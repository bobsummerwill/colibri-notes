#!/usr/bin/env bash
# fix-nsys.sh - repair Nsight Systems on Ubuntu/Debian and remove the shim.
#
# The distro package installs nsys (agent) under
#   /usr/lib/<arch>/nsight-systems/target-linux-x64
# but the tools (QdstrmImporter and friends) under
#   /usr/lib/nsight-systems/host-linux-x64
# nsys looks for the importer as a sibling of the agent directory, so every
# profile stays a .qdstrm. This script creates that sibling link, removes the
# ~/.local/bin/nsys shim and its PATH lines, and verifies with a real kernel.
#
# usage: sudo bash fix-nsys.sh [--perf] [--dry-run]
#   --perf     also set kernel.perf_event_paranoid=1 (enables CPU sampling)
#   --dry-run  print what would change, touch nothing
set -euo pipefail

WITH_PERF=0
DRY=0
for a in "$@"; do
  case "$a" in
    --perf) WITH_PERF=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help)
      sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

run() {
  if [ "$DRY" = 1 ]; then printf '  would: %s\n' "$*"; else "$@"; fi
}

if [ "$(id -u)" -ne 0 ]; then
  if [ "$DRY" = 1 ]; then
    echo "note: not root; a real run re-executes itself with sudo" >&2
  else
    exec sudo -E bash "$0" "$@"
  fi
fi

echo "== 1. locate the two halves"
NSYS_BIN=$(command -v nsys || true)
if [ -z "$NSYS_BIN" ]; then
  echo "ERROR: nsys is not installed on this host." >&2
  exit 1
fi
NSYS_REAL=$(readlink -f "$NSYS_BIN")
TARGET_DIR=$(dirname "$NSYS_REAL")
PREFIX=$(dirname "$TARGET_DIR")
LINK="$PREFIX/host-linux-x64"

echo "   nsys           : $NSYS_BIN -> $NSYS_REAL"
echo "   agent dir      : $TARGET_DIR"

NEEDS_FIX=0
if [ -x "$LINK/QdstrmImporter" ]; then
  echo "   tools dir      : $LINK (self-contained install)"
  echo "   this host does not have the Debian packaging split; nothing to link."
else
  NEEDS_FIX=1
  HOST_DIR=/usr/lib/nsight-systems/host-linux-x64
  if [ ! -x "$HOST_DIR/QdstrmImporter" ]; then
    HOST_DIR=$(dirname "$(dpkg-query -L nsight-systems 2>/dev/null |
      grep -m1 '/host-linux-x64/QdstrmImporter$' || true)")
  fi
  echo "   tools dir      : $HOST_DIR"
  echo "   missing sibling: $LINK"
  if [ ! -x "$HOST_DIR/QdstrmImporter" ]; then
    echo "ERROR: QdstrmImporter not found for $NSYS_REAL." >&2
    echo "       This host has neither a self-contained install nor the" >&2
    echo "       Ubuntu nsight-systems packages; nothing to repair." >&2
    exit 1
  fi
fi

if [ "$NEEDS_FIX" = 1 ]; then
  echo "== 2. link the tools beside the agent"
  if [ -e "$LINK" ] || [ -L "$LINK" ]; then
    echo "   already present, leaving it alone"
  else
    run ln -s "$HOST_DIR" "$LINK"
  fi
else
  echo "== 2. no link needed on this host"
fi

echo "== 3. remove the user-level nsys shim"
shopt -s nullglob
for home in /home/* /root; do
  f="$home/.local/bin/nsys"
  if [ -f "$f" ] && grep -q 'Nsight Systems CLI shim' "$f" 2>/dev/null; then
    echo "   removing $f"
    run rm -f "$f"
  fi
  for rc in "$home/.bashrc" "$home/.profile"; do
    [ -f "$rc" ] || continue
    if grep -q 'user-local tools (nsys shim)' "$rc" 2>/dev/null; then
      echo "   cleaning PATH block in $rc"
      if [ "$DRY" = 1 ]; then
        echo "     would remove the '# user-local tools (nsys shim)' block"
      else
        python3 - "$rc" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
s = s.replace('\n# user-local tools (nsys shim)\nexport PATH="$HOME/.local/bin:$PATH"\n', '\n')
open(p, 'w').write(s)
PY
      fi
    fi
  done
done

if [ "$WITH_PERF" = 1 ]; then
  echo "== 4. enable CPU sampling (perf_event_paranoid=1)"
  run sysctl -w kernel.perf_event_paranoid=1
  if [ "$DRY" = 0 ]; then
    echo 'kernel.perf_event_paranoid=1' > /etc/sysctl.d/50-nsys-perf.conf
    echo "   persisted in /etc/sysctl.d/50-nsys-perf.conf"
  else
    echo "   would persist in /etc/sysctl.d/50-nsys-perf.conf"
  fi
else
  echo "== 4. CPU sampling left as is (pass --perf to enable it)"
fi

echo "== 5. verify with a real kernel"
if [ "$DRY" = 1 ]; then
  echo "   skipped in --dry-run"
  exit 0
fi
if [ "$NEEDS_FIX" = 1 ]; then
  test -x "$LINK/QdstrmImporter" || { echo "ERROR: link is not usable" >&2; exit 1; }
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cat > "$tmp/add.cu" <<'EOF'
#include <cstdio>
__global__ void add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
int main(void) {
    int n = 1 << 20;
    size_t s = (size_t)n * sizeof(float);
    float *a, *b, *c;
    cudaMalloc(&a, s); cudaMalloc(&b, s); cudaMalloc(&c, s);
    add<<<(n + 255) / 256, 256>>>(a, b, c, n);
    printf("kernel status %d\n", (int)cudaDeviceSynchronize());
    return 0;
}
EOF
if command -v nvcc >/dev/null 2>&1; then
  nvcc -O2 -arch=sm_86 "$tmp/add.cu" -o "$tmp/add" >/dev/null 2>&1 || true
fi
if [ -x "$tmp/add" ]; then
  # NOTE: this nsys build's in-process --stats path fails with
  # "no such function: median"; run "nsys stats" as its own process.
  nsys profile -o "$tmp/p" --force-overwrite true --trace=cuda \
    "$tmp/add" >"$tmp/log" 2>&1 || true
  nsys stats --report gpukernsum "$tmp/p.nsys-rep" >"$tmp/stats" 2>&1 || true
  if grep -q 'add(const float' "$tmp/stats" && [ -f "$tmp/p.nsys-rep" ]; then
    echo "   OK: plain nsys wrote a report and the kernel summary lists 'add'"
  else
    echo "   WARNING: report/kernel not found; see $tmp (kept for inspection)" >&2
    trap - EXIT
    exit 1
  fi
else
  echo "   nvcc or the test binary unavailable; checking report generation only"
  nsys profile -o "$tmp/p" --force-overwrite true true >/dev/null 2>&1 || true
  [ -f "$tmp/p.nsys-rep" ] && echo "   OK: plain nsys wrote a report" || {
    echo "   ERROR: no report written" >&2; exit 1; }
fi

echo
echo "Done. Open a new shell (or re-login) so PATH no longer has the shim,"
echo "then use plain 'nsys'. Reports: $(command -v nsys) stats <file>.nsys-rep"
