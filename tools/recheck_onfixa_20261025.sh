#!/bin/bash
# recheck_onfixa_20261025.sh (counter, 2026-10-25 night): PR #2 and the split stack rebased onto #2401 (fix A dd03a9de).
#   1. make -C c check on PR #2's tip and split parts 1-3 (part 4's tree = PR #2's), 4 at a time
#   2. per commit, PR #2 and the split: build, GPU-tier, +Vulkan and test-hooks compiles, dsv4/v4 python tests (discover)
# Results: ~/colibri-notes/manifests/counter-20261025-onfixa/
set -u
P=/home/unagi/Projects; R=$P/colibri
M=/home/unagi/colibri-notes/manifests/counter-20261025-onfixa; mkdir -p "$M/logs"
BASE=fix/dsv4-mirror-guard-overshoot
declare -A TIP=([pr2]=docs5/pr2-two-gpu-decode [split1]=split/pr2-1-guard [split2]=split/pr2-2-settings [split3]=split/pr2-3-vram-first)
for n in "${!TIP[@]}"; do
    w=$P/wt-o-$n; sha=$(git -C "$R" rev-parse "${TIP[$n]}")
    [ -d "$w" ] || git -C "$R" worktree add -q --detach "$w" "$sha"
    git -C "$w" checkout -q --detach "$sha"
done
echo "# make -C c check, counter ($(gcc --version | head -1)), $(date -Is), base $BASE $(git -C $R rev-parse --short $BASE)" > "$M/make-check.txt"
check() {
    local n=$1 w=$P/wt-o-$1 t0=$(date +%s)
    make -C "$w/c" check > "$M/logs/check-$n.log" 2>&1; local rc=$?
    echo "$n ${TIP[$n]} $(git -C "$w" rev-parse --short HEAD) make_check=$rc $(grep -E '^Ran [0-9]+ tests' "$M/logs/check-$n.log" | tail -1) $(tail -5 "$M/logs/check-$n.log" | grep -E '^(OK|FAILED)' | tail -1) $(( $(date +%s) - t0 ))s" >> "$M/make-check.txt"
}
for n in pr2 split1 split2 split3; do check "$n" & done
wait
CF="-O2 -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function -fopenmp -pthread -march=native -fsyntax-only"
percommit() { # range out
    local w=$P/wt-o-pc c
    [ -d "$w" ] || git -C "$R" worktree add -q --detach "$w" $BASE
    { echo "# Per-commit check, counter ($(gcc --version | head -1)), $(date -Is), range $1"
      echo "# (1) make -f Makefile.deepseek-v4 deepseek-v4 (clean) (2) deepseek_v4.c -DCOLI_V4_GPU_TIER -DCOLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER (3) +COLI_VULKAN (4) tests/test_deepseek_v4.c -DCOLI_V4_TEST_HOOKS (5) unittest discover test_dsv4*/test_v4*"; } > "$2"
    for c in $(git -C "$R" rev-list --reverse "$1"); do
        git -C "$w" checkout -q --detach "$c"
        ( cd "$w/c" && make -f Makefile.deepseek-v4 deepseek-v4-clean >/dev/null 2>&1; rm -f deepseek_v4 ./*.o
          make -f Makefile.deepseek-v4 deepseek-v4 -j12 > "$M/logs/pc-build.log" 2>&1; b=$?
          gcc $CF -DCOLI_V4_GPU_TIER -DCOLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER deepseek_v4.c > "$M/logs/pc-gt.log" 2>&1; g=$?
          gcc $CF -DCOLI_V4_GPU_TIER -DCOLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER -DCOLI_VULKAN deepseek_v4.c > "$M/logs/pc-vk.log" 2>&1; v=$?
          gcc $CF -DCOLI_V4_TEST_HOOKS -I. tests/test_deepseek_v4.c > "$M/logs/pc-test.log" 2>&1; t=$?
          { python3 -m unittest discover -s tests -t tests -p 'test_dsv4*.py' && \
            python3 -m unittest discover -s tests -t tests -p 'test_v4*.py'; } > "$M/logs/pc-py.log" 2>&1; p=$?
          d=$(cat "$M"/logs/pc-{build,gt,vk,test}.log | grep -E 'warning:|error:' | grep -vc 'lto-wrapper')
          echo "$(git rev-parse --short=10 HEAD) build=$b gpu_tier=$g vk=$v test=$t py=$p diag=$d  $(git log -1 --format=%s | cut -c1-80)" ) >> "$2"
    done
}
percommit $BASE..docs5/pr2-two-gpu-decode "$M/pr2-percommit.txt"
percommit $BASE..split/pr2-4-two-gpu "$M/pr2-split-percommit.txt"
echo "ALL DONE $(date -Is)" >> "$M/make-check.txt"
