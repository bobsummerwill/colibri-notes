#!/bin/bash
# build_r9f.sh -- create and CUDA-build the F42-fix trees for run_f42fix.sh (shadowfax, CPU only:
# git worktrees + nvcc/gcc compile, no GPU use). Same make line as final2/build_r8.sh.
#   r9-pr1v5         4feb6d48 (pr1/v5 tip: the routing-books fix c7f79815 + the rest of PR #1), plain
#   r9-pr1v5-probe   4feb6d48 + final2/probe/f42fix_probe_patch.py (logging only, V4_DEC_TRACE=1|2)
#   r9-pr1-probe     93b74183 (old pr1 tip, no fix) + the same probe patch
# Records each probe diff as final2/probe/<tree>.diff (+ .sha256). Logs:
# final2/builds/<tree>.build.log ("tree=... sha=..." first line, "EXIT=<rc>" last line).
# Refuses to touch an existing tree directory.
set -u
F2=/home/maplemercury/colibri-bench/final2
L=$F2/builds
REPO=/tmp/colibri
PATCH=$F2/probe/f42fix_probe_patch.py
mkdir -p "$L" "$F2/probe"
build_one() { # name sha probe(0|1)
    local T=$1 SHA=$2 PROBE=$3
    local DIR=/tmp/cw/$T LOG=$L/$T.build.log extra=
    if [ -e "$DIR" ]; then echo "$DIR exists; not touching it" >&2; return 1; fi
    git -C "$REPO" cat-file -e "$SHA^{commit}" || { echo "$SHA not in $REPO" >&2; return 1; }
    git -C "$REPO" worktree add --detach "$DIR" "$SHA" > /dev/null 2>&1 || { echo "worktree add $DIR failed" >&2; return 1; }
    if [ "$PROBE" = 1 ]; then
        python3 "$PATCH" "$DIR" || { echo "probe patch failed on $DIR" >&2; return 1; }
        git -C "$DIR" diff > "$F2/probe/$T.diff"
        sha256sum < "$F2/probe/$T.diff" | cut -c1-64 > "$F2/probe/$T.diff.sha256"
        [ "$(git -C "$DIR" status --porcelain --untracked-files=no)" = " M c/deepseek_v4.c" ] || { echo "unexpected changes in $DIR" >&2; return 1; }
        extra=" probe_diff_sha256=$(cat "$F2/probe/$T.diff.sha256") probe_patch_sha256=$(sha256sum < "$PATCH" | cut -c1-64)"
    else
        [ -z "$(git -C "$DIR" status --porcelain --untracked-files=no)" ] || { echo "$DIR not clean" >&2; return 1; }
    fi
    ( cd "$DIR/c" && echo "tree=$DIR sha=$(git rev-parse HEAD) start=$(date -Is)$extra" && \
      make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j8; \
      echo "EXIT=$? end=$(date -Is)" ) > "$LOG" 2>&1
    local warn
    warn=$(grep -iE "warning|error" "$LOG" | grep -vE "^lto-wrapper: (warning: using serial compilation|note: see)" | wc -l)
    echo "$T: $(tail -1 "$LOG") warn_err_lines=$warn nvcc_lines=$(grep -c nvcc "$LOG") binary_sha256=$(sha256sum "$DIR/c/deepseek_v4" 2>/dev/null | cut -c1-64)$extra"
    grep -q '^EXIT=0' "$LOG" && [ "$warn" = 0 ]
}
rc=0
build_one r9-pr1v5 4feb6d487973e064abc53486f379032aa402efc6 0 > "$L/r9f.r9-pr1v5.result" 2>&1 &
p1=$!
build_one r9-pr1v5-probe 4feb6d487973e064abc53486f379032aa402efc6 1 > "$L/r9f.r9-pr1v5-probe.result" 2>&1 &
p2=$!
build_one r9-pr1-probe 93b741839677ec457868fd71ab668cc1249b9a22 1 > "$L/r9f.r9-pr1-probe.result" 2>&1 &
p3=$!
wait $p1 || rc=1; wait $p2 || rc=1; wait $p3 || rc=1
cat "$L"/r9f.r9-pr1v5.result "$L"/r9f.r9-pr1v5-probe.result "$L"/r9f.r9-pr1-probe.result
echo "BUILD_R9F_DONE rc=$rc $(date -Is)" | tee "$L/done.r9f"
exit $rc
