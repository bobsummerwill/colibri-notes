#!/bin/bash
# build_f29.sh (2026-10-26): PR #2's split steps on dev c1c06ac6 for final29, CUDA sm_86 NO_TC (final6 line, as f27-*).
# S1 = split/pr2-1-guard 6a191e90, S2 = split/pr2-2-settings c45b6962, S3 = split/pr2-3-vram-first 695a7be4 (fork).
# Worktrees cw22/f29-s1..s3; logs and diagnostics in final29/builds. Run by wait_then_f29.sh after the chain ends.
W=~/colibri-bench/cw22
B=~/colibri-bench/final29/builds; mkdir -p $B
systemctl --user is-active -q cbench-chain2 && { echo "cbench-chain2 still running: not building" >> $B/status; exit 1; }
MAKE_CUDA="make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j14"
git -C $W/repo fetch -q https://github.com/bobsummerwill/colibri.git \
    split/pr2-1-guard split/pr2-2-settings split/pr2-3-vram-first >> $B/status 2>&1
for spec in s1:6a191e90 s2:c45b6962 s3:695a7be4; do
  IFS=: read n ref <<< "$spec"
  ref=$(git -C $W/repo rev-parse "$ref^{commit}") || { echo "f29-$n: $ref missing" >> $B/status; continue; }
  t=$W/f29-$n
  [ -d $t ] || git -C $W/repo worktree add -q --detach $t $ref || { echo "f29-$n worktree failed" >> $B/status; continue; }
  [ "$(git -C $t rev-parse HEAD)" = "$ref" ] || { echo "f29-$n exists at another commit" >> $B/status; continue; }
  ( cd $t/c && echo "tree=$t sha=$(git rev-parse HEAD) tree_sha=$(git rev-parse "HEAD^{tree}") start=$(date -Is)" &&
    env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=$HOME nice -n 15 $MAKE_CUDA; echo "EXIT=$? end=$(date -Is)" ) > $B/f29-$n.build.log 2>&1
  grep -iE "warning|error" $B/f29-$n.build.log | grep -v lto-wrapper > $B/f29-$n.diagnostics.txt
  echo "f29-$n $(git -C $t rev-parse --short HEAD) $(tail -1 $B/f29-$n.build.log) warnings=$(wc -l < $B/f29-$n.diagnostics.txt) bin=$(sha256sum $t/c/deepseek_v4 2>/dev/null | cut -c1-16)" >> $B/status
done
echo "done-f29" >> $B/status
