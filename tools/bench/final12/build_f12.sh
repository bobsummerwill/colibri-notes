#!/bin/bash
# final12: the CUDA build of f12/pr2 (PR #2 + F18 fix + the decode stage names), final6's make line
W=~/colibri-bench/cw22
B=~/colibri-bench/final12/builds; mkdir -p $B
MAKE_LINE="make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j14"
cd $W/repo; t=$W/f12-pr2s
[ -d $t ] || git worktree add -q --detach $t f12/pr2 || exit 1
( cd $t/c && echo "tree=$t sha=$(git rev-parse HEAD) tree_sha=$(git rev-parse 'HEAD^{tree}') start=$(date -Is)" &&
  env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=$HOME nice -n 15 $MAKE_LINE; echo "EXIT=$? end=$(date -Is)" ) > $B/f12-pr2s.build.log 2>&1
echo "f12-pr2s $(tail -1 $B/f12-pr2s.build.log) warnings=$(grep -iE "warning|error" $B/f12-pr2s.build.log | grep -vc lto-wrapper)" >> $B/status
echo done >> $B/status
