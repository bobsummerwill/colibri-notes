#!/bin/bash
# final10: CUDA builds (final6's make line) of dev 61646cbd and fix A, logs in final10/builds/<tree>.build.log
W=~/colibri-bench/cw22
B=~/colibri-bench/final10/builds; mkdir -p $B
MAKE_LINE="make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j12"
cd $W/repo
for b in f10-dev:61646cbd07da458569deb882586a390ea945e90b f10-fixa:fix/dsv4-mirror-guard-overshoot; do
  n=${b%%:*}; ref=${b#*:}; t=$W/$n
  [ -d $t ] || git worktree add -q --detach $t $ref || exit 1
  ( cd $t/c && echo "tree=$t sha=$(git rev-parse HEAD) tree_sha=$(git rev-parse 'HEAD^{tree}') start=$(date -Is)" &&
    env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=$HOME nice -n 15 $MAKE_LINE; echo "EXIT=$? end=$(date -Is)" ) > $B/$n.build.log 2>&1
  echo "$n $(tail -1 $B/$n.build.log) warnings=$(grep -iE "warning|error" $B/$n.build.log | grep -vc lto-wrapper)" >> $B/status
done
echo done >> $B/status
