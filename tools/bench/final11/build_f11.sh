#!/bin/bash
# final11: the CUDA build of PR #2 + the F18 fix (branch f11/pr2-f18) with final6's make line, log in
# final11/builds/f11-pr2f.build.log; and the diagnostic builds v2 (CUDA=1 VK=1, vkdiag_patch.py v2 as one local commit
# on dev 61646cbd and PR #1 47c8ec51, never pushed), logs cw22/build-vk11d-{dev,pr1}.log
W=~/colibri-bench/cw22
B=~/colibri-bench/final11/builds; mkdir -p $B
G=$HOME/colibri-bench/vulkansdk/1.4.363.0/x86_64/bin/glslc
MAKE_LINE="make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j14"
cd $W/repo
t=$W/f11-pr2f
[ -d $t ] || git worktree add -q --detach $t f11/pr2-f18 || exit 1
( cd $t/c && echo "tree=$t sha=$(git rev-parse HEAD) tree_sha=$(git rev-parse 'HEAD^{tree}') start=$(date -Is)" &&
  env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=$HOME nice -n 15 $MAKE_LINE; echo "EXIT=$? end=$(date -Is)" ) > $B/f11-pr2f.build.log 2>&1
echo "f11-pr2f $(tail -1 $B/f11-pr2f.build.log) warnings=$(grep -iE "warning|error" $B/f11-pr2f.build.log | grep -vc lto-wrapper)" >> $B/status
for b in dev:61646cbd07da458569deb882586a390ea945e90b pr1:47c8ec5165384009f5d573a6ad9e229483e579de; do
  n=${b%%:*}; sha=${b#*:}; t=$W/f11d-$n
  if [ ! -d $t ]; then
    git worktree add -q --detach $t $sha && python3 -I ~/colibri-bench/final11/vkdiag_patch.py $t/c/deepseek_v4.c &&
      git -C $t -c user.name=final11 -c user.email=final11@localhost commit -q -am "DIAGNOSTIC ONLY (final11): log the Vulkan dense hook's declines with their reasons" || exit 1
  fi
  ( cd $t/c && echo "tree=$t sha=$(git rev-parse HEAD)" && CPATH=$HOME/colibri-bench/vulkansdk/1.4.363.0/x86_64/include nice -n 15 make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 VK=1 GLSLC=$G CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j14; echo "EXIT=$?" ) > $W/build-vk11d-$n.log 2>&1
  echo "$n $(tail -1 $W/build-vk11d-$n.log) warnings=$(grep -iE "warning|error" $W/build-vk11d-$n.log | grep -vc lto-wrapper)" >> $B/status
done
echo done >> $B/status
