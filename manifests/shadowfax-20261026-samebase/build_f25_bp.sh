#!/bin/bash
# final25 (2026-10-26): same-base run. Pure-Vulkan builds (CUDA=1 VK=1, final22 make line) of B dd03a9de and P 5b3f4601.
# The CUDA builds of B and P are final22's f22-base / f22-pr2 (binaries verified against final23 tips.txt).
W=~/colibri-bench/cw22
B=~/colibri-bench/final25/builds; mkdir -p $B
G=$HOME/colibri-bench/vulkansdk/1.4.363.0/x86_64/bin/glslc
MAKE_VK="make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 VK=1 GLSLC=$G CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j14"
cd $W/repo
for spec in basevk:dd03a9de pr2vk:5b3f4601; do
  IFS=: read n ref <<< "$spec"; t=$W/f25-$n
  [ -d $t ] || git worktree add -q --detach $t $ref || { echo "f25-$n worktree failed" >> $B/status; continue; }
  ( cd $t/c && echo "tree=$t sha=$(git rev-parse HEAD) tree_sha=$(git rev-parse "HEAD^{tree}") start=$(date -Is)" &&
    env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=$HOME CPATH=$HOME/colibri-bench/vulkansdk/1.4.363.0/x86_64/include nice -n 15 $MAKE_VK; echo "EXIT=$? end=$(date -Is)" ) > $B/f25-$n.build.log 2>&1
  echo "f25-$n $(tail -1 $B/f25-$n.build.log) warnings=$(grep -iE "warning|error" $B/f25-$n.build.log | grep -vc lto-wrapper) bin=$(sha256sum $t/c/deepseek_v4 | cut -c1-16)" >> $B/status
done
echo done-bp >> $B/status
