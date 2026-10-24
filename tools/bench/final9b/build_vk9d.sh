#!/bin/bash
# final9b: the diagnostic CUDA=1 VK=1 builds (vkdiag_patch.py as one local commit on each tip; never pushed)
W=~/colibri-bench/cw22
G=$HOME/colibri-bench/vulkansdk/1.4.363.0/x86_64/bin/glslc
export CPATH=$HOME/colibri-bench/vulkansdk/1.4.363.0/x86_64/include
cd $W/repo
for b in dev:61646cbd07da458569deb882586a390ea945e90b pr1:47c8ec5165384009f5d573a6ad9e229483e579de; do
  n=${b%%:*}; sha=${b#*:}; t=$W/f9d-$n
  if [ ! -d $t ]; then
    git worktree add -q --detach $t $sha && python3 -I ~/colibri-bench/final9b/vkdiag_patch.py $t/c/deepseek_v4.c &&
      git -C $t -c user.name=final9b -c user.email=final9b@localhost commit -q -am "DIAGNOSTIC ONLY (final9b): log the Vulkan dense hook's CPU fallbacks" || exit 1
  fi
  ( cd $t/c && echo "tree=$t sha=$(git rev-parse HEAD)" && nice -n 15 make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 VK=1 GLSLC=$G CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j12; echo "EXIT=$?" ) > $W/build-vk9d-$n.log 2>&1
  echo "$n $(tail -1 $W/build-vk9d-$n.log) warnings=$(grep -iE "warning|error" $W/build-vk9d-$n.log | grep -vc lto-wrapper)" >> $W/build-vk9d.status
done
echo done >> $W/build-vk9d.status
