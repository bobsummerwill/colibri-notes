#!/bin/bash
# CUDA=1 VK=1 engine builds of dev and PR #1 (final6 Vulkan arms); glslc from the LunarG SDK in ~/colibri-bench/vulkansdk
W=~/colibri-bench/cw22
G=$HOME/colibri-bench/vulkansdk/1.4.363.0/x86_64/bin/glslc
export CPATH=$HOME/colibri-bench/vulkansdk/1.4.363.0/x86_64/include   # vulkan/vulkan.h; the loader is dlopened at run time
cd $W/repo
for b in dev:7df8c27c95acf7019dd60714ec1c42fa7e36b730 pr1:ebe82123cb83d5b969caedd50203dc9c156b348a; do
  n=${b%%:*}; sha=${b#*:}; t=$W/f6v-$n
  [ -d $t ] || git worktree add -q --detach $t $sha
  ( cd $t/c && echo "tree=$t sha=$(git rev-parse HEAD)" && nice -n 15 make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 VK=1 GLSLC=$G CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j6; echo "EXIT=$?" ) > $W/build-vk-$n.log 2>&1
  echo "$n $(tail -1 $W/build-vk-$n.log) warnings=$(grep -iE "warning|error" $W/build-vk-$n.log | grep -vc lto-wrapper)" >> $W/build-vk.status
done
echo done >> $W/build-vk.status
