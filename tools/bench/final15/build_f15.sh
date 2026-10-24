#!/bin/bash
# final15: the CUDA=1 VK=1 build of f15/fixb (dev 61646cbd + the uploader's host images aligned inside calloc), build_vk9.sh's make line
W=~/colibri-bench/cw22
G=$HOME/colibri-bench/vulkansdk/1.4.363.0/x86_64/bin/glslc
export CPATH=$HOME/colibri-bench/vulkansdk/1.4.363.0/x86_64/include
B=~/colibri-bench/final15/builds; mkdir -p $B
cd $W/repo; t=$W/f15-fixb
[ -d $t ] || git worktree add -q --detach $t f15/fixb || exit 1
( cd $t/c && echo "tree=$t sha=$(git rev-parse HEAD) start=$(date -Is)" &&
  nice -n 15 make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 VK=1 GLSLC=$G CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j14; echo "EXIT=$? end=$(date -Is)" ) > $B/f15-fixb.build.log 2>&1
echo "f15-fixb $(tail -1 $B/f15-fixb.build.log) warnings=$(grep -iE "warning|error" $B/f15-fixb.build.log | grep -vc lto-wrapper)" >> $B/status
echo done >> $B/status
