#!/bin/bash
# CUDA-build the r6 worktrees (CPU only: nvcc compile, no GPU use). Logs in ~/colibri-bench/final/builds/.
L=/home/maplemercury/colibri-bench/final/builds
mkdir -p $L
b() {
  local t=$1
  local log=$L/$t.build.log
  ( cd /tmp/cw/$t/c && echo "tree=/tmp/cw/$t sha=$(git rev-parse HEAD) start=$(date -Is)" && \
    make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j8; \
    echo "EXIT=$? end=$(date -Is)" ) > $log 2>&1
}
for t in "$@"; do b $t & done
wait
echo BUILD_ALL_DONE > $L/done.$(echo "$@" | tr ' ' '_')
