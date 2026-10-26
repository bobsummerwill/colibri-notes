#!/bin/bash
# 2026-10-26 same-base run (c4m27): generic CUDA sm_120 builds (CUDA 12.8, no DeepGEMM; build22.sh flags for m1766/mbase)
# and CPU-only builds (no CUDA; Makefile.deepseek-v4 default flags, -march=native) of the given trees.
# usage: build27.sh NAME:REF[:cuda|cpu] ...   (trees dg/s25<NAME>, git worktrees of ~/Projects/colibri)
cd ~/colibri-bench/dg
CU="CUDA_HOME=/usr/local/cuda-12.8 NVCC=/usr/local/cuda-12.8/bin/nvcc"
for spec in "$@"; do
  IFS=: read n ref kind <<< "$spec"; t=s27$n
  [ -d $t ] || git -C ~/Projects/colibri worktree add -q --detach ~/colibri-bench/dg/$t $ref || { echo "$t worktree failed" >> logs/build27.status; continue; }
  [ "$(git -C $t rev-parse HEAD)" = "$(git -C $t rev-parse $ref^{commit})" ] || { echo "$t not at $ref" >> logs/build27.status; continue; }
  if [ "$kind" = cpu ]; then flags=""; else flags="CUDA=1 CUDA_ARCH=sm_120 $CU"; fi
  ( cd $t/c && echo "tree=$t sha=$(git rev-parse HEAD) kind=$kind flags=$flags start=$(date -Is)" &&
    env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=$HOME nice -n 10 make -f Makefile.deepseek-v4 deepseek-v4 $flags -j12;
    echo "EXIT=$? end=$(date -Is)" ) > logs/build27-$t.log 2>&1
  echo "$t $(git -C $t rev-parse --short HEAD) $(tail -1 logs/build27-$t.log) warn=$(grep -iE "warning|error" logs/build27-$t.log | grep -vc lto-wrapper) bin=$(sha256sum $t/c/deepseek_v4 2>/dev/null | cut -c1-16) cuda_linked=$(ldd $t/c/deepseek_v4 2>/dev/null | grep -c cudart)" >> logs/build27.status
done; echo "done $*" >> logs/build27.status
