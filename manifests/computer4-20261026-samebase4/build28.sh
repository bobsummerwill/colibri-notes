#!/bin/bash
# 2026-10-26 (c4m28): PR #2's split steps on dev c1c06ac6, generic CUDA sm_120 builds (CUDA 12.8, no DeepGEMM), as
# build27.sh. S1 = split/pr2-1-guard 6a191e90, S2 = split/pr2-2-settings c45b6962, S3 = split/pr2-3-vram-first 695a7be4.
# Trees dg/s28s1..s3 (git worktrees of ~/Projects/colibri). Run by c4m28/wait_then_c4m28.sh after c4m27 ends.
cd ~/colibri-bench/dg
systemctl --user is-active -q c4m27 && { echo "c4m27 still running: not building" >> logs/build28.status; exit 1; }
CU="CUDA_HOME=/usr/local/cuda-12.8 NVCC=/usr/local/cuda-12.8/bin/nvcc"
git -C ~/Projects/colibri fetch -q origin split/pr2-1-guard split/pr2-2-settings split/pr2-3-vram-first >> logs/build28.status 2>&1
for spec in s1:6a191e90 s2:c45b6962 s3:695a7be4; do
  IFS=: read n ref <<< "$spec"; t=s28$n
  [ -d $t ] || git -C ~/Projects/colibri worktree add -q --detach ~/colibri-bench/dg/$t $ref || { echo "$t worktree failed" >> logs/build28.status; continue; }
  [ "$(git -C $t rev-parse HEAD)" = "$(git -C $t rev-parse $ref^{commit})" ] || { echo "$t not at $ref" >> logs/build28.status; continue; }
  ( cd $t/c && echo "tree=$t sha=$(git rev-parse HEAD) flags=CUDA=1 CUDA_ARCH=sm_120 $CU start=$(date -Is)" &&
    env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=$HOME nice -n 10 make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_ARCH=sm_120 $CU -j12;
    echo "EXIT=$? end=$(date -Is)" ) > logs/build28-$t.log 2>&1
  echo "$t $(git -C $t rev-parse --short HEAD) $(tail -1 logs/build28-$t.log) warn=$(grep -iE "warning|error" logs/build28-$t.log | grep -vc lto-wrapper) bin=$(sha256sum $t/c/deepseek_v4 2>/dev/null | cut -c1-16) cuda_linked=$(ldd $t/c/deepseek_v4 2>/dev/null | grep -c cudart)" >> logs/build28.status
done; echo "done-c4m28" >> logs/build28.status
