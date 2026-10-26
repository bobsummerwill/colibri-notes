#!/bin/bash
# C+ (fork samebase/cplus) for c4m26: fetch, then the generic CUDA sm_120 build (dg/s26c) and the CPU-only build
# (dg/s26ccpu), build25.sh's make lines. Run after c4m25 has finished (it would compete for the CPU).
# usage: build26.sh [COMMIT]   (default: the fetched branch tip; pass the expected sha to pin it)
flock -n ~/colibri-bench/c4m25/.lock true || { echo "c4m25 is still running; not building"; exit 5; }
cd ~/Projects/colibri && git fetch -q origin samebase/cplus:refs/remotes/origin/samebase/cplus || { echo "fetch failed"; exit 4; }
ref=${1:-$(git rev-parse origin/samebase/cplus)}
git cat-file -t "$ref" >/dev/null || { echo "no commit $ref"; exit 4; }
ref=$(git rev-parse "$ref^{commit}"); echo "C+ = $ref"
cd ~/colibri-bench/dg
CU="CUDA_HOME=/usr/local/cuda-12.8 NVCC=/usr/local/cuda-12.8/bin/nvcc"
for spec in c:cuda ccpu:cpu; do
  IFS=: read n kind <<< "$spec"; t=s26$n
  [ -d $t ] || git -C ~/Projects/colibri worktree add -q --detach ~/colibri-bench/dg/$t $ref || { echo "$t worktree failed" >> logs/build26.status; continue; }
  [ "$(git -C $t rev-parse HEAD)" = "$ref" ] || { echo "$t not at $ref" >> logs/build26.status; continue; }
  if [ "$kind" = cpu ]; then flags=""; else flags="CUDA=1 CUDA_ARCH=sm_120 $CU"; fi
  ( cd $t/c && echo "tree=$t sha=$(git rev-parse HEAD) kind=$kind flags=$flags start=$(date -Is)" &&
    env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=$HOME nice -n 10 make -f Makefile.deepseek-v4 deepseek-v4 $flags -j12;
    echo "EXIT=$? end=$(date -Is)" ) > logs/build26-$t.log 2>&1
  grep -iE "warning|error" logs/build26-$t.log | grep -v lto-wrapper > logs/build26-$t.diagnostics.txt
  echo "$t $(git -C $t rev-parse --short HEAD) $(tail -1 logs/build26-$t.log) warn=$(wc -l < logs/build26-$t.diagnostics.txt) bin=$(sha256sum $t/c/deepseek_v4 2>/dev/null | cut -c1-16) cuda_linked=$(ldd $t/c/deepseek_v4 2>/dev/null | grep -c cudart)" >> logs/build26.status
done; echo "done $ref" >> logs/build26.status
tail -3 logs/build26.status
