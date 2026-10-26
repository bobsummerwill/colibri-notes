#!/bin/bash
# final27/28 (2026-10-26): same-base run restarted on dev c1c06ac6 (#2401-#2404 merged upstream).
# B = c1c06ac6, P = samebase/pr2 11c40038, M = samebase/m1766 ecc92586, C+ = samebase/cplus b659ac8d.
# CUDA: final6 make line (sm_86 NO_TC) -> cw22/f27-{base,pr2,m1766,cplus}; pure Vulkan (CUDA=1 VK=1, final22 line) -> f27-*vk.
# usage: build_f27.sh cuda|vk   (two streams can run in parallel)
W=~/colibri-bench/cw22
B=~/colibri-bench/final27/builds; mkdir -p $B
G=$HOME/colibri-bench/vulkansdk/1.4.363.0/x86_64/bin/glslc
MAKE_CUDA="make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j14"
MAKE_VK="make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 VK=1 GLSLC=$G CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j14"
kind=${1:?cuda|vk}
for spec in base:c1c06ac6 pr2:11c40038 m1766:ecc92586 cplus:b659ac8d; do
  IFS=: read n ref <<< "$spec"
  ref=$(git -C $W/repo rev-parse "$ref^{commit}") || { echo "f27-$n: $ref missing" >> $B/status; continue; }
  if [ $kind = vk ]; then n=${n}vk; ML="$MAKE_VK"; X="CPATH=$HOME/colibri-bench/vulkansdk/1.4.363.0/x86_64/include"; else ML="$MAKE_CUDA"; X=; fi
  t=$W/f27-$n
  [ -d $t ] || git -C $W/repo worktree add -q --detach $t $ref || { echo "f27-$n worktree failed" >> $B/status; continue; }
  [ "$(git -C $t rev-parse HEAD)" = "$ref" ] || { echo "f27-$n exists at another commit" >> $B/status; continue; }
  ( cd $t/c && echo "tree=$t sha=$(git rev-parse HEAD) tree_sha=$(git rev-parse "HEAD^{tree}") start=$(date -Is)" &&
    env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=$HOME $X nice -n 15 $ML; echo "EXIT=$? end=$(date -Is)" ) > $B/f27-$n.build.log 2>&1
  grep -iE "warning|error" $B/f27-$n.build.log | grep -v lto-wrapper > $B/f27-$n.diagnostics.txt
  echo "f27-$n $(git -C $t rev-parse --short HEAD) $(tail -1 $B/f27-$n.build.log) warnings=$(wc -l < $B/f27-$n.diagnostics.txt) bin=$(sha256sum $t/c/deepseek_v4 2>/dev/null | cut -c1-16)" >> $B/status
done
echo "done-$kind" >> $B/status
