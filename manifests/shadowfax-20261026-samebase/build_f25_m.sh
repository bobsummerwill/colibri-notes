#!/bin/bash
# final25: M = #1766 merged onto dd03a9de (fork branch samebase/m1766): CUDA (final6 make line, sm_86 NO_TC) -> cw22/f25-m1766,
# pure-Vulkan (CUDA=1 VK=1, final22 line) -> cw22/f25-m1766vk.  usage: build_f25_m.sh [COMMIT]
W=~/colibri-bench/cw22
B=~/colibri-bench/final25/builds; mkdir -p $B
G=$HOME/colibri-bench/vulkansdk/1.4.363.0/x86_64/bin/glslc
MAKE_CUDA="make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j14"
MAKE_VK="make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 VK=1 GLSLC=$G CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j14"
cd $W/repo && git fetch -q https://github.com/bobsummerwill/colibri.git samebase/m1766:refs/remotes/fork/samebase/m1766 || { echo "fetch failed" >> $B/status; exit 4; }
ref=${1:-$(git rev-parse fork/samebase/m1766)}; ref=$(git rev-parse "$ref^{commit}") || exit 4
for spec in m1766:cuda m1766vk:vk; do
  IFS=: read n kind <<< "$spec"; t=$W/f25-$n
  [ -d $t ] || git worktree add -q --detach $t $ref || { echo "f25-$n worktree failed" >> $B/status; continue; }
  [ "$(git -C $t rev-parse HEAD)" = "$ref" ] || { echo "f25-$n exists at another commit" >> $B/status; continue; }
  if [ $kind = vk ]; then ML="$MAKE_VK"; X="CPATH=$HOME/colibri-bench/vulkansdk/1.4.363.0/x86_64/include"; else ML="$MAKE_CUDA"; X=; fi
  ( cd $t/c && echo "tree=$t sha=$(git rev-parse HEAD) tree_sha=$(git rev-parse "HEAD^{tree}") start=$(date -Is)" &&
    env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=$HOME $X nice -n 15 $ML; echo "EXIT=$? end=$(date -Is)" ) > $B/f25-$n.build.log 2>&1
  echo "f25-$n $(git -C $t rev-parse --short HEAD) $(tail -1 $B/f25-$n.build.log) warnings=$(grep -iE "warning|error" $B/f25-$n.build.log | grep -vc lto-wrapper) bin=$(sha256sum $t/c/deepseek_v4 | cut -c1-16)" >> $B/status
done
echo done-m >> $B/status
