#!/bin/bash
# final26: C+ = fork branch samebase/cplus (PR #2 5b3f4601 merged with samebase M): CUDA (final6 make line, sm_86 NO_TC)
# -> cw22/f26-cplus, pure-Vulkan (CUDA=1 VK=1, final22 line) -> cw22/f26-cplusvk. Run only after final25 has ended.
# usage: build_f26_c.sh [COMMIT]   (default: the fork's samebase/cplus tip)
W=~/colibri-bench/cw22
B=~/colibri-bench/final26/builds; mkdir -p $B
G=$HOME/colibri-bench/vulkansdk/1.4.363.0/x86_64/bin/glslc
MAKE_CUDA="make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j14"
MAKE_VK="make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 VK=1 GLSLC=$G CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j14"
systemctl --user is-active -q colibri-final25 && { echo "final25 still running; not building" >> $B/status; exit 5; }
cd $W/repo && git fetch -q https://github.com/bobsummerwill/colibri.git samebase/cplus:refs/remotes/fork/samebase/cplus || { echo "fetch failed" >> $B/status; exit 4; }
ref=${1:-$(git rev-parse fork/samebase/cplus)}; ref=$(git rev-parse "$ref^{commit}") || exit 4
for spec in cplus:cuda cplusvk:vk; do
  IFS=: read n kind <<< "$spec"; t=$W/f26-$n
  [ -d $t ] || git worktree add -q --detach $t $ref || { echo "f26-$n worktree failed" >> $B/status; continue; }
  [ "$(git -C $t rev-parse HEAD)" = "$ref" ] || { echo "f26-$n exists at another commit" >> $B/status; continue; }
  if [ $kind = vk ]; then ML="$MAKE_VK"; X="CPATH=$HOME/colibri-bench/vulkansdk/1.4.363.0/x86_64/include"; else ML="$MAKE_CUDA"; X=; fi
  ( cd $t/c && echo "tree=$t sha=$(git rev-parse HEAD) tree_sha=$(git rev-parse "HEAD^{tree}") start=$(date -Is)" &&
    env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=$HOME $X nice -n 15 $ML; echo "EXIT=$? end=$(date -Is)" ) > $B/f26-$n.build.log 2>&1
  # nvcc's own diagnostics, kept verbatim (lto-wrapper's serial-compilation notes are not warnings)
  grep -iE "warning|error" $B/f26-$n.build.log | grep -v lto-wrapper > $B/f26-$n.diagnostics.txt
  echo "f26-$n $(git -C $t rev-parse --short HEAD) $(tail -1 $B/f26-$n.build.log) warnings=$(wc -l < $B/f26-$n.diagnostics.txt) bin=$(sha256sum $t/c/deepseek_v4 2>/dev/null | cut -c1-16)" >> $B/status
done
echo done-c >> $B/status
