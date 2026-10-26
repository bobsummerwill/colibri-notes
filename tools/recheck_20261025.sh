#!/bin/bash
# recheck_20261025.sh (counter, 2026-10-25): the branches rebased onto upstream dev 77530b49, checked before any GPU run.
#   1. make -C c check on dev and on each tip (3 at a time, each in its own worktree ~/Projects/wt-r-<name>)
#   2. per commit, PR #2 and the split stack: the five compiles + the dsv4/v4 python tests (as pr2-linear-percommit.txt)
#   3. Vulkan (VK=1): build warnings, vk-memtype-check, vk-tier-check, and test_vk_chain on Lavapipe, for dev, fix B,
#      the initializers and the subgroup env; the subgroup env also on the iGPU at COLI_VK_SUBGROUP=16
# Results: ~/colibri-notes/manifests/counter-20261025/
set -u
P=/home/unagi/Projects; R=$P/colibri
M=/home/unagi/colibri-notes/manifests/counter-20261025; mkdir -p "$M/logs"
declare -A TIP=([dev]=upstream/dev [pr2]=docs5/pr2-two-gpu-decode [split1]=split/pr2-1-guard [split2]=split/pr2-2-settings
                [split3]=split/pr2-3-vram-first [fixa]=fix/dsv4-mirror-guard-overshoot [fixb]=fix/vk-host-image-alloc
                [init]=fix/vk-chain-test-initializers [sg]=fix/vk-subgroup-size-env)
for n in "${!TIP[@]}"; do
    w=$P/wt-r-$n; sha=$(git -C "$R" rev-parse "${TIP[$n]}")
    [ -d "$w" ] || git -C "$R" worktree add -q --detach "$w" "$sha"
    git -C "$w" checkout -q --detach "$sha"
done
echo "# make -C c check, counter ($(gcc --version | head -1)), $(date -Is), dev $(git -C $R rev-parse --short upstream/dev)" > "$M/make-check.txt"
check() { # name
    local n=$1 w=$P/wt-r-$1 t0=$(date +%s)
    make -C "$w/c" check > "$M/logs/check-$n.log" 2>&1; local rc=$?
    echo "$n ${TIP[$n]} $(git -C "$w" rev-parse --short HEAD) make_check=$rc $(grep -E '^Ran [0-9]+ tests' "$M/logs/check-$n.log" | tail -1) $(tail -5 "$M/logs/check-$n.log" | grep -E '^(OK|FAILED)' | tail -1) $(( $(date +%s) - t0 ))s" >> "$M/make-check.txt"
}
# 1. three at a time
N=0
for n in dev pr2 fixa fixb init sg split1 split2 split3; do
    check "$n" & N=$((N + 1))
    [ $((N % 3)) = 0 ] && wait
done
wait

# 2. per commit
CF="-O2 -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function -fopenmp -pthread -march=native -fsyntax-only"
percommit() { # range out
    local w=$P/wt-r-pc c
    [ -d "$w" ] || git -C "$R" worktree add -q --detach "$w" upstream/dev
    { echo "# Per-commit check, counter ($(gcc --version | head -1)), $(date -Is), range $1"
      echo "# (1) make -f Makefile.deepseek-v4 deepseek-v4 (clean) (2) deepseek_v4.c -DCOLI_V4_GPU_TIER -DCOLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER (3) +COLI_VULKAN (4) tests/test_deepseek_v4.c -DCOLI_V4_TEST_HOOKS (5) dsv4/v4 python tests present at the commit"; } > "$2"
    for c in $(git -C "$R" rev-list --reverse "$1"); do
        git -C "$w" checkout -q --detach "$c"
        ( cd "$w/c" && make -f Makefile.deepseek-v4 deepseek-v4-clean >/dev/null 2>&1; rm -f deepseek_v4 ./*.o
          make -f Makefile.deepseek-v4 deepseek-v4 -j12 > "$M/logs/pc-build.log" 2>&1; b=$?
          gcc $CF -DCOLI_V4_GPU_TIER -DCOLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER deepseek_v4.c > "$M/logs/pc-gt.log" 2>&1; g=$?
          gcc $CF -DCOLI_V4_GPU_TIER -DCOLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER -DCOLI_VULKAN deepseek_v4.c > "$M/logs/pc-vk.log" 2>&1; v=$?
          gcc $CF -DCOLI_V4_TEST_HOOKS -I. tests/test_deepseek_v4.c > "$M/logs/pc-test.log" 2>&1; t=$?
          py=$(ls tests/test_dsv4*.py tests/test_v4*.py 2>/dev/null | sed 's#tests/##; s#\.py$##' | tr '\n' ' ')
          (cd tests && python3 -m unittest $py) > "$M/logs/pc-py.log" 2>&1; p=$?
          d=$(cat "$M"/logs/pc-{build,gt,vk,test}.log | grep -E 'warning:|error:' | grep -vc 'lto-wrapper')
          echo "$(git rev-parse --short=10 HEAD) build=$b gpu_tier=$g vk=$v test=$t py=$p diag=$d  $(git log -1 --format=%s | cut -c1-80)" ) >> "$2"
    done
}
percommit upstream/dev..docs5/pr2-two-gpu-decode "$M/pr2-percommit.txt"
percommit upstream/dev..split/pr2-4-two-gpu "$M/pr2-split-percommit.txt"
percommit upstream/dev..fix/dsv4-mirror-guard-overshoot "$M/fixa-percommit.txt"

# 3. Vulkan
LVP=/usr/share/vulkan/icd.d/lvp_icd.json
echo "# Vulkan, counter, $(date -Is), $(vulkaninfo --summary 2>/dev/null | grep -m1 driverInfo | xargs)" > "$M/vulkan.txt"
for n in dev fixb init sg; do
    w=$P/wt-r-$n
    ( cd "$w/c" && make clean >/dev/null 2>&1; make VK=1 -j12 tests/test_vk_tier tests/test_vk_memtype tests/test_vk_chain > "$M/logs/vk-build-$n.log" 2>&1 ); rc=$?
    echo "$n $(git -C "$w" rev-parse --short HEAD) build rc=$rc warnings=$(grep -c 'warning:' "$M/logs/vk-build-$n.log")" >> "$M/vulkan.txt"
    for t in vk-memtype-check vk-tier-check; do
        ( cd "$w/c" && timeout 900 make VK=1 $t ) > "$M/logs/$t-$n.log" 2>&1
        echo "$n $t rc=$?" >> "$M/vulkan.txt"
    done
    t0=$(date +%s); ( cd "$w/c" && VK_DRIVER_FILES=$LVP timeout 1800 stdbuf -oL ./tests/test_vk_chain shaders/qmatmul.spv ) > "$M/logs/lvp-chain-$n.log" 2>&1
    echo "$n test_vk_chain lavapipe rc=$? $(( $(date +%s) - t0 ))s $(tail -1 "$M/logs/lvp-chain-$n.log")" >> "$M/vulkan.txt"
done
w=$P/wt-r-sg; t0=$(date +%s)
( cd "$w/c" && VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.json COLI_VK_SUBGROUP=16 timeout 2400 stdbuf -oL ./tests/test_vk_chain shaders/qmatmul.spv ) > "$M/logs/igpu-chain-sg16.log" 2>&1
echo "sg test_vk_chain iGPU COLI_VK_SUBGROUP=16 rc=$? $(( $(date +%s) - t0 ))s $(tail -1 "$M/logs/igpu-chain-sg16.log")" >> "$M/vulkan.txt"
echo "ALL DONE $(date -Is)" >> "$M/make-check.txt"
