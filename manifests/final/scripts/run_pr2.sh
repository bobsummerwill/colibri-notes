#!/bin/bash
# PR #2 two-GPU decode re-measurement, final tip pr2/two-gpu-decode d0e19de23c3baf314ee932347c42d5f6d49ef4fb.
# shadowfax only. Stops llama.cpp ONCE; the EXIT trap ALWAYS restarts it and waits for /health ok.
# Launch (own transient unit, survives an ssh drop):
#   systemd-run --user --unit=colibri-final-pr2 --collect -p TimeoutStopSec=900 \
#     -p WorkingDirectory=/home/maplemercury/colibri-bench/final /bin/bash /home/maplemercury/colibri-bench/final/run_pr2.sh
#   progress: tail /home/maplemercury/colibri-bench/final/pr2/run.log
#
# Client bench_decode.py: 6 short prompts x 300 tokens, streamed, greedy (temperature ignored);
# decode tok/s per prompt, full text. --ram 40 --ctx 8192, COMMON_ENV of final_common.sh, a cold
# server per sample, environment built with env -i so COLI_CUDA_MOE_DOUBLE is unset in EVERY arm.
# V4_SERVE_STATS_LOG=1 everywhere (stderr stats lines only, no behaviour change).
# Arms (switches as documented in docs/deepseek-v4.md at d0e19de2, "Decode on two GPUs" + env table):
#   G1  one GPU, no PR #2 switch                                     (tree pr2)
#   X1  G1 + V4_GPU_RESIDENT_FIRST=1        (VRAM-first lookups)
#   X2  X1 + DSV4_CUDA_PEER_DEVICE=1 DSV4_CUDA_PEER_COMPUTE=0   (peer pool on GPU1, read in place over NVLink)
#   X3  X1 + DSV4_CUDA_PEER_DEVICE=1        (peer compute on GPU1; DSV4_CUDA_PEER_COMPUTE default 1)
#   3 samples each, interleaved: round k runs the arms rotated by k-1 (G1 X1 X2 X3 / X1 X2 X3 G1 / X2 X3 G1 X1);
#   then P1 = pr1 tip 93b74183 with G1's env, one sample (no-regression sanity check; pr1 has no
#   V4_SERVE_STATS_LOG, the variable is set but unread there).
# Dependencies: V4_GPU_RESIDENT_FIRST only acts with the dual expert loader, which the Linux
# Makefile compiles in (-DCOLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER); V4_LOADER_LANES=3 (GPU tier)
# is in COMMON_ENV; peer compute needs DSV4_CUDA_BATCHED (default 1), same SM on both cards, P2P.
# Mode proof per sample (mode.txt, checked again in summary.txt): "v4_gpu peer-mirrors device=1 ...
# mode=p2p-read" (X2) / "mode=compute" (X3), none for G1/X1/P1; v4_serve_turn
# vram_first_hits_total = 0 (G1) / > 0 (X1-X3); the engine's real environment in engine_environ.txt.
# Output: /home/maplemercury/colibri-bench/final/pr2/<arm>/s<k>/, meta/, run.log, summary.{txt,json};
# packed as final/pr2.tgz + .sha256.
# Expected GPU window ~2 h 20 min (per sample: G1 ~11.5 min, X1 ~11, X2 ~9.5, X3 ~8.5, P1 ~11.5).

EXP=pr2
MAX_GPU_S=$((200 * 60))     # hard cap: no new sample once the llama-down window would pass 200 min
source /home/maplemercury/colibri-bench/final/final_common.sh

PR1_SHA=93b741839677ec457868fd71ab668cc1249b9a22
PR2_SHA=d0e19de23c3baf314ee932347c42d5f6d49ef4fb
PR1=/tmp/cw/r6-pr1
PR2=/tmp/cw/r6-pr2
MODE_RE='peer-mirrors|peer mirror pool|peer device|mirror-reserve|moe-batch=|moe-double|DSV4 CUDA\] device|^v4_serve_turn|^v4_dec_moe|out of memory'

declare -A TREE XENV EST
TREE[G1]=$PR2; XENV[G1]="V4_SERVE_STATS_LOG=1";                                                       EST[G1]=700
TREE[X1]=$PR2; XENV[X1]="V4_SERVE_STATS_LOG=1 V4_GPU_RESIDENT_FIRST=1";                               EST[X1]=690
TREE[X2]=$PR2; XENV[X2]="V4_SERVE_STATS_LOG=1 V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1 DSV4_CUDA_PEER_COMPUTE=0"; EST[X2]=600
TREE[X3]=$PR2; XENV[X3]="V4_SERVE_STATS_LOG=1 V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1";       EST[X3]=530
TREE[P1]=$PR1; XENV[P1]="V4_SERVE_STATS_LOG=1";                                                       EST[P1]=700
TMO=1500

lock_or_exit
log "run_pr2.sh start (pid $$)"
preflight_common bench_decode.py
preflight_tree $PR1 $PR1_SHA
preflight_tree $PR2 $PR2_SHA
preflight_switch $PR2 V4_GPU_RESIDENT_FIRST DSV4_CUDA_PEER_DEVICE DSV4_CUDA_PEER_COMPUTE V4_SERVE_STATS_LOG COLI_CUDA_MOE_DOUBLE
for v in V4_GPU_RESIDENT_FIRST DSV4_CUDA_PEER_DEVICE DSV4_CUDA_PEER_COMPUTE V4_SERVE_STATS_LOG; do
    grep -q "$v" $PR2/docs/deepseek-v4.md || die_pre "$v not in $PR2/docs/deepseek-v4.md"
done
grep -q -- '-DCOLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER' $PR2/c/Makefile.deepseek-v4 || die_pre "dual loader not in the Makefile"
[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2 ] || die_pre "second GPU not visible"

arm_trap
capture_meta $PR1 $PR2
nvidia-smi topo -p2p r > "$OUT/meta/p2p_read.txt" 2>&1
stop_llama

ORDER="G1 X1 X2 X3"
for k in 1 2 3; do
    for arm in $(rotate $((k - 1)) $ORDER); do
        if over_budget $(est_for $arm ${EST[$arm]}); then log "budget: skipping $arm s$k (GPU window $(gpu_elapsed) s)"; continue; fi
        # shellcheck disable=SC2086
        sample $arm $k ${TREE[$arm]} decode $TMO ${XENV[$arm]}
        log "mode $arm s$k: $(grep -hE 'peer-mirrors|peer mirror pool|peer device' "$OUT/$arm/s$k/mode.txt" | head -2 | tr '\n' ' ') $(grep -h '^v4_serve_turn' "$OUT/$arm/s$k/mode.txt" | tail -1 | grep -oE 'vram_first_hits_total=[0-9]+')"
    done
done
if over_budget ${EST[P1]}; then
    log "budget: skipping P1"
else
    sample P1 1 ${TREE[P1]} decode $TMO ${XENV[P1]}
fi
finish pr2
