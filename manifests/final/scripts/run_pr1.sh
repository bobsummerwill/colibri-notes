#!/bin/bash
# PR #1 prefill benchmark, final tip pr1/reap-prefill 93b741839677ec457868fd71ab668cc1249b9a22.
# shadowfax only. Stops llama.cpp ONCE; the EXIT trap ALWAYS restarts it and waits for /health ok.
# Launch (own transient unit, survives an ssh drop):
#   systemd-run --user --unit=colibri-final-pr1 --collect -p TimeoutStopSec=900 \
#     -p WorkingDirectory=/home/maplemercury/colibri-bench/final /bin/bash /home/maplemercury/colibri-bench/final/run_pr1.sh
#   progress: tail /home/maplemercury/colibri-bench/final/pr1/run.log
#
# Arms (all: --ram 40 --ctx 8192, COMMON_ENV of final_common.sh, a cold server per sample,
# client bench_pr1.py: request 1 = 1992-token prompt / 100 tokens, request 2 = 3005-token / 60):
#   A  dev 7b55a9b3 (reference: no GPU batch for 132 experts, CPU union)   both requests
#   B  pr1 tip 93b74183                                                     both
#   C  B + COLI_CUDA_MOE_DOUBLE=1   (double bank)                           both
#   E  B + DSV4_MOE_EXPERT_MAJOR=0  (expert-major MoE kernel off)           first request
#   F  B + DSV4_MM_WARP=0           (warp dense kernel off)                 first request
#   3 samples per arm, interleaved: round k runs the arms rotated by k-1
#   (A B C E F / B C E F A / C E F A B). Then one verify run:
#   V  B + DSV4_MOE_EXPERT_MAJOR_VERIFY=1 DSV4_MM_WARP_VERIFY=1             first request
#      (both kernels run beside the old ones, the old results are kept, bit mismatches counted;
#      the counters print every 500 calls and on every mismatch -> V/s1/verify_counters.txt)
# Switch names checked against docs/deepseek-v4.md at 93b74183 (env table: COLI_CUDA_MOE_DOUBLE,
# DSV4_MOE_EXPERT_MAJOR, DSV4_MM_WARP, DSV4_MM_WARP_VERIFY, "_VERIFY=1") and, at run time, against
# the getenv() calls of the tree (preflight).
# Output: /home/maplemercury/colibri-bench/final/pr1/<arm>/s<k>/ (serve.log, bench.jsonl, sample.txt,
# engine_environ.txt, procs.txt, nvsmi.csv, gpu_*.csv, mode.txt), meta/, run.log, summary.{txt,json};
# packed as final/pr1.tgz + .sha256.
# Expected GPU window ~72 min (A ~10 min, B ~4, C ~3.7, E ~2.5, F ~1.8 per sample; V ~3).

EXP=pr1
MAX_GPU_S=$((150 * 60))     # hard cap: no new sample once the llama-down window would pass 150 min
source /home/maplemercury/colibri-bench/final/final_common.sh

DEV_SHA=7b55a9b3792527f52d1a2337fd9e312b9a7983e3
PR1_SHA=93b741839677ec457868fd71ab668cc1249b9a22
DEV=/tmp/cw/r6-dev
PR1=/tmp/cw/r6-pr1
MODE_RE='moe-batch=|moe-double|mirror-reserve|verify|_VERIFY|EXPERT_MAJOR|MM_WARP|DSV4 CUDA\] device|out of memory'

declare -A TREE CLIENT TMO XENV
TREE[A]=$DEV; CLIENT[A]=pr1:both;  TMO[A]=1800; XENV[A]=""
TREE[B]=$PR1; CLIENT[B]=pr1:both;  TMO[B]=900;  XENV[B]=""
TREE[C]=$PR1; CLIENT[C]=pr1:both;  TMO[C]=900;  XENV[C]="COLI_CUDA_MOE_DOUBLE=1"
TREE[E]=$PR1; CLIENT[E]=pr1:first; TMO[E]=600;  XENV[E]="DSV4_MOE_EXPERT_MAJOR=0"
TREE[F]=$PR1; CLIENT[F]=pr1:first; TMO[F]=600;  XENV[F]="DSV4_MM_WARP=0"
TREE[V]=$PR1; CLIENT[V]=pr1:first; TMO[V]=900;  XENV[V]="DSV4_MOE_EXPERT_MAJOR_VERIFY=1 DSV4_MM_WARP_VERIFY=1"
EST_S_A=620; EST_S=260      # budget guard estimates (A / any other arm)

lock_or_exit
log "run_pr1.sh start (pid $$)"
preflight_common bench_pr1.py
preflight_tree $DEV $DEV_SHA
preflight_tree $PR1 $PR1_SHA
preflight_switch $PR1 COLI_CUDA_MOE_DOUBLE DSV4_MOE_EXPERT_MAJOR DSV4_MM_WARP DSV4_MOE_EXPERT_MAJOR_VERIFY DSV4_MM_WARP_VERIFY
grep -q 'COLI_CUDA_MOE_DOUBLE' $PR1/docs/deepseek-v4.md && grep -q 'DSV4_MOE_EXPERT_MAJOR' $PR1/docs/deepseek-v4.md \
    && grep -q 'DSV4_MM_WARP_VERIFY' $PR1/docs/deepseek-v4.md || die_pre "switch names not in $PR1/docs/deepseek-v4.md"

arm_trap
capture_meta $DEV $PR1
stop_llama

ORDER="A B C E F"
for k in 1 2 3; do
    for arm in $(rotate $((k - 1)) $ORDER); do
        est=$EST_S; [ $arm = A ] && est=$EST_S_A
        est=$(est_for $arm $est)
        if over_budget $est; then log "budget: skipping $arm s$k (GPU window $(gpu_elapsed) s)"; continue; fi
        # shellcheck disable=SC2086
        sample $arm $k ${TREE[$arm]} ${CLIENT[$arm]} ${TMO[$arm]} ${XENV[$arm]}
    done
done
if over_budget $EST_S; then
    log "budget: skipping V"
else
    sample V 1 ${TREE[V]} ${CLIENT[V]} ${TMO[V]} ${XENV[V]}
    { grep -hE "verify calls=|_VERIFY" "$OUT/V/s1/serve.log"; } > "$OUT/V/s1/verify_counters.txt"
    log "verify counters (last lines):"; tail -4 "$OUT/V/s1/verify_counters.txt" | tee -a "$OUT/run.log"
    log "verify lines with a nonzero mismatch: $(grep -cE '(mismatches|fc1_mismatch|fc2_mismatch)=[1-9]' "$OUT/V/s1/verify_counters.txt")"
fi
finish pr1
