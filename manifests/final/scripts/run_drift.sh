#!/bin/bash
# PR #0 upload-race validation: does greedy decode on upstream dev give different output across
# identical cold runs, and is pr0/dsv4-fixes identical every time?
# shadowfax only. Stops llama.cpp ONCE; the EXIT trap ALWAYS restarts it and waits for /health ok.
# Total llama-down window capped at 90 min (MAX_GPU_S), phase 2 is fitted into what is left.
# Launch (own transient unit, survives an ssh drop):
#   systemd-run --user --unit=colibri-final-drift --collect -p TimeoutStopSec=900 \
#     -p WorkingDirectory=/home/maplemercury/colibri-bench/final /bin/bash /home/maplemercury/colibri-bench/final/run_drift.sh
#   (optional: -E DRIFT_PHASE2=force|off; default auto)
#   progress: tail /home/maplemercury/colibri-bench/final/drift/run.log
#
# Background: on dev 7b55a9b3, decode-mirror uploads (upload()) and in-place recycles
# (dsv4_cuda_tensor_refill_fp4 from pageable slabs past the 20 GiB page-lock cap) use a
# legacy-stream cudaMemcpy that is not ordered against the non-blocking compute stream; pr0
# a2c8ba7a (b2c8ca61, h2d()) queues them on the compute stream.
#
# Phase 1 (always): arms D = dev 7b55a9b3, Z = pr0 tip a2c8ba7a, same env (COMMON_ENV of
#   final_common.sh, one GPU, no MOE_DOUBLE), client bench_decode.py (6 prompts x 300 tokens),
#   3 cold runs each, interleaved D Z / Z D / D Z. Then final_tools.py phase1 compares the
#   full texts per prompt (DECISION=DIFF when dev's 3 runs differ anywhere).
# Phase 2 (only when phase 1 found NO text difference on dev, DRIFT_PHASE2=auto): the same
#   requests on the probe trees /tmp/cw/r6-dev-probe and /tmp/cw/r6-pr0-probe (the same SHAs
#   plus a logging-only patch, final/probe/*.diff, gated by V4_LOGIT_LOG=1: per greedy token
#   "v4_logit tok= logit= hidden=<FNV-1a of the final hidden row>", and "v4_cpu_moe" for any
#   decode token-layer that missed the fused GPU path). Prompts per run (bench_decode_sub.py,
#   the first N of bench_decode.py's prompts) are sized from phase 1's per-prompt time so the
#   whole window stays under 90 min: 4 runs Dp Zp Dp Zp with the largest N that fits (N >= 1;
#   with phase 1 at ~71 min that is N = 1, 300 tokens per run), else Dp Dp (+ Zp if time is
#   left). Comparison: first differing logit/hidden line between runs (summary.txt).
# Output: /home/maplemercury/colibri-bench/final/drift/<arm>/s<k>/, phase1.json, meta/, run.log,
# summary.{txt,json}; packed as final/drift.tgz + .sha256.
# Expected GPU window: phase 1 ~72 min (6 x ~11.8 min), phase 2 <= the rest of the 90 min.

EXP=drift
MAX_GPU_S=$((90 * 60))
source /home/maplemercury/colibri-bench/final/final_common.sh

DEV_SHA=7b55a9b3792527f52d1a2337fd9e312b9a7983e3
PR0_SHA=a2c8ba7a149a31a781e84a992aa980afdb93497d
DEV=/tmp/cw/r6-dev
PR0=/tmp/cw/r6-pr0
DEVP=/tmp/cw/r6-dev-probe
PR0P=/tmp/cw/r6-pr0-probe
PHASE2=${DRIFT_PHASE2:-auto}
MODE_RE='mirror-reserve|moe-batch=|moe-double|DSV4 CUDA\] device|expert-mirrors|out of memory'
EST1=720        # phase-1 sample estimate (s): ~690 s of client + start/stop
TMO1=1200

probe_tree_ok() { # tree sha
    local t=$1 sha=$2 name; name=$(basename "$1")
    ( preflight_tree "$t" "$sha" 1 ) || return 1
    [ -f "$FINAL/probe/$name.diff" ] || return 1
    [ "$(git -C "$t" diff | sha256sum | cut -c1-64)" = "$(sha256sum < "$FINAL/probe/$name.diff" | cut -c1-64)" ] || return 1
    strings "$t/c/deepseek_v4" | grep -q 'v4_logit tok=' || return 1
}

lock_or_exit
log "run_drift.sh start (pid $$), DRIFT_PHASE2=$PHASE2"
preflight_common bench_decode.py bench_decode_sub.py
preflight_tree $DEV $DEV_SHA
preflight_tree $PR0 $PR0_SHA
PROBE_OK=0
if probe_tree_ok $DEVP $DEV_SHA && probe_tree_ok $PR0P $PR0_SHA; then PROBE_OK=1; else log "probe trees not usable: phase 2 disabled"; fi

arm_trap
if [ $PROBE_OK = 1 ]; then capture_meta $DEV $PR0 $DEVP $PR0P; cp "$FINAL"/probe/* "$OUT/meta/"; else capture_meta $DEV $PR0; fi
stop_llama

# ---------------- phase 1
declare -A TREE
TREE[D]=$DEV; TREE[Z]=$PR0; TREE[Dp]=$DEVP; TREE[Zp]=$PR0P
for k in 1 2 3; do
    for arm in $(rotate $((k - 1)) D Z); do
        if over_budget $(est_for $arm $EST1); then log "budget: skipping $arm s$k (GPU window $(gpu_elapsed) s)"; continue; fi
        sample $arm $k ${TREE[$arm]} decode $TMO1
    done
done
P1OUT=$(python3 "$FINAL/final_tools.py" phase1 "$OUT" 2>&1)
echo "$P1OUT" | tee -a "$OUT/run.log"
DECISION=$(echo "$P1OUT" | sed -n 's/^DECISION=//p')
PER_PROMPT_S=$(echo "$P1OUT" | sed -n 's/^PER_PROMPT_S=//p'); PER_PROMPT_S=${PER_PROMPT_S:-120}
READY_S=$(echo "$P1OUT" | sed -n 's/^READY_S=//p'); READY_S=${READY_S:-30}
[ "$PER_PROMPT_S" -ge 30 ] 2>/dev/null || PER_PROMPT_S=120
[ "$READY_S" -ge 0 ] 2>/dev/null || READY_S=30
log "phase 1: DECISION=$DECISION per_prompt=${PER_PROMPT_S}s ready=${READY_S}s GPU window $(gpu_elapsed) s"

# ---------------- phase 2 (probe trees)
RUN2=0
case "$PHASE2" in
    off)   log "phase 2: off by DRIFT_PHASE2" ;;
    force) RUN2=1 ;;
    *)     if [ "$DECISION" = NODIFF ]; then RUN2=1; else log "phase 2: not needed/possible (DECISION=$DECISION)"; fi ;;
esac
[ $RUN2 = 1 ] && [ $PROBE_OK = 0 ] && { log "phase 2: wanted but the probe trees are not usable"; RUN2=0; }
if [ $RUN2 = 1 ]; then
    OVER=45                                         # stop + idle wait per run
    REM=$(( MAX_GPU_S - $(gpu_elapsed) - 240 ))
    fit() { local n=$(( ($1 - READY_S - OVER) / PER_PROMPT_S )); [ $n -gt 6 ] && n=6; [ $n -lt 0 ] && n=0; echo $n; }
    N4=$(fit $(( REM / 4 ))); N2=$(fit $(( REM / 2 )))
    if [ "$N4" -ge 1 ]; then PLAN="Dp:1 Zp:1 Dp:2 Zp:2"; N=$N4
    elif [ "$N2" -ge 1 ]; then PLAN="Dp:1 Dp:2 Zp:1 Zp:2"; N=$N2
    else PLAN=""; N=0; fi
    log "phase 2: remaining ${REM}s, N4=$N4 N2=$N2 -> plan '$PLAN' with up to $N prompts per run"
    for item in $PLAN; do
        arm=${item%%:*}; k=${item#*:}
        rem=$(( MAX_GPU_S - $(gpu_elapsed) - 240 ))
        n=$(fit "$rem"); [ "$n" -gt "$N" ] && n=$N
        if [ "$n" -lt 1 ]; then log "budget: skipping $arm s$k (remaining ${rem}s)"; continue; fi
        tmo=$(( n * PER_PROMPT_S * 3 / 2 + 60 ))
        cap=$(( rem - READY_S - OVER )); [ $tmo -gt $cap ] && tmo=$cap
        sample $arm $k ${TREE[$arm]} decodesub:$n $tmo V4_LOGIT_LOG=1
        log "$arm s$k: $(grep -c '^v4_logit ' "$OUT/$arm/s$k/serve.log") logit lines, $(grep -c '^v4_cpu_moe ' "$OUT/$arm/s$k/serve.log") cpu-moe lines"
    done
fi
finish drift
