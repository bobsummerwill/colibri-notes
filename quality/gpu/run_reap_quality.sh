#!/bin/bash
# REAP arm of the quality comparison (protocol v2). shadowfax only, user maplemercury.
# Stops llama.cpp (llama-server-active) ONCE; the EXIT trap of final2/final_common.sh ALWAYS
# restarts it and polls /health until ok (restore()). Nothing is killed by pattern: the colibri
# server runs as its own transient unit (colibri-quality-serve) and is stopped by unit name.
#
# Launch (own transient unit, survives an ssh drop; RuntimeMaxSec is only a backstop, its SIGTERM
# also ends in the EXIT trap, and TimeoutStopSec leaves the trap 15 min to restore llama):
#   systemd-run --user --unit=colibri-quality --collect -p TimeoutStopSec=900 -p RuntimeMaxSec=11400 \
#     -p WorkingDirectory=/home/maplemercury/colibri-bench/quality \
#     /bin/bash /home/maplemercury/colibri-bench/quality/gpu/run_reap_quality.sh
#   progress: tail ~/colibri-bench/quality/gpu/quality/run.log ; answers: results/reap.jsonl
# Preflight only (CPU only, takes and releases the locks, never touches llama):
#   PREFLIGHT_ONLY=1 bash ~/colibri-bench/quality/gpu/run_reap_quality.sh
#
# Order (protocol v2): the Qwen arms are already in results/ (the gate below refuses otherwise).
# Window: llama stop (<= 2 min) + server ready (seconds) + client (soft budget BUDGET_MIN for
# starting prompts, hard stop HARD_MIN aborts an answer in flight) + restore (<= 3 min).
# Expected ~100-115 min with Qwen-like answer lengths; worst case ~HARD_MIN + 8 min.

EXP=quality
[ "${PREFLIGHT_ONLY:-0}" = 1 ] && EXP=quality-preflight
BUDGET_MIN=${BUDGET_MIN:-120}
HARD_MIN=${HARD_MIN:-150}
MAX_GPU_S=$(( (HARD_MIN + 15) * 60 ))   # recorded; the client's hard stop is what bounds the window

COMMON=/home/maplemercury/colibri-bench/final2/final_common.sh
COMMON_SHA=87f3fb9160254814879c4f951e2271dd3bfd541fbdd9cbfa5074e81538df4588   # as used by run_f42.sh / run_race.sh
if [ "$(sha256sum < "$COMMON" | cut -c1-64)" != "$COMMON_SHA" ]; then
    echo "PREFLIGHT FAILED: $COMMON changed (sha256 != $COMMON_SHA)" >&2
    exit 4
fi
# shellcheck source=/dev/null
source "$COMMON"

Q=/home/maplemercury/colibri-bench/quality
OUT=$Q/gpu/$EXP                    # evidence next to the quality run (final_common uses $OUT at call time)
EXP_UNIT=colibri-quality           # this runner's unit (preflight_common allows only it)
SERVE_UNIT=colibri-quality-serve
PR2_SHA=d0e19de23c3baf314ee932347c42d5f6d49ef4fb
PR2=/tmp/cw/r6-pr2
# The brief's serve environment = COMMON_ENV of final_common.sh + these two (two-GPU decode)
EXTRA_ENV=(V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1)
QWEN_RESULTS=${QWEN_RESULTS:-}     # default: results/qwen-hermes.jsonl,results/qwen-greedy.jsonl

lock_or_exit                       # final2/.gpu.lock AND final/.gpu.lock, flock -n, before any trap
log "run_reap_quality.sh start (pid $$) budget=${BUDGET_MIN} hard=${HARD_MIN} min"
preflight_common                   # model dir, port 8090 free, no other colibri unit, final_tools selftest
preflight_tree $PR2 $PR2_SHA       # worktree at d0e19de2, clean, built, build log EXIT=0
preflight_switch $PR2 V4_GPU_RESIDENT_FIRST DSV4_CUDA_PEER_DEVICE COLI_CUDA_MOE_BATCH
preflight_doc $PR2 V4_GPU_RESIDENT_FIRST DSV4_CUDA_PEER_DEVICE
for f in prompts.json clients/qclient_common.py clients/colibri_client.py; do
    [ -f "$Q/$f" ] || die_pre "$Q/$f missing"
done
( cd "$Q" && python3 clients/colibri_client.py --dry-run --budget-min "$BUDGET_MIN" --hard-stop-min "$HARD_MIN" \
      ${QWEN_RESULTS:+--qwen-results "$QWEN_RESULTS"} ) > "$OUT/gate.txt" 2>&1 \
    || { cat "$OUT/gate.txt" >> "$OUT/run.log"; die_pre "quality gate failed (Qwen arms missing or truncated), see $OUT/gate.txt"; }
cat "$OUT/gate.txt" >> "$OUT/run.log"

if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then log "PREFLIGHT_ONLY: all checks passed; exiting before llama is touched"; exit 0; fi
arm_trap
capture_meta $PR2
cp "$Q/prompts.json" "$Q"/clients/*.py "$0" "$OUT/meta/" 2>/dev/null
(cd "$OUT/meta" && sha256sum prompts.json qclient_common.py colibri_client.py "$(basename "$0")" > quality_sha256s.txt)
stop_llama                         # T0 = start of the llama-down window; exits if GPUs stay busy

start_serve() { # attempt
    local d=$OUT/serve$1
    mkdir -p "$d"
    local envs=()
    mapfile -t envs < <(merge_env "${BASE_ENV[@]}" "${COMMON_ENV[@]}" "${EXTRA_ENV[@]}")
    gpu_idle_wait 120 || { log "GPUs not idle before serve $1"; return 1; }
    local i
    for i in $(seq 1 30); do ss -ltnH "sport = :$PORT" | grep -q . || break; sleep 2; done
    ss -ltnH "sport = :$PORT" | grep -q . && { log "port $PORT busy before serve $1"; return 1; }
    { echo "server_env=${envs[*]}"; echo "serve_cmd=python3 $PR2/c/coli serve $SERVE_ARGS"; echo "start=$(date -Is)"; } > "$d/serve.txt"
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
        --format=csv -lms "$NVSMI_MS" > "$d/nvsmi.csv" 2>&1 &
    NVSMI_PID=$!
    systemd-run --user --unit="$SERVE_UNIT" --collect -p WorkingDirectory="$PR2" \
        /usr/bin/env -i "${envs[@]}" /bin/bash -c "exec python3 $PR2/c/coli serve $SERVE_ARGS > $d/serve.log 2>&1" \
        >> "$OUT/run.log" 2>&1
    local t_start; t_start=$(now)
    while [ $(( $(now) - t_start )) -lt 900 ]; do
        grep -q "listening on" "$d/serve.log" 2>/dev/null && { echo "ready_s=$(( $(now) - t_start ))" >> "$d/serve.txt"; record_procs "$d"; return 0; }
        systemctl --user is-active -q "$SERVE_UNIT" || break
        sleep 2
    done
    log "colibri serve $1 never became ready"
    return 1
}
stop_serve() {
    systemctl --user stop "$SERVE_UNIT" 2>/dev/null
    local i
    for i in $(seq 1 60); do systemctl --user is-active -q "$SERVE_UNIT" || break; sleep 2; done
    stop_sampler
    gpu_idle_wait 120 || log "warning: compute processes remain after stopping $SERVE_UNIT"
}

run_client() { # budget_min hard_min
    ( cd "$Q" && timeout $(( $2 * 60 + 1500 )) python3 clients/colibri_client.py --wait-ready-s 1200 \
          --budget-min "$1" --hard-stop-min "$2" ${QWEN_RESULTS:+--qwen-results "$QWEN_RESULTS"} ) \
        >> "$OUT/client.log" 2>&1
}

crc=99
if start_serve 1; then
    log "serve ready; client starts (GPU window $(gpu_elapsed) s)"
    run_client "$BUDGET_MIN" "$HARD_MIN"; crc=$?
    log "client exit $crc (GPU window $(gpu_elapsed) s)"
fi
stop_serve
# exit 3 = colibri stopped answering mid-run: ONE restart, with whatever budget is left
used_min=$(( $(gpu_elapsed) / 60 ))
if [ "$crc" = 3 ] && [ $(( BUDGET_MIN - used_min )) -ge 15 ]; then
    log "restarting colibri once (used ${used_min} min)"
    if start_serve 2; then
        run_client $(( BUDGET_MIN - used_min )) $(( HARD_MIN - used_min )); crc=$?
        log "client (2nd serve) exit $crc"
    fi
    stop_serve
fi
restore                            # llama back first, then the summary
python3 - "$Q/results/reap.jsonl" <<'PY' 2>&1 | tee -a "$OUT/run.log"
import json, sys, collections
c = collections.Counter(); rows = []
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    if r.get("type") == "result":
        rows.append(r)
last = {}
for r in rows:
    if r["prompt_id"] not in last or r.get("status") == "ok" or last[r["prompt_id"]].get("status") != "ok":
        last[r["prompt_id"]] = r
for pid, r in sorted(last.items(), key=lambda kv: kv[1].get("priority", 99)):
    c[r.get("status")] += 1
    print(f"{pid:18s} {r.get('status'):15s} finish={r.get('finish_reason')} tok={r.get('completion_tokens')} "
          f"expected={r.get('expected_tokens')} total_s={r.get('total_s')} rate={r.get('decode_tok_s')}")
print("status counts:", dict(c))
PY
log "ALL DONE (client exit $crc)"
