#!/bin/bash
# run_c4m26.sh (2026-10-26, computer4) -- Stage 2 of colibri-notes review/PLAN_SAME_BASE_20261026.md: C+ (fork
# samebase/cplus, $C_SHA = PR #2 5b3f4601 merged with samebase M) on one RTX 5070 12 GB and CPU-only, with a same-session
# anchor of the base (dd03a9de, c4m25's trees). Full V4 Flash, --ram 40 --ctx 8192, c4m25's serve environment.
#   GPU (generic CUDA sm_120, CUDA 12.8): C (V4_GPU_RESIDENT_FIRST=1) x2, B x1, interleaved C B C; bench_decode.py 6 x 300.
#   The reserve is c4m25's pilot decision (its bench/meta/gpu_reserve.txt), so C+ runs as M/B/P did; RESERVE=none|2000
#   overrides, and without c4m25's file RESERVE must be given.
#   CPU-only (no-CUDA builds): CC x2, CB x1, bench_decode_cpu.py (the first 2 prompts x 128 tokens).
# MODE=all (gpu, cpu; the default) | gpu | cpu.  C_SHA must be set. Run after c4m25 has finished (its lock is checked).
set -u
W=$HOME/colibri-bench/c4m26
MODE=${MODE:-all}; RESERVE=${RESERVE:-c4m25}
C_SHA=${C_SHA:?set C_SHA (the samebase/cplus commit built into dg/s26c)}
MODEL=$HOME/colibri-models/deepseek-v4-flash
PORT=8090
DG=$HOME/colibri-bench/dg
declare -A TREE=([B]=$DG/s25base [C]=$DG/s26c [CB]=$DG/s25basecpu [CC]=$DG/s26ccpu)
declare -A WANT=([B]=dd03a9de [C]=$C_SHA [CB]=dd03a9de [CC]=$C_SHA)
declare -A XENV=([B]="" [C]="V4_GPU_RESIDENT_FIRST=1" [CB]="" [CC]="")
BASE_ENV=(PATH=/usr/local/bin:/usr/bin:/bin HOME=$HOME USER=$USER LOGNAME=$USER LANG=en_US.UTF-8
          LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64)
GPU_ENV=(COLI_CUDA=1 COLI_GPU=0 DSV4_CUDA_DEVICE=0 CUDA_DENSE=1 COLI_CUDA_PIPE=2 COLI_CUDA_ATTN_BATCH=1
         V4_MOE_REFILL_GROUP=12 V4_LOADER_LANES=3 V4_PREFIX_CKPT_DISK=0)
CPU_ENV=(COLI_CUDA=0 DSV4_CUDA=0 V4_MOE_REFILL_GROUP=12 V4_LOADER_LANES=3 V4_PREFIX_CKPT_DISK=0)
SERVE_ARGS="--model $MODEL --ram 40 --ctx 8192 --host 127.0.0.1 --port $PORT"
ORDER_GPU=${ORDER_GPU:-"C:1 B:1 C:2"}
ORDER_CPU=${ORDER_CPU:-"CC:1 CB:1 CC:2"}

exec 9> "$W/.lock"; flock -n 9 || { echo "another run_c4m26 holds $W/.lock"; exit 4; }
exec 8> "$HOME/colibri-bench/c4m25/.lock"; flock -n 8 || { echo "c4m25 is still running (holds its lock)"; exit 4; }
OUT=$W/bench
[ "$MODE" != cpu ] && [ -d "$OUT" ] && [ -n "$(ls -A "$OUT")" ] && mv "$OUT" "$OUT.old-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT/meta"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/run.log"; }
die() { log "PREFLIGHT FAILED: $*"; exit 4; }

# ---- preflight
for a in "${!TREE[@]}"; do t=${TREE[$a]}
    [ -x "$t/c/deepseek_v4" ] || die "no binary in $t"
    [ -z "$(git -C "$t" status --porcelain --untracked-files=no)" ] || die "$t has local changes"
    [ "$(git -C "$t" rev-parse HEAD)" = "$(git -C "$t" rev-parse "${WANT[$a]}^{commit}")" ] || die "$t is not at ${WANT[$a]}"
    case $a in CB|CC) ldd "$t/c/deepseek_v4" | grep -q cudart && die "$t (CPU arm) links cudart";;
               *) ldd "$t/c/deepseek_v4" | grep -q cudart || die "$t (GPU arm) does not link cudart";; esac
done
[ "$(ls "$MODEL"/model-*-of-*.safetensors | wc -l)" = 48 ] || die "model incomplete"
pgrep -x deepseek_v4 >/dev/null && die "a deepseek_v4 process is running"
ss -ltnH "sport = :$PORT" | grep -q . && die "port $PORT in use"
other=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -v voxtype)
[ -z "$other" ] || die "other GPU processes: $other"
[ "$(df -Pk "$W" | awk 'NR==2{print $4}')" -gt 5000000 ] || die "less than 5 GB free"
{ uname -a; nvidia-smi; free -g; lscpu | grep -E 'Model name|^CPU\(s\)'
  for a in B C CB CC; do t=${TREE[$a]}; echo "$a tree=$t head=$(git -C "$t" rev-parse HEAD) binary=$(sha256sum "$t/c/deepseek_v4" | cut -c1-64)"; done
  echo "GPU_ENV=${GPU_ENV[*]}"; echo "CPU_ENV=${CPU_ENV[*]}"; echo "SERVE_ARGS=$SERVE_ARGS"; echo "MODE=$MODE RESERVE=$RESERVE"; } > "$OUT/meta/host-$(date +%H%M%S).txt" 2>&1
cp "$0" "$W/clients/bench_decode.py" "$W/clients/bench_decode_cpu.py" "$W/summary_c4m26.py" "$OUT/meta/"
log "run_c4m26 start: MODE=$MODE RESERVE=$RESERVE C=$C_SHA"
[ "${PREFLIGHT_ONLY:-0}" = 1 ] && { log "preflight ok (PREFLIGHT_ONLY=1, nothing run)"; exit 0; }

SEQ=$(cat "$OUT/meta/seq" 2>/dev/null || echo 0); SPID=; NPID=
cleanup() { [ -n "$NPID" ] && kill "$NPID" 2>/dev/null; [ -n "$SPID" ] && kill -TERM -- "-$SPID" 2>/dev/null; }
trap 'cleanup; log "signal: exiting"; exit 143' TERM INT HUP
sample() { # OUTDIR ARM K EXTRA_ENV...
    local out=$1 arm=$2 k=$3; shift 3
    local tree=${TREE[$arm]} d=$out/$arm/s$k t0=$(date +%s) client=bench_decode.py tmo=3600 env_arr
    case $arm in CB|CC) env_arr=("${CPU_ENV[@]}"); client=bench_decode_cpu.py; tmo=5400;; *) env_arr=("${GPU_ENV[@]}");; esac
    mkdir -p "$d"; SEQ=$((SEQ + 1)); echo $SEQ > "$OUT/meta/seq"
    for i in $(seq 1 30); do ss -ltnH "sport = :$PORT" | grep -q . || break; sleep 2; done
    { echo "arm=$arm"; echo "sample=$k"; echo "seq=$SEQ"; echo "tree=$tree"; echo "tree_head=$(git -C "$tree" rev-parse HEAD)"
      echo "tree_dirty_lines=$(git -C "$tree" status --porcelain --untracked-files=no | wc -l)"
      echo "binary_sha256=$(sha256sum "$tree/c/deepseek_v4" | cut -c1-64)"
      echo "server_env=${BASE_ENV[*]} ${env_arr[*]} ${XENV[$arm]} $*"; echo "client=$client"; echo "start=$(date -Is)"; } > "$d/sample.txt"
    log "=== sample $SEQ: $arm s$k tree=$tree $*"
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu,clocks.sm \
        --format=csv -lms 5000 > "$d/nvsmi.csv" 2>&1 & NPID=$!
    # shellcheck disable=SC2086
    setsid env -i "${BASE_ENV[@]}" "${env_arr[@]}" ${XENV[$arm]} "$@" bash -c "cd '$tree' && exec python3 c/coli serve $SERVE_ARGS" > "$d/serve.log" 2>&1 &
    SPID=$!
    local ready=0 rc=99
    for i in $(seq 1 600); do grep -q "listening on" "$d/serve.log" 2>/dev/null && { ready=1; break; }; kill -0 "$SPID" 2>/dev/null || break; sleep 2; done
    echo "ready=$ready" >> "$d/sample.txt"
    if [ $ready = 1 ]; then
        timeout $tmo python3 "$W/clients/$client" > "$d/bench.jsonl" 2> "$d/bench.err"; rc=$?
        python3 -I "$W/summary_c4m26.py" --brief "$d/bench.jsonl" > "$d/bench.out" 2>&1
    fi
    echo "client_rc=$rc" >> "$d/sample.txt"
    kill -TERM -- "-$SPID" 2>/dev/null
    for i in $(seq 1 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$SPID" 2>/dev/null; SPID=
    kill "$NPID" 2>/dev/null; NPID=
    echo "wall_s=$(( $(date +%s) - t0 ))" >> "$d/sample.txt"
    log "--- $arm s$k: ready=$ready client_rc=$rc wall=$(( $(date +%s) - t0 ))s $(tr '\n' ' ' < "$d/bench.out" 2>/dev/null)"
    log "check $arm s$k: $(grep -c 'out of memory' "$d/serve.log") OOM lines, $(grep -cE 'Segmentation|Aborted|Traceback' "$d/serve.log") crash lines; $(grep -m4 -E 'resident|placement=|layers=|moe-batch=|mirror-reserve' "$d/serve.log" | tr '\n' ' ' | cut -c1-300)"
}

XRES=""
case $RESERVE in
    c4m25) f=$HOME/colibri-bench/c4m25/bench/meta/gpu_reserve.txt
           [ -f "$f" ] || { log "no $f (c4m25's pilot decision); set RESERVE=none|2000"; exit 4; }
           XRES=$(cat "$f"); log "reserve: c4m25's pilot decision '${XRES:-no reserve setting}'";;
    2000) XRES="DSV4_CUDA_VRAM_RESERVE_MB=2000";;
    none) XRES="";;
    *) log "RESERVE must be c4m25, none or 2000"; exit 4;;
esac
echo "$XRES" > "$OUT/meta/gpu_reserve.txt"
if [ "$MODE" = all ] || [ "$MODE" = gpu ]; then
    for item in $ORDER_GPU; do sample "$OUT" "${item%%:*}" "${item#*:}" $XRES; done
fi
if [ "$MODE" = all ] || [ "$MODE" = cpu ]; then
    for item in $ORDER_CPU; do sample "$OUT" "${item%%:*}" "${item#*:}"; done
fi
python3 -I "$W/summary_c4m26.py" "$OUT" > "$OUT/summary.txt" 2>&1
tee -a "$OUT/run.log" < "$OUT/summary.txt"
tar -C "$W" -czf "$W/$(basename "$OUT").tgz" "$(basename "$OUT")" && sha256sum "$W/$(basename "$OUT").tgz" > "$W/$(basename "$OUT").tgz.sha256"
log "ALL SAMPLES DONE"
