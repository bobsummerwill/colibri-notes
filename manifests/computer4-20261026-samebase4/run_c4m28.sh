#!/bin/bash
# run_c4m28.sh (2026-10-26, computer4) -- PR #2's four-way split on dev c1c06ac6, one RTX 5070, after c4m27. S1 = fork
# split/pr2-1-guard 6a191e90, S2 = split/pr2-2-settings c45b6962, S3 = split/pr2-3-vram-first 695a7be4
# (V4_GPU_RESIDENT_FIRST=1); S4 = P's tree, no arm. Anchor B (dg/s27base). GPU only, 1 each, B S1 S2 S3; no pilot: the
# reserve c4m27's pilot decided (c4m27/bench/meta/gpu_reserve.txt; RESERVE=none|2000 overrides). One text group: every
# arm must give B's texts. Trees dg/s28s1..s3 (dg/build28.sh). c4m27's runner, new arms. Below: c4m27's header.
# run_c4m27.sh (2026-10-26, computer4) -- the same-base run RESTARTED on dev c1c06ac6 (#2401-#2404 merged upstream;
# c4m25 stopped): c4m25 and c4m26 in one run. B = c1c06ac6, M = fork samebase/m1766 ($M_SHA, default ecc92586),
# P = samebase/pr2 11c40038 (PR #2 rebased), C = samebase/cplus ($C_SHA, default b659ac8d; PR #2 + #1766).
#   GPU arms B M P C (P and C with V4_GPU_RESIDENT_FIRST=1), CPU-only CB CM CP CC, 2 each, interleaved; the pilot (M at
#   defaults) decides the reserve as before. CPU arms are named explicitly (CB|CM|CP|CC): GPU arm C is C+.
# Below: c4m25's header.
# run_c4m25.sh (2026-10-26, computer4) -- the same-base run on one RTX 5070 12 GB and CPU-only (colibri-notes
# review/PLAN_SAME_BASE_20261026.md, Stage 1). Every tree on dd03a9de (dev 77530b49 + #2401): B = dd03a9de, M = #1766
# merged onto it (fork samebase/m1766, $M_SHA), P = PR #2 5b3f4601. Full V4 Flash, --ram 40 --ctx 8192.
#   GPU (generic CUDA sm_120 builds, CUDA 12.8, no DeepGEMM; c4m22's serve environment, so no COLI_CUDA_MOE_BATCH: the
#   bank doesn't fit 12 GB anyway, c4f21): B, M, P (V4_GPU_RESIDENT_FIRST=1), bench_decode.py 6 x 300 tokens, 2 each.
#   First a pilot: M once at the defaults. More than PILOT_OOM_MAX "out of memory" lines in its serve.log (#1766 on its
#   old base logged 300-540k) -> every GPU arm gets DSV4_CUDA_VRAM_RESERVE_MB=2000 (RESERVE=auto, the default;
#   RESERVE=none|2000 forces it).
#   CPU-only (no-CUDA builds, -march=native): CB, CM, CP, bench_decode_cpu.py (the first 2 prompts x 128 tokens), 2 each.
# MODE=all (pilot, gpu, cpu; the default) | pilot | gpu | cpu.  M_SHA must be set.
set -u
W=$HOME/colibri-bench/c4m28
MODE=gpu; RESERVE=${RESERVE:-c4m27}; PILOT_OOM_MAX=${PILOT_OOM_MAX:-100}
M_SHA=${M_SHA:-ecc92586}; C_SHA=${C_SHA:-b659ac8d}
MODEL=$HOME/colibri-models/deepseek-v4-flash
PORT=8090
DG=$HOME/colibri-bench/dg
declare -A TREE=([B]=$DG/s27base [S1]=$DG/s28s1 [S2]=$DG/s28s2 [S3]=$DG/s28s3)
declare -A WANT=([B]=c1c06ac6 [S1]=6a191e90 [S2]=c45b6962 [S3]=695a7be4)
declare -A XENV=([B]="" [S1]="" [S2]="" [S3]="V4_GPU_RESIDENT_FIRST=1")
BASE_ENV=(PATH=/usr/local/bin:/usr/bin:/bin HOME=$HOME USER=$USER LOGNAME=$USER LANG=en_US.UTF-8
          LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64)
GPU_ENV=(COLI_CUDA=1 COLI_GPU=0 DSV4_CUDA_DEVICE=0 CUDA_DENSE=1 COLI_CUDA_PIPE=2 COLI_CUDA_ATTN_BATCH=1
         V4_MOE_REFILL_GROUP=12 V4_LOADER_LANES=3 V4_PREFIX_CKPT_DISK=0)
CPU_ENV=(COLI_CUDA=0 DSV4_CUDA=0 V4_MOE_REFILL_GROUP=12 V4_LOADER_LANES=3 V4_PREFIX_CKPT_DISK=0)
SERVE_ARGS="--model $MODEL --ram 40 --ctx 8192 --host 127.0.0.1 --port $PORT"
ORDER_GPU=${ORDER_GPU:-"B:1 S1:1 S2:1 S3:1"}
ORDER_CPU=""

exec 9> "$W/.lock"; flock -n 9 || { echo "another run_c4m28 holds $W/.lock"; exit 4; }
for o in c4m25 c4m26 c4m27; do exec 8> "$HOME/colibri-bench/$o/.lock"; flock -n 8 || { echo "$o is running (holds its lock)"; exit 4; }; exec 8>&-; done
OUT=$W/bench; [ "$MODE" = pilot ] && OUT=$W/pilot
[ "$MODE" != cpu ] && [ -d "$OUT" ] && [ -n "$(ls -A "$OUT")" ] && mv "$OUT" "$OUT.old-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT/meta"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/run.log"; }
die() { log "PREFLIGHT FAILED: $*"; exit 4; }

# ---- preflight
for a in "${!TREE[@]}"; do t=${TREE[$a]}
    [ -x "$t/c/deepseek_v4" ] || die "no binary in $t"
    [ -z "$(git -C "$t" status --porcelain --untracked-files=no)" ] || die "$t has local changes"
    [ "$(git -C "$t" rev-parse HEAD)" = "$(git -C "$t" rev-parse "${WANT[$a]}^{commit}")" ] || die "$t is not at ${WANT[$a]}"
    case $a in CB|CM|CP|CC) ldd "$t/c/deepseek_v4" | grep -q cudart && die "$t (CPU arm) links cudart";;
               *) ldd "$t/c/deepseek_v4" | grep -q cudart || die "$t (GPU arm) does not link cudart";; esac
done
[ "$(ls "$MODEL"/model-*-of-*.safetensors | wc -l)" = 48 ] || die "model incomplete"
pgrep -x deepseek_v4 >/dev/null && die "a deepseek_v4 process is running"
ss -ltnH "sport = :$PORT" | grep -q . && die "port $PORT in use"
other=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -v voxtype)
[ -z "$other" ] || die "other GPU processes: $other"
[ "$(df -Pk "$W" | awk 'NR==2{print $4}')" -gt 5000000 ] || die "less than 5 GB free"
{ uname -a; nvidia-smi; free -g; lscpu | grep -E 'Model name|^CPU\(s\)'
  for a in B S1 S2 S3; do t=${TREE[$a]}; echo "$a tree=$t head=$(git -C "$t" rev-parse HEAD) binary=$(sha256sum "$t/c/deepseek_v4" | cut -c1-64)"; done
  echo "GPU_ENV=${GPU_ENV[*]}"; echo "CPU_ENV=${CPU_ENV[*]}"; echo "SERVE_ARGS=$SERVE_ARGS"; echo "MODE=$MODE RESERVE=$RESERVE"; } > "$OUT/meta/host-$(date +%H%M%S).txt" 2>&1
cp "$0" "$W/clients/bench_decode.py" "$W/clients/bench_decode_cpu.py" "$W/summary_c4m28.py" "$OUT/meta/"
log "run_c4m28 start: MODE=$MODE RESERVE=$RESERVE S1=6a191e90 S2=c45b6962 S3=695a7be4"
[ "${PREFLIGHT_ONLY:-0}" = 1 ] && { log "preflight ok (PREFLIGHT_ONLY=1, nothing run)"; exit 0; }

SEQ=$(cat "$OUT/meta/seq" 2>/dev/null || echo 0); SPID=; NPID=
cleanup() { [ -n "$NPID" ] && kill "$NPID" 2>/dev/null; [ -n "$SPID" ] && kill -TERM -- "-$SPID" 2>/dev/null; }
trap 'cleanup; log "signal: exiting"; exit 143' TERM INT HUP
sample() { # OUTDIR ARM K EXTRA_ENV...
    local out=$1 arm=$2 k=$3; shift 3
    local tree=${TREE[$arm]} d=$out/$arm/s$k t0=$(date +%s) client=bench_decode.py tmo=3600 env_arr
    case $arm in CB|CM|CP|CC) env_arr=("${CPU_ENV[@]}"); client=bench_decode_cpu.py; tmo=5400;; *) env_arr=("${GPU_ENV[@]}");; esac
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
        python3 -I "$W/summary_c4m28.py" --brief "$d/bench.jsonl" > "$d/bench.out" 2>&1
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
if [ "$MODE" = all ] || [ "$MODE" = pilot ]; then
    if [ "$RESERVE" = auto ]; then
        PD=$W/pilot-$(date +%Y%m%d-%H%M%S)
        sample "$PD" M 1
        n=$(grep -c 'out of memory' "$PD/M/s1/serve.log" 2>/dev/null)
        if [ "${n:-0}" -gt "$PILOT_OOM_MAX" ]; then XRES="DSV4_CUDA_VRAM_RESERVE_MB=2000"; fi
        log "pilot: M at the defaults logged ${n:-?} 'out of memory' lines -> GPU arms get '${XRES:-no reserve setting}'"
    fi
fi
if [ "$RESERVE" = c4m27 ]; then   # c4m28: c4m27's pilot decided it
    f=$HOME/colibri-bench/c4m27/bench/meta/gpu_reserve.txt
    [ -f "$f" ] || die "no $f (c4m27's pilot never decided a reserve); set RESERVE=none|2000"
    XRES=$(cat "$f"); log "reserve from c4m27's pilot: '${XRES:-no reserve setting}'"
fi
[ "$RESERVE" = 2000 ] && XRES="DSV4_CUDA_VRAM_RESERVE_MB=2000"
echo "$XRES" > "$OUT/meta/gpu_reserve.txt"
if [ "$MODE" = all ] || [ "$MODE" = gpu ]; then
    for item in $ORDER_GPU; do sample "$OUT" "${item%%:*}" "${item#*:}" $XRES; done
fi
if [ "$MODE" = all ] || [ "$MODE" = cpu ]; then
    for item in $ORDER_CPU; do sample "$OUT" "${item%%:*}" "${item#*:}"; done
fi
python3 -I "$W/summary_c4m28.py" "$OUT" > "$OUT/summary.txt" 2>&1
tee -a "$OUT/run.log" < "$OUT/summary.txt"
tar -C "$W" -czf "$W/$(basename "$OUT").tgz" "$(basename "$OUT")" && sha256sum "$W/$(basename "$OUT").tgz" > "$W/$(basename "$OUT").tgz.sha256"
log "ALL SAMPLES DONE"
