#!/bin/bash
# run_c4m22.sh (2026-10-26, computer4) -- the one-GPU half of the comparison asked for on JustVugg/colibri#1766: its dev
# base 88f25a60 (MB) vs #1766 ca55947c (M1), same machine, same prompts, the same serve environment for both. Generic
# CUDA sm_120 builds (CUDA 12.8, no DeepGEMM), RTX 5070 12 GB, full V4 Flash, bench_decode.py (6 prompts x 300 tokens).
# #1766's DSV4_CUDA_RESIDENT_EXPERTS=auto (its default) is left as it is. W = a discarded warm-up on the base.
# Then tests_c4m22.sh (DeepGEMM dsv4-cuda-test of PR #2 5b3f4601, rebased onto #2401).
set -u
W=$HOME/colibri-bench/c4m22
OUT=$W/bench
MODEL=$HOME/colibri-models/deepseek-v4-flash
PORT=8090
DEV=$HOME/colibri-bench/dg/mbase; PR1=$HOME/colibri-bench/dg/m1766
BASE_ENV=(PATH=/usr/local/bin:/usr/bin:/bin HOME=$HOME USER=$USER LOGNAME=$USER LANG=en_US.UTF-8
          LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64)
COMMON_ENV=(COLI_CUDA=1 COLI_GPU=0 DSV4_CUDA_DEVICE=0 CUDA_DENSE=1 COLI_CUDA_PIPE=2 COLI_CUDA_ATTN_BATCH=1
            V4_MOE_REFILL_GROUP=12 V4_LOADER_LANES=3 V4_PREFIX_CKPT_DISK=0)
SERVE_ARGS="--model $MODEL --ram 40 --ctx 8192 --host 127.0.0.1 --port $PORT"
ORDER="W:1 MB:1 M1:1 M1:2 MB:2"
declare -A TREE=([W]=$DEV [MB]=$DEV [M1]=$PR1)
declare -A XENV=([W]="" [MB]="" [M1]="")

[ -d "$OUT" ] && [ -n "$(ls -A "$OUT")" ] && mv "$OUT" "$OUT.old-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT/meta"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/run.log"; }
for t in "$DEV" "$PR1"; do
    [ -x "$t/c/deepseek_v4" ] || { log "no binary in $t"; exit 4; }
    [ -z "$(git -C "$t" status --porcelain --untracked-files=no)" ] || { log "$t has local changes"; exit 4; }
done
[ "$(git -C "$DEV" rev-parse HEAD)" = "$(git -C "$DEV" rev-parse 88f25a60)" ] || { log "base tree moved"; exit 4; }
[ "$(git -C "$PR1" rev-parse HEAD)" = ca55947c62b1d342d3e0aa890e2959dec1b1de5e ] || { log "#1766 tree moved"; exit 4; }
[ "$(ls "$MODEL"/model-*-of-*.safetensors | wc -l)" = 48 ] || { log "model incomplete"; exit 4; }
{ uname -a; nvidia-smi; nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv; free -g; lscpu | grep -E 'Model name|^CPU\(s\)';
  for t in "$DEV" "$PR1"; do echo "tree=$t head=$(git -C "$t" rev-parse HEAD) binary=$(sha256sum "$t/c/deepseek_v4" | cut -c1-64)"; ldd "$t/c/deepseek_v4" | grep -E 'cuda|cublas'; done
  echo "COMMON_ENV=${COMMON_ENV[*]}"; echo "SERVE_ARGS=$SERVE_ARGS"; } > "$OUT/meta/host.txt" 2>&1
cp "$0" "$W/clients/bench_decode.py" "$W/summary_c4m22.py" "$OUT/meta/"
log "run_c4m22 start: $ORDER"
SEQ=0; SPID=; NPID=
cleanup() { [ -n "$NPID" ] && kill "$NPID" 2>/dev/null; [ -n "$SPID" ] && kill -TERM -- "-$SPID" 2>/dev/null; }
trap 'cleanup; log "signal: exiting"; exit 143' TERM INT HUP
for item in $ORDER; do
    arm=${item%%:*}; k=${item#*:}; tree=${TREE[$arm]}; d=$OUT/$arm/s$k; mkdir -p "$d"; SEQ=$((SEQ + 1)); t0=$(date +%s)
    for i in $(seq 1 30); do ss -ltnH "sport = :$PORT" | grep -q . || break; sleep 2; done
    { echo "arm=$arm"; echo "sample=$k"; echo "seq=$SEQ"; echo "tree=$tree"; echo "tree_head=$(git -C "$tree" rev-parse HEAD)"
      echo "tree_dirty_lines=$(git -C "$tree" status --porcelain --untracked-files=no | wc -l)"
      echo "binary_sha256=$(sha256sum "$tree/c/deepseek_v4" | cut -c1-64)"
      echo "server_env=${BASE_ENV[*]} ${COMMON_ENV[*]} ${XENV[$arm]}"; echo "start=$(date -Is)"; } > "$d/sample.txt"
    log "=== sample $SEQ: $arm s$k tree=$tree"
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu,clocks.sm \
        --format=csv -lms 5000 > "$d/nvsmi.csv" 2>&1 & NPID=$!
    setsid env -i "${BASE_ENV[@]}" "${COMMON_ENV[@]}" ${XENV[$arm]} bash -c "cd '$tree' && exec python3 c/coli serve $SERVE_ARGS" > "$d/serve.log" 2>&1 &
    SPID=$!
    ready=0
    for i in $(seq 1 450); do grep -q "listening on" "$d/serve.log" 2>/dev/null && { ready=1; break; }; kill -0 "$SPID" 2>/dev/null || break; sleep 2; done
    echo "ready=$ready" >> "$d/sample.txt"
    rc=99
    if [ $ready = 1 ]; then
        timeout 3600 python3 "$W/clients/bench_decode.py" > "$d/bench.jsonl" 2> "$d/bench.err"; rc=$?
        python3 -I "$W/summary_c4m22.py" --brief "$d/bench.jsonl" > "$d/bench.out" 2>&1
    fi
    echo "client_rc=$rc" >> "$d/sample.txt"
    kill -TERM -- "-$SPID" 2>/dev/null
    for i in $(seq 1 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$SPID" 2>/dev/null; SPID=
    kill "$NPID" 2>/dev/null; NPID=
    echo "wall_s=$(( $(date +%s) - t0 ))" >> "$d/sample.txt"
    log "--- $arm s$k: ready=$ready client_rc=$rc wall=$(( $(date +%s) - t0 ))s $(tr '\n' ' ' < "$d/bench.out" 2>/dev/null)"
    log "check $arm s$k: $(grep -cE 'out of memory|Segmentation|Aborted|Traceback' "$d/serve.log") error lines; $(grep -m3 -E 'resident|DSV4_CUDA_DEVICES|layer range|moe-batch=' "$d/serve.log" | tr '\n' ' ' | cut -c1-300)"
done
python3 -I "$W/summary_c4m22.py" "$OUT" > "$OUT/summary.txt" 2>&1
tee -a "$OUT/run.log" < "$OUT/summary.txt"
tar -C "$W" -czf "$W/bench.tgz" bench && sha256sum "$W/bench.tgz" > "$W/bench.tgz.sha256"
log "ALL SAMPLES DONE"; bash $W/tests_c4m22.sh
