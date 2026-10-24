#!/bin/bash
# run_c4fa.sh (2026-10-24, overnight, computer4) -- upstream fix A (fix/dsv4-mirror-guard-overshoot bd7103fd) against
# dev 61646cbd on the DeepGEMM build, RTX 5070 12 GB: the full DeepSeek V4 Flash, default configuration, both
# bench_pr1.py prompts. Here dev's handled "kv comp allocation: out of memory" failures varied from 0 to 1,066 a
# sample and changed prompt 2's text in 2 of 7 runs (c4ab2). A = dev, B = fix A (named so final6_tools.py checks
# them); W discarded; 5 pairs. From run_c4ab3.sh (same per-sample files).
set -u
W=$HOME/colibri-bench/c4fa
OUT=$W/bench
MODEL=$HOME/colibri-models/deepseek-v4-flash
PORT=8090
DEV=$HOME/colibri-bench/dg/dev9; PR1=$HOME/colibri-bench/dg/fixa
BASE_ENV=(PATH=/usr/local/bin:/usr/bin:/bin HOME=$HOME USER=$USER LOGNAME=$USER LANG=en_US.UTF-8
          LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64)
COMMON_ENV=(COLI_CUDA=1 COLI_GPU=0 DSV4_CUDA_DEVICE=0 CUDA_DENSE=1 COLI_CUDA_PIPE=2 COLI_CUDA_ATTN_BATCH=1
            V4_MOE_REFILL_GROUP=12 V4_LOADER_LANES=3 V4_PREFIX_CKPT_DISK=0)
SERVE_ARGS="--model $MODEL --ram 40 --ctx 8192 --host 127.0.0.1 --port $PORT"
ORDER="W:1 A:1 B:1 B:2 A:2 A:3 B:3 B:4 A:4 A:5 B:5"
declare -A TREE=([W]=$DEV [A]=$DEV [B]=$PR1)
declare -A XENV=([W]="" [A]="" [B]="")

[ -d "$OUT" ] && [ -n "$(ls -A "$OUT")" ] && mv "$OUT" "$OUT.old-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT/meta"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/run.log"; }
for t in "$DEV" "$PR1"; do
    [ -x "$t/c/deepseek_v4" ] || { log "no binary in $t"; exit 4; }
    [ -z "$(git -C "$t" status --porcelain --untracked-files=no)" ] || { log "$t has local changes"; exit 4; }
done
[ "$(git -C "$DEV" rev-parse HEAD)" = 61646cbd07da458569deb882586a390ea945e90b ] || { log "dev tree moved"; exit 4; }
[ "$(git -C "$PR1" rev-parse HEAD)" = bd7103fd5e76db05500f03bcf1b72b5da1f7bbeb ] || { log "pr1 tree moved"; exit 4; }
[ "$(ls "$MODEL"/model-*-of-*.safetensors | wc -l)" = 48 ] || { log "model incomplete"; exit 4; }
{ uname -a; nvidia-smi; nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv; free -g; lscpu | grep -E 'Model name|^CPU\(s\)';
  for t in "$DEV" "$PR1"; do echo "tree=$t head=$(git -C "$t" rev-parse HEAD) binary=$(sha256sum "$t/c/deepseek_v4" | cut -c1-64)"; ldd "$t/c/deepseek_v4" | grep -E 'cuda|cublas'; done
  echo "COMMON_ENV=${COMMON_ENV[*]}"; echo "SERVE_ARGS=$SERVE_ARGS"; } > "$OUT/meta/host.txt" 2>&1
cp "$0" "$W/clients/bench_pr1.py" "$OUT/meta/"
log "run_c4ab start: $ORDER"
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
        timeout 3600 python3 "$W/clients/bench_pr1.py" "$d/bench.jsonl" > "$d/bench.out" 2> "$d/bench.err"; rc=$?
    fi
    echo "client_rc=$rc" >> "$d/sample.txt"
    kill -TERM -- "-$SPID" 2>/dev/null
    for i in $(seq 1 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$SPID" 2>/dev/null; SPID=
    kill "$NPID" 2>/dev/null; NPID=
    echo "wall_s=$(( $(date +%s) - t0 ))" >> "$d/sample.txt"
    log "--- $arm s$k: ready=$ready client_rc=$rc wall=$(( $(date +%s) - t0 ))s $(tr '\n' ' ' < "$d/bench.out" 2>/dev/null)"
    python3 -I "$W/final6_tools.py" check "$arm" "$d" 2>&1 | tee -a "$OUT/run.log"
done
python3 -I "$W/final6_tools.py" summarize final6 "$OUT" > "$OUT/summary.txt" 2>&1
grep -v '^JSON ' "$OUT/summary.txt" | tee -a "$OUT/run.log"
tar -C "$W" -czf "$W/bench.tgz" bench && sha256sum "$W/bench.tgz" > "$W/bench.tgz.sha256"
log "ALL SAMPLES DONE"
