#!/bin/bash
# run_c4ab3.sh (2026-10-23, final8 on computer4) -- why the default configuration's texts change under VRAM pressure:
# c4ab2 showed dev (A) and PR #1 (W) each change prompt 2's text when '[DSV4 CUDA] kv comp allocation: out of memory'
# sends layers' attention to the CPU. Arms: CA/CB = dev/PR #1 with COLI_CUDA_ATTN_BATCH=0 (all attention on the CPU);
# HA/HB = dev/PR #1 with DSV4_CUDA_VRAM_RESERVE_MB=2000 (mirrors leave 2 GB, so the kv comp buffer can grow).
# Otherwise run_c4ab.sh (final6's question on computer4's DeepGEMM build):
# default configuration (no COLI_CUDA_MOE_BATCH), upstream dev 7df8c27c vs PR #1 (b286167e), both
# bench_pr1.py prompts, W A B B A A B (W discarded). RTX 5070 12 GB (Voxtype keeps ~1.5 GB, so the bank's
# ~10 GB row does not fit and is not run). On DeepGEMM builds DSV4_MM_WARP and DSV4_MOE_EXPERT_MAJOR are
# compiled out: this checks that the rest of PR #1 leaves the DeepGEMM path's texts and speed alone.
# Same per-sample files as final6 (sample.txt, bench.jsonl, serve.log, nvsmi.csv), so final6_tools.py
# summarizes it. Each server is its own process group, stopped by TERM to the group.
set -u
W=$HOME/colibri-bench/c4ab3
OUT=$W/bench
MODEL=$HOME/colibri-models/deepseek-v4-flash
PORT=8090
DEV=$HOME/colibri-bench/dg/dev; PR1=$HOME/colibri-bench/dg/pr1b
BASE_ENV=(PATH=/usr/local/bin:/usr/bin:/bin HOME=$HOME USER=$USER LOGNAME=$USER LANG=en_US.UTF-8
          LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64)
COMMON_ENV=(COLI_CUDA=1 COLI_GPU=0 DSV4_CUDA_DEVICE=0 CUDA_DENSE=1 COLI_CUDA_PIPE=2 COLI_CUDA_ATTN_BATCH=1
            V4_MOE_REFILL_GROUP=12 V4_LOADER_LANES=3 V4_PREFIX_CKPT_DISK=0)
SERVE_ARGS="--model $MODEL --ram 40 --ctx 8192 --host 127.0.0.1 --port $PORT"
ORDER="HA:1 HB:1 HB:2 HA:2 CA:1 CB:1"
declare -A TREE=([HA]=$DEV [HB]=$PR1 [CA]=$DEV [CB]=$PR1)
declare -A XENV=([HA]="DSV4_CUDA_VRAM_RESERVE_MB=2000" [HB]="DSV4_CUDA_VRAM_RESERVE_MB=2000" [CA]="COLI_CUDA_ATTN_BATCH=0" [CB]="COLI_CUDA_ATTN_BATCH=0")

[ -d "$OUT" ] && [ -n "$(ls -A "$OUT")" ] && mv "$OUT" "$OUT.old-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT/meta"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/run.log"; }
for t in "$DEV" "$PR1"; do
    [ -x "$t/c/deepseek_v4" ] || { log "no binary in $t"; exit 4; }
    [ -z "$(git -C "$t" status --porcelain --untracked-files=no)" ] || { log "$t has local changes"; exit 4; }
done
[ "$(git -C "$DEV" rev-parse HEAD)" = 7df8c27c95acf7019dd60714ec1c42fa7e36b730 ] || { log "dev tree moved"; exit 4; }
[ "$(git -C "$PR1" rev-parse HEAD)" = b286167e614bb208e5a84a25812d7fb29786b230 ] || { log "pr1 tree moved"; exit 4; }
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
