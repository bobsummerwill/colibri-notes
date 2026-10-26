#!/bin/bash
# run_final24.sh (2026-10-26 early) -- final24: final23's follow-up. final23's M1F/M2F (#1766 on the full model) lost the
#   expert bank (handled OOM; moe-batch off) and took 388 s on the second long prompt (P0F: 124 s). MBF = #1766's dev
#   base 88f25a60 on the full model (does the base lose the bank too?); M1R/M2R = #1766 on one / two GPUs with
#   DSV4_CUDA_VRAM_RESERVE_MB=4700 (#2401's bank-aware reserve is 4681). One each. final23's runner, new arms.
# MAX_GPU_S from the environment.
EXP=${EXP_NAME:-bench}
[ "${PILOT:-0}" = 1 ] && EXP=pilot
W22=/home/maplemercury/colibri-bench/cw22
BASE=$W22/f22-base; PR2=$W22/f22-pr2; MBASE=$W22/f22-mbase; M1766=$W22/f22-m1766
NVF="-O3 -std=c++17 -ftz=false -arch=sm_86 -Xcompiler=-Wall,-Wextra -DCOLI_DSV4_NO_TC"
MAX_GPU_S=${MAX_GPU_S:?set MAX_GPU_S (s)}   # cap on the llama-down window (<= 4 h; ends before 05:44)
NVSMI_MS=5000
source /home/maplemercury/colibri-bench/final24/final_common.sh

REPO=$W22/repo
DEV_SHA=77530b49ddd2f4e6ad1b3bb52500491e0c8ed9bb
PR2_SHA=5b3f4601
BASE_SHA=dd03a9de
MBASE_SHA=88f25a60
M1766_SHA=ca55947c
TZ_NAME=America/Vancouver
STOP_AT="${STOP_AT:?set STOP_AT=\x27YYYY-MM-DD HH:MM:SS\x27}"
STOP_EPOCH_WANT=$(date -d "$STOP_AT" +%s)
LATEST_STOP_S=300
MAX_EARLY_S=3600
MODE_RE='moe-batch=|mirror-reserve|peer-mirrors|DSV4 CUDA\] device|out of memory|CUDA is lost|PEER_FAULT|DSV4_CUDA_DEVICES|[Rr]esiden|layer[s]? [0-9]+-[0-9]+|device [0-9]: layers'
export TZ=$TZ_NAME

FULL=/home/maplemercury/colibri-models/deepseek-v4-flash
REAP=/home/maplemercury/colibri-models/deepseek-v4-flash-reap
MODEL=$REAP
declare -A TREE CLIENT TMO XENV EST FAILED MDL CTX
arm() { # NAME TREE MODEL CTX CLIENT TIMEOUT EST [KEY=VAL...]
    local n=$1; TREE[$n]=$2; MDL[$n]=$3; CTX[$n]=$4; CLIENT[$n]=$5; TMO[$n]=$6; EST[$n]=$7; shift 7; XENV[$n]="$*"
}
BANKE="COLI_CUDA_MOE_BATCH=1"
PEER="V4_SERVE_STATS_LOG=1 V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1"
arm MBF "$MBASE" "$FULL" 8192 pr1:both 3600 560  $BANKE
arm MBR "$MBASE" "$FULL" 8192 pr1:both 3600 330  $BANKE DSV4_CUDA_VRAM_RESERVE_MB=4700
arm M1R "$M1766" "$FULL" 8192 pr1:both 3600 560  $BANKE DSV4_CUDA_VRAM_RESERVE_MB=4700
arm M2R "$M1766" "$FULL" 8192 pr1:both 3600 560  $BANKE DSV4_CUDA_VRAM_RESERVE_MB=4700 DSV4_CUDA_DEVICES=0,1
ORDER1=${ORDER1_OVERRIDE:-"MBF:1 M1R:1 M2R:1"}   # part 2 (bench2): MBR:1 M1R:2 M2R:2 MBR:2
ORDER=""
ORDER3=""
[ -n "${ORDER_OVERRIDE:-}" ] && ORDER=$ORDER_OVERRIDE

lock_or_exit
trap 'log "signal before the llama stop: exiting, llama untouched"; [ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null; exit 143' TERM INT HUP
log "run_final24.sh start (pid $$, unit $(sed 's#.*/##' /proc/self/cgroup), PILOT=${PILOT:-0})"

foreign_gpu_pids() {   # every process of llama-server-active's cgroup is llama's
    local cg allowed
    cg=$(systemctl --user show -p ControlGroup --value llama-server-active 2>/dev/null)
    allowed=$(cat "/sys/fs/cgroup$cg/cgroup.procs" 2>/dev/null | tr '\n' ' ')
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u | grep -v '^$' \
        | while read -r p; do case " $allowed " in *" $p "*) ;; *) echo "$p" ;; esac; done
}

# ---- 0. the clock
[ "$(timedatectl show -p Timezone --value 2>/dev/null)" = "$TZ_NAME" ] || die_pre "system time zone is not $TZ_NAME"
STOP_EPOCH=$(date -d "$STOP_AT" +%s)
[ "$STOP_EPOCH" = "$STOP_EPOCH_WANT" ] || die_pre "$STOP_AT resolves to $STOP_EPOCH, not $STOP_EPOCH_WANT"
lead=$(( STOP_EPOCH - $(now) ))
log "llama stop planned at $(date -d "@$STOP_EPOCH" '+%F %T %Z') ($lead s from now)"
[ "$lead" -le "$MAX_EARLY_S" ] || die_pre "started $lead s before $STOP_AT (more than $MAX_EARLY_S s early)"
[ "$lead" -ge "-$LATEST_STOP_S" ] || die_pre "started $(( -lead )) s after $STOP_AT: the window is missed"

# ---- 1. the tips and builds
declare -A BIN
for t in "$BASE:$BASE_SHA" "$PR2:$PR2_SHA" "$MBASE:$MBASE_SHA" "$M1766:$M1766_SHA"; do
    dir=${t%%:*}; full=$(git -C "$REPO" rev-parse "${t#*:}^{commit}") || die_pre "${t#*:} not in $REPO"
    preflight_tree "$dir" "$full"
    BIN[$dir]=$(sha256sum "$dir/c/deepseek_v4" | cut -c1-64)
    echo "$(basename "$dir")=$full binary=${BIN[$dir]}" >> "$OUT/meta/tips.txt"
done
log "tips: $(tr '\n' ' ' < "$OUT/meta/tips.txt")"

# ---- 3. preflight (nothing touched)
preflight_common bench_pr1.py bench_decode.py
[ -f "$REAP/config.json" ] || die_pre "REAP model missing at $REAP"
[ -f "$FULL/config.json" ] || die_pre "full model missing at $FULL"
preflight_switch "$PR2" V4_GPU_RESIDENT_FIRST DSV4_CUDA_PEER_DEVICE DSV4_CUDA_PEER_FAULT
preflight_switch "$M1766" DSV4_CUDA_DEVICES DSV4_CUDA_RESIDENT_EXPERTS
[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2 ] || die_pre "second GPU not visible"
[ -z "$(foreign_gpu_pids)" ] || die_pre "other GPU compute processes: $(foreign_gpu_pids | tr '\n' ' ')"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/meta/gpu_apps_preflight.txt" 2>&1
[ "$(df -Pk "$FINAL" | awk 'NR==2{print $4}')" -gt 2000000 ] || die_pre "less than 2 GB free in $FINAL"
total_est=0
for item in $ORDER1 $ORDER $ORDER3; do total_est=$((total_est + ${EST[${item%%:*}]})); done
log "plan: $ORDER1 | $ORDER | $ORDER3 | CUDA tests (estimated $total_est s of samples + no tests; cap $MAX_GPU_S s)"
[ $((total_est + 240)) -le "$MAX_GPU_S" ] || die_pre "plan estimate exceeds the cap"
log "preflight ok"

# ---- 4. wait for STOP_AT (re-check 20 s before)
recheck() {
    local why= dir
    systemctl --user is-active -q "$LLAMA_UNIT" && llama_healthy || why="$why llama not healthy;"
    ss -ltnH "sport = :$PORT" | grep -q . && why="$why port $PORT in use;"
    [ -z "$(other_colibri_units)" ] || why="$why other colibri unit: $(other_colibri_units | tr '\n' ' ');"
    for dir in "$BASE" "$PR2" "$MBASE" "$M1766"; do
        [ "$(sha256sum "$dir/c/deepseek_v4" | cut -c1-64)" = "${BIN[$dir]}" ] || why="$why $dir binary changed;"
        [ -z "$(git -C "$dir" status --porcelain --untracked-files=no)" ] || why="$why $dir has local changes;"
    done
    [ -z "$(foreign_gpu_pids)" ] || why="$why other GPU processes: $(foreign_gpu_pids | tr '\n' ' ');"
    [ -n "$why" ] && die_pre "re-check before the llama stop:$why"
    log "re-check ok"
}
log "waiting until $STOP_AT ($(( STOP_EPOCH - $(now) )) s)"
wait_until $((STOP_EPOCH - 20))
recheck
wait_until "$STOP_EPOCH"
[ "$(now)" -le $((STOP_EPOCH + LATEST_STOP_S)) ] || die_pre "it is $(date '+%T'), later than $LATEST_STOP_S s after $STOP_AT: the window is missed"

# ---- 5. the GPU window
arm_trap
capture_meta "$BASE" "$PR2" "$MBASE" "$M1766"
stop_llama
start_watchdog

run_order() {
for item in $1; do
    arm=${item%%:*}; k=${item#*:}
    if over_budget "$(est_for "$arm" "${EST[$arm]}")"; then log "budget: skipping $arm s$k (GPU window $(gpu_elapsed) s)"; continue; fi
    MODEL=${MDL[$arm]}
    SERVE_ARGS="--model ${MDL[$arm]} --ram 40 --ctx ${CTX[$arm]} --host 127.0.0.1 --port $PORT"
    # shellcheck disable=SC2086
    sample "$arm" "$k" "${TREE[$arm]}" "${CLIENT[$arm]}" "${TMO[$arm]}" ${XENV[$arm]}
    python3 "$FINAL/final24_tools.py" check "$arm" "$OUT/$arm/s$k" 2>&1 | tee -a "$OUT/run.log"
    [ "${PIPESTATUS[0]}" != 0 ] && log "$arm s$k broke a premise (recorded; the run goes on)"
done
}
run_order "$ORDER1"
run_order "$ORDER"
run_order "$ORDER3"


finish
