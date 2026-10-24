#!/bin/bash
# run_final15.sh (2026-10-24) -- final15 (OVERNIGHT_20261024.md), REAP-150B, pure Vulkan: fix B candidate f15/fixb
# (final15_tools.py): J2 fixb pool 12 GiB, C1 dev the same (control), J1 fixb with the default pool. final14's runner otherwise.
EXP=bench
[ "${PILOT:-0}" = 1 ] && EXP=pilot
W22=/home/maplemercury/colibri-bench/cw22
DEVV=$W22/f9v-dev                       # CUDA=1 VK=1 build of dev 61646cbd (cw22/build_vk9.sh, final9)
DEV=$W22/f10-dev                         # CUDA build of dev 61646cbd (final10)
PR2F=$W22/f11-pr2f                       # CUDA build of PR #2 + the F18 fix (build_f11.sh)
D2DEV=$W22/f11d-dev; D2PR1=$W22/f11d-pr1 # CUDA=1 VK=1 diagnostic builds v2 (build_f11.sh)
NVF="-O3 -std=c++17 -ftz=false -arch=sm_86 -Xcompiler=-Wall,-Wextra -DCOLI_DSV4_NO_TC"
MAX_GPU_S=$((120 * 60))         # cap on the llama-down window (2 h; the plan is ~45 min)
NVSMI_MS=5000
source /home/maplemercury/colibri-bench/final15/final_common.sh

REPO=$W22/repo
DEV_SHA=61646cbd07da458569deb882586a390ea945e90b
PR2F_REF=f11/pr2-f18
CLIENT_SHA_PR1=eca4360bdd5a0601c0a7138f49464874a648486519235bbc22d40997c3e2de53     # final/clients/bench_pr1.py
TZ_NAME=America/Vancouver
STOP_AT="${STOP_AT:?set STOP_AT=\x27YYYY-MM-DD HH:MM:SS\x27}"
STOP_EPOCH_WANT=$(date -d "$STOP_AT" +%s)
LATEST_STOP_S=300
MAX_EARLY_S=3600
MODE_RE='moe-batch=|mirror-reserve|DSV4 CUDA\] device|out of memory|[Vv]ulkan|\[VK|vk_'
export TZ=$TZ_NAME

FULL=/home/maplemercury/colibri-models/deepseek-v4-flash
REAP=/home/maplemercury/colibri-models/deepseek-v4-flash-reap
MODEL=$REAP
declare -A TREE CLIENT TMO XENV EST FAILED MDL CTX
arm() { # NAME TREE MODEL CTX CLIENT TIMEOUT EST [KEY=VAL...]
    local n=$1; TREE[$n]=$2; MDL[$n]=$3; CTX[$n]=$4; CLIENT[$n]=$5; TMO[$n]=$6; EST[$n]=$7; shift 7; XENV[$n]="$*"
}
VKO="COLI_VULKAN=1 DSV4_CUDA=0 COLI_CUDA=0"
FIXB=$W22/f15-fixb                         # CUDA=1 VK=1 build of f15/fixb (build_f15.sh)
arm J2 "$FIXB" "$REAP" 8192 pr1:both 3600 300  $VKO COLI_VK_TIER_HOST_GB=12
arm C1 "$DEVV" "$REAP" 8192 pr1:both 3600 300  $VKO COLI_VK_TIER_HOST_GB=12
arm J1 "$FIXB" "$REAP" 8192 pr1:both 3600 300  $VKO
ORDER1="J2:1 C1:1"
ORDER="J1:1 J2:2 C1:2 J1:2 J2:3 C1:3"
[ -n "${ORDER_OVERRIDE:-}" ] && ORDER=$ORDER_OVERRIDE

lock_or_exit
trap 'log "signal before the llama stop: exiting, llama untouched"; [ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null; exit 143' TERM INT HUP
log "run_final15.sh start (pid $$, unit $(sed 's#.*/##' /proc/self/cgroup), PILOT=${PILOT:-0}, model $MODEL)"

foreign_gpu_pids() {   # every process of llama-server-active's cgroup is llama's (Tabby's model runs in a child)
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
[ "$(git -C "$DEVV" rev-parse HEAD)" = "$DEV_SHA" ] && grep -q '^EXIT=0' "$W22/build-vk9-dev.log" || die_pre "$DEVV is not a clean VK build of dev"
grep -q '^EXIT=0' "$FINAL/builds/f15-fixb.build.log" || die_pre "$FIXB is not a clean build"
preflight_tree "$FIXB" "$(git -C "$REPO" rev-parse f15/fixb)"
echo "dev_sha=$DEV_SHA fixb_sha=$(git -C "$FIXB" rev-parse HEAD)" > "$OUT/meta/tips.txt"
DEVV_BIN=$(sha256sum "$DEVV/c/deepseek_v4" | cut -c1-64)
log "tips: dev $DEV_SHA, VK binary $DEVV_BIN"

# ---- 3. preflight (nothing touched)
preflight_common bench_pr1.py bench_f6.py bench_decode.py
[ "$(sha256sum < "$CLIENTS/bench_pr1.py" | cut -c1-64)" = "$CLIENT_SHA_PR1" ] || die_pre "bench_pr1.py is not the 2026-10-15 client"
[ -f "$REAP/config.json" ] || die_pre "REAP model missing at $REAP"
preflight_switch "$DEVV" COLI_VK_TIER_HOST COLI_VK_TIER_HOST_GB COLI_VK_UP_IMPORT
[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2 ] || die_pre "second GPU not visible"
[ -z "$(foreign_gpu_pids)" ] || die_pre "other GPU compute processes: $(foreign_gpu_pids | tr '\n' ' ')"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/meta/gpu_apps_preflight.txt" 2>&1
[ "$(usage_state | cut -d' ' -f1)" = "size=0" ] || [ "$(usage_state)" = missing ] \
    || log "WARNING: $MODEL/.coli_usage is not empty ($(usage_state)): the RAM pins start from a history (recorded, not an abort)"
[ "$(df -Pk "$FINAL" | awk 'NR==2{print $4}')" -gt 2000000 ] || die_pre "less than 2 GB free in $FINAL"
total_est=0
for item in $ORDER1 $ORDER; do total_est=$((total_est + ${EST[${item%%:*}]})); done
log "plan: $ORDER (estimated $total_est s of samples; cap $MAX_GPU_S s, watchdog at $((MAX_GPU_S - HARD_MARGIN_S)) s)"
[ $((total_est + 240)) -le "$MAX_GPU_S" ] || die_pre "plan estimate $total_est s exceeds the cap"
log "preflight ok"

# ---- 4. wait for STOP_AT (re-check 20 s before)
recheck() {
    local why=
    systemctl --user is-active -q "$LLAMA_UNIT" && llama_healthy || why="$why llama not healthy;"
    ss -ltnH "sport = :$PORT" | grep -q . && why="$why port $PORT in use;"
    [ -z "$(other_colibri_units)" ] || why="$why other colibri unit: $(other_colibri_units | tr '\n' ' ');"
    [ "$(sha256sum "$DEVV/c/deepseek_v4" | cut -c1-64)" = "$DEVV_BIN" ] || why="$why $DEVV binary changed;"
    [ -z "$(git -C "$DEVV" status --porcelain --untracked-files=no)" ] || why="$why $DEVV has local changes;"
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
capture_meta "$DEVV" "$FIXB"
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
    python3 "$FINAL/final15_tools.py" check "$arm" "$OUT/$arm/s$k" 2>&1 | tee -a "$OUT/run.log"
    # final9: a premise failure (an OOM kill above all) is recorded and the plan goes on
    [ "${PIPESTATUS[0]}" != 0 ] && log "$arm s$k broke a premise (recorded; the run goes on)"
done
}
run_order "$ORDER1"

run_order "$ORDER"

finish
