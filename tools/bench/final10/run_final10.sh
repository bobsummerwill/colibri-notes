#!/bin/bash
# run_final10.sh (2026-10-24) -- final10 (OVERNIGHT_20261024.md): 1. upstream fix A (fix/dsv4-mirror-guard-overshoot
# bd7103fd on dev 61646cbd) against dev on the full DeepSeek V4 Flash in the default configuration, W A F F A A F
# (W discarded), counting the handled "kv comp allocation" failures; 2. the Vulkan host pool OOM: dev, REAP, pure
# Vulkan, default pool, memory by class every 5 s (M), and the same with COLI_VK_UP_IMPORT=0 (M2). final9's runner
# otherwise; CUDA builds by build_f10.sh before this run.

EXP=bench
[ "${PILOT:-0}" = 1 ] && EXP=pilot
W22=/home/maplemercury/colibri-bench/cw22
DEVV=$W22/f9v-dev                       # CUDA=1 VK=1 build of dev 61646cbd (cw22/build_vk9.sh, final9)
DEV=$W22/f10-dev; FIXA=$W22/f10-fixa     # CUDA builds (build_f10.sh): dev 61646cbd and fix A
MAX_GPU_S=$((180 * 60))         # cap on the llama-down window (3 h; the plan is ~1.4 h)
NVSMI_MS=5000
source /home/maplemercury/colibri-bench/final10/final_common.sh

REPO=$W22/repo
DEV_SHA=61646cbd07da458569deb882586a390ea945e90b
FIXA_SHA=bd7103fd
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
MODEL=$FULL
declare -A TREE CLIENT TMO XENV EST FAILED MDL CTX
arm() { # NAME TREE MODEL CTX CLIENT TIMEOUT EST [KEY=VAL...]
    local n=$1; TREE[$n]=$2; MDL[$n]=$3; CTX[$n]=$4; CLIENT[$n]=$5; TMO[$n]=$6; EST[$n]=$7; shift 7; XENV[$n]="$*"
}
VKO="COLI_VULKAN=1 DSV4_CUDA=0 COLI_CUDA=0"
arm W  "$DEV"  "$FULL" 8192 pr1:both 3600 700
arm A  "$DEV"  "$FULL" 8192 pr1:both 3600 700
arm F  "$FIXA" "$FULL" 8192 pr1:both 3600 700
arm M  "$DEVV" "$REAP" 8192 pr1:both 3600 400  $VKO
arm M2 "$DEVV" "$REAP" 8192 pr1:both 3600 400  $VKO COLI_VK_UP_IMPORT=0
ORDER="W:1 A:1 F:1 F:2 A:2 A:3 F:3 M:1 M2:1"
[ -n "${ORDER_OVERRIDE:-}" ] && ORDER=$ORDER_OVERRIDE

lock_or_exit
trap 'log "signal before the llama stop: exiting, llama untouched"; [ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null; exit 143' TERM INT HUP
log "run_final10.sh start (pid $$, unit $(sed 's#.*/##' /proc/self/cgroup), PILOT=${PILOT:-0}, model $MODEL)"

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
FIXA_SHA=$(git -C "$REPO" rev-parse --verify -q "$FIXA_SHA^{commit}") || die_pre "fix A not in $REPO"
[ "$(git -C "$REPO" rev-parse "$FIXA_SHA^")" = "$DEV_SHA" ] || die_pre "fix A is not one commit on dev $DEV_SHA"
{ echo "resolved_at=$(date -Is)"; echo "dev_sha=$DEV_SHA"; echo "fixa_sha=$FIXA_SHA"; git -C "$REPO" log -1 --format=%s "$FIXA_SHA";
  git -C "$REPO" diff --stat "$DEV_SHA" "$FIXA_SHA" | tail -1; } > "$OUT/meta/tips.txt" 2>&1
preflight_tree "$DEV" "$DEV_SHA"
preflight_tree "$FIXA" "$FIXA_SHA"
[ "$(git -C "$DEVV" rev-parse HEAD)" = "$DEV_SHA" ] && grep -q '^EXIT=0' "$W22/build-vk9-dev.log" || die_pre "$DEVV is not a clean VK build of dev"
DEV_BIN_SHA=$(sha256sum "$DEV/c/deepseek_v4" | cut -c1-64)
PR1_BIN=$(sha256sum "$FIXA/c/deepseek_v4" | cut -c1-64)
DEVV_BIN=$(sha256sum "$DEVV/c/deepseek_v4" | cut -c1-64)
log "tips: dev $DEV_SHA, fix A $FIXA_SHA; binaries $DEV_BIN_SHA $PR1_BIN, VK $DEVV_BIN"

# ---- 3. preflight (nothing touched)
preflight_common bench_pr1.py bench_f6.py bench_decode.py
[ "$(sha256sum < "$CLIENTS/bench_pr1.py" | cut -c1-64)" = "$CLIENT_SHA_PR1" ] || die_pre "bench_pr1.py is not the 2026-10-15 client"
[ -f "$REAP/config.json" ] || die_pre "REAP model missing at $REAP"
[ "$(python3 -c "import json;print(json.load(open('$FULL/config.json')).get('n_routed_experts'))")" = 256 ] || die_pre "$FULL is not the 256-expert checkpoint"
preflight_switch "$DEVV" COLI_VK_TIER_HOST_GB COLI_VK_UP_IMPORT
[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2 ] || die_pre "second GPU not visible"
[ -z "$(foreign_gpu_pids)" ] || die_pre "other GPU compute processes: $(foreign_gpu_pids | tr '\n' ' ')"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/meta/gpu_apps_preflight.txt" 2>&1
[ "$(usage_state | cut -d' ' -f1)" = "size=0" ] || [ "$(usage_state)" = missing ] \
    || log "WARNING: $MODEL/.coli_usage is not empty ($(usage_state)): the RAM pins start from a history (recorded, not an abort)"
[ "$(df -Pk "$FINAL" | awk 'NR==2{print $4}')" -gt 2000000 ] || die_pre "less than 2 GB free in $FINAL"
total_est=0
for item in $ORDER; do total_est=$((total_est + ${EST[${item%%:*}]})); done
log "plan: $ORDER (estimated $total_est s of samples; cap $MAX_GPU_S s, watchdog at $((MAX_GPU_S - HARD_MARGIN_S)) s)"
[ $((total_est + 240)) -le "$MAX_GPU_S" ] || die_pre "plan estimate $total_est s exceeds the cap"
log "preflight ok"

# ---- 4. wait for STOP_AT (re-check 20 s before)
recheck() {
    local why=
    systemctl --user is-active -q "$LLAMA_UNIT" && llama_healthy || why="$why llama not healthy;"
    ss -ltnH "sport = :$PORT" | grep -q . && why="$why port $PORT in use;"
    [ -z "$(other_colibri_units)" ] || why="$why other colibri unit: $(other_colibri_units | tr '\n' ' ');"
    [ "$(sha256sum "$FIXA/c/deepseek_v4" | cut -c1-64)" = "$PR1_BIN" ] || why="$why $FIXA binary changed;"
    [ "$(sha256sum "$DEV/c/deepseek_v4" | cut -c1-64)" = "$DEV_BIN_SHA" ] || why="$why $DEV binary changed;"
    [ "$(sha256sum "$DEVV/c/deepseek_v4" | cut -c1-64)" = "$DEVV_BIN" ] || why="$why $DEVV binary changed;"
    local t
    for t in "$DEV" "$FIXA" "$DEVV"; do
        [ -z "$(git -C "$t" status --porcelain --untracked-files=no)" ] || why="$why $t has local changes;"
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
capture_meta "$DEV" "$FIXA" "$DEVV"
stop_llama
start_watchdog

for item in $ORDER; do
    arm=${item%%:*}; k=${item#*:}
    if over_budget "$(est_for "$arm" "${EST[$arm]}")"; then log "budget: skipping $arm s$k (GPU window $(gpu_elapsed) s)"; continue; fi
    MODEL=${MDL[$arm]}
    SERVE_ARGS="--model ${MDL[$arm]} --ram 40 --ctx ${CTX[$arm]} --host 127.0.0.1 --port $PORT"
    # shellcheck disable=SC2086
    sample "$arm" "$k" "${TREE[$arm]}" "${CLIENT[$arm]}" "${TMO[$arm]}" ${XENV[$arm]}
    python3 "$FINAL/final10_tools.py" check "$arm" "$OUT/$arm/s$k" 2>&1 | tee -a "$OUT/run.log"
    # final9: a premise failure (an OOM kill above all) is recorded and the plan goes on
    [ "${PIPESTATUS[0]}" != 0 ] && log "$arm s$k broke a premise (recorded; the run goes on)"
done

# ---- 6. F18 probe with the host-wait variant (cw22/f18hw 1b93f1ef, -DCOLI_F18_PROBE, debug only): a real __trap()
# on the peer GPU in the decode MoE, before/after its down-projection, with device 0's stream waiting on the peer's
# join (modes 1-2) and with the host waiting for it (DSV4_F18_HOSTWAIT=1, modes 3-4). Does device 0 survive?
# Last, because a faulted context could disturb what follows.
F18HW=$W22/f18hw
if over_budget 600; then log "budget: skipping the F18 probe"; elif [ ! -x "$F18HW/c/f18_trap_cuda_test" ]; then log "F18 probe binary missing"; else
    ( cd "$F18HW/c" && CUDA_VISIBLE_DEVICES=0,1 timeout 400 ./f18_trap_cuda_test ) > "$OUT/meta/f18-probe.log" 2>&1
    log "F18 probe: rc=$? $(grep -E 'RESULT|AFTER|HANG|f18 probe:|survive' "$OUT/meta/f18-probe.log" | tr '\n' ' ' | cut -c1-1500)"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$OUT/meta/f18-probe.log" 2>&1
fi

finish
