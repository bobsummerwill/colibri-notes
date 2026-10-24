#!/bin/bash
# run_final9.sh (2026-10-24) -- final9 (TEST_PLAN_20261024.md): the merged tips, dev 61646cbd against PR #1 47c8ec51,
# on the REAP-150B checkpoint, Vulkan only. 1: pure Vulkan with the DEFAULT host pool (dev's 0f786bfa + 5c0469ae
# resize it; before them it OOM-killed the server after prompt 1): WQ warm-up, QD dev x3, QE PR #1 x3, interleaved.
# 2: why Vulkan + CUDA gives the bank and the union different prompt-2 texts: VA9 / VB9 / VD9 / VE9 / VX9, one each.
# final8's runner otherwise (same clients, units, llama handling, watchdog, OOM score); a premise failure is
# recorded and the run goes on (an OOM kill is a result here, not a reason to skip the arm).

EXP=bench
[ "${PILOT:-0}" = 1 ] && EXP=pilot
W22=/home/maplemercury/colibri-bench/cw22
DEVV=$W22/f9v-dev; PR1V=$W22/f9v-pr1   # CUDA=1 VK=1 builds (cw22/build_vk9.sh), logs build-vk9-{dev,pr1}.log
MAX_GPU_S=$((240 * 60))         # cap on the llama-down window (4 h; the plan is ~1.2 h)
NVSMI_MS=5000
source /home/maplemercury/colibri-bench/final9/final_common.sh

REPO=$W22/repo
DEV_SHA=61646cbd07da458569deb882586a390ea945e90b
PR1_SHA=47c8ec5165384009f5d573a6ad9e229483e579de
CLIENT_SHA_PR1=eca4360bdd5a0601c0a7138f49464874a648486519235bbc22d40997c3e2de53     # final/clients/bench_pr1.py
TZ_NAME=America/Vancouver
STOP_AT="${STOP_AT:?set STOP_AT=\x27YYYY-MM-DD HH:MM:SS\x27}"
STOP_EPOCH_WANT=$(date -d "$STOP_AT" +%s)
LATEST_STOP_S=300
MAX_EARLY_S=3600
MODE_RE='moe-batch=|mirror-reserve|DSV4 CUDA\] device|out of memory|[Vv]ulkan|\[VK|vk_'
export TZ=$TZ_NAME

MODEL=/home/maplemercury/colibri-models/deepseek-v4-flash-reap
declare -A TREE CLIENT TMO XENV EST FAILED MDL CTX
arm() { # NAME TREE MODEL CTX CLIENT TIMEOUT EST [KEY=VAL...]
    local n=$1; TREE[$n]=$2; MDL[$n]=$3; CTX[$n]=$4; CLIENT[$n]=$5; TMO[$n]=$6; EST[$n]=$7; shift 7; XENV[$n]="$*"
}
BANKE="COLI_CUDA_MOE_BATCH=1"
RV="DSV4_CUDA_VRAM_RESERVE_MB=3100"
VKO="COLI_VULKAN=1 DSV4_CUDA=0 COLI_CUDA=0"
arm WQ  "$DEVV" "$MODEL" 8192 pr1:both 3600 400  $VKO
arm QD  "$DEVV" "$MODEL" 8192 pr1:both 3600 400  $VKO
arm QE  "$PR1V" "$MODEL" 8192 pr1:both 3600 400  $VKO
arm VA9 "$DEVV" "$MODEL" 8192 pr1:both 2400 600  COLI_VULKAN=1 $RV
arm VB9 "$PR1V" "$MODEL" 8192 pr1:both 2400 450  $BANKE COLI_VULKAN=1 $RV
arm VD9 "$DEVV" "$MODEL" 8192 pr1:both 2400 600  COLI_VULKAN=1 COLI_VK_DENSE=0 $RV
arm VE9 "$PR1V" "$MODEL" 8192 pr1:both 2400 450  $BANKE COLI_VULKAN=1 COLI_VK_DENSE=0 $RV
arm VX9 "$DEVV" "$MODEL" 8192 pr1:both 2400 600  COLI_VULKAN=0 $RV
ORDER="WQ:1 QD:1 QE:1 QE:2 QD:2 QD:3 QE:3 VA9:1 VB9:1 VD9:1 VE9:1 VX9:1"
[ -n "${ORDER_OVERRIDE:-}" ] && ORDER=$ORDER_OVERRIDE

lock_or_exit
trap 'log "signal before the llama stop: exiting, llama untouched"; [ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null; exit 143' TERM INT HUP
log "run_final9.sh start (pid $$, unit $(sed 's#.*/##' /proc/self/cgroup), PILOT=${PILOT:-0}, model $MODEL)"

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

# ---- 1. the tips
isanc() { git -C "$REPO" merge-base --is-ancestor "$1" "$2" && echo yes || echo NO; }
{
    echo "resolved_at=$(date -Is)"; echo "repo=$REPO"; echo "model=$MODEL"
    echo "dev_sha=$DEV_SHA"; echo "dev_subject=$(git -C "$REPO" log -1 --format=%s "$DEV_SHA")"
    echo "pr1_sha=$PR1_SHA"; echo "pr1_tree=$(git -C "$REPO" rev-parse "$PR1_SHA^{tree}")"
    echo "pr1_subject=$(git -C "$REPO" log -1 --format=%s "$PR1_SHA")"
    echo "dev_is_ancestor_of_pr1=$(isanc "$DEV_SHA" "$PR1_SHA")"
    echo "pr1_vs_dev_files="; git -C "$REPO" diff --stat "$DEV_SHA" "$PR1_SHA" | tail -1
} > "$OUT/meta/tips.txt" 2>&1
grep -q '^dev_is_ancestor_of_pr1=yes' "$OUT/meta/tips.txt" || die_pre "dev $DEV_SHA is not an ancestor of $PR1_SHA"
log "tips: dev $DEV_SHA, pr1 $PR1_SHA"

# ---- 2. the builds (cw22/build_vk9.sh, before this run)
for t in "$DEVV:$DEV_SHA:dev" "$PR1V:$PR1_SHA:pr1"; do
    d=${t%%:*}; r=${t#*:}; sha=${r%%:*}; n=${r#*:}; blog=$W22/build-vk9-$n.log
    [ "$(git -C "$d" rev-parse HEAD 2>/dev/null)" = "$sha" ] || die_pre "$d is not at $sha"
    [ -z "$(git -C "$d" status --porcelain --untracked-files=no)" ] || die_pre "$d has local changes"
    [ -x "$d/c/deepseek_v4" ] || die_pre "$d/c/deepseek_v4 not built"
    grep -q "sha=$sha" "$blog" && grep -q '^EXIT=0' "$blog" || die_pre "$blog is not a clean build of $sha"
    newer=$(find "$d/c" -maxdepth 1 -type f \( -name '*.c' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' -o -name 'Makefile*' \) -newer "$d/c/deepseek_v4" | head -3)
    [ -z "$newer" ] || die_pre "$d binary older than its sources: $newer"
    cp "$blog" "$BUILDS/$(basename "$d").build.log" 2>/dev/null || { mkdir -p "$BUILDS"; cp "$blog" "$BUILDS/$(basename "$d").build.log"; }
    log "VK build $d: $(git -C "$d" rev-parse --short HEAD) binary $(sha256sum "$d/c/deepseek_v4" | cut -c1-16)"
done
DEV_BIN_SHA=$(sha256sum "$DEVV/c/deepseek_v4" | cut -c1-64)
PR1_BIN=$(sha256sum "$PR1V/c/deepseek_v4" | cut -c1-64)

# ---- 3. preflight (nothing touched)
preflight_common bench_pr1.py bench_f6.py bench_decode.py
[ "$(sha256sum < "$CLIENTS/bench_pr1.py" | cut -c1-64)" = "$CLIENT_SHA_PR1" ] || die_pre "bench_pr1.py is not the 2026-10-15 client"
[ -f "$MODEL/config.json" ] || die_pre "REAP model missing at $MODEL"
preflight_switch "$DEVV" COLI_VK_TIER_HOST_GB COLI_VK_TIER_STREAM_ROWS COLI_VK_TIER_BALANCE COLI_VK_DENSE DSV4_CUDA_VRAM_RESERVE_MB
preflight_switch "$PR1V" COLI_VK_TIER_HOST_GB COLI_VK_TIER_STREAM_ROWS COLI_VK_TIER_BALANCE COLI_VK_DENSE DSV4_CUDA_VRAM_RESERVE_MB
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
log "preflight ok: dev binary $DEV_BIN_SHA, pr1 binary $PR1_BIN"

# ---- 4. wait for STOP_AT (re-check 20 s before)
recheck() {
    local why=
    systemctl --user is-active -q "$LLAMA_UNIT" && llama_healthy || why="$why llama not healthy;"
    ss -ltnH "sport = :$PORT" | grep -q . && why="$why port $PORT in use;"
    [ -z "$(other_colibri_units)" ] || why="$why other colibri unit: $(other_colibri_units | tr '\n' ' ');"
    [ "$(sha256sum "$PR1V/c/deepseek_v4" | cut -c1-64)" = "$PR1_BIN" ] || why="$why $PR1V binary changed;"
    [ "$(sha256sum "$DEVV/c/deepseek_v4" | cut -c1-64)" = "$DEV_BIN_SHA" ] || why="$why $DEVV binary changed;"
    local t
    for t in "$DEVV" "$PR1V"; do
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
capture_meta "$DEVV" "$PR1V"
stop_llama
start_watchdog

for item in $ORDER; do
    arm=${item%%:*}; k=${item#*:}
    if over_budget "$(est_for "$arm" "${EST[$arm]}")"; then log "budget: skipping $arm s$k (GPU window $(gpu_elapsed) s)"; continue; fi
    SERVE_ARGS="--model ${MDL[$arm]} --ram 40 --ctx ${CTX[$arm]} --host 127.0.0.1 --port $PORT"
    # shellcheck disable=SC2086
    sample "$arm" "$k" "${TREE[$arm]}" "${CLIENT[$arm]}" "${TMO[$arm]}" ${XENV[$arm]}
    python3 "$FINAL/final9_tools.py" check "$arm" "$OUT/$arm/s$k" 2>&1 | tee -a "$OUT/run.log"
    # final9: a premise failure (an OOM kill above all) is recorded and the plan goes on
    [ "${PIPESTATUS[0]}" != 0 ] && log "$arm s$k broke a premise (recorded; the run goes on)"
done

finish
