#!/bin/bash
# run_final8.sh (2026-10-23) -- final8, the questions final7 left (TEST_PLAN_20261023.md, investigations.md §7), on
# final6's trees and builds: pure Vulkan with only the pool fixed (QF dev x3, QG PR #1 x2), and the Vulkan + CUDA
# arms with one VRAM reserve for all (VA2 / VA3 / VU2 / VB2 x2, RA2 / RB2 x1; VA3 = dev with VU2's exact environment). REAP-150B only. final7's runner otherwise.

EXP=bench
[ "${PILOT:-0}" = 1 ] && EXP=pilot
W22=/home/maplemercury/colibri-bench/cw22
PR1=$W22/f6-pr1
PR2=$W22/f6-pr2
DEV=$W22/f6-dev
F18=$W22/f18
MAX_GPU_S=$((1200 * 60))        # hard cap on the llama-down window (20 h; Bob 2026-10-23: downtime is fine)
BUILDS=/home/maplemercury/colibri-bench/final6/builds   # final7: final6's build logs of the same trees
export F18_AT=${F18_AT:-3000}   # the peer pass the probe traps (~43 a token: early in the first answer)
NVSMI_MS=5000
source /home/maplemercury/colibri-bench/final8/final_common.sh

REPO=$W22/repo
DEV_SHA=7df8c27c95acf7019dd60714ec1c42fa7e36b730
PR1_SHA_WANT=b286167e
PR2_SHA_WANT=0523e8e0
F18_SHA=b62d10680ff947f0942d854f6a1a2e3a32cbbf1a
CLIENT_SHA_PR1=eca4360bdd5a0601c0a7138f49464874a648486519235bbc22d40997c3e2de53     # final/clients/bench_pr1.py
TZ_NAME=America/Vancouver
STOP_AT="${STOP_AT:?set STOP_AT=\x27YYYY-MM-DD HH:MM:SS\x27}"
STOP_EPOCH_WANT=$(date -d "$STOP_AT" +%s)
LATEST_STOP_S=300
MAX_EARLY_S=3600
MODE_RE='moe-batch=|moe-double|mirror-reserve|peer-mirrors|peer mirror pool|peer device|DSV4 CUDA\] device|mm-warp verify|^v4_serve_turn|^v4_dec_moe|out of memory|[Vv]ulkan|\[VK|vk_|MTP|mtp|dspark|DSpark|draft'
export TZ=$TZ_NAME

FULL=/home/maplemercury/colibri-models/deepseek-v4-flash
REAP=/home/maplemercury/colibri-models/deepseek-v4-flash-reap
declare -A TREE CLIENT TMO XENV EST FAILED MDL CTX
arm() { # NAME TREE MODEL CTX CLIENT TIMEOUT EST [KEY=VAL...]
    local n=$1; TREE[$n]=$2; MDL[$n]=$3; CTX[$n]=$4; CLIENT[$n]=$5; TMO[$n]=$6; EST[$n]=$7; shift 7; XENV[$n]="$*"
}
BANKE="COLI_CUDA_MOE_BATCH=1"
DEVV=$W22/f6v-dev; PR1V=$W22/f6v-pr1   # CUDA=1 VK=1 builds (final6's build_vk.sh)
RV="DSV4_CUDA_VRAM_RESERVE_MB=3100"
arm QF "$DEVV" "$REAP" 8192 pr1:both     3600 400  COLI_VULKAN=1 DSV4_CUDA=0 COLI_CUDA=0 COLI_VK_TIER_HOST_GB=12
arm QG "$PR1V" "$REAP" 8192 pr1:both     3600 400  COLI_VULKAN=1 DSV4_CUDA=0 COLI_CUDA=0 COLI_VK_TIER_HOST_GB=12
arm VA2 "$DEVV" "$REAP" 8192 pr1:both    2400 600  $BANKE COLI_VULKAN=1 $RV
arm VA3 "$DEVV" "$REAP" 8192 pr1:both    2400 600  COLI_VULKAN=1 $RV
arm VU2 "$PR1V" "$REAP" 8192 pr1:both    2400 600  COLI_VULKAN=1 $RV
arm VB2 "$PR1V" "$REAP" 8192 pr1:both    2400 450  $BANKE COLI_VULKAN=1 $RV
arm RA2 "$DEV"  "$REAP" 8192 pr1:both    2400 600  $BANKE $RV
arm RB2 "$PR1"  "$REAP" 8192 pr1:both    2400 450  $BANKE $RV
CORE=" "
ORDER="QF:1 QG:1 QF:2 QG:2 QF:3 VA2:1 VU2:1 VA3:1 VB2:1 VB2:2 VA3:2 VU2:2 VA2:2 RA2:1 RB2:1"
[ -n "${ORDER_OVERRIDE:-}" ] && ORDER=$ORDER_OVERRIDE

lock_or_exit
trap 'log "signal before the llama stop: exiting, llama untouched"; [ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null; exit 143' TERM INT HUP
log "run_final8.sh start (pid $$, unit $(sed 's#.*/##' /proc/self/cgroup), PILOT=${PILOT:-0}, model $MODEL)"

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
PR1_SHA=$(git -C "$REPO" rev-parse --verify -q "$PR1_SHA_WANT^{commit}") || die_pre "$PR1_SHA_WANT not in $REPO"
PR2_SHA=$(git -C "$REPO" rev-parse --verify -q "$PR2_SHA_WANT^{commit}") || die_pre "$PR2_SHA_WANT not in $REPO"
isanc() { git -C "$REPO" merge-base --is-ancestor "$1" "$2" && echo yes || echo NO; }
{
    echo "resolved_at=$(date -Is)"; echo "repo=$REPO"; echo "model=$MODEL"
    echo "dev_sha=$DEV_SHA"; echo "dev_subject=$(git -C "$REPO" log -1 --format=%s "$DEV_SHA")"
    echo "pr1_sha=$PR1_SHA"; echo "pr1_tree=$(git -C "$REPO" rev-parse "$PR1_SHA^{tree}")"
    echo "pr1_subject=$(git -C "$REPO" log -1 --format=%s "$PR1_SHA")"
    echo "pr2_sha=$PR2_SHA"; echo "pr2_subject=$(git -C "$REPO" log -1 --format=%s "$PR2_SHA")"
    echo "dev_is_ancestor_of_pr1=$(isanc "$DEV_SHA" "$PR1_SHA")"; echo "pr1_is_ancestor_of_pr2=$(isanc "$PR1_SHA" "$PR2_SHA")"
    echo "pr1_vs_dev_files="; git -C "$REPO" diff --stat "$DEV_SHA" "$PR1_SHA" | tail -1
} > "$OUT/meta/tips.txt" 2>&1
grep -q '^dev_is_ancestor_of_pr1=yes' "$OUT/meta/tips.txt" || die_pre "dev $DEV_SHA is not an ancestor of $PR1_SHA"
log "tips: dev $DEV_SHA, pr1 $PR1_SHA, pr2 $PR2_SHA"

# ---- 2. builds: final6's trees as they are (checked below); the probe tree built as a server (CPU only)
DEV_BIN_SHA=$(sha256sum "$DEV/c/deepseek_v4" | cut -c1-64)
PR1_BIN=$(sha256sum "$PR1/c/deepseek_v4" | cut -c1-64)
PR2_BIN=$(sha256sum "$PR2/c/deepseek_v4" | cut -c1-64)

for t in "$DEVV" "$PR1V"; do
    if [ -x "$t/c/deepseek_v4" ] && grep -q '^EXIT=0' "$W22/build-vk-${t##*-}.log" 2>/dev/null; then
        log "VK build $t: $(git -C "$t" rev-parse --short HEAD) binary $(sha256sum "$t/c/deepseek_v4" | cut -c1-16)"
    else
        log "WARNING: no clean VK build in $t: its arms are skipped"; [ "$t" = "$DEVV" ] && { FAILED[VA]=1; FAILED[QA]=1; FAILED[QP]=1; FAILED[QO]=1; } || { FAILED[VB]=1; FAILED[VU]=1; FAILED[QQ]=1; }
    fi
done
[ "$(git -C "$DEVV" rev-parse HEAD 2>/dev/null)" = "$DEV_SHA" ] || die_pre "$DEVV is not at dev"
[ "$(git -C "$PR1V" rev-parse HEAD 2>/dev/null)" = "$PR1_SHA" ] || die_pre "$PR1V is not at pr1"

# ---- 3. preflight (nothing touched)
preflight_common bench_pr1.py bench_f6.py bench_decode.py
[ "$(sha256sum < "$CLIENTS/bench_decode.py" | cut -c1-64)" = 06b2475ac64144ca512bee9d2fc85211227276f808b903e001227ac1ed78b644 ] || die_pre "bench_decode.py is not the 2026-10-15 client"
[ -f "$REAP/config.json" ] || die_pre "REAP model missing at $REAP"
[ "$(sha256sum < "$CLIENTS/bench_pr1.py" | cut -c1-64)" = "$CLIENT_SHA_PR1" ] || die_pre "bench_pr1.py is not the 2026-10-15 client"
[ -f "$MODEL/config.json" ] || die_pre "no $MODEL/config.json (download not finished?)"
[ -z "$(find "$MODEL" -name '*.incomplete' -print -quit)" ] || die_pre "$MODEL has .incomplete files (download not finished)"
nshard=$(ls "$MODEL"/model-*-of-*.safetensors 2>/dev/null | wc -l)
want_shard=$(ls "$MODEL"/model-*-of-*.safetensors 2>/dev/null | head -1 | sed 's/.*-of-0*\([0-9]*\)\.safetensors/\1/')
[ -n "$want_shard" ] && [ "$nshard" = "$want_shard" ] || die_pre "$MODEL has $nshard shards, expected ${want_shard:-?}"
[ "$(python3 -c "import json;print(json.load(open('$MODEL/config.json')).get('n_routed_experts'))")" = 256 ] \
    || die_pre "$MODEL is not the 256-expert checkpoint"
preflight_tree "$DEV" "$DEV_SHA"
preflight_tree "$PR1" "$PR1_SHA"
preflight_tree "$PR2" "$PR2_SHA"
preflight_switch "$DEV" DSV4_CUDA_VRAM_RESERVE_MB DSV4_HYBRID
preflight_switch "$PR2" DSV4_HYBRID
preflight_switch "$DEVV" COLI_VK_TIER_HOST_GB COLI_VK_TIER_STREAM_ROWS COLI_VK_TIER_BALANCE
preflight_switch "$PR1V" COLI_VK_TIER_HOST_GB COLI_VK_TIER_STREAM_ROWS COLI_VK_TIER_BALANCE
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
log "preflight ok: dev binary $DEV_BIN_SHA, pr1 $PR1_SHA binary $PR1_BIN"

# ---- 4. wait for STOP_AT (re-check 20 s before)
recheck() {
    local why=
    systemctl --user is-active -q "$LLAMA_UNIT" && llama_healthy || why="$why llama not healthy;"
    ss -ltnH "sport = :$PORT" | grep -q . && why="$why port $PORT in use;"
    [ -z "$(other_colibri_units)" ] || why="$why other colibri unit: $(other_colibri_units | tr '\n' ' ');"
    [ "$(sha256sum "$PR1/c/deepseek_v4" | cut -c1-64)" = "$PR1_BIN" ] || why="$why $PR1 binary changed;"
    [ "$(sha256sum "$DEV/c/deepseek_v4" | cut -c1-64)" = "$DEV_BIN_SHA" ] || why="$why $DEV binary changed;"
    [ "$(sha256sum "$PR2/c/deepseek_v4" | cut -c1-64)" = "$PR2_BIN" ] || why="$why $PR2 binary changed;"
    local t
    for t in "$DEV" "$PR1" "$PR2"; do
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
capture_meta "$DEV" "$PR1" "$PR2" "$DEVV" "$PR1V"
stop_llama
start_watchdog

NFAIL=0
for item in $ORDER; do
    arm=${item%%:*}; k=${item#*:}
    if [ -n "${FAILED[$arm]}" ]; then log "skipping $arm s$k: an earlier $arm sample broke a premise"; continue; fi
    if over_budget "$(est_for "$arm" "${EST[$arm]}")"; then log "budget: skipping $arm s$k (GPU window $(gpu_elapsed) s)"; continue; fi
    MODEL=${MDL[$arm]}   # final6: the model and context of this arm (sample() and usage_state read these)
    SERVE_ARGS="--model $MODEL --ram 40 --ctx ${CTX[$arm]} --host 127.0.0.1 --port $PORT"
    # shellcheck disable=SC2086
    sample "$arm" "$k" "${TREE[$arm]}" "${CLIENT[$arm]}" "${TMO[$arm]}" ${XENV[$arm]}
    python3 "$FINAL/final8_tools.py" check "$arm" "$OUT/$arm/s$k" 2>&1 | tee -a "$OUT/run.log"
    if [ "${PIPESTATUS[0]}" != 0 ]; then
        FAILED[$arm]=1
        case "$CORE" in *" $arm "*) NFAIL=$((NFAIL + 1)) ;; esac
        if [ "$NFAIL" -ge 2 ]; then log "ABORT after $arm s$k: a second core sample broke a premise"; exit 8; fi
        log "$arm s$k broke a premise: the rest of $arm is skipped"
    fi
done

finish
