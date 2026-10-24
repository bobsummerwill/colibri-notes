#!/bin/bash
# run_final11.sh (2026-10-24) -- final11 (OVERNIGHT_20261024.md), REAP-150B: 1. DA2 / DB2, the Vulkan dense hook's
# declines with their reasons (diagnostic builds v2); 2. M3 (pure Vulkan, default pool, smaps every 15 s) and M4
# (the pool off); 3. PR #2 + the F18 fix: dsv4-cuda-test and cuda-test, then S7 (a sticky fault in a real server,
# DSV4_CUDA_PEER_FAULT=3000:sticky); 4. PR #2 decode retest WR D0 G1 X3 X3 G1 D0 D0 G1 X3. final10's runner otherwise;
# builds by build_f11.sh before this run.

EXP=bench
[ "${PILOT:-0}" = 1 ] && EXP=pilot
W22=/home/maplemercury/colibri-bench/cw22
DEVV=$W22/f9v-dev                       # CUDA=1 VK=1 build of dev 61646cbd (cw22/build_vk9.sh, final9)
DEV=$W22/f10-dev                         # CUDA build of dev 61646cbd (final10)
PR2F=$W22/f11-pr2f                       # CUDA build of PR #2 + the F18 fix (build_f11.sh)
D2DEV=$W22/f11d-dev; D2PR1=$W22/f11d-pr1 # CUDA=1 VK=1 diagnostic builds v2 (build_f11.sh)
NVF="-O3 -std=c++17 -ftz=false -arch=sm_86 -Xcompiler=-Wall,-Wextra -DCOLI_DSV4_NO_TC"
MAX_GPU_S=$((300 * 60))         # cap on the llama-down window (5 h; the plan is ~2.8 h)
NVSMI_MS=5000
source /home/maplemercury/colibri-bench/final11/final_common.sh

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
RV="DSV4_CUDA_VRAM_RESERVE_MB=3100"
BANKE="COLI_CUDA_MOE_BATCH=1"
R2="$BANKE V4_SERVE_STATS_LOG=1"
arm DA2 "$D2DEV" "$REAP" 8192 pr1:both 2400 600  COLI_VULKAN=1 $RV
arm DB2 "$D2PR1" "$REAP" 8192 pr1:both 2400 450  $BANKE COLI_VULKAN=1 $RV
arm M3  "$DEVV"  "$REAP" 8192 pr1:both 3600 400  $VKO
arm M4  "$DEVV"  "$REAP" 8192 pr1:both 3600 400  $VKO COLI_VK_TIER_HOST=0
arm S7  "$PR2F"  "$REAP" 8192 decode   2400 900  $BANKE V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1 DSV4_CUDA_PEER_FAULT=3000:sticky
arm WR  "$PR2F"  "$REAP" 8192 decode   1500 700  $R2
arm D0  "$DEV"   "$REAP" 8192 decode   1500 720  $BANKE
arm G1  "$PR2F"  "$REAP" 8192 decode   1500 700  $R2
arm X3  "$PR2F"  "$REAP" 8192 decode   1500 520  $R2 V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1
ORDER1="DA2:1 DB2:1 M3:1 M4:1"   # then the CUDA tests at PR #2 + F18 (section 6), S7, the decode retest
ORDER="S7:1 WR:1 D0:1 G1:1 X3:1 X3:2 G1:2 D0:2 D0:3 G1:3 X3:3"
[ -n "${ORDER_OVERRIDE:-}" ] && ORDER=$ORDER_OVERRIDE

lock_or_exit
trap 'log "signal before the llama stop: exiting, llama untouched"; [ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null; exit 143' TERM INT HUP
log "run_final11.sh start (pid $$, unit $(sed 's#.*/##' /proc/self/cgroup), PILOT=${PILOT:-0}, model $MODEL)"

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
PR2F_SHA=$(git -C "$PR2F" rev-parse HEAD)
{ echo "resolved_at=$(date -Is)"; echo "dev_sha=$DEV_SHA"; echo "pr2f_sha=$PR2F_SHA $(git -C "$PR2F" log -1 --format=%s)";
  echo "pr2f_parent=$(git -C "$PR2F" rev-parse HEAD^)"; echo "d2dev=$(git -C "$D2DEV" rev-parse HEAD) d2pr1=$(git -C "$D2PR1" rev-parse HEAD)"; } > "$OUT/meta/tips.txt" 2>&1
preflight_tree "$DEV" "$DEV_SHA"
preflight_tree "$PR2F" "$PR2F_SHA"
[ "$(git -C "$DEVV" rev-parse HEAD)" = "$DEV_SHA" ] && grep -q '^EXIT=0' "$W22/build-vk9-dev.log" || die_pre "$DEVV is not a clean VK build of dev"
for t in "$D2DEV:dev" "$D2PR1:pr1"; do d=${t%%:*}; n=${t#*:}
    grep -q '^EXIT=0' "$W22/build-vk11d-$n.log" && [ -x "$d/c/deepseek_v4" ] || die_pre "$d: no clean diagnostic build"
    [ "$(git -C "$d" diff --name-only HEAD^ HEAD)" = c/deepseek_v4.c ] || die_pre "$d: the diagnostic commit changes more than deepseek_v4.c"
done
DEV_BIN_SHA=$(sha256sum "$DEV/c/deepseek_v4" | cut -c1-64)
PR1_BIN=$(sha256sum "$PR2F/c/deepseek_v4" | cut -c1-64)
DEVV_BIN=$(sha256sum "$DEVV/c/deepseek_v4" | cut -c1-64)
log "tips: dev $DEV_SHA, PR #2 + F18 $PR2F_SHA; binaries $DEV_BIN_SHA $PR1_BIN, VK $DEVV_BIN"

# ---- 3. preflight (nothing touched)
preflight_common bench_pr1.py bench_f6.py bench_decode.py
[ "$(sha256sum < "$CLIENTS/bench_pr1.py" | cut -c1-64)" = "$CLIENT_SHA_PR1" ] || die_pre "bench_pr1.py is not the 2026-10-15 client"
[ -f "$REAP/config.json" ] || die_pre "REAP model missing at $REAP"
preflight_switch "$DEVV" COLI_VK_TIER_HOST
preflight_switch "$PR2F" V4_GPU_RESIDENT_FIRST DSV4_CUDA_PEER_DEVICE DSV4_CUDA_PEER_FAULT
grep -q ':sticky' "$PR2F/c/backend_cuda_dsv4.cu" || die_pre "$PR2F has no :sticky fault mode"
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
    [ "$(sha256sum "$PR2F/c/deepseek_v4" | cut -c1-64)" = "$PR1_BIN" ] || why="$why $PR2F binary changed;"
    [ "$(sha256sum "$DEV/c/deepseek_v4" | cut -c1-64)" = "$DEV_BIN_SHA" ] || why="$why $DEV binary changed;"
    [ "$(sha256sum "$DEVV/c/deepseek_v4" | cut -c1-64)" = "$DEVV_BIN" ] || why="$why $DEVV binary changed;"
    local t
    for t in "$DEV" "$PR2F" "$DEVV" "$D2DEV" "$D2PR1"; do
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
capture_meta "$DEV" "$PR2F" "$DEVV" "$D2DEV" "$D2PR1"
stop_llama
start_watchdog

run_order() {
for item in $1; do
    arm=${item%%:*}; k=${item#*:}
    if over_budget "$(est_for "$arm" "${EST[$arm]}")"; then log "budget: skipping $arm s$k (GPU window $(gpu_elapsed) s)"; continue; fi
    MODEL=${MDL[$arm]}
    [ "$arm" = M3 ] && { mkdir -p "$OUT/$arm/s$k"; echo 3 > "$OUT/$arm/s$k/smaps_every"; }
    SERVE_ARGS="--model ${MDL[$arm]} --ram 40 --ctx ${CTX[$arm]} --host 127.0.0.1 --port $PORT"
    # shellcheck disable=SC2086
    sample "$arm" "$k" "${TREE[$arm]}" "${CLIENT[$arm]}" "${TMO[$arm]}" ${XENV[$arm]}
    python3 "$FINAL/final11_tools.py" check "$arm" "$OUT/$arm/s$k" 2>&1 | tee -a "$OUT/run.log"
    # final9: a premise failure (an OOM kill above all) is recorded and the plan goes on
    [ "${PIPESTATUS[0]}" != 0 ] && log "$arm s$k broke a premise (recorded; the run goes on)"
done
}
run_order "$ORDER1"

# ---- 6. PR #2 + F18: dsv4-cuda-test (the peer tests, the new :sticky scenario among them) and cuda-test
if over_budget 1800; then log "budget: skipping the CUDA tests"; else
    ( cd "$PR2F/c" && CUDA_VISIBLE_DEVICES=0,1 timeout 900 make dsv4-cuda-test CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NVCCFLAGS="$NVF" ) > "$OUT/meta/dsv4-cuda-test-pr2f.log" 2>&1
    log "dsv4-cuda-test pr2f: rc=$? ($(grep -ciE 'warning' "$OUT/meta/dsv4-cuda-test-pr2f.log") warning lines; $(grep -E 'dsv4 peer MoE:|sticky:' "$OUT/meta/dsv4-cuda-test-pr2f.log" | tr '\n' ' '))"
    ( cd "$PR2F" && CUDA_VISIBLE_DEVICES=0,1 timeout 900 make -C c cuda-test CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NVCCFLAGS="$NVF" ) > "$OUT/meta/cuda-test-pr2f.log" 2>&1
    log "cuda-test pr2f: rc=$?"
fi

run_order "$ORDER"

finish
