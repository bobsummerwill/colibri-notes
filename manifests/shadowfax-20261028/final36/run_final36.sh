#!/bin/bash
# run_final36.sh (2026-10-28) -- final36: #2405 (PR #2) refreshed onto dev d68e16b6. Trees cw22/f36-base (dev d68e16b6)
#   and cw22/f36-pr2 (f1e56b00 = PR #2 11c40038 merged with d68e16b6), built by final36/build_f36.sh (final27's line).
#   Re-measures every row of #2405's description on the new base, both trees in the same session, interleaved:
#     REAP decode (bench_decode.py, 6 x 300 tokens): B1, P1 (VRAM-first), P2 (+ peer GPU 1), 2 samples each.
#     V4 Flash fdec (3 x 300 tokens): DB1F, DP1F, DP2F, 2 samples each.
#     V4 Flash long prompts (bench_pr1.py both, TTFT): B1F, P1F, P2F, 1 sample each.
#     S7: the F18 fault test (final13's S7c): DSV4_CUDA_PEER_FAULT=3000:sticky, all 6 answers must complete.
#   Then dsv4-cuda-test and cuda-test on the PR tree (final27's step 6). final30's runner otherwise unchanged.
EXP=${EXP_NAME:-bench}
[ "${PILOT:-0}" = 1 ] && EXP=pilot
[ "${DRY_RUN:-0}" = 1 ] && EXP=dryrun
W22=/home/maplemercury/colibri-bench/cw22
BASE=$W22/f36-base; PR2=$W22/f36-pr2
NVF="-O3 -std=c++17 -ftz=false -arch=sm_86 -Xcompiler=-Wall,-Wextra -DCOLI_DSV4_NO_TC"
MAX_GPU_S=${MAX_GPU_S:?set MAX_GPU_S (s)}   # cap on the llama-down window (<= 4 h; ends before 05:44)
NVSMI_MS=5000
BUILDS=${BUILDS:-/home/maplemercury/colibri-bench/final36/builds}
source /home/maplemercury/colibri-bench/final36/final_common.sh

REPO=$W22/repo
DEV_SHA=d68e16b6
PR2_SHA=f1e56b00
BASE_SHA=d68e16b6
TZ_NAME=America/Vancouver
STOP_AT="${STOP_AT:?set STOP_AT=\x27YYYY-MM-DD HH:MM:SS\x27}"
STOP_EPOCH_WANT=$(date -d "$STOP_AT" +%s)
LATEST_STOP_S=300
MAX_EARLY_S=3600
MODE_RE='\[VK|host pool|placement=|v4_gpu device=[0-9] layers|moe-batch=|mirror-reserve|peer-mirrors|DSV4 CUDA\] device|out of memory|CUDA is lost|PEER_FAULT|DSV4_CUDA_DEVICES|[Rr]esiden|layer[s]? [0-9]+-[0-9]+|device [0-9]: layers'
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
# EST (s, sample wall incl. load): final27 B1 632, P1 614, P2 ~460; final30 fdec ~660/600; final27 pr1:both 241-300.
arm B1   "$BASE" "$REAP" 8192 decode   1800 650  $BANKE
arm P1   "$PR2"  "$REAP" 8192 decode   1800 650  $BANKE V4_GPU_RESIDENT_FIRST=1
arm P2   "$PR2"  "$REAP" 8192 decode   1800 480  $BANKE $PEER
arm DB1F "$BASE" "$FULL" 8192 fdec     2400 680  $BANKE
arm DP1F "$PR2"  "$FULL" 8192 fdec     2400 680  $BANKE V4_GPU_RESIDENT_FIRST=1
arm DP2F "$PR2"  "$FULL" 8192 fdec     2400 620  $BANKE $PEER
arm B1F  "$BASE" "$FULL" 8192 pr1:both 3600 280  $BANKE
arm P1F  "$PR2"  "$FULL" 8192 pr1:both 3600 300  $BANKE V4_GPU_RESIDENT_FIRST=1
arm P2F  "$PR2"  "$FULL" 8192 pr1:both 3600 300  $BANKE $PEER
arm S7   "$PR2"  "$REAP" 8192 decode   1800 720  $BANKE V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1 DSV4_CUDA_PEER_FAULT=3000:sticky
ORDER1=${ORDER1_OVERRIDE:-"B1:1 P1:1 P2:1 P2:2 P1:2 B1:2"}
ORDER=${ORDER2_OVERRIDE:-"DB1F:1 DP1F:1 DP2F:1 DP2F:2 DP1F:2 DB1F:2"}
ORDER3=${ORDER3_OVERRIDE:-"B1F:1 P1F:1 P2F:1 S7:1"}

lock_or_exit
trap 'log "signal before the llama stop: exiting, llama untouched"; [ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null; exit 143' TERM INT HUP
log "run_final36.sh start (pid $$, unit $(sed 's#.*/##' /proc/self/cgroup), PILOT=${PILOT:-0})"

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
[ "$lead" -le "$MAX_EARLY_S" ] || [ "${DRY_RUN:-0}" = 1 ] || die_pre "started $lead s before $STOP_AT (more than $MAX_EARLY_S s early)"
[ "$lead" -ge "-$LATEST_STOP_S" ] || die_pre "started $(( -lead )) s after $STOP_AT: the window is missed"

# ---- 1. the tips and builds
declare -A BIN
for t in "$BASE:$BASE_SHA" "$PR2:$PR2_SHA"; do
    dir=${t%%:*}; full=$(git -C "$REPO" rev-parse "${t#*:}^{commit}") || die_pre "${t#*:} not in $REPO"
    preflight_tree "$dir" "$full"
    BIN[$dir]=$(sha256sum "$dir/c/deepseek_v4" | cut -c1-64)
    echo "$(basename "$dir")=$full binary=${BIN[$dir]}" >> "$OUT/meta/tips.txt"
done
log "tips: $(tr '\n' ' ' < "$OUT/meta/tips.txt")"

# ---- 3. preflight (nothing touched)
preflight_common bench_decode.py bench_pr1.py bench_fdec.py
[ -f "$FULL/config.json" ] || die_pre "full model missing at $FULL"
[ -f "$REAP/config.json" ] || die_pre "REAP model missing at $REAP"
preflight_switch "$PR2" V4_GPU_RESIDENT_FIRST DSV4_CUDA_PEER_DEVICE DSV4_CUDA_PEER_FAULT
# final28 and final29 must be over (their locks are taken by lock_or_exit; this names them if a unit is somehow still up)
for u in colibri-build36; do systemctl --user is-active -q $u && soft_pre "$u is still running"; done
[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2 ] || die_pre "second GPU not visible"
[ -z "$(foreign_gpu_pids)" ] || soft_pre "other GPU compute processes: $(foreign_gpu_pids | tr '\n' ' ')"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/meta/gpu_apps_preflight.txt" 2>&1
[ "$(df -Pk "$FINAL" | awk 'NR==2{print $4}')" -gt 2000000 ] || die_pre "less than 2 GB free in $FINAL"
total_est=0
for item in $ORDER1 $ORDER $ORDER3; do total_est=$((total_est + ${EST[${item%%:*}]})); done
log "plan: $ORDER1 | $ORDER | $ORDER3 | no tests (estimated $total_est s of samples; cap $MAX_GPU_S s)"
[ $((total_est + 240)) -le "$MAX_GPU_S" ] || die_pre "plan estimate exceeds the cap"
[ "$MAX_GPU_S" -le 14400 ] || die_pre "MAX_GPU_S $MAX_GPU_S is more than 4 h"
for item in $ORDER1 $ORDER $ORDER3; do [ -n "${TREE[${item%%:*}]}" ] || die_pre "no arm ${item%%:*}"; done
python3 "$FINAL/final36_tools.py" arms $ORDER1 $ORDER $ORDER3 >> "$OUT/run.log" 2>&1 || die_pre "an arm of the plan has no final36_tools.py ARMS entry"
# final36: each arm's environment (COMMON_ENV + its extra env, as sample() merges it) against its ARMS premises
for item in $ORDER1 $ORDER $ORDER3; do
    a=${item%%:*}
    # shellcheck disable=SC2046,SC2086
    python3 "$FINAL/final36_tools.py" envcheck "$a" $(merge_env "${COMMON_ENV[@]}" ${XENV[$a]}) >> "$OUT/run.log" 2>&1 || die_pre "arm $a's environment breaks its premises (see run.log)"
done
if [ "${DRY_RUN:-0}" = 1 ]; then
    for item in $ORDER1 $ORDER $ORDER3; do a=${item%%:*}; log "DRY arm $a: tree=$(basename "${TREE[$a]}") client=${CLIENT[$a]} timeout=${TMO[$a]} est=${EST[$a]} env='${XENV[$a]}'"; done
    log "DRY_RUN: preflight done (${DRY_SOFT:-0} box-busy checks recorded, see above); exiting before the wait, llama untouched"
    exit 0
fi
log "preflight ok"

# ---- 4. wait for STOP_AT (re-check 20 s before)
recheck() {
    local why= dir
    systemctl --user is-active -q "$LLAMA_UNIT" && llama_healthy || why="$why llama not healthy;"
    ss -ltnH "sport = :$PORT" | grep -q . && why="$why port $PORT in use;"
    [ -z "$(other_colibri_units)" ] || why="$why other colibri unit: $(other_colibri_units | tr '\n' ' ');"
    for dir in "$BASE" "$PR2"; do
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
capture_meta "$BASE" "$PR2"
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
    python3 "$FINAL/final36_tools.py" check "$arm" "$OUT/$arm/s$k" 2>&1 | tee -a "$OUT/run.log"
    [ "${PIPESTATUS[0]}" != 0 ] && log "$arm s$k broke a premise (recorded; the run goes on)"
done
}
run_order "$ORDER1"
run_order "$ORDER"
run_order "$ORDER3"


# ---- 6. the CUDA tests on the PR tree (last: they rebuild objects in the tree; deepseek_v4 itself is untouched)
cuda_tests() { # tree name
    local t=$1 n=$2
    if over_budget 900; then log "budget: skipping the CUDA tests of $n"; return; fi
    ( cd "$t/c" && CUDA_VISIBLE_DEVICES=0,1 timeout 900 make dsv4-cuda-test CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NVCCFLAGS="$NVF" ) > "$OUT/meta/dsv4-cuda-test-$n.log" 2>&1
    log "dsv4-cuda-test $n: rc=$? ($(grep -ciE 'warning' "$OUT/meta/dsv4-cuda-test-$n.log") warning lines; $(grep -E 'dsv4 peer MoE:|sticky:' "$OUT/meta/dsv4-cuda-test-$n.log" | tr '\n' ' '))"
    if over_budget 900; then log "budget: skipping cuda-test of $n"; return; fi
    ( cd "$t" && CUDA_VISIBLE_DEVICES=0,1 timeout 900 make -C c cuda-test CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NVCCFLAGS="$NVF" ) > "$OUT/meta/cuda-test-$n.log" 2>&1
    log "cuda-test $n: rc=$?"
}
cuda_tests "$PR2" pr2

finish
