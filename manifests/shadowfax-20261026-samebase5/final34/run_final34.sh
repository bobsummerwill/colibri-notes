#!/bin/bash
# run_final34.sh (2026-10-26) -- final34: PR #2's four-way split on dev c1c06ac6, each step on one GPU, REAP decode, after
#   the final27/final28 chain. S1 = split/pr2-1-guard 6a191e90, S2 = split/pr2-2-settings c45b6962, S3 =
#   split/pr2-3-vram-first 695a7be4 (V4_GPU_RESIDENT_FIRST=1); S4 = P's tree, so no arm. Anchors in the same session:
#   B1 (f27-base) and P1 (f27-pr2, VRAM-first, no peer). 2 each for S1-S3, 1 each for the anchors, interleaved
#   B1 S1 S2 S3 P1 S3 S2 S1. One text group: every arm must give B's texts. Trees cw22/f29-s1..s3 (build_f29.sh).
#   No CUDA tests (the steps' tests ran on counter). final27's runner, new arms. Below: final27's header.
# run_final27.sh (2026-10-26) -- final27: the same-base run RESTARTED on dev c1c06ac6 (#2401-#2404 merged upstream;
#   final25/26 stopped). Trees cw22/f27-*: B = c1c06ac6, P = samebase/pr2 11c40038 (PR #2 rebased, same 11 patches),
#   M = samebase/m1766 ecc92586, C+ = samebase/cplus b659ac8d. Built by final27/build_f27.sh. Below: final25's header.
# run_final25.sh (2026-10-26) -- final25: the same-base run, Stage 1 of colibri-notes review/PLAN_SAME_BASE_20261026.md.
#   Every tree on dd03a9de (dev 77530b49 + #2401): B = dd03a9de (f22-base), M = #1766 merged onto it (fork samebase/m1766,
#   f25-m1766), P = PR #2 5b3f4601 (f22-pr2). One base, so every CUDA arm must give the same texts (one group per model).
#   REAP-150B, bench_decode.py, 2 each, interleaved: B1, M1, P1 (VRAM-first, no peer) on one GPU; M2
#   (DSV4_CUDA_DEVICES=0,1), P2 (peer GPU 1) on two. The same on the full V4 Flash (bench_pr1.py), 2 each. Pure Vulkan on
#   one GPU (VKB/VKM/VKP, CUDA=1 VK=1 builds, final22's VD environment and client), 1 each. Then dsv4-cuda-test and
#   cuda-test on B, P, M and #1766's test_dsv4_multigpu_cuda 0,1 on M.
# MAX_GPU_S from the environment. final24's runner, new arms and trees.
EXP=${EXP_NAME:-bench}
[ "${PILOT:-0}" = 1 ] && EXP=pilot
W22=/home/maplemercury/colibri-bench/cw22
BASE=$W22/f27-base; PR2=$W22/f27-pr2
CPLUS=$W22/f27-cplus; CF4=$W22/f33-cplusf4
NVF="-O3 -std=c++17 -ftz=false -arch=sm_86 -Xcompiler=-Wall,-Wextra -DCOLI_DSV4_NO_TC"
MAX_GPU_S=${MAX_GPU_S:?set MAX_GPU_S (s)}   # cap on the llama-down window (<= 4 h; ends before 05:44)
NVSMI_MS=5000
source /home/maplemercury/colibri-bench/final34/final_common.sh

REPO=$W22/repo
DEV_SHA=c1c06ac6
PR2_SHA=11c40038
BASE_SHA=c1c06ac6
CPLUS_SHA=b659ac8d   # final34: C+ as benchmarked in final28
CF4_SHA=e2933955    # final34: fork samebase/cplus e2933955 = dec72809 + dev table f4
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
VKO="COLI_VULKAN=1 DSV4_CUDA=0 COLI_CUDA=0"
arm B1  "$BASE"  "$REAP" 8192 decode   1800 640  $BANKE
arm C1  "$CPLUS" "$REAP" 8192 decode   1800 600  $BANKE V4_GPU_RESIDENT_FIRST=1
arm N1  "$CF4"   "$REAP" 8192 decode   1800 600  $BANKE V4_GPU_RESIDENT_FIRST=1
arm C2  "$CPLUS" "$REAP" 8192 decode   1800 460  $BANKE $PEER
arm N2  "$CF4"   "$REAP" 8192 decode   1800 460  $BANKE $PEER
ORDER1=${ORDER1_OVERRIDE:-"B1:1 C2:1 N2:1 C1:1 N1:1 N2:2 C2:2"}   # final34: C+ f4 fix on REAP
ORDER=""
ORDER3=""

lock_or_exit
trap 'log "signal before the llama stop: exiting, llama untouched"; [ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null; exit 143' TERM INT HUP
log "run_final34.sh start (pid $$, unit $(sed 's#.*/##' /proc/self/cgroup), PILOT=${PILOT:-0})"

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
for t in "$BASE:$BASE_SHA" "$PR2:$PR2_SHA" "$CPLUS:$CPLUS_SHA" "$CF4:$CF4_SHA"; do
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
preflight_switch "$CPLUS" V4_GPU_RESIDENT_FIRST DSV4_CUDA_PEER_DEVICE DSV4_CUDA_PEER_FAULT
preflight_switch "$CF4" V4_GPU_RESIDENT_FIRST DSV4_CUDA_PEER_DEVICE DSV4_CUDA_PEER_FAULT
[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2 ] || die_pre "second GPU not visible"
[ -z "$(foreign_gpu_pids)" ] || die_pre "other GPU compute processes: $(foreign_gpu_pids | tr '\n' ' ')"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/meta/gpu_apps_preflight.txt" 2>&1
[ "$(df -Pk "$FINAL" | awk 'NR==2{print $4}')" -gt 2000000 ] || die_pre "less than 2 GB free in $FINAL"
total_est=0
for item in $ORDER1 $ORDER $ORDER3; do total_est=$((total_est + ${EST[${item%%:*}]})); done
log "plan: $ORDER1 | no tests (estimated $total_est s of samples; cap $MAX_GPU_S s)"
[ $((total_est + 240)) -le "$MAX_GPU_S" ] || die_pre "plan estimate exceeds the cap"
log "preflight ok"

# ---- 4. wait for STOP_AT (re-check 20 s before)
recheck() {
    local why= dir
    systemctl --user is-active -q "$LLAMA_UNIT" && llama_healthy || why="$why llama not healthy;"
    ss -ltnH "sport = :$PORT" | grep -q . && why="$why port $PORT in use;"
    [ -z "$(other_colibri_units)" ] || why="$why other colibri unit: $(other_colibri_units | tr '\n' ' ');"
    for dir in "$BASE" "$PR2" "$S1T" "$S2T" "$S3T"; do
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
capture_meta "$BASE" "$PR2" "$S1T" "$S2T" "$S3T"
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
    python3 "$FINAL/final34_tools.py" check "$arm" "$OUT/$arm/s$k" 2>&1 | tee -a "$OUT/run.log"
    [ "${PIPESTATUS[0]}" != 0 ] && log "$arm s$k broke a premise (recorded; the run goes on)"
done
}
run_order "$ORDER1"
run_order "$ORDER"
run_order "$ORDER3"


# ---- 6. no CUDA tests here (each split step's builds and tests: counter, before)
finish
