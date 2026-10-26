#!/bin/bash
# run_final28.sh (2026-10-26) -- final28: the same-base run RESTARTED on dev c1c06ac6 (#2401-#2404 merged upstream;
#   final25/26 stopped). Trees cw22/f27-*: B = c1c06ac6, P = samebase/pr2 11c40038 (PR #2 rebased, same 11 patches),
#   M = samebase/m1766 ecc92586, C+ = samebase/cplus b659ac8d. Built by final27/build_f27.sh. Below: final26's header.
# run_final26.sh (2026-10-26) -- final26: Stage 2 of colibri-notes review/PLAN_SAME_BASE_20261026.md, the same-base C+ run.
#   C+ = PR #2 5b3f4601 merged with samebase M (#1766 on dd03a9de): fork samebase/cplus, built into cw22/f26-cplus (CUDA)
#   and cw22/f26-cplusvk (CUDA=1 VK=1) by build_f26_c.sh. Every tree on dd03a9de, so one text group per model.
#   REAP-150B, bench_decode.py: C1 (C+ one GPU, VRAM-first) x2, C2 (C+ + PR #2's peer on GPU 1) x2, same-session anchors
#   B1, P2, M2, M1 x1 (final25's trees f22-base, f22-pr2, f25-m1766), interleaved. Then the ablation of #1766's switches
#   on M, one GPU, x1 each (MX all off; AC AH AP AI AR AW one off). Full V4 Flash (bench_pr1.py): C1F, C2F x2, anchors
#   B1F, P2F x1. Pure Vulkan VKC x1. Then dsv4-cuda-test and cuda-test on C+, #1766's test_dsv4_multigpu_cuda 0,1 and
#   #1766's CUDA unit tests on C+.
# Starts only after final25 (its lock is taken). MAX_GPU_S from the environment. final25's runner, new arms and trees.
EXP=${EXP_NAME:-bench}
[ "${PILOT:-0}" = 1 ] && EXP=pilot
W22=/home/maplemercury/colibri-bench/cw22
BASE=$W22/f27-base; PR2=$W22/f27-pr2; M1766=$W22/f27-m1766
CPLUS=$W22/f27-cplus; CPLUSVK=$W22/f27-cplusvk
NVF="-O3 -std=c++17 -ftz=false -arch=sm_86 -Xcompiler=-Wall,-Wextra -DCOLI_DSV4_NO_TC"
MAX_GPU_S=${MAX_GPU_S:?set MAX_GPU_S (s)}   # cap on the llama-down window (<= 4 h; ends before 05:44)
NVSMI_MS=5000
source /home/maplemercury/colibri-bench/final28/final_common.sh

REPO=$W22/repo
DEV_SHA=c1c06ac6
PR2_SHA=11c40038
BASE_SHA=c1c06ac6
M1766_SHA=${M1766_SHA:-ecc92586}   # samebase/m1766 (#1766 + dev c1c06ac6), built into f27-m1766
CPLUS_SHA=${CPLUS_SHA:-b659ac8d}   # samebase/cplus (PR #2 + #1766 on dev c1c06ac6), built into f27-cplus
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
ABL_ALL="DSV4_CUDA_COMPRESSOR=0 DSV4_CUDA_HEAD=0 DSV4_CUDA_HEAD_BATCH=0 DSV4_CUDA_INDEXER_PREP=0 DSV4_CUDA_INDEXER_HEADS=0 DSV4_CUDA_RESIDENT_INDEXER=0 DSV4_CUDA_WO_DECODE=0 DSV4_CUDA_RESIDENT_EXPERTS=0"
arm C1  "$CPLUS"   "$REAP" 8192 decode   1800 600  $BANKE V4_GPU_RESIDENT_FIRST=1
arm C2  "$CPLUS"   "$REAP" 8192 decode   1800 460  $BANKE $PEER
arm B1  "$BASE"    "$REAP" 8192 decode   1800 640  $BANKE
arm P2  "$PR2"     "$REAP" 8192 decode   1800 460  $BANKE $PEER
arm M2  "$M1766"   "$REAP" 8192 decode   1800 480  $BANKE DSV4_CUDA_DEVICES=0,1
arm M1  "$M1766"   "$REAP" 8192 decode   1800 600  $BANKE
arm MX  "$M1766"   "$REAP" 8192 decode   1800 640  $BANKE $ABL_ALL
arm AC  "$M1766"   "$REAP" 8192 decode   1800 620  $BANKE DSV4_CUDA_COMPRESSOR=0
arm AH  "$M1766"   "$REAP" 8192 decode   1800 620  $BANKE DSV4_CUDA_HEAD=0
arm AP  "$M1766"   "$REAP" 8192 decode   1800 620  $BANKE DSV4_CUDA_INDEXER_PREP=0
arm AI  "$M1766"   "$REAP" 8192 decode   1800 620  $BANKE DSV4_CUDA_INDEXER_HEADS=0
arm AR  "$M1766"   "$REAP" 8192 decode   1800 620  $BANKE DSV4_CUDA_RESIDENT_INDEXER=0
arm AW  "$M1766"   "$REAP" 8192 decode   1800 620  $BANKE DSV4_CUDA_WO_DECODE=0
arm C1F "$CPLUS"   "$FULL" 8192 pr1:both 3600 330  $BANKE V4_GPU_RESIDENT_FIRST=1
arm C2F "$CPLUS"   "$FULL" 8192 pr1:both 3600 300  $BANKE $PEER
arm B1F "$BASE"    "$FULL" 8192 pr1:both 3600 260  $BANKE
arm P2F "$PR2"     "$FULL" 8192 pr1:both 3600 300  $BANKE $PEER
arm VKC "$CPLUSVK" "$REAP" 8192 pr1:both 3600 320  $VKO
ORDER1=${ORDER1_OVERRIDE:-"C1:1 C2:1 B1:1 P2:1 M2:1 M1:1 C2:2 C1:2 MX:1 AC:1 AH:1 AP:1 AI:1 AR:1 AW:1"}
ORDER=${ORDER2_OVERRIDE:-"C1F:1 C2F:1 B1F:1 P2F:1 C2F:2 C1F:2"}
ORDER3=${ORDER3_OVERRIDE:-"VKC:1"}

lock_or_exit
trap 'log "signal before the llama stop: exiting, llama untouched"; [ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null; exit 143' TERM INT HUP
log "run_final28.sh start (pid $$, unit $(sed 's#.*/##' /proc/self/cgroup), PILOT=${PILOT:-0})"

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
for t in "$BASE:$BASE_SHA" "$PR2:$PR2_SHA" "$M1766:$M1766_SHA" "$CPLUS:$CPLUS_SHA" "$CPLUSVK:$CPLUS_SHA"; do
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
preflight_switch "$M1766" DSV4_CUDA_DEVICES DSV4_CUDA_RESIDENT_EXPERTS DSV4_CUDA_COMPRESSOR DSV4_CUDA_HEAD DSV4_CUDA_HEAD_BATCH DSV4_CUDA_INDEXER_PREP DSV4_CUDA_INDEXER_HEADS DSV4_CUDA_RESIDENT_INDEXER DSV4_CUDA_WO_DECODE
preflight_switch "$CPLUS" V4_GPU_RESIDENT_FIRST DSV4_CUDA_PEER_DEVICE DSV4_CUDA_PEER_FAULT DSV4_CUDA_DEVICES DSV4_CUDA_RESIDENT_EXPERTS
# final25 must be over (its lock is taken by lock_or_exit; this names it if its unit is somehow still up)
systemctl --user is-active -q colibri-final25 && die_pre "colibri-final25 is still running"
[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2 ] || die_pre "second GPU not visible"
[ -z "$(foreign_gpu_pids)" ] || die_pre "other GPU compute processes: $(foreign_gpu_pids | tr '\n' ' ')"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/meta/gpu_apps_preflight.txt" 2>&1
[ "$(df -Pk "$FINAL" | awk 'NR==2{print $4}')" -gt 2000000 ] || die_pre "less than 2 GB free in $FINAL"
total_est=0
for item in $ORDER1 $ORDER $ORDER3; do total_est=$((total_est + ${EST[${item%%:*}]})); done
log "plan: $ORDER1 | $ORDER | $ORDER3 | CUDA tests (estimated $total_est s of samples + ~2700 s of tests; cap $MAX_GPU_S s)"
[ $((total_est + 2700 + 240)) -le "$MAX_GPU_S" ] || die_pre "plan estimate exceeds the cap"
log "preflight ok"

# ---- 4. wait for STOP_AT (re-check 20 s before)
recheck() {
    local why= dir
    systemctl --user is-active -q "$LLAMA_UNIT" && llama_healthy || why="$why llama not healthy;"
    ss -ltnH "sport = :$PORT" | grep -q . && why="$why port $PORT in use;"
    [ -z "$(other_colibri_units)" ] || why="$why other colibri unit: $(other_colibri_units | tr '\n' ' ');"
    for dir in "$BASE" "$PR2" "$M1766" "$CPLUS" "$CPLUSVK"; do
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
capture_meta "$BASE" "$PR2" "$M1766" "$CPLUS" "$CPLUSVK"
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
    python3 "$FINAL/final28_tools.py" check "$arm" "$OUT/$arm/s$k" 2>&1 | tee -a "$OUT/run.log"
    [ "${PIPESTATUS[0]}" != 0 ] && log "$arm s$k broke a premise (recorded; the run goes on)"
done
}
run_order "$ORDER1"
run_order "$ORDER"
run_order "$ORDER3"


# ---- 6. the CUDA tests (last: they rebuild objects in the trees; final22 showed deepseek_v4 itself is untouched)
cuda_tests() { # tree name
    local t=$1 n=$2
    if over_budget 900; then log "budget: skipping the CUDA tests of $n"; return; fi
    ( cd "$t/c" && CUDA_VISIBLE_DEVICES=0,1 timeout 900 make dsv4-cuda-test CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NVCCFLAGS="$NVF" ) > "$OUT/meta/dsv4-cuda-test-$n.log" 2>&1
    log "dsv4-cuda-test $n: rc=$? ($(grep -ciE 'warning' "$OUT/meta/dsv4-cuda-test-$n.log") warning lines; $(grep -E 'dsv4 peer MoE:|sticky:' "$OUT/meta/dsv4-cuda-test-$n.log" | tr '\n' ' '))"
    if over_budget 900; then log "budget: skipping cuda-test of $n"; return; fi
    ( cd "$t" && CUDA_VISIBLE_DEVICES=0,1 timeout 900 make -C c cuda-test CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NVCCFLAGS="$NVF" ) > "$OUT/meta/cuda-test-$n.log" 2>&1
    log "cuda-test $n: rc=$?"
}
cuda_tests "$CPLUS" cplus
# #1766's own CUDA tests, on C+ (they carry over in the merge): the multi-GPU lifecycle test, then every other
# tests/test_dsv4_*_cuda / resident target Makefile.deepseek-v4 defines, each built and run on its own (a failed
# build is recorded, not fatal).
if ! over_budget 900; then
    ( cd "$CPLUS/c" && timeout 900 make -f Makefile.deepseek-v4 tests/test_dsv4_multigpu_cuda CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 &&
      CUDA_VISIBLE_DEVICES=0,1 timeout 600 ./tests/test_dsv4_multigpu_cuda 0,1 ) > "$OUT/meta/cplus-multigpu-test.log" 2>&1
    log "C+ test_dsv4_multigpu_cuda 0,1: rc=$? ($(tail -1 "$OUT/meta/cplus-multigpu-test.log"))"
fi
for tgt in $(grep -oE '^tests/test_dsv4_[a-z_]+(_cuda)?|^tests/test_dsv4_resident_[a-z_]+' "$CPLUS/c/Makefile.deepseek-v4" | sed 's/:.*//' | sort -u | grep -v multigpu); do
    over_budget 600 && { log "budget: skipping $tgt"; continue; }
    n=$(basename "$tgt")
    ( cd "$CPLUS/c" && timeout 600 make -f Makefile.deepseek-v4 "$tgt" CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 &&
      CUDA_VISIBLE_DEVICES=0,1 timeout 300 "./$tgt" ) > "$OUT/meta/cplus-$n.log" 2>&1
    log "C+ $n: rc=$? ($(tail -1 "$OUT/meta/cplus-$n.log" | cut -c1-160))"
done

finish
