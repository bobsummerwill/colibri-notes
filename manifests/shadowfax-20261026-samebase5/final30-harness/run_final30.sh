#!/bin/bash
# run_final30.sh (2026-10-26) -- final30: the two open findings of colibri-notes review/PLAN_SAME_BASE_20261026.md
#   ("Investigations added (Bob, ~14:20)"), after final29. Trees cw22/f27-* (B = dev c1c06ac6, P = samebase/pr2 11c40038,
#   M = samebase/m1766 ecc92586, C+ = samebase/cplus b659ac8d); no Vulkan trees, no CUDA tests. All on the full V4 Flash,
#   one sample per arm.
#   F1 (#1766 13-14% slower to first token than dev on long prompts), bench_pr1.py both prompts, one GPU:
#     A:  anchors AB1F (B), AM1F (M); M with one #1766 switch off each (FAC compressor, FAP indexer prepare, FAI indexer
#         heads, FAR resident indexer, FAH head, FAHB head batch, FAW WO decode) and FMX (final28's ABL_ALL, all off).
#     A': PB1F, PM1F = AB1F, AM1F with DSV4_ATTN_PROF=1 V4_MOE_BATCH_PROF=1 (per-phase prefill split; timing not used).
#   F2 (PR #2's peer helps only the second full-model prompt):
#     B:  client fdec (bench_decode.py's first 3 prompts, 300 tokens, greedy): DB1F, DP1F, DP2F, DM1F, DM2F, DC1F, DC2F.
#     B': client pr1:swap (bench_pr1.py's two requests in reverse order): SP1F (P VRAM-first), SP2F (P + peer).
#   final28's runner (STOP_AT, MAX_GPU_S, preflight, llama stop/restart, check after each sample, summary, packing).
#   DRY_RUN=1: clock, trees, switches, selftest and plan checked, the box-busy checks only recorded; exits before the wait.
# Below: final28's header.
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
[ "${DRY_RUN:-0}" = 1 ] && EXP=dryrun
W22=/home/maplemercury/colibri-bench/cw22
BASE=$W22/f27-base; PR2=$W22/f27-pr2; M1766=$W22/f27-m1766
CPLUS=$W22/f27-cplus
MAX_GPU_S=${MAX_GPU_S:?set MAX_GPU_S (s)}   # cap on the llama-down window (<= 4 h; ends before 05:44)
NVSMI_MS=5000
BUILDS=${BUILDS:-/home/maplemercury/colibri-bench/final27/builds}   # final30: the f27-* trees' build logs (final27/build_f27.sh); no builds here
source /home/maplemercury/colibri-bench/final30/final_common.sh

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
MODEL=$FULL   # final30: every arm is on the full model
declare -A TREE CLIENT TMO XENV EST FAILED MDL CTX
arm() { # NAME TREE MODEL CTX CLIENT TIMEOUT EST [KEY=VAL...]
    local n=$1; TREE[$n]=$2; MDL[$n]=$3; CTX[$n]=$4; CLIENT[$n]=$5; TMO[$n]=$6; EST[$n]=$7; shift 7; XENV[$n]="$*"
}
BANKE="COLI_CUDA_MOE_BATCH=1"
PEER="V4_SERVE_STATS_LOG=1 V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1"
ABL_ALL="DSV4_CUDA_COMPRESSOR=0 DSV4_CUDA_HEAD=0 DSV4_CUDA_HEAD_BATCH=0 DSV4_CUDA_INDEXER_PREP=0 DSV4_CUDA_INDEXER_HEADS=0 DSV4_CUDA_RESIDENT_INDEXER=0 DSV4_CUDA_WO_DECODE=0 DSV4_CUDA_RESIDENT_EXPERTS=0"
PROF="DSV4_ATTN_PROF=1 V4_MOE_BATCH_PROF=1"   # both read by c/deepseek_v4.c (getenv) in B and M (preflight_switch below)
# EST (s, sample wall incl. load): final27 full pr1:both B1F 241, M1F 267, P 240, M2F 265; fdec ~3 x 300 tokens at ~2 tok/s
# + 3 short prefills + load; ablation / profiled arms padded (a switch off may be slower).
# Part A (F1): full, pr1:both, one GPU
arm AB1F "$BASE"  "$FULL" 8192 pr1:both 3600 270  $BANKE
arm AM1F "$M1766" "$FULL" 8192 pr1:both 3600 290  $BANKE
arm FAC  "$M1766" "$FULL" 8192 pr1:both 3600 300  $BANKE DSV4_CUDA_COMPRESSOR=0
arm FAP  "$M1766" "$FULL" 8192 pr1:both 3600 300  $BANKE DSV4_CUDA_INDEXER_PREP=0
arm FAI  "$M1766" "$FULL" 8192 pr1:both 3600 300  $BANKE DSV4_CUDA_INDEXER_HEADS=0
arm FAR  "$M1766" "$FULL" 8192 pr1:both 3600 300  $BANKE DSV4_CUDA_RESIDENT_INDEXER=0
arm FAH  "$M1766" "$FULL" 8192 pr1:both 3600 300  $BANKE DSV4_CUDA_HEAD=0
arm FAHB "$M1766" "$FULL" 8192 pr1:both 3600 300  $BANKE DSV4_CUDA_HEAD_BATCH=0
arm FAW  "$M1766" "$FULL" 8192 pr1:both 3600 300  $BANKE DSV4_CUDA_WO_DECODE=0
arm FMX  "$M1766" "$FULL" 8192 pr1:both 3600 320  $BANKE $ABL_ALL
# Part A' (F1, profiled; timing not used)
arm PB1F "$BASE"  "$FULL" 8192 pr1:both 3600 320  $BANKE $PROF
arm PM1F "$M1766" "$FULL" 8192 pr1:both 3600 320  $BANKE $PROF
# Part B (F2): full, fdec (3 x 300 tokens)
arm DB1F "$BASE"  "$FULL" 8192 fdec     2400 660  $BANKE
arm DP1F "$PR2"   "$FULL" 8192 fdec     2400 660  $BANKE V4_GPU_RESIDENT_FIRST=1
arm DP2F "$PR2"   "$FULL" 8192 fdec     2400 600  $BANKE $PEER
arm DM1F "$M1766" "$FULL" 8192 fdec     2400 660  $BANKE
arm DM2F "$M1766" "$FULL" 8192 fdec     2400 600  $BANKE DSV4_CUDA_DEVICES=0,1
arm DC1F "$CPLUS" "$FULL" 8192 fdec     2400 660  $BANKE V4_GPU_RESIDENT_FIRST=1
arm DC2F "$CPLUS" "$FULL" 8192 fdec     2400 600  $BANKE $PEER
# Part B' (F2): full, bench_pr1.py's two requests in reverse order
arm SP1F "$PR2"   "$FULL" 8192 pr1:swap 3600 270  $BANKE V4_GPU_RESIDENT_FIRST=1
arm SP2F "$PR2"   "$FULL" 8192 pr1:swap 3600 270  $BANKE $PEER
ORDER1=${ORDER1_OVERRIDE:-"AB1F:1 AM1F:1 FAC:1 FAP:1 FAI:1 FAR:1 FAH:1 FAHB:1 FAW:1 FMX:1 PB1F:1 PM1F:1"}
ORDER=${ORDER2_OVERRIDE:-"DB1F:1 DP1F:1 DP2F:1 DM1F:1 DM2F:1 DC1F:1 DC2F:1"}
ORDER3=${ORDER3_OVERRIDE:-"SP1F:1 SP2F:1"}

lock_or_exit
trap 'log "signal before the llama stop: exiting, llama untouched"; [ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null; exit 143' TERM INT HUP
log "run_final30.sh start (pid $$, unit $(sed 's#.*/##' /proc/self/cgroup), PILOT=${PILOT:-0})"

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
for t in "$BASE:$BASE_SHA" "$PR2:$PR2_SHA" "$M1766:$M1766_SHA" "$CPLUS:$CPLUS_SHA"; do
    dir=${t%%:*}; full=$(git -C "$REPO" rev-parse "${t#*:}^{commit}") || die_pre "${t#*:} not in $REPO"
    preflight_tree "$dir" "$full"
    BIN[$dir]=$(sha256sum "$dir/c/deepseek_v4" | cut -c1-64)
    echo "$(basename "$dir")=$full binary=${BIN[$dir]}" >> "$OUT/meta/tips.txt"
done
log "tips: $(tr '\n' ' ' < "$OUT/meta/tips.txt")"

# ---- 3. preflight (nothing touched)
preflight_common bench_pr1.py bench_fdec.py
grep -q '"swap"' "$CLIENTS/bench_pr1.py" || die_pre "$CLIENTS/bench_pr1.py has no BENCH_REQS=swap"
[ -f "$FULL/config.json" ] || die_pre "full model missing at $FULL"
preflight_switch "$PR2" V4_GPU_RESIDENT_FIRST DSV4_CUDA_PEER_DEVICE DSV4_CUDA_PEER_FAULT
preflight_switch "$BASE" DSV4_ATTN_PROF V4_MOE_BATCH_PROF
preflight_switch "$M1766" DSV4_ATTN_PROF V4_MOE_BATCH_PROF
preflight_switch "$M1766" DSV4_CUDA_DEVICES DSV4_CUDA_RESIDENT_EXPERTS DSV4_CUDA_COMPRESSOR DSV4_CUDA_HEAD DSV4_CUDA_HEAD_BATCH DSV4_CUDA_INDEXER_PREP DSV4_CUDA_INDEXER_HEADS DSV4_CUDA_RESIDENT_INDEXER DSV4_CUDA_WO_DECODE
preflight_switch "$CPLUS" V4_GPU_RESIDENT_FIRST DSV4_CUDA_PEER_DEVICE DSV4_CUDA_PEER_FAULT DSV4_CUDA_DEVICES DSV4_CUDA_RESIDENT_EXPERTS
# final28 and final29 must be over (their locks are taken by lock_or_exit; this names them if a unit is somehow still up)
for u in colibri-final28 colibri-final29 cbench-chain2; do systemctl --user is-active -q $u && soft_pre "$u is still running"; done
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
python3 "$FINAL/final30_tools.py" arms $ORDER1 $ORDER $ORDER3 >> "$OUT/run.log" 2>&1 || die_pre "an arm of the plan has no final30_tools.py ARMS entry"
# final30: each arm's environment (COMMON_ENV + its extra env, as sample() merges it) against its ARMS premises
for item in $ORDER1 $ORDER $ORDER3; do
    a=${item%%:*}
    # shellcheck disable=SC2046,SC2086
    python3 "$FINAL/final30_tools.py" envcheck "$a" $(merge_env "${COMMON_ENV[@]}" ${XENV[$a]}) >> "$OUT/run.log" 2>&1 || die_pre "arm $a's environment breaks its premises (see run.log)"
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
    for dir in "$BASE" "$PR2" "$M1766" "$CPLUS"; do
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
capture_meta "$BASE" "$PR2" "$M1766" "$CPLUS"
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
    python3 "$FINAL/final30_tools.py" check "$arm" "$OUT/$arm/s$k" 2>&1 | tee -a "$OUT/run.log"
    [ "${PIPESTATUS[0]}" != 0 ] && log "$arm s$k broke a premise (recorded; the run goes on)"
done
}
run_order "$ORDER1"
run_order "$ORDER"
run_order "$ORDER3"


# ---- 6. no CUDA tests here (final28 ran them on C+)
finish
