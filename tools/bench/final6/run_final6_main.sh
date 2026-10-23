#!/bin/bash
# run_final6_main.sh (2026-10-22; run_final6.sh plus the F18 probe at the end, a separate file so the
# running pilot's script is never edited under it) -- run_final5.sh adapted to JustVugg's request on #2264: the default
# configuration (no COLI_CUDA_MOE_BATCH) on the FULL DeepSeek V4 Flash, upstream dev against PR #1
# merged with that dev, prompt and decode, interleaved. DSV4_MM_WARP (auto) is on by default in
# PR #1's mm_batch, so this is the configuration every FP8 checkpoint on a generic CUDA build gets.
#   A = upstream dev 7df8c27c, B = docs5/pr1-reap-prefill merged with it + the unused-variable fix (b286167e),
#   PR2 = docs5/pr2-two-gpu-decode with both merged (0523e8e0).
# Trees and builds live under ~/colibri-bench/cw22 (persistent; /tmp vanishes on reboot).
# Arms (final6_tools.py's ARMS has each arm's premises; every sample is a fresh server):
#   1. the question, full model, default configuration: W warm-up, A dev, B PR #1, interleaved
#      W A B B A A B; then V = B + DSV4_MM_WARP_VERIFY=1 (bit-identity on this model, not timed)
#   2. the 256-expert bank (never run): AK dev / BK PR #1 with COLI_CUDA_MOE_BATCH=1, BD = BK +
#      COLI_CUDA_MOE_DOUBLE=1, 3 each interleaved
#   3. speculative decoding with the bank on (F2): AS dev / BS PR #1, V4_DRAFT=4 (n-gram drafts),
#      first prompt only; texts must equal the plain runs', one bank creation per request
#   4. CC: PR #1, default configuration, both requests at once (bench_f6.py parallel)
#   5. LA dev / LB PR #1: one ~8k-token prompt, default configuration, --ctx 16384
#   6. PR #2 after the dev merge, REAP-150B, final5's decode arms: WR warm-up, D0 dev, G1 / R1 / X3
#      PR #2 (switches off / VRAM-first / + second GPU), 3 each interleaved
#   7. PR #2 paths never run end to end: P0 (peer pool, DSV4_CUDA_PEER_COMPUTE=0, F45),
#      HY (DSV4_HYBRID=1), NB (DSV4_CUDA_BATCHED=0), one sample each
#   A premise failure in A/B/V aborts after the second one; elsewhere it only skips that arm.
# PILOT=1: P (B's config) and PK (BK's config), one sample each, to check memory and timing, then
#   dsv4-cuda-test at both merged tips and cuda-test at PR2, then llama comes back.
# Everything else (locks, builds, preflight, the llama stop / EXIT-trap restore, watchdog, evidence)
# is final5's; see final5/run_final5.sh and final_common.sh.

EXP=bench
[ "${PILOT:-0}" = 1 ] && EXP=pilot
W22=/home/maplemercury/colibri-bench/cw22
PR1=$W22/f6-pr1
PR2=$W22/f6-pr2
DEV=$W22/f6-dev
if [ "${PILOT:-0}" = 1 ]; then
    MAX_GPU_S=$((120 * 60))     # pilot: two samples + the CUDA tests
else
    MAX_GPU_S=$((1200 * 60))    # final6: hard cap on the llama-down window (20 h; Bob: Hermes may stay down until the tests finish)
fi
NVSMI_MS=5000
source /home/maplemercury/colibri-bench/final6/final_common.sh

REPO=$W22/repo
DEV_SHA=7df8c27c95acf7019dd60714ec1c42fa7e36b730
PR1_SHA_WANT=b286167e
PR2_SHA_WANT=0523e8e0
CLIENT_SHA_PR1=eca4360bdd5a0601c0a7138f49464874a648486519235bbc22d40997c3e2de53     # final/clients/bench_pr1.py
TZ_NAME=America/Vancouver
STOP_AT="${STOP_AT:?set STOP_AT=\x27YYYY-MM-DD HH:MM:SS\x27}"
STOP_EPOCH_WANT=$(date -d "$STOP_AT" +%s)
LATEST_STOP_S=300
MAX_EARLY_S=5400            # builds run first (about 25 min for three trees)
MODE_RE='moe-batch=|moe-double|mirror-reserve|peer-mirrors|peer mirror pool|peer device|DSV4 CUDA\] device|mm-warp verify|^v4_serve_turn|^v4_dec_moe|out of memory|[Vv]ulkan|\[VK|vk_|MTP|mtp|dspark|DSpark|draft'
export TZ=$TZ_NAME

FULL=/home/maplemercury/colibri-models/deepseek-v4-flash
REAP=/home/maplemercury/colibri-models/deepseek-v4-flash-reap
declare -A TREE CLIENT TMO XENV EST FAILED MDL CTX
arm() { # NAME TREE MODEL CTX CLIENT TIMEOUT EST [KEY=VAL...]
    local n=$1; TREE[$n]=$2; MDL[$n]=$3; CTX[$n]=$4; CLIENT[$n]=$5; TMO[$n]=$6; EST[$n]=$7; shift 7; XENV[$n]="$*"
}
BANKE="COLI_CUDA_MOE_BATCH=1"
arm W  "$PR1" "$FULL" 8192  pr1:both     3600 1300
arm P  "$PR1" "$FULL" 8192  pr1:both     3600 1300
arm PK "$PR1" "$FULL" 8192  pr1:both     3600 800  $BANKE
arm A  "$DEV" "$FULL" 8192  pr1:both     3600 1300
arm B  "$PR1" "$FULL" 8192  pr1:both     3600 1300
arm V  "$PR1" "$FULL" 8192  pr1:both     3600 1500 DSV4_MM_WARP_VERIFY=1
arm AK "$DEV" "$FULL" 8192  pr1:both     3600 800  $BANKE
arm BK "$PR1" "$FULL" 8192  pr1:both     3600 800  $BANKE
arm BD "$PR1" "$FULL" 8192  pr1:both     3600 800  $BANKE COLI_CUDA_MOE_DOUBLE=1
arm AS "$DEV" "$FULL" 8192  pr1:first    3600 1200 $BANKE V4_DRAFT=4
arm BS "$PR1" "$FULL" 8192  pr1:first    3600 1200 $BANKE V4_DRAFT=4
arm CC "$PR1" "$FULL" 8192  f6:parallel  3600 1300
arm LA "$DEV" "$FULL" 16384 f6:long      5400 1800
arm LB "$PR1" "$FULL" 16384 f6:long      5400 1800
R2="$BANKE V4_SERVE_STATS_LOG=1"
arm WR "$PR2" "$REAP" 8192  decode       1500 700  $R2
arm D0 "$DEV" "$REAP" 8192  decode       1500 720  $BANKE
arm G1 "$PR2" "$REAP" 8192  decode       1500 700  $R2
arm R1 "$PR2" "$REAP" 8192  decode       1500 650  $R2 V4_GPU_RESIDENT_FIRST=1
arm X3 "$PR2" "$REAP" 8192  decode       1500 520  $R2 V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1
arm P0 "$PR2" "$REAP" 8192  decode       1500 700  $R2 V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1 DSV4_CUDA_PEER_COMPUTE=0
arm HY "$PR2" "$REAP" 8192  decode       1500 700  $R2 V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1 DSV4_HYBRID=1
arm NB "$PR2" "$REAP" 8192  decode       1500 700  $R2 V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1 DSV4_CUDA_BATCHED=0
DEVV=$W22/f6v-dev; PR1V=$W22/f6v-pr1   # CUDA=1 VK=1 builds (build_vk.sh, LunarG glslc), outside build_tree
arm RB "$PR1"  "$REAP" 8192  pr1:both     1800 650  $BANKE
arm VA "$DEVV" "$REAP" 8192  pr1:both     2400 900  $BANKE COLI_VULKAN=1
arm VB "$PR1V" "$REAP" 8192  pr1:both     2400 650  $BANKE COLI_VULKAN=1
arm QA "$DEVV" "$REAP" 8192  pr1:both     3600 900  COLI_VULKAN=1 DSV4_CUDA=0 COLI_CUDA=0
arm QB "$PR1V" "$REAP" 8192  pr1:both     3600 900  COLI_VULKAN=1 DSV4_CUDA=0 COLI_CUDA=0
arm KA "$DEV"  "$FULL" 32768 f6:long16    5400 1500 $BANKE
arm KB "$PR1"  "$FULL" 32768 f6:long16    5400 1500 $BANKE
arm MA "$DEV"  "$FULL" 8192  f6:short     3600 1200 $BANKE V4_MTP=1 V4_DRAFT=3
arm MB "$PR1"  "$FULL" 8192  f6:short     3600 1200 $BANKE V4_MTP=1 V4_DRAFT=3
arm SK "$PR2"  "$REAP" 8192  f6:soak      5400 2700 $BANKE V4_SERVE_STATS_LOG=1 V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1
CORE=" A B V "
if [ "${PILOT:-0}" = 1 ]; then
    ORDER="P:1 PK:1"
else
    ORDER="W:1 A:1 B:1 B:2 A:2 A:3 B:3 V:1"
    ORDER="$ORDER AK:1 BK:1 BD:1 BK:2 BD:2 AK:2 BD:3 AK:3 BK:3"
    ORDER="$ORDER AS:1 BS:1 CC:1 LA:1 LB:1"
    ORDER="$ORDER WR:1 D0:1 G1:1 R1:1 X3:1 G1:2 R1:2 X3:2 D0:2 R1:3 X3:3 D0:3 G1:3"
    ORDER="$ORDER P0:1 HY:1 NB:1"
    ORDER="$ORDER RB:1 VA:1 VB:1 QA:1 QB:1 SK:1 KA:1 KB:1 MA:1 MB:1"
fi

lock_or_exit
trap 'log "signal before the llama stop: exiting, llama untouched"; [ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null; exit 143' TERM INT HUP
log "run_final6.sh start (pid $$, unit $(sed 's#.*/##' /proc/self/cgroup), PILOT=${PILOT:-0}, model $MODEL)"

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

# ---- 2. CUDA builds (CPU only)
build_tree "$DEV" "$DEV_SHA"
DEV_BIN_SHA=$(sha256sum "$DEV/c/deepseek_v4" | cut -c1-64)
build_tree "$PR1" "$PR1_SHA"
PR1_BIN=$(sha256sum "$PR1/c/deepseek_v4" | cut -c1-64)
build_tree "$PR2" "$PR2_SHA"
PR2_BIN=$(sha256sum "$PR2/c/deepseek_v4" | cut -c1-64)

for t in "$DEVV" "$PR1V"; do
    if [ -x "$t/c/deepseek_v4" ] && grep -q '^EXIT=0' "$W22/build-vk-${t##*-}.log" 2>/dev/null; then
        log "VK build $t: $(git -C "$t" rev-parse --short HEAD) binary $(sha256sum "$t/c/deepseek_v4" | cut -c1-16)"
    else
        log "WARNING: no clean VK build in $t: its arms are skipped"; [ "$t" = "$DEVV" ] && { FAILED[VA]=1; FAILED[QA]=1; } || { FAILED[VB]=1; FAILED[QB]=1; }
    fi
done
[ "$(git -C "$DEVV" rev-parse HEAD 2>/dev/null)" = "$DEV_SHA" ] || { FAILED[VA]=1; FAILED[QA]=1; }
[ "$(git -C "$PR1V" rev-parse HEAD 2>/dev/null)" = "$PR1_SHA" ] || { FAILED[VB]=1; FAILED[QB]=1; }

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
grep -q 'getenv("DSV4_MM_WARP")' "$PR1/c/backend_cuda_dsv4.cu" || die_pre "$PR1 does not read DSV4_MM_WARP"
grep -q 'DSV4_MM_WARP' "$DEV/c/backend_cuda_dsv4.cu" && die_pre "$DEV already has DSV4_MM_WARP"
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
capture_meta "$DEV" "$PR1" "$PR2"
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
    python3 "$FINAL/final6_tools.py" check "$arm" "$OUT/$arm/s$k" 2>&1 | tee -a "$OUT/run.log"
    if [ "${PIPESTATUS[0]}" != 0 ]; then
        FAILED[$arm]=1
        case "$CORE" in *" $arm "*) NFAIL=$((NFAIL + 1)) ;; esac
        if [ "$NFAIL" -ge 2 ]; then log "ABORT after $arm s$k: a second core sample broke a premise"; exit 8; fi
        log "$arm s$k broke a premise: the rest of $arm is skipped"
    fi
done

# ---- 6. CUDA tests at the tips while llama is still down (results, not aborts; the main run repeats the
# pilot's at b286167e / 0523e8e0, whose generic code is identical to ebe82123 / 073f18d2: same SASS)
if true; then
    NVF="-O3 -std=c++17 -ftz=false -arch=sm_86 -Xcompiler=-Wall,-Wextra -DCOLI_DSV4_NO_TC"
    for t in "$PR1" "$PR2"; do
        n=$(basename "$t")
        if over_budget 900; then log "budget: skipping dsv4-cuda-test at $n (GPU window $(gpu_elapsed) s)"; continue; fi
        ( cd "$t/c" && CUDA_VISIBLE_DEVICES=0,1 timeout 900 make dsv4-cuda-test CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NVCCFLAGS="$NVF" ) > "$OUT/meta/dsv4-cuda-test-$n.log" 2>&1
        rc=$?
        log "dsv4-cuda-test $n: rc=$rc ($(grep -ciE 'warning' "$OUT/meta/dsv4-cuda-test-$n.log") warning lines)"
    done
    if over_budget 900; then log "budget: skipping cuda-test"; else
        ( cd "$PR2" && CUDA_VISIBLE_DEVICES=0,1 timeout 900 make -C c cuda-test CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NVCCFLAGS="$NVF" ) > "$OUT/meta/cuda-test-pr2.log" 2>&1
        log "cuda-test pr2: rc=$?"
    fi
fi
# ---- 7. F18 probe (debug build ~/colibri-bench/cw22/f18, -DCOLI_F18_PROBE): a real __trap() on the
# peer GPU inside the decode MoE, before and after its down-projection; each scenario runs in a child
# with a 60 s alarm, so a hang is a result. Last, because a faulted context could disturb what follows.
if [ "${PILOT:-0}" != 1 ]; then
    F18=$W22/f18
    if over_budget 600; then log "budget: skipping the F18 probe"; elif [ ! -x "$F18/c/f18_trap_cuda_test" ]; then log "F18 probe binary missing"; else
        ( cd "$F18/c" && CUDA_VISIBLE_DEVICES=0,1 timeout 300 ./f18_trap_cuda_test ) > "$OUT/meta/f18-probe.log" 2>&1
        log "F18 probe: rc=$? $(grep -E 'RESULT|AFTER|HANG|f18 probe:' "$OUT/meta/f18-probe.log" | tr '\n' ' ')"
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$OUT/meta/f18-probe.log" 2>&1
    fi
fi
finish
