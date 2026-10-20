#!/bin/bash
# run_final5.sh (2026-10-19) -- run_final4.sh adapted to the stack on upstream dev 736a0682 (PR #0 merged):
#   A/D0 = upstream dev 736a0682, B/C = refresh/pr1-reap-prefill, G1/R1/X3 = refresh/pr2-two-gpu-decode,
#   all from /tmp/colibri, built into /tmp/cw5/r15-*. STOP_AT is given at launch.
# Arms (every one 3 samples except the warm-up; order rotated per round):
#   W   warm-up, B's config, discarded                                           pr1  prefill client
#   A   dev (no GPU batch for 132 experts: CPU union)                            dev  prefill client
#   B   pr1 tip (COLI_CUDA_MOE_BATCH=1 takes the 132-expert checkpoint)           pr1  prefill client
#   C   B + COLI_CUDA_MOE_DOUBLE=1                                               pr1  prefill client
#   D0  dev on the decode client (no PR2 code)                                   dev  decode client
#   G1  pr2 tip, V4_SERVE_STATS_LOG=1 only (every PR2 switch off)                 pr2  decode client
#   R1  G1 + V4_GPU_RESIDENT_FIRST=1 (one GPU)                                   pr2  decode client
#   X3  R1 + DSV4_CUDA_PEER_DEVICE=1 (peer compute on GPU 1)                     pr2  decode client
# One changed variable per comparison: A->B (the PR1 engine), B->C (double bank), G1->R1 (VRAM-first),
# R1->X3 (second GPU); D0->G1 checks that PR2 with its switches off behaves like dev.
# Everything else (locks, builds, preflight, the llama stop/EXIT-trap restore, watchdog, evidence,
# CUDA tests at the end) is final4's; see final4/run_final4.sh for the full description.
# final5: the GPU-exclusivity check accepts every process in llama-server-active's cgroup (Tabby).

EXP=bench
PR1=/tmp/cw5/r15-pr1
PR2=/tmp/cw5/r15-pr2
if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
    EXP=bench-preflight
    BUILDS=/home/maplemercury/colibri-bench/final5/builds-dry
    PR1=/tmp/cw5/f5dry-pr1
    PR2=/tmp/cw5/f5dry-pr2
    DEV_DRY=/tmp/cw5/f5dry-dev
fi
MAX_GPU_S=$((300 * 60))     # final5: hard cap on the llama-down window (samples + CUDA tests)
NVSMI_MS=5000
source /home/maplemercury/colibri-bench/final5/final_common.sh

REPO=/tmp/colibri
PR1_BRANCH=refresh/pr1-reap-prefill
PR2_BRANCH=refresh/pr2-two-gpu-decode
EXPECT_PR1=${EXPECT_PR1:-unset}    # the tips when this script was written (information only)
EXPECT_PR2=${EXPECT_PR2:-unset}
DEV_SHA=736a0682d38f0a1c6592f34b1c6749fea221da42   # final5: upstream dev after PR #0 merged
DEV=/tmp/cw5/r15-dev
[ "${PREFLIGHT_ONLY:-0}" = 1 ] && DEV=$DEV_DRY
DEV_BIN_SHA=   # final4: set after the dev build below
CLIENT_SHA_PR1=eca4360bdd5a0601c0a7138f49464874a648486519235bbc22d40997c3e2de53     # final/clients/bench_pr1.py
CLIENT_SHA_DECODE=06b2475ac64144ca512bee9d2fc85211227276f808b903e001227ac1ed78b644  # final/clients/bench_decode.py
TZ_NAME=America/Vancouver
STOP_AT="${STOP_AT:?set STOP_AT=\x27YYYY-MM-DD HH:MM:SS\x27}"
STOP_EPOCH_WANT=$(date -d "$STOP_AT" +%s)   # final4: given at launch
LATEST_STOP_S=300           # no llama stop after 01:05:00
MAX_EARLY_S=1800            # refuse to start more than 30 min before 01:00:00
MODE_RE='moe-batch=|moe-double|mirror-reserve|peer-mirrors|peer mirror pool|peer device|DSV4 CUDA\] device|^v4_serve_turn|^v4_dec_moe|out of memory'
export TZ=$TZ_NAME

declare -A TREE CLIENT TMO XENV EST FAILED
TREE[W]=$PR1;  CLIENT[W]=pr1:both;  TMO[W]=900;   XENV[W]="";                                                        EST[W]=270
TREE[A]=$DEV;  CLIENT[A]=pr1:both;  TMO[A]=1800;  XENV[A]="";                                                        EST[A]=640
TREE[B]=$PR1;  CLIENT[B]=pr1:both;  TMO[B]=900;   XENV[B]="";                                                        EST[B]=270
TREE[C]=$PR1;  CLIENT[C]=pr1:both;  TMO[C]=900;   XENV[C]="COLI_CUDA_MOE_DOUBLE=1";                                  EST[C]=245
TREE[G1]=$PR2; CLIENT[G1]=decode;   TMO[G1]=1500; XENV[G1]="V4_SERVE_STATS_LOG=1";                                   EST[G1]=700
TREE[X3]=$PR2; CLIENT[X3]=decode;   TMO[X3]=1500; XENV[X3]="V4_SERVE_STATS_LOG=1 V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1"; EST[X3]=520
TREE[R1]=$PR2; CLIENT[R1]=decode;   TMO[R1]=1500; XENV[R1]="V4_SERVE_STATS_LOG=1 V4_GPU_RESIDENT_FIRST=1";                         EST[R1]=650
TREE[D0]=$DEV; CLIENT[D0]=decode;   TMO[D0]=1500; XENV[D0]="";                                                        EST[D0]=720
ORDER="W:1 A:1 B:1 C:1 B:2 C:2 A:2 C:3 A:3 B:3 D0:1 G1:1 R1:1 X3:1 G1:2 R1:2 X3:2 D0:2 R1:3 X3:3 D0:3 G1:3"

lock_or_exit
# before the llama stop a signal just ends the script (nothing touched; the locks go with it)
trap 'log "signal before the llama stop: exiting, llama untouched"; [ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null; exit 143' TERM INT HUP
log "run_final4.sh start (pid $$, unit $(sed 's#.*/##' /proc/self/cgroup), PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}, trees $PR1 $PR2)"
if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
    unlock_all
    log "PREFLIGHT_ONLY: the three GPU locks were free; released again at once"
fi


# final4: no GPU compute process except llama-server-active's own (someone else's job on the GPUs)
foreign_gpu_pids() {   # final5: every process of llama-server-active's cgroup is llama's (Tabby's model runs in a child)
    local cg allowed
    cg=$(systemctl --user show -p ControlGroup --value llama-server-active 2>/dev/null)
    allowed=$(cat "/sys/fs/cgroup$cg/cgroup.procs" 2>/dev/null | tr '\n' ' ')
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u | grep -v '^$' \
        | while read -r p; do case " $allowed " in *" $p "*) ;; *) echo "$p" ;; esac; done
}

# ---- 0. the clock: shadowfax local time is America/Vancouver, 01:00:00 is the agreed stop
[ "$(timedatectl show -p Timezone --value 2>/dev/null)" = "$TZ_NAME" ] || die_pre "system time zone is not $TZ_NAME"
STOP_EPOCH=$(date -d "$STOP_AT" +%s)
[ "$STOP_EPOCH" = "$STOP_EPOCH_WANT" ] || die_pre "$STOP_AT resolves to $STOP_EPOCH, not $STOP_EPOCH_WANT"
lead=$(( STOP_EPOCH - $(now) ))
log "llama stop planned at $(date -d "@$STOP_EPOCH" '+%F %T %Z') ($lead s from now)"
if [ "${PREFLIGHT_ONLY:-0}" != 1 ]; then
    [ "$lead" -le "$MAX_EARLY_S" ] || die_pre "started $lead s before $STOP_AT (more than $MAX_EARLY_S s early)"
    [ "$lead" -ge "-$LATEST_STOP_S" ] || die_pre "started $(( -lead )) s after $STOP_AT: the window is missed"
fi

# ---- 1. the tips, as they are now
PR1_SHA=$(git -C "$REPO" rev-parse --verify -q "refs/heads/$PR1_BRANCH^{commit}") || die_pre "branch $PR1_BRANCH not in $REPO"
PR2_SHA=$(git -C "$REPO" rev-parse --verify -q "refs/heads/$PR2_BRANCH^{commit}") || die_pre "branch $PR2_BRANCH not in $REPO"
PR1_TREE=$(git -C "$REPO" rev-parse "$PR1_SHA^{tree}")
PR2_TREE=$(git -C "$REPO" rev-parse "$PR2_SHA^{tree}")
isanc() { git -C "$REPO" merge-base --is-ancestor "$1" "$2" && echo yes || echo NO; }
REF_T=
{
    echo "resolved_at=$(date -Is)"
    echo "repo=$REPO"
    echo "dev_sha=$DEV_SHA"
    echo "pr1_branch=$PR1_BRANCH"; echo "pr1_sha=$PR1_SHA"; echo "pr1_tree=$PR1_TREE"
    echo "pr1_subject=$(git -C "$REPO" log -1 --format=%s "$PR1_SHA")"; echo "pr1_committer_date=$(git -C "$REPO" log -1 --format=%cI "$PR1_SHA")"
    echo "pr1_is_expected_$EXPECT_PR1=$([ "$PR1_SHA" = "$EXPECT_PR1" ] && echo yes || echo NO)"
    echo "pr2_branch=$PR2_BRANCH"; echo "pr2_sha=$PR2_SHA"; echo "pr2_tree=$PR2_TREE"
    echo "pr2_subject=$(git -C "$REPO" log -1 --format=%s "$PR2_SHA")"; echo "pr2_committer_date=$(git -C "$REPO" log -1 --format=%cI "$PR2_SHA")"
    echo "pr2_is_expected_$EXPECT_PR2=$([ "$PR2_SHA" = "$EXPECT_PR2" ] && echo yes || echo NO)"
    echo "pr2_tree_equals_ref_pr2_target_v5=$([ -n "$REF_T" ] && { [ "$REF_T" = "$PR2_TREE" ] && echo yes || echo "NO ($REF_T)"; } || echo n/a)"
    echo "dev_is_ancestor_of_pr1=$(isanc "$DEV_SHA" "$PR1_SHA")"; echo "pr1_is_ancestor_of_pr2=$(isanc "$PR1_SHA" "$PR2_SHA")"
    echo "commits_dev_to_pr1=$(git -C "$REPO" rev-list --count "$DEV_SHA..$PR1_SHA")"
    echo "commits_pr1_to_pr2=$(git -C "$REPO" rev-list --count "$PR1_SHA..$PR2_SHA")"
    echo "pr1_commits="; git -C "$REPO" log --format='  %H %s' "$DEV_SHA..$PR1_SHA"
    echo "pr2_commits="; git -C "$REPO" log --format='  %H %s' "$PR1_SHA..$PR2_SHA"
} > "$OUT/meta/tips.txt" 2>&1
log "tips: $PR1_BRANCH=$PR1_SHA (tree $PR1_TREE) $PR2_BRANCH=$PR2_SHA (tree $PR2_TREE); $(grep -E '^(pr1_is_expected|pr2_is_expected|pr2_tree_equals|pr1_is_ancestor|dev_is_ancestor)' "$OUT/meta/tips.txt" | tr '\n' ' ')"
grep -q '^pr1_is_ancestor_of_pr2=yes' "$OUT/meta/tips.txt" || log "WARNING: the pr2 tip does not descend from the pr1 tip (recorded; the run goes on)"

# ---- 2. CUDA builds of dev and both tips (CPU only)
build_tree "$DEV" "$DEV_SHA"
DEV_BIN_SHA=$(sha256sum "$DEV/c/deepseek_v4" | cut -c1-64)
build_tree "$PR1" "$PR1_SHA"
build_tree "$PR2" "$PR2_SHA"
PR1_BIN=$(sha256sum "$PR1/c/deepseek_v4" | cut -c1-64)
PR2_BIN=$(sha256sum "$PR2/c/deepseek_v4" | cut -c1-64)

# ---- 3. preflight (nothing touched)
preflight_common bench_pr1.py bench_decode.py
[ "$(sha256sum < "$CLIENTS/bench_pr1.py" | cut -c1-64)" = "$CLIENT_SHA_PR1" ] || die_pre "bench_pr1.py is not the 2026-10-15 client"
[ "$(sha256sum < "$CLIENTS/bench_decode.py" | cut -c1-64)" = "$CLIENT_SHA_DECODE" ] || die_pre "bench_decode.py is not the 2026-10-15 client"
preflight_tree "$DEV" "$DEV_SHA"
preflight_tree "$PR1" "$PR1_SHA"
preflight_tree "$PR2" "$PR2_SHA"
preflight_switch "$PR1" COLI_CUDA_MOE_BATCH COLI_CUDA_MOE_DOUBLE
preflight_doc "$PR1" COLI_CUDA_MOE_BATCH COLI_CUDA_MOE_DOUBLE
preflight_switch "$PR2" COLI_CUDA_MOE_BATCH COLI_CUDA_MOE_DOUBLE V4_GPU_RESIDENT_FIRST DSV4_CUDA_PEER_DEVICE DSV4_CUDA_PEER_COMPUTE V4_SERVE_STATS_LOG
preflight_doc "$PR2" V4_GPU_RESIDENT_FIRST DSV4_CUDA_PEER_DEVICE DSV4_CUDA_PEER_COMPUTE V4_SERVE_STATS_LOG
grep -q -- '-DCOLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER' "$PR2/c/Makefile.deepseek-v4" || die_pre "dual expert loader not in the $PR2 Makefile"
# the mode lines the per-sample check looks for exist in the sources
grep -q 'moe-batch=off' "$DEV/c/deepseek_v4.c" || die_pre "no moe-batch=off line in $DEV"
for t in "$PR1" "$PR2"; do
    for p in 'moe-batch=on bank=%d-experts' 'moe-double=on' 'mirror-reserve='; do
        grep -qF "$p" "$t/c/deepseek_v4.c" || die_pre "no '$p' line in $t/c/deepseek_v4.c"
    done
done
for p in 'v4_gpu peer-mirrors device=%d reserve=%lldMB mode=compute' 'v4_serve_turn ' 'vram_first_hits_total='; do
    grep -qF "$p" "$PR2/c/deepseek_v4.c" || die_pre "no '$p' line in $PR2/c/deepseek_v4.c"
done
[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2 ] || die_pre "second GPU not visible"
[ -z "$(foreign_gpu_pids)" ] || die_pre "other GPU compute processes: $(foreign_gpu_pids | tr '\n' ' ')"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/meta/gpu_apps_preflight.txt" 2>&1
log "GPU compute processes now: $(tail -n +2 "$OUT/meta/gpu_apps_preflight.txt" | tr '\n' ';')"
[ "$(usage_state | cut -d' ' -f1)" = "size=0" ] || [ "$(usage_state)" = missing ] \
    || log "WARNING: $MODEL/.coli_usage is not empty ($(usage_state)): the RAM pins start from a history (recorded, not an abort)"
[ "$(df -Pk "$FINAL" | awk 'NR==2{print $4}')" -gt 2000000 ] || die_pre "less than 2 GB free in $FINAL"
total_est=0
for item in $ORDER; do total_est=$((total_est + ${EST[${item%%:*}]})); done
log "plan: $ORDER (estimated $total_est s of samples; cap $MAX_GPU_S s, watchdog at $((MAX_GPU_S - HARD_MARGIN_S)) s)"
[ $((total_est + 240)) -le "$MAX_GPU_S" ] || die_pre "plan estimate $total_est s exceeds the cap"
log "preflight ok: pr1 $PR1_SHA binary $PR1_BIN, pr2 $PR2_SHA binary $PR2_BIN, dev binary $DEV_BIN_SHA"

if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
    log "PREFLIGHT_ONLY: all checks passed; would now sleep $lead s until $STOP_AT and stop llama; exiting, llama untouched"
    exit 0
fi

# ---- 4. wait for 01:00:00 (re-check 20 s before)
recheck() {
    local why=
    systemctl --user is-active -q "$LLAMA_UNIT" && llama_healthy || why="$why llama not healthy;"
    ss -ltnH "sport = :$PORT" | grep -q . && why="$why port $PORT in use;"
    [ -z "$(other_colibri_units)" ] || why="$why other colibri unit: $(other_colibri_units | tr '\n' ' ');"
    [ "$(sha256sum "$PR1/c/deepseek_v4" | cut -c1-64)" = "$PR1_BIN" ] || why="$why $PR1 binary changed;"
    [ "$(sha256sum "$PR2/c/deepseek_v4" | cut -c1-64)" = "$PR2_BIN" ] || why="$why $PR2 binary changed;"
    [ "$(sha256sum "$DEV/c/deepseek_v4" | cut -c1-64)" = "$DEV_BIN_SHA" ] || why="$why $DEV binary changed;"
    local t
    for t in "$DEV" "$PR1" "$PR2"; do
        [ -z "$(git -C "$t" status --porcelain --untracked-files=no)" ] || why="$why $t has local changes;"
    done
    [ "$(git -C "$PR1" rev-parse HEAD)" = "$PR1_SHA" ] && [ "$(git -C "$PR2" rev-parse HEAD)" = "$PR2_SHA" ] || why="$why a tree HEAD moved;"
    [ -z "$(foreign_gpu_pids)" ] || why="$why other GPU processes: $(foreign_gpu_pids | tr '\n' ' ');"
    [ -n "$why" ] && die_pre "re-check before the llama stop:$why"
    log "re-check ok; branch tips now: $PR1_BRANCH=$(git -C "$REPO" rev-parse -q --verify "refs/heads/$PR1_BRANCH") $PR2_BRANCH=$(git -C "$REPO" rev-parse -q --verify "refs/heads/$PR2_BRANCH") (the run uses the tips resolved at $(grep '^resolved_at=' "$OUT/meta/tips.txt" | cut -d= -f2))"
}
log "waiting until $STOP_AT ($(( STOP_EPOCH - $(now) )) s)"
wait_until $((STOP_EPOCH - 20))
recheck
wait_until "$STOP_EPOCH"
[ "$(now)" -le $((STOP_EPOCH + LATEST_STOP_S)) ] || die_pre "it is $(date '+%T'), later than $LATEST_STOP_S s after $STOP_AT: the window is missed"

# ---- 5. the GPU window
arm_trap
capture_meta "$DEV" "$PR1" "$PR2"
cp "$OUT/meta/tips.txt" "$BUILDS/$(basename "$PR1").result" "$BUILDS/$(basename "$PR2").result" "$OUT/meta/" 2>/dev/null
stop_llama
start_watchdog

NFAIL=0
for item in $ORDER; do
    arm=${item%%:*}; k=${item#*:}
    if [ -n "${FAILED[$arm]}" ]; then log "skipping $arm s$k: an earlier $arm sample broke a premise"; continue; fi
    if over_budget "$(est_for "$arm" "${EST[$arm]}")"; then log "budget: skipping $arm s$k (GPU window $(gpu_elapsed) s)"; continue; fi
    # shellcheck disable=SC2086
    sample "$arm" "$k" "${TREE[$arm]}" "${CLIENT[$arm]}" "${TMO[$arm]}" ${XENV[$arm]}
    python3 "$FINAL/final_tools.py" check "$arm" "$OUT/$arm/s$k" 2>&1 | tee -a "$OUT/run.log"
    if [ "${PIPESTATUS[0]}" != 0 ]; then
        FAILED[$arm]=1; NFAIL=$((NFAIL + 1))
        if [ "$NFAIL" -ge 2 ]; then log "ABORT after $arm s$k: a second sample broke a premise"; exit 8; fi
        log "$arm s$k broke a premise: the rest of $arm is skipped"
    fi
done

# ---- 6. final4: CUDA tests on both GPUs while llama is still down (results, not aborts)
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
finish
