#!/bin/bash
# F42 fix: does pr1/v5's routing-books fix (c7f79815 "GPU-bank prefill keeps the expert store's
# routing books", tip 4feb6d48) recover the decode slowdown after a GPU-bank prefill (F42, confirmed
# in final2/f42v2: on identical text GB 7.4 % slower than the CPU union U, +57 % RAM-store misses in
# the first ~25 decode tokens, because the bank refill counted each routed expert once per layer
# per prompt: flat usage, pins ranked by id, 7-9 of 16 pins active, first-read recency)?
# Same request, client, reserve and probe as f42v2, on the PR #1 code line:
#   FX   fixed probe build (4feb6d48 + probe), GPU-bank prefill
#   UX   FX + COLI_CUDA_MOE_BATCH_MIN=100000 (CPU union, same 3100 MiB reserve; the gate is checked
#        below exactly as in run_f42v2.sh)
#   OX   unfixed probe build (old pr1 tip 93b74183 + the same probe), GPU-bank prefill: the F42
#        reference on this code line; OX -> FX changes only the engine commit (the fix; the code diff
#        93b74183..4feb6d48 in c/deepseek_v4.c is that commit's)
#   W    warm-up: FX's config on the fixed probe build, ONE sample first, discarded
#   FXn  FX's env on the plain fixed build (no probe), 3 samples: the probe changes neither speed
#        nor text on this line (lowest priority: last in each place it appears, skipped first by
#        the budget)
# Expected: FX's decode-start books look like UX's (layer requests ~1065, 16 active pins ranked by
# usage, usage > 1), its early-token RAM misses drop towards UX's, and FX recovers part or all of
# the OX-UX decode gap; TTFT FX vs OX shows any prefill cost of the bookkeeping.
#
# shadowfax only. Stops llama.cpp ONCE; the EXIT trap ALWAYS restarts it and waits for /health ok.
# Launch (own transient unit, survives an ssh drop):
#   systemd-run --user --unit=colibri-final2-f42fix --collect -p TimeoutStopSec=900 \
#     -p WorkingDirectory=/home/maplemercury/colibri-bench/final2 /bin/bash /home/maplemercury/colibri-bench/final2/run_f42fix.sh
#   progress: tail /home/maplemercury/colibri-bench/final2/f42fix/run.log
# CPU-only check first (nothing stopped or started, exits before the trap exists):
#   PREFLIGHT_ONLY=1 /bin/bash /home/maplemercury/colibri-bench/final2/run_f42fix.sh   (log: final2/f42fix-preflight/run.log)
#
# Builds (build_r9f.sh; same make line as build_r8.sh):
#   /tmp/cw/r9-pr1v5-probe  4feb6d48 + final2/probe/f42fix_probe_patch.py (diff probe/r9-pr1v5-probe.diff)
#   /tmp/cw/r9-pr1-probe    93b74183 + the same patch (diff probe/r9-pr1-probe.diff; the same +/- lines)
#   /tmp/cw/r9-pr1v5        4feb6d48 unmodified
# The probe is f42v2_probe_patch.py adapted to the PR #1 line, which has no V4_SERVE_STATS_LOG decode
# counters: it keeps its own (calls, total_s, fetch_wait_s, attach_s; same places and timers as pr2's
# v4_dec_moe; no uploads counter) and adds a store-wait ms per token-layer. V4_DEC_TRACE=2 prints
# v4_dec_trace_start / v4_dec_res / v4_dec_trace / v4_dec_layer on stderr only.
# Every sample: --ram 40 --ctx 8192, COMMON_ENV of final_common.sh (COLI_CUDA_MOE_BATCH=1), a cold
# server per sample (env -i), DSV4_DECODE_PROF=1, V4_DEC_TRACE=2 (read by the probe builds only).
# V4_SERVE_STATS_LOG is not set: the PR #1 line does not read it.
# Client bench_f42v2.py (unchanged): bench_pr1.py REQUESTS[0] byte for byte, reference answer 32534c06...
# Order: W, then 5 rounds of FX/OX/UX in rotating order, FXn after rounds 2, 4 and 5.
# Expected window ~48 min (FX/OX/FXn/W ~100 s, UX ~235 s per sample, + llama stop/restore); hard cap
# MAX_GPU_S = 55 min (a sample that would overrun it, with 240 s kept for the restore, is skipped).
# Abort (EXIT trap restores llama) when a premise breaks: a failed client, no probe lines in a probe
# arm, a reserve other than 3100 MiB, the bank used in UX or not used in FX/OX/FXn/W.
# Captured per sample (final2/f42fix/<arm>/s<k>/): serve.log, bench.jsonl, nvsmi.csv every 1 s,
# sample.txt, engine_environ.txt, procs.txt, gpu_*.csv, mode.txt.
# Summary: analyze_f42fix.py summarize f42fix (summary.txt / summary.json), then f42fix.tgz.
# Uses final_common.sh unchanged (client case "f42v2").

EXP=f42fix
[ "${PREFLIGHT_ONLY:-0}" = 1 ] && EXP=f42fix-preflight
MAX_GPU_S=$((55 * 60))      # hard cap on the llama-down window
NVSMI_MS=1000
source /home/maplemercury/colibri-bench/final2/final_common.sh

FIX_SHA=4feb6d487973e064abc53486f379032aa402efc6
OLD_SHA=93b741839677ec457868fd71ab668cc1249b9a22
FIXN=/tmp/cw/r9-pr1v5
FIXN_BIN_SHA=a17d0a0269396f3570c35c0514df4b227bc82530e8b8f73ffb152b918cd5ee1c
FIXP=/tmp/cw/r9-pr1v5-probe
FIXP_DIFF_SHA=d72e7c57c642aa619146cc141b0672db78a376ac7f3fc713971b7a143b139342
FIXP_BIN_SHA=98ecc0cbe898fadb0bd55060a7e5c8544ae817e096e45b0df166f90fac188b8f
OLDP=/tmp/cw/r9-pr1-probe
OLDP_DIFF_SHA=b500a731e16e913ce3c37ff165fa834aaa4d9327cd49c81d11f7645e0e3979a1
OLDP_BIN_SHA=f5ef189a1e4b319e317a95688aa2da34d5dba62c199e15e740b7389d52f57993
PATCH_SHA=2562ff545038b4a16da60f942303a6b4374413f2c839cd9a940d667ad7c1e78a
REF_TEXT_SHA=32534c06abd640d59d2eb39ab6609051f63e9876f533aad1d75dd5cd775835ba
MODE_RE='moe-batch=on|moe-batch=off|moe-double|mirror-reserve|mirror-cache:|DSV4 CUDA\] device|v4_hot_policy|ram_tiers|^v4_dec_trace_start|out of memory'
STATS="DSV4_DECODE_PROF=1 V4_DEC_TRACE=2"

declare -A XENV TREE EST PATHW
XENV[W]="$STATS";                                   TREE[W]=$FIXP;   EST[W]=115;   PATHW[W]=bank
XENV[FX]="$STATS";                                  TREE[FX]=$FIXP;  EST[FX]=115;  PATHW[FX]=bank
XENV[UX]="$STATS COLI_CUDA_MOE_BATCH_MIN=100000";   TREE[UX]=$FIXP;  EST[UX]=250;  PATHW[UX]=union
XENV[OX]="$STATS";                                  TREE[OX]=$OLDP;  EST[OX]=115;  PATHW[OX]=bank
XENV[FXn]="$STATS";                                 TREE[FXn]=$FIXN; EST[FXn]=115; PATHW[FXn]=bank
ORDER="W:1 FX:1 OX:1 UX:1 UX:2 OX:2 FX:2 FXn:1 OX:3 UX:3 FX:3 FX:4 UX:4 OX:4 FXn:2 UX:5 FX:5 OX:5 FXn:3"
TMO=900
is_probe() { [ "$1" = "$FIXP" ] || [ "$1" = "$OLDP" ]; }

# a probe tree: SHA + exactly the recorded probe diff, built after it by build_r9f.sh
preflight_probe() { # tree sha diff_sha bin_sha
    local t=$1 sha=$2 dsha=$3 bsha=$4 n; n=$(basename "$1")
    preflight_tree "$t" "$sha" 1
    [ -f "$FINAL/probe/$n.diff" ] || die_pre "$FINAL/probe/$n.diff missing"
    [ "$(sha256sum < "$FINAL/probe/$n.diff" | cut -c1-64)" = "$dsha" ] || die_pre "$FINAL/probe/$n.diff is not the recorded diff"
    [ "$(git -C "$t" diff | sha256sum | cut -c1-64)" = "$dsha" ] || die_pre "$t diff differs from the recorded one"
    [ "$(git -C "$t" status --porcelain --untracked-files=no)" = " M c/deepseek_v4.c" ] || die_pre "$t: changes beyond c/deepseek_v4.c"
    head -1 "$(build_log "$t")" | grep -q "probe_diff_sha256=$dsha probe_patch_sha256=$PATCH_SHA" || die_pre "build log of $t is not for this diff/patch"
    [ "$(sha256sum "$t/c/deepseek_v4" | cut -c1-64)" = "$bsha" ] || die_pre "$t binary is not the recorded build"
    strings "$t/c/deepseek_v4" | grep -q 'v4_dec_trace ord=' || die_pre "$t binary has no probe"
}

# COLI_CUDA_MOE_BATCH_MIN: read once, in the bank's per-call gate, and nowhere in the reserve sizing
preflight_min_gate() {
    local src=$1/c/deepseek_v4.c
    [ "$(grep -c 'getenv("COLI_CUDA_MOE_BATCH_MIN")' $src)" = 1 ] || die_pre "COLI_CUDA_MOE_BATCH_MIN read more than once in $src"
    grep -q 'v4_gpu_moe_batch_hint_tokens < minimum) return -1;' $src || die_pre "MOE_BATCH_MIN gate not found in $src"
    grep -q 'coli_v4_gpu_moe_batch_hint(fresh);' $src || die_pre "the prefill hint is not the fresh token count in $src"
    local f body
    for f in v4_gpu_mirror_reserve_size v4_gpu_mirror_reserve_mb; do
        body=$(awk -v f="$f" '$0 ~ "^static [a-z ]*" f "\\(" && $0 ~ /\{$/ {on=1} on {print} on && /^}/ {exit}' $src)
        [ -n "$body" ] || die_pre "$f not found in $src"
        if echo "$body" | grep -qE 'MOE_BATCH_MIN|hint_tokens|minimum'; then die_pre "$f depends on the bank gate"; fi
    done
    [ "$(grep -c 'getenv("COLI_CUDA_MOE_BATCH_MIN")' $1/c/*.cu $1/c/*.h 2>/dev/null | awk -F: '{s+=$NF} END {print s+0}')" = 0 ] \
        || die_pre "COLI_CUDA_MOE_BATCH_MIN read outside deepseek_v4.c"
}

# after each sample: the premises of the comparison; a broken one aborts (the trap restores llama)
check_sample() { # arm k
    local arm=$1 k=$2 d=$OUT/$1/s$2
    local crc reserve bank refill start text ttft chunks
    crc=$(grep '^client_rc=' "$d/sample.txt" | cut -d= -f2)
    reserve=$(grep -ohE 'mirror-reserve=[0-9]+MiB' "$d/serve.log" | tr '\n' ' ')
    bank=$(grep -c 'moe-batch=on bank=' "$d/serve.log")
    refill=$(grep -c 'moe-batch refill ' "$d/serve.log")
    start=$(grep -m1 '^v4_dec_trace_start ' "$d/serve.log" | grep -oE 'bank=[0-9]+ union=[0-9]+ rows=[0-9]+')
    read -r text ttft chunks < <(python3 -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')][0]; print(r['text_sha256'], r['ttft_s'], r['decode_chunks'])" "$d/bench.jsonl" 2>/dev/null)
    log "check $arm s$k: client_rc=$crc reserve='$reserve' bank_lines=$bank refill_lines=$refill prefill_paths='$start' ttft=$ttft chunks=$chunks text=${text:0:12} ref=$([ "$text" = "$REF_TEXT_SHA" ] && echo yes || echo NO) trace_lines=$(grep -c '^v4_dec_trace ' "$d/serve.log") layer_lines=$(grep -c '^v4_dec_layer ' "$d/serve.log") res_lines=$(grep -c '^v4_dec_res ' "$d/serve.log")"
    log "stats $arm s$k: $(grep -h '^v4_dec_trace ord=25 ' "$d/serve.log" | head -1 | cut -c1-260) | lreq0=$(grep -m1 '^v4_dec_res ' "$d/serve.log" | grep -oE 'lreq=[0-9]+')"
    local why=
    [ "$crc" = 0 ] || why="client_rc=$crc"
    [ "$reserve" = "mirror-reserve=3100MiB " ] || why="$why reserve='$reserve' (want one 3100MiB line)"
    if [ "${PATHW[$arm]}" = union ]; then
        [ "$bank" = 0 ] && [ "$refill" = 0 ] || why="$why GPU bank used in $arm (bank_lines=$bank refill_lines=$refill)"
        [[ "$start" =~ ^bank=0\ union=[1-9] ]] || why="$why prefill paths '$start' (want bank=0 union>0)"
    else
        [ "$bank" -ge 1 ] && [ "$refill" -ge 1 ] || why="$why GPU bank not used in $arm (bank_lines=$bank refill_lines=$refill)"
        if is_probe "${TREE[$arm]}"; then
            [[ "$start" =~ ^bank=[1-9] ]] || why="$why prefill paths '$start' (want bank>0)"
            [[ "$start" =~ \ union=0\  ]] || log "NOTE: $arm s$k prefill paths '$start': some chunk-layers fell back to the CPU union"
        fi
    fi
    if is_probe "${TREE[$arm]}"; then
        [ "$(grep -c '^v4_dec_trace ' "$d/serve.log")" -ge 2 ] || why="$why no per-token probe lines"
    elif grep -qE '^v4_dec_(trace|trace_start|layer|res) ' "$d/serve.log"; then
        why="$why probe lines from the non-probe binary"
    fi
    if [ -n "$why" ]; then
        log "ABORT after $arm s$k: premise broken:$why"
        exit 8
    fi
    [ "$text" = "$REF_TEXT_SHA" ] || log "NOTE: $arm s$k text ${text:0:12} differs from the reference ${REF_TEXT_SHA:0:12} (a result, not an abort)"
}

# restore llama first, then summarise and pack the evidence (final_common.sh's finish, with this analyzer)
finish_f42fix() {
    restore
    python3 "$FINAL/analyze_f42fix.py" summarize "$OUT" > "$OUT/summary.txt" 2>&1
    tee -a "$OUT/run.log" < "$OUT/summary.txt"
    log "ALL DONE; packing $FINAL/$EXP.tgz"
    tar -C "$FINAL" -czf "$FINAL/$EXP.tgz" "$EXP" && (cd "$FINAL" && sha256sum "$EXP.tgz" > "$EXP.tgz.sha256")
    log "packed $FINAL/$EXP.tgz $(cut -c1-64 "$FINAL/$EXP.tgz.sha256")"
}

lock_or_exit
log "run_f42fix.sh start (pid $$)"
preflight_common bench_f42v2.py bench_common.py bench_pr1.py
grep -q 'f42v2)  *timeout' "$FINAL/final_common.sh" || die_pre "final_common.sh has no f42v2 client"
preflight_tree $FIXN $FIX_SHA
[ "$(sha256sum $FIXN/c/deepseek_v4 | cut -c1-64)" = "$FIXN_BIN_SHA" ] || die_pre "$FIXN binary is not the recorded build"
if strings $FIXN/c/deepseek_v4 | grep -q 'v4_dec_trace'; then die_pre "$FIXN binary carries the probe"; fi
preflight_probe $FIXP $FIX_SHA $FIXP_DIFF_SHA $FIXP_BIN_SHA
preflight_probe $OLDP $OLD_SHA $OLDP_DIFF_SHA $OLDP_BIN_SHA
[ "$(sha256sum < "$FINAL/probe/f42fix_probe_patch.py" | cut -c1-64)" = "$PATCH_SHA" ] || die_pre "probe patch script changed"
# the same probe on both trees: identical added/removed lines (only hunk offsets and blob ids differ)
cmp -s <(grep -E '^[-+]' "$FINAL/probe/r9-pr1v5-probe.diff" | grep -vE '^(\+\+\+|---) ') \
       <(grep -E '^[-+]' "$FINAL/probe/r9-pr1-probe.diff" | grep -vE '^(\+\+\+|---) ') || die_pre "the two probe diffs differ"
# OX -> FX is one engine commit: the engine sources (c/ without c/tests) differ between the trees
# by exactly c7f79815's added/removed lines
FIX_COMMIT=c7f7981508113c80bee015de6d29fed4e78524bc
cmp -s <(git -C /tmp/colibri diff $OLD_SHA $FIX_SHA -- c ':!c/tests' | grep -E '^[-+]' | grep -vE '^(\+\+\+|---) ') \
       <(git -C /tmp/colibri diff $FIX_COMMIT^ $FIX_COMMIT -- c ':!c/tests' | grep -E '^[-+]' | grep -vE '^(\+\+\+|---) ') \
    || die_pre "the engine code between $OLD_SHA and $FIX_SHA is not the fix commit's"
for t in $FIXP $OLDP; do
    preflight_switch $t COLI_CUDA_MOE_BATCH COLI_CUDA_MOE_BATCH_MIN DSV4_DECODE_PROF V4_DEC_TRACE
    preflight_min_gate $t
    grep -q 'mirror-reserve=' $t/c/deepseek_v4.c || die_pre "no mirror-reserve log line in $t"
    grep -q 'fprintf(stderr, "decprof position=' $t/c/deepseek_v4.c || die_pre "no decprof line in $t"
done
preflight_switch $FIXN COLI_CUDA_MOE_BATCH COLI_CUDA_MOE_BATCH_MIN DSV4_DECODE_PROF
preflight_doc $FIXN COLI_CUDA_MOE_BATCH COLI_CUDA_MOE_BATCH_MIN DSV4_DECODE_PROF
preflight_min_gate $FIXN
[ "$(usage_state | cut -d' ' -f1)" = "size=0" ] || [ "$(usage_state)" = missing ] \
    || die_pre "$MODEL/.coli_usage is not empty ($(usage_state)): the RAM pins would start from a history"
python3 "$CLIENTS/bench_f42v2.py" --check > /dev/null || die_pre "bench_f42v2.py prompt check failed"
python3 "$FINAL/analyze_f42fix.py" selftest > /dev/null 2>&1 || die_pre "analyze_f42fix.py selftest failed"
total_est=0
for item in $ORDER; do total_est=$((total_est + ${EST[${item%%:*}]})); done
log "plan: $ORDER (estimated $total_est s of samples, cap $MAX_GPU_S s)"
[ $((total_est + 240)) -le "$MAX_GPU_S" ] || die_pre "plan estimate $total_est s exceeds the cap"

if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then log "PREFLIGHT_ONLY: all checks passed; exiting before llama is touched"; exit 0; fi
arm_trap
capture_meta $FIXN $FIXP $OLDP
cp "$FINAL/analyze_f42fix.py" "$FINAL/analyze_f42v2.py" "$FINAL/build_r9f.sh" "$OUT/meta/" 2>/dev/null
(cd "$OUT/meta" && sha256sum ./*.sh ./*.py ./*.diff ./*.build.log > sha256s.txt 2>/dev/null)
stop_llama

for item in $ORDER; do
    arm=${item%%:*}; k=${item#*:}
    if over_budget $(est_for $arm ${EST[$arm]}); then log "budget: skipping $arm s$k (GPU window $(gpu_elapsed) s)"; continue; fi
    # shellcheck disable=SC2086
    sample $arm $k ${TREE[$arm]} f42v2 $TMO ${XENV[$arm]}
    check_sample $arm $k
done
finish_f42fix
