# final_common.sh (final2) -- shared helpers for run_f42.sh and run_race.sh (shadowfax, user
# maplemercury). Sourced, never run on its own. Derived from final/final_common.sh (same rules,
# same llama handling); what changed is marked "final2:". The caller sets EXP and MAX_GPU_S
# before sourcing, then calls, in this order:
#   lock_or_exit; preflight_* checks (nothing touched yet); arm_trap;
#   capture_meta; stop_llama; sample ...; finish
# Rules kept here: llama.cpp (user unit llama-server-active, port 18080) is stopped once and
# ALWAYS restarted by the EXIT trap, then polled until /health says ok; colibri serves only on
# 127.0.0.1:8090 as its own transient user unit; nothing is killed by pattern, only by PID or unit.

FINAL=/home/maplemercury/colibri-bench/final2
OLD_FINAL=/home/maplemercury/colibri-bench/final    # final2: the r6 build logs and the old lock live there
OUT=$FINAL/$EXP
CLIENTS=$FINAL/clients
MODEL=/home/maplemercury/colibri-models/deepseek-v4-flash-reap
PORT=8090
LLAMA_UNIT=llama-server-active
LLAMA_HEALTH=http://127.0.0.1:18080/health
SERVE_UNIT=colibri-final2-$EXP-serve
EXP_UNIT=colibri-final2-$EXP       # the runner's own unit (launch line in each run_*.sh)
SERVE_ARGS="--model $MODEL --ram 40 --ctx 8192 --host 127.0.0.1 --port $PORT"
# The server gets exactly this environment (env -i): nothing inherited from the
# user manager, so a stray COLI_*/DSV4_*/V4_* variable can never leak in.
BASE_ENV=(PATH=/usr/local/bin:/usr/bin:/bin HOME=/home/maplemercury USER=maplemercury LOGNAME=maplemercury LANG=en_US.UTF-8)
# Same COMMON_ENV as every earlier final run (COLI_CUDA_PIPE=2 is kept for comparability; the
# deepseek_v4 binary never reads it -- only colibri.c does).
COMMON_ENV=(COLI_CUDA=1 COLI_GPU=0 DSV4_CUDA_DEVICE=0 CUDA_DENSE=1 COLI_CUDA_PIPE=2 COLI_CUDA_ATTN_BATCH=1 COLI_CUDA_MOE_BATCH=1 V4_MOE_REFILL_GROUP=12 V4_LOADER_LANES=3 V4_PREFIX_CKPT_DISK=0)
NVSMI_MS=${NVSMI_MS:-5000}   # final2: nvidia-smi sampling interval per sample (ms)

T0=0            # epoch seconds when llama was stopped (start of the GPU window)
NVSMI_PID=      # background nvidia-smi sampler of the running sample
SEQ=0           # global sample order
RESTORED=0
LLAMA_STOPPED=0
declare -A LAST_ARM_S   # measured wall seconds of the last sample of each arm

log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/run.log"; }
# The engine's expert-usage history (seeds the RAM pin ranking at startup) is
# saved only on a clean engine close; servers stopped by `systemctl stop` never
# close cleanly, so it has stayed empty (0 bytes since 2026-10-12). Recorded
# before/after every sample so a cold sample provably started without history.
usage_state() { stat -c 'size=%s mtime=%Y' "$MODEL/.coli_usage" 2>/dev/null || echo missing; }
now() { date +%s; }
gpu_elapsed() { [ "$T0" -gt 0 ] && echo $(( $(now) - T0 )) || echo 0; }

# ---- lock: one final GPU script at a time. Taken BEFORE the trap exists, so a
# second script that loses the race exits without touching llama. final2: BOTH the final2 lock
# and the old final/ lock, so neither a final2 script nor an old final/ script can overlap.
# A previous $OUT (an earlier or aborted run) is moved aside, never mixed with or deleted.
lock_or_exit() {
    exec 9>"$FINAL/.gpu.lock"
    if ! flock -n 9; then
        echo "another colibri GPU script holds $FINAL/.gpu.lock; not starting" >&2
        exit 3
    fi
    exec 8>"$OLD_FINAL/.gpu.lock"
    if ! flock -n 8; then
        echo "another colibri GPU script holds $OLD_FINAL/.gpu.lock; not starting" >&2
        exit 3
    fi
    if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then
        mv "$OUT" "$OUT.old-$(date +%Y%m%d-%H%M%S)"
    fi
    mkdir -p "$OUT/meta"
}

die_pre() { echo "PREFLIGHT FAILED: $*" | tee -a "$OUT/run.log" >&2; exit 4; }

# final2: the build log of a tree, in final2/builds (r7-*) or final/builds (r6-*)
build_log() { # tree
    local n; n=$(basename "$1")
    if [ -f "$FINAL/builds/$n.build.log" ]; then echo "$FINAL/builds/$n.build.log"; else echo "$OLD_FINAL/builds/$n.build.log"; fi
}

# preflight_tree DIR SHA [allow_dirty]: worktree at SHA, clean (or the probe diff only), built.
preflight_tree() {
    local tree=$1 sha=$2 allow_dirty=${3:-0}
    [ -d "$tree/c" ] || die_pre "$tree missing"
    local head; head=$(git -C "$tree" rev-parse HEAD 2>/dev/null)
    [ "$head" = "$sha" ] || die_pre "$tree HEAD $head != $sha"
    if [ "$allow_dirty" = 0 ] && [ -n "$(git -C "$tree" status --porcelain --untracked-files=no)" ]; then
        die_pre "$tree has local changes"
    fi
    [ -x "$tree/c/deepseek_v4" ] || die_pre "$tree/c/deepseek_v4 not built"
    local blog; blog=$(build_log "$tree")
    grep -q "sha=$sha" "$blog" 2>/dev/null && grep -q '^EXIT=0' "$blog" || die_pre "build log $blog not a clean build of $sha"
    [ "$tree/c/deepseek_v4" -nt "$tree/c/deepseek_v4.c" ] || die_pre "$tree binary older than its source"
}

# probe_tree_ok TREE SHA: the probe build of SHA -- only the probe diff (byte-identical to
# probe/<name>.diff), built after it, and the binary carries the probe's format string.
probe_tree_ok() {
    local t=$1 sha=$2 name; name=$(basename "$1")
    ( preflight_tree "$t" "$sha" 1 ) || return 1
    [ -f "$FINAL/probe/$name.diff" ] || return 1
    [ "$(git -C "$t" diff | sha256sum | cut -c1-64)" = "$(sha256sum < "$FINAL/probe/$name.diff" | cut -c1-64)" ] || return 1
    [ "$(git -C "$t" status --porcelain --untracked-files=no)" = " M c/deepseek_v4.c" ] || return 1
    strings "$t/c/deepseek_v4" | grep -q 'v4_logit tok=' || return 1
}

# preflight_switch TREE NAME...: the engine reads each switch (getenv in the source).
preflight_switch() {
    local tree=$1; shift
    local v
    for v in "$@"; do
        grep -q "getenv(\"$v\")" "$tree"/c/*.c "$tree"/c/*.cu 2>/dev/null || die_pre "$tree does not read $v"
    done
}
# preflight_doc TREE NAME...: each switch is documented in the tree's docs/deepseek-v4.md.
preflight_doc() {
    local tree=$1; shift
    local v
    for v in "$@"; do grep -q "$v" "$tree/docs/deepseek-v4.md" || die_pre "$v not in $tree/docs/deepseek-v4.md"; done
}

preflight_common() {
    [ -d "$MODEL" ] || die_pre "model dir missing"
    local f
    for f in "$@"; do [ -f "$CLIENTS/$f" ] || die_pre "client $CLIENTS/$f missing"; done
    if ss -ltnH "sport = :$PORT" | grep -q .; then die_pre "port $PORT already in use"; fi
    if systemctl --user is-active -q "$SERVE_UNIT"; then die_pre "$SERVE_UNIT already active"; fi
    if systemctl --user is-active -q colibri-bench; then die_pre "old colibri-bench unit active"; fi
    local u
    for u in $(systemctl --user list-units --plain --no-legend --state=active,activating,deactivating 'colibri-*' 2>/dev/null | awk '{print $1}'); do
        case "$u" in "$EXP_UNIT"|"$EXP_UNIT.service") ;; *) die_pre "another colibri unit is active: $u" ;; esac
    done
    python3 "$FINAL/final_tools.py" selftest > /dev/null 2>&1 || die_pre "final_tools.py selftest failed"
    local free_kb; free_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    log "preflight ok: MemAvailable=$((free_kb/1024)) MiB, llama $(systemctl --user is-active $LLAMA_UNIT)"
}

# ---- restore llama: the EXIT trap (also called explicitly at the end).
stop_sampler() {
    if [ -n "$NVSMI_PID" ]; then kill "$NVSMI_PID" 2>/dev/null; wait "$NVSMI_PID" 2>/dev/null; NVSMI_PID=; fi
}
llama_healthy() { curl -s -m 3 "$LLAMA_HEALTH" 2>/dev/null | grep -q '"status": *"ok"'; }
restore() {
    local rc=$?
    [ "$RESTORED" = 1 ] && return 0
    RESTORED=1
    trap '' TERM INT HUP    # nothing interrupts the restore
    stop_sampler
    systemctl --user stop "$SERVE_UNIT" 2>/dev/null
    local i
    for i in $(seq 1 60); do systemctl --user is-active -q "$SERVE_UNIT" || break; sleep 2; done
    log "restore: starting $LLAMA_UNIT (script status $rc, GPU window $(gpu_elapsed) s)"
    systemctl --user start "$LLAMA_UNIT"
    local ok=0
    for i in $(seq 1 120); do llama_healthy && { ok=1; break; }; sleep 5; done
    if [ $ok = 0 ]; then
        log "restore: llama not healthy after 600 s, starting it again"
        systemctl --user start "$LLAMA_UNIT"
        for i in $(seq 1 60); do llama_healthy && { ok=1; break; }; sleep 5; done
    fi
    local h; h=$(curl -s -m 3 "$LLAMA_HEALTH")
    log "restore: llama health '$h' (ok=$ok) at $(date -Is); llama-down window $(gpu_elapsed) s"
    mkdir -p "$OUT/meta"
    { echo "restored_at=$(date -Is)"; echo "health=$h"; echo "ok=$ok"; echo "gpu_window_s=$(gpu_elapsed)";
      systemctl --user status "$LLAMA_UNIT" --no-pager 2>&1 | head -5;
      nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv; } > "$OUT/meta/llama_after.txt" 2>&1
    return 0
}
arm_trap() {
    trap restore EXIT
    trap 'log "signal received, exiting (EXIT trap restores llama)"; exit 143' TERM INT HUP
}

# ---- GPU idle: no compute process on either card.
gpu_idle_wait() { # $1 = max seconds
    local i
    for i in $(seq 1 $(( ${1:-120} / 3 ))); do
        [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -cE '^[[:space:]]*[0-9]+')" = 0 ] && return 0
        sleep 3
    done
    return 1
}

stop_llama() {
    { echo "checked_at=$(date -Is)"; echo "active=$(systemctl --user is-active $LLAMA_UNIT)";
      echo "health=$(curl -s -m 3 $LLAMA_HEALTH)";
      nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv;
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } > "$OUT/meta/llama_before.txt" 2>&1
    log "stopping $LLAMA_UNIT (the EXIT trap restarts it)"
    T0=$(now)
    LLAMA_STOPPED=1
    systemctl --user stop "$LLAMA_UNIT"
    if ! gpu_idle_wait 120; then
        log "GPUs still have compute processes 120 s after the llama stop; aborting"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a "$OUT/run.log"
        exit 5
    fi
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/idle gpu /' | tee -a "$OUT/run.log"
}

# ---- experiment metadata (nvidia-smi queries only, no GPU work).
tree_info() { # tree
    local t=$1
    echo "tree=$t"
    echo "  head=$(git -C "$t" rev-parse HEAD)"
    echo "  parent=$(git -C "$t" rev-parse HEAD^)"
    echo "  subject=$(git -C "$t" log -1 --format=%s)"
    echo "  status_lines=$(git -C "$t" status --porcelain --untracked-files=no | wc -l)"
    [ -n "$(git -C "$t" status --porcelain --untracked-files=no)" ] && echo "  diff_sha256=$(git -C "$t" diff | sha256sum | cut -c1-64)"
    echo "  binary_sha256=$(sha256sum "$t/c/deepseek_v4" | cut -c1-64)"
    echo "  binary_mtime=$(stat -c %y "$t/c/deepseek_v4")"
    echo "  build_log=$(build_log "$t")"
}
capture_meta() { # trees...
    local m=$OUT/meta; mkdir -p "$m"
    {
        echo "exp=$EXP captured_at=$(date -Is) host=$(hostname) runner_pid=$$"
        echo "runner_unit=$(cat /proc/self/cgroup | sed 's#.*/##')"
        uname -a; grep -E '^(PRETTY_NAME|VERSION)=' /etc/os-release
        lscpu | grep -E 'Model name|Socket|Core|Thread|^CPU\(s\)|MHz'
        echo "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
        free -b; free -g
        gcc --version | head -1; nvcc --version | tail -2; python3 --version
        nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current,power.limit --format=csv
        nvidia-smi topo -m
        nvidia-smi nvlink -s 2>&1 | head -16
        lsblk -o NAME,MODEL,SIZE,TYPE,MOUNTPOINT | grep -v loop
        findmnt -T "$MODEL"; df -h "$MODEL"; ls -la --time-style=full-iso "$MODEL" | head -20
        echo "coli_usage $(usage_state)"
    } > "$m/host.txt" 2>&1
    local t
    for t in "$@"; do tree_info "$t"; done > "$m/trees.txt" 2>&1
    cp "$0" "$FINAL/final_common.sh" "$FINAL/final_tools.py" "$m/" 2>/dev/null
    cp "$CLIENTS"/*.py "$m/" 2>/dev/null
    cp "$FINAL"/probe/* "$m/" 2>/dev/null
    for t in "$@"; do cp "$(build_log "$t")" "$m/" 2>/dev/null; done
    (cd "$m" && sha256sum ./*.sh ./*.py ./*.diff ./*.build.log > sha256s.txt 2>/dev/null)
    { echo "COMMON_ENV=${COMMON_ENV[*]}"; echo "BASE_ENV=${BASE_ENV[*]}"; echo "SERVE_ARGS=$SERVE_ARGS"; echo "NVSMI_MS=$NVSMI_MS"; } > "$m/common_env.txt"
}

# final2: KEY=VAL... merged left to right, a later KEY replacing an earlier one in place (so an
# arm's COLI_CUDA_MOE_BATCH=0 replaces COMMON_ENV's =1 instead of appearing twice); one per line.
merge_env() {
    local -A val=(); local -a order=(); local kv k
    for kv in "$@"; do
        k=${kv%%=*}
        [ -n "${val[$k]+x}" ] || order+=("$k")
        val[$k]=$kv
    done
    for k in "${order[@]}"; do printf '%s\n' "${val[$k]}"; done
}

# ---- one cold sample: fresh server unit, client, stop, GPUs idle again.
# sample ARM K TREE CLIENT TIMEOUT_S [KEY=VAL...]
#   CLIENT: pr1:first | pr1:both | decode | decodesub:N | f42 | race:short:N | race:long:N
sample() {
    local arm=$1 k=$2 tree=$3 client=$4 tmo=$5; shift 5
    local d=$OUT/$arm/s$k s_epoch
    s_epoch=$(now)
    mkdir -p "$d"
    SEQ=$((SEQ + 1))
    local envs=()
    mapfile -t envs < <(merge_env "${BASE_ENV[@]}" "${COMMON_ENV[@]}" "$@")
    log "=== sample $SEQ: $arm s$k tree=$tree client=$client timeout=${tmo}s extra='$*' (GPU window $(gpu_elapsed) s)"
    if ! gpu_idle_wait 120; then log "GPUs not idle before $arm s$k; aborting"; exit 6; fi
    local i
    for i in $(seq 1 30); do ss -ltnH "sport = :$PORT" | grep -q . || break; sleep 2; done
    if ss -ltnH "sport = :$PORT" | grep -q .; then log "port $PORT still busy before $arm s$k; aborting"; exit 7; fi
    {
        echo "arm=$arm"; echo "sample=$k"; echo "seq=$SEQ"; echo "tree=$tree"
        echo "tree_head=$(git -C "$tree" rev-parse HEAD)"
        echo "tree_dirty_lines=$(git -C "$tree" status --porcelain --untracked-files=no | wc -l)"
        echo "tree_diff_sha256=$(git -C "$tree" diff | sha256sum | cut -c1-64)"
        echo "binary_sha256=$(sha256sum "$tree/c/deepseek_v4" | cut -c1-64)"
        echo "client=$client"; echo "timeout_s=$tmo"
        echo "extra_env=$*"; echo "server_env=${envs[*]}"
        echo "serve_cmd=python3 $tree/c/coli serve $SERVE_ARGS"
        echo "usage_before=$(usage_state)"
        echo "start=$(date -Is)"; echo "start_epoch=$(now)"
    } > "$d/sample.txt"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv > "$d/gpu_before.csv" 2>&1
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.sm,pcie.link.gen.current,pcie.link.width.current \
        --format=csv -lms "$NVSMI_MS" > "$d/nvsmi.csv" 2>&1 &
    NVSMI_PID=$!
    systemd-run --user --unit="$SERVE_UNIT" --collect -p WorkingDirectory="$tree" \
        /usr/bin/env -i "${envs[@]}" /bin/bash -c "exec python3 $tree/c/coli serve $SERVE_ARGS > $d/serve.log 2>&1" \
        >> "$OUT/run.log" 2>&1
    local ready=0 t_start; t_start=$(now)
    while [ $(( $(now) - t_start )) -lt 900 ]; do
        grep -q "listening on" "$d/serve.log" 2>/dev/null && { ready=1; break; }
        systemctl --user is-active -q "$SERVE_UNIT" || break
        sleep 2
    done
    { echo "ready=$ready"; echo "ready_s=$(( $(now) - t_start ))"; } >> "$d/sample.txt"
    local crc=99
    if [ $ready = 1 ]; then
        record_procs "$d"
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > "$d/gpu_ready.csv" 2>&1
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$d/gpu_apps.csv" 2>&1
        local c_start; c_start=$(now)
        case "$client" in
            pr1:*)       BENCH_REQS=${client#pr1:} timeout "$tmo" python3 "$CLIENTS/bench_pr1.py" "$d/bench.jsonl" > "$d/bench.out" 2> "$d/bench.err" ;;
            decode)      timeout "$tmo" python3 "$CLIENTS/bench_decode.py" > "$d/bench.jsonl" 2> "$d/bench.err" ;;
            decodesub:*) BENCH_NPROMPTS=${client#decodesub:} timeout "$tmo" python3 "$CLIENTS/bench_decode_sub.py" "$d/bench.jsonl" > "$d/bench.out" 2> "$d/bench.err" ;;
            f42)         timeout "$tmo" python3 "$CLIENTS/bench_f42.py" "$d/bench.jsonl" > "$d/bench.out" 2> "$d/bench.err" ;;
            race:*:*)    local rm=${client#race:}; RACE_MODE=${rm%%:*} RACE_NTOK=${rm#*:} \
                             timeout "$tmo" python3 "$CLIENTS/bench_race.py" "$d/bench.jsonl" > "$d/bench.out" 2> "$d/bench.err" ;;
            *)           log "unknown client $client"; false ;;
        esac
        crc=$?
        echo "client_s=$(( $(now) - c_start ))" >> "$d/sample.txt"
    else
        log "server for $arm s$k never became ready"
    fi
    echo "client_rc=$crc" >> "$d/sample.txt"
    echo "server_active_at_end=$(systemctl --user is-active $SERVE_UNIT)" >> "$d/sample.txt"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > "$d/gpu_end.csv" 2>&1
    systemctl --user stop "$SERVE_UNIT" 2>/dev/null
    for i in $(seq 1 60); do systemctl --user is-active -q "$SERVE_UNIT" || break; sleep 2; done
    gpu_idle_wait 120 || log "warning: compute processes remain after stopping $SERVE_UNIT"
    stop_sampler
    { echo "end=$(date -Is)"; echo "end_epoch=$(now)"; echo "wall_s=$(( $(now) - s_epoch ))"; echo "usage_after=$(usage_state)"; } >> "$d/sample.txt"
    if [ "$(grep '^usage_before=' "$d/sample.txt" | cut -d= -f2-)" != "$(usage_state)" ]; then
        log "WARNING: $MODEL/.coli_usage changed during $arm s$k ($(usage_state)): later samples start with a usage history"
    fi
    LAST_ARM_S[$arm]=$(( $(now) - s_epoch ))
    grep -E "${MODE_RE:-peer-mirrors|moe-batch=|moe-double|mirror-reserve}" "$d/serve.log" 2>/dev/null \
        | grep -v "moe-batch refill" | cut -c1-400 > "$d/mode.txt"
    grep -ciE "out of memory|FAILED|Traceback|error" "$d/serve.log" > "$d/error_count.txt" 2>/dev/null
    log "--- $arm s$k done: client_rc=$crc ready=$ready $(tail -c 300 "$d/bench.out" 2>/dev/null | tr '\n' ' ')"
    if [ -f "$d/bench.jsonl" ]; then
        python3 "$FINAL/final_tools.py" brief "$d/bench.jsonl" 2>&1 | tee -a "$OUT/run.log"
    fi
    return 0
}

# record the processes of the serve unit (exe, argv) and the engine's environment
record_procs() { # dir
    local d=$1 cg pid exe
    cg=$(systemctl --user show -p ControlGroup --value "$SERVE_UNIT")
    : > "$d/procs.txt"
    for pid in $(cat "/sys/fs/cgroup$cg/cgroup.procs" 2>/dev/null); do
        exe=$(readlink "/proc/$pid/exe")
        echo "pid=$pid exe=$exe argv=$(tr '\0' ' ' < /proc/$pid/cmdline)" >> "$d/procs.txt"
        case "$exe" in
            */c/deepseek_v4) tr '\0' '\n' < "/proc/$pid/environ" | sort > "$d/engine_environ.txt" ;;
            */python3*)      tr '\0' '\n' < "/proc/$pid/environ" | sort > "$d/serve_environ.txt" ;;
        esac
    done
}

# rotate N WORD...: the words rotated left by N (interleaving order per round)
rotate() {
    local n=$1; shift
    local a=("$@") len=$# i
    for ((i = 0; i < len; i++)); do printf '%s ' "${a[$(( (i + n) % len ))]}"; done
}

# est_for ARM DEFAULT_S: the larger of the default and the arm's last measured sample
est_for() { local m=${LAST_ARM_S[$1]:-0}; [ "$m" -gt "$2" ] && echo "$m" || echo "$2"; }
# over_budget EST_S: would a sample of EST_S seconds overrun MAX_GPU_S (with
# 240 s kept for stopping the last server and restoring llama)?
over_budget() { [ $(( $(gpu_elapsed) + $1 + 240 )) -gt "$MAX_GPU_S" ]; }

# finish KIND: restore llama first, then summarise and pack the evidence.
finish() {
    local kind=$1
    restore
    python3 "$FINAL/final_tools.py" summarize "$kind" "$OUT" > "$OUT/summary.txt" 2>&1
    cat "$OUT/summary.txt" | tee -a "$OUT/run.log"
    log "ALL DONE; packing $FINAL/$EXP.tgz"
    tar -C "$FINAL" -czf "$FINAL/$EXP.tgz" "$EXP" && (cd "$FINAL" && sha256sum "$EXP.tgz" > "$EXP.tgz.sha256")
    log "packed $FINAL/$EXP.tgz $(cut -c1-64 "$FINAL/$EXP.tgz.sha256")"
}
