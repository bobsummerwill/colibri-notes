# final_common.sh (final13, 2026-10-24: final12's, new paths and units; final12, 2026-10-24: final11's, new paths and units; final11, 2026-10-24: final10's, new paths and units; SMAPS_EVERY=n also snapshots smaps every n samples (5 s each); final10, 2026-10-24: final9's, new paths and units, and memx.csv (memory by class) per sample; final9, 2026-10-24: final8's, new paths and units; final8, 2026-10-23: final7's, new paths and units; final7, 2026-10-23: final6's with an OOM score and a host-memory record per sample;
# final6, 2026-10-22: final5's, for the full DeepSeek V4 Flash in the default configuration (no COLI_CUDA_MOE_BATCH), dev 7df8c27c vs PR #1 merged with it)
# (final5: copy of final4 for the stack on dev 736a0682, 2026-10-19) -- shared helpers for run_final3.sh (shadowfax, user maplemercury).
# Sourced, never run on its own. Derived from final2/final_common.sh (same rules, same llama
# handling, same sample layout); what changed is marked "final3:". The caller sets EXP, MAX_GPU_S
# and (optionally) BUILDS before sourcing, then calls, in this order:
#   lock_or_exit; build_tree ...; preflight_* checks (nothing touched yet); wait_until ...;
#   arm_trap; capture_meta; stop_llama; start_watchdog; sample ...; finish
# Rules kept here: llama.cpp (user unit llama-server-active, port 18080) is stopped once and
# ALWAYS restarted by the EXIT trap, then polled until /health says ok; colibri serves only on
# 127.0.0.1:8090 as its own transient user unit; nothing is killed by pattern, only by PID or unit.
# final3: a hard cap -- a watchdog sends TERM to the runner MAX_GPU_S - HARD_MARGIN_S after the
# llama stop, so llama is back within MAX_GPU_S even if a sample hangs; the client runs in the
# background and the runner waits with the `wait` builtin, so that TERM is acted on at once.
# final3: the summary and tarball are written by the EXIT trap too (after llama is healthy), so
# an aborted run still leaves summary.txt and the .tgz.

FINAL=/home/maplemercury/colibri-bench/final13
FINAL2=/home/maplemercury/colibri-bench/final12    # final13: the final12 lock is taken too
OLD_FINAL=/home/maplemercury/colibri-bench/final11  # final13: and the final11 lock
OUT=$FINAL/$EXP
CLIENTS=$FINAL/clients
BUILDS=${BUILDS:-$FINAL/builds}
MODEL=/home/maplemercury/colibri-models/deepseek-v4-flash   # final6: the full 256-expert checkpoint
PORT=8090
LLAMA_UNIT=llama-server-active
LLAMA_HEALTH=http://127.0.0.1:18080/health
EXP_UNIT=colibri-final13            # the runner's own unit (the timer's service; launch line in run_final3.sh)
SERVE_UNIT=colibri-final13-serve
SERVE_ARGS="--model $MODEL --ram 40 --ctx 8192 --host 127.0.0.1 --port $PORT"
# The server gets exactly this environment (env -i): nothing inherited from the
# user manager, so a stray COLI_*/DSV4_*/V4_* variable can never leak in.
BASE_ENV=(PATH=/usr/local/bin:/usr/bin:/bin HOME=/home/maplemercury USER=maplemercury LOGNAME=maplemercury LANG=en_US.UTF-8)
# Same COMMON_ENV as every earlier final run (COLI_CUDA_PIPE=2 is kept for comparability; the
# deepseek_v4 binary never reads it -- only colibri.c does).
# final6: the default configuration the maintainer asked for: the same line WITHOUT COLI_CUDA_MOE_BATCH=1
# (prefill MoE on the CPU union, attention batched on the GPU, where DSV4_MM_WARP's auto shapes run).
COMMON_ENV=(COLI_CUDA=1 COLI_GPU=0 DSV4_CUDA_DEVICE=0 CUDA_DENSE=1 COLI_CUDA_PIPE=2 COLI_CUDA_ATTN_BATCH=1 V4_MOE_REFILL_GROUP=12 V4_LOADER_LANES=3 V4_PREFIX_CKPT_DISK=0)
NVSMI_MS=${NVSMI_MS:-5000}         # nvidia-smi sampling interval per sample (ms); 5 s as in final/
HARD_MARGIN_S=${HARD_MARGIN_S:-120} # final3: the watchdog fires this long before MAX_GPU_S (restore takes ~11 s)
MAKE_LINE="make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j8"

T0=0            # epoch seconds when llama was stopped (start of the GPU window)
NVSMI_PID=      # background nvidia-smi sampler of the running sample
CLIENT_PID=     # final3: background client of the running sample
WATCHDOG_PID=   # final3: hard-cap watchdog
SLEEP_PID=      # final3: the sleep of wait_until
SEQ=0           # global sample order
RESTORED=0
LLAMA_STOPPED=0
FINISHED=0
SUMMARY_KIND=final13
declare -A LAST_ARM_S   # measured wall seconds of the last sample of each arm

log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/run.log"; }
# The engine's expert-usage history (seeds the RAM pin ranking at startup) is
# saved only on a clean engine close; servers stopped by `systemctl stop` never
# close cleanly, so it has stayed empty (0 bytes since 2026-10-12). Recorded
# before/after every sample so a cold sample provably started without history.
usage_state() { stat -c 'size=%s mtime=%Y' "$MODEL/.coli_usage" 2>/dev/null || echo missing; }
now() { date +%s; }
gpu_elapsed() { [ "$T0" -gt 0 ] && echo $(( $(now) - T0 )) || echo 0; }

# ---- locks: one colibri GPU script at a time. Taken BEFORE any trap exists, so a script that
# loses the race exits without touching llama. final3: the final3, final2 AND final/ locks, so no
# final, final2 or final3 script can overlap. A previous $OUT is moved aside, never mixed or deleted.
lock_or_exit() {
    local fd=9 f
    for f in "$FINAL/.gpu.lock" "$FINAL2/.gpu.lock" "$OLD_FINAL/.gpu.lock"; do
        eval "exec $fd>\"\$f\""
        if ! flock -n $fd; then
            echo "another colibri GPU script holds $f; not starting" >&2
            exit 3
        fi
        fd=$((fd - 1))
    done
    if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then
        mv "$OUT" "$OUT.old-$(date +%Y%m%d-%H%M%S)"
    fi
    mkdir -p "$OUT/meta"
}
# final3: release the three locks (the dry preflight holds them only for a moment)
unlock_all() { local fd; for fd in 9 8 7; do flock -u $fd 2>/dev/null; eval "exec $fd>&-"; done; }

die_pre() { echo "PREFLIGHT FAILED: $*" | tee -a "$OUT/run.log" >&2; exit 4; }

# final3: the build log of a tree: $BUILDS, then final3/builds, final2/builds, final/builds
build_log() { # tree
    local n d; n=$(basename "$1")
    for d in "$BUILDS" "$FINAL/builds" "$FINAL2/builds" "$OLD_FINAL/builds"; do
        [ -f "$d/$n.build.log" ] && { echo "$d/$n.build.log"; return; }
    done
    echo "$BUILDS/$n.build.log"
}

# final3: build_tree DIR SHA -- detached worktree of /tmp/colibri at SHA, CUDA build with the make
# line of every earlier final build. A missing DIR is created; an existing DIR is used only when it
# is a clean worktree at exactly SHA (then rebuilt from clean), anything else is refused. Log:
# $BUILDS/<name>.build.log ("tree=... sha=..." first line, "EXIT=<rc>" last line); a result line
# in $BUILDS/<name>.result. A failed build exits (die_pre) before llama is touched.
build_tree() {
    local dir=$1 sha=$2 n blog reuse=0 head warn nvl bsha
    n=$(basename "$dir"); blog=$BUILDS/$n.build.log
    mkdir -p "$BUILDS"
    git -C "$REPO" cat-file -e "$sha^{commit}" 2>/dev/null || die_pre "$sha not in $REPO"
    if [ -e "$dir" ]; then
        head=$(git -C "$dir" rev-parse HEAD 2>/dev/null)
        [ "$head" = "$sha" ] || die_pre "$dir exists at '${head:-not a worktree}', not $sha; not touching it"
        [ -z "$(git -C "$dir" status --porcelain --untracked-files=no)" ] || die_pre "$dir exists with local changes; not touching it"
        reuse=1
        log "build: $dir already exists, clean, at $sha: rebuilding it from clean"
    else
        git -C "$REPO" worktree add --detach "$dir" "$sha" >> "$OUT/run.log" 2>&1 || die_pre "git worktree add $dir $sha failed"
    fi
    log "build: $n at $sha ($MAKE_LINE)"
    (
        cd "$dir/c" || exit 9
        echo "tree=$dir sha=$(git rev-parse HEAD) tree_sha=$(git rev-parse 'HEAD^{tree}') reuse=$reuse start=$(date -Is)"
        [ "$reuse" = 1 ] && { make -f Makefile.deepseek-v4 deepseek-v4-clean CUDA=1 > /dev/null 2>&1; rm -f deepseek_v4 ./*.o; }
        echo "build_env=${BASE_ENV[*]}"
        # shellcheck disable=SC2086
        env -i "${BASE_ENV[@]}" $MAKE_LINE
        echo "EXIT=$? end=$(date -Is)"
    ) > "$blog" 2>&1
    warn=$(grep -iE "warning|error" "$blog" | grep -vE "^lto-wrapper: (warning: using serial compilation|note: see)" | wc -l)
    nvl=$(grep -c '/usr/bin/nvcc' "$blog")
    bsha=$(sha256sum "$dir/c/deepseek_v4" 2>/dev/null | cut -c1-64)
    echo "$n: $(tail -1 "$blog") warn_err_lines=$warn nvcc_lines=$nvl binary_sha256=${bsha:-none}" | tee "$BUILDS/$n.result" | tee -a "$OUT/run.log"
    grep -q '^EXIT=0' "$blog" || die_pre "CUDA build of $sha in $dir failed (see $blog)"
    [ -n "$bsha" ] || die_pre "CUDA build of $sha in $dir left no binary"
    [ "$nvl" -ge 1 ] || die_pre "CUDA build of $sha in $dir did not run nvcc"
    [ "$warn" = 0 ] || log "WARNING: the build of $n has $warn warning/error lines (recorded, not an abort; see $blog)"
}

# preflight_tree DIR SHA: worktree at SHA, clean, built by a logged clean build of SHA, and the
# binary newer than every C/CUDA/header/Makefile source of c/ (final3: all of them, not one file).
preflight_tree() {
    local tree=$1 sha=$2
    [ -d "$tree/c" ] || die_pre "$tree missing"
    local head; head=$(git -C "$tree" rev-parse HEAD 2>/dev/null)
    [ "$head" = "$sha" ] || die_pre "$tree HEAD $head != $sha"
    [ -z "$(git -C "$tree" status --porcelain --untracked-files=no)" ] || die_pre "$tree has local changes"
    [ -x "$tree/c/deepseek_v4" ] || die_pre "$tree/c/deepseek_v4 not built"
    local blog; blog=$(build_log "$tree")
    grep -q "sha=$sha" "$blog" 2>/dev/null && grep -q '^EXIT=0' "$blog" || die_pre "build log $blog not a clean build of $sha"
    local newer
    newer=$(find "$tree/c" -maxdepth 1 -type f \( -name '*.c' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' -o -name 'Makefile*' \) -newer "$tree/c/deepseek_v4" | head -3)
    [ -z "$newer" ] || die_pre "$tree binary older than its sources: $newer"
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

llama_healthy() { curl -s -m 3 "$LLAMA_HEALTH" 2>/dev/null | grep -q -E '"status": *"(ok|healthy)"'; }   # final5: Tabby answers "healthy"

# final3: no colibri unit active except this runner's service and its timer
other_colibri_units() {
    local u
    for u in $(systemctl --user list-units --all --plain --no-legend --state=active,activating,deactivating,reloading 'colibri-*' 2>/dev/null | awk '{print $1}'); do
        case "$u" in "$EXP_UNIT.service"|"$EXP_UNIT.timer") ;; *) echo "$u" ;; esac
    done
}

preflight_common() {
    [ -d "$MODEL" ] || die_pre "model dir missing"
    local f
    for f in "$@"; do [ -f "$CLIENTS/$f" ] || die_pre "client $CLIENTS/$f missing"; done
    if ss -ltnH "sport = :$PORT" | grep -q .; then die_pre "port $PORT already in use"; fi
    if systemctl --user is-active -q "$SERVE_UNIT"; then die_pre "$SERVE_UNIT already active"; fi
    if systemctl --user is-active -q colibri-bench; then die_pre "old colibri-bench unit active"; fi
    local others; others=$(other_colibri_units | tr '\n' ' ')
    [ -z "$others" ] || die_pre "another colibri unit is active: $others"
    systemctl --user is-active -q "$LLAMA_UNIT" || die_pre "$LLAMA_UNIT is not active"
    llama_healthy || die_pre "llama /health is not ok: '$(curl -s -m 3 "$LLAMA_HEALTH")'"
    python3 "$FINAL/final13_tools.py" selftest > "$OUT/meta/selftest.txt" 2>&1 || die_pre "final_tools.py selftest failed (see $OUT/meta/selftest.txt)"
    local free_kb; free_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    log "preflight common ok: MemAvailable=$((free_kb/1024)) MiB, llama $(systemctl --user is-active $LLAMA_UNIT) health ok, port $PORT free, no other colibri unit"
}

# ---- final3: sleep until an epoch second without ever waking early; the sleep runs in the
# background and the runner waits with `wait`, so a TERM is acted on at once.
wait_until() { # epoch
    local t=$1 r
    while :; do
        r=$(( t - $(now) ))
        [ "$r" -le 0 ] && break
        [ "$r" -gt 30 ] && r=30
        sleep "$r" & SLEEP_PID=$!
        wait "$SLEEP_PID"; SLEEP_PID=
    done
}

# ---- restore llama: the EXIT trap (also called explicitly at the end).
stop_sampler() {
    if [ -n "$NVSMI_PID" ]; then kill "$NVSMI_PID" 2>/dev/null; wait "$NVSMI_PID" 2>/dev/null; NVSMI_PID=; fi
}
stop_client() {
    if [ -n "$CLIENT_PID" ]; then kill "$CLIENT_PID" 2>/dev/null; wait "$CLIENT_PID" 2>/dev/null; CLIENT_PID=; fi
}
restore() { # [script status]
    local rc=${1:-$?}
    [ "$RESTORED" = 1 ] && return 0
    RESTORED=1
    trap '' TERM INT HUP    # nothing interrupts the restore
    [ -n "$WATCHDOG_PID" ] && kill "$WATCHDOG_PID" 2>/dev/null
    stop_client
    stop_sampler
    [ -n "$MEM_PID" ] && kill "$MEM_PID" 2>/dev/null
    systemctl --user stop "$SERVE_UNIT" 2>/dev/null
    local i
    for i in $(seq 1 60); do systemctl --user is-active -q "$SERVE_UNIT" || break; sleep 2; done
    log "restore: starting $LLAMA_UNIT (script status $rc, GPU window $(gpu_elapsed) s)"
    systemctl --user start "$LLAMA_UNIT"
    local ok=0
    for i in $(seq 1 120); do llama_healthy && { ok=1; break; }; sleep 5; done
    if [ $ok = 0 ]; then
        log "restore: llama not healthy after 600 s, starting it again"
        systemctl --user reset-failed "$LLAMA_UNIT" 2>/dev/null
        systemctl --user start "$LLAMA_UNIT"
        for i in $(seq 1 60); do llama_healthy && { ok=1; break; }; sleep 5; done
    fi
    local h; h=$(curl -s -m 3 "$LLAMA_HEALTH")
    log "restore: llama health '$h' (ok=$ok) at $(date -Is); llama-down window $(gpu_elapsed) s"
    mkdir -p "$OUT/meta"
    { echo "restored_at=$(date -Is)"; echo "health=$h"; echo "ok=$ok"; echo "gpu_window_s=$(gpu_elapsed)"; echo "script_status=$rc";
      systemctl --user status "$LLAMA_UNIT" --no-pager 2>&1 | head -5;
      nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv; } > "$OUT/meta/llama_after.txt" 2>&1
    return 0
}
# final3: the EXIT trap -- llama first, then (only once llama was stopped) the summary and tarball
on_exit() {
    local rc=$?
    restore "$rc"
    [ "$LLAMA_STOPPED" = 1 ] && [ "$FINISHED" = 0 ] && { log "exit status $rc: writing the summary and tarball of what ran"; summarize_and_pack; }
    return 0
}
arm_trap() {
    trap on_exit EXIT
    trap 'log "signal received, exiting (EXIT trap restores llama)"; exit 143' TERM INT HUP
}

# ---- final3: hard cap. MAX_GPU_S - HARD_MARGIN_S after the llama stop, TERM to the runner (its
# trap exits, the EXIT trap stops the client and the server and restarts llama). The watchdog
# polls every 5 s and ends by itself when the runner is gone.
start_watchdog() {
    local main=$$ deadline=$(( T0 + MAX_GPU_S - HARD_MARGIN_S ))
    (
        trap - EXIT TERM INT HUP
        while kill -0 "$main" 2>/dev/null; do
            if [ "$(date +%s)" -ge "$deadline" ]; then
                echo "[$(date '+%F %T')] HARD CAP: llama has been down $(( $(date +%s) - T0 )) s (cap $MAX_GPU_S s); TERM to the runner $main" >> "$OUT/run.log"
                kill -TERM "$main"
                exit 0
            fi
            sleep 5
        done
    ) &
    WATCHDOG_PID=$!
    log "watchdog $WATCHDOG_PID: TERM at $(date -d "@$deadline" '+%F %T') (llama stop + $((MAX_GPU_S - HARD_MARGIN_S)) s)"
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
    log "stopping $LLAMA_UNIT at $(date '+%F %T.%N') (the EXIT trap restarts it)"
    T0=$(now)
    LLAMA_STOPPED=1
    { echo "llama_stop_epoch=$T0"; echo "llama_stop_at=$(date -Is -d "@$T0")"; } >> "$OUT/meta/llama_before.txt"
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
    echo "  tree_sha=$(git -C "$t" rev-parse 'HEAD^{tree}')"
    echo "  parent=$(git -C "$t" rev-parse HEAD^)"
    echo "  subject=$(git -C "$t" log -1 --format=%s)"
    echo "  status_lines=$(git -C "$t" status --porcelain --untracked-files=no | wc -l)"
    echo "  binary_sha256=$(sha256sum "$t/c/deepseek_v4" | cut -c1-64)"
    echo "  binary_mtime=$(stat -c %y "$t/c/deepseek_v4")"
    echo "  build_log=$(build_log "$t")"
}
capture_meta() { # trees...
    local m=$OUT/meta; mkdir -p "$m"
    {
        echo "exp=$EXP captured_at=$(date -Is) host=$(hostname) runner_pid=$$"
        echo "runner_unit=$(sed 's#.*/##' /proc/self/cgroup)"
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
    cp "$0" "$FINAL/final_common.sh" "$FINAL/final13_tools.py" "$m/" 2>/dev/null
    cp "$CLIENTS"/*.py "$m/" 2>/dev/null
    for t in "$@"; do cp "$(build_log "$t")" "$m/" 2>/dev/null; done
    (cd "$m" && sha256sum ./*.sh ./*.py ./*.build.log > sha256s.txt 2>/dev/null)
    { echo "COMMON_ENV=${COMMON_ENV[*]}"; echo "BASE_ENV=${BASE_ENV[*]}"; echo "SERVE_ARGS=$SERVE_ARGS"; echo "NVSMI_MS=$NVSMI_MS";
      echo "MAX_GPU_S=$MAX_GPU_S"; echo "HARD_MARGIN_S=$HARD_MARGIN_S"; } > "$m/common_env.txt"
    nvidia-smi topo -p2p r > "$m/p2p_read.txt" 2>&1
}

# KEY=VAL... merged left to right, a later KEY replacing an earlier one in place; one per line.
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
#   CLIENT: pr1:first | pr1:both | decode
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
        echo "binary_sha256=$(sha256sum "$tree/c/deepseek_v4" | cut -c1-64)"
        echo "client=$client"; echo "timeout_s=$tmo"
        echo "extra_env=$*"; echo "server_env=${envs[*]}"
        echo "serve_cmd=python3 $tree/c/coli serve $SERVE_ARGS"
        echo "usage_before=$(usage_state)"
        echo "start=$(date -Is)"; echo "start_epoch=$(now)"; echo "gpu_window_at_start_s=$(gpu_elapsed)"
    } > "$d/sample.txt"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv > "$d/gpu_before.csv" 2>&1
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.sm,pcie.link.gen.current,pcie.link.width.current \
        --format=csv -lms "$NVSMI_MS" > "$d/nvsmi.csv" 2>&1 &
    NVSMI_PID=$!
    # final7: OOMScoreAdjust=1000, so a host OOM kills this server first, never another process. Not a
    # MemoryMax: final6's QB unit reported a 15.1G memory peak while the kernel's OOM report counted 44 GB of
    # anon RSS, so the cgroup does not see most of it (likely the Vulkan host pool's driver-pinned pages).
    systemd-run --user --unit="$SERVE_UNIT" --collect -p WorkingDirectory="$tree" -p OOMScoreAdjust=1000 \
        /usr/bin/env -i "${envs[@]}" /bin/bash -c "exec python3 $tree/c/coli serve $SERVE_ARGS > $d/serve.log 2>&1" \
        >> "$OUT/run.log" 2>&1
    mem_sampler "$d" &
    MEM_PID=$!
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
        # final3: the client runs in the background and is waited for, so a TERM (hard cap,
        # systemctl stop) interrupts the wait at once; the EXIT trap then stops the client.
        case "$client" in
            pr1:*)  BENCH_REQS=${client#pr1:} timeout "$tmo" python3 "$CLIENTS/bench_pr1.py" "$d/bench.jsonl" > "$d/bench.out" 2> "$d/bench.err" &
                    CLIENT_PID=$! ;;
            decode) timeout "$tmo" python3 "$CLIENTS/bench_decode.py" > "$d/bench.jsonl" 2> "$d/bench.err" &
                    CLIENT_PID=$! ;;
            f6:*)   BENCH_MODE=${client#f6:} timeout "$tmo" python3 "$CLIENTS/bench_f6.py" "$d/bench.jsonl" > "$d/bench.out" 2> "$d/bench.err" &
                    CLIENT_PID=$! ;;   # final6: the ~8k prompt and the two-at-once requests
            *)      log "unknown client $client"; CLIENT_PID= ;;
        esac
        if [ -n "$CLIENT_PID" ]; then wait "$CLIENT_PID"; crc=$?; CLIENT_PID=; fi
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
    [ -n "$MEM_PID" ] && { kill "$MEM_PID" 2>/dev/null; wait "$MEM_PID" 2>/dev/null; MEM_PID=; }
    # final7: an OOM kill of the server is a result (QB in final6), recorded from the user journal
    echo "oom_kill=$(journalctl --user -u "$SERVE_UNIT" --since "@$s_epoch" --no-pager 2>/dev/null | grep -c 'oom-kill\|killed by the OOM killer')" >> "$d/sample.txt"
    { echo "end=$(date -Is)"; echo "end_epoch=$(now)"; echo "wall_s=$(( $(now) - s_epoch ))"; echo "usage_after=$(usage_state)"; } >> "$d/sample.txt"
    if [ "$(grep '^usage_before=' "$d/sample.txt" | cut -d= -f2-)" != "$(usage_state)" ]; then
        log "WARNING: $MODEL/.coli_usage changed during $arm s$k ($(usage_state)): later samples start with a usage history"
    fi
    LAST_ARM_S[$arm]=$(( $(now) - s_epoch ))
    grep -E "${MODE_RE:-peer-mirrors|moe-batch=|moe-double|mirror-reserve}" "$d/serve.log" 2>/dev/null \
        | grep -v "moe-batch refill" | cut -c1-400 > "$d/mode.txt"
    grep -ciE "out of memory|FAILED|Traceback|error" "$d/serve.log" > "$d/error_count.txt" 2>/dev/null
    log "--- $arm s$k done: client_rc=$crc ready=$ready wall=${LAST_ARM_S[$arm]}s $(tail -c 300 "$d/bench.out" 2>/dev/null | tr '\n' ' ')"
    if [ -f "$d/bench.jsonl" ]; then
        python3 "$FINAL/final13_tools.py" brief "$d/bench.jsonl" 2>&1 | tee -a "$OUT/run.log"
    fi
    return 0
}

# final7: every 5 s, MemAvailable and the engine's VmRSS (kB) -> mem.csv, while the serve unit lives
MEM_PID=
mem_sampler() { # dir
    local d=$1 cg pid rss
    trap - EXIT TERM INT HUP
    echo "epoch,mem_available_kb,engine_rss_kb" > "$d/mem.csv"
    while :; do
        cg=$(systemctl --user show -p ControlGroup --value "$SERVE_UNIT" 2>/dev/null); rss=0
        for pid in $(cat "/sys/fs/cgroup$cg/cgroup.procs" 2>/dev/null); do
            case "$(readlink "/proc/$pid/exe")" in */c/deepseek_v4) rss=$(awk '/^VmRSS/{print $2}' "/proc/$pid/status" 2>/dev/null) ;; esac
        done
        echo "$(date +%s),$(awk '/MemAvailable/{print $2}' /proc/meminfo),${rss:-0}" >> "$d/mem.csv"
        # final10: the engine's memory by class and the box's unevictable / locked pages (kB) -> memx.csv
        local st=; for pid in $(cat "/sys/fs/cgroup$cg/cgroup.procs" 2>/dev/null); do
            case "$(readlink "/proc/$pid/exe")" in */c/deepseek_v4) st=/proc/$pid/status ;; esac
        done
        # final10: the engine's mappings by size once it passes 50 and 55 GiB (which allocations outgrow the plan?)
        if [ -n "$st" ] && [ "${rss:-0}" -gt 52428800 ] && [ ! -s "$d/smaps_50g.txt" ]; then cp "${st%/status}/smaps" "$d/smaps_50g.txt" 2>/dev/null; fi
        if [ -n "$st" ] && [ "${rss:-0}" -gt 57671680 ] && [ ! -s "$d/smaps_55g.txt" ]; then cp "${st%/status}/smaps" "$d/smaps_55g.txt" 2>/dev/null; fi
        # final11: SMAPS_EVERY=n (per sample, from the arm's environment file): a snapshot every n passes
        if [ -n "$st" ] && [ -f "$d/smaps_every" ]; then
            SMAPS_N=$(( ${SMAPS_N:-0} + 1 ))
            [ $(( SMAPS_N % $(cat "$d/smaps_every") )) = 0 ] && cp "${st%/status}/smaps" "$d/smaps_$(date +%s).txt" 2>/dev/null
        fi
        [ -s "$d/memx.csv" ] || echo "epoch,rss_anon,rss_file,vm_lck,vm_pin,mem_free,cached,unevictable,mlocked,anon_pages,mapped" > "$d/memx.csv"
        echo "$(date +%s),$(awk '/^RssAnon/{a=$2}/^RssFile/{f=$2}/^VmLck/{l=$2}/^VmPin/{p=$2}END{printf "%d,%d,%d,%d",a,f,l,p}' "${st:-/dev/null}" 2>/dev/null),$(awk '/^MemFree:/{a=$2}/^Cached:/{b=$2}/^Unevictable:/{c=$2}/^Mlocked:/{d=$2}/^AnonPages:/{e=$2}/^Mapped:/{f=$2}END{printf "%d,%d,%d,%d,%d,%d",a,b,c,d,e,f}' /proc/meminfo)" >> "$d/memx.csv"
        sleep 5
    done
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

# est_for ARM DEFAULT_S: the larger of the default and the arm's last measured sample
est_for() { local m=${LAST_ARM_S[$1]:-0}; [ "$m" -gt "$2" ] && echo "$m" || echo "$2"; }
# over_budget EST_S: would a sample of EST_S seconds overrun MAX_GPU_S (with
# 240 s kept for stopping the last server and restoring llama)?
over_budget() { [ $(( $(gpu_elapsed) + $1 + 240 )) -gt "$MAX_GPU_S" ]; }

# final3: summary + tarball (CPU only; called after llama is back)
summarize_and_pack() {
    [ "$FINISHED" = 1 ] && return 0
    FINISHED=1
    python3 "$FINAL/final13_tools.py" summarize "$SUMMARY_KIND" "$OUT" > "$OUT/summary.txt" 2>&1
    tee -a "$OUT/run.log" < "$OUT/summary.txt"
    log "packing $FINAL/$EXP.tgz"
    tar -C "$FINAL" -czf "$FINAL/$EXP.tgz" "$EXP" && (cd "$FINAL" && sha256sum "$EXP.tgz" > "$EXP.tgz.sha256")
    log "packed $FINAL/$EXP.tgz $(cut -c1-64 "$FINAL/$EXP.tgz.sha256" 2>/dev/null)"
}

# finish: restore llama first, then summarise and pack the evidence.
finish() {
    restore 0
    log "ALL SAMPLES DONE"
    summarize_and_pack
}
