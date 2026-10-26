#!/bin/bash
# chain_f27_f28.sh (2026-10-26): the same-base run restarted on dev c1c06ac6. final27 (B/M/P, Stage 1) then final28
# (C+, same-session anchors, #1766 switch ablation; Stage 2), back to back. Each runner stops llama itself and restores it
# at its end; between the two, wait until llama has answered healthy for ~2 min. hermes is left alone (stopped by Bob).
# Run as a user unit named cbench-chain2 (not colibri-*, which the runners' preflight refuses).
L=~/colibri-bench/final28/chain.log
log() { echo "[$(date "+%F %T")] $*" >> $L; }
healthy() { systemctl --user is-active -q llama-server-active && curl -s -m 10 http://127.0.0.1:8080/health | grep -q "\"status\": *\"healthy\""; }
wait_healthy() { local ok=0; while [ $ok -lt 8 ]; do if healthy; then ok=$((ok + 1)); else ok=0; fi; sleep 15; done; }
run_one() { # name runner [env...]
    local n=$1 r=$2; shift 2
    wait_healthy
    log "$n: llama healthy, starting"
    systemd-run --user --unit colibri-$n --collect -E STOP_AT="$(date -d "+2 min" "+%F %T")" "$@" "$r" >> $L 2>&1
    sleep 30
    while systemctl --user is-active -q colibri-$n; do sleep 60; done
    log "$n: unit ended; last lines: $(tail -2 ~/colibri-bench/$n/bench/run.log | tr "\n" " ")"
}
log "chain2 start"
run_one final27 /home/maplemercury/colibri-bench/final27/run_final27.sh -E MAX_GPU_S=14400
run_one final28 /home/maplemercury/colibri-bench/final28/run_final28.sh -E MAX_GPU_S=15000
log "chain2 done (hermes left stopped for Bob)"
