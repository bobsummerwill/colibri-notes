#!/bin/bash
# chain_f22_f23.sh (2026-10-25 night): shadowfax is taken by a hermes bakeoff until its 05:30 deadline (failsafe restore
# 05:44). Wait until no bakeoff runs and production llama has been healthy for 10 minutes, then run final22, then final23,
# hermes stays paused (Bob brings it back). Each runner stops llama itself and restores it at its end.
L=~/colibri-bench/final23/chain.log
log() { echo "[$(date "+%F %T")] $*" >> $L; }
healthy() { systemctl --user is-active -q llama-server-active && curl -s -m 10 http://127.0.0.1:8080/health | grep -q "\"status\": *\"healthy\""; }
wait_clear() { # a stable production lane and no bakeoff
    local ok=0
    while :; do
        if pgrep -f "[b]akeoff/bakeoff.py run" >/dev/null || ! healthy; then ok=0; else ok=$((ok + 1)); fi
        [ $ok -ge 20 ] && return; sleep 30
    done
}
run_one() { # name runner [extra env]
    local n=$1 r=$2; shift 2
    wait_clear
    log "$n: lane clear, starting"
    systemd-run --user --unit colibri-$n --collect -E STOP_AT="$(date -d "+2 min" "+%F %T")" "$@" "$r" >> $L 2>&1
    sleep 30
    while systemctl --user is-active -q colibri-$n; do sleep 60; done
    log "$n: unit ended; last lines: $(tail -2 ~/colibri-bench/$n/bench/run.log | tr "\n" " ")"
}
log "chain start"
run_one final22 /home/maplemercury/colibri-bench/final22/run_final22.sh
run_one final23 /home/maplemercury/colibri-bench/final23/run_final23.sh -E MAX_GPU_S=14400
log "chain done (hermes left paused for Bob: hermes resume)"
