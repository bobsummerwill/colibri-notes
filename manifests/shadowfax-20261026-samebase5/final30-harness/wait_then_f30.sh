#!/bin/bash
# wait_then_f30.sh (2026-10-26): run as user unit cbench-f30 (not colibri-*). Waits until final29's waiter (cbench-f29) has
# ended and llama has answered healthy for ~2 min, then runs final30 (review/PLAN_SAME_BASE_20261026.md, "Investigations
# added": F1 #1766 TTFT ablation, F2 fdec / swapped pr1) like the chain runs its runners. No builds: trees cw22/f27-*.
# hermes is left alone. wait_then_f29.sh's waiter.
L=~/colibri-bench/final30/chain.log
MAX_GPU_S=11200   # final30's arm estimates total 8600 s; +30% = 11180 s (<= 4 h)
log() { echo "[$(date "+%F %T")] $*" >> $L; }
healthy() { systemctl --user is-active -q llama-server-active && curl -s -m 10 http://127.0.0.1:8080/health | grep -q "\"status\": *\"healthy\""; }
wait_healthy() { local ok=0; while [ $ok -lt 8 ]; do if healthy; then ok=$((ok + 1)); else ok=0; fi; sleep 15; done; }
log "cbench-f30 start: waiting for cbench-f29"
while systemctl --user is-active -q cbench-f29; do sleep 120; done
log "cbench-f29 ended ($(tail -1 ~/colibri-bench/final29/chain.log))"
while systemctl --user is-active -q colibri-final29; do log "colibri-final29 still active; waiting"; sleep 120; done
wait_healthy
log "final30: llama healthy, starting (MAX_GPU_S=$MAX_GPU_S)"
systemd-run --user --unit colibri-final30 --collect -E STOP_AT="$(date -d "+2 min" "+%F %T")" -E MAX_GPU_S=$MAX_GPU_S \
    /home/maplemercury/colibri-bench/final30/run_final30.sh >> $L 2>&1
sleep 30
while systemctl --user is-active -q colibri-final30; do sleep 60; done
log "final30: unit ended; last lines: $(tail -2 ~/colibri-bench/final30/bench/run.log | tr "\n" " ")"
log "cbench-f30 done (hermes left stopped for Bob)"
