#!/bin/bash
# wait_then_f29.sh (2026-10-26): run as user unit cbench-f29 (not colibri-*). Waits until the final27/final28 chain
# (cbench-chain2) has ended and llama has answered healthy for ~2 min, then builds the split trees (build_f29.sh) and
# runs final29 like the chain runs its runners. hermes is left alone.
L=~/colibri-bench/final29/chain.log
log() { echo "[$(date "+%F %T")] $*" >> $L; }
healthy() { systemctl --user is-active -q llama-server-active && curl -s -m 10 http://127.0.0.1:8080/health | grep -q "\"status\": *\"healthy\""; }
wait_healthy() { local ok=0; while [ $ok -lt 8 ]; do if healthy; then ok=$((ok + 1)); else ok=0; fi; sleep 15; done; }
log "cbench-f29 start: waiting for cbench-chain2"
while systemctl --user is-active -q cbench-chain2; do sleep 120; done
log "cbench-chain2 ended ($(tail -1 ~/colibri-bench/final28/chain.log))"
wait_healthy
log "llama healthy; building"
/home/maplemercury/colibri-bench/final29/build_f29.sh
log "builds: $(tr '\n' ' ' < ~/colibri-bench/final29/builds/status)"
grep -q "EXIT=0" ~/colibri-bench/final29/builds/f29-s1.build.log && grep -q "EXIT=0" ~/colibri-bench/final29/builds/f29-s2.build.log &&
    grep -q "EXIT=0" ~/colibri-bench/final29/builds/f29-s3.build.log || { log "a build failed: final29 not started"; exit 1; }
wait_healthy
log "final29: llama healthy, starting"
systemd-run --user --unit colibri-final29 --collect -E STOP_AT="$(date -d "+2 min" "+%F %T")" -E MAX_GPU_S=7200 \
    /home/maplemercury/colibri-bench/final29/run_final29.sh >> $L 2>&1
sleep 30
while systemctl --user is-active -q colibri-final29; do sleep 60; done
log "final29: unit ended; last lines: $(tail -2 ~/colibri-bench/final29/bench/run.log | tr "\n" " ")"
log "cbench-f29 done (hermes left stopped for Bob)"
