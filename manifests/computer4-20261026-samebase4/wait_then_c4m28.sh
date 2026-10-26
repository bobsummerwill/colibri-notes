#!/bin/bash
# wait_then_c4m28.sh (2026-10-26): run as user unit c4m28-wait. Waits for c4m27 to end, builds the split trees
# (dg/build28.sh), then runs c4m28 (the reserve c4m27's pilot decided) as unit c4m28.
L=~/colibri-bench/c4m28/wait.log
log() { echo "[$(date "+%F %T")] $*" >> $L; }
log "c4m28-wait start: waiting for c4m27"
while systemctl --user is-active -q c4m27; do sleep 120; done
log "c4m27 ended ($(tail -1 ~/colibri-bench/c4m27/bench/run.log 2>/dev/null))"
~/colibri-bench/dg/build28.sh
log "builds: $(tr '\n' ' ' < ~/colibri-bench/dg/logs/build28.status)"
for t in s28s1 s28s2 s28s3; do grep -q "^EXIT=0" ~/colibri-bench/dg/logs/build28-$t.log || { log "$t build failed: c4m28 not started"; exit 1; }; done
systemd-run --user --unit c4m28 --collect /home/zombietiger/colibri-bench/c4m28/run_c4m28.sh >> $L 2>&1
sleep 30
while systemctl --user is-active -q c4m28; do sleep 60; done
log "c4m28 ended: $(tail -3 ~/colibri-bench/c4m28/bench/run.log 2>/dev/null | tr '\n' ' ')"
