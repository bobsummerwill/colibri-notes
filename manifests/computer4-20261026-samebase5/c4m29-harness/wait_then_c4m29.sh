#!/bin/bash
# wait_then_c4m29.sh (2026-10-26): run as user unit c4m29-wait. Waits until c4m28's waiter (c4m28-wait) has ended and no
# c4m28 unit is active, then runs c4m29 (F1 on the 5070: B, M, MX with bench_pr1.py; no builds, trees dg/s27base, s27m)
# as unit c4m29. wait_then_c4m28.sh's waiter.
L=~/colibri-bench/c4m29/wait.log
log() { echo "[$(date "+%F %T")] $*" >> $L; }
log "c4m29-wait start: waiting for c4m28-wait and c4m28"
while systemctl --user is-active -q c4m28-wait || systemctl --user is-active -q c4m28; do sleep 120; done
log "c4m28 ended ($(tail -1 ~/colibri-bench/c4m28/wait.log 2>/dev/null))"
systemd-run --user --unit c4m29 --collect /home/zombietiger/colibri-bench/c4m29/run_c4m29.sh >> $L 2>&1
sleep 30
while systemctl --user is-active -q c4m29; do sleep 60; done
log "c4m29 ended: $(tail -3 ~/colibri-bench/c4m29/bench/run.log 2>/dev/null | tr '\n' ' ')"
