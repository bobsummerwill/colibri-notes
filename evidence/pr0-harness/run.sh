#!/bin/bash
# GPU0 window for the PR #0 harnesses on the rebased stack. Stops llama, ALWAYS restores it.
H=~/colibri-bench/pr0-harness-20261018; cd $H; exec > $H/run.log 2>&1
restore() { systemctl --user start llama-server-active; for i in $(seq 60); do curl -sf -m 3 127.0.0.1:18080/health && { echo " restored $(date +%T)"; return; }; sleep 5; done; echo "RESTORE FAILED"; }
others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d " " | grep -vx "$(systemctl --user show -p MainPID --value llama-server-active)" | grep -v "^$")
[ -z "$others" ] || { echo "abort: other GPU processes $others"; exit 1; }
trap restore EXIT; trap "exit 143" TERM INT
echo "start $(date -Is)"; systemctl --user stop llama-server-active; sleep 3
for k in 62eaa9dc ce2d4eff 760cb44a; do
  echo "== $k"
  CUDA_VISIBLE_DEVICES=0 timeout 600 ./bin/h2d_race-$k 300 20 2>&1 | grep -v "device 0" | tail -1
  CUDA_VISIBLE_DEVICES=0 timeout 120 ./bin/bank_lasterr-$k 2>&1 | grep -v "device 0" | tail -1
  CUDA_VISIBLE_DEVICES=0 timeout 120 ./bin/shutdown_reinit-$k 2>&1 | grep -v "device 0" | tail -2 | tr "\n" " "; echo
done
echo "end $(date -Is)"
