#!/bin/bash
# Runs on shadowfax as maplemercury. Always restores llama.cpp on exit.
B=~/colibri-bench
restore() {
  systemctl --user stop colibri-bench.service 2>/dev/null
  sleep 3
  systemctl --user start llama-server-active.service
  for i in $(seq 1 40); do
    s=$(curl -s -m 3 http://127.0.0.1:18080/health)
    echo "$s" | grep -q ok && { echo "restore: llama health $s"; break; }
    sleep 5
  done
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
}
trap restore EXIT

serve() { # $1 = log file, rest = extra -E args
  local log=$1; shift
  systemctl --user stop colibri-bench.service 2>/dev/null; sleep 2
  systemd-run --user --unit=colibri-bench --collect -p WorkingDirectory=${COLI_DIR:-/tmp/colibri} \
    -E COLI_CUDA=1 -E COLI_GPU=0 -E DSV4_CUDA_DEVICE=0 -E CUDA_DENSE=1 -E COLI_CUDA_PIPE=2 \
    -E COLI_CUDA_ATTN_BATCH=1 -E COLI_CUDA_MOE_BATCH=1 -E V4_MOE_REFILL_GROUP=12 -E V4_LOADER_LANES=3 \
    -E V4_PREFIX_CKPT_DISK=0 "$@" \
    /bin/bash -c "exec $NSYS_PREFIX python3 ${COLI_DIR:-/tmp/colibri}/c/coli serve --model /home/maplemercury/colibri-models/deepseek-v4-flash-reap --ram ${COLI_RAM:-40} --ctx 8192 --host 127.0.0.1 --port 8090 > $log 2>&1"
  until grep -q "listening on" "$log" 2>/dev/null || ! systemctl --user is-active -q colibri-bench; do sleep 3; done
}

# usage: run_x.sh LABEL BENCH.py [extra -E args...]
L=$1; BENCH=$2; shift 2
systemctl --user stop llama-server-active.service; sleep 3
echo "=== RUN $L"
serve $B/serve-$L.log "$@"
(cd $B && timeout 2400 python3 $BENCH > bench-$L.jsonl 2> bench-$L.err)
if [ -n "$NSYS_PREFIX" ]; then nsys stop --session=colibri 2>&1 | tail -2; for i in $(seq 1 60); do ls $B/nsys-$L.nsys-rep >/dev/null 2>&1 && break; sleep 5; done; ls -la $B/nsys-$L.nsys-rep; fi
cat $B/bench-$L.jsonl $B/bench-$L.err
grep -E "moe-batch=|out of memory|bank alloc" $B/serve-$L.log
