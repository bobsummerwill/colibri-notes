#!/bin/bash
# Double-bank nondeterminism probe on shadowfax. Stops llama.cpp once, ALWAYS restores it.
B=~/colibri-bench/dbldebug2
mkdir -p $B
restore() {
  systemctl --user stop colibri-bench.service 2>/dev/null; sleep 3
  systemctl --user start llama-server-active.service
  for i in $(seq 1 40); do s=$(curl -s -m 3 http://127.0.0.1:18080/health); echo "$s" | grep -q ok && { echo "restore: llama health $s"; break; }; sleep 5; done
}
trap restore EXIT
COMMON="-E COLI_CUDA=1 -E COLI_GPU=0 -E DSV4_CUDA_DEVICE=0 -E CUDA_DENSE=1 -E COLI_CUDA_PIPE=2 -E COLI_CUDA_ATTN_BATCH=1 -E COLI_CUDA_MOE_BATCH=1 -E V4_MOE_REFILL_GROUP=12 -E V4_LOADER_LANES=3 -E V4_PREFIX_CKPT_DISK=0"
serve() {
  local tree=$1 log=$2; shift 2
  systemctl --user stop colibri-bench.service 2>/dev/null; sleep 2
  systemd-run --user --unit=colibri-bench --collect -p WorkingDirectory=$tree $COMMON "$@" \
    /bin/bash -c "exec python3 $tree/c/coli serve --model /home/maplemercury/colibri-models/deepseek-v4-flash-reap --ram 40 --ctx 8192 --host 127.0.0.1 --port 8090 > $log 2>&1"
  until grep -q "listening on" "$log" 2>/dev/null || ! systemctl --user is-active -q colibri-bench; do sleep 3; done
}
sample() { # arm k order extra...
  local arm=$1 k=$2 order=$3; shift 3
  local d=$B/$arm/s$k; mkdir -p $d
  echo "=== $arm sample $k ($(date +%T))"
  serve /tmp/cw/dbg-pr1 $d/serve.log "$@"
  BENCH_REQS=$order timeout 3000 python3 ~/colibri-bench/bench_pr1b.py $d/bench.jsonl 2> $d/bench.err
}
systemctl --user stop llama-server-active.service; sleep 5
sample Bdbg 1 both -E V4_LOGIT_LOG=1
for k in 1 2 3; do sample Cdbg $k both -E COLI_CUDA_MOE_DOUBLE=1 -E V4_LOGIT_LOG=1; done
for k in 1 2; do sample Cblk $k both -E COLI_CUDA_MOE_DOUBLE=1 -E CUDA_LAUNCH_BLOCKING=1 -E V4_LOGIT_LOG=1; done
for k in 1 2; do sample Cfirst $k second -E COLI_CUDA_MOE_DOUBLE=1 -E V4_LOGIT_LOG=1; done
echo "ALL DONE"
