#!/bin/bash
# PR #1 benchmark on shadowfax. Stops llama.cpp once, ALWAYS restores it on exit.
# Arms (all: --ram 40 --ctx 8192 and the common GPU env below; cold server per sample):
#   A = upstream dev 7b55a9b3, B = pr1/fixed 48767c01, C = B + COLI_CUDA_MOE_DOUBLE=1,
#   E = B + DSV4_MOE_EXPERT_MAJOR=0 (first request only), F = B + DSV4_MM_WARP=0 (first request only)
# Samples interleaved across arms. Finally one verify run (pr1, both VERIFY modes, first request).
B=~/colibri-bench/pr1final
mkdir -p $B
restore() {
  systemctl --user stop colibri-bench.service 2>/dev/null; sleep 3
  systemctl --user start llama-server-active.service
  for i in $(seq 1 40); do s=$(curl -s -m 3 http://127.0.0.1:18080/health); echo "$s" | grep -q ok && { echo "restore: llama health $s"; break; }; sleep 5; done
}
trap restore EXIT
COMMON="-E COLI_CUDA=1 -E COLI_GPU=0 -E DSV4_CUDA_DEVICE=0 -E CUDA_DENSE=1 -E COLI_CUDA_PIPE=2 -E COLI_CUDA_ATTN_BATCH=1 -E COLI_CUDA_MOE_BATCH=1 -E V4_MOE_REFILL_GROUP=12 -E V4_LOADER_LANES=3 -E V4_PREFIX_CKPT_DISK=0"
serve() { # $1 tree, $2 log, rest extra -E
  local tree=$1 log=$2; shift 2
  systemctl --user stop colibri-bench.service 2>/dev/null; sleep 2
  systemd-run --user --unit=colibri-bench --collect -p WorkingDirectory=$tree $COMMON "$@" \
    /bin/bash -c "exec python3 $tree/c/coli serve --model /home/maplemercury/colibri-models/deepseek-v4-flash-reap --ram 40 --ctx 8192 --host 127.0.0.1 --port 8090 > $log 2>&1"
  until grep -q "listening on" "$log" 2>/dev/null || ! systemctl --user is-active -q colibri-bench; do sleep 3; done
}
sample() { # $1 arm, $2 k, $3 tree, $4 reqs, rest extra -E
  local arm=$1 k=$2 tree=$3 reqs=$4; shift 4
  local d=$B/$arm/s$k; mkdir -p $d
  echo "=== $arm sample $k ($(date +%T))"
  serve $tree $d/serve.log "$@"
  BENCH_REQS=$reqs timeout 3000 python3 ~/colibri-bench/bench_pr1.py $d/bench.jsonl 2> $d/bench.err
  grep -hE "moe-batch=|out of memory|FAILED|rror" $d/serve.log | sort | uniq -c | head -5
}
systemctl --user stop llama-server-active.service; sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader
PR1=/tmp/cw/bench-pr1f
for k in 1 2 3; do
  sample Bf $k $PR1 both
  sample Cf $k $PR1 both -E COLI_CUDA_MOE_DOUBLE=1
done
echo "ALL DONE"
