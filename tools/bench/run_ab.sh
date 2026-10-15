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
  systemd-run --user --unit=colibri-bench --collect -p WorkingDirectory=/tmp/colibri \
    -E COLI_CUDA=1 -E COLI_GPU=0 -E DSV4_CUDA_DEVICE=0 -E CUDA_DENSE=1 -E COLI_CUDA_PIPE=2 \
    -E COLI_CUDA_ATTN_BATCH=1 -E COLI_CUDA_MOE_BATCH=1 -E V4_MOE_REFILL_GROUP=12 -E V4_LOADER_LANES=3 \
    -E V4_PREFIX_CKPT_DISK=0 "$@" \
    /bin/bash -c "exec python3 /tmp/colibri/c/coli serve --model /home/maplemercury/colibri-models/deepseek-v4-flash-reap --ram 40 --ctx 8192 --host 127.0.0.1 --port 8090 > $log 2>&1"
  until grep -q "listening on" "$log" 2>/dev/null || ! systemctl --user is-active -q colibri-bench; do sleep 3; done
}

systemctl --user stop llama-server-active.service; sleep 3

echo "=== RUN A: sized bank, default reserve"
serve $B/serve-A.log
(cd $B && timeout 1500 python3 bench_reap.py > benchA.jsonl 2> benchA.err)
cat $B/benchA.jsonl $B/benchA.err
grep -E "moe-batch=|out of memory|bank alloc" $B/serve-A.log

echo "=== RUN B: profiled long_prefill (cold)"
python3 - <<'EOF'
s = open("/home/maplemercury/colibri-bench/bench_reap.py").read()
i = s.index("filler2=")
open("/home/maplemercury/colibri-bench/bench_prof.py", "w").write(s[:i])
EOF
serve $B/serve-B.log -E DSV4_ATTN_PROF=1 -E DSV4_CUDA_MOE_PROF=1
(cd $B && timeout 1500 python3 bench_prof.py > benchB.jsonl 2> benchB.err)
cat $B/benchB.jsonl $B/benchB.err
wc -l $B/serve-B.log
