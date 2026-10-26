#!/bin/bash
# igpu_subgroup_20261026.sh (counter): decode speed on the Intel Meteor Lake iGPU (Mesa ANV) with the Vulkan chain, with
# the pipelines pinned at subgroup size 16 (COLI_VK_SUBGROUP=16, #2404) vs the device's reported 32 (unset, dev's pin).
# OLMoE-1B-7B-0924-Instruct Q4_K_M (allenai's GGUF, imported), the #2404 tip d0765490 built with VK=1, coli serve +
# bench_decode.py (6 prompts x 300 tokens, streamed), 3 client runs per server, servers in the order 16, 32, 32, 16:
# each start's time to "listening" shows whether the chain_moe.spv compile at 32 (38+ min in test_vk_chain) repeats.
# Waits for counter's colibri-recheck2 so the CPU is quiet.
set -u
while systemctl --user is-active -q colibri-recheck2; do sleep 60; done
T=/home/unagi/Projects/wt-r-sg; M=/home/unagi/colibri-notes/manifests/counter-20261026-igpu; mkdir -p "$M"
MODEL=/home/unagi/colibri-models/olmoe/olmoe-q4km; PORT=8090
log() { echo "[$(date '+%F %T')] $*" | tee -a "$M/run.log"; }
{ echo "tree=$T head=$(git -C $T rev-parse HEAD) binary=$(sha256sum $T/c/olmoe | cut -c1-64)"; cat /home/unagi/colibri-models/olmoe/olmoe.sha256
  vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverInfo|apiVersion" | head -4; lscpu | grep "Model name"; free -g | head -2; } > "$M/host.txt"
n=0
for sg in 16 32 32 16; do
    n=$((n + 1)); d=$M/s$n-sg$sg; mkdir -p "$d"
    if [ $sg = 32 ]; then X=""; else X="COLI_VK_SUBGROUP=$sg"; fi
    log "=== server $n: subgroup $sg (${X:-unset: the reported 32})"
    t0=$(date +%s)
    ( cd $T/c && exec env VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.json COLI_VULKAN=1 COLI_VK_CHAIN=1 $X \
        python3 coli serve --model $MODEL --ram 16 --ctx 4096 --host 127.0.0.1 --port $PORT ) > "$d/serve.log" 2>&1 &
    S=$!
    ready=0
    for i in $(seq 1 1800); do grep -q "listening on" "$d/serve.log" 2>/dev/null && { ready=1; break; }; kill -0 $S 2>/dev/null || break; sleep 2; done
    log "server $n ready=$ready after $(( $(date +%s) - t0 )) s"
    if [ $ready = 1 ]; then
        for r in 1 2 3; do
            timeout 3600 python3 /home/unagi/colibri-notes/tools/bench_decode_anymodel.py > "$d/bench-r$r.jsonl" 2> "$d/bench-r$r.err"
            log "server $n run $r rc=$? $(python3 -c "
import json,sys,statistics,hashlib
rs=[json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
d=[r['decode_tok_s'] for r in rs if r.get('decode_tok_s')]
print('rows=%d mean_decode=%.2f tok/s median_ttft=%.2f s texts=%s'%(len(rs),statistics.mean(d) if d else 0,statistics.median([r['ttft_s'] for r in rs]) if rs else 0,' '.join(hashlib.sha256(r['text'].encode()).hexdigest()[:8] for r in rs)))" "$d/bench-r$r.jsonl" 2>&1)"
        done
    fi
    kill -TERM $S 2>/dev/null; sleep 5; pkill -KILL -x olmoe 2>/dev/null
    for i in $(seq 1 30); do ss -ltnH "sport = :$PORT" | grep -q . || break; sleep 2; done
done
log "ALL DONE"
