#!/usr/bin/env python3
"""final30 client fdec: bench_decode.py's first 3 prompts on the full V4 Flash, 300 tokens each, temperature 0 (greedy,
so the texts compare between arms), streaming. One JSON record per prompt on stdout, bench_decode.py's fields
(name, ttft_s, decode_chunks, decode_s, decode_tok_s, total_s, usage, vram, text_head, text) plus position and
text_sha256. Names fdec_0..fdec_2 (not decode_k: bench_decode.py's REAP records sample at 0.7)."""
import hashlib, json, time, urllib.request, sys, subprocess
URL="http://127.0.0.1:8090/v1/chat/completions"
def run(name, prompt, max_tokens, position):
    body=json.dumps({"model":"deepseek-v4-colibri","messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":0,"stream":True}).encode()
    req=urllib.request.Request(URL,data=body,headers={"Content-Type":"application/json"})
    t0=time.time(); first=None; n=0; text=""; usage=None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line=line.decode().strip()
            if not line.startswith("data:") or line=="data: [DONE]": continue
            d=json.loads(line[5:])
            if d.get("usage"): usage=d["usage"]
            for c in d.get("choices",[]):
                delta=c.get("delta",{}).get("content") or c.get("delta",{}).get("reasoning_content") or ""
                if delta:
                    if first is None: first=time.time()
                    n+=1; text+=delta
    t1=time.time()
    vram=subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader"],capture_output=True,text=True).stdout.split()
    res={"name":name,"position":position,"ttft_s":round(first-t0,2) if first else None,"decode_chunks":n,"decode_s":round(t1-first,2) if first else None,
         "decode_tok_s":round((n-1)/(t1-first),3) if first and n>1 else None,"total_s":round(t1-t0,2),"usage":usage,"vram":vram,"text_head":text[:200],"text":text,
         "text_sha256":hashlib.sha256(text.encode()).hexdigest()}
    print(json.dumps(res), flush=True)
    print(name, res["ttft_s"], res["decode_tok_s"], n, res["text_sha256"][:12], file=sys.stderr, flush=True)
# bench_decode.py's PROMPTS[:3], verbatim
PROMPTS=['Write a detailed explanation of how a refrigerator works, covering the thermodynamic cycle, the role of the compressor, condenser, expansion valve and evaporator.', 'Write a short story about a lighthouse keeper who discovers a message in a bottle from the future.', 'Explain, step by step, how to implement a hash map in C with open addressing and linear probing, including resizing.']
for k,p in enumerate(PROMPTS):
    run("fdec_%d"%k, p, 300, k+1)
