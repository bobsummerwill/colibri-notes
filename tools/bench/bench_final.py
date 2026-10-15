import json, time, urllib.request, sys, subprocess
URL="http://127.0.0.1:8090/v1/chat/completions"
def run(name, prompt, max_tokens):
    body=json.dumps({"model":"deepseek-v4-colibri","messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":0.7,"stream":True}).encode()
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
    res={"name":name,"ttft_s":round(first-t0,2) if first else None,"decode_chunks":n,"decode_s":round(t1-first,2) if first else None,
         "decode_tok_s":round((n-1)/(t1-first),3) if first and n>1 else None,"total_s":round(t1-t0,2),"usage":usage,"vram":vram,"text_head":text[:200],"text":text}
    print(json.dumps(res), flush=True)
short="Explain the memory hierarchy trick that MoE inference engines use to run very large models on consumer hardware, in 3 sentences."
run("short_cold", short, 200)
run("short_warm", short, 200)
filler=" ".join(["The quick brown fox jumps over the lazy dog while the committee reviews section %d of the quarterly infrastructure report."%i for i in range(90)])
run("long_prefill", filler+"\n\nSummarize the above in one sentence.", 100)
filler2=" ".join(["Station %d logged a temperature of %d degrees and a wind speed of %d knots during the overnight survey."%(i,10+i%17,3+i%11) for i in range(130)])
run("long_prefill_2", filler2+"\n\nWhich station had the highest wind speed? Answer briefly.", 60)
