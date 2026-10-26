import json, time, urllib.request, sys, subprocess
# bench_decode_anymodel.py: bench_decode.py (shadowfax final runs) with the model id taken from the server's /v1/models
URL="http://127.0.0.1:8090/v1/chat/completions"
MODEL=json.load(urllib.request.urlopen("http://127.0.0.1:8090/v1/models", timeout=60))["data"][0]["id"]
def run(name, prompt, max_tokens):
    body=json.dumps({"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":0.7,"stream":True}).encode()
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
PROMPTS=['Write a detailed explanation of how a refrigerator works, covering the thermodynamic cycle, the role of the compressor, condenser, expansion valve and evaporator.', 'Write a short story about a lighthouse keeper who discovers a message in a bottle from the future.', 'Explain, step by step, how to implement a hash map in C with open addressing and linear probing, including resizing.', 'Describe the causes and consequences of the 2008 global financial crisis for a general audience.', 'Write a guide for a beginner gardener on growing tomatoes on a balcony, from seed to harvest.', 'Explain how public-key cryptography works, using RSA as an example, and why it is hard to break.']
for k,p in enumerate(PROMPTS):
    run("decode_%d"%k, p, 300)
