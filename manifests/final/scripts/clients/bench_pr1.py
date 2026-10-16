#!/usr/bin/env python3
"""PR #1 benchmark client: fixed prompts, greedy, records TTFT, decode rate and full text."""
import hashlib, json, os, sys, time, urllib.request

URL = "http://127.0.0.1:8090/v1/chat/completions"
OUT = sys.argv[1]

filler = " ".join(["The quick brown fox jumps over the lazy dog while the committee reviews section %d of the quarterly infrastructure report." % i for i in range(90)])
filler2 = " ".join(["Station %d logged a temperature of %d degrees and a wind speed of %d knots during the overnight survey." % (i, 10 + i % 17, 3 + i % 11) for i in range(130)])
REQUESTS = [
    ("long_prefill", filler + "\n\nSummarize the above in one sentence.", 100, 1992),
    ("long_prefill_2", filler2 + "\n\nWhich station had the highest wind speed? Answer briefly.", 60, 3005),
]
if os.environ.get("BENCH_REQS") == "first":
    REQUESTS = REQUESTS[:1]


def run(name, prompt, max_tokens, prompt_tokens, position):
    body = json.dumps({"model": "deepseek-v4-colibri", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; n = 0; text = ""
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            for c in d.get("choices", []):
                delta = c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content") or ""
                if delta:
                    if first is None:
                        first = time.time()
                    n += 1; text += delta
    t1 = time.time()
    ttft = first - t0
    return {"name": name, "position": position, "prompt_tokens": prompt_tokens, "ttft_s": round(ttft, 3),
            "prefill_tok_s": round(prompt_tokens / ttft, 4), "decode_chunks": n,
            "decode_tok_s": round((n - 1) / (t1 - first), 4) if n > 1 else None,
            "total_s": round(t1 - t0, 3), "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


with open(OUT, "w") as f:
    for pos, (name, prompt, mt, pt) in enumerate(REQUESTS, 1):
        rec = run(name, prompt, mt, pt, pos)
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(name, rec["ttft_s"], rec["decode_tok_s"], rec["text_sha256"][:12], flush=True)
