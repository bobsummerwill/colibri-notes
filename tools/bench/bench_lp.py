#!/usr/bin/env python3
"""Determinism probe: same prompts as bench_pr1.py, non-streamed, with per-token logprobs."""
import hashlib, json, os, sys, time, urllib.request

URL = "http://127.0.0.1:8090/v1/chat/completions"
OUT = sys.argv[1]
filler = " ".join(["The quick brown fox jumps over the lazy dog while the committee reviews section %d of the quarterly infrastructure report." % i for i in range(90)])
filler2 = " ".join(["Station %d logged a temperature of %d degrees and a wind speed of %d knots during the overnight survey." % (i, 10 + i % 17, 3 + i % 11) for i in range(130)])
REQ = [("long_prefill", filler + "\n\nSummarize the above in one sentence.", 100),
       ("long_prefill_2", filler2 + "\n\nWhich station had the highest wind speed? Answer briefly.", 60)]
if os.environ.get("BENCH_ORDER") == "second_only":
    REQ = REQ[1:]

with open(OUT, "w") as f:
    for name, prompt, mt in REQ:
        body = json.dumps({"model": "deepseek-v4-colibri", "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": mt, "temperature": 0, "logprobs": True, "top_logprobs": 2}).encode()
        req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r:
            d = json.loads(r.read())
        ch = d["choices"][0]
        text = ch["message"].get("content") or ""
        lp = ch.get("logprobs") or {}
        toks = [(c.get("token"), c.get("logprob"), [(t.get("token"), t.get("logprob")) for t in (c.get("top_logprobs") or [])]) for c in (lp.get("content") or [])]
        rec = {"name": name, "total_s": round(time.time() - t0, 2), "text": text,
               "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "tokens": toks}
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(name, rec["total_s"], rec["text_sha256"][:12], len(toks), flush=True)
