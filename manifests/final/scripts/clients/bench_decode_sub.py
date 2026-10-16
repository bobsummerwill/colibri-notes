#!/usr/bin/env python3
"""Drift phase-2 client: the first N prompts of bench_decode.py, same requests.

The prompts are read from bench_decode.py next to this file (ast, never
executed), so they are byte-identical to phase 1. Each request is exactly
bench_decode.py's: model deepseek-v4-colibri, one user message, max_tokens 300,
temperature 0.7 (ignored: the engine is greedy), streamed. One JSON line per
request in the output file, flushed as it completes, with the full text and
its sha256.

usage: BENCH_NPROMPTS=N bench_decode_sub.py OUT.jsonl   (N default 6)
"""
import ast, hashlib, json, os, sys, time, urllib.request

URL = "http://127.0.0.1:8090/v1/chat/completions"
HERE = os.path.dirname(os.path.abspath(__file__))


def load_prompts():
    tree = ast.parse(open(os.path.join(HERE, "bench_decode.py")).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PROMPTS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("PROMPTS not found in bench_decode.py")


def run(name, prompt, max_tokens):
    body = json.dumps({"model": "deepseek-v4-colibri", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.7, "stream": True}).encode()
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
    return {"name": name, "ttft_s": round(first - t0, 2) if first else None, "decode_chunks": n,
            "decode_s": round(t1 - first, 2) if first else None,
            "decode_tok_s": round((n - 1) / (t1 - first), 3) if first and n > 1 else None,
            "total_s": round(t1 - t0, 2), "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


def main():
    prompts = load_prompts()
    n = int(os.environ.get("BENCH_NPROMPTS", len(prompts)))
    with open(sys.argv[1], "w") as f:
        for k, p in enumerate(prompts[:max(1, min(n, len(prompts)))]):
            rec = run("decode_%d" % k, p, 300)
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(rec["name"], rec["decode_tok_s"], rec["text_sha256"][:12], flush=True)


if __name__ == "__main__":
    main()
