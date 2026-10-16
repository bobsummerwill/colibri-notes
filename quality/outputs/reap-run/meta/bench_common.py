"""Shared pieces of the final2 clients (bench_f42.py, bench_race.py). Imported, never run.

Prompts are taken from the final/ clients' source with ast (never executed), so they are
byte-identical to the earlier runs, and checked against fixed sha256s:
  FILLER        bench_pr1.py's `filler` (the 1,992-token prompt of the PR #1 benchmark without its
                last line "\\n\\nSummarize the above in one sentence.")
  SHORT_PROMPT  bench_decode.py PROMPTS[0] (refrigerator)
The 1,992-token request itself cannot give a 200-token answer: greedy decoding ends it with EOS
after 24 tokens whatever max_tokens says (the server has no ignore_eos), so the long requests here
keep the filler byte for byte and replace only the closing question with LONG_Q, which asks for a
long answer.
"""
import ast, hashlib, json, os, time, urllib.request

URL = "http://127.0.0.1:8090/v1/chat/completions"
HERE = os.path.dirname(os.path.abspath(__file__))
FILLER_SHA256 = "06301a34d3a71ffd979dbed0013f6b18b2736326dc0c86712b2590965407386f"
SHORT_SHA256 = "d4d43e5e55de8bf50683e2ab879615a220ff30ddc9faa783398b56c398f6ad5a"
LONG_Q = ("\n\nDescribe the above text in detail: what kind of document it seems to be, which words repeat "
          "in every sentence, and how the section numbers change from one sentence to the next. Then write a "
          "short commentary on what such a report might be for. Write at least 300 words.")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _assign(path, name):
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return node.value
    raise SystemExit("%s not found in %s" % (name, path))


def load_filler():
    expr = _assign(os.path.join(HERE, "bench_pr1.py"), "filler")
    # a pure expression (" ".join of a %-formatted list comprehension over range): evaluated with
    # no builtins but range
    text = eval(compile(ast.Expression(expr), "bench_pr1.py:filler", "eval"), {"__builtins__": {"range": range}})
    if sha(text) != FILLER_SHA256:
        raise SystemExit("bench_pr1.py filler changed: %s" % sha(text))
    return text


def load_short_prompt():
    prompts = ast.literal_eval(_assign(os.path.join(HERE, "bench_decode.py"), "PROMPTS"))
    if sha(prompts[0]) != SHORT_SHA256:
        raise SystemExit("bench_decode.py PROMPTS[0] changed: %s" % sha(prompts[0]))
    return prompts[0]


def long_prompt():
    return load_filler() + LONG_Q


def stream(name, prompt, max_tokens, opener=None):
    """One streamed greedy chat request; every content/reasoning delta is a chunk with its arrival
    time. Returns the result record (full text, per-chunk offsets from the request start)."""
    body = json.dumps({"model": "deepseek-v4-colibri", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    opener = opener or urllib.request.urlopen
    t0 = time.time(); times = []; parts = []; finish = None; usage = None
    with opener(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content") or ""
                if delta:
                    times.append(time.time()); parts.append(delta)
                if c.get("finish_reason"):
                    finish = c["finish_reason"]
    t1 = time.time()
    text = "".join(parts)
    n = len(times)
    first = times[0] if times else None
    return {"name": name, "prompt_sha256": sha(prompt), "prompt_chars": len(prompt), "max_tokens": max_tokens,
            "t0_epoch": round(t0, 4), "ttft_s": round(first - t0, 3) if first else None, "decode_chunks": n,
            "decode_s": round(times[-1] - first, 3) if n > 1 else None,
            "decode_tok_s": round((n - 1) / (times[-1] - first), 4) if n > 1 and times[-1] > first else None,
            "total_s": round(t1 - t0, 3), "finish_reason": finish, "usage": usage,
            "chunk_t": [round(t - t0, 4) for t in times], "chunk_text": parts,
            "text": text, "text_sha256": sha(text)}


def write(f, rec):
    f.write(json.dumps(rec) + "\n"); f.flush()
    print(rec["name"], rec["ttft_s"], rec["decode_tok_s"], rec["decode_chunks"], rec["finish_reason"],
          rec["text_sha256"][:12], flush=True)
