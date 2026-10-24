#!/usr/bin/env python3
"""final6 client for the two new request shapes (bench_pr1.py stays the client of every other arm).

  BENCH_MODE=long      one ~8k-token prompt (60 tokens decoded); the server's usage gives the count
  BENCH_MODE=parallel  bench_pr1.py's two requests at the same moment, from two threads; each record
                       is bench_pr1.py's, so the texts compare with the sequential arms'
  BENCH_MODE=long16    one ~16k-token prompt (60 tokens decoded)
  BENCH_MODE=short     bench_pr1.py's first prompt, 24 tokens decoded (MTP drafting is slow here)
  BENCH_MODE=soak      a 6-turn conversation (each answer goes back into the history), run 3 times on
                       the same server: records rep<r>_turn<t>; every repeat must match the first
Usage: BENCH_MODE=... bench_f6.py OUT.jsonl
"""
import hashlib, json, os, sys, threading, time, urllib.request

URL = "http://127.0.0.1:8090/v1/chat/completions"
OUT = sys.argv[1]
MODE = os.environ.get("BENCH_MODE", "")

# bench_pr1.py's requests, verbatim
filler = " ".join(["The quick brown fox jumps over the lazy dog while the committee reviews section %d of the quarterly infrastructure report." % i for i in range(90)])
filler2 = " ".join(["Station %d logged a temperature of %d degrees and a wind speed of %d knots during the overnight survey." % (i, 10 + i % 17, 3 + i % 11) for i in range(130)])
PR1_REQUESTS = [
    ("long_prefill", filler + "\n\nSummarize the above in one sentence.", 100, 1992),
    ("long_prefill_2", filler2 + "\n\nWhich station had the highest wind speed? Answer briefly.", 60, 3005),
]
# ~8k tokens: a varied ledger, so the prompt is not one sentence repeated (bench_pr1.py's are)
ITEMS = ["valves", "pumps", "cables", "filters", "sensors", "bearings", "gaskets", "relays", "fuses", "hoses"]
SITES = ["Ashford", "Brightwater", "Calder", "Dunmore", "Eastleigh", "Fernhill", "Glenrock", "Harwood"]
filler8k = " ".join(["Entry %d: the %s depot shipped %d %s to bay %d on day %d, and %d were returned for inspection."
                     % (i, SITES[i % len(SITES)], 20 + (i * 37) % 180, ITEMS[(i * 7) % len(ITEMS)], 1 + (i * 13) % 40,
                        1 + (i * 11) % 30, (i * 5) % 9) for i in range(300)])
filler16k = " ".join(["Entry %d: the %s depot shipped %d %s to bay %d on day %d, and %d were returned for inspection."
                      % (i, SITES[i % len(SITES)], 20 + (i * 37) % 180, ITEMS[(i * 7) % len(ITEMS)], 1 + (i * 13) % 40,
                         1 + (i * 11) % 30, (i * 5) % 9) for i in range(600)])
LONG16_REQUEST = ("prompt_16k", filler16k + "\n\nWhich depot shipped the most units in a single entry, and how many? Answer briefly.", 60, None)
SOAK_TURNS = ["Give me three ideas for a weekend project with a Raspberry Pi.",
              "Pick the second idea and list the parts I would need.",
              "Write the first ten lines of a Python script for it.",
              "What could go wrong with that script on a cold morning?",
              "Summarize our conversation so far in two sentences.",
              "Now suggest a name for the project."]
LONG_REQUEST = ("prompt_8k", filler8k + "\n\nWhich depot shipped the most units in a single entry, and how many? Answer briefly.", 60, None)


def run(name, prompt, max_tokens, prompt_tokens, position, messages=None):
    body = json.dumps({"model": "deepseek-v4-colibri", "messages": messages or [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; n = 0; text = ""; usage = None
    with urllib.request.urlopen(req, timeout=7200) as r:
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
                    if first is None:
                        first = time.time()
                    n += 1; text += delta
    t1 = time.time()
    if prompt_tokens is None and usage:
        prompt_tokens = usage.get("prompt_tokens")
    ttft = first - t0
    return {"name": name, "position": position, "prompt_tokens": prompt_tokens, "ttft_s": round(ttft, 3),
            "prefill_tok_s": round(prompt_tokens / ttft, 4) if prompt_tokens else None, "decode_chunks": n,
            "decode_tok_s": round((n - 1) / (t1 - first), 4) if n > 1 else None,
            "total_s": round(t1 - t0, 3), "start_s": round(t0, 3), "usage": usage,
            "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


def write(f, rec):
    f.write(json.dumps(rec) + "\n"); f.flush()
    print(rec["name"], rec["ttft_s"], rec["decode_tok_s"], rec["text_sha256"][:12], flush=True)


with open(OUT, "w") as f:
    if MODE == "long":
        write(f, run(*LONG_REQUEST, 1))
    elif MODE == "long16":
        write(f, run(*LONG16_REQUEST, 1))
    elif MODE == "short":
        write(f, run(PR1_REQUESTS[0][0] + "_24", PR1_REQUESTS[0][1], 24, PR1_REQUESTS[0][3], 1))
    elif MODE == "soak":
        pos = 0
        for rep in range(1, 4):
            history = []
            for t, question in enumerate(SOAK_TURNS, 1):
                history.append({"role": "user", "content": question})
                pos += 1
                rec = run("rep%d_turn%d" % (rep, t), None, 120, None, pos, messages=list(history))
                history.append({"role": "assistant", "content": rec["text"]})
                write(f, rec)
    elif MODE == "parallel":
        recs, errs = [None, None], []

        def worker(i):
            try:
                recs[i] = run(*PR1_REQUESTS[i], i + 1)
            except Exception as e:  # recorded, and the client fails
                errs.append("%s: %r" % (PR1_REQUESTS[i][0], e))
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for rec in recs:
            if rec:
                write(f, rec)
        if errs:
            print("errors: " + "; ".join(errs), file=sys.stderr)
            sys.exit(1)
    else:
        print("BENCH_MODE must be long, long16, short, soak or parallel", file=sys.stderr)
        sys.exit(2)
