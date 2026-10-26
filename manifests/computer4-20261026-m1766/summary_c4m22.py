#!/usr/bin/env python3
"""summary_c4m22.py OUT | --brief BENCH.JSONL: per-arm decode tok/s and TTFT (bench_decode.py rows), texts by arm."""
import hashlib, json, os, statistics, sys

def rows(p):
    out = []
    for line in open(p):
        line = line.strip()
        if line.startswith("{"):
            r = json.loads(line); r["sha"] = hashlib.sha256(r.get("text", "").encode()).hexdigest()[:12]; out.append(r)
    return out

if sys.argv[1] == "--brief":
    for r in rows(sys.argv[2]):
        print("%-10s ttft=%s decode=%s chunks=%s text=%s" % (r["name"], r["ttft_s"], r["decode_tok_s"], r["decode_chunks"], r["sha"]))
    sys.exit(0)
out = sys.argv[1]
arms = {}
for arm in sorted(os.listdir(out)):
    if arm in ("meta",) or not os.path.isdir(os.path.join(out, arm)):
        continue
    for s in sorted(os.listdir(os.path.join(out, arm))):
        p = os.path.join(out, arm, s, "bench.jsonl")
        if os.path.exists(p):
            arms.setdefault(arm, []).append((s, rows(p)))
mean_dec = {}
for arm, samples in arms.items():
    print("arm %s" % arm)
    per = []
    for s, rs in samples:
        decs = [r["decode_tok_s"] for r in rs if r.get("decode_tok_s")]
        m = statistics.mean(decs) if decs else None
        per.append(m)
        print("  %s: %d rows (%s tokens), mean decode %s tok/s, median TTFT %s s, texts %s" % (
            s, len(rs), "/".join(str(r.get("decode_chunks")) for r in rs), round(m, 3) if m else None, statistics.median([r["ttft_s"] for r in rs if r.get("ttft_s")] or [0]),
            " ".join(r["sha"][:8] for r in rs)))
    good = [x for x in per if x]
    mean_dec[arm] = statistics.median(good) if good else None
    print("  median of per-sample mean decode: %s tok/s" % (round(mean_dec[arm], 3) if mean_dec[arm] else None))
for new, old in (("M1", "MB"), ("M1R", "MR")):
    if mean_dec.get(new) and mean_dec.get(old):
        print("%s/%s (#1766 vs its dev base, one RTX 5070): mean decode ratio %.3f" % (new, old, mean_dec[new] / mean_dec[old]))
