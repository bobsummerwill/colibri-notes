#!/usr/bin/env python3
"""summary_c4m28.py (2026-10-26: c4m27's for the PR #2 split arms S1-S3 and anchor B; one GPU group). summary_c4m25.py OUT | --brief BENCH.JSONL (2026-10-26, from summary_c4m22.py): same-base run on computer4. Per-arm decode
tok/s and TTFT (bench_decode.py rows), handled OOM lines, and the text gate: one base, so every arm of a group must give
the same texts as the group's first sample (GPU group B/M/P/C; CPU group CB/CM/CP/CC)."""
import hashlib, json, os, re, statistics, sys

GROUPS = {"gpu": ["B", "S1", "S2", "S3"]}
RATIOS = [("S1", "B", "PR #2 step 1 (per-cache guard) vs dev, one RTX 5070"),
          ("S2", "B", "PR #2 steps 1-2 (+ pin cap, stats log) vs dev, one RTX 5070"),
          ("S3", "B", "PR #2 steps 1-3 (+ VRAM-first) vs dev, one RTX 5070"),
          ("S2", "S1", "what step 2 adds, one RTX 5070"), ("S3", "S2", "what step 3 adds, one RTX 5070")]

def rows(p):
    out = []
    for line in open(p):
        line = line.strip()
        if line.startswith("{"):
            r = json.loads(line); r["sha"] = hashlib.sha256(r.get("text", "").encode()).hexdigest()[:12]; out.append(r)
    return out

def seq(d):
    try:
        return int(re.search(r"^seq=(\d+)", open(os.path.join(d, "sample.txt")).read(), re.M).group(1))
    except Exception:
        return 10**9

if sys.argv[1] == "--brief":
    for r in rows(sys.argv[2]):
        print("%-10s ttft=%s decode=%s chunks=%s text=%s" % (r["name"], r["ttft_s"], r["decode_tok_s"], r["decode_chunks"], r["sha"]))
    sys.exit(0)
out = sys.argv[1]
arms, problems = {}, []
for arm in sorted(os.listdir(out)):
    if arm == "meta" or not os.path.isdir(os.path.join(out, arm)):
        continue
    for s in sorted(os.listdir(os.path.join(out, arm))):
        d = os.path.join(out, arm, s); p = os.path.join(d, "bench.jsonl")
        if os.path.exists(p):
            log = open(os.path.join(d, "serve.log"), errors="replace").read() if os.path.exists(os.path.join(d, "serve.log")) else ""
            arms.setdefault(arm, []).append((s, seq(d), rows(p), len(re.findall(r"out of memory", log)),
                                             bool(re.search(r"moe-batch=off \(bank allocation failed", log))))
ref = {}
for g, members in GROUPS.items():
    first = sorted((sq, a, rs) for a in members for (_, sq, rs, _, _) in arms.get(a, []))
    if first:
        ref[g] = {r["name"]: (r["sha"], first[0][1]) for r in first[0][2]}
mean_dec = {}
for arm, samples in arms.items():
    g = next((k for k, v in GROUPS.items() if arm in v), None)
    print("arm %s" % arm)
    per = []
    for s, sq, rs, ooms, bank_lost in samples:
        decs = [r["decode_tok_s"] for r in rs if r.get("decode_tok_s")]
        m = statistics.mean(decs) if decs else None
        per.append(m)
        diff = [r["name"] for r in rs if g and r["name"] in ref[g] and r["sha"] != ref[g][r["name"]][0]]
        if diff:
            problems.append("%s %s: texts differ from %s's for %s" % (arm, s, ref[g][diff[0]][1], " ".join(diff)))
        if ooms:
            problems.append("%s %s: %d 'out of memory' lines in serve.log" % (arm, s, ooms))
        print("  %s (seq %s): %d rows (%s tokens), mean decode %s tok/s, median TTFT %s s, OOM lines %d%s, texts %s%s" % (
            s, sq, len(rs), "/".join(str(r.get("decode_chunks")) for r in rs), round(m, 3) if m else None,
            statistics.median([r["ttft_s"] for r in rs if r.get("ttft_s")] or [0]), ooms, " BANK LOST" if bank_lost else "",
            " ".join(r["sha"][:8] for r in rs), "" if not diff else "  DIFFERS: " + " ".join(diff)))
        if not rs:
            problems.append("%s %s: no rows" % (arm, s))
    good = [x for x in per if x]
    mean_dec[arm] = statistics.median(good) if good else None
    print("  median of per-sample mean decode: %s tok/s" % (round(mean_dec[arm], 3) if mean_dec[arm] else None))
for new, old, what in RATIOS:
    if mean_dec.get(new) and mean_dec.get(old):
        print("%s/%s (%s): mean decode ratio %.3f" % (new, old, what, mean_dec[new] / mean_dec[old]))
print("GATES: %s" % ("PASS" if not problems else "FAIL"))
for p in problems:
    print("  problem: " + p)
