#!/usr/bin/env python3
"""summary_c4m29.py OUT | --brief BENCH.JSONL | --selftest (2026-10-26, from summary_c4m28.py): F1 on computer4 (one RTX 5070),
full V4 Flash, bench_pr1.py (long_prefill 1,992 tokens, long_prefill_2 3,005 tokens). Arms B (dev), M (#1766), MX (#1766, every
switch off). Per arm and prompt: TTFT, prefill tok/s, decode tok/s; ratios M/B, MX/B, MX/M per prompt (TTFT is the question).
Text gate: one group, every arm must give B's texts (MX: a difference is a note); final27's / final6's full references
(REF_OLD) are a note. Environment premise from sample.txt: M has no #1766 switch set, MX has every one at 0."""
import hashlib, json, os, re, statistics, sys, tempfile

GROUPS = {"gpu": ["B", "M", "MX"]}
NOTE_ONLY = ("MX",)
ABL_ALL = ["DSV4_CUDA_COMPRESSOR", "DSV4_CUDA_HEAD", "DSV4_CUDA_HEAD_BATCH", "DSV4_CUDA_INDEXER_PREP", "DSV4_CUDA_INDEXER_HEADS",
           "DSV4_CUDA_RESIDENT_INDEXER", "DSV4_CUDA_WO_DECODE", "DSV4_CUDA_RESIDENT_EXPERTS"]
REF_OLD = {"long_prefill": "a397814f650a80340c40cca3197295a49dd5d2dc9d55211f27473036bb4efd64",
           "long_prefill_2": "8b3f437baa347d887b5dada5bd950743d49b3181fdab9f27a87f64ad0433d4c0"}
RATIOS = [("M", "B", "#1766 vs dev, one RTX 5070, full"), ("MX", "B", "#1766 all switches off vs dev (1.0 = back to dev)"),
          ("MX", "M", "#1766 all switches off vs on")]
PROMPTS = ("long_prefill", "long_prefill_2")


def rows(p):
    out = []
    for line in open(p):
        line = line.strip()
        if line.startswith("{"):
            r = json.loads(line)
            r["sha_full"] = hashlib.sha256(r.get("text", "").encode()).hexdigest(); r["sha"] = r["sha_full"][:12]
            out.append(r)
    return out


def meta(d):
    out = {}
    try:
        for line in open(os.path.join(d, "sample.txt")):
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1); out[k] = v
    except FileNotFoundError:
        pass
    return out


def env_problems(arm, m):
    env = dict(kv.split("=", 1) for kv in m.get("server_env", "").split() if "=" in kv)
    p = []
    if arm in ("B", "M"):
        p += ["%s set" % k for k in ABL_ALL if k in env]
    if arm == "MX":
        p += ["%s not 0" % k for k in ABL_ALL if env.get(k) != "0"]
    if env.get("COLI_CUDA") != "1":
        p.append("COLI_CUDA not 1")
    return p


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(xs) if xs else None


def summarize(out):
    arms, problems, notes = {}, [], []
    for arm in sorted(os.listdir(out)):
        if arm == "meta" or not os.path.isdir(os.path.join(out, arm)):
            continue
        for s in sorted(os.listdir(os.path.join(out, arm))):
            d = os.path.join(out, arm, s); p = os.path.join(d, "bench.jsonl"); m = meta(d)
            log = open(os.path.join(d, "serve.log"), errors="replace").read() if os.path.exists(os.path.join(d, "serve.log")) else ""
            rs = rows(p) if os.path.exists(p) else []
            arms.setdefault(arm, []).append((s, int(m.get("seq", 10**9)), rs, len(re.findall(r"out of memory", log)), m, log))
    ref = {}
    for g, members in GROUPS.items():
        first = sorted((sq, a, rs) for a in members if a not in NOTE_ONLY for (_, sq, rs, _, _, _) in arms.get(a, []) if rs)
        if first:
            ref[g] = {r["name"]: (r["sha"], first[0][1]) for r in first[0][2]}
    lines, per_prompt = [], {}
    for arm, samples in arms.items():
        g = next((k for k, v in GROUPS.items() if arm in v), None)
        lines.append("arm %s%s" % (arm, "  (switch ablation: a text difference is a note)" if arm in NOTE_ONLY else ""))
        for s, sq, rs, ooms, m, log in samples:
            if m.get("client_rc") != "0":
                problems.append("%s %s: client rc %s" % (arm, s, m.get("client_rc")))
            if m.get("tree_dirty_lines", "0") != "0":
                problems.append("%s %s: tree has local changes" % (arm, s))
            for e in env_problems(arm, m):
                problems.append("%s %s: environment: %s" % (arm, s, e))
            if arm in ("M", "MX") and "devices=1 placement=contiguous-layers" not in log:
                problems.append("%s %s: no 'devices=1 placement=contiguous-layers' in serve.log" % (arm, s))
            if len(rs) < 2:
                problems.append("%s %s: %d rows, expected 2" % (arm, s, len(rs)))
            if re.search(r"Traceback|Segmentation fault|Aborted|core dumped", log):
                problems.append("%s %s: crash lines in serve.log" % (arm, s))
            diff = [r["name"] for r in rs if g in ref and r["name"] in ref[g] and r["sha"] != ref[g][r["name"]][0]]
            if diff:
                (notes if arm in NOTE_ONLY else problems).append("%s %s: texts differ from %s's for %s" % (arm, s, ref[g][diff[0]][1], " ".join(diff)))
            old = [r["name"] for r in rs if r["name"] in REF_OLD and r["sha_full"] != REF_OLD[r["name"]]]
            if old:
                notes.append("%s %s: %s differ from final27's full references" % (arm, s, " ".join(old)))
            if ooms:
                notes.append("%s %s: %d 'out of memory' lines in serve.log" % (arm, s, ooms))
            for r in rs:
                per_prompt.setdefault(arm, {}).setdefault(r["name"], []).append(r)
                lines.append("  %s (seq %s) %-15s ttft=%s s prefill=%s tok/s decode=%s tok/s chunks=%s text=%s%s" % (
                    s, sq, r["name"], r.get("ttft_s"), r.get("prefill_tok_s"), r.get("decode_tok_s"), r.get("decode_chunks"),
                    r["sha"][:8], "  DIFFERS" if r["name"] in diff else ""))
    ratios = []
    for new, old, what in RATIOS:
        if new in per_prompt and old in per_prompt:
            parts = []
            for n in PROMPTS:
                a = med([r.get("ttft_s") for r in per_prompt[new].get(n, [])]); b = med([r.get("ttft_s") for r in per_prompt[old].get(n, [])])
                c = med([r.get("decode_tok_s") for r in per_prompt[new].get(n, [])]); e = med([r.get("decode_tok_s") for r in per_prompt[old].get(n, [])])
                if a and b:
                    parts.append("%s ttft %.4f" % (n, a / b))
                if c and e:
                    parts.append("%s decode %.4f" % (n, c / e))
            ratios.append("%s/%s (%s): %s" % (new, old, what, ", ".join(parts)))
    return lines, ratios, problems, notes


def main(out):
    lines, ratios, problems, notes = summarize(out)
    print("C4M29 SUMMARY (F1 on one RTX 5070: dev B vs #1766 M vs #1766 all switches off MX, full V4 Flash, bench_pr1.py)")
    for l in lines + ratios:
        print(l)
    print("GATES: %s" % ("PASS" if not problems else "FAIL"))
    for p in problems:
        print("  problem: " + p)
    for p in notes:
        print("  note: " + p)


def _fake(base, arm, seq, texts, env, log):
    d = os.path.join(base, arm, "s1"); os.makedirs(d)
    open(os.path.join(d, "sample.txt"), "w").write("arm=%s\nseq=%d\ntree_dirty_lines=0\nserver_env=COLI_CUDA=1 %s\nclient_rc=0\n" % (arm, seq, env))
    with open(os.path.join(d, "bench.jsonl"), "w") as f:
        for i, (n, t) in enumerate(zip(PROMPTS, texts)):
            f.write(json.dumps({"name": n, "position": i + 1, "ttft_s": 250.0 + 100 * i + (20 if arm != "B" else 0),
                                "prefill_tok_s": 8.0, "decode_tok_s": 1.3, "decode_chunks": 30, "text": t}) + "\n")
    open(os.path.join(d, "serve.log"), "w").write(log)


def selftest():
    one = "v4_gpu tier=dense-matvec device=0 devices=1 placement=contiguous-layers\n"
    mx = " ".join(k + "=0" for k in ABL_ALL)
    with tempfile.TemporaryDirectory() as base:
        _fake(base, "B", 1, ("a", "b"), "", "")
        _fake(base, "M", 2, ("a", "b"), "", one)
        _fake(base, "MX", 3, ("a", "x"), mx, one)
        lines, ratios, problems, notes = summarize(base)
        assert not problems, problems
        assert any(n.startswith("MX s1: texts differ from B's for long_prefill_2") for n in notes), notes
        assert any(r.startswith("M/B") and "long_prefill ttft 1.0800" in r for r in ratios), ratios
    with tempfile.TemporaryDirectory() as base:
        _fake(base, "B", 1, ("a", "b"), "", "")
        _fake(base, "M", 2, ("a", "z"), "DSV4_CUDA_HEAD=0", "")
        _fake(base, "MX", 3, ("a", "b"), "DSV4_CUDA_HEAD=0", one)
        lines, ratios, problems, notes = summarize(base)
        p = " | ".join(problems)
        assert "M s1: texts differ from B's for long_prefill_2" in p and "M s1: environment: DSV4_CUDA_HEAD set" in p, p
        assert "M s1: no 'devices=1" in p and "MX s1: environment: DSV4_CUDA_COMPRESSOR not 0" in p, p
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1] == "--brief":
        for r in rows(sys.argv[2]):
            print("%-15s ttft=%s prefill=%s decode=%s chunks=%s text=%s" % (r["name"], r.get("ttft_s"), r.get("prefill_tok_s"),
                                                                           r.get("decode_tok_s"), r.get("decode_chunks"), r["sha"]))
    elif sys.argv[1] == "--selftest":
        selftest()
    else:
        main(sys.argv[1])
