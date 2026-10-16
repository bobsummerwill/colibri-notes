#!/usr/bin/env python3
"""Result tools for the final colibri GPU runs (run_pr1.sh, run_pr2.sh, run_drift.sh).

CPU only; reads <exp>/<arm>/s<k>/{sample.txt,bench.jsonl,serve.log,engine_environ.txt,nvsmi.csv}.

  final_tools.py brief BENCH.jsonl            one line per request (logged after each sample)
  final_tools.py phase1 DRIFT_DIR             drift phase-1 text comparison; prints DECISION=...
  final_tools.py summarize pr1|pr2|drift DIR  writes DIR/summary.json, prints a readable summary
"""
import glob, hashlib, json, os, re, statistics, sys


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path, errors="replace"):
        line = line.strip()
        if line.startswith("{"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "text" not in r:     # an old client that kept only text_head: never full-text evidence
                r["text"] = r.get("text_head", "")
                r["text_truncated"] = True
            if "text_sha256" not in r:
                r["text_sha256"] = sha(r["text"])
            rows.append(r)
    return rows


def kv(path):
    out = {}
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                out[k] = v
    return out


def env_file(path):
    return kv(path)


def lines(path):
    return open(path, errors="replace").read().splitlines() if os.path.exists(path) else []


def samples(base, arm):
    out = []
    for d in sorted(glob.glob(os.path.join(base, arm, "s*")), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0)):
        s = {"dir": d, "k": int(re.sub(r"\D", "", os.path.basename(d)) or 0), "meta": kv(os.path.join(d, "sample.txt")),
             "rows": read_jsonl(os.path.join(d, "bench.jsonl")), "log": lines(os.path.join(d, "serve.log")),
             "env": env_file(os.path.join(d, "engine_environ.txt"))}
        out.append(s)
    return out


def usage_changed(base):
    """samples whose .coli_usage state changed while they ran"""
    bad = []
    for st in glob.glob(os.path.join(base, "*", "s*", "sample.txt")):
        m = kv(st)
        if m.get("usage_before") and m.get("usage_after") and m["usage_before"] != m["usage_after"]:
            bad.append("%s: %s -> %s" % (os.path.relpath(os.path.dirname(st), base), m["usage_before"], m["usage_after"]))
    return bad


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(xs), 4) if xs else None


def gpu_max_mem(d, index):
    """max memory.used (MiB) of GPU `index` in the sample's nvidia-smi samples"""
    best = None
    for line in lines(os.path.join(d, "nvsmi.csv"))[1:]:
        f = [x.strip() for x in line.split(",")]
        if len(f) > 2 and f[1] == str(index):
            m = re.match(r"(\d+)", f[2])
            if m:
                best = max(best or 0, int(m.group(1)))
    return best


def grep(log, pat):
    r = re.compile(pat)
    return [l for l in log if r.search(l)]


def brief(path):
    for r in read_jsonl(path):
        print("   %-16s ttft=%-8s decode_tok_s=%-7s chunks=%-4s sha=%s" % (
            r.get("name"), r.get("ttft_s"), r.get("decode_tok_s"), r.get("decode_chunks"), r["text_sha256"][:12]))


def first_diff(a, b):
    return next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None if len(a) == len(b) else min(len(a), len(b)))


def env_check(s, must, must_not):
    """must: {VAR: value}; must_not: [VAR]; checked on the engine's real environment"""
    env = s["env"]
    if not env:
        return ["engine_environ.txt missing"]
    bad = ["%s=%r (want %r)" % (k, env.get(k), v) for k, v in must.items() if env.get(k) != v]
    bad += ["%s set (=%r) but must be unset" % (k, env[k]) for k in must_not if k in env]
    return bad


COMMON = {"COLI_CUDA": "1", "COLI_GPU": "0", "DSV4_CUDA_DEVICE": "0", "CUDA_DENSE": "1", "COLI_CUDA_PIPE": "2",
          "COLI_CUDA_ATTN_BATCH": "1", "COLI_CUDA_MOE_BATCH": "1", "V4_MOE_REFILL_GROUP": "12",
          "V4_LOADER_LANES": "3", "V4_PREFIX_CKPT_DISK": "0"}


# ---------------------------------------------------------------- PR1
PR1_ARMS = {
    "A": ({}, ["COLI_CUDA_MOE_DOUBLE"], [r"moe-batch=off"]),
    "B": ({}, ["COLI_CUDA_MOE_DOUBLE", "DSV4_MOE_EXPERT_MAJOR", "DSV4_MM_WARP"], [r"moe-batch=on bank=132-experts"]),
    "C": ({"COLI_CUDA_MOE_DOUBLE": "1"}, [], [r"moe-batch=on bank=132-experts", r"moe-double=on"]),
    "E": ({"DSV4_MOE_EXPERT_MAJOR": "0"}, ["COLI_CUDA_MOE_DOUBLE"], [r"moe-batch=on bank=132-experts"]),
    "F": ({"DSV4_MM_WARP": "0"}, ["COLI_CUDA_MOE_DOUBLE"], [r"moe-batch=on bank=132-experts"]),
    "V": ({"DSV4_MOE_EXPERT_MAJOR_VERIFY": "1", "DSV4_MM_WARP_VERIFY": "1"}, ["COLI_CUDA_MOE_DOUBLE"],
          [r"moe-batch=on bank=132-experts", r"mm-warp verify calls=", r"moe-expert-major verify calls="]),
}


def verify_counters(log):
    mw = [tuple(map(int, m.groups())) for m in (re.search(r"mm-warp verify calls=(\d+) mismatches=(\d+)", l) for l in log) if m]
    em = [tuple(map(int, m.groups())) for m in
          (re.search(r"moe-expert-major verify calls=(\d+) fc1_mismatch=(\d+) fc2_mismatch=(\d+)", l) for l in log) if m]
    return {"mm_warp_lines": len(mw), "mm_warp_max_calls": max((c for c, _ in mw), default=0),
            "mm_warp_max_mismatches": max((x for _, x in mw), default=None),
            "expert_major_lines": len(em), "expert_major_max_calls": max((c for c, _, _ in em), default=0),
            "expert_major_max_fc1_mismatch": max((a for _, a, _ in em), default=None),
            "expert_major_max_fc2_mismatch": max((b for _, _, b in em), default=None),
            "nonzero_lines": [l for l in log if re.search(r"verify calls=.*(mismatches|fc1_mismatch|fc2_mismatch)=[1-9]", l)][:20],
            "raw_tail": [l for l in log if "verify calls=" in l][-6:],
            "notes": [l for l in log if "_VERIFY" in l][:6]}


def summarize_pr1(base):
    out = {"kind": "pr1", "arms": {}}
    ref = {}
    a_samples = samples(base, "A")
    for name in ("long_prefill", "long_prefill_2"):
        shas = {r["text_sha256"] for s in a_samples for r in s["rows"] if r["name"] == name}
        ref[name] = {"sha256s": sorted(shas), "consistent": len(shas) == 1}
    out["reference_A"] = ref
    for arm, (extra, unset, modes) in PR1_ARMS.items():
        ss = samples(base, arm)
        if not ss:
            continue
        a = {"samples": [], "requests": {}}
        for s in ss:
            must = dict(COMMON, **extra)
            a["samples"].append({
                "k": s["k"], "seq": s["meta"].get("seq"), "tree_head": s["meta"].get("tree_head"),
                "binary_sha256": s["meta"].get("binary_sha256"), "client_rc": s["meta"].get("client_rc"),
                "ready_s": s["meta"].get("ready_s"), "env_problems": env_check(s, must, unset),
                "mode_missing": [p for p in modes if not grep(s["log"], p)],
                "oom_or_fail_lines": len(grep(s["log"], r"out of memory|FAILED|Traceback")),
                "requests": {r["name"]: {"ttft_s": r.get("ttft_s"), "prefill_tok_s": r.get("prefill_tok_s"),
                                         "decode_tok_s": r.get("decode_tok_s"), "sha": r["text_sha256"]} for r in s["rows"]}})
            if arm == "V":
                a["verify"] = verify_counters(s["log"])
        for name in ("long_prefill", "long_prefill_2"):
            rs = [x["requests"][name] for x in a["samples"] if name in x["requests"]]
            if not rs:
                continue
            refsha = ref[name]["sha256s"][0] if ref[name]["consistent"] and ref[name]["sha256s"] else None
            a["requests"][name] = {
                "n": len(rs), "ttft_s": [r["ttft_s"] for r in rs], "prefill_tok_s": [r["prefill_tok_s"] for r in rs],
                "decode_tok_s": [r["decode_tok_s"] for r in rs], "median_ttft_s": med([r["ttft_s"] for r in rs]),
                "median_prefill_tok_s": med([r["prefill_tok_s"] for r in rs]),
                "shas": [r["sha"][:12] for r in rs],
                "identical_to_A": (all(r["sha"] == refsha for r in rs) if refsha else None)}
        out["arms"][arm] = a
    lines_out = ["PR1 prefill benchmark: %s" % base]
    for name in ("long_prefill", "long_prefill_2"):
        lines_out.append("A reference %s: %s consistent=%s" % (name, [x[:12] for x in ref[name]["sha256s"]], ref[name]["consistent"]))
    for arm, a in out["arms"].items():
        for name, r in a["requests"].items():
            base_ttft = out["arms"].get("A", {}).get("requests", {}).get(name, {}).get("median_ttft_s")
            b_ttft = out["arms"].get("B", {}).get("requests", {}).get(name, {}).get("median_ttft_s")
            sp = " vsA=%.3fx" % (base_ttft / r["median_ttft_s"]) if base_ttft and r["median_ttft_s"] else ""
            sb = " vsB=%.3fx" % (b_ttft / r["median_ttft_s"]) if b_ttft and r["median_ttft_s"] and arm != "B" else ""
            lines_out.append("%s %-15s n=%d ttft=%s median=%s tok/s=%s%s%s identical_to_A=%s shas=%s" % (
                arm, name, r["n"], r["ttft_s"], r["median_ttft_s"], r["median_prefill_tok_s"], sp, sb, r["identical_to_A"], r["shas"]))
        for s in a["samples"]:
            if s["env_problems"] or s["mode_missing"] or s["client_rc"] not in ("0", None) or s["oom_or_fail_lines"]:
                lines_out.append("  !! %s s%d rc=%s env=%s mode_missing=%s fail_lines=%d" % (
                    arm, s["k"], s["client_rc"], s["env_problems"], s["mode_missing"], s["oom_or_fail_lines"]))
        if "verify" in a:
            v = a["verify"]
            lines_out.append("V verify: mm-warp lines=%d max_calls=%d max_mismatches=%s | expert-major lines=%d max_calls=%d fc1=%s fc2=%s | nonzero=%d" % (
                v["mm_warp_lines"], v["mm_warp_max_calls"], v["mm_warp_max_mismatches"], v["expert_major_lines"],
                v["expert_major_max_calls"], v["expert_major_max_fc1_mismatch"], v["expert_major_max_fc2_mismatch"], len(v["nonzero_lines"])))
    return out, lines_out


# ---------------------------------------------------------------- decode helpers (PR2, drift)
def decode_stats(s):
    rows = s["rows"]
    toks = [r.get("decode_tok_s") for r in rows if r.get("decode_tok_s")]
    dec_s = sum(r.get("decode_s") or 0 for r in rows)
    chunks = sum((r.get("decode_chunks") or 1) - 1 for r in rows)
    return {"n_prompts": len(rows), "mean_tok_s": round(statistics.mean(toks), 4) if toks else None,
            "agg_tok_s": round(chunks / dec_s, 4) if dec_s else None, "per_prompt_tok_s": toks,
            "shas": [r["text_sha256"] for r in rows], "texts": [r.get("text", "") for r in rows],
            "total_s": round(sum(r.get("total_s") or 0 for r in rows), 2)}


def compare_texts(ref, other):
    """per-prompt: identical? else first differing character"""
    res = []
    for i, (a, b) in enumerate(zip(ref["texts"], other["texts"])):
        res.append(None if a == b else first_diff(a, b))
    return res


PR2_ARMS = {
    # arm: (must-have env beyond COMMON, must-be-unset, required log patterns, forbidden log patterns, vram_first expectation)
    "G1": ({"V4_SERVE_STATS_LOG": "1"}, ["COLI_CUDA_MOE_DOUBLE", "V4_GPU_RESIDENT_FIRST", "DSV4_CUDA_PEER_DEVICE", "DSV4_CUDA_PEER_COMPUTE"],
           [r"v4_serve_turn "], [r"peer-mirrors"], "zero"),
    "X1": ({"V4_SERVE_STATS_LOG": "1", "V4_GPU_RESIDENT_FIRST": "1"}, ["COLI_CUDA_MOE_DOUBLE", "DSV4_CUDA_PEER_DEVICE", "DSV4_CUDA_PEER_COMPUTE"],
           [r"v4_serve_turn "], [r"peer-mirrors"], "positive"),
    "X2": ({"V4_SERVE_STATS_LOG": "1", "V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1", "DSV4_CUDA_PEER_COMPUTE": "0"},
           ["COLI_CUDA_MOE_DOUBLE"], [r"v4_gpu peer-mirrors device=1 .*mode=p2p-read", r"v4_serve_turn "],
           [r"mode=compute", r"peer mirror pool off", r"peer device \d+ failed"], "positive"),
    "X3": ({"V4_SERVE_STATS_LOG": "1", "V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1"},
           ["COLI_CUDA_MOE_DOUBLE", "DSV4_CUDA_PEER_COMPUTE"], [r"v4_gpu peer-mirrors device=1 .*mode=compute", r"v4_serve_turn "],
           [r"mode=p2p-read", r"peer mirror pool off", r"peer device \d+ failed"], "positive"),
    "P1": ({"V4_SERVE_STATS_LOG": "1"}, ["COLI_CUDA_MOE_DOUBLE", "V4_GPU_RESIDENT_FIRST", "DSV4_CUDA_PEER_DEVICE", "DSV4_CUDA_PEER_COMPUTE"],
           [], [r"peer-mirrors"], None),
}


def serve_stats(log):
    turns = grep(log, r"^v4_serve_turn ")
    dec = grep(log, r"^v4_dec_moe ")
    vf = None
    if turns:
        m = re.search(r"vram_first_hits_total=(\d+)", turns[-1])
        vf = int(m.group(1)) if m else None
    return {"turns": len(turns), "vram_first_hits_total": vf, "last_turn": turns[-1] if turns else None,
            "last_dec_moe": dec[-1] if dec else None, "peer_lines": grep(log, r"peer-mirrors|peer mirror pool|peer device")[:6],
            "tok_s_log": [float(m.group(1)) for m in (re.search(r" tok_s=([\d.]+)", t) for t in turns) if m]}


def summarize_pr2(base):
    out = {"kind": "pr2", "arms": {}}
    ref = None
    g1 = samples(base, "G1")
    if g1 and g1[0]["rows"]:
        ref = decode_stats(g1[0])
    for arm, (extra, unset, need, forbid, vf_expect) in PR2_ARMS.items():
        ss = samples(base, arm)
        if not ss:
            continue
        a = {"samples": []}
        for s in ss:
            st = decode_stats(s)
            sv = serve_stats(s["log"])
            problems = env_check(s, dict(COMMON, **extra), unset)
            problems += ["missing log line /%s/" % p for p in need if not grep(s["log"], p)]
            problems += ["forbidden log line /%s/: %s" % (p, grep(s["log"], p)[0][:160]) for p in forbid if grep(s["log"], p)]
            if vf_expect == "zero" and sv["vram_first_hits_total"] not in (0, None):
                problems.append("vram_first_hits_total=%s, want 0" % sv["vram_first_hits_total"])
            if vf_expect == "positive" and not sv["vram_first_hits_total"]:
                problems.append("vram_first_hits_total=%s, want > 0" % sv["vram_first_hits_total"])
            if len(s["rows"]) != 6:
                problems.append("only %d of 6 prompts" % len(s["rows"]))
            a["samples"].append({
                "k": s["k"], "seq": s["meta"].get("seq"), "tree_head": s["meta"].get("tree_head"),
                "binary_sha256": s["meta"].get("binary_sha256"), "client_rc": s["meta"].get("client_rc"),
                "mean_tok_s": st["mean_tok_s"], "agg_tok_s": st["agg_tok_s"], "per_prompt_tok_s": st["per_prompt_tok_s"],
                "shas": [x[:12] for x in st["shas"]], "total_s": st["total_s"],
                "text_vs_G1s1": compare_texts(ref, st) if ref else None,
                "gpu1_max_mib": gpu_max_mem(s["dir"], 1), "gpu0_max_mib": gpu_max_mem(s["dir"], 0),
                "serve_stats": sv, "mode_ok": not problems, "problems": problems})
        a["median_mean_tok_s"] = med([x["mean_tok_s"] for x in a["samples"]])
        a["median_agg_tok_s"] = med([x["agg_tok_s"] for x in a["samples"]])
        a["all_identical_to_G1s1"] = all(x["text_vs_G1s1"] is not None and all(v is None for v in x["text_vs_G1s1"]) and
                                         len(x["shas"]) == 6 for x in a["samples"])
        out["arms"][arm] = a
    L = ["PR2 decode re-measurement: %s" % base]
    g = out["arms"].get("G1", {}).get("median_mean_tok_s")
    x1 = out["arms"].get("X1", {}).get("median_mean_tok_s")
    for arm, a in out["arms"].items():
        m = a["median_mean_tok_s"]
        rel = (" vsG1=%.3fx" % (m / g) if g and m else "") + (" vsX1=%.3fx" % (m / x1) if x1 and m and arm in ("X2", "X3") else "")
        L.append("%s median(mean of 6)=%s median(agg)=%s%s identical_to_G1s1=%s samples=%s" % (
            arm, m, a["median_agg_tok_s"], rel, a["all_identical_to_G1s1"], [x["mean_tok_s"] for x in a["samples"]]))
        for x in a["samples"]:
            L.append("   s%d seq=%s rc=%s gpu0_max=%sMiB gpu1_max=%sMiB vram_first=%s mode_ok=%s %s" % (
                x["k"], x["seq"], x["client_rc"], x["gpu0_max_mib"], x["gpu1_max_mib"],
                x["serve_stats"]["vram_first_hits_total"], x["mode_ok"], "; ".join(x["problems"])[:300]))
            if x["serve_stats"]["peer_lines"]:
                L.append("      " + x["serve_stats"]["peer_lines"][0][:200])
    return out, L


# ---------------------------------------------------------------- drift
def phase1(base, quiet=False):
    D, Z = samples(base, "D"), samples(base, "Z")
    res = {"D": [decode_stats(s) for s in D], "Z": [decode_stats(s) for s in Z]}

    def distinct(stats):
        per = []
        n = max((len(s["shas"]) for s in stats), default=0)
        for i in range(n):
            per.append(len({s["shas"][i] for s in stats if len(s["shas"]) > i}))
        return per

    d_distinct, z_distinct = distinct(res["D"]), distinct(res["Z"])
    dev_differs = any(x > 1 for x in d_distinct)
    z_differs = any(x > 1 for x in z_distinct)
    dz = None
    if res["D"] and res["Z"]:
        dz = compare_texts(res["D"][0], res["Z"][0])
    within_d = [compare_texts(res["D"][0], s) for s in res["D"][1:]] if res["D"] else []
    within_z = [compare_texts(res["Z"][0], s) for s in res["Z"][1:]] if res["Z"] else []
    per_prompt = [r.get("total_s") for s in D + Z for r in s["rows"] if r.get("total_s")]
    ready = [float(s["meta"]["ready_s"]) for s in D + Z if s["meta"].get("ready_s")]
    complete_d = sum(1 for s in res["D"] if s["n_prompts"] == 6)
    out = {"dev_text_differs": dev_differs, "pr0_text_differs": z_differs, "dev_distinct_per_prompt": d_distinct,
           "pr0_distinct_per_prompt": z_distinct, "dev_vs_pr0_s1_first_diff": dz, "dev_within_first_diff": within_d,
           "pr0_within_first_diff": within_z, "complete_dev_samples": complete_d,
           "per_prompt_s": round(statistics.median(per_prompt), 1) if per_prompt else 120.0,
           "ready_s": round(statistics.median(ready), 1) if ready else 30.0,
           "mean_tok_s": {"D": [s["mean_tok_s"] for s in res["D"]], "Z": [s["mean_tok_s"] for s in res["Z"]]},
           "shas": {"D": [[x[:12] for x in s["shas"]] for s in res["D"]], "Z": [[x[:12] for x in s["shas"]] for s in res["Z"]]}}
    out["decision"] = "DIFF" if dev_differs else ("NODIFF" if complete_d >= 2 else "INCOMPLETE")
    return out


def logit_lines(log):
    return [l.strip() for l in log if l.startswith("v4_logit ")]


def cmp_logits(a, b):
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), None)
    tok = next((i for i in range(n) if a[i].split()[1] != b[i].split()[1]), None)
    return {"compared": n, "len_a": len(a), "len_b": len(b), "first_diff_line": first, "first_token_diff": tok,
            "n_diff_lines": sum(1 for i in range(n) if a[i] != b[i]),
            "first_a": a[first] if first is not None else None, "first_b": b[first] if first is not None else None}


def summarize_drift(base):
    p1 = phase1(base)
    out = {"kind": "drift", "phase1": p1, "phase2": None}
    Dp, Zp = samples(base, "Dp"), samples(base, "Zp")
    if Dp or Zp:
        ll = {("Dp", s["k"]): logit_lines(s["log"]) for s in Dp}
        ll.update({("Zp", s["k"]): logit_lines(s["log"]) for s in Zp})
        pairs = {}
        if len(Dp) >= 2:
            pairs["Dp1_vs_Dp2"] = cmp_logits(ll[("Dp", Dp[0]["k"])], ll[("Dp", Dp[1]["k"])])
        if len(Zp) >= 2:
            pairs["Zp1_vs_Zp2"] = cmp_logits(ll[("Zp", Zp[0]["k"])], ll[("Zp", Zp[1]["k"])])
        if Dp and Zp:
            pairs["Dp1_vs_Zp1"] = cmp_logits(ll[("Dp", Dp[0]["k"])], ll[("Zp", Zp[0]["k"])])
        p1ref = {"Dp": samples(base, "D"), "Zp": samples(base, "Z")}
        per = []
        for arm, ss in (("Dp", Dp), ("Zp", Zp)):
            for s in ss:
                st = decode_stats(s)
                ref = decode_stats(p1ref[arm][0]) if p1ref[arm] else None
                per.append({"arm": arm, "k": s["k"], "logit_lines": len(ll[(arm, s["k"])]), "prompts": st["n_prompts"],
                            "cpu_moe_lines": len(grep(s["log"], r"^v4_cpu_moe ")), "client_rc": s["meta"].get("client_rc"),
                            "probe_on": bool(s["env"].get("V4_LOGIT_LOG") == "1"),
                            "text_vs_phase1_s1": compare_texts(ref, st) if ref else None})
        out["phase2"] = {"pairs": pairs, "samples": per}
    L = ["Drift (PR0 upload race): %s" % base,
         "phase1 decision=%s dev_text_differs=%s pr0_text_differs=%s" % (p1["decision"], p1["dev_text_differs"], p1["pr0_text_differs"]),
         "  dev distinct texts per prompt=%s pr0=%s" % (p1["dev_distinct_per_prompt"], p1["pr0_distinct_per_prompt"]),
         "  dev s1 vs pr0 s1 first differing char per prompt=%s (None = identical)" % p1["dev_vs_pr0_s1_first_diff"],
         "  dev s1 vs s2,s3=%s pr0 s1 vs s2,s3=%s" % (p1["dev_within_first_diff"], p1["pr0_within_first_diff"]),
         "  decode tok/s D=%s Z=%s" % (p1["mean_tok_s"]["D"], p1["mean_tok_s"]["Z"])]
    if out["phase2"]:
        for k, v in out["phase2"]["pairs"].items():
            L.append("phase2 %s: compared=%d (%d/%d) first_diff_line=%s first_token_diff=%s n_diff=%d" % (
                k, v["compared"], v["len_a"], v["len_b"], v["first_diff_line"], v["first_token_diff"], v["n_diff_lines"]))
            if v["first_a"]:
                L.append("    a: %s\n    b: %s" % (v["first_a"], v["first_b"]))
        for x in out["phase2"]["samples"]:
            L.append("  %s s%d probe_on=%s logit_lines=%d prompts=%d cpu_moe_lines=%d rc=%s text_vs_phase1=%s" % (
                x["arm"], x["k"], x["probe_on"], x["logit_lines"], x["prompts"], x["cpu_moe_lines"], x["client_rc"], x["text_vs_phase1_s1"]))
    else:
        L.append("phase2: not run")
    return out, L


def strip_texts(o):
    if isinstance(o, dict):
        return {k: strip_texts(v) for k, v in o.items() if k != "texts"}
    if isinstance(o, list):
        return [strip_texts(v) for v in o]
    return o


def main(argv):
    if len(argv) >= 2 and argv[0] == "brief":
        brief(argv[1]); return 0
    if len(argv) >= 2 and argv[0] == "phase1":
        r = phase1(argv[1])
        json.dump(strip_texts(r), open(os.path.join(argv[1], "phase1.json"), "w"), indent=1)
        print("DECISION=%s" % r["decision"]); print("PER_PROMPT_S=%d" % int(r["per_prompt_s"])); print("READY_S=%d" % int(r["ready_s"]))
        print("# dev distinct per prompt %s, pr0 %s, dev-vs-pr0 %s" % (r["dev_distinct_per_prompt"], r["pr0_distinct_per_prompt"], r["dev_vs_pr0_s1_first_diff"]))
        return 0
    if len(argv) >= 3 and argv[0] == "summarize":
        fn = {"pr1": summarize_pr1, "pr2": summarize_pr2, "drift": summarize_drift}[argv[1]]
        out, L = fn(argv[2])
        out["usage_history_changed"] = usage_changed(argv[2])
        L.append("usage history (.coli_usage) changed during: %s" % (out["usage_history_changed"] or "no sample"))
        json.dump(strip_texts(out), open(os.path.join(argv[2], "summary.json"), "w"), indent=1)
        print("\n".join(L))
        return 0
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
