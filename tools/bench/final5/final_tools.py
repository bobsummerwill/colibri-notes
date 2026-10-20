#!/usr/bin/env python3
"""Result tools for the final3 colibri GPU run (run_final3.sh). CPU only.

Reads <exp>/<arm>/s<k>/{sample.txt,bench.jsonl,serve.log,engine_environ.txt,nvsmi.csv} and
<exp>/meta/{tips.txt,llama_before.txt,llama_after.txt}. Derived from final/final_tools.py (same
sample layout, same PR1/PR2 arm checks); what is new is the fixed references below and the gates.

  final_tools.py brief BENCH.jsonl            one line per request (logged after each sample)
  final_tools.py check ARM SAMPLE_DIR         premise check after a sample; exit 1 = premise broken
  final_tools.py summarize final4 DIR         writes DIR/summary.json, prints a readable summary
  final_tools.py selftest                     the check and the summary on synthetic samples

Arms (run_final3.sh):  W warm-up (B's config, discarded) | A dev 7b55a9b3 | B pr1 tip |
C pr1 tip + COLI_CUDA_MOE_DOUBLE=1 (client bench_pr1.py, both requests) | G1 pr2 tip |
X3 pr2 tip + V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1 (client bench_decode.py, 6 x 300).
"""
import glob, hashlib, json, os, re, shutil, statistics, sys, tempfile

DEV_SHA = "736a0682d38f0a1c6592f34b1c6749fea221da42"   # final5: upstream dev after PR #0 merged (references below: older bases)
DEV_SHA_FINAL4 = "62eaa9dc249b67567b9cbc83e43fbc875a8b94c1"

# Fixed references (2026-10-15 final runs, colibri-claude manifests/final/pr1|pr2/stats.json).
# PR1: the dev reference text of each request (A, identical in all 3 samples and in B, C, E, F, V).
REF_PR1 = {"long_prefill": "32534c06abd640d59d2eb39ab6609051f63e9876f533aad1d75dd5cd775835ba",
           "long_prefill_2": "1f0192038c425512912a1df7d56dca6670c50f01935e664f0219589b802cdc8d"}
# PR2: the G1 s1 text of each prompt (identical in every G1/X1/X2/X3/P1 sample of that run).
REF_G1 = {"decode_0": "2aa40adc7d4deb49fa84db4e070a48f4d14d10981ed8103efe83388292c84e9b",
          "decode_1": "643fa3d128e4c41c671ffc829f6d554531d259c2eb075820085d8981e985af80",
          "decode_2": "d64c91874f9ee22c50b943803c51eeeca79f4b613b4e551456b6735d57a318da",
          "decode_3": "9fffce519994640443407e699f48b4c1762e8a23af2b6719531f317892cc5231",
          "decode_4": "e3bb3209f760833a7e440b7371e35ceb90dbac2c459f65ab75131c179db1ebda",
          "decode_5": "31cbd1c1700fd7f64dda1d79142ea7d5e2d1b442ccc6e879c44bee7a3bfb439e"}
# Old arm means (n=3) of the 2026-10-15 runs: PR1 on old pr1 tip 93b74183, PR2 on old pr2 tip d0e19de2.
OLD_PR1 = {  # arm -> request -> (ttft_s, prefill_tok_s, decode_tok_s)
    "A": {"long_prefill": (220.4573, 9.0369, 2.1793), "long_prefill_2": (334.4023, 8.9862, 2.8145)},
    "B": {"long_prefill": (82.117, 24.2582, 2.0185), "long_prefill_2": (119.3343, 25.1814, 2.7474)},
    "C": {"long_prefill": (69.7723, 28.5502, 2.0217), "long_prefill_2": (106.2687, 28.2776, 2.734)},
}
OLD_PR2 = {"G1": 2.866, "X3": 4.0433, "X3_vs_G1": 1.4108}   # mean over prompts of decode tok/s, mean of 3 samples

PR1_REQS = ("long_prefill", "long_prefill_2")
PLAN = {"W": 1, "A": 3, "B": 3, "C": 3, "G1": 1, "X3": 1}   # samples expected per arm


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
            if "text" not in r:
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
                out[k.strip()] = v
    return out


def lines(path):
    return open(path, errors="replace").read().splitlines() if os.path.exists(path) else []


def load_sample(d):
    return {"dir": d, "k": int(re.sub(r"\D", "", os.path.basename(d)) or 0), "meta": kv(os.path.join(d, "sample.txt")),
            "rows": read_jsonl(os.path.join(d, "bench.jsonl")), "log": lines(os.path.join(d, "serve.log")),
            "env": kv(os.path.join(d, "engine_environ.txt"))}


def samples(base, arm):
    ds = sorted(glob.glob(os.path.join(base, arm, "s*")), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0))
    return [load_sample(d) for d in ds if os.path.isdir(d)]


def usage_changed(base):
    bad = []
    for st in glob.glob(os.path.join(base, "*", "s*", "sample.txt")):
        m = kv(st)
        if m.get("usage_before") and m.get("usage_after") and m["usage_before"] != m["usage_after"]:
            bad.append("%s: %s -> %s" % (os.path.relpath(os.path.dirname(st), base), m["usage_before"], m["usage_after"]))
    return bad


def stats(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    if not xs:
        return None
    m = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else None
    return {"mean": round(m, 4), "stdev": round(sd, 4) if sd is not None else None,
            "cv_pct": round(100 * sd / m, 2) if sd is not None and m else None,
            "median": round(statistics.median(xs), 4), "min": min(xs), "max": max(xs), "n": len(xs)}


def ratio(a, b):
    return round(a / b, 4) if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b else None


def gpu_max_mem(d, index):
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
        ref = REF_PR1.get(r.get("name")) or REF_G1.get(r.get("name"))
        print("   %-16s ttft=%-8s decode_tok_s=%-7s chunks=%-4s sha=%s ref=%s" % (
            r.get("name"), r.get("ttft_s"), r.get("decode_tok_s"), r.get("decode_chunks"), r["text_sha256"][:12],
            "-" if not ref else ("yes" if r["text_sha256"] == ref else "NO")))


def first_diff(a, b):
    return next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None if len(a) == len(b) else min(len(a), len(b)))


def env_check(s, must, must_not):
    env = s["env"]
    if not env:
        return ["engine_environ.txt missing"]
    bad = ["%s=%r (want %r)" % (k, env.get(k), v) for k, v in must.items() if env.get(k) != v]
    bad += ["%s set (=%r) but must be unset" % (k, env[k]) for k in must_not if k in env]
    return bad


COMMON = {"COLI_CUDA": "1", "COLI_GPU": "0", "DSV4_CUDA_DEVICE": "0", "CUDA_DENSE": "1", "COLI_CUDA_PIPE": "2",
          "COLI_CUDA_ATTN_BATCH": "1", "COLI_CUDA_MOE_BATCH": "1", "V4_MOE_REFILL_GROUP": "12",
          "V4_LOADER_LANES": "3", "V4_PREFIX_CKPT_DISK": "0"}
PR2_SWITCHES = ["V4_GPU_RESIDENT_FIRST", "DSV4_CUDA_PEER_DEVICE", "DSV4_CUDA_PEER_COMPUTE", "V4_SERVE_STATS_LOG"]

# arm: (tree key in tips.txt, client rows, must env beyond COMMON, must be unset, required log lines, forbidden log lines)
ARMS = {
    "W":  ("pr1", 2, {}, ["COLI_CUDA_MOE_DOUBLE"] + PR2_SWITCHES, [r"moe-batch=on bank=132-experts"], [r"moe-double=on", r"moe-batch=off"]),
    "A":  ("dev", 2, {}, ["COLI_CUDA_MOE_DOUBLE"] + PR2_SWITCHES, [r"moe-batch=off"], [r"moe-batch=on"]),
    "B":  ("pr1", 2, {}, ["COLI_CUDA_MOE_DOUBLE"] + PR2_SWITCHES, [r"moe-batch=on bank=132-experts"], [r"moe-double=on", r"moe-batch=off"]),
    "C":  ("pr1", 2, {"COLI_CUDA_MOE_DOUBLE": "1"}, PR2_SWITCHES, [r"moe-batch=on bank=132-experts", r"moe-double=on"], [r"moe-batch=off"]),
    "G1": ("pr2", 6, {"V4_SERVE_STATS_LOG": "1"}, ["COLI_CUDA_MOE_DOUBLE", "V4_GPU_RESIDENT_FIRST", "DSV4_CUDA_PEER_DEVICE", "DSV4_CUDA_PEER_COMPUTE"],
           [r"^v4_serve_turn "], [r"peer-mirrors"]),
    "X3": ("pr2", 6, {"V4_SERVE_STATS_LOG": "1", "V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1"},
           ["COLI_CUDA_MOE_DOUBLE", "DSV4_CUDA_PEER_COMPUTE"], [r"v4_gpu peer-mirrors device=1 .*mode=compute", r"^v4_serve_turn "],
           [r"mode=p2p-read", r"peer mirror pool off", r"peer device \d+ failed"]),
    # final5: R1 = G1 + VRAM-first only (one GPU); D0 = upstream dev on the decode client (no PR2 code at all)
    "R1": ("pr2", 6, {"V4_SERVE_STATS_LOG": "1", "V4_GPU_RESIDENT_FIRST": "1"},
           ["COLI_CUDA_MOE_DOUBLE", "DSV4_CUDA_PEER_DEVICE", "DSV4_CUDA_PEER_COMPUTE"], [r"^v4_serve_turn "], [r"peer-mirrors"]),
    "D0": ("dev", 6, {}, ["COLI_CUDA_MOE_DOUBLE"] + PR2_SWITCHES, [], [r"peer-mirrors", r"^v4_serve_turn "]),
}


def tips(base):
    t = kv(os.path.join(base, "meta", "tips.txt"))
    return {"dev": t.get("dev_sha", DEV_SHA), "pr1": t.get("pr1_sha"), "pr2": t.get("pr2_sha"), "raw": t}


def vram_first(log):
    turns = grep(log, r"^v4_serve_turn ")
    if not turns:
        return None
    m = re.search(r"vram_first_hits_total=(\d+)", turns[-1])
    return int(m.group(1)) if m else None


def premise(arm, s, tp):
    """problems that break the comparison (exit 1 in `check`), and notes (results, not aborts)"""
    key, nrows, extra, unset, need, forbid = ARMS[arm]
    m, log = s["meta"], s["log"]
    bad, notes = [], []
    if m.get("ready") != "1":
        bad.append("server never ready")
    if m.get("client_rc") != "0":
        bad.append("client_rc=%s" % m.get("client_rc"))
    if len(s["rows"]) != nrows:
        bad.append("%d of %d requests" % (len(s["rows"]), nrows))
    want = tp.get(key)
    if want and m.get("tree_head") != want:
        bad.append("tree_head %s != %s tip %s" % (m.get("tree_head"), key, want))
    if m.get("tree_dirty_lines") not in ("0", None):
        bad.append("tree had %s changed files" % m.get("tree_dirty_lines"))
    bad += env_check(s, dict(COMMON, **extra), unset)
    bad += ["missing log line /%s/" % p for p in need if not grep(log, p)]
    bad += ["forbidden log line /%s/: %s" % (p, grep(log, p)[0][:160]) for p in forbid if grep(log, p)]
    if arm == "G1" and vram_first(log) not in (0, None):
        bad.append("vram_first_hits_total=%s, want 0" % vram_first(log))
    if arm in ("X3", "R1") and not vram_first(log):
        bad.append("vram_first_hits_total=%s, want > 0" % vram_first(log))
    oom = grep(log, r"out of memory|Traceback")
    if oom:
        bad.append("serve.log: %s" % oom[0][:160])
    refs = REF_G1 if arm in ("G1", "X3", "R1", "D0") else REF_PR1
    for r in s["rows"]:
        ref = refs.get(r.get("name"))
        if ref and r["text_sha256"] != ref:
            notes.append("%s text %s differs from the reference %s" % (r.get("name"), r["text_sha256"][:12], ref[:12]))
    return bad, notes


def check(arm, d):
    base = os.path.dirname(os.path.dirname(os.path.abspath(d)))
    s = load_sample(d)
    bad, notes = premise(arm, s, tips(base))
    res = grep(s["log"], r"mirror-reserve=\d+MiB")
    print("check %s s%d: %s | reserve=%s | %s" % (arm, s["k"], "OK" if not bad else "PREMISE BROKEN: " + "; ".join(bad),
          re.findall(r"mirror-reserve=\d+MiB", " ".join(res)) or "-", "; ".join(notes) or "texts = reference"))
    return 1 if bad else 0


# ---------------------------------------------------------------- summary
def summarize_final4(base):
    tp = tips(base)
    out = {"kind": "final4", "tips": tp["raw"], "arms": {}, "comparisons": {}, "gates": {"problems": [], "notes": []}}
    P, N = out["gates"]["problems"], out["gates"]["notes"]
    for arm in PLAN:   # final5: final4 arms only
        ss = samples(base, arm)
        if len(ss) != PLAN[arm]:
            P.append("%s: %d of %d samples ran" % (arm, len(ss), PLAN[arm]))
        if not ss:
            continue
        a = {"samples": []}
        for s in ss:
            bad, notes = premise(arm, s, tp)
            x = {"k": s["k"], "seq": s["meta"].get("seq"), "start": s["meta"].get("start"), "wall_s": s["meta"].get("wall_s"),
                 "ready_s": s["meta"].get("ready_s"), "tree_head": s["meta"].get("tree_head"),
                 "binary_sha256": s["meta"].get("binary_sha256"), "client_rc": s["meta"].get("client_rc"),
                 "premise_problems": bad, "notes": notes,
                 "gpu0_max_mib": gpu_max_mem(s["dir"], 0), "gpu1_max_mib": gpu_max_mem(s["dir"], 1),
                 "mirror_reserve": re.findall(r"mirror-reserve=\d+MiB", " ".join(grep(s["log"], r"mirror-reserve="))),
                 "requests": {}}
            for r in s["rows"]:
                ref = (REF_G1 if ARMS[arm][0] == "pr2" else REF_PR1).get(r.get("name"))
                x["requests"][r.get("name")] = {k: r.get(k) for k in ("ttft_s", "prefill_tok_s", "decode_tok_s", "decode_chunks", "total_s")}
                x["requests"][r.get("name")]["text_sha256"] = r["text_sha256"]
                x["requests"][r.get("name")]["identical_to_reference"] = (r["text_sha256"] == ref) if ref else None
            if ARMS[arm][0] == "pr2":
                toks = [r.get("decode_tok_s") for r in s["rows"] if r.get("decode_tok_s")]
                x["mean_tok_s"] = round(statistics.mean(toks), 4) if toks else None
                x["engine_tok_s"] = [float(m.group(1)) for m in (re.search(r" tok_s=([\d.]+)", t) for t in grep(s["log"], r"^v4_serve_turn ")) if m]
                x["vram_first_hits_total"] = vram_first(s["log"])
                x["peer_lines"] = [l[:200] for l in grep(s["log"], r"peer-mirrors|peer mirror pool|peer device")[:3]]
            a["samples"].append(x)
            for p in bad:
                P.append("%s s%d: %s" % (arm, s["k"], p))
            for n in notes:
                (P if arm != "W" else N).append("%s s%d: %s" % (arm, s["k"], n))
        bins = {x["binary_sha256"] for x in a["samples"]}
        if len(bins) > 1:
            P.append("%s: samples ran different binaries %s" % (arm, sorted(b[:12] for b in bins if b)))
        if ARMS[arm][0] == "pr2":
            a["mean_tok_s"] = stats([x["mean_tok_s"] for x in a["samples"]])
            a["per_prompt_tok_s"] = {n: stats([x["requests"].get(n, {}).get("decode_tok_s") for x in a["samples"]]) for n in REF_G1}
            a["all_identical_to_G1_1015"] = all(x["requests"].get(n, {}).get("identical_to_reference") for x in a["samples"] for n in REF_G1)
        else:
            a["requests"] = {}
            for n in PR1_REQS:
                a["requests"][n] = {k: stats([x["requests"].get(n, {}).get(k) for x in a["samples"]])
                                    for k in ("ttft_s", "prefill_tok_s", "decode_tok_s")}
                a["requests"][n]["identical_to_dev_reference"] = all(x["requests"].get(n, {}).get("identical_to_reference") for x in a["samples"])
        out["arms"][arm] = a

    def mean_of(arm, n, k):
        st = out["arms"].get(arm, {}).get("requests", {}).get(n, {}).get(k)
        return st["mean"] if st else None

    def median_of(arm, n, k):
        st = out["arms"].get(arm, {}).get("requests", {}).get(n, {}).get(k)
        return st["median"] if st else None
    C = out["comparisons"]
    for n in PR1_REQS:
        c = {}
        for t, b in (("B", "A"), ("C", "B"), ("C", "A")):
            c["prefill_%s_vs_%s_mean" % (t, b)] = ratio(mean_of(t, n, "prefill_tok_s"), mean_of(b, n, "prefill_tok_s"))
            c["prefill_%s_vs_%s_median" % (t, b)] = ratio(median_of(t, n, "prefill_tok_s"), median_of(b, n, "prefill_tok_s"))
        c["decode_B_vs_A_mean"] = ratio(mean_of("B", n, "decode_tok_s"), mean_of("A", n, "decode_tok_s"))
        c["decode_C_vs_B_mean"] = ratio(mean_of("C", n, "decode_tok_s"), mean_of("B", n, "decode_tok_s"))
        for arm in ("A", "B", "C"):
            old = OLD_PR1[arm][n]
            c["%s_vs_1015_ttft" % arm] = ratio(mean_of(arm, n, "ttft_s"), old[0])
            c["%s_vs_1015_prefill" % arm] = ratio(mean_of(arm, n, "prefill_tok_s"), old[1])
            c["%s_vs_1015_decode" % arm] = ratio(mean_of(arm, n, "decode_tok_s"), old[2])
        C[n] = c
    g = (out["arms"].get("G1", {}).get("mean_tok_s") or {}).get("mean")
    x3 = (out["arms"].get("X3", {}).get("mean_tok_s") or {}).get("mean")
    C["pr2"] = {"G1_mean_tok_s": g, "X3_mean_tok_s": x3, "G1_vs_1015": ratio(g, OLD_PR2["G1"]), "X3_vs_1015": ratio(x3, OLD_PR2["X3"]),
                "X3_vs_G1": ratio(x3, g), "X3_vs_G1_1015": OLD_PR2["X3_vs_G1"]}
    lb, la = kv(os.path.join(base, "meta", "llama_before.txt")), kv(os.path.join(base, "meta", "llama_after.txt"))
    out["llama"] = {"stop_at": lb.get("llama_stop_at"), "restored_at": la.get("restored_at"), "health_ok": la.get("ok"),
                    "down_s": la.get("gpu_window_s"), "script_status": la.get("script_status")}
    if la and la.get("ok") != "1":
        P.append("llama not healthy after the restore (%s)" % la.get("health"))
    out["usage_history_changed"] = usage_changed(base)
    out["gates"]["pass"] = not P

    # ---- readable
    t = tp["raw"]
    L = ["final4 benchmark: %s" % base,
         "tips (resolved %s): pr1 %s %s (tree %s), pr2 %s %s (tree %s), dev %s" % (
             t.get("resolved_at"), t.get("pr1_branch"), t.get("pr1_sha"), t.get("pr1_tree"),
             t.get("pr2_branch"), t.get("pr2_sha"), t.get("pr2_tree"), tp["dev"]),
         "llama: stopped %s, restored %s, down %s s, health ok=%s, script status %s" % (
             out["llama"]["stop_at"], out["llama"]["restored_at"], out["llama"]["down_s"], out["llama"]["health_ok"], out["llama"]["script_status"])]
    for arm in ("W", "A", "B", "C"):
        a = out["arms"].get(arm)
        if not a:
            L.append("%s: no samples" % arm)
            continue
        for n in PR1_REQS:
            r = a["requests"][n]
            f = lambda k: ("%s (sd %s)" % (r[k]["mean"], r[k]["stdev"])) if r[k] else "-"
            L.append("%s %-15s n=%s ttft=%s prefill_tok_s=%s decode_tok_s=%s identical_to_dev_ref=%s%s" % (
                arm, n, (r["ttft_s"] or {}).get("n"), f("ttft_s"), f("prefill_tok_s"), f("decode_tok_s"),
                r["identical_to_dev_reference"], "  (warm-up, discarded)" if arm == "W" else ""))
        L.append("   per sample: " + "; ".join("s%d seq=%s wall=%ss reserve=%s %s" % (
            x["k"], x["seq"], x["wall_s"], ",".join(x["mirror_reserve"]) or "-",
            " ".join("%s:%s/%s/%s" % (n[-1] if n.endswith("2") else "1", q.get("ttft_s"), q.get("prefill_tok_s"), q.get("decode_tok_s"))
                     for n, q in x["requests"].items())) for x in a["samples"]))
    for n in PR1_REQS:
        c = C[n]
        L.append("PR1 %-15s prefill B/A=%s (median %s) C/B=%s C/A=%s | decode B/A=%s C/B=%s | vs 10-15: A pf %s dec %s, B pf %s dec %s, C pf %s dec %s" % (
            n, c["prefill_B_vs_A_mean"], c["prefill_B_vs_A_median"], c["prefill_C_vs_B_mean"], c["prefill_C_vs_A_mean"],
            c["decode_B_vs_A_mean"], c["decode_C_vs_B_mean"], c["A_vs_1015_prefill"], c["A_vs_1015_decode"],
            c["B_vs_1015_prefill"], c["B_vs_1015_decode"], c["C_vs_1015_prefill"], c["C_vs_1015_decode"]))
    for arm in ("G1", "X3"):
        a = out["arms"].get(arm)
        if not a:
            L.append("%s: no samples" % arm)
            continue
        for x in a["samples"]:
            L.append("%s s%d mean_tok_s=%s per_prompt=%s engine=%s identical_to_G1_1015=%s vram_first=%s gpu0_max=%s gpu1_max=%s %s" % (
                arm, x["k"], x["mean_tok_s"], [x["requests"].get(n, {}).get("decode_tok_s") for n in REF_G1], x["engine_tok_s"],
                [x["requests"].get(n, {}).get("identical_to_reference") for n in REF_G1], x["vram_first_hits_total"],
                x["gpu0_max_mib"], x["gpu1_max_mib"], (x["peer_lines"] or [""])[0][:120]))
    p = C["pr2"]
    L.append("PR2 G1=%s (10-15 %s, ratio %s) X3=%s (10-15 %s, ratio %s) X3/G1=%s (10-15 %s)" % (
        p["G1_mean_tok_s"], OLD_PR2["G1"], p["G1_vs_1015"], p["X3_mean_tok_s"], OLD_PR2["X3"], p["X3_vs_1015"], p["X3_vs_G1"], OLD_PR2["X3_vs_G1"]))
    L.append("usage history (.coli_usage) changed during: %s" % (out["usage_history_changed"] or "no sample"))
    L.append("GATES: %s (%d problems)" % ("PASS" if out["gates"]["pass"] else "FAIL", len(P)))
    L += ["  !! " + x for x in P] + ["  note: " + x for x in N]
    return out, L


# ---------------------------------------------------------------- selftest
def _fake(base, arm, k, seq, tree_head, rows, log, env_extra, rc="0", bin_sha="b" * 64):
    d = os.path.join(base, arm, "s%d" % k)
    os.makedirs(d)
    with open(os.path.join(d, "sample.txt"), "w") as f:
        f.write("arm=%s\nsample=%d\nseq=%d\ntree_head=%s\ntree_dirty_lines=0\nbinary_sha256=%s\nready=1\nready_s=4\n"
                "client_rc=%s\nwall_s=200\nusage_before=size=0 mtime=1\nusage_after=size=0 mtime=1\nstart=x\n" % (arm, k, seq, tree_head, bin_sha, rc))
    with open(os.path.join(d, "bench.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(d, "serve.log"), "w") as f:
        f.write("\n".join(log) + "\nOpenAI-compatible API listening on http://127.0.0.1:8090/v1\n")
    env = dict(COMMON, PATH="/usr/bin", **env_extra)
    with open(os.path.join(d, "engine_environ.txt"), "w") as f:
        f.write("\n".join("%s=%s" % kv_ for kv_ in sorted(env.items())) + "\n")
    with open(os.path.join(d, "nvsmi.csv"), "w") as f:
        f.write("timestamp, index, memory.used [MiB]\nt, 0, 21214 MiB\nt, 1, %s MiB\n" % (23137 if arm == "X3" else 83))
    return d


def selftest():
    tmp = tempfile.mkdtemp(prefix="final4-selftest-")
    try:
        base = os.path.join(tmp, "bench")
        os.makedirs(os.path.join(base, "meta"))
        P1, P2 = "1" * 40, "2" * 40
        with open(os.path.join(base, "meta", "tips.txt"), "w") as f:
            f.write("resolved_at=now\npr1_branch=pr1/v6\npr1_sha=%s\npr1_tree=t1\npr2_branch=pr2/v5\npr2_sha=%s\npr2_tree=t2\ndev_sha=%s\n" % (P1, P2, DEV_SHA))
        with open(os.path.join(base, "meta", "llama_after.txt"), "w") as f:
            f.write("restored_at=now\nhealth={\"status\":\"ok\"}\nok=1\ngpu_window_s=4700\nscript_status=0\n")

        def pr1rows(ttft, dec, wrong=False):
            return [{"name": "long_prefill", "ttft_s": ttft, "prefill_tok_s": round(1992 / ttft, 4), "decode_tok_s": dec, "decode_chunks": 25,
                     "text": "x", "text_sha256": REF_PR1["long_prefill"] if not wrong else "f" * 64},
                    {"name": "long_prefill_2", "ttft_s": ttft * 1.45, "prefill_tok_s": round(3005 / (ttft * 1.45), 4), "decode_tok_s": dec + 0.7,
                     "decode_chunks": 60, "text": "y", "text_sha256": REF_PR1["long_prefill_2"]}]

        def decrows(tok):
            return [{"name": n, "ttft_s": 4.0, "decode_tok_s": tok, "decode_chunks": 300, "decode_s": 100, "total_s": 104,
                     "text": n, "text_sha256": h} for n, h in REF_G1.items()]
        bank = ["v4_gpu mirror-reserve=3100MiB (132-slot bank)", "v4_gpu moe-batch=on bank=132-experts"]
        seq = 0
        seq += 1; _fake(base, "W", 1, seq, P1, pr1rows(82.0, 2.15), bank, {})
        for k in (1, 2, 3):
            seq += 1; _fake(base, "A", k, seq, DEV_SHA, pr1rows(220.0, 2.18), ["v4_gpu moe-batch=off (unsupported model geometry)"], {}, bin_sha="a" * 64)
            seq += 1; _fake(base, "B", k, seq, P1, pr1rows(82.0, 2.15), bank, {})
            seq += 1; _fake(base, "C", k, seq, P1, pr1rows(70.0, 2.15), bank + ["v4_gpu moe-double=on"], {"COLI_CUDA_MOE_DOUBLE": "1"})
        turn = "v4_serve_turn prompt=33 completion=300 tok_s=%s vram_first_hits_total=%d"
        seq += 1; _fake(base, "G1", 1, seq, P2, decrows(2.87), [turn % ("2.87", 0)] * 6, {"V4_SERVE_STATS_LOG": "1"}, bin_sha="c" * 64)
        seq += 1; _fake(base, "X3", 1, seq, P2, decrows(4.04), ["v4_gpu peer-mirrors device=1 reserve=20000MB mode=compute"] + [turn % ("4.04", 5000)] * 6,
                        {"V4_SERVE_STATS_LOG": "1", "V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1"}, bin_sha="c" * 64)
        out, L = summarize_final4(base)
        out5, L5 = summarize_final5(base)   # final5: smoke test on the same synthetic samples
        assert out5["arms"]["B"]["all_texts_identical_to_run_dev"] and not out5["gates"]["pass"]   # D0/R1 missing -> problems
        assert out["gates"]["pass"], out["gates"]["problems"]
        c = out["comparisons"]["long_prefill"]
        assert abs(c["prefill_B_vs_A_mean"] - 220.0 / 82.0) < 0.01, c
        assert out["comparisons"]["pr2"]["X3_vs_G1"] == round(4.04 / 2.87, 4)
        assert out["arms"]["G1"]["all_identical_to_G1_1015"] and out["arms"]["A"]["requests"]["long_prefill"]["identical_to_dev_reference"]
        assert check("X3", os.path.join(base, "X3", "s1")) == 0 and check("B", os.path.join(base, "B", "s2")) == 0
        # broken premises and a changed text must be caught
        bad = os.path.join(tmp, "bad")
        shutil.copytree(base, bad)
        shutil.rmtree(os.path.join(bad, "C", "s3"))
        _fake(bad, "C", 3, 99, P1, pr1rows(70.0, 2.15, wrong=True), bank, {})          # no moe-double env/line, wrong text
        shutil.rmtree(os.path.join(bad, "X3", "s1"))
        _fake(bad, "X3", 1, 98, P2, decrows(3.0), [turn % ("3.0", 0)], {"V4_SERVE_STATS_LOG": "1", "COLI_CUDA_MOE_DOUBLE": "1"}, rc="124", bin_sha="c" * 64)
        out2, L2 = summarize_final4(bad)
        probs = " | ".join(out2["gates"]["problems"])
        assert not out2["gates"]["pass"]
        for want in ("C s3: COLI_CUDA_MOE_DOUBLE=None", "C s3: missing log line /moe-double=on/", "C s3: long_prefill text",
                     "X3 s1: client_rc=124", "X3 s1: COLI_CUDA_MOE_DOUBLE set", "X3 s1: vram_first_hits_total=0, want > 0",
                     "X3 s1: missing log line /v4_gpu peer-mirrors"):
            assert want in probs, (want, probs)
        assert check("X3", os.path.join(bad, "X3", "s1")) == 1 and check("C", os.path.join(bad, "C", "s3")) == 1
        print("selftest ok")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def strip_texts(o):
    if isinstance(o, dict):
        return {k: strip_texts(v) for k, v in o.items() if k not in ("texts", "text")}
    if isinstance(o, list):
        return [strip_texts(v) for v in o]
    return o


# ---------------------------------------------------------------- final5 summary
PLAN5 = {"W": 1, "A": 3, "B": 3, "C": 3, "D0": 3, "G1": 3, "R1": 3, "X3": 3}
PR1_ARMS5, PR2_ARMS5 = ("W", "A", "B", "C"), ("D0", "G1", "R1", "X3")
DEC_REQS = tuple(sorted(REF_G1))


def summarize_final5(base):
    """Like summarize_final4, plus D0/R1 and 3 samples per decode arm. Texts are compared with THIS run's
    dev arm (A s1 for the prefill client, D0 s1 for the decode client): a difference is a gate problem.
    A difference from the older fixed references is only a note (the base moved by 574 commits)."""
    tp = tips(base)
    out = {"kind": "final5", "tips": tp["raw"], "arms": {}, "comparisons": {}, "gates": {"problems": [], "notes": []}}
    P, N = out["gates"]["problems"], out["gates"]["notes"]
    ref_run = {}
    for dev_arm in ("A", "D0"):
        ss = samples(base, dev_arm)
        if ss:
            ref_run.update({r.get("name"): r["text_sha256"] for r in ss[0]["rows"]})
    out["run_reference_texts"] = ref_run
    for arm in PR1_ARMS5 + PR2_ARMS5:
        ss = samples(base, arm)
        if len(ss) != PLAN5[arm]:
            P.append("%s: %d of %d samples ran" % (arm, len(ss), PLAN5[arm]))
        if not ss:
            continue
        a = {"samples": []}
        for s in ss:
            bad, notes = premise(arm, s, tp)
            x = {"k": s["k"], "seq": s["meta"].get("seq"), "start": s["meta"].get("start"), "wall_s": s["meta"].get("wall_s"),
                 "ready_s": s["meta"].get("ready_s"), "tree_head": s["meta"].get("tree_head"),
                 "binary_sha256": s["meta"].get("binary_sha256"), "client_rc": s["meta"].get("client_rc"),
                 "premise_problems": bad, "notes_vs_old_references": notes,
                 "gpu0_max_mib": gpu_max_mem(s["dir"], 0), "gpu1_max_mib": gpu_max_mem(s["dir"], 1),
                 "mirror_reserve": re.findall(r"mirror-reserve=\d+MiB", " ".join(grep(s["log"], r"mirror-reserve="))),
                 "requests": {}}
            for r in s["rows"]:
                n = r.get("name")
                x["requests"][n] = {k: r.get(k) for k in ("ttft_s", "prefill_tok_s", "decode_tok_s", "decode_chunks", "total_s")}
                x["requests"][n]["text_sha256"] = r["text_sha256"]
                x["requests"][n]["identical_to_run_dev"] = (r["text_sha256"] == ref_run[n]) if n in ref_run else None
                if n in ref_run and r["text_sha256"] != ref_run[n]:
                    (P if arm != "W" else N).append("%s s%d: %s text %s differs from this run's dev text %s" % (
                        arm, s["k"], n, r["text_sha256"][:12], ref_run[n][:12]))
            if arm in PR2_ARMS5:
                toks = [r.get("decode_tok_s") for r in s["rows"] if r.get("decode_tok_s")]
                x["mean_tok_s"] = round(statistics.mean(toks), 4) if toks else None
                x["vram_first_hits_total"] = vram_first(s["log"])
                x["peer_lines"] = [l[:200] for l in grep(s["log"], r"peer-mirrors|peer mirror pool|peer device")[:3]]
            a["samples"].append(x)
            P.extend("%s s%d: %s" % (arm, s["k"], p) for p in bad)
            N.extend("%s s%d: %s" % (arm, s["k"], n) for n in notes)
        bins = {x["binary_sha256"] for x in a["samples"]}
        if len(bins) > 1:
            P.append("%s: samples ran different binaries %s" % (arm, sorted(b[:12] for b in bins if b)))
        if arm in PR2_ARMS5:
            a["mean_tok_s"] = stats([x["mean_tok_s"] for x in a["samples"]])
            a["per_prompt_tok_s"] = {n: stats([x["requests"].get(n, {}).get("decode_tok_s") for x in a["samples"]]) for n in DEC_REQS}
        else:
            a["requests"] = {n: {k: stats([x["requests"].get(n, {}).get(k) for x in a["samples"]])
                                 for k in ("ttft_s", "prefill_tok_s", "decode_tok_s")} for n in PR1_REQS}
        a["all_texts_identical_to_run_dev"] = all(v.get("identical_to_run_dev") is not False
                                                  for x in a["samples"] for v in x["requests"].values())
        out["arms"][arm] = a

    def st(arm, n, k, f):
        v = out["arms"].get(arm, {}).get("requests", {}).get(n, {}).get(k)
        return v[f] if v else None
    C = out["comparisons"]
    for n in PR1_REQS:
        c = {}
        for t2, b2 in (("B", "A"), ("C", "B"), ("C", "A")):
            for f in ("mean", "median"):
                c["ttft_%s_over_%s_%s" % (b2, t2, f)] = ratio(st(b2, n, "ttft_s", f), st(t2, n, "ttft_s", f))
                c["prefill_%s_vs_%s_%s" % (t2, b2, f)] = ratio(st(t2, n, "prefill_tok_s", f), st(b2, n, "prefill_tok_s", f))
            c["decode_%s_vs_%s_mean" % (t2, b2)] = ratio(st(t2, n, "decode_tok_s", "mean"), st(b2, n, "decode_tok_s", "mean"))
        C[n] = c

    def dm(arm, f="mean"):
        v = out["arms"].get(arm, {}).get("mean_tok_s")
        return v[f] if v else None
    C["decode"] = {arm + "_mean_tok_s": dm(arm) for arm in PR2_ARMS5}
    C["decode"].update({arm + "_median_tok_s": dm(arm, "median") for arm in PR2_ARMS5})
    for t2, b2 in (("G1", "D0"), ("R1", "G1"), ("X3", "R1"), ("X3", "G1"), ("X3", "D0")):
        C["decode"]["%s_vs_%s_mean" % (t2, b2)] = ratio(dm(t2), dm(b2))
        C["decode"]["%s_vs_%s_median" % (t2, b2)] = ratio(dm(t2, "median"), dm(b2, "median"))
    lb, la = kv(os.path.join(base, "meta", "llama_before.txt")), kv(os.path.join(base, "meta", "llama_after.txt"))
    out["llama"] = {"stop_at": lb.get("llama_stop_at"), "restored_at": la.get("restored_at"), "health_ok": la.get("ok"),
                    "down_s": la.get("gpu_window_s"), "script_status": la.get("script_status")}
    if la and la.get("ok") != "1":
        P.append("llama not healthy after the restore (%s)" % la.get("health"))
    out["usage_history_changed"] = usage_changed(base)
    out["gates"]["pass"] = not P

    L = ["final5 benchmark: %s" % base,
         "dev %s | pr1 %s | pr2 %s" % (tp["dev"], tp["pr1"], tp["pr2"]),
         "gates: %s (%d problems, %d notes)" % ("PASS" if out["gates"]["pass"] else "FAIL", len(P), len(N))]
    L += ["  problem: " + p for p in P[:40]]
    L.append("")
    L.append("Prefill client (bench_pr1.py): mean [median] over samples; texts vs this run's dev (A s1)")
    for arm in PR1_ARMS5:
        a = out["arms"].get(arm)
        if not a:
            continue
        for n in PR1_REQS:
            r = a["requests"][n]
            L.append("  %-2s %-15s ttft %8s s [%8s]  prefill %7s tok/s [%7s]  decode %6s tok/s  n=%s  texts_same=%s%s" % (
                arm, n, (r["ttft_s"] or {}).get("mean"), (r["ttft_s"] or {}).get("median"),
                (r["prefill_tok_s"] or {}).get("mean"), (r["prefill_tok_s"] or {}).get("median"),
                (r["decode_tok_s"] or {}).get("mean"), (r["ttft_s"] or {}).get("n"), a["all_texts_identical_to_run_dev"],
                "  (warm-up, discarded)" if arm == "W" else ""))
    for n in PR1_REQS:
        c = C[n]
        L.append("  %s: TTFT speedup B vs A %sx (median %sx), C vs A %sx (median %sx), C vs B %sx; decode B/A %s C/B %s" % (
            n, c["ttft_A_over_B_mean"], c["ttft_A_over_B_median"], c["ttft_A_over_C_mean"], c["ttft_A_over_C_median"],
            c["ttft_B_over_C_mean"], c["decode_B_vs_A_mean"], c["decode_C_vs_B_mean"]))
    L.append("")
    L.append("Decode client (bench_decode.py, 6 x 300 tokens): per-sample mean over prompts; texts vs this run's dev (D0 s1)")
    for arm in PR2_ARMS5:
        a = out["arms"].get(arm)
        if not a:
            continue
        L.append("  %-2s mean %s tok/s [median %s, cv %s%%, n=%s] samples %s  vram_first_hits %s  texts_same=%s" % (
            arm, (a["mean_tok_s"] or {}).get("mean"), (a["mean_tok_s"] or {}).get("median"), (a["mean_tok_s"] or {}).get("cv_pct"),
            (a["mean_tok_s"] or {}).get("n"), [x["mean_tok_s"] for x in a["samples"]],
            [x["vram_first_hits_total"] for x in a["samples"]], a["all_texts_identical_to_run_dev"]))
    d = C["decode"]
    L.append("  G1/D0 %s (PR2 switches off vs dev)  R1/G1 %s (VRAM-first)  X3/R1 %s (second GPU)  X3/G1 %s  X3/D0 %s" % (
        d["G1_vs_D0_mean"], d["R1_vs_G1_mean"], d["X3_vs_R1_mean"], d["X3_vs_G1_mean"], d["X3_vs_D0_mean"]))
    L.append("")
    L.append("llama: stopped %s, restored %s, healthy=%s, down %s s, script status %s" % (
        out["llama"]["stop_at"], out["llama"]["restored_at"], out["llama"]["health_ok"], out["llama"]["down_s"], out["llama"]["script_status"]))
    if N:
        L.append("notes (%d): %s" % (len(N), "; ".join(N[:8])))
    return out, L


def main(argv):
    if len(argv) >= 2 and argv[0] == "brief":
        brief(argv[1]); return 0
    if len(argv) >= 3 and argv[0] == "check":
        return check(argv[1], argv[2])
    if len(argv) >= 1 and argv[0] == "selftest":
        return selftest()
    if len(argv) >= 3 and argv[0] == "summarize" and argv[1] == "final5":
        out, L = summarize_final5(argv[2])
        json.dump(out, open(os.path.join(argv[2], "summary.json"), "w"), indent=2)
        print("\n".join(L))
        return 0 if out["gates"]["pass"] else 2
    if len(argv) >= 3 and argv[0] == "summarize" and argv[1] == "final4":
        out, L = summarize_final4(argv[2])
        json.dump(strip_texts(out), open(os.path.join(argv[2], "summary.json"), "w"), indent=1)
        print("\n".join(L))
        return 0
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
