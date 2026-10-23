#!/usr/bin/env python3
"""final6_tools.py (2026-10-22) -- checks and summary for run_final6.sh.

The question (JustVugg on #2264): DSV4_MM_WARP is on by default, so does PR #1 change the
default configuration (no COLI_CUDA_MOE_BATCH) on the full DeepSeek V4 Flash? Around it, the
untested items of open-problems.md that this box can test, in one long GPU window.

Arms (ARMS below): tree, model, client, the reference group whose first sample every other
sample of the group must match byte for byte, environment that must / must not be set, log
patterns that must / must not appear. No fixed references: the full model never ran before, and
PR #2's REAP references are from older bases (a difference from them is only a note).

  final6_tools.py brief  BENCH.JSONL
  final6_tools.py check  ARM SAMPLE_DIR
  final6_tools.py summarize final6 OUT_DIR
  final6_tools.py selftest
"""
import hashlib, json, os, re, statistics, sys, tempfile

BANK = r"moe-batch=on bank=256-experts"
PR2_OFF = ["V4_GPU_RESIDENT_FIRST", "DSV4_CUDA_PEER_DEVICE", "DSV4_CUDA_PEER_COMPUTE", "DSV4_HYBRID", "DSV4_CUDA_BATCHED"]
NOT_DEFAULT = ["COLI_CUDA_MOE_BATCH", "COLI_CUDA_MOE_DOUBLE", "DSV4_MM_WARP", "DSV4_MM_WARP_VERIFY", "V4_DRAFT", "V4_MTP"]

# arm: (tree, model, group, must_env, must_not_env, must_log, must_not_log, rows, core)
#   group: the arms whose texts must all equal the group's first sample (per request name)
#   core: a premise failure here counts toward the abort; elsewhere it only skips the arm
ARMS = {
    # 1. the question: default configuration, full model, dev vs PR #1
    "W":  ("pr1", "full", "full", {}, NOT_DEFAULT, [], [r"moe-batch=on"], 2, False),
    "P":  ("pr1", "full", "full", {}, NOT_DEFAULT, [], [r"moe-batch=on"], 2, True),
    "A":  ("dev", "full", "full", {}, NOT_DEFAULT, [], [r"moe-batch=on"], 2, True),
    "B":  ("pr1", "full", "full", {}, NOT_DEFAULT, [], [r"moe-batch=on"], 2, True),
    "V":  ("pr1", "full", "full", {"DSV4_MM_WARP_VERIFY": "1"}, [v for v in NOT_DEFAULT if v != "DSV4_MM_WARP_VERIFY"],
           [r"mm-warp verify calls=\d+ mismatches=0"], [r"mm-warp verify calls=\d+ mismatches=[1-9]", r"moe-batch=on"], 2, False),
    # 2. the 256-expert bank, never run before (F7/F8/F12): dev vs PR #1, and the double bank
    "AK": ("dev", "full", "full", {"COLI_CUDA_MOE_BATCH": "1"}, ["COLI_CUDA_MOE_DOUBLE", "V4_DRAFT"], [BANK], [r"moe-batch=off"], 2, False),
    "BK": ("pr1", "full", "full", {"COLI_CUDA_MOE_BATCH": "1"}, ["COLI_CUDA_MOE_DOUBLE", "V4_DRAFT"], [BANK, r"mirror-reserve="], [r"moe-batch=off"], 2, False),
    "PK": ("pr1", "full", "full", {"COLI_CUDA_MOE_BATCH": "1"}, ["COLI_CUDA_MOE_DOUBLE", "V4_DRAFT"], [BANK, r"mirror-reserve="], [r"moe-batch=off"], 2, False),
    "BD": ("pr1", "full", "full", {"COLI_CUDA_MOE_BATCH": "1", "COLI_CUDA_MOE_DOUBLE": "1"}, ["V4_DRAFT"], [BANK, r"moe-double=on"], [r"moe-batch=off"], 2, False),
    # 3. a long prompt (~8k tokens, --ctx 16384), default configuration, one sample each
    "LA": ("dev", "full", "long", {}, NOT_DEFAULT, [], [r"moe-batch=on"], 1, False),
    "LB": ("pr1", "full", "long", {}, NOT_DEFAULT, [], [r"moe-batch=on"], 1, False),
    # 4. speculative decoding with the bank on (F2): n-gram drafts on the repetitive first prompt.
    #    The texts must equal the plain runs' (a draft never changes a token); bank lines are counted.
    "AS": ("dev", "full", "full", {"COLI_CUDA_MOE_BATCH": "1", "V4_DRAFT": "4"}, ["COLI_CUDA_MOE_DOUBLE"], [BANK], [], 1, False),
    "BS": ("pr1", "full", "full", {"COLI_CUDA_MOE_BATCH": "1", "V4_DRAFT": "4"}, ["COLI_CUDA_MOE_DOUBLE"], [BANK], [], 1, False),
    # 5. two requests at once (default configuration): each text must equal its sequential one
    "CC": ("pr1", "full", "full", {}, NOT_DEFAULT, [], [r"moe-batch=on"], 2, False),
    # 8. Vulkan + CUDA builds (F22: the router's Vulkan gate matmul), REAP-150B prefill with
    #    COLI_CUDA_MOE_BATCH=1 and COLI_VULKAN=1; RB is the same without Vulkan (the reference)
    "RB": ("pr1", "reap", "reap-pf", {"COLI_CUDA_MOE_BATCH": "1"}, ["COLI_VULKAN", "V4_DRAFT"], [r"moe-batch=on bank=132-experts"], [], 2, False),
    "VA": ("dev-vk", "reap", "reap-pf", {"COLI_CUDA_MOE_BATCH": "1", "COLI_VULKAN": "1"}, ["V4_DRAFT"], [], [], 2, False),
    "VB": ("pr1-vk", "reap", "reap-pf", {"COLI_CUDA_MOE_BATCH": "1", "COLI_VULKAN": "1"}, ["V4_DRAFT"], [r"moe-batch=on bank=132-experts"], [], 2, False),
    #    pure Vulkan (CUDA off): dev vs PR #1, compared with each other (Vulkan's matmuls may differ from CUDA's)
    "QA": ("dev-vk", "reap", "reap-vk", {"COLI_VULKAN": "1", "DSV4_CUDA": "0", "COLI_CUDA": "0"}, ["COLI_CUDA_MOE_BATCH", "V4_DRAFT"],
           [], [r"moe-batch=on", r"DSV4 CUDA\] device"], 2, False),
    "QB": ("pr1-vk", "reap", "reap-vk", {"COLI_VULKAN": "1", "DSV4_CUDA": "0", "COLI_CUDA": "0"}, ["COLI_CUDA_MOE_BATCH", "V4_DRAFT"],
           [], [r"moe-batch=on", r"DSV4 CUDA\] device"], 2, False),
    # 9. a ~16k-token prompt with the bank on (--ctx 32768), dev vs PR #1
    "KA": ("dev", "full", "long16", {"COLI_CUDA_MOE_BATCH": "1"}, ["V4_DRAFT"], [BANK], [r"moe-batch=off"], 1, False),
    "KB": ("pr1", "full", "long16", {"COLI_CUDA_MOE_BATCH": "1"}, ["V4_DRAFT"], [BANK], [r"moe-batch=off"], 1, False),
    # 10. MTP drafting with the bank on, 24 tokens: the two texts must match, and (a note) be a prefix of A's
    "MA": ("dev", "full", "mtp", {"COLI_CUDA_MOE_BATCH": "1", "V4_MTP": "1", "V4_DRAFT": "3"}, [], [BANK], [], 1, False),
    "MB": ("pr1", "full", "mtp", {"COLI_CUDA_MOE_BATCH": "1", "V4_MTP": "1", "V4_DRAFT": "3"}, [], [BANK], [], 1, False),
    # 11. soak / multi-turn: PR #2 with everything on, one server, the 6-turn conversation 3 times
    "SK": ("pr2", "reap", "soak", {"COLI_CUDA_MOE_BATCH": "1", "V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1"}, [], [], [], 18, False),
    # 6. PR #2 after the dev merge, on REAP, final5's decode arms (COLI_CUDA_MOE_BATCH=1 as in final5)
    "WR": ("pr2", "reap", "reap-dec", {"COLI_CUDA_MOE_BATCH": "1"}, PR2_OFF, [], [], 6, False),
    "D0": ("dev", "reap", "reap-dec", {"COLI_CUDA_MOE_BATCH": "1"}, PR2_OFF, [], [], 6, False),
    "G1": ("pr2", "reap", "reap-dec", {"COLI_CUDA_MOE_BATCH": "1", "V4_SERVE_STATS_LOG": "1"}, PR2_OFF, [r"^v4_serve_turn"], [], 6, False),
    "R1": ("pr2", "reap", "reap-dec", {"V4_GPU_RESIDENT_FIRST": "1"}, ["DSV4_CUDA_PEER_DEVICE", "DSV4_HYBRID", "DSV4_CUDA_BATCHED"], [r"vram_first_hits_total=[1-9]"], [], 6, False),
    "X3": ("pr2", "reap", "reap-dec", {"V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1"}, ["DSV4_CUDA_PEER_COMPUTE", "DSV4_HYBRID", "DSV4_CUDA_BATCHED"],
           [r"peer-mirrors device=1"], [], 6, False),
    # 7. PR #2 paths never run end to end: peer pool without peer compute (F45), hybrid, unbatched
    "P0": ("pr2", "reap", "reap-dec", {"V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1", "DSV4_CUDA_PEER_COMPUTE": "0"}, [], [r"peer-mirrors device=1"], [], 6, False),
    "HY": ("pr2", "reap", "reap-dec", {"V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1", "DSV4_HYBRID": "1"}, [], [], [], 6, False),
    "NB": ("pr2", "reap", "reap-dec", {"V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1", "DSV4_CUDA_BATCHED": "0"}, [], [], [], 6, False),
}
# final5's six REAP decode texts (on 736a0682, identical to 2026-10-15): a difference is a note only
REF_G1_OLD = {"decode_0": "2aa40adc7d4deb49fa84db4e070a48f4d14d10981ed8103efe83388292c84e9b"}
TIMED = ["A", "B", "AK", "BK", "BD", "D0", "G1", "R1", "X3"]
COMPARE = [("B", "A", "PR #1 vs dev, default configuration"), ("BK", "AK", "PR #1 vs dev, 256-expert bank"),
           ("BD", "BK", "double bank vs single"), ("G1", "D0", "PR #2 switches off vs dev"),
           ("R1", "G1", "VRAM-first"), ("X3", "R1", "second GPU"), ("LB", "LA", "PR #1 vs dev, ~8k prompt"),
           ("KB", "KA", "PR #1 vs dev, ~16k prompt, bank"), ("VB", "VA", "PR #1 vs dev, Vulkan + CUDA build"), ("QB", "QA", "PR #1 vs dev, pure Vulkan"),
           ("MB", "MA", "PR #1 vs dev, MTP drafting")]


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_jsonl(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                r = json.loads(line)
                if "text_sha256" not in r and "text" in r:
                    r["text_sha256"] = sha(r["text"])
                rows.append(r)
    except FileNotFoundError:
        pass
    return rows


def kv(path):
    out = {}
    try:
        with open(path) as f:
            for line in f:
                if "=" in line:
                    k, v = line.rstrip("\n").split("=", 1)
                    out[k] = v
    except FileNotFoundError:
        pass
    return out


def text(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def load_sample(d):
    return {"dir": d, "k": int(os.path.basename(d)[1:]), "meta": kv(os.path.join(d, "sample.txt")),
            "rows": read_jsonl(os.path.join(d, "bench.jsonl")), "log": text(os.path.join(d, "serve.log"))}


def samples(base, arm):
    d = os.path.join(base, arm)
    if not os.path.isdir(d):
        return []
    return sorted((load_sample(os.path.join(d, s)) for s in os.listdir(d) if re.fullmatch(r"s\d+", s)),
                  key=lambda s: s["k"])


def gpu_peak_mib(d, index):
    peak = 0
    for line in text(os.path.join(d, "nvsmi.csv")).splitlines()[1:]:
        f = [x.strip() for x in line.split(",")]
        if len(f) > 2 and f[1] == str(index):
            m = re.match(r"(\d+)", f[2])
            if m:
                peak = max(peak, int(m.group(1)))
    return peak


def gpu_series_mib(d, index):
    out = []
    for line in text(os.path.join(d, "nvsmi.csv")).splitlines()[1:]:
        f = [x.strip() for x in line.split(",")]
        if len(f) > 2 and f[1] == str(index):
            m = re.match(r"(\d+)", f[2])
            if m:
                out.append(int(m.group(1)))
    return out


def env_of(s):
    return dict(kv.split("=", 1) for kv in s["meta"].get("server_env", "").split() if "=" in kv)


def premise(arm, s):
    """Problems that make a sample unusable (not a slow one: a wrong one)."""
    tree, model, group, must_env, must_not_env, must_log, must_not_log, rows, core = ARMS[arm]
    p = []
    m = s["meta"]
    if m.get("ready") != "1":
        p.append("server never ready")
    if m.get("client_rc") != "0":
        p.append("client rc %s" % m.get("client_rc"))
    if m.get("tree_dirty_lines", "0") != "0":
        p.append("tree has local changes")
    env = env_of(s)
    for k, v in must_env.items():
        if env.get(k) != v:
            p.append("%s=%s not in the server environment" % (k, v))
    for k in must_not_env:
        if k in env and k not in must_env:
            p.append("%s set in the server environment" % k)
    log = s["log"]
    for pat in must_log:
        if not re.search(pat, log, re.M):
            p.append("no /%s/ in serve.log" % pat)
    for pat in must_not_log:
        if re.search(pat, log, re.M):
            p.append("/%s/ in serve.log" % pat)
    # "[DSV4 CUDA] <what> allocation: out of memory" is a handled failure (the engine falls back and goes
    # on): counted per sample (handled_oom) and compared between arms, not a premise. Anything else is.
    rest = HANDLED_OOM.sub("", log)
    if re.search(r"out of memory|Traceback|Segmentation fault|Aborted|core dumped", rest, re.I):
        p.append("out of memory / crash in serve.log")
    if len(s["rows"]) < rows:
        p.append("%d request rows, expected %d" % (len(s["rows"]), rows))
    if arm == "SK":
        first = {r["name"].split("_", 1)[1]: r.get("text_sha256") for r in s["rows"] if r.get("name", "").startswith("rep1_")}
        for r in s["rows"]:
            t = r.get("name", "").split("_", 1)[-1]
            if t in first and r.get("text_sha256") != first[t]:
                p.append("%s differs from rep1_%s (the same conversation on the same server)" % (r["name"], t))
        mem = gpu_series_mib(s["dir"], 0)
        if len(mem) > 20:
            q = len(mem) // 4
            early, late = max(mem[q:2 * q] or [0]), max(mem[3 * q:] or [0])
            if late > early + 1024:
                p.append("GPU0 memory grew %d -> %d MiB over the soak" % (early, late))
    if arm in ("AS", "BS"):
        n = len(re.findall(BANK, log))
        if n != len(s["rows"]):
            p.append("%d bank creations for %d requests: speculative batches re-created the bank (F2)" % (n, len(s["rows"])))
    return p


HANDLED_OOM = re.compile(r"^\[DSV4 CUDA\] [a-z0-9 _-]+ allocation: out of memory$", re.M)


def handled_oom(log):
    """{what: count} of handled allocation failures ("[DSV4 CUDA] kv comp allocation: out of memory")."""
    out = {}
    for m in HANDLED_OOM.finditer(log):
        what = m.group(0)[len("[DSV4 CUDA] "):].split(" allocation")[0]
        out[what] = out.get(what, 0) + 1
    return out


def verify_counts(log):
    hits = re.findall(r"mm-warp verify calls=(\d+) mismatches=(\d+)", log)
    return (int(hits[-1][0]), int(hits[-1][1])) if hits else None


def brief(path):
    for r in read_jsonl(path):
        print("  %-16s ttft=%s s prefill=%s tok/s decode=%s tok/s chunks=%s text=%s" % (
            r.get("name"), r.get("ttft_s"), r.get("prefill_tok_s"), r.get("decode_tok_s"),
            r.get("decode_chunks"), r.get("text_sha256", "")[:12]))


def check(arm, d):
    s = load_sample(d)
    p = premise(arm, s)
    extra = ""
    vc = verify_counts(s["log"])
    if vc:
        extra = " mm-warp verify calls=%d mismatches=%d" % vc
    ho = handled_oom(s["log"])
    if ho:
        extra += " handled allocation failures %s" % ho
    print("check %s s%d: %s; GPU0 peak %d MiB GPU1 peak %d MiB%s" % (
        arm, s["k"], "; ".join(p) if p else "ok", gpu_peak_mib(d, 0), gpu_peak_mib(d, 1), extra))
    return 1 if p else 0


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(xs), 4) if xs else None


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.mean(xs), 4) if xs else None


def cv(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.pstdev(xs) / statistics.mean(xs), 4) if len(xs) > 1 and statistics.mean(xs) else None


def summarize(base):
    out = {"kind": "final6", "arms": {}, "comparisons": {}, "group_references": {},
           "gates": {"problems": [], "notes": []}}
    present = [a for a in ARMS if samples(base, a)]
    # the reference text of each (group, request): the group's first non-warm-up sample by run order
    order = sorted(((int(s["meta"].get("seq", 0)), a, s) for a in present if a not in ("W", "WR") for s in samples(base, a)),
                   key=lambda t: t[0])
    ref = {}
    for _, a, s in order:
        g = ARMS[a][2]
        for r in s["rows"]:
            ref.setdefault(g, {}).setdefault(r.get("name"), (r.get("text_sha256"), "%s s%d" % (a, s["k"])))
    out["group_references"] = {g: {n: {"sha256": v[0], "from": v[1]} for n, v in d.items()} for g, d in ref.items()}
    for arm in present:
        tree, model, group = ARMS[arm][:3]
        a = {"tree": tree, "model": model, "group": group, "samples": [], "requests": {}}
        for s in samples(base, arm):
            x = {"k": s["k"], "seq": s["meta"].get("seq"), "wall_s": s["meta"].get("wall_s"),
                 "binary_sha256": s["meta"].get("binary_sha256"), "premise_problems": premise(arm, s),
                 "gpu0_peak_mib": gpu_peak_mib(s["dir"], 0), "gpu1_peak_mib": gpu_peak_mib(s["dir"], 1),
                 "mm_warp_verify": verify_counts(s["log"]), "handled_oom": handled_oom(s["log"]), "requests": {}}
            for r in s["rows"]:
                n = r.get("name")
                rs = ref.get(group, {}).get(n)
                same = (r.get("text_sha256") == rs[0]) if rs else None
                x["requests"][n] = {k: r.get(k) for k in ("ttft_s", "prefill_tok_s", "decode_tok_s", "decode_chunks",
                                                          "prompt_tokens", "text_sha256")}
                x["requests"][n]["identical_to_group_reference"] = same
                if same is False and arm not in ("W", "WR"):
                    out["gates"]["problems"].append("%s s%d %s text differs from %s's" % (arm, s["k"], n, rs[1]))
                if group == "reap-dec" and n in REF_G1_OLD and r.get("text_sha256") != REF_G1_OLD[n]:
                    out["gates"]["notes"].append("%s s%d %s differs from final5's text (older base)" % (arm, s["k"], n))
            if x["premise_problems"] and arm not in ("W", "WR"):
                out["gates"]["problems"].append("%s s%d: %s" % (arm, s["k"], "; ".join(x["premise_problems"])))
            a["samples"].append(x)
        names = sorted({n for x in a["samples"] for n in x["requests"]})
        for n in names:
            vals = [x["requests"].get(n, {}) for x in a["samples"]]
            a["requests"][n] = {"ttft_s_median": med([v.get("ttft_s") for v in vals]),
                                "ttft_cv": cv([v.get("ttft_s") for v in vals]),
                                "prefill_tok_s_median": med([v.get("prefill_tok_s") for v in vals]),
                                "decode_tok_s_median": med([v.get("decode_tok_s") for v in vals]),
                                "decode_tok_s_mean": mean([v.get("decode_tok_s") for v in vals])}
        a["per_sample_mean_decode"] = [mean([r.get("decode_tok_s") for r in x["requests"].values()]) for x in a["samples"]]
        a["median_of_sample_mean_decode"] = med(a["per_sample_mean_decode"])
        if len({x["binary_sha256"] for x in a["samples"]}) > 1:
            out["gates"]["problems"].append("%s ran more than one binary" % arm)
        if arm in TIMED and len(a["samples"]) < 3:
            out["gates"]["notes"].append("%s has %d samples, fewer than 3" % (arm, len(a["samples"])))
        out["arms"][arm] = a
    a_text = next((r.get("text") for s in samples(base, "A") for r in s["rows"] if r.get("name") == "long_prefill"), None)
    for arm in ("MA", "MB"):
        for s in samples(base, arm):
            for r in s["rows"]:
                if a_text is not None and r.get("text") is not None and not a_text.startswith(r["text"]):
                    out["gates"]["notes"].append("%s s%d: the 24-token MTP answer is not a prefix of A's answer" % (arm, s["k"]))
    for arm, a in out["arms"].items():
        tot = sum(sum(x["handled_oom"].values()) for x in a["samples"])
        a["handled_oom_total"] = tot
        if tot:
            out["gates"]["notes"].append("%s: %d handled allocation failures over %d samples (%s)" % (
                arm, tot, len(a["samples"]), "; ".join("s%d %s" % (x["k"], x["handled_oom"]) for x in a["samples"] if x["handled_oom"])))
    for new, old, what in COMPARE:
        if new in out["arms"] and old in out["arms"]:
            rn, ro = out["arms"][new]["requests"], out["arms"][old]["requests"]
            c = {"what": what}
            for n in sorted(set(rn) & set(ro)):
                if rn[n]["ttft_s_median"] and ro[n]["ttft_s_median"]:
                    c[n + "_ttft_ratio"] = round(rn[n]["ttft_s_median"] / ro[n]["ttft_s_median"], 4)
                if rn[n]["decode_tok_s_median"] and ro[n]["decode_tok_s_median"]:
                    c[n + "_decode_ratio"] = round(rn[n]["decode_tok_s_median"] / ro[n]["decode_tok_s_median"], 4)
            if out["arms"][new]["median_of_sample_mean_decode"] and out["arms"][old]["median_of_sample_mean_decode"]:
                c["mean_decode_ratio"] = round(out["arms"][new]["median_of_sample_mean_decode"] /
                                               out["arms"][old]["median_of_sample_mean_decode"], 4)
            out["comparisons"]["%s/%s" % (new, old)] = c
    core_ok = all(a in out["arms"] and len(out["arms"][a]["samples"]) >= 3 for a in ("A", "B"))
    core_problems = [p for p in out["gates"]["problems"] if p.split()[0] in ("A", "B", "V")]
    out["gates"]["question_answered"] = core_ok and not core_problems
    out["gates"]["pass"] = not out["gates"]["problems"]
    return out


def print_summary(out):
    print("FINAL6 SUMMARY (full DeepSeek V4 Flash; REAP-150B for PR #2's arms)")
    for arm, a in out["arms"].items():
        print("arm %s (%s, %s, group %s)%s" % (arm, a["tree"], a["model"], a["group"], "  (warm-up, discarded)" if arm in ("W", "WR") else ""))
        for x in a["samples"]:
            print("  s%d seq=%s wall=%ss GPU peak %d/%d MiB %s%s%s" % (
                x["k"], x["seq"], x["wall_s"], x["gpu0_peak_mib"], x["gpu1_peak_mib"],
                "; ".join(x["premise_problems"]) or "ok",
                "" if not x["handled_oom"] else " handled allocation failures %s" % x["handled_oom"],
                "" if not x["mm_warp_verify"] else " mm-warp verify (calls, mismatches)=%s" % (x["mm_warp_verify"],)))
            for n, r in x["requests"].items():
                print("    %-16s ttft=%s decode=%s identical=%s" % (n, r["ttft_s"], r["decode_tok_s"], r["identical_to_group_reference"]))
        for n, r in a["requests"].items():
            print("  median %-16s ttft=%s s (cv %s) prefill=%s tok/s decode=%s tok/s (mean %s)" % (
                n, r["ttft_s_median"], r["ttft_cv"], r["prefill_tok_s_median"], r["decode_tok_s_median"], r["decode_tok_s_mean"]))
        print("  median of per-sample mean decode: %s tok/s" % a["median_of_sample_mean_decode"])
    for k, c in out["comparisons"].items():
        print("%-6s %s" % (k, c))
    print("QUESTION (A vs B, V) ANSWERED CLEANLY: %s" % out["gates"]["question_answered"])
    print("GATES: %s" % ("PASS" if out["gates"]["pass"] else "FAIL"))
    for p in out["gates"]["problems"]:
        print("  problem: " + p)
    for p in out["gates"]["notes"]:
        print("  note: " + p)
    print("JSON " + json.dumps(out, sort_keys=True))


def _fake(base, arm, k, seq, rows, env, log, rc="0"):
    d = os.path.join(base, arm, "s%d" % k)
    os.makedirs(d)
    with open(os.path.join(d, "sample.txt"), "w") as f:
        f.write("seq=%d\ntree_dirty_lines=0\nbinary_sha256=%s\nready=1\nclient_rc=%s\nserver_env=%s\nwall_s=100\n"
                % (seq, ARMS[arm][0], rc, env))
    with open(os.path.join(d, "bench.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(d, "serve.log"), "w") as f:
        f.write(log)


def selftest():
    env = "COLI_CUDA=1 CUDA_DENSE=1 COLI_CUDA_ATTN_BATCH=1"
    rows = lambda t, txt="x": [{"name": "long_prefill", "ttft_s": t, "decode_tok_s": 1.0, "text": txt},
                               {"name": "long_prefill_2", "ttft_s": t * 1.5, "decode_tok_s": 1.2, "text": "y"}]
    with tempfile.TemporaryDirectory() as base:
        seq = 0
        for k in (1, 2, 3):
            seq += 1; _fake(base, "A", k, seq, rows(300.0), env, "v4_gpu moe-batch=off\n")
            seq += 1; _fake(base, "B", k, seq, rows(290.0), env, "")
        seq += 1; _fake(base, "V", 1, seq, rows(310.0), env + " DSV4_MM_WARP_VERIFY=1", "mm-warp verify calls=500 mismatches=0 (x)\n")
        seq += 1; _fake(base, "BK", 1, seq, rows(100.0), env + " COLI_CUDA_MOE_BATCH=1",
                        "v4_gpu mirror-reserve=4681MiB\nv4_gpu moe-batch=on bank=256-experts\n")
        seq += 1; _fake(base, "BS", 1, seq, rows(100.0)[:1], env + " COLI_CUDA_MOE_BATCH=1 V4_DRAFT=4",
                        "v4_gpu moe-batch=on bank=256-experts\n")
        out = summarize(base)
        assert out["gates"]["pass"], out["gates"]
        assert out["gates"]["question_answered"]
        assert out["comparisons"]["B/A"]["long_prefill_ttft_ratio"] == round(290 / 300, 4)
    with tempfile.TemporaryDirectory() as base:
        seq = 0
        for k in (1, 2, 3):
            seq += 1; _fake(base, "A", k, seq, rows(300.0), env, "")
            seq += 1; _fake(base, "B", k, seq, rows(290.0, "z" if k == 2 else "x"), env + " COLI_CUDA_MOE_BATCH=1",
                            "v4_gpu moe-batch=on bank=256-experts\n")
        seq += 1; _fake(base, "V", 1, seq, rows(310.0), env + " DSV4_MM_WARP_VERIFY=1", "mm-warp verify calls=3 mismatches=2 (x)\n")
        seq += 1; _fake(base, "BS", 1, seq, rows(100.0)[:1], env + " COLI_CUDA_MOE_BATCH=1 V4_DRAFT=4",
                        "v4_gpu moe-batch=on bank=256-experts\nv4_gpu moe-batch=on bank=256-experts\n")
        out = summarize(base)
        probs = " ".join(out["gates"]["problems"])
        for want in ("differs", "COLI_CUDA_MOE_BATCH set", "moe-batch=on", "mismatches=[1-9]", "F2"):
            assert want in probs, (want, probs)
        assert not out["gates"]["question_answered"]
    with tempfile.TemporaryDirectory() as base:
        rows = [{"name": "rep%d_turn%d" % (r, t), "ttft_s": 1.0, "decode_tok_s": 3.0, "text": "t%d" % t + ("!" if (r, t) == (3, 4) else "")}
                for r in (1, 2, 3) for t in range(1, 7)]
        _fake(base, "SK", 1, 1, rows, "COLI_CUDA_MOE_BATCH=1 V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1", "")
        p = premise("SK", load_sample(os.path.join(base, "SK", "s1")))
        assert len(p) == 1 and "rep3_turn4" in p[0], p
    with tempfile.TemporaryDirectory() as base:
        rows2 = [{"name": "long_prefill", "ttft_s": 1.0, "decode_tok_s": 1.0, "text": "x"},
                 {"name": "long_prefill_2", "ttft_s": 1.0, "decode_tok_s": 1.0, "text": "y"}]
        _fake(base, "A", 1, 1, rows2, "COLI_CUDA=1", "[DSV4 CUDA] kv comp allocation: out of memory\n" * 3)
        s1 = load_sample(os.path.join(base, "A", "s1"))
        assert premise("A", s1) == [] and handled_oom(s1["log"]) == {"kv comp": 3}, (premise("A", s1), handled_oom(s1["log"]))
        _fake(base, "A", 2, 2, rows2, "COLI_CUDA=1", "CUDA error: out of memory at foo\n")
        assert premise("A", load_sample(os.path.join(base, "A", "s2")))
    print("selftest ok")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "brief":
        brief(sys.argv[2])
    elif cmd == "check":
        sys.exit(check(sys.argv[2], sys.argv[3]))
    elif cmd == "summarize":
        print_summary(summarize(sys.argv[3]))
    elif cmd == "selftest":
        sys.exit(selftest())
    else:
        print(__doc__)
        sys.exit(2)
