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
VK_ONLY = {"COLI_VULKAN": "1", "DSV4_CUDA": "0", "COLI_CUDA": "0"}
VK_PIN = {"COLI_VK_TIER_HOST_GB": "12", "COLI_VK_TIER_STREAM_ROWS": "16", "COLI_VK_TIER_BALANCE": "0"}
VK_PINS = list(VK_PIN)
VK_PIN_LOG = [r"cold experts with 16 rows or more go to the device \(COLI_VK_TIER_STREAM_ROWS", r"host pool on, \d+ experts in 12\.00 GiB"]
F18_AT = os.environ.get("F18_AT", "3000")
PR2_OFF = ["V4_GPU_RESIDENT_FIRST", "DSV4_CUDA_PEER_DEVICE", "DSV4_CUDA_PEER_COMPUTE", "DSV4_HYBRID", "DSV4_CUDA_BATCHED"]
NOT_DEFAULT = ["COLI_CUDA_MOE_BATCH", "COLI_CUDA_MOE_DOUBLE", "DSV4_MM_WARP", "DSV4_MM_WARP_VERIFY", "V4_DRAFT", "V4_MTP"]

# arm: (tree, model, group, must_env, must_not_env, must_log, must_not_log, rows, core)
#   group: the arms whose texts must all equal the group's first sample (per request name)
#   core: a premise failure here counts toward the abort; elsewhere it only skips the arm
ARMS = {
    # 1. the 256-expert bank, like for like (final6's AK lost dev's bank on its second prompt: dev's default
    #    reserve, 2800 MiB, is below the bank's 3264 MiB once the decode mirrors grow, as dev's own comment says).
    #    WK warm-up; AR = dev with PR #1's reserve (DSV4_CUDA_VRAM_RESERVE_MB=4681); BK PR #1; AK dev as it is,
    #    expected to lose the bank on prompt 2 again (a note, not a premise: that is the result).
    "WK": ("pr1", "full", "full", {"COLI_CUDA_MOE_BATCH": "1"}, ["COLI_CUDA_MOE_DOUBLE", "V4_DRAFT"], [BANK], [r"moe-batch=off"], 2, False),
    "AR": ("dev", "full", "full", {"COLI_CUDA_MOE_BATCH": "1", "DSV4_CUDA_VRAM_RESERVE_MB": "4681"}, ["COLI_CUDA_MOE_DOUBLE", "V4_DRAFT"],
           [BANK], [r"moe-batch=off"], 2, True),
    "BK": ("pr1", "full", "full", {"COLI_CUDA_MOE_BATCH": "1"}, ["COLI_CUDA_MOE_DOUBLE", "V4_DRAFT", "DSV4_CUDA_VRAM_RESERVE_MB"],
           [BANK, r"mirror-reserve="], [r"moe-batch=off"], 2, True),
    "AK": ("dev", "full", "full", {"COLI_CUDA_MOE_BATCH": "1"}, ["COLI_CUDA_MOE_DOUBLE", "V4_DRAFT", "DSV4_CUDA_VRAM_RESERVE_MB"], [BANK], [], 2, False),
    # 2. pure Vulkan (CUDA off), REAP: QA dev three times as it is (is it reproducible at all?), then the
    #    tier's run-time choices pinned (pool size, stream threshold, no CPU balance): QP dev, QQ PR #1
    "QA": ("dev-vk", "reap", "vk-free", VK_ONLY, ["COLI_CUDA_MOE_BATCH", "V4_DRAFT"] + VK_PINS,
           [], [r"moe-batch=on", r"DSV4 CUDA\] device"], 2, False),
    "QP": ("dev-vk", "reap", "vk-pin", dict(VK_ONLY, **VK_PIN), ["COLI_CUDA_MOE_BATCH", "V4_DRAFT"],
           VK_PIN_LOG, [r"moe-batch=on", r"DSV4 CUDA\] device"], 2, False),
    "QQ": ("pr1-vk", "reap", "vk-pin", dict(VK_ONLY, **VK_PIN), ["COLI_CUDA_MOE_BATCH", "V4_DRAFT"],
           VK_PIN_LOG, [r"moe-batch=on", r"DSV4 CUDA\] device"], 2, False),
    #    dev with final6 QB's pool (15.75 GiB): expected to be OOM-killed like QB (a result, recorded)
    "QO": ("dev-vk", "reap", "vk-qo", dict(VK_ONLY, COLI_VK_TIER_HOST_GB="15.75"), ["COLI_CUDA_MOE_BATCH", "V4_DRAFT"], [], [], 2, False),
    # 3. Vulkan + CUDA build, REAP, COLI_VULKAN=1: VA dev (its bank is refused on REAP: CPU union), VU PR #1
    #    without the bank (the union: must equal VA), VB PR #1 with the bank (as final6); repeats of each
    "VA": ("dev-vk", "reap", "vkc-union", {"COLI_CUDA_MOE_BATCH": "1", "COLI_VULKAN": "1"}, ["V4_DRAFT"], [], [], 2, False),
    "VU": ("pr1-vk", "reap", "vkc-union", {"COLI_VULKAN": "1"}, ["COLI_CUDA_MOE_BATCH", "V4_DRAFT"], [], [r"moe-batch=on"], 2, False),
    "VB": ("pr1-vk", "reap", "vkc-bank", {"COLI_CUDA_MOE_BATCH": "1", "COLI_VULKAN": "1"}, ["V4_DRAFT"], [r"moe-batch=on bank=132-experts"], [], 2, False),
    # 4. DSV4_HYBRID (dev's switch; its split comes from live bandwidth EMAs): D0 dev reference, HD dev + hybrid
    #    twice, HG PR #2 switches off + hybrid, HY PR #2 all on + hybrid twice (as final6)
    "D0": ("dev", "reap", "reap-dec", {"COLI_CUDA_MOE_BATCH": "1"}, PR2_OFF, [], [], 6, False),
    "HD": ("dev", "reap", "hyb-dev", {"COLI_CUDA_MOE_BATCH": "1", "DSV4_HYBRID": "1"}, ["V4_GPU_RESIDENT_FIRST", "DSV4_CUDA_PEER_DEVICE"], [], [], 6, False),
    "HG": ("pr2", "reap", "hyb-pr2-off", {"COLI_CUDA_MOE_BATCH": "1", "DSV4_HYBRID": "1"}, ["V4_GPU_RESIDENT_FIRST", "DSV4_CUDA_PEER_DEVICE"], [], [], 6, False),
    "HY": ("pr2", "reap", "hyb-pr2-on", {"V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1", "DSV4_HYBRID": "1"}, [], [], [], 6, False),
    # 5. F18 in a real server: the probe build (PR #2 + -DCOLI_F18_PROBE), PR #2 all on, a __trap() on the peer
    #    mid-way through the first answer; the client goes on with the other five
    "F7": ("f18", "reap", "f18", {"V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1", "DSV4_F18_TRAP": F18_AT},
           [], [r"F18 probe: __trap\(\) on the peer"], [], 6, False),
}
# final6's texts per (group, request), the same requests on the same models: a difference is a note
REF_OLD = {
    "full": {"long_prefill": "a397814f650a80340c40cca3197295a49dd5d2dc9d55211f27473036bb4efd64",
             "long_prefill_2": "8b3f437baa347d887b5dada5bd950743d49b3181fdab9f27a87f64ad0433d4c0"},
    "reap-dec": {"decode_0": "2aa40adc7d4deb49fa84db4e070a48f4d14d10981ed8103efe83388292c84e9b",
                 "decode_1": "643fa3d128e4c41c671ffc829f6d554531d259c2eb075820085d8981e985af80",
                 "decode_2": "d64c91874f9ee22c50b943803c51eeeca79f4b613b4e551456b6735d57a318da",
                 "decode_3": "9fffce519994640443407e699f48b4c1762e8a23af2b6719531f317892cc5231",
                 "decode_4": "e3bb3209f760833a7e440b7371e35ceb90dbac2c459f65ab75131c179db1ebda",
                 "decode_5": "31cbd1c1700fd7f64dda1d79142ea7d5e2d1b442ccc6e879c44bee7a3bfb439e"},
}
for g in ("hyb-dev", "hyb-pr2-off", "hyb-pr2-on", "f18"):
    REF_OLD[g] = REF_OLD["reap-dec"]
REF_OLD["vk-free"] = REF_OLD["vk-pin"] = {"long_prefill": "6ec1e4b7afc8aad9ff13e7daa5a60ca8771c0d5a0d3a387844aa9164f02e2d0e",
                                         "long_prefill_2": "f85783e3112bfb87e50b6bd769b3864244d3b72838219fdbf05aa94a7aab1261"}  # final6 QA
TIMED = ["AR", "BK", "QP", "QQ"]
COMPARE = [("BK", "AR", "PR #1 vs dev, 256-expert bank, dev with PR #1's reserve"), ("BK", "AK", "PR #1 vs dev as it is, 256-expert bank"),
           ("AK", "AR", "dev: default reserve vs 4681 MiB"),
           ("QQ", "QP", "PR #1 vs dev, pure Vulkan, tier pinned"), ("VU", "VA", "PR #1 vs dev, Vulkan + CUDA, CPU union"),
           ("VB", "VA", "PR #1 bank vs dev union, Vulkan + CUDA"), ("HD", "D0", "dev: hybrid vs not"), ("HG", "HD", "PR #2 off vs dev, hybrid")]

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


# final7: case-insensitive ("[DSV4 CUDA] expert FC2 bank allocation: out of memory", dev's bank on its second prompt)
HANDLED_OOM = re.compile(r"^\[DSV4 CUDA\] [a-z0-9 _-]+ allocation: out of memory$", re.M | re.I)


def handled_oom(log):
    """{what: count} of handled allocation failures ("[DSV4 CUDA] kv comp allocation: out of memory")."""
    out = {}
    for m in HANDLED_OOM.finditer(log):
        what = m.group(0)[len("[DSV4 CUDA] "):].split(" allocation")[0]
        out[what] = out.get(what, 0) + 1
    return out


def mem_record(d):
    """final7: (min MemAvailable GiB, peak engine VmRSS GiB) from mem.csv, or (None, None)."""
    av, rss = [], []
    for line in text(os.path.join(d, "mem.csv")).splitlines()[1:]:
        f = line.split(",")
        if len(f) == 3 and f[1].isdigit() and f[2].isdigit():
            av.append(int(f[1])); rss.append(int(f[2]))
    g = 1024.0 * 1024.0
    return (round(min(av) / g, 2) if av else None, round(max(rss) / g, 2) if rss else None)


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
    out = {"kind": "final7", "arms": {}, "comparisons": {}, "group_references": {},
           "gates": {"problems": [], "notes": []}}
    present = [a for a in ARMS if samples(base, a)]
    # the reference text of each (group, request): the group's first non-warm-up sample by run order
    order = sorted(((int(s["meta"].get("seq", 0)), a, s) for a in present if a not in ("W", "WR", "WK") for s in samples(base, a)),
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
                 "mm_warp_verify": verify_counts(s["log"]), "handled_oom": handled_oom(s["log"]),
                 "oom_kill": s["meta"].get("oom_kill"), "mem_available_min_gib": mem_record(s["dir"])[0],
                 "engine_rss_peak_gib": mem_record(s["dir"])[1],
                 "bank_lost": bool(re.search(r"moe-batch=off \(bank allocation failed", s["log"])), "requests": {}}
            for r in s["rows"]:
                n = r.get("name")
                rs = ref.get(group, {}).get(n)
                same = (r.get("text_sha256") == rs[0]) if rs else None
                x["requests"][n] = {k: r.get(k) for k in ("ttft_s", "prefill_tok_s", "decode_tok_s", "decode_chunks",
                                                          "prompt_tokens", "text_sha256")}
                x["requests"][n]["identical_to_group_reference"] = same
                if same is False and arm not in ("W", "WR", "WK"):
                    out["gates"]["problems"].append("%s s%d %s text differs from %s's" % (arm, s["k"], n, rs[1]))
                old = REF_OLD.get(group, {}).get(n)
                x["requests"][n]["identical_to_final6"] = (r.get("text_sha256") == old) if old else None
                if old and r.get("text_sha256") != old:
                    out["gates"]["notes"].append("%s s%d %s differs from final6's text for its group" % (arm, s["k"], n))
            if x["premise_problems"] and arm not in ("W", "WR", "WK"):
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
    core_ok = all(a in out["arms"] and len(out["arms"][a]["samples"]) >= 3 for a in ("AR", "BK"))
    core_problems = [p for p in out["gates"]["problems"] if p.split()[0] in ("AR", "BK")]
    out["gates"]["question_answered"] = core_ok and not core_problems
    out["gates"]["pass"] = not out["gates"]["problems"]
    return out


def print_summary(out):
    print("FINAL7 SUMMARY (follow-ups to final6; full DeepSeek V4 Flash for the bank, REAP-150B for the rest)")
    for arm, a in out["arms"].items():
        print("arm %s (%s, %s, group %s)%s" % (arm, a["tree"], a["model"], a["group"], "  (warm-up, discarded)" if arm in ("W", "WR", "WK") else ""))
        for x in a["samples"]:
            print("  s%d seq=%s wall=%ss GPU peak %d/%d MiB RSS peak %s GiB MemAvailable min %s GiB oom_kill=%s%s %s%s%s" % (
                x["k"], x["seq"], x["wall_s"], x["gpu0_peak_mib"], x["gpu1_peak_mib"],
                x["engine_rss_peak_gib"], x["mem_available_min_gib"], x["oom_kill"], " BANK LOST" if x["bank_lost"] else "",
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
    print("BANK QUESTION (BK vs AR) ANSWERED CLEANLY: %s" % out["gates"]["question_answered"])
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
    env = "COLI_CUDA=1 CUDA_DENSE=1 COLI_CUDA_ATTN_BATCH=1 COLI_CUDA_MOE_BATCH=1"
    rows = lambda t, txt="x": [{"name": "long_prefill", "ttft_s": t, "decode_tok_s": 1.0, "text": txt},
                               {"name": "long_prefill_2", "ttft_s": t * 1.5, "decode_tok_s": 1.2, "text": "y"}]
    bank = "v4_gpu mirror-reserve=4681MiB\nv4_gpu moe-batch=on bank=256-experts\n"
    with tempfile.TemporaryDirectory() as base:
        seq = 0
        for k in (1, 2, 3):
            seq += 1; _fake(base, "AR", k, seq, rows(130.0), env + " DSV4_CUDA_VRAM_RESERVE_MB=4681", "v4_gpu moe-batch=on bank=256-experts\n")
            seq += 1; _fake(base, "BK", k, seq, rows(86.0), env, bank)
        seq += 1; _fake(base, "AK", 1, seq, rows(130.0), env, "v4_gpu moe-batch=on bank=256-experts\n"
                        "[DSV4 CUDA] expert FC2 bank allocation: out of memory\nv4_gpu moe-batch=off (bank allocation failed; CPU union stays)\n")
        out = summarize(base)
        assert out["gates"]["pass"], out["gates"]
        assert out["gates"]["question_answered"]
        assert out["comparisons"]["BK/AR"]["long_prefill_ttft_ratio"] == round(86 / 130, 4)
        ak = out["arms"]["AK"]["samples"][0]
        assert ak["bank_lost"] and ak["handled_oom"] == {"expert FC2 bank": 1}, ak
    with tempfile.TemporaryDirectory() as base:
        seq = 0
        for k in (1, 2, 3):
            seq += 1; _fake(base, "AR", k, seq, rows(130.0), env, "v4_gpu moe-batch=on bank=256-experts\n")
            seq += 1; _fake(base, "BK", k, seq, rows(86.0, "z" if k == 2 else "x"), env, bank)
        out = summarize(base)
        probs = " ".join(out["gates"]["problems"])
        for want in ("differs", "DSV4_CUDA_VRAM_RESERVE_MB=4681 not in"):
            assert want in probs, (want, probs)
        assert not out["gates"]["question_answered"]
    with tempfile.TemporaryDirectory() as base:
        venv = "COLI_VULKAN=1 DSV4_CUDA=0 COLI_CUDA=0 COLI_VK_TIER_HOST_GB=12 COLI_VK_TIER_STREAM_ROWS=16 COLI_VK_TIER_BALANCE=0"
        log = ("[VK] tier deepseek_v4: host pool on, 819 experts in 12.00 GiB of system RAM\n"
               "[VK] tier deepseek_v4 stream: a step of 768 rows; cold experts with 16 rows or more go to the device (COLI_VK_TIER_STREAM_ROWS)\n")
        _fake(base, "QP", 1, 1, rows(100.0), venv, log)
        _fake(base, "QQ", 1, 2, rows(100.0), venv, log)
        _fake(base, "QQ", 2, 3, rows(100.0), venv, "")
        out = summarize(base)
        probs = " ".join(out["gates"]["problems"])
        assert "QQ s2" in probs and "QQ s1" not in probs and "QP" not in probs, probs
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
