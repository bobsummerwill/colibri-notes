#!/usr/bin/env python3
"""final27_tools.py (2026-10-26, final25's for the restart on dev c1c06ac6; trees f27-*). final25_tools.py (2026-10-26, from final24_tools.py; final24_tools.py 2026-10-26 from final23_tools.py; final23_tools.py 2026-10-26 from final22_tools.py; final22_tools.py 2026-10-25 night from final21_tools.py; final21_tools.py 2026-10-25 from final13_tools.py; final13_tools.py 2026-10-24 from final12_tools.py; final12_tools.py from final11_tools.py; final11_tools.py from final10_tools.py; final10_tools.py from final9_tools.py; final9_tools.py from final8_tools.py; final6_tools.py 2026-10-22 before it) -- checks and summary for run_final13.sh.

final9: the merged tips on Vulkan (ARMS below). The rest of this docstring is final6's.

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
VK_FIX = {"COLI_VULKAN": "1", "DSV4_CUDA": "0", "COLI_CUDA": "0", "COLI_VK_TIER_HOST_GB": "12"}
VK_PIN_LOG = [r"cold experts with 16 rows or more go to the device \(COLI_VK_TIER_STREAM_ROWS", r"host pool on, \d+ experts in 12\.00 GiB"]
F18_AT = os.environ.get("F18_AT", "3000")
PR2_OFF = ["V4_GPU_RESIDENT_FIRST", "DSV4_CUDA_PEER_DEVICE", "DSV4_CUDA_PEER_COMPUTE", "DSV4_HYBRID", "DSV4_CUDA_BATCHED"]
NOT_DEFAULT = ["COLI_CUDA_MOE_BATCH", "COLI_CUDA_MOE_DOUBLE", "DSV4_MM_WARP", "DSV4_MM_WARP_VERIFY", "V4_DRAFT", "V4_MTP"]

# arm: (tree, model, group, must_env, must_not_env, must_log, must_not_log, rows, core)
#   group: the arms whose texts must all equal the group's first sample (per request name)
#   core: a premise failure here counts toward the abort; elsewhere it only skips the arm
PR2_PINS = ["DSV4_CUDA_PEER_COMPUTE", "DSV4_HYBRID", "DSV4_CUDA_BATCHED"]
ARMS = {
    # final25 (2026-10-26): the same-base run (review/PLAN_SAME_BASE_20261026.md, Stage 1). Every tree on dd03a9de (dev
    # 77530b49 + #2401): B = dd03a9de, M = #1766 merged onto it (samebase/m1766), P = PR #2 5b3f4601. One base, so ONE
    # text group per model: every CUDA arm must equal the group's first sample byte for byte (REAP: also final11's D0
    # texts, a note if not). #1766's serve line (COLI_CUDA_ATTN_BATCH=1, set for every arm, COLI_CUDA_MOE_BATCH=1)
    # everywhere; no DSV4_CUDA_VRAM_RESERVE_MB (#2401's bank-aware default; #1766's preload reserve at its default).
    #   x1 = one GPU: B1 base, M1 #1766, P1 PR #2 VRAM-first (no peer); x2: M2 #1766 DSV4_CUDA_DEVICES=0,1, P2 PR #2 peer GPU 1.
    "B1": ("base", "reap", "reap-dec", {"COLI_CUDA_MOE_BATCH": "1"}, ["DSV4_CUDA_DEVICES", "DSV4_CUDA_RESIDENT_EXPERTS", "DSV4_CUDA_VRAM_RESERVE_MB"] + PR2_OFF, [], [], 6, False),
    "M1": ("m25", "reap", "reap-dec", {"COLI_CUDA_MOE_BATCH": "1"}, ["DSV4_CUDA_DEVICES", "DSV4_CUDA_RESIDENT_EXPERTS", "DSV4_CUDA_VRAM_RESERVE_MB"] + PR2_OFF,
           [r"devices=1 placement=contiguous-layers"], [], 6, False),
    "P1": ("pr2", "reap", "reap-dec", {"COLI_CUDA_MOE_BATCH": "1", "V4_GPU_RESIDENT_FIRST": "1"},
           ["DSV4_CUDA_DEVICES", "DSV4_CUDA_VRAM_RESERVE_MB", "DSV4_CUDA_PEER_DEVICE"] + PR2_PINS + ["DSV4_CUDA_PEER_FAULT"], [], [r"peer-mirrors device=1", r"CUDA is lost"], 6, False),
    "M2": ("m25", "reap", "reap-dec", {"COLI_CUDA_MOE_BATCH": "1", "DSV4_CUDA_DEVICES": "0,1"}, ["DSV4_CUDA_RESIDENT_EXPERTS", "DSV4_CUDA_VRAM_RESERVE_MB"] + PR2_OFF,
           [r"v4_gpu device=1 layers=\d+\.\.42"], [], 6, False),
    "P2": ("pr2", "reap", "reap-dec", {"COLI_CUDA_MOE_BATCH": "1", "V4_SERVE_STATS_LOG": "1", "V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1"},
           ["DSV4_CUDA_DEVICES", "DSV4_CUDA_VRAM_RESERVE_MB"] + PR2_PINS + ["DSV4_CUDA_PEER_FAULT"], [r"peer-mirrors device=1"], [r"CUDA is lost"], 6, False),
    # the same five on the full V4 Flash, bench_pr1.py (1,992- and 3,005-token prompts), 2 each
    "B1F": ("base", "full", "full", {"COLI_CUDA_MOE_BATCH": "1"}, ["DSV4_CUDA_DEVICES", "DSV4_CUDA_RESIDENT_EXPERTS", "DSV4_CUDA_VRAM_RESERVE_MB"] + PR2_OFF, [], [], 2, False),
    "M1F": ("m25", "full", "full", {"COLI_CUDA_MOE_BATCH": "1"}, ["DSV4_CUDA_DEVICES", "DSV4_CUDA_RESIDENT_EXPERTS", "DSV4_CUDA_VRAM_RESERVE_MB"] + PR2_OFF,
            [r"devices=1 placement=contiguous-layers"], [], 2, False),
    "P1F": ("pr2", "full", "full", {"COLI_CUDA_MOE_BATCH": "1", "V4_GPU_RESIDENT_FIRST": "1"},
            ["DSV4_CUDA_DEVICES", "DSV4_CUDA_VRAM_RESERVE_MB", "DSV4_CUDA_PEER_DEVICE"] + PR2_PINS + ["DSV4_CUDA_PEER_FAULT"], [], [r"peer-mirrors device=1", r"CUDA is lost"], 2, False),
    "M2F": ("m25", "full", "full", {"COLI_CUDA_MOE_BATCH": "1", "DSV4_CUDA_DEVICES": "0,1"}, ["DSV4_CUDA_RESIDENT_EXPERTS", "DSV4_CUDA_VRAM_RESERVE_MB"] + PR2_OFF,
            [r"v4_gpu device=1 layers=\d+\.\.42"], [], 2, False),
    "P2F": ("pr2", "full", "full", {"COLI_CUDA_MOE_BATCH": "1", "V4_SERVE_STATS_LOG": "1", "V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1"},
            ["DSV4_CUDA_DEVICES", "DSV4_CUDA_VRAM_RESERVE_MB"] + PR2_PINS + ["DSV4_CUDA_PEER_FAULT"], [r"peer-mirrors device=1"], [r"CUDA is lost"], 2, False),
    # pure Vulkan on one 3090 (final22's VD environment), REAP, bench_pr1.py, 1 each. final22 saw pure-Vulkan texts change
    # between two runs of one tree (VD s1/s2), so each is its own group: texts are compared by hand, not gated.
    "VKB": ("basevk", "reap", "VKB", VK_ONLY, ["COLI_CUDA_MOE_BATCH", "V4_DRAFT", "COLI_VK_UP_IMPORT", "GLIBC_TUNABLES"] + VK_PINS, [], [], 2, False),
    "VKM": ("m25vk", "reap", "VKM", VK_ONLY, ["COLI_CUDA_MOE_BATCH", "V4_DRAFT", "COLI_VK_UP_IMPORT", "GLIBC_TUNABLES"] + VK_PINS, [], [], 2, False),
    "VKP": ("pr2vk", "reap", "VKP", VK_ONLY, ["COLI_CUDA_MOE_BATCH", "V4_DRAFT", "COLI_VK_UP_IMPORT", "GLIBC_TUNABLES"] + VK_PINS, [], [], 2, False),
    # kept for the selftest (final22's arms)
    "D0": ("base", "reap", "reap-dec", {"COLI_CUDA_MOE_BATCH": "1"}, PR2_OFF, [], [], 6, True),
    "X3": ("pr2", "reap", "reap-dec", {"COLI_CUDA_MOE_BATCH": "1", "V4_SERVE_STATS_LOG": "1", "V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1"},
           PR2_PINS + ["DSV4_CUDA_PEER_FAULT"], [r"peer-mirrors device=1"], [r"CUDA is lost"], 6, True),
    "FD": ("base", "full", "full-d", {}, NOT_DEFAULT + PR2_OFF + ["DSV4_CUDA_VRAM_RESERVE_MB", "COLI_VULKAN"], [], [r"moe-batch=on"], 2, True),
    "FP": ("pr2", "full", "full-d", {"V4_SERVE_STATS_LOG": "1", "V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1"},
           NOT_DEFAULT + PR2_PINS + ["DSV4_CUDA_PEER_FAULT", "DSV4_CUDA_VRAM_RESERVE_MB", "COLI_VULKAN"], [r"peer-mirrors device=1"],
           [r"moe-batch=on", r"CUDA is lost"], 2, True),
    "VD": ("devvk", "reap", "VD", VK_ONLY, ["COLI_CUDA_MOE_BATCH", "V4_DRAFT", "COLI_VK_UP_IMPORT", "GLIBC_TUNABLES"] + VK_PINS,
           [r"host pool on, \d+ experts in"], [], 2, False),
    "VB": ("fixbvk", "reap", "VB", VK_ONLY, ["COLI_CUDA_MOE_BATCH", "V4_DRAFT", "COLI_VK_UP_IMPORT", "GLIBC_TUNABLES"] + VK_PINS,
           [r"host pool on, \d+ experts in"], [], 2, False),
    # kept for the selftest (final21's arms)
    "S7": ("pr2", "reap", "s7", {"COLI_CUDA_MOE_BATCH": "1", "V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1",
                                 "DSV4_CUDA_PEER_FAULT": "3000:sticky"}, [], [r"CUDA is lost for this process", r"DSV4_CUDA_PEER_FAULT: peer-compute pass #3000"],
           [r"block computation failed"], 6, False),
    "A": ("dev", "full", "full", {}, NOT_DEFAULT + ["DSV4_CUDA_VRAM_RESERVE_MB", "COLI_VULKAN"], [], [r"moe-batch=on"], 2, True),
    "F": ("fixa", "full", "full", {}, NOT_DEFAULT + ["DSV4_CUDA_VRAM_RESERVE_MB", "COLI_VULKAN"], [], [r"moe-batch=on"], 2, True),
}
WARMUP = ("W", "WR", "WK", "WQ")
REF_OLD = {"reap-dec": {"decode_0": "2aa40adc7d4deb49fa84db4e070a48f4d14d10981ed8103efe83388292c84e9b",
                        "decode_1": "643fa3d128e4c41c671ffc829f6d554531d259c2eb075820085d8981e985af80",
                        "decode_2": "d64c91874f9ee22c50b943803c51eeeca79f4b613b4e551456b6735d57a318da",
                        "decode_3": "9fffce519994640443407e699f48b4c1762e8a23af2b6719531f317892cc5231",
                        "decode_4": "e3bb3209f760833a7e440b7371e35ceb90dbac2c459f65ab75131c179db1ebda",
                        "decode_5": "31cbd1c1700fd7f64dda1d79142ea7d5e2d1b442ccc6e879c44bee7a3bfb439e"},
           "full": {"long_prefill": "a397814f650a80340c40cca3197295a49dd5d2dc9d55211f27473036bb4efd64",
                    "long_prefill_2": "8b3f437baa347d887b5dada5bd950743d49b3181fdab9f27a87f64ad0433d4c0"}}
KNOWN = {"32534c06abd640d59d2eb39ab6609051f63e9876f533aad1d75dd5cd775835ba": "REAP CUDA/pure-VK p1",
         "1f0192038c425512912a1df7d56dca6670c50f01935e664f0219589b802cdc8d": "REAP CUDA/pure-VK p2"}
KNOWN.update({v: "D0 ref " + k for k, v in REF_OLD["reap-dec"].items()})
KNOWN.update({v: "full ref " + k + " (final6)" for k, v in REF_OLD["full"].items()})
TIMED = []
COMPARE = [("M1", "B1", "#1766 vs base, one RTX 3090, REAP"),
           ("P1", "B1", "PR #2 VRAM-first (no peer) vs base, one GPU, REAP"),
           ("M2", "B1", "#1766 two GPUs (layer split) vs base on one, REAP"),
           ("P2", "B1", "PR #2 two GPUs (peer) vs base on one, REAP"),
           ("M2", "M1", "#1766, what its second GPU adds, REAP"),
           ("P2", "P1", "PR #2, what its second GPU adds, REAP"),
           ("P2", "M2", "PR #2 vs #1766, two GPUs, same base, REAP"),
           ("P1", "M1", "PR #2 vs #1766, one GPU, same base, REAP"),
           ("M1F", "B1F", "#1766 vs base, one RTX 3090, full model"),
           ("P1F", "B1F", "PR #2 VRAM-first (no peer) vs base, one GPU, full model"),
           ("M2F", "B1F", "#1766 two GPUs (layer split) vs base on one, full model"),
           ("P2F", "B1F", "PR #2 two GPUs (peer) vs base on one, full model"),
           ("M2F", "M1F", "#1766, what its second GPU adds, full model"),
           ("P2F", "P1F", "PR #2, what its second GPU adds, full model"),
           ("P2F", "M2F", "PR #2 vs #1766, two GPUs, same base, full model"),
           ("P1F", "M1F", "PR #2 vs #1766, one GPU, same base, full model"),
           ("VKM", "VKB", "#1766 vs base, pure Vulkan, REAP"),
           ("VKP", "VKB", "PR #2 vs base, pure Vulkan, REAP"),
           ("X3", "D0", "selftest pair"), ("FP", "FD", "selftest pair")]


def upload_rates(log):
    """final12: the tier's measured upload rates (GB/s) from its stream lines."""
    return [float(x) for x in re.findall(r"an upload of [0-9.]+ MiB at ([0-9.]+) GB/s", log)]


def vkdiag(log):
    """final11: the [VKDIAG] lines of the v2 diagnostic builds, by prompt (split at v4_ckpt store prompt_end=...)."""
    cut = log.find("v4_ckpt store prompt_end=")
    parts = (log, "") if cut < 0 else (log[:cut], log[cut:])
    out = {}
    for name, part in zip(("p1", "after_p1"), parts):
        why = re.findall(r"\[VKDIAG\] declined: why=(\w+) fmt=(\d+) rows=(\d+) cols=(\d+) batch=(\d+)", part)
        agg = {}
        for w, f, r, c, bt in why:
            k = "%s fmt=%s %sx%s" % (w, f, r, c)
            agg[k] = agg.get(k, 0) + 1
        tallies = re.findall(r"\[VKDIAG\] hook calls: (.*)", part)
        out[name] = {"declined": agg, "last_tally": tallies[-1] if tallies else None,
                     "refused": len(re.findall(r"\[VKDIAG\] (hook upload refused|forget)", part))}
    return out


def memx(d):
    """final10: peaks (GiB) of memx.csv's columns, or {}."""
    rows = text(os.path.join(d, "memx.csv")).splitlines()
    if len(rows) < 2:
        return {}
    head = rows[0].split(",")
    cols = {h: [] for h in head}
    for line in rows[1:]:
        f = line.split(",")
        if len(f) == len(head) and all(x.lstrip("-").isdigit() for x in f):
            for h, x in zip(head, f):
                cols[h].append(int(x))
    g = 1024.0 * 1024.0
    out = {h + "_peak_gib": round(max(v) / g, 2) for h, v in cols.items() if v and h != "epoch"}
    if cols.get("mem_free"):
        out["mem_free_min_gib"] = round(min(cols["mem_free"]) / g, 2)
    return out

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
    out = {"kind": "final27", "arms": {}, "comparisons": {}, "group_references": {},
           "gates": {"problems": [], "notes": []}}
    present = [a for a in ARMS if samples(base, a)]
    # the reference text of each (group, request): the group's first non-warm-up sample by run order
    order = sorted(((int(s["meta"].get("seq", 0)), a, s) for a in present if a not in WARMUP for s in samples(base, a)),
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
                 "oom_kill": s["meta"].get("oom_kill"), "memx": memx(s["dir"]), "vkdiag": None, "upload_gbps": upload_rates(s["log"]), "mem_available_min_gib": mem_record(s["dir"])[0],
                 "engine_rss_peak_gib": mem_record(s["dir"])[1],
                 "bank_lost": bool(re.search(r"moe-batch=off \(bank allocation failed", s["log"])), "requests": {}}
            for r in s["rows"]:
                n = r.get("name")
                rs = ref.get(group, {}).get(n)
                same = (r.get("text_sha256") == rs[0]) if rs else None
                x["requests"][n] = {k: r.get(k) for k in ("ttft_s", "prefill_tok_s", "decode_tok_s", "decode_chunks",
                                                          "prompt_tokens", "text_sha256")}
                x["requests"][n]["identical_to_group_reference"] = same
                if same is False and arm not in WARMUP:
                    out["gates"]["problems"].append("%s s%d %s text differs from %s's" % (arm, s["k"], n, rs[1]))
                old = REF_OLD.get(group, {}).get(n)
                x["requests"][n]["identical_to_final8"] = (r.get("text_sha256") == old) if old else None
                if old and r.get("text_sha256") != old:
                    out["gates"]["notes"].append("%s s%d %s differs from final8's text for its group" % (arm, s["k"], n))
            if x["premise_problems"] and arm not in WARMUP:
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
    core_ok = all(a in out["arms"] and len(out["arms"][a]["samples"]) >= 3 for a in ("I1", "I0"))
    core_problems = [p for p in out["gates"]["problems"] if p.split()[0] in ("I1", "I0")]
    out["gates"]["question_answered"] = core_ok and not core_problems
    out["gates"]["pass"] = not out["gates"]["problems"]
    return out


def print_summary(out):
    print("FINAL27 SUMMARY (same base c1c06ac6 (dev, #2401-#2404 merged): B, M = #1766 merged, P = PR #2 11c40038; 2x RTX 3090)")
    for arm, a in out["arms"].items():
        print("arm %s (%s, %s, group %s)%s" % (arm, a["tree"], a["model"], a["group"], "  (warm-up, discarded)" if arm in WARMUP else ""))
        for x in a["samples"]:
            print("  s%d seq=%s wall=%ss GPU peak %d/%d MiB RSS peak %s GiB MemAvailable min %s GiB oom_kill=%s%s %s%s%s" % (
                x["k"], x["seq"], x["wall_s"], x["gpu0_peak_mib"], x["gpu1_peak_mib"],
                x["engine_rss_peak_gib"], x["mem_available_min_gib"], x["oom_kill"], " BANK LOST" if x["bank_lost"] else "",
                "; ".join(x["premise_problems"]) or "ok",
                "" if not x["handled_oom"] else " handled allocation failures %s" % x["handled_oom"],
                "" if not x["mm_warp_verify"] else " mm-warp verify (calls, mismatches)=%s" % (x["mm_warp_verify"],)))
            if x["memx"]:
                print("    memx %s" % json.dumps(x["memx"]))
            if x.get("upload_gbps"):
                u = x["upload_gbps"]
                print("    upload GB/s: median %.2f of %d stream lines" % (sorted(u)[len(u) // 2], len(u)))
            for n, r in x["requests"].items():
                h = r.get("text_sha256") or ""
                print("    %-16s ttft=%s decode=%s identical=%s text=%s (%s)" % (n, r["ttft_s"], r["decode_tok_s"], r["identical_to_group_reference"],
                                                                         h[:8], KNOWN.get(h, "new")))
        for n, r in a["requests"].items():
            print("  median %-16s ttft=%s s (cv %s) prefill=%s tok/s decode=%s tok/s (mean %s)" % (
                n, r["ttft_s_median"], r["ttft_cv"], r["prefill_tok_s_median"], r["decode_tok_s_median"], r["decode_tok_s_mean"]))
        print("  median of per-sample mean decode: %s tok/s" % a["median_of_sample_mean_decode"])
    for k, c in out["comparisons"].items():
        print("%-6s %s" % (k, c))
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
    rows6 = lambda txt=None: [{"name": "decode_%d" % i, "ttft_s": 6.3, "decode_tok_s": 3.9,
                               "text": txt or REF_OLD["reap-dec"]["decode_%d" % i]} for i in range(6)]
    rows2 = lambda txt=None: [{"name": n, "ttft_s": 160.0, "decode_tok_s": 1.9, "text": txt or n} for n in ("long_prefill", "long_prefill_2")]
    with tempfile.TemporaryDirectory() as base:
        env = "COLI_CUDA_MOE_BATCH=1 V4_SERVE_STATS_LOG=1 V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1"
        _fake(base, "D0", 1, 1, rows6(), "COLI_CUDA_MOE_BATCH=1", "")
        _fake(base, "X3", 1, 2, rows6(), env, "peer-mirrors device=1\n")
        _fake(base, "S7", 1, 3, rows6("h"), env.replace(" V4_SERVE_STATS_LOG=1", "") + " DSV4_CUDA_PEER_FAULT=3000:sticky",
              "DSV4_CUDA_PEER_FAULT: peer-compute pass #3000\nCUDA is lost for this process\nrequest failed: block computation failed in attention\n")
        _fake(base, "FD", 1, 4, rows2(), "", "")
        _fake(base, "FP", 1, 5, rows2("x"), "V4_SERVE_STATS_LOG=1 V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1", "peer-mirrors device=1\n")
        _fake(base, "VD", 1, 6, rows2()[:1], "COLI_VULKAN=1 DSV4_CUDA=0 COLI_CUDA=0", "[VK] tier deepseek_v4: host pool on, 900 experts in 16.00 GiB\n", rc="None")
        _fake(base, "VB", 1, 7, rows2(), "COLI_VULKAN=1 DSV4_CUDA=0 COLI_CUDA=0", "[VK] tier deepseek_v4: host pool on, 900 experts in 16.00 GiB\n")
        out = summarize(base)
        probs = " ".join(out["gates"]["problems"])
        assert "X3" not in probs and "D0" not in probs, probs
        assert "S7 s1" in probs, probs
        assert "FP s1 long_prefill text differs" in probs, probs
        assert "VD s1" in probs and "VB s1" not in probs, probs
        assert "X3/D0" in out["comparisons"] and "FP/FD" in out["comparisons"], out["comparisons"]
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
