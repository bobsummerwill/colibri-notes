#!/usr/bin/env python3
"""final30_tools.py (2026-10-26, final28's for final30: the two open findings F1 (#1766 slower to first token on the
full model) and F2 (PR #2's peer helps only the second full-model prompt), review/PLAN_SAME_BASE_20261026.md; trees f27-*,
all arms on the full V4 Flash, one sample each. Groups: "full" (bench_pr1.py, in order or swapped; texts must equal the
group's first sample AND final27's / final6's references, REF_OLD["full"]); "full-fdec" (bench_fdec.py; texts must equal the
group's first sample, DB1F). Ablation (FA*, FMX) and profiled (PB1F, PM1F) arms: a text difference is a note.
  final30_tools.py arms ARM[:k]...          every arm has an ARMS entry
  final30_tools.py envcheck ARM KEY=VAL...  the arm's environment premises against this environment
final28_tools.py (2026-10-26, final26's for the restart on dev c1c06ac6; trees f27-*). final26_tools.py (2026-10-26, from final25_tools.py; final25_tools.py 2026-10-26 from final24_tools.py; final24_tools.py 2026-10-26 from final23_tools.py; final23_tools.py 2026-10-26 from final22_tools.py; final22_tools.py 2026-10-25 night from final21_tools.py; final21_tools.py 2026-10-25 from final13_tools.py; final13_tools.py 2026-10-24 from final12_tools.py; final12_tools.py from final11_tools.py; final11_tools.py from final10_tools.py; final10_tools.py from final9_tools.py; final9_tools.py from final8_tools.py; final6_tools.py 2026-10-22 before it) -- checks and summary for run_final13.sh.

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
NOT_PROF = ["DSV4_ATTN_PROF", "V4_MOE_BATCH_PROF"]
PROF = {"DSV4_ATTN_PROF": "1", "V4_MOE_BATCH_PROF": "1"}
ONE_GPU_M = [r"devices=1 placement=contiguous-layers"]
ABL_ALL = {'DSV4_CUDA_COMPRESSOR': '0', 'DSV4_CUDA_HEAD': '0', 'DSV4_CUDA_HEAD_BATCH': '0', 'DSV4_CUDA_INDEXER_PREP': '0',
           'DSV4_CUDA_INDEXER_HEADS': '0', 'DSV4_CUDA_RESIDENT_INDEXER': '0', 'DSV4_CUDA_WO_DECODE': '0', 'DSV4_CUDA_RESIDENT_EXPERTS': '0'}
SWITCHES = [k for k in ABL_ALL if k != "DSV4_CUDA_RESIDENT_EXPERTS"]
B_ENV = {"COLI_CUDA_MOE_BATCH": "1"}
P1_ENV = {"COLI_CUDA_MOE_BATCH": "1", "V4_GPU_RESIDENT_FIRST": "1"}
P2_ENV = {"COLI_CUDA_MOE_BATCH": "1", "V4_SERVE_STATS_LOG": "1", "V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1"}
B_NOT = ["DSV4_CUDA_DEVICES", "DSV4_CUDA_RESIDENT_EXPERTS", "DSV4_CUDA_VRAM_RESERVE_MB"] + PR2_OFF + SWITCHES
M1_NOT = ["DSV4_CUDA_DEVICES", "DSV4_CUDA_RESIDENT_EXPERTS", "DSV4_CUDA_VRAM_RESERVE_MB"] + PR2_OFF
P1_NOT = ["DSV4_CUDA_DEVICES", "DSV4_CUDA_VRAM_RESERVE_MB", "DSV4_CUDA_PEER_DEVICE"] + PR2_PINS + ["DSV4_CUDA_PEER_FAULT"] + SWITCHES
P2_NOT = ["DSV4_CUDA_DEVICES", "DSV4_CUDA_VRAM_RESERVE_MB"] + PR2_PINS + ["DSV4_CUDA_PEER_FAULT"] + SWITCHES
C1_NOT = ["DSV4_CUDA_DEVICES", "DSV4_CUDA_RESIDENT_EXPERTS", "DSV4_CUDA_VRAM_RESERVE_MB", "DSV4_CUDA_PEER_DEVICE"] + PR2_PINS + ["DSV4_CUDA_PEER_FAULT"] + SWITCHES
C2_NOT = ["DSV4_CUDA_DEVICES", "DSV4_CUDA_RESIDENT_EXPERTS", "DSV4_CUDA_VRAM_RESERVE_MB"] + PR2_PINS + ["DSV4_CUDA_PEER_FAULT"] + SWITCHES


def _m_one(off):
    """M, one GPU, BANKE plus the switches in `off` (each =0); every other #1766 switch must be unset."""
    env = dict(B_ENV); env.update(off)
    return ("m1766", "full", "full", env, M1_NOT + [k for k in SWITCHES if k not in off] + NOT_PROF, ONE_GPU_M, [], 2, False)


ARMS = {
    # final30 Part A (F1): full model, bench_pr1.py both prompts, one GPU. Anchors AB1F (B), AM1F (M); M with one #1766
    # switch off each, and FMX = final28's ABL_ALL (all off). #1766's one-GPU placement line must be in M's serve.log.
    "AB1F": ("base", "full", "full", B_ENV, B_NOT + NOT_PROF, [], [r"peer-mirrors device=1"], 2, False),
    "AM1F": _m_one({}),
    "FAC": _m_one({"DSV4_CUDA_COMPRESSOR": "0"}),
    "FAP": _m_one({"DSV4_CUDA_INDEXER_PREP": "0"}),
    "FAI": _m_one({"DSV4_CUDA_INDEXER_HEADS": "0"}),
    "FAR": _m_one({"DSV4_CUDA_RESIDENT_INDEXER": "0"}),
    "FAH": _m_one({"DSV4_CUDA_HEAD": "0"}),
    "FAHB": _m_one({"DSV4_CUDA_HEAD_BATCH": "0"}),
    "FAW": _m_one({"DSV4_CUDA_WO_DECODE": "0"}),
    "FMX": ("m1766", "full", "full", dict(B_ENV, **ABL_ALL), ["DSV4_CUDA_DEVICES", "DSV4_CUDA_VRAM_RESERVE_MB"] + PR2_OFF + NOT_PROF, ONE_GPU_M, [], 2, False),
    # Part A' (F1): the anchors with DSV4_ATTN_PROF=1 V4_MOE_BATCH_PROF=1 (per-phase prefill split; timing not used)
    "PB1F": ("base", "full", "full", dict(B_ENV, **PROF), B_NOT, [], [r"peer-mirrors device=1"], 2, False),
    "PM1F": ("m1766", "full", "full", dict(B_ENV, **PROF), M1_NOT + SWITCHES, ONE_GPU_M, [], 2, False),
    # Part B (F2): bench_fdec.py (bench_decode.py's first 3 prompts, 300 tokens, greedy), group full-fdec (no stored
    # reference: DB1F, the first sample, sets it). P1/C1 = VRAM-first one GPU, P2/C2 = PR #2's peer on GPU 1, M2 = #1766 0,1.
    "DB1F": ("base", "full", "full-fdec", B_ENV, B_NOT + NOT_PROF, [], [r"peer-mirrors device=1"], 3, False),
    "DP1F": ("pr2", "full", "full-fdec", P1_ENV, P1_NOT + NOT_PROF, [], [r"peer-mirrors device=1", r"CUDA is lost"], 3, False),
    "DP2F": ("pr2", "full", "full-fdec", P2_ENV, P2_NOT + NOT_PROF, [r"peer-mirrors device=1"], [r"CUDA is lost"], 3, False),
    "DM1F": ("m1766", "full", "full-fdec", B_ENV, M1_NOT + SWITCHES + NOT_PROF, ONE_GPU_M, [], 3, False),
    "DM2F": ("m1766", "full", "full-fdec", {"COLI_CUDA_MOE_BATCH": "1", "DSV4_CUDA_DEVICES": "0,1"},
             ["DSV4_CUDA_RESIDENT_EXPERTS", "DSV4_CUDA_VRAM_RESERVE_MB"] + PR2_OFF + SWITCHES + NOT_PROF, [r"v4_gpu device=1 layers=\d+\.\.42"], [], 3, False),
    "DC1F": ("cplus", "full", "full-fdec", P1_ENV, C1_NOT + NOT_PROF, ONE_GPU_M, [r"peer-mirrors device=1", r"CUDA is lost"], 3, False),
    "DC2F": ("cplus", "full", "full-fdec", P2_ENV, C2_NOT + NOT_PROF, [r"peer-mirrors device=1", r"devices=1 placement=contiguous-layers"], [r"CUDA is lost"], 3, False),
    # Part B' (F2): bench_pr1.py with BENCH_REQS=swap (long_prefill_2 first). Records are named by request, so they sit in
    # the "full" group: texts must equal AB1F's and final27's references.
    "SP1F": ("pr2", "full", "full", P1_ENV, P1_NOT + NOT_PROF, [], [r"peer-mirrors device=1", r"CUDA is lost"], 2, False),
    "SP2F": ("pr2", "full", "full", P2_ENV, P2_NOT + NOT_PROF, [r"peer-mirrors device=1"], [r"CUDA is lost"], 2, False),
    # kept for the selftest (final22's arms)
    "D0": ("base", "reap", "reap-dec", {"COLI_CUDA_MOE_BATCH": "1"}, PR2_OFF, [], [], 6, True),
    "X3": ("pr2", "reap", "reap-dec", {"COLI_CUDA_MOE_BATCH": "1", "V4_SERVE_STATS_LOG": "1", "V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1"},
           PR2_PINS + ["DSV4_CUDA_PEER_FAULT"], [r"peer-mirrors device=1"], [r"CUDA is lost"], 6, True),
}
PLAN = ["AB1F", "AM1F", "FAC", "FAP", "FAI", "FAR", "FAH", "FAHB", "FAW", "FMX", "PB1F", "PM1F",
        "DB1F", "DP1F", "DP2F", "DM1F", "DM2F", "DC1F", "DC2F", "SP1F", "SP2F"]
WARMUP = ("W", "WR", "WK", "WQ")
NOTE_ONLY = ("FAC", "FAP", "FAI", "FAR", "FAH", "FAHB", "FAW", "FMX", "PB1F", "PM1F")   # ablation / profiled: a text difference is a note
PROFILED = ("PB1F", "PM1F")
STRICT_OLD = ("full",)   # final30: a difference from REF_OLD in these groups is a problem (not a note) outside NOTE_ONLY
REF_OLD = {"reap-dec": {"decode_0": "2aa40adc7d4deb49fa84db4e070a48f4d14d10981ed8103efe83388292c84e9b",
                        "decode_1": "643fa3d128e4c41c671ffc829f6d554531d259c2eb075820085d8981e985af80",
                        "decode_2": "d64c91874f9ee22c50b943803c51eeeca79f4b613b4e551456b6735d57a318da",
                        "decode_3": "9fffce519994640443407e699f48b4c1762e8a23af2b6719531f317892cc5231",
                        "decode_4": "e3bb3209f760833a7e440b7371e35ceb90dbac2c459f65ab75131c179db1ebda",
                        "decode_5": "31cbd1c1700fd7f64dda1d79142ea7d5e2d1b442ccc6e879c44bee7a3bfb439e"},
           # final27's full references (B1F s1; identical to final6's)
           "full": {"long_prefill": "a397814f650a80340c40cca3197295a49dd5d2dc9d55211f27473036bb4efd64",
                    "long_prefill_2": "8b3f437baa347d887b5dada5bd950743d49b3181fdab9f27a87f64ad0433d4c0"}}
KNOWN = {}
KNOWN.update({v: "D0 ref " + k for k, v in REF_OLD["reap-dec"].items()})
KNOWN.update({v: "full ref " + k + " (final27/final6)" for k, v in REF_OLD["full"].items()})
TIMED = []
COMPARE = [("AM1F", "AB1F", "F1: #1766 vs dev, one GPU, full, TTFT (anchor)")]
for _a, _what in (("FAC", "compressor"), ("FAP", "indexer prepare"), ("FAI", "indexer heads"), ("FAR", "resident indexer"),
                  ("FAH", "head"), ("FAHB", "head batch"), ("FAW", "WO decode"), ("FMX", "every switch")):
    COMPARE.append((_a, "AM1F", "F1: #1766 with %s off vs #1766, TTFT per prompt" % _what))
    COMPARE.append((_a, "AB1F", "F1: #1766 with %s off vs dev, TTFT per prompt (1.0 = back to dev)" % _what))
COMPARE += [("DP2F", "DP1F", "F2: PR #2, what its peer adds, fdec decode"),
            ("DM2F", "DM1F", "F2: #1766, what its second GPU adds, fdec decode"),
            ("DC2F", "DC1F", "F2: C+, what its peer adds, fdec decode"),
            ("DP2F", "DM2F", "F2: PR #2 vs #1766, two GPUs, fdec decode"),
            ("DC2F", "DM2F", "F2: C+ vs #1766, two GPUs, fdec decode"),
            ("DM1F", "DB1F", "F2: #1766 vs dev, one GPU, fdec decode"),
            ("DP1F", "DB1F", "F2: PR #2 VRAM-first vs dev, one GPU, fdec decode"),
            ("DC1F", "DB1F", "F2: C+ VRAM-first vs dev, one GPU, fdec decode"),
            ("SP2F", "SP1F", "F2: PR #2, what its peer adds, prompts swapped (long_prefill_2 first), per prompt"),
            ("X3", "D0", "selftest pair")]


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
    out = {"kind": "final30", "arms": {}, "comparisons": {}, "group_references": {},
           "gates": {"problems": [], "notes": []}}
    present = [a for a in ARMS if samples(base, a)]
    # the reference text of each (group, request): the group's first non-warm-up sample by run order
    order = sorted(((int(s["meta"].get("seq", 0)), a, s) for a in present if a not in WARMUP for s in samples(base, a)),
                   key=lambda t: t[0])
    ref = {}
    for _, a, s in order:
        if a in NOTE_ONLY:
            continue
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
                                                          "decode_s", "position", "prompt_tokens", "text_sha256")}
                x["requests"][n]["identical_to_group_reference"] = same
                if same is False and arm in NOTE_ONLY:
                    out["gates"]["notes"].append("%s s%d %s text differs from %s's (ablation)" % (arm, s["k"], n, rs[1]))
                elif same is False and arm not in WARMUP:
                    out["gates"]["problems"].append("%s s%d %s text differs from %s's" % (arm, s["k"], n, rs[1]))
                old = REF_OLD.get(group, {}).get(n)
                x["requests"][n]["identical_to_final8"] = (r.get("text_sha256") == old) if old else None
                if old and r.get("text_sha256") != old:
                    if group in STRICT_OLD and arm not in NOTE_ONLY and arm not in WARMUP:
                        out["gates"]["problems"].append("%s s%d %s text differs from final27's reference for %s" % (arm, s["k"], n, group))
                    else:
                        out["gates"]["notes"].append("%s s%d %s differs from final27's reference text for its group" % (arm, s["k"], n))
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
    print("FINAL30 SUMMARY (same base c1c06ac6, full V4 Flash, 2x RTX 3090): F1 #1766 TTFT ablation (A, profiled A'), F2 fdec decode (B) and swapped pr1 (B')")
    for arm, a in out["arms"].items():
        print("arm %s (%s, %s, group %s)%s" % (arm, a["tree"], a["model"], a["group"], "  (warm-up, discarded)" if arm in WARMUP else
                                              "  (profiled: timing not used)" if arm in PROFILED else ""))
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
                print("    %-16s pos=%s ttft=%s decode=%s chunks=%s identical=%s text=%s (%s)" % (n, r.get("position"), r["ttft_s"], r["decode_tok_s"],
                                                                         r.get("decode_chunks"), r["identical_to_group_reference"], h[:8], KNOWN.get(h, "new")))
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


def envcheck(arm, kvs):
    """The arm's environment premises (must_env / must_not_env) against KEY=VAL... (the runner's merged env)."""
    s = {"meta": {"ready": "1", "client_rc": "0", "server_env": " ".join(kvs)}, "log": "\n".join(ARMS[arm][5]).replace("\\d+", "21").replace("\\.", "."), "rows": [{}] * ARMS[arm][7], "dir": "/nonexistent"}
    p = [x for x in premise(arm, s) if "serve.log" not in x]
    print("envcheck %s: %s" % (arm, "; ".join(p) if p else "ok"))
    return 1 if p else 0


def selftest():
    rows2 = lambda txt=None, order=("long_prefill", "long_prefill_2"): [
        {"name": n, "position": i + 1, "ttft_s": 90.0 + i, "decode_tok_s": 1.9, "decode_chunks": 30,
         "text": txt or None, "text_sha256": None if txt else REF_OLD["full"][n]} for i, n in enumerate(order)]
    rows3 = lambda txt="t", dec=1.9: [{"name": "fdec_%d" % i, "position": i + 1, "ttft_s": 5.0, "decode_chunks": 300, "decode_s": 150.0,
                                       "decode_tok_s": dec, "text": txt + str(i)} for i in range(3)]
    def clean(rows):
        for r in rows:
            if r["text"] is None:
                del r["text"]
        return rows
    one = "\nv4_gpu tier=dense-matvec device=0 devices=1 placement=contiguous-layers\n"
    with tempfile.TemporaryDirectory() as base:
        _fake(base, "AB1F", 1, 1, clean(rows2()), "COLI_CUDA_MOE_BATCH=1", "")
        _fake(base, "AM1F", 1, 2, clean(rows2()), "COLI_CUDA_MOE_BATCH=1", one)
        _fake(base, "FAC", 1, 3, rows2("c"), "COLI_CUDA_MOE_BATCH=1 DSV4_CUDA_COMPRESSOR=0", one)        # text differs: a note
        _fake(base, "FAH", 1, 4, clean(rows2()), "COLI_CUDA_MOE_BATCH=1 DSV4_CUDA_HEAD=0 DSV4_CUDA_WO_DECODE=0", one)  # two switches: premise
        _fake(base, "FMX", 1, 5, clean(rows2()), "COLI_CUDA_MOE_BATCH=1 " + " ".join("%s=%s" % kv for kv in ABL_ALL.items()), one)
        _fake(base, "PM1F", 1, 6, rows2("p"), "COLI_CUDA_MOE_BATCH=1 DSV4_ATTN_PROF=1 V4_MOE_BATCH_PROF=1", one)
        _fake(base, "DB1F", 1, 7, rows3(), "COLI_CUDA_MOE_BATCH=1", "")
        _fake(base, "DP1F", 1, 8, rows3(), "COLI_CUDA_MOE_BATCH=1 V4_GPU_RESIDENT_FIRST=1", "")
        _fake(base, "DP2F", 1, 9, rows3(dec=2.4), "COLI_CUDA_MOE_BATCH=1 V4_SERVE_STATS_LOG=1 V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1", "peer-mirrors device=1\n")
        _fake(base, "DM1F", 1, 10, rows3("x"), "COLI_CUDA_MOE_BATCH=1", one)                              # fdec text differs: problem
        _fake(base, "DM2F", 1, 11, rows3()[:2], "COLI_CUDA_MOE_BATCH=1 DSV4_CUDA_DEVICES=0,1", "v4_gpu device=1 layers=21..42\n")  # 2 rows: problem
        _fake(base, "SP1F", 1, 12, clean(rows2(order=("long_prefill_2", "long_prefill"))), "COLI_CUDA_MOE_BATCH=1 V4_GPU_RESIDENT_FIRST=1", "")
        _fake(base, "SP2F", 1, 13, rows2("s", order=("long_prefill_2", "long_prefill")), "COLI_CUDA_MOE_BATCH=1 V4_SERVE_STATS_LOG=1 V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1", "peer-mirrors device=1\n")
        _fake(base, "DC1F", 1, 14, rows3(), "COLI_CUDA_MOE_BATCH=1 V4_GPU_RESIDENT_FIRST=1", "")          # no one-GPU line: premise
        out = summarize(base)
        probs = " ".join(out["gates"]["problems"]); notes = " ".join(out["gates"]["notes"])
        for ok in ("AB1F", "AM1F", "FAC", "FMX", "PM1F", "DB1F", "DP1F", "DP2F", "SP1F"):
            assert not [q for q in out["gates"]["problems"] if q.startswith(ok + " ")], (ok, probs)
        assert "FAC s1 long_prefill text differs" in notes and "PM1F s1" in notes, notes
        assert "FAH s1: DSV4_CUDA_WO_DECODE set" in probs, probs
        assert "DM1F s1 fdec_0 text differs from DB1F s1's" in probs, probs
        assert "DM2F s1: 2 request rows, expected 3" in probs, probs
        assert "SP2F s1 long_prefill_2 text differs from AB1F s1's" in probs, probs
        assert "SP2F s1 long_prefill_2 text differs from final27's reference" in probs, probs
        assert "DC1F s1: no /devices=1 placement" in probs, probs
        assert out["group_references"]["full-fdec"]["fdec_0"]["from"] == "DB1F s1", out["group_references"]
        assert out["group_references"]["full"]["long_prefill"]["from"] == "AB1F s1", out["group_references"]
        c = out["comparisons"]
        assert c["DP2F/DP1F"]["fdec_0_decode_ratio"] == round(2.4 / 1.9, 4) and c["DP2F/DP1F"]["mean_decode_ratio"], c["DP2F/DP1F"]
        assert c["FAC/AB1F"]["long_prefill_ttft_ratio"] == 1.0 and "SP2F/SP1F" in c and "FMX/AM1F" in c, c
        assert out["arms"]["SP1F"]["samples"][0]["requests"]["long_prefill_2"]["position"] == 1
    for a in PLAN:
        assert a in ARMS, a
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
    elif cmd == "arms":
        miss = [x.split(":")[0] for x in sys.argv[2:] if x.split(":")[0] not in ARMS]
        print("arms: %s" % ("missing " + " ".join(miss) if miss else "all %d in ARMS" % len(sys.argv[2:])))
        sys.exit(1 if miss else 0)
    elif cmd == "envcheck":
        sys.exit(envcheck(sys.argv[2], sys.argv[3:]))
    else:
        print(__doc__)
        sys.exit(2)
