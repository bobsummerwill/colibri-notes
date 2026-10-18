#!/usr/bin/env python3
"""Build the PR #1 experiment manifests (upstream docs/experiments/manifest.example.json format,
validated by c/experiment_manifest.py) from the final4 run on shadowfax (2026-10-17, run_final4.sh).
Adapted from ../final/pr1/make_manifests_pr1.py.

usage: make_manifests_final4.py RAW_DIR [ENGINE_REPO]
  RAW_DIR     = the extracted bench.tgz's bench/ directory (holds W/ A/ B/ C/ G1/ X3/ meta/ run.log summary.json)
  ENGINE_REPO = optional colibri checkout; if given, the post-run amend of rb/pr1/reap-prefill is checked
                with read-only git commands (see notes.branch_after_run)
Also reads (inputs, not rewritten):
  bench.tgz, bench.tgz.sha256        the whole run (its sha256 is checked)
  ../final/pr1/pr1-*.json            the 2026-10-15 manifests (for the comparison note)
Writes, next to this script:
  evidence/final4-arm-<X>.tgz   deterministic per-arm tarball of RAW_DIR/<X>/ (what a manifest's evidence names)
  pr1-*.json                    one manifest per (changed variable, request), each with a notes block
  stats.json                    per-sample and per-arm numbers, the correctness gate and the notes

PR #2 arms G1/X3 ran one sample each: no manifest (the validator needs >= 3 samples per arm); see pr2-NOTE.md.
"""
import decimal, glob, gzip, hashlib, io, json, os, re, statistics, subprocess, sys, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(sys.argv[1])
ENGINE_REPO = sys.argv[2] if len(sys.argv) > 2 else None

# ---------------------------------------------------------------- the run's identity (meta/tips.txt, meta/trees.txt)
TGZ = os.path.join(HERE, "bench.tgz")
TGZ_SHA = hashlib.sha256(open(TGZ, "rb").read()).hexdigest()
assert open(TGZ + ".sha256").read().split()[0] == TGZ_SHA, "bench.tgz does not match bench.tgz.sha256"

DEV = "62eaa9dc249b67567b9cbc83e43fbc875a8b94c1"
PR1 = "7e9d2d20816398c25dc979e952ce924cc1012fc3"
PR1_AMENDED = "4412cc1b60371424dbbdd479ca9fde054ec55186"  # rb/pr1/reap-prefill after the run (docs-only amend)
OLD_DEV = "7b55a9b3792527f52d1a2337fd9e312b9a7983e3"   # the 2026-10-15 manifests' base


def kv(path):
    out = {}
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                out[k.strip()] = v
    return out


TIPS = kv(os.path.join(RAW, "meta", "tips.txt"))
assert TIPS["dev_sha"] == DEV and TIPS["pr1_sha"] == PR1 and TIPS["pr1_branch"] == "rb/pr1/reap-prefill", TIPS
assert TIPS["dev_is_ancestor_of_pr1"] == "yes", TIPS
TREES = {}
for block in open(os.path.join(RAW, "meta", "trees.txt")).read().split("tree=")[1:]:
    d = dict(l.strip().split("=", 1) for l in block.splitlines()[1:] if "=" in l)
    TREES[d["head"]] = d
BIN = {DEV: TREES[DEV]["binary_sha256"], PR1: TREES[PR1]["binary_sha256"]}
assert all(TREES[c]["status_lines"] == "0" for c in BIN), TREES
ARM_TREE = {"A": DEV, "B": PR1, "C": PR1}
ARM_EXTRA = {"A": {}, "B": {}, "C": {"COLI_CUDA_MOE_DOUBLE": "1"}}
REQS = {"long_prefill": 1992, "long_prefill_2": 3005}

# the exact prompts of bench_pr1.py (meta/bench_pr1.py in the tarball); prompt_hash = sha256(json.dumps(PROMPTS))
filler = " ".join(["The quick brown fox jumps over the lazy dog while the committee reviews section %d of the quarterly infrastructure report." % i for i in range(90)])
filler2 = " ".join(["Station %d logged a temperature of %d degrees and a wind speed of %d knots during the overnight survey." % (i, 10 + i % 17, 3 + i % 11) for i in range(130)])
PROMPTS = [filler + "\n\nSummarize the above in one sentence.", filler2 + "\n\nWhich station had the highest wind speed? Answer briefly."]
PROMPT_HASH = hashlib.sha256(json.dumps(PROMPTS).encode()).hexdigest()
assert PROMPT_HASH == "13fc61ddae676a87f55d68c32d7488f90f0c99af634f30dc846eaa8d1707257f", PROMPT_HASH
BENCH_SRC = open(os.path.join(RAW, "meta", "bench_pr1.py")).read()
assert all(s in BENCH_SRC for s in (
    '"The quick brown fox jumps over the lazy dog while the committee reviews section %d of the quarterly infrastructure report." % i for i in range(90)',
    '"Station %d logged a temperature of %d degrees and a wind speed of %d knots during the overnight survey." % (i, 10 + i % 17, 3 + i % 11) for i in range(130)',
    '"\\n\\nSummarize the above in one sentence.", 100, 1992', '"\\n\\nWhich station had the highest wind speed? Answer briefly.", 60, 3005')), \
    "meta/bench_pr1.py prompts differ from the ones hashed here"

# the fixed 2026-10-15 dev reference texts (meta/final_tools.py REF_PR1)
TOOLS = open(os.path.join(RAW, "meta", "final_tools.py")).read()
REF_1015 = {n: re.search(r'"%s": "([0-9a-f]{64})"' % n, TOOLS).group(1) for n in REQS}

# ---------------------------------------------------------------- host (meta/host.txt, run.log)
HOST = open(os.path.join(RAW, "meta", "host.txt")).read()
RUNLOG = open(os.path.join(RAW, "run.log")).read()


def hx(pat, text=HOST):
    m = re.search(pat, text, re.M)
    assert m, pat
    return m.group(1)


CPU_MODEL = hx(r"^Model name:\s+(.+?)\s*$")
assert CPU_MODEL == "AMD Ryzen 9 9950X3D 16-Core Processor", CPU_MODEL
CPUS, CORES, TPC = int(hx(r"^CPU\(s\):\s+(\d+)")), int(hx(r"^Core\(s\) per socket:\s+(\d+)")), int(hx(r"^Thread\(s\) per core:\s+(\d+)"))
SOCKETS = int(hx(r"^Socket\(s\):\s+(\d+)"))
MAXMHZ = float(hx(r"^CPU max MHz:\s+([\d.]+)"))
GOV = hx(r"^governor=(\S+)")
MEM_B = int(re.search(r"^Mem:\s+(\d+)\s", HOST.split("total        used")[1], re.M).group(1))  # the `free -b` table
SWAP_B = int(re.search(r"^Swap:\s+(\d+)\s", HOST.split("total        used")[1], re.M).group(1))
KERNEL = hx(r"^Linux shadowfax (\S+) ")
OSNAME = hx(r'^PRETTY_NAME="([^"]+)"')
GCC = hx(r"^gcc \(.*\) (\S+)$")
NVCC = hx(r"Cuda compilation tools, release [\d.]+, V(\S+)")
PY = hx(r"^Python (\S+)")
GPUS = [m.groups() for m in re.finditer(
    r"^\d, (NVIDIA GeForce RTX 3090), GPU-[\w-]+, ([\d.]+), (\d+) MiB, (\d+), (\d+), \d+, \d+,", HOST, re.M)]
assert len(GPUS) == 2 and GPUS[0] == GPUS[1], GPUS
GPU_NAME, DRIVER, GPU_MIB, PCIE_GEN_MAX, PCIE_W_MAX = GPUS[0]
assert "GPU0\t X \tNV4" in HOST, "NVLink topology"
DISK = hx(r"^nvme0n1\s+(WD_BLACK SN850X 4000GB)")
ROOTFS = re.search(r"^/\s+/dev/nvme0n1p2 (\S+)\s+(\S+)", HOST, re.M).groups()
MODEL_KIB = int(hx(r"^total (\d+)$"))  # `ls -la` of the model directory, 1 KiB blocks
MEMAVAIL_MIB = int(hx(r"preflight common ok: MemAvailable=(\d+) MiB", RUNLOG))
LLAMA_DOWN = hx(r"\[(\S+ \S+)\] stopping llama-server-active", RUNLOG).replace(" ", "T")
LLAMA_UP = hx(r"restore: llama health \S+ \(ok=1\) at (\S+?);", RUNLOG)[:19]

# ---------------------------------------------------------------- per-sample data


def nvsmi(path):
    """peak memory.used per GPU and the PCIe links seen in the nvidia-smi log of one sample"""
    peak, links = {}, {}
    for line in open(path):
        p = [x.strip() for x in line.split(",")]
        if len(p) < 11 or p[0].startswith("timestamp"):
            continue
        i = p[1]
        peak[i] = max(peak.get(i, 0), int(p[2].split()[0]))
        links.setdefault(i, set()).add("gen%s x%s" % (p[9], p[10]))
    return {"mem_used_mib_max": peak, "links": {i: sorted(v) for i, v in links.items()}}


def load(arm):
    out = []
    for d in sorted(glob.glob(os.path.join(RAW, arm, "s*")), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)))):
        meta = kv(os.path.join(d, "sample.txt"))
        env = kv(os.path.join(d, "engine_environ.txt"))
        rows = [json.loads(l) for l in open(os.path.join(d, "bench.jsonl")) if l.strip().startswith("{")]
        for r in rows:  # recompute the hash from the stored full text
            assert hashlib.sha256(r["text"].encode()).hexdigest() == r["text_sha256"], (d, r["name"])
        out.append({"dir": os.path.relpath(d, RAW), "meta": meta, "env": env, "rows": {r["name"]: r for r in rows},
                    "serve_log": open(os.path.join(d, "serve.log"), errors="replace").read(),
                    "gpu": nvsmi(os.path.join(d, "nvsmi.csv"))})
    return out


COMMON_ENV = {"COLI_CUDA": "1", "COLI_GPU": "0", "DSV4_CUDA_DEVICE": "0", "CUDA_DENSE": "1", "COLI_CUDA_PIPE": "2",
              "COLI_CUDA_ATTN_BATCH": "1", "COLI_CUDA_MOE_BATCH": "1", "V4_MOE_REFILL_GROUP": "12",
              "V4_LOADER_LANES": "3", "V4_PREFIX_CKPT_DISK": "0", "ram_gib": "40", "ctx": "8192", "temperature": "0"}
CE = kv(os.path.join(RAW, "meta", "common_env.txt"))
assert dict(x.split("=", 1) for x in CE["COMMON_ENV"].split()) == {k: v for k, v in COMMON_ENV.items() if k.isupper()}, CE
assert "--ram 40 --ctx 8192 --host 127.0.0.1 --port 8090" in CE["SERVE_ARGS"], CE


def env_ok(s, arm):
    want = {k: v for k, v in COMMON_ENV.items() if k.isupper()}
    want.update(ARM_EXTRA[arm])
    want.update({"RAM_GB": "40", "CTX": "8192"})
    bad = [k for k, v in want.items() if s["env"].get(k) != v]
    bad += ["%s set" % k for k in ("COLI_CUDA_MOE_DOUBLE",) if k in s["env"] and k not in ARM_EXTRA[arm]]
    if s["meta"].get("tree_head") != ARM_TREE[arm] or s["meta"].get("binary_sha256") != BIN[ARM_TREE[arm]]:
        bad.append("tree/binary")
    if s["meta"].get("client_rc") != "0" or s["meta"].get("tree_dirty_lines") != "0":
        bad.append("client_rc/dirty")
    if s["meta"].get("usage_before") != "size=0 mtime=1791867517" or s["meta"].get("usage_after") != s["meta"].get("usage_before"):
        bad.append(".coli_usage not empty/unchanged")
    log = s["serve_log"]
    if "no usage history says which, so it fills as experts are used" not in log:
        bad.append("serve.log: expert-cache prewarm line missing")
    batch = re.findall(r"v4_gpu moe-batch=(\S+)", log)
    if arm == "A" and batch != ["off"]:
        bad.append("serve.log: dev should refuse moe-batch, got %s" % batch)
    if arm != "A" and (not batch or any(b != "on" for b in batch)):
        bad.append("serve.log: moe-batch %s" % batch)
    if (arm == "C") != ("v4_gpu moe-double=on" in log):
        bad.append("serve.log: moe-double state")
    return bad


ARMS = {a: load(a) for a in ARM_TREE}
assert all(len(ss) == 3 for ss in ARMS.values()), {a: len(ss) for a, ss in ARMS.items()}
REF = {name: {s["rows"][name]["text_sha256"] for s in ARMS["A"]} for name in REQS}
assert all(len(v) == 1 for v in REF.values()), REF
REF = {k: next(iter(v)) for k, v in REF.items()}
A_EQ_1015 = REF == REF_1015


def gate(arm):
    """every request of every sample of this arm: full text sha256 == arm A's, env/tree/binary/serve.log checked"""
    problems = []
    for s in ARMS[arm]:
        problems += ["%s: %s" % (s["dir"], p) for p in env_ok(s, arm)]
        for name in REQS:
            r = s["rows"].get(name)
            if r is None:
                problems.append("%s %s missing" % (s["dir"], name))
            elif r["text_sha256"] != REF[name]:
                problems.append("%s %s sha %s != A %s" % (s["dir"], name, r["text_sha256"][:12], REF[name][:12]))
    return problems


def evidence(arm):
    buf = io.BytesIO()
    # gzip header mtime 0 and no file name: the tarball is byte-identical on every run
    with gzip.GzipFile(filename="", fileobj=buf, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as t:
        for f in sorted(glob.glob(os.path.join(RAW, arm, "s*", "*"))):
            info = t.gettarinfo(f, arcname=os.path.join("final4", os.path.relpath(f, RAW)))
            info.mtime = 0; info.uid = info.gid = 0; info.uname = info.gname = ""
            with open(f, "rb") as fh:
                t.addfile(info, fh)
    data = buf.getvalue()
    os.makedirs(os.path.join(HERE, "evidence"), exist_ok=True)
    path = os.path.join(HERE, "evidence", "final4-arm-%s.tgz" % arm)
    open(path, "wb").write(data)
    return {"uri": os.path.relpath(path, HERE), "sha256": hashlib.sha256(data).hexdigest(),
            "contains": "final4/%s/s*/ (serve.log, bench.jsonl with full texts, sample.txt, engine_environ.txt, serve_environ.txt, procs.txt, nvsmi.csv, gpu_*.csv, mode.txt, bench.out/err, error_count.txt)" % arm,
            "full_run": "bench.tgz (sha256 %s, also in bench.tgz.sha256): all arms incl. W/G1/X3, meta/ (host, tips, trees, scripts, build logs, llama before/after), run.log, summary.json" % TGZ_SHA}


EVID = {}


def stat(xs):
    return {"mean": round(statistics.mean(xs), 4), "stdev": round(statistics.stdev(xs), 4),
            "cv_pct": round(100 * statistics.stdev(xs) / statistics.mean(xs), 2),
            "median": round(statistics.median(xs), 4), "min": min(xs), "max": max(xs), "n": len(xs)}


def dmean(xs, places=4):
    """mean of values recorded with <= 4 decimals, rounded half-up on the exact decimal mean"""
    m = sum(decimal.Decimal(str(x)) for x in xs) / len(xs)
    return float(m.quantize(decimal.Decimal(1).scaleb(-places), rounding=decimal.ROUND_HALF_UP))


def vals(arm, request, key):
    return [s["rows"][request][key] for s in ARMS[arm]]


def pct(a, b):
    return 100.0 * (a / b - 1)


def hm(ts):
    return ts[11:19]


# ---------------------------------------------------------------- notes
ORDER = sorted(((int(s["meta"]["seq"]), "%s s%s" % (a, s["meta"]["sample"])) for a, ss in ARMS.items() for s in ss))
W = load("W")[0]
assert W["meta"]["seq"] == "1" and W["meta"]["tree_head"] == PR1 and W["rows"]["long_prefill"]["text_sha256"] == REF["long_prefill"]
ORDER_TEXT = " ".join(n for _, n in ORDER)
assert ORDER_TEXT == "A s1 B s1 C s1 B s2 C s2 A s2 C s3 A s3 B s3", ORDER_TEXT

BASE_NOTE = (
    "New base: arm A is upstream dev %s ('%s'); the 2026-10-15 manifests (../final/pr1) used dev %s. The pr1 tip %s "
    "(rb/pr1/reap-prefill, tree %s) is %s commits on top of this base (meta/tips.txt: dev_is_ancestor_of_pr1=yes). "
    "dev still refuses the batched GPU MoE for the 132-expert model: every A serve.log has 'v4_gpu moe-batch=off "
    "(unsupported model geometry)', every B/C serve.log 'v4_gpu moe-batch=on bank=132-experts', and only C "
    "'v4_gpu moe-double=on'." % (DEV, TREES[DEV]["subject"], OLD_DEV, PR1, TIPS["pr1_tree"], TIPS["commits_dev_to_pr1"]))

PREWARM_NOTE = (
    "Expert-cache prewarm: the base includes upstream #2111 (463f8443, 'a serve fills the expert cache before its first "
    "request (#2109)'), so both dev and the pr1 tip prewarm at serve start. At --ram 40 the plan holds 59 of 132 experts "
    "per layer, so the prewarm picks experts from .coli_usage, which was empty (size 0, same mtime) before and after "
    "every sample (sample.txt usage_before/usage_after; summary.json usage_history_changed []). Every A/B/C serve.log "
    "says '[v4] expert cache: 59 of 132 experts per layer fit and no usage history says which, so it fills as experts "
    "are used': the prewarm read nothing, and the expert cache started empty in every sample as on 2026-10-15.")

A1 = vals("A", "long_prefill", "ttft_s")
CACHE_NOTE = (
    "The OS page cache was not dropped between samples. The run began with one discarded warm-up sample W (seq 1, a "
    "separate server process on the pr1 tip, both requests; texts identical to the reference) before A s1. Unlike the "
    "2026-10-15 run, no host journal was captured for this window, so the serve unit's memory peak (used then to show "
    "round 1 was partly cold) is not available and the cache state per sample cannot be stated from evidence. "
    "Observed: A request-1 TTFT s1 %.3f s vs s2/s3 %.3f/%.3f s (s1 %.2f%% slower than the s2/s3 mean); medians are "
    "over all 3 samples. Run order (seq 2-10): %s." % (
        A1[0], A1[1], A1[2], pct(A1[0], statistics.mean(A1[1:])), ORDER_TEXT))


def spread(xs):
    return {"stdev": statistics.stdev(xs), "cv_pct": 100 * statistics.stdev(xs) / statistics.mean(xs)}


TT = {(a, n): spread(vals(a, n, "ttft_s")) for a in ARMS for n in REQS}
TK = {(a, n): spread(vals(a, n, "prefill_tok_s")) for a in ARMS for n in REQS}
VAR_NOTE = ("TTFT stdev / CV over 3 samples, every arm and request: %s; the largest prompt tok/s CV is %.2f%%." % (
    ", ".join("%s r%d %.3f s / %.2f%%" % (a, 1 if n == "long_prefill" else 2, v["stdev"], v["cv_pct"]) for (a, n), v in TT.items()),
    max(v["cv_pct"] for v in TK.values())))

# the 2026-10-15 manifests (same machine, same client, previous base and an earlier pr1 commit)
OLDM = {}
for f in ("pr1-engine-1992.json", "pr1-engine-3005.json", "pr1-moe-double-1992.json", "pr1-moe-double-3005.json"):
    OLDM[f] = json.load(open(os.path.join(HERE, "..", "final", "pr1", f)))
OLD_PR1 = OLDM["pr1-engine-1992.json"]["trial"]["config"]["engine_commit"]
assert OLDM["pr1-engine-1992.json"]["baseline"]["config"]["engine_commit"] == OLD_DEV
OLD_MED = {}
for f, m in OLDM.items():
    req = "long_prefill" if "1992" in f else "long_prefill_2"
    for side in ("baseline", "trial"):
        arm = m[side]["samples"]["sample_dirs"][0].split("/")[0]
        OLD_MED[(arm, req)] = m[side]["median_tok_s"]
        assert m[side]["samples"]["text_sha256"] == [REF[req]] * 3, (f, side)  # same texts then
REPRO = {k: {"median_tok_s_2026_10_15": v, "median_tok_s_final4": statistics.median(vals(k[0], k[1], "prefill_tok_s")),
             "delta_pct": round(pct(statistics.median(vals(k[0], k[1], "prefill_tok_s")), v), 2),
             "commit_2026_10_15": OLD_DEV if k[0] == "A" else OLD_PR1, "commit_final4": ARM_TREE[k[0]]}
         for k, v in OLD_MED.items()}
OLD_RATIO = {f: m["speedup_median_tok_s"] for f, m in OLDM.items()}

DECODE = {n: {"A_mean": dmean(vals("A", n, "decode_tok_s")), "B_mean": dmean(vals("B", n, "decode_tok_s")),
              "C_mean": dmean(vals("C", n, "decode_tok_s"))} for n in REQS}
for n, d in DECODE.items():
    d["B_vs_A_pct"] = round(pct(d["B_mean"], d["A_mean"]), 2)
    d["C_vs_B_pct"] = round(pct(d["C_mean"], d["B_mean"]), 2)
CHUNKS = {n: sorted({s["rows"][n]["decode_chunks"] for ss in ARMS.values() for s in ss}) for n in REQS}
DECODE_NOTE = (
    "Observation (not this manifest's hypothesis): decode right after the prefill, bench_pr1.py decode_tok_s (streamed "
    "chunks per second after the first chunk; %s chunks on request 1, %s on request 2), 3-sample means: request 1 "
    "A %.4f, B %.4f (%+.2f%% vs A), C %.4f (%+.2f%% vs B); request 2 A %.4f, B %.4f (%+.2f%%), C %.4f (%+.2f%%). The "
    "texts are byte-identical, so this is a speed difference only; the cause is not established by this run." % (
        "/".join(map(str, CHUNKS["long_prefill"])), "/".join(map(str, CHUNKS["long_prefill_2"])),
        DECODE["long_prefill"]["A_mean"], DECODE["long_prefill"]["B_mean"], DECODE["long_prefill"]["B_vs_A_pct"],
        DECODE["long_prefill"]["C_mean"], DECODE["long_prefill"]["C_vs_B_pct"],
        DECODE["long_prefill_2"]["A_mean"], DECODE["long_prefill_2"]["B_mean"], DECODE["long_prefill_2"]["B_vs_A_pct"],
        DECODE["long_prefill_2"]["C_mean"], DECODE["long_prefill_2"]["C_vs_B_pct"]))

N_OUT = {a: sum(len(s["rows"]) for s in ss) for a, ss in ARMS.items()}
ALL_IDENTICAL = all(r["text_sha256"] == REF[n] for ss in ARMS.values() for s in ss for n, r in s["rows"].items())
OUTPUT_NOTE = (
    "%d outputs in 9 samples (%s), %s byte-identical (full-text sha256, recomputed from the stored text) to arm A's "
    "per-request text; A's texts are %s the fixed 2026-10-15 dev references (meta/final_tools.py REF_PR1: %s). The "
    "discarded warm-up W also matched." % (
        sum(N_OUT.values()), ", ".join("%s %d" % kv for kv in N_OUT.items()), "all" if ALL_IDENTICAL else "NOT all",
        "identical to" if A_EQ_1015 else "DIFFERENT from", ", ".join("%s %s" % (n, h[:12]) for n, h in REF_1015.items())))

GPU_PEAK = {a: {i: max(s["gpu"]["mem_used_mib_max"][i] for s in ss) for i in ("0", "1")} for a, ss in ARMS.items()}
GPU_LINKS = sorted({l for ss in ARMS.values() for s in ss for l in s["gpu"]["links"]["0"]})
RESERVE = {a: sorted(set(re.findall(r"mirror-reserve=\d+MiB", " ".join(s["serve_log"] for s in ss)))) for a, ss in ARMS.items()}
GPU_NOTE = ("Peak memory.used (nvsmi.csv, max over the arm's samples): GPU0 %s MiB; GPU1 %s MiB (unused). GPU0 link "
            "seen during samples: %s (PCIe gen%s x%s-capable cards, meta/host.txt). Decode-mirror VRAM reserve from "
            "serve.log: %s." % (
                ", ".join("%s %d" % (a, v["0"]) for a, v in GPU_PEAK.items()),
                "/".join(sorted({str(v["1"]) for v in GPU_PEAK.values()})), ", ".join(GPU_LINKS), PCIE_GEN_MAX, PCIE_W_MAX,
                ", ".join("%s %s" % (a, "/".join(v) or "none printed") for a, v in RESERVE.items())))

HOST_NOTE = (
    "llama.cpp (llama-server-active) was stopped %s and restored %s PDT (health ok, run.log), so the PR arms ran "
    "without it on the GPUs. The host journal (Hermes watchdog / cron activity) was not captured for this run, so "
    "whether other host jobs ran during samples cannot be stated from evidence." % (hm(LLAMA_DOWN), hm(LLAMA_UP)))

BRANCH_NOTE = None
if ENGINE_REPO:
    def git(*a):
        return subprocess.run(["git", "-C", ENGINE_REPO] + list(a), check=True, capture_output=True, text=True).stdout.strip()
    tip_now = git("rev-parse", "rb/pr1/reap-prefill")
    changed = git("diff", "--name-only", PR1, tip_now).split()
    assert git("rev-parse", tip_now + "^") == git("rev-parse", PR1 + "^"), "the amend should keep the parent"
    BRANCH_NOTE = (
        "After the run, rb/pr1/reap-prefill was amended (local ref now %s, committed %s): same parent as the measured "
        "%s, diff limited to %s, so the measured engine code equals the current branch tip; the manifests record the "
        "measured commit %s." % (tip_now, git("log", "-1", "--format=%cI", tip_now), PR1[:8],
                                  ", ".join(changed) or "nothing", PR1))
    assert all(not f.startswith("c/") for f in changed), changed

WARMUP = {"long_prefill": (0, "warmup_runs 0: request 1 (long_prefill) is the first request the client sends to each "
                              "freshly started server process; no warm-up request precedes it in that process. (The run's "
                              "discarded W sample was a separate server process before A s1; see notes.cache_state.)"),
          "long_prefill_2": (1, "warmup_runs 1: request 2 (long_prefill_2) is served by the same server process right "
                                "after request 1 (long_prefill, the 1992-token prompt, max_tokens 100), which is its one "
                                "warm-up here; request 1 is itself measured, in the matching -1992 manifest")}

MODEL = ("DeepSeek V4 Flash REAP 150B (puwaer/DeepSeek-V4-Flash-0731-reap-150b, 132 routed experts, fp4 experts / "
         "fp8 dense), colibri safetensors layout, %.1f GiB on disk" % (MODEL_KIB / 2 ** 20))
BUILD = "cd c && make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j8"
assert ("(%s)" % BUILD.split("&& ")[1]) in RUNLOG, "build command differs from run.log"
RUN = ("python3 c/coli serve --model <REAP dir> --ram 40 --ctx 8192 --host 127.0.0.1 --port 8090, server environment "
       "exactly PATH/HOME/USER/LOGNAME/LANG + the config's env keys (env -i); one freshly started server process per "
       "sample (OS page cache not dropped between samples, see notes.cache_state); one discarded warm-up sample W on "
       "the pr1 tip first, then 3 samples per arm interleaved with the arm order rotated per round "
       "(A B C / B C A / C A B); client bench_pr1.py sends request 1 (long_prefill: 1992-token prompt, max_tokens 100) "
       "then request 2 (long_prefill_2: 3005-token prompt, max_tokens 60), streamed, greedy; "
       "tok_s = prompt_tokens / time-to-first-token of the named request")

HW = {
    "cpu": "%s, %d cores / %d threads, %d socket, governor %s, max %.2f GHz (engine: 16 OpenMP compute threads + "
           "3 expert-loader lanes)" % (CPU_MODEL.replace(" 16-Core Processor", ""), CORES * SOCKETS, CPUS, SOCKETS, GOV, MAXMHZ / 1000),
    "ram": "%.1f GiB (free -b; %.1f GiB swap); engine --ram 40 (ram_tiers: 31.59 GiB routed-expert RAM cache = 59 of 132 "
           "experts per layer, 6.23 GiB dense resident); MemAvailable %.1f GiB (%d MiB) at preflight with llama.cpp "
           "still loaded" % (MEM_B / 2 ** 30, SWAP_B / 2 ** 30, MEMAVAIL_MIB / 1024, MEMAVAIL_MIB),
    "storage": "%s NVMe, %s root filesystem (%s), model read with O_DIRECT (v4_ssd_io mode=direct-aligned); OS page "
               "cache not dropped between samples" % (DISK.replace("4000GB", "4 TB"), ROOTFS[0], ROOTFS[1]),
    "os": "%s, kernel %s, gcc %s, Python %s, CUDA toolkit %s (generic build CUDA_ARCH=sm_86 NO_TC=1), NVIDIA driver %s; "
          "2x %s %d GB, PCIe gen%s x%s-capable, GPU0 ran at %s during samples, NVLink NV4 bridge; only GPU0 used "
          "(COLI_GPU=0 DSV4_CUDA_DEVICE=0)" % (OSNAME, KERNEL, GCC, PY, NVCC, DRIVER, GPU_NAME,
                                               int(GPU_MIB) // 1024, PCIE_GEN_MAX, PCIE_W_MAX, "/".join(GPU_LINKS)),
}
for k in ("ram", "storage", "cpu", "os"):
    assert HW[k]
assert "ram_tiers available=40.00GiB dense=resident(6.23GiB) target_slots=59 target_cache=31.59GiB" in ARMS["A"][0]["serve_log"]
assert "16 compute threads" in ARMS["A"][0]["serve_log"]

UNFILLED_NOTE = (
    "Fields of the 2026-10-15 generator not filled from this run's evidence: the host journal (serve-unit memory peaks, "
    "watchdog and cron activity) was not captured, so there is no cold/warm-round analysis and no host-activity "
    "statement; there is no verify arm V and no E/F arms, so no expert-major / mm-warp manifests; memory type (DDR5), "
    "the cpufreq driver (amd-pstate) and the NVMe PCIe generation are not recorded in this run's meta/host.txt and are "
    "omitted from hardware; the model's HF id and expert format come from the model directory being the same as on "
    "2026-10-15 (meta/host.txt lists the same directory and file mtimes of 2026-10-12), not from a new hash of the weights.")

ROUND_NOTES = {}


def run(arm, cfg, request):
    ss = ARMS[arm]
    rows = [s["rows"][request] for s in ss]
    speeds = [r["prefill_tok_s"] for r in rows]
    probs = gate(arm)
    if arm not in EVID:
        EVID[arm] = evidence(arm)
    return {
        "config": cfg,
        "samples": {"tok_s": speeds, "ttft_s": [r["ttft_s"] for r in rows], "decode_tok_s": [r["decode_tok_s"] for r in rows],
                    "prompt_tokens": [r["prompt_tokens"] for r in rows],
                    "text_sha256": [r["text_sha256"] for r in rows], "sample_dirs": [s["dir"] for s in ss],
                    "run_order_seq": [int(s["meta"]["seq"]) for s in ss],
                    "tok_s_mean": round(statistics.mean(speeds), 4), "tok_s_stdev": round(statistics.stdev(speeds), 4),
                    "ttft_s_mean": round(statistics.mean([r["ttft_s"] for r in rows]), 3),
                    "ttft_s_stdev": round(statistics.stdev([r["ttft_s"] for r in rows]), 3)},
        "median_tok_s": statistics.median(speeds),
        "engine": {"commit": ARM_TREE[arm], "binary_sha256": BIN[ARM_TREE[arm]]},
        "quality": {"method": ("greedy (temperature 0): the full output text of both requests (long_prefill, long_prefill_2) "
                               "in every sample of this arm has the same sha256 (recomputed from the stored text) as "
                               "upstream dev (arm A, itself identical across its 3 samples and to the 2026-10-15 dev "
                               "reference); the engine's real environment (/proc/<pid>/environ), tree SHA, binary sha256, "
                               "empty unchanged .coli_usage and the serve.log moe-batch/moe-double lines checked per sample"),
                    "passed": not probs, "problems": probs},
        "evidence": EVID[arm],
    }


OUT = []
ENG = "engine_commit"


def manifest(name, hypothesis, changed, base_arm, base_val, trial_arm, trial_val, request, fixed=None):
    bc = dict(COMMON_ENV); bc.update(fixed or {}); bc[changed] = base_val
    tc = dict(COMMON_ENV); tc.update(fixed or {}); tc[changed] = trial_val
    b = run(base_arm, bc, request); t = run(trial_arm, tc, request)
    ratio = t["median_tok_s"] / b["median_tok_s"]
    rec = {"version": 1, "hypothesis": hypothesis, "commit": PR1, "model": MODEL,
           "command": "build: %s; run: %s; metric: request '%s' (%d prompt tokens); baseline arm %s, trial arm %s" % (
               BUILD, RUN, request, REQS[request], base_arm, trial_arm),
           "prompt_hash": PROMPT_HASH, "hardware": HW, "warmup_runs": WARMUP[request][0],
           "changed_variables": [changed], "baseline": b, "trial": t,
           "outcome": "improvement" if ratio > 1.01 else ("regression" if ratio < 0.99 else "no-change"),
           "speedup_median_tok_s": round(ratio, 4)}
    rn = 1 if request == "long_prefill" else 2
    rec["notes"] = {
        "run": "final4, shadowfax, samples %s - %s PDT (llama.cpp down %s - %s); bench.tgz sha256 %s" % (
            hm(min(s["meta"]["start"] for ss in ARMS.values() for s in ss)), hm(max(s["meta"]["end"] for ss in ARMS.values() for s in ss)),
            hm(LLAMA_DOWN), hm(LLAMA_UP), TGZ_SHA),
        "base": BASE_NOTE,
        "prewarm": PREWARM_NOTE,
        "warmup": WARMUP[request][1],
        "cache_state": CACHE_NOTE,
        "variability": "TTFT stdev / CV over 3 samples: %s; prompt tok/s CV %s" % (
            "; ".join("%s r%d %.3f s / %.2f%%" % (a, rn, TT[(a, request)]["stdev"], TT[(a, request)]["cv_pct"]) for a in (base_arm, trial_arm)),
            "; ".join("%s %.2f%%" % (a, TK[(a, request)]["cv_pct"]) for a in (base_arm, trial_arm))),
        "earlier_run_2026_10_15": "; ".join("%s median %.4f tok/s on 2026-10-15 (%s) -> %.4f here (%s, %+.2f%%)" % (
            a, REPRO[(a, request)]["median_tok_s_2026_10_15"], REPRO[(a, request)]["commit_2026_10_15"][:8],
            REPRO[(a, request)]["median_tok_s_final4"], REPRO[(a, request)]["commit_final4"][:8], REPRO[(a, request)]["delta_pct"])
            for a in (base_arm, trial_arm)) + "; median ratio then %.4f (../final/pr1/%s), now %.4f. Both arms changed commit "
            "(new dev base; pr1 %s then), so this is a comparison across commits, not a repeat." % (
                OLD_RATIO[name], name, round(ratio, 4), OLD_PR1[:8]),
        "outputs": OUTPUT_NOTE,
        "gpu": GPU_NOTE,
        "host": HOST_NOTE,
        "not_filled": UNFILLED_NOTE,
    }
    if BRANCH_NOTE:
        rec["notes"]["branch_after_run"] = BRANCH_NOTE
    if changed == ENG:
        rec["notes"]["observation_decode"] = DECODE_NOTE
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(rec, f, indent=2); f.write("\n")
    OUT.append(name)
    print("%-26s %-11s %8.4f -> %8.4f tok/s (%.4fx) quality %s/%s" % (
        name, rec["outcome"], b["median_tok_s"], t["median_tok_s"], ratio, b["quality"]["passed"], t["quality"]["passed"]))


manifest("pr1-engine-1992.json",
         "Batched GPU prefill for expert-pruned checkpoints (pr1 tip) speeds up REAP-132 prefill over upstream dev, which "
         "refuses the GPU batch path for a 132-expert model (moe-batch=off) and runs the CPU expert union",
         ENG, "A", DEV, "B", PR1, "long_prefill")
manifest("pr1-engine-3005.json",
         "Batched GPU prefill for expert-pruned checkpoints (pr1 tip) also speeds up REAP-132 prefill over upstream dev "
         "(CPU expert union, moe-batch=off) on a longer prompt: request 2 (3005 tokens), which each server process serves "
         "right after request 1 (the 1992-token prompt of pr1-engine-1992.json, counted here as its one warm-up run)",
         ENG, "A", DEV, "B", PR1, "long_prefill_2")
manifest("pr1-moe-double-1992.json",
         "The double expert bank (COLI_CUDA_MOE_DOUBLE=1: the next bank is filled on the aux stream while the current one "
         "computes) speeds up batched REAP-132 prefill over the single bank, with outputs unchanged",
         "COLI_CUDA_MOE_DOUBLE", "B", "unset (default 0)", "C", "1", "long_prefill", fixed={ENG: PR1})
manifest("pr1-moe-double-3005.json",
         "The double expert bank (COLI_CUDA_MOE_DOUBLE=1) also speeds up batched REAP-132 prefill over the single bank, "
         "with outputs unchanged, on request 2 (3005 tokens), which each server process serves right after request 1 "
         "(the 1992-token prompt of pr1-moe-double-1992.json, counted here as its one warm-up run)",
         "COLI_CUDA_MOE_DOUBLE", "B", "unset (default 0)", "C", "1", "long_prefill_2", fixed={ENG: PR1})

# per-arm statistics and the gate, for the report
stats = {"bench_tgz_sha256": TGZ_SHA, "reference_A_sha256": REF, "reference_2026_10_15_sha256": REF_1015,
         "A_identical_to_2026_10_15_reference": A_EQ_1015, "arms": {}}
for arm, ss in ARMS.items():
    a = {"samples": [], "requests": {}, "gate_problems": gate(arm)}
    for s in ss:
        a["samples"].append({"dir": s["dir"], "seq": int(s["meta"]["seq"]), "start": s["meta"].get("start"),
                             "end": s["meta"].get("end"), "wall_s": int(s["meta"]["wall_s"]), "ready_s": int(s["meta"]["ready_s"]),
                             "gpu_mem_used_mib_max": s["gpu"]["mem_used_mib_max"], "gpu_links": s["gpu"]["links"],
                             "requests": {n: {k: r[k] for k in ("ttft_s", "prefill_tok_s", "decode_tok_s", "decode_chunks", "total_s", "text_sha256")}
                                          for n, r in s["rows"].items()}})
    for name in REQS:
        a["requests"][name] = {k: stat(vals(arm, name, k)) for k in ("ttft_s", "prefill_tok_s", "decode_tok_s")}
        a["requests"][name]["identical_to_A"] = all(r == REF[name] for r in vals(arm, name, "text_sha256"))
    stats["arms"][arm] = a
stats["notes"] = {"base": BASE_NOTE, "prewarm": PREWARM_NOTE, "cache_state": CACHE_NOTE, "variability": VAR_NOTE,
                  "earlier_run_2026_10_15_table": {"%s %s" % k: v for k, v in REPRO.items()},
                  "earlier_run_2026_10_15_ratios": OLD_RATIO, "observation_decode": DECODE_NOTE,
                  "observation_decode_table": DECODE, "outputs": OUTPUT_NOTE, "gpu": GPU_NOTE,
                  "gpu_mem_used_mib_max_per_arm": GPU_PEAK, "warmup": {n: v[1] for n, v in WARMUP.items()},
                  "host": HOST_NOTE, "not_filled": UNFILLED_NOTE, "branch_after_run": BRANCH_NOTE,
                  "pr2": "G1/X3 ran one sample each; no manifest (see pr2-NOTE.md)"}
stats["manifests"] = OUT
stats["evidence"] = EVID
with open(os.path.join(HERE, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2); f.write("\n")
for arm, a in stats["arms"].items():
    for name, r in a["requests"].items():
        print("%s %-15s ttft mean %8.3f sd %6.3f | tok/s median %7.4f mean %7.4f sd %6.4f | identical_to_A=%s" % (
            arm, name, r["ttft_s"]["mean"], r["ttft_s"]["stdev"], r["prefill_tok_s"]["median"], r["prefill_tok_s"]["mean"],
            r["prefill_tok_s"]["stdev"], r["identical_to_A"]))
    if a["gate_problems"]:
        print("  GATE PROBLEMS %s: %s" % (arm, a["gate_problems"]))
