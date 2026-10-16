#!/usr/bin/env python3
"""Build the PR #1 experiment manifests (upstream docs/experiments/manifest.example.json format,
validated by c/experiment_manifest.py) from the final run on shadowfax (run_pr1.sh).

usage: make_manifests_pr1.py RAW_DIR
  RAW_DIR = the extracted pr1.tgz (the directory that holds A/ B/ C/ E/ F/ V/ meta/ run.log).
Also reads (inputs, not rewritten):
  hardware.json                      hardware text; its MemAvailable, page-cache and PCIe wording is checked below
  evidence/pr1-host-journal.txt      shadowfax journal / Hermes cron extract of the run window
                                     (capture_host_journal_pr1.sh): serve-unit memory peaks, watchdog, cron
  ../../evidence/pr1bench-raw.tgz    the 2026-10-14 run (dev vs pr1 48767c01), for the reproduction check
  ../../pr1-*.json                   the 2026-10-14 manifests (their engine commits)
Writes, next to this script:
  evidence/pr1-arm-<X>.tgz   deterministic per-arm tarball of RAW_DIR/<X>/ (what a manifest's evidence names)
  pr1-*.json                 one manifest per (changed variable, request), each with a notes block
  stats.json                 per-sample and per-arm numbers (mean, stdev, CV, median), the correctness gate and the notes
"""
import datetime, decimal, glob, gzip, hashlib, io, json, os, re, statistics, sys, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(sys.argv[1])
DEV = "7b55a9b3792527f52d1a2337fd9e312b9a7983e3"
PR1 = "93b741839677ec457868fd71ab668cc1249b9a22"
BIN = {DEV: "beb037276ed54cd4c54a5e043bf876a38c4de900de661d8b30b75c314b36d127",
       PR1: "a0a65edd3a5041a040bdd6a8abed1fedff1a2226d303a58a177bc87d8fce7c5f"}
ARM_TREE = {"A": DEV, "B": PR1, "C": PR1, "E": PR1, "F": PR1, "V": PR1}
ARM_EXTRA = {"A": {}, "B": {}, "C": {"COLI_CUDA_MOE_DOUBLE": "1"}, "E": {"DSV4_MOE_EXPERT_MAJOR": "0"},
             "F": {"DSV4_MM_WARP": "0"}, "V": {"DSV4_MOE_EXPERT_MAJOR_VERIFY": "1", "DSV4_MM_WARP_VERIFY": "1"}}
REQS = {"long_prefill": 1992, "long_prefill_2": 3005}

# the exact prompts of bench_pr1.py; prompt_hash = sha256(json.dumps(PROMPTS)), as in the earlier manifests
filler = " ".join(["The quick brown fox jumps over the lazy dog while the committee reviews section %d of the quarterly infrastructure report." % i for i in range(90)])
filler2 = " ".join(["Station %d logged a temperature of %d degrees and a wind speed of %d knots during the overnight survey." % (i, 10 + i % 17, 3 + i % 11) for i in range(130)])
PROMPTS = [filler + "\n\nSummarize the above in one sentence.", filler2 + "\n\nWhich station had the highest wind speed? Answer briefly."]
PROMPT_HASH = hashlib.sha256(json.dumps(PROMPTS).encode()).hexdigest()
assert PROMPT_HASH == "13fc61ddae676a87f55d68c32d7488f90f0c99af634f30dc846eaa8d1707257f", PROMPT_HASH

HW = json.load(open(os.path.join(HERE, "hardware.json")))
RUNLOG = open(os.path.join(RAW, "run.log")).read()
MEMAVAIL_MIB = int(re.search(r"preflight ok: MemAvailable=(\d+) MiB", RUNLOG).group(1))
# hardware.json must carry the preflight figure in GiB (1024^3) with the MiB value next to it
assert "MemAvailable %.1f GiB (%d MiB)" % (MEMAVAIL_MIB / 1024, MEMAVAIL_MIB) in HW["ram"], (MEMAVAIL_MIB, HW["ram"])
LLAMA_DOWN = re.search(r"\[(\S+ \S+)\] stopping llama-server-active", RUNLOG).group(1).replace(" ", "T")
LLAMA_UP = re.search(r"restore: llama health \S+ \(ok=1\) at (\S+?);", RUNLOG).group(1)[:19]
# the cards' PCIe capability vs the link they ran at (meta/host.txt: nvidia-smi --query-gpu ... pcie.link.*)
PCIE = [tuple(map(int, m.groups())) for m in re.finditer(
    r"^\d, NVIDIA GeForce RTX 3090, GPU-[\w-]+, [\d.]+, \d+ MiB, (\d+), (\d+), \d+, (\d+),",
    open(os.path.join(RAW, "meta", "host.txt")).read(), re.M)]
assert len(PCIE) == 2 and all(p == PCIE[0] for p in PCIE), PCIE
assert "PCIe %d.0 x%d-capable cards running at link width x%d" % PCIE[0] in HW["os"], (PCIE, HW["os"])
MODEL = ("DeepSeek V4 Flash REAP 150B (puwaer/DeepSeek-V4-Flash-0731-reap-150b, 132 routed experts, fp4 experts / "
         "fp8 dense), colibri safetensors layout, 79 GiB on disk")
COMMON_ENV = {"COLI_CUDA": "1", "COLI_GPU": "0", "DSV4_CUDA_DEVICE": "0", "CUDA_DENSE": "1", "COLI_CUDA_PIPE": "2",
              "COLI_CUDA_ATTN_BATCH": "1", "COLI_CUDA_MOE_BATCH": "1", "V4_MOE_REFILL_GROUP": "12",
              "V4_LOADER_LANES": "3", "V4_PREFIX_CKPT_DISK": "0", "ram_gib": "40", "ctx": "8192", "temperature": "0"}
BUILD = "cd c && make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j8"
RUN = ("python3 c/coli serve --model <REAP dir> --ram 40 --ctx 8192 --host 127.0.0.1 --port 8090, server environment "
       "exactly PATH/HOME/USER/LOGNAME/LANG + the config's env keys (env -i); one freshly started server process per "
       "sample (OS page cache not dropped between samples, see notes.cache_state), "
       "3 samples per arm interleaved with the arm order rotated per round (A B C E F / B C E F A / C E F A B); "
       "client bench_pr1.py sends request 1 (long_prefill: 1992-token prompt, max_tokens 100) and, for arms A/B/C, "
       "then request 2 (long_prefill_2: 3005-token prompt, max_tokens 60), streamed, greedy; "
       "tok_s = prompt_tokens / time-to-first-token of the named request")


def kv(path):
    out = {}
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                out[k] = v
    return out


def load(arm):
    out = []
    for d in sorted(glob.glob(os.path.join(RAW, arm, "s*")), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)))):
        meta = kv(os.path.join(d, "sample.txt"))
        env = kv(os.path.join(d, "engine_environ.txt"))
        rows = [json.loads(l) for l in open(os.path.join(d, "bench.jsonl")) if l.strip().startswith("{")] \
            if os.path.exists(os.path.join(d, "bench.jsonl")) else []
        for r in rows:  # recompute the hash from the stored full text
            assert hashlib.sha256(r["text"].encode()).hexdigest() == r["text_sha256"], (d, r["name"])
        out.append({"dir": os.path.relpath(d, RAW), "meta": meta, "env": env, "rows": {r["name"]: r for r in rows},
                    "gpu": nvsmi(os.path.join(d, "nvsmi.csv"))})
    return out


def nvsmi(path):
    """peak memory.used per GPU and the PCIe links seen in the 1 s nvidia-smi log of one sample"""
    peak, links = {}, {}
    for line in open(path):
        p = [x.strip() for x in line.split(",")]
        if len(p) < 11 or p[0].startswith("timestamp"):
            continue
        i = p[1]
        peak[i] = max(peak.get(i, 0), int(p[2].split()[0]))
        links.setdefault(i, set()).add("gen%s x%s" % (p[9], p[10]))
    return {"mem_used_mib_max": peak, "links": {i: sorted(v) for i, v in links.items()}}


def env_ok(s, arm):
    want = {k: v for k, v in COMMON_ENV.items() if k.isupper()}
    want.update(ARM_EXTRA[arm])
    bad = [k for k, v in want.items() if s["env"].get(k) != v]
    switches = ("COLI_CUDA_MOE_DOUBLE", "DSV4_MOE_EXPERT_MAJOR", "DSV4_MM_WARP", "DSV4_MOE_EXPERT_MAJOR_VERIFY", "DSV4_MM_WARP_VERIFY")
    bad += ["%s set" % k for k in switches if k in s["env"] and k not in ARM_EXTRA[arm]]
    if s["meta"].get("tree_head") != ARM_TREE[arm] or s["meta"].get("binary_sha256") != BIN[ARM_TREE[arm]]:
        bad.append("tree/binary")
    if s["meta"].get("client_rc") != "0" or s["meta"].get("tree_dirty_lines") != "0":
        bad.append("client_rc/dirty")
    return bad


ARMS = {a: load(a) for a in ARM_TREE}
REF = {name: {s["rows"][name]["text_sha256"] for s in ARMS["A"] if name in s["rows"]} for name in REQS}
assert all(len(v) == 1 for v in REF.values()), REF
REF = {k: next(iter(v)) for k, v in REF.items()}


def gate(arm):
    """every request of every sample of this arm: full text sha256 == arm A's, env/tree/binary checked"""
    problems = []
    for s in ARMS[arm]:
        problems += ["%s: %s" % (s["dir"], p) for p in env_ok(s, arm)]
        for name, r in s["rows"].items():
            if r["text_sha256"] != REF[name]:
                problems.append("%s %s sha %s != A %s" % (s["dir"], name, r["text_sha256"][:12], REF[name][:12]))
    return problems


def evidence(arm):
    buf = io.BytesIO()
    # gzip header mtime 0 and no file name: the tarball is byte-identical on every run
    with gzip.GzipFile(filename="", fileobj=buf, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as t:
        for f in sorted(glob.glob(os.path.join(RAW, arm, "s*", "*"))):
            info = t.gettarinfo(f, arcname=os.path.join("pr1", os.path.relpath(f, RAW)))
            info.mtime = 0; info.uid = info.gid = 0; info.uname = info.gname = ""
            with open(f, "rb") as fh:
                t.addfile(info, fh)
    data = buf.getvalue()
    os.makedirs(os.path.join(HERE, "evidence"), exist_ok=True)
    path = os.path.join(HERE, "evidence", "pr1-arm-%s.tgz" % arm)
    open(path, "wb").write(data)
    return {"uri": os.path.relpath(path, HERE), "sha256": hashlib.sha256(data).hexdigest(),
            "contains": "pr1/%s/s*/ (serve.log, bench.jsonl with full texts, sample.txt, engine_environ.txt, procs.txt, nvsmi.csv, gpu_*.csv, mode.txt)" % arm,
            "full_run": "pr1.tgz (sha256 in pr1.tgz.sha256): all arms, meta/ (host, trees, scripts, llama before/after), run.log, summary.json"}


EVID = {}


def stat(xs):
    return {"mean": round(statistics.mean(xs), 4), "stdev": round(statistics.stdev(xs), 4) if len(xs) > 1 else None,
            "cv_pct": round(100 * statistics.stdev(xs) / statistics.mean(xs), 2) if len(xs) > 1 else None,
            "median": round(statistics.median(xs), 4), "min": min(xs), "max": max(xs), "n": len(xs)}


def dmean(xs, places=4):
    """mean of values recorded with <= 4 decimals, rounded half-up on the exact decimal mean (2.24645 -> 2.2465)"""
    m = sum(decimal.Decimal(str(x)) for x in xs) / len(xs)
    return float(m.quantize(decimal.Decimal(1).scaleb(-places), rounding=decimal.ROUND_HALF_UP))


def mean_of(arm, request, key, seqs=None):
    """mean of one per-request field over an arm's samples (seqs: restrict to these sample numbers, e.g. (2, 3))"""
    xs = [s["rows"][request][key] for s in ARMS[arm] if request in s["rows"] and
          (seqs is None or int(s["meta"]["sample"]) in seqs)]
    return dmean(xs)


def pct(a, b):
    return 100.0 * (a / b - 1)


# ---------------------------------------------------------------- host side of the window (evidence/pr1-host-journal.txt)
JOURNAL = os.path.join(HERE, "evidence", "pr1-host-journal.txt")
JLINES = open(JOURNAL).read().splitlines()


def in_window(ts):
    return LLAMA_DOWN <= ts[:19] <= LLAMA_UP


MEMPEAK, cur = {}, None  # sample dir -> systemd "memory peak" of the serve unit (cgroup; counts page cache it faulted in)
for l in JLINES:
    m = re.search(r"Started colibri-final-pr1-serve\.service .*/final/pr1/(\w+/s\d+)/serve\.log", l)
    if m:
        cur = m.group(1)
        continue
    m = re.search(r"colibri-final-pr1-serve\.service: Consumed .*?, ([\d.]+)G memory peak", l)
    if m and cur:
        MEMPEAK[cur] = float(m.group(1)); cur = None
ALL_SAMPLES = sorted(((int(s["meta"]["seq"]), s["dir"]) for ss in ARMS.values() for s in ss))
assert sorted(MEMPEAK) == sorted(d for _, d in ALL_SAMPLES), (sorted(MEMPEAK), ALL_SAMPLES)
ROUND1 = [d for q, d in ALL_SAMPLES if q <= 5]  # A B C E F s1: the first round of the rotation
LATER = [d for q, d in ALL_SAMPLES if q > 5]
LATER_PEAKS = sorted({MEMPEAK[d] for d in LATER})
assert len(LATER_PEAKS) == 1 and all(MEMPEAK[d] > LATER_PEAKS[0] for d in ROUND1), (MEMPEAK, LATER_PEAKS)
assert "%.1fG-%.1fG vs %.1fG in all %d later samples" % (min(MEMPEAK[d] for d in ROUND1), max(MEMPEAK[d] for d in ROUND1),
                                                          LATER_PEAKS[0], len(LATER)) in HW["storage"], HW["storage"]

WATCHDOG = []  # shadowfax-ensure-active-stack runs started while llama.cpp was down
for l in JLINES:
    m = re.match(r"(\S+) \S+ systemd\[\d+\]: Starting shadowfax-ensure-active-stack\.service", l)
    if m:
        WATCHDOG.append({"start": m.group(1)[:19], "_json": []})
        continue
    if not WATCHDOG:
        continue
    m = re.match(r"(\S+) \S+ systemd\[\d+\]: (Failed to start|Finished) shadowfax-ensure-active-stack\.service", l)
    if m:
        WATCHDOG[-1].update(end=m.group(1)[:19], result="failed" if m.group(2) == "Failed to start" else "ok")
        continue
    m = re.match(r"\S+ \S+ env\[\d+\]: (.*)$", l)
    if m:
        WATCHDOG[-1]["_json"].append(m.group(1))
for w in WATCHDOG:
    js = w.pop("_json")
    d = json.loads("\n".join(js)) if js else None
    w.update(logged=bool(d), backend=d["statuses"]["backend"] if d else None,
             probe_error=d["exact_output_probe"].get("error") if d else None,
             restarted_units=d["restarted_units"] if d else None, recovered=d["recovered"] if d else None)
WD_IN = [w for w in WATCHDOG if in_window(w["start"])]
WD_AFTER = [w for w in WATCHDOG if w["start"][:19] > LLAMA_UP]
assert WD_IN and all(w["result"] == "failed" for w in WD_IN), WD_IN
assert all(not w["restarted_units"] and w["recovered"] is False and w["backend"] is False
           for w in WD_IN if w["logged"]), WD_IN  # nothing restarted, backend reported unhealthy
WD_GAPS = sorted({round((datetime.datetime.fromisoformat(b["start"]) - datetime.datetime.fromisoformat(a["start"])).total_seconds())
                  for a, b in zip(WD_IN, WD_IN[1:])})

CRON = [json.loads(l.split(" ", 1)[1]) for l in JLINES if l.startswith("cron_execution ")]
CRON_IN = [c for c in CRON if in_window(c["claimed_at"])]
SCOPE_CPU = {m.group(1): m.group(2) for m in (re.search(r"hermes-worker-cron-\w+-exec-(\w+)\.scope: Consumed ([\d.]+s) CPU time", l)
                                              for l in JLINES) if m}
SAMPLE_SPAN = {s["dir"]: (s["meta"]["start"][:19], s["meta"]["end"][:19]) for ss in ARMS.values() for s in ss}
for c in CRON_IN:
    c["cpu_time"] = SCOPE_CPU.get(c["id"])
    c["overlapping_samples"] = [d for _, d in ALL_SAMPLES
                                if SAMPLE_SPAN[d][0] < c["finished_at"][:19] and c["claimed_at"][:19] < SAMPLE_SPAN[d][1]]


def hm(ts):
    return ts[11:19]


HOST_NOTE = (
    "Host side effect of the window (llama.cpp down %s-%s PDT; evidence/pr1-host-journal.txt): the Hermes "
    "shadowfax-ensure-active-stack watchdog ran every 5 min (start to start %d-%d s; %d runs, %s-%s), reported the "
    "backend unhealthy each time (all %d exited 1; %d logged 'backend unhealthy' with restarted_units [] and recovered "
    "false, %d logged no output) and restarted nothing; the first run after the restore (%s) passed. Hermes cron ran "
    "without an LLM backend: %s." % (
        hm(LLAMA_DOWN), hm(LLAMA_UP), WD_GAPS[0], WD_GAPS[-1], len(WD_IN), hm(WD_IN[0]["start"]), hm(WD_IN[-1]["end"]),
        len(WD_IN), sum(w["logged"] for w in WD_IN), sum(not w["logged"] for w in WD_IN), hm(WD_AFTER[0]["start"]),
        "; ".join("%s (%s) %s-%s, %s ('%s'), %s CPU, during samples %s" % (
            c["job_name"], c["job_schedule"], hm(c["claimed_at"]), hm(c["finished_at"]), c["status"], c["error"],
            c["cpu_time"], " ".join(c["overlapping_samples"])) for c in CRON_IN) or "no cron execution in the window"))
assert WD_AFTER and WD_AFTER[0]["result"] == "ok"

# ---------------------------------------------------------------- cache state (round 1 partly cold)
S1_DECODE_LOWEST = {arm: min(ARMS[arm], key=lambda s: s["rows"]["long_prefill"]["decode_tok_s"])["meta"]["sample"] == "1"
                    for arm in "ABCEF"}
A1 = [s["rows"]["long_prefill"]["ttft_s"] for s in ARMS["A"]]
CACHE_NOTE = (
    "The OS page cache was not dropped between samples. Round 1 (samples 1-5: %s) was partly cold: the serve unit's "
    "systemd memory peak (its cgroup, which counts the page cache the unit faulted in) was %s, against %.1fG in all %d "
    "later samples (evidence/pr1-host-journal.txt); from round 2 on the page cache was warm. Round 1 holds the lowest "
    "request-1 decode rate of %s and the slowest A request-1 TTFT (A s1 %.1f s vs %s s). Medians are over all 3 samples; "
    "the round-1-excluded ratios (s2/s3 means) are in notes.round1_excluded." % (
        " ".join(ROUND1), ", ".join("%.1fG %s" % (MEMPEAK[d], d) for d in ROUND1), LATER_PEAKS[0], len(LATER),
        "every arm" if all(S1_DECODE_LOWEST.values()) else "arms " + " ".join(a for a, v in S1_DECODE_LOWEST.items() if v),
        A1[0], "/".join("%.1f" % x for x in A1[1:])))
assert A1[0] == max(A1)

# ---------------------------------------------------------------- variability
def spread(xs):  # unrounded, so the text rounds once (stat() rounds to 4 places first)
    return {"stdev": statistics.stdev(xs), "cv_pct": 100 * statistics.stdev(xs) / statistics.mean(xs)}


TT = {(arm, n): spread([s["rows"][n]["ttft_s"] for s in ARMS[arm] if n in s["rows"]])
      for arm in "ABCEF" for n in REQS if any(n in s["rows"] for s in ARMS[arm])}
EXC = ("A", "long_prefill")
CV_REST = max(v["cv_pct"] for k, v in TT.items() if k != EXC)
SD_SORTED = sorted(TT.items(), key=lambda kv: -kv[1]["stdev"])
assert SD_SORTED[0][0] == EXC and SD_SORTED[1][0] == ("A", "long_prefill_2"), SD_SORTED[:2]
VAR_NOTE = (
    "TTFT coefficient of variation (stdev/mean, 3 samples) is <= %.2f%% for every arm and request except A request 1 "
    "(%.2f%%, from the partly cold A s1); TTFT stdev is <= %.3f s except A request 1 (%.3f s) and A request 2 (%.3f s). "
    "Per arm and request (ttft stdev s / CV %%): %s." % (
        CV_REST, TT[EXC]["cv_pct"], SD_SORTED[2][1]["stdev"], TT[EXC]["stdev"], TT[("A", "long_prefill_2")]["stdev"],
        ", ".join("%s r%d %.3f/%.2f" % (a, 1 if n == "long_prefill" else 2, v["stdev"], v["cv_pct"]) for (a, n), v in TT.items())))

# ---------------------------------------------------------------- the 2026-10-14 run (manifests/pr1-*.json, evidence/pr1bench-raw.tgz)
EARLIER_TGZ = os.path.normpath(os.path.join(HERE, "..", "..", "evidence", "pr1bench-raw.tgz"))
EARLIER = {}
with tarfile.open(EARLIER_TGZ) as t:
    for mem in t.getmembers():
        m = re.match(r"pr1bench/(\w)/s(\d)/bench\.jsonl$", mem.name)
        if m:
            for l in t.extractfile(mem).read().decode().splitlines():
                if l.strip().startswith("{"):
                    r = json.loads(l)
                    EARLIER.setdefault(m.group(1), {}).setdefault(r["name"], {})[int(m.group(2))] = r
    EARLIER_SCRIPT = t.extractfile("run_pr1_bench.sh").read().decode()
EARLIER_SHA = hashlib.sha256(open(EARLIER_TGZ, "rb").read()).hexdigest()
EM = {f: json.load(open(os.path.normpath(os.path.join(HERE, "..", "..", f))))
      for f in ("pr1-engine-1992.json", "pr1-expert-major-1992.json", "pr1-mm-warp-1992.json")}
EARLIER_COMMIT = {"A": EM["pr1-engine-1992.json"]["baseline"]["config"]["engine_commit"],
                  "B": EM["pr1-engine-1992.json"]["trial"]["config"]["engine_commit"],
                  "E": EM["pr1-expert-major-1992.json"]["baseline"]["config"]["engine_commit"],
                  "F": EM["pr1-mm-warp-1992.json"]["baseline"]["config"]["engine_commit"]}
m = re.search(r"B = pr1/fixed (\w+), C = B \+", EARLIER_SCRIPT)  # C ran on B's tree (no 10-14 manifest for C)
assert m and EARLIER_COMMIT["B"].startswith(m.group(1)), (m and m.group(1), EARLIER_COMMIT)
EARLIER_COMMIT["C"] = EARLIER_COMMIT["B"]
assert EARLIER_COMMIT["A"] == DEV and all(EARLIER_COMMIT[a] != PR1 for a in "BCEF"), EARLIER_COMMIT
REPRO = {}
for arm in "ABCEF":
    for n in REQS:
        rs = [s["rows"][n] for s in ARMS[arm] if n in s["rows"]]
        if rs:
            old = statistics.median(r["prefill_tok_s"] for r in EARLIER[arm][n].values())
            new = statistics.median(r["prefill_tok_s"] for r in rs)
            REPRO[(arm, n)] = {"median_tok_s_2026_10_14": old, "median_tok_s_final": new, "delta_pct": round(pct(new, old), 2),
                               "commit_2026_10_14": EARLIER_COMMIT[arm], "commit_final": ARM_TREE[arm],
                               "same_texts_2026_10_14": sorted({r["text_sha256"] for r in EARLIER[arm][n].values()}) == [REF[n]]}
DELTAS = [v["delta_pct"] for v in REPRO.values()]
LOWEST = min(REPRO.items(), key=lambda kv: kv[1]["delta_pct"])
# the verify arm V (1 sample, request 1, both VERIFY modes) ran on both days too, on B's tree (pr1 48767c01 on 10-14);
# it is kept out of DELTAS (not a manifest arm) but stated, so "every arm" does not overstate
assert re.search(r"^\s+sample B \$k \$PR1 both$", EARLIER_SCRIPT, re.M) and re.search(
    r"^sample V 1 \$PR1 first -E DSV4_MOE_EXPERT_MAJOR_VERIFY=1 -E DSV4_MM_WARP_VERIFY=1$", EARLIER_SCRIPT, re.M)
_vo = statistics.median(r["prefill_tok_s"] for r in EARLIER["V"]["long_prefill"].values())
_vn = statistics.median(s["rows"]["long_prefill"]["prefill_tok_s"] for s in ARMS["V"])
REPRO_V = {"median_tok_s_2026_10_14": _vo, "median_tok_s_final": _vn, "delta_pct": round(pct(_vn, _vo), 2), "n": len(ARMS["V"]),
           "commit_2026_10_14": EARLIER_COMMIT["B"], "commit_final": ARM_TREE["V"],
           "same_texts_2026_10_14": sorted({r["text_sha256"] for r in EARLIER["V"]["long_prefill"].values()}) == [REF["long_prefill"]]}
V_REPRO_TEXT = "the 1-sample verify arm V, request 1: %.4f -> %.4f (%+.2f%%)" % (_vo, _vn, REPRO_V["delta_pct"])
REPRO_NOTE = (
    "Against the 2026-10-14 run (manifests/pr1-*.json, evidence/pr1bench-raw.tgz): every A/B/C/E/F arm/request median "
    "tok/s is %s, by %.2f%% to %.2f%% (largest: %s request %d, %.2f%%); %s. That run's A was the same dev commit; its "
    "B, C, E, F, V were pr1 %s, an earlier commit than this tip %s, so for B..F and V this is a comparison across commits, "
    "not a strict repeat. Per arm/request: %s." % (
        "slightly lower" if max(DELTAS) < 0 else "within", min(DELTAS), max(DELTAS), LOWEST[0][0],
        1 if LOWEST[0][1] == "long_prefill" else 2, LOWEST[1]["delta_pct"], V_REPRO_TEXT, EARLIER_COMMIT["B"][:8], PR1[:8],
        ", ".join("%s r%d %.4f -> %.4f (%+.2f%%)" % (a, 1 if n == "long_prefill" else 2, v["median_tok_s_2026_10_14"],
                                                      v["median_tok_s_final"], v["delta_pct"]) for (a, n), v in REPRO.items())))

# ---------------------------------------------------------------- decode after prefill: pr1 tip vs dev (observation)
def dec(src, arm, n, seqs):
    if src == "final":
        return mean_of(arm, n, "decode_tok_s", seqs)
    return dmean([r["decode_tok_s"] for k, r in EARLIER[arm][n].items() if seqs is None or k in seqs])


DECODE = {}
for src in ("final", "2026_10_14"):
    for n in REQS:
        a23, b23 = dec(src, "A", n, (2, 3)), dec(src, "B", n, (2, 3))
        a, b = dec(src, "A", n, None), dec(src, "B", n, None)
        DECODE["%s %s" % (src, n)] = {"A_s2s3": round(a23, 4), "B_s2s3": round(b23, 4), "B_vs_A_s2s3_pct": round(pct(b23, a23), 2),
                                      "A_all": round(a, 4), "B_all": round(b, 4), "B_vs_A_all_pct": round(pct(b, a), 2)}
assert all(v["B_vs_A_s2s3_pct"] < 0 and v["B_vs_A_all_pct"] < 0 for v in DECODE.values()), DECODE
F1, F2 = DECODE["final long_prefill"], DECODE["final long_prefill_2"]
E1, E2 = DECODE["2026_10_14 long_prefill"], DECODE["2026_10_14 long_prefill_2"]
CHUNKS = {n: sorted({s["rows"][n]["decode_chunks"] for ss in ARMS.values() for s in ss if n in s["rows"]}) for n in REQS}
DECODE_NOTE = (
    "Observation (not this manifest's hypothesis; already documented in the PR docs, docs/deepseek-v4.md 'Expert-pruned "
    "checkpoints (REAP)', review finding F42): decode right after the GPU-batch prefill is slower on the pr1 tip than "
    "after dev's CPU-union prefill. Request 1: B %.4f vs A %.4f chunks/s (s2/s3 means, %.1f%%; all 3 samples %.1f%%); "
    "request 2: B %.3f vs A %.3f (%.1f%%; all 3 samples %.1f%%). Seen in both runs: on 2026-10-14 (pr1 %s) %.1f%% and "
    "%.1f%% (s2/s3 means). Decode rate = bench_pr1.py decode_tok_s, streamed chunks per second after the first chunk "
    "(%s chunks on request 1, %s on request 2); the texts are byte-identical, so this is a speed difference only; "
    "the cause is not established." % (
        F1["B_s2s3"], F1["A_s2s3"], F1["B_vs_A_s2s3_pct"], F1["B_vs_A_all_pct"], F2["B_s2s3"], F2["A_s2s3"],
        F2["B_vs_A_s2s3_pct"], F2["B_vs_A_all_pct"], EARLIER_COMMIT["B"][:8], E1["B_vs_A_s2s3_pct"], E2["B_vs_A_s2s3_pct"],
        "/".join(map(str, CHUNKS["long_prefill"])), "/".join(map(str, CHUNKS["long_prefill_2"]))))

# ---------------------------------------------------------------- outputs and GPU memory
N_OUT = {arm: sum(len(s["rows"]) for s in ss) for arm, ss in ARMS.items()}
ALL_IDENTICAL = all(r["text_sha256"] == REF[n] for ss in ARMS.values() for s in ss for n, r in s["rows"].items())
OUTPUT_NOTE = (
    "%d outputs in %d samples (%s), %s byte-identical (full-text sha256) to arm A's per-request reference; %d of them "
    "come from the pr1-tip arms %s." % (
        sum(N_OUT.values()), len(ALL_SAMPLES), ", ".join("%s %d" % kv for kv in N_OUT.items()),
        "all" if ALL_IDENTICAL else "NOT all", sum(v for a, v in N_OUT.items() if a != "A"), " ".join(a for a in ARMS if a != "A")))
GPU_PEAK = {arm: {i: max(s["gpu"]["mem_used_mib_max"][i] for s in ss) for i in ("0", "1")} for arm, ss in ARMS.items()}
GPU_LINKS = sorted({l for ss in ARMS.values() for s in ss for l in s["gpu"]["links"]["0"]})
GPU_NOTE = ("Peak memory.used (nvsmi.csv, 1 s samples, max over the arm's samples): GPU0 %s MiB; GPU1 %s MiB (unused). "
            "GPU0 link seen during samples: %s; the cards are PCIe %d.0 x%d-capable and ran at x%d (meta/host.txt)." % (
                ", ".join("%s %d" % (a, v["0"]) for a, v in GPU_PEAK.items()),
                "/".join(sorted({str(v["1"]) for v in GPU_PEAK.values()})), ", ".join(GPU_LINKS), *PCIE[0]))

WARMUP = {"long_prefill": (0, "warmup_runs 0: request 1 (long_prefill) is the first request the client sends to each "
                              "freshly started server process; no warm-up request precedes it"),
          "long_prefill_2": (1, "warmup_runs 1: request 2 (long_prefill_2) is served by the same server process right "
                                "after request 1 (long_prefill, the 1992-token prompt, max_tokens 100), which is its one "
                                "warm-up here; request 1 is itself measured, in the matching -1992 manifest")}


ROUND1_EXCLUDED = {}


def round1_excluded(name, base_arm, trial_arm, request):
    b = mean_of(base_arm, request, "prefill_tok_s", (2, 3)); t = mean_of(trial_arm, request, "prefill_tok_s", (2, 3))
    ROUND1_EXCLUDED[name] = "s2/s3 mean tok/s (round 1 excluded): %s %.4f, %s %.4f, ratio %.4f" % (base_arm, b, trial_arm, t, t / b)
    return ROUND1_EXCLUDED[name]



def run(arm, cfg, request):
    ss = [s for s in ARMS[arm] if request in s["rows"]]
    rows = [s["rows"][request] for s in ss]
    speeds = [r["prefill_tok_s"] for r in rows]
    probs = gate(arm)
    if arm not in EVID:
        EVID[arm] = evidence(arm)
    return {
        "config": cfg,
        "samples": {"tok_s": speeds, "ttft_s": [r["ttft_s"] for r in rows], "decode_tok_s": [r["decode_tok_s"] for r in rows],
                    "text_sha256": [r["text_sha256"] for r in rows], "sample_dirs": [s["dir"] for s in ss],
                    "run_order_seq": [int(s["meta"]["seq"]) for s in ss],
                    "tok_s_mean": round(statistics.mean(speeds), 4), "tok_s_stdev": round(statistics.stdev(speeds), 4),
                    "ttft_s_mean": round(statistics.mean([r["ttft_s"] for r in rows]), 3),
                    "ttft_s_stdev": round(statistics.stdev([r["ttft_s"] for r in rows]), 3)},
        "median_tok_s": statistics.median(speeds),
        "engine": {"commit": ARM_TREE[arm], "binary_sha256": BIN[ARM_TREE[arm]]},
        "quality": {"method": ("greedy (temperature 0): the full output text of every request (long_prefill and, where run, "
                               "long_prefill_2) in every sample of this arm has the same sha256 as upstream dev (arm A, "
                               "itself identical across its 3 samples); the engine's real environment (/proc/<pid>/environ), "
                               "tree SHA and binary sha256 checked per sample"),
                    "passed": not probs, "problems": probs},
        "evidence": EVID[arm],
    }


OUT = []


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
        "warmup": WARMUP[request][1],
        "cache_state": CACHE_NOTE,
        "round1_excluded": round1_excluded(name, base_arm, trial_arm, request),
        "variability": "TTFT stdev / CV over 3 samples: %s; %s" % (
            "; ".join("%s r%d %.3f s / %.2f%%" % (a, rn, TT[(a, request)]["stdev"], TT[(a, request)]["cv_pct"])
                      for a in (base_arm, trial_arm)), VAR_NOTE.split(" Per arm and request")[0]),
        "earlier_run_2026_10_14": "; ".join("%s median %.4f tok/s on 2026-10-14 (%s) -> %.4f here (%+.2f%%)" % (
            a, REPRO[(a, request)]["median_tok_s_2026_10_14"], REPRO[(a, request)]["commit_2026_10_14"][:8],
            REPRO[(a, request)]["median_tok_s_final"], REPRO[(a, request)]["delta_pct"]) for a in (base_arm, trial_arm)) +
            "; all A/B/C/E/F arm/request medians of this run are %.2f%% to %.2f%% vs 2026-10-14, all lower; %s; B..F "
            "and V ran an earlier pr1 commit then" % (min(DELTAS), max(DELTAS), V_REPRO_TEXT),
        "outputs": OUTPUT_NOTE,
        "host": HOST_NOTE,
    }
    if changed == ENG:
        rec["notes"]["observation_decode"] = DECODE_NOTE
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(rec, f, indent=2); f.write("\n")
    OUT.append(name)
    print("%-28s %-11s %8.3f -> %8.3f tok/s (%.3fx) quality %s/%s" % (
        name, rec["outcome"], b["median_tok_s"], t["median_tok_s"], ratio, b["quality"]["passed"], t["quality"]["passed"]))


ENG = "engine_commit"
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
manifest("pr1-expert-major-1992.json",
         "The expert-major fp4 MoE kernel (default on) is faster than the per-route kernel in the batched prefill MoE",
         "DSV4_MOE_EXPERT_MAJOR", "E", "0", "B", "unset (default 1)", "long_prefill", fixed={ENG: PR1})
manifest("pr1-mm-warp-1992.json",
         "The warp-per-row dense batch matmul (auto: I<=1024, i.e. attn.wq_b) is faster than mm_batch in batched prefill",
         "DSV4_MM_WARP", "F", "0", "B", "unset (auto)", "long_prefill", fixed={ENG: PR1})

# per-arm statistics and the gate, for the report
stats = {"reference_A_sha256": REF, "arms": {}}
for arm, ss in ARMS.items():
    a = {"samples": [], "requests": {}, "gate_problems": gate(arm)}
    for s in ss:
        a["samples"].append({"dir": s["dir"], "seq": int(s["meta"]["seq"]), "start": s["meta"].get("start"),
                             "end": s["meta"].get("end"),
                             "wall_s": int(s["meta"].get("wall_s", 0)), "ready_s": int(s["meta"].get("ready_s", 0)),
                             "serve_unit_memory_peak_gib": MEMPEAK[s["dir"]],
                             "gpu_mem_used_mib_max": s["gpu"]["mem_used_mib_max"], "gpu_links": s["gpu"]["links"],
                             "requests": {n: {k: r[k] for k in ("ttft_s", "prefill_tok_s", "decode_tok_s", "decode_chunks", "total_s", "text_sha256")}
                                          for n, r in s["rows"].items()}})
    for name in REQS:
        rs = [s["rows"][name] for s in ss if name in s["rows"]]
        if rs:
            a["requests"][name] = {"ttft_s": stat([r["ttft_s"] for r in rs]), "prefill_tok_s": stat([r["prefill_tok_s"] for r in rs]),
                                   "decode_tok_s": stat([r["decode_tok_s"] for r in rs]),
                                   "identical_to_A": all(r["text_sha256"] == REF[name] for r in rs)}
    stats["arms"][arm] = a
vlog = os.path.join(RAW, "V", "s1", "verify_counters.txt")
vl = open(vlog).read().splitlines() if os.path.exists(vlog) else []
mw = [tuple(map(int, m.groups())) for m in (re.search(r"mm-warp verify calls=(\d+) mismatches=(\d+)", l) for l in vl) if m]
em = [tuple(map(int, m.groups())) for m in (re.search(r"moe-expert-major verify calls=(\d+) fc1_mismatch=(\d+) fc2_mismatch=(\d+)", l) for l in vl) if m]
stats["verify_V"] = {"file": "V/s1/verify_counters.txt", "lines": vl,
                     "mm_warp_max_calls": max((c for c, _ in mw), default=0), "mm_warp_max_mismatches": max((x for _, x in mw), default=None),
                     "expert_major_max_calls": max((c for c, _, _ in em), default=0),
                     "expert_major_max_fc1_mismatch": max((a for _, a, _ in em), default=None),
                     "expert_major_max_fc2_mismatch": max((b for _, _, b in em), default=None),
                     "nonzero_mismatch_lines": [l for l in vl if re.search(r"(mismatches|fc1_mismatch|fc2_mismatch)=[1-9]", l)],
                     "note": "backend_cuda_dsv4.cu prints moe-expert-major every 100 calls and mm-warp every 500 calls, and either on any call with a mismatch (the cumulative mismatch count is in every line); max_calls is therefore a lower bound on the calls checked, and no line with a nonzero count means zero mismatches"}
stats["notes"] = {
    "cache_state": CACHE_NOTE,
    "serve_unit_memory_peak_gib": {d: MEMPEAK[d] for _, d in ALL_SAMPLES},
    "round1_excluded": ROUND1_EXCLUDED,
    "variability": VAR_NOTE,
    "earlier_run_2026_10_14": REPRO_NOTE,
    "earlier_run_2026_10_14_table": {"%s %s" % k: v for k, v in REPRO.items()},
    "earlier_run_2026_10_14_verify_arm_V": REPRO_V,
    "earlier_run_2026_10_14_evidence": {"tgz": "../../evidence/pr1bench-raw.tgz", "sha256": EARLIER_SHA,
                                        "manifests": sorted(EM)},
    "observation_decode": DECODE_NOTE,
    "observation_decode_table": DECODE,
    "outputs": OUTPUT_NOTE,
    "gpu_memory": GPU_NOTE,
    "gpu_mem_used_mib_max_per_arm": GPU_PEAK,
    "memavailable_preflight": "%d MiB = %.1f GiB (run.log preflight, /proc/meminfo MemAvailable, llama.cpp still loaded)" % (
        MEMAVAIL_MIB, MEMAVAIL_MIB / 1024),
    "warmup": {n: v[1] for n, v in WARMUP.items()},
    "host": HOST_NOTE,
    "host_watchdog_runs": WD_IN + WD_AFTER[:1],
    "host_cron_executions": CRON_IN,
    "host_evidence": {"file": "evidence/pr1-host-journal.txt", "sha256": hashlib.sha256(open(JOURNAL, "rb").read()).hexdigest(),
                      "captured_by": "capture_host_journal_pr1.sh (read-only journalctl / sqlite on shadowfax)"},
}
stats["manifests"] = OUT
stats["evidence"] = EVID
with open(os.path.join(HERE, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2); f.write("\n")
for arm, a in stats["arms"].items():
    for name, r in a["requests"].items():
        print("%s %-15s ttft mean %8.3f sd %6.3f | tok/s mean %7.3f sd %6.4f | identical_to_A=%s" % (
            arm, name, r["ttft_s"]["mean"], r["ttft_s"]["stdev"] or 0, r["prefill_tok_s"]["mean"], r["prefill_tok_s"]["stdev"] or 0, r["identical_to_A"]))
    if a["gate_problems"]:
        print("  GATE PROBLEMS %s: %s" % (arm, a["gate_problems"]))
