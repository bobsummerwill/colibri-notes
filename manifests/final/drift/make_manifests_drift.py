#!/usr/bin/env python3
"""Build the PR #0 drift (upload race) records from the final run on shadowfax (run_drift.sh).

usage: make_manifests_drift.py RAW_DIR
  RAW_DIR = the extracted drift.tgz (the directory that holds D/ Z/ Dp/ Zp/ meta/).
Reads, next to this script: hardware.json, and evidence/hermes-window.txt (what Hermes did while llama.cpp was
stopped; captured read-only on shadowfax by evidence/hermes_window.py, not part of drift.tgz).
Writes, next to this script:
  evidence/drift-arm-<X>.tgz   deterministic per-arm tarball of RAW_DIR/<X>/
  pr0-engine-decode.json       one manifest: engine_commit dev 7b55a9b3 -> pr0 a2c8ba7a, one-GPU decode
                               (bench_decode.py), 3 cold samples per arm, full-text identity gate
  stats.json                   per-prompt, per-sample full-text sha256 (distinct counts and first differing
                               character), decode tok/s, the phase-2 per-token logit/hidden-hash comparison
                               of the probe builds, GPU memory, and the per-sample checks
What the run tested: whether greedy decode on upstream dev gives different text across identical cold
runs (the decode-mirror upload race that pr0 b2c8ca61 fixes by queuing h2d() on the compute stream),
and whether pr0 is identical every time. The manifest records only what the evidence supports:
pr0's decode speed and output against dev on this configuration. It does NOT record a drift fix,
because dev did not drift in these runs, and the configuration lacked every documented trigger of the race
(long first request, smaller mirror budget, double bank, refills forced onto the pageable path).
Metric: per sample, decode tok/s = mean over the 6 prompts of bench_decode.py's per-prompt
decode_tok_s ((chunks - 1) / (time of the last chunk - time of the first chunk), client side).
"""
import ast, glob, gzip, hashlib, io, itertools, json, math, os, re, statistics, sys, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(sys.argv[1])
DEV = "7b55a9b3792527f52d1a2337fd9e312b9a7983e3"
PR0 = "a2c8ba7a149a31a781e84a992aa980afdb93497d"
BIN = {"D": "beb037276ed54cd4c54a5e043bf876a38c4de900de661d8b30b75c314b36d127",
       "Z": "d26bc5526a6c05b03e254a5259da0503d02f1083d682124a6587fff124621812",
       "Dp": "dea9bd9d0214cf090e196758fea665caa3f272da65b60aebb4c0d7af32092ba0",
       "Zp": "f8d5d9ed927c2af243467a225c7c179147c869c1612599d821e74224e840f19f"}
ARM_TREE = {"D": DEV, "Z": PR0, "Dp": DEV, "Zp": PR0}
PROBE_DIFF = {"Dp": "4568b67e85165e61e91b0875d441825f76b25ddd5f007c363bb2db931dabdf8b",
              "Zp": "bc1cec431a2cc79207fe0d263486bc17f6c25c461544523e20cce1c321d9e7f9"}
ARM_EXTRA = {"D": {}, "Z": {}, "Dp": {"V4_LOGIT_LOG": "1"}, "Zp": {"V4_LOGIT_LOG": "1"}}
SWITCHES = ("COLI_CUDA_MOE_DOUBLE", "V4_GPU_RESIDENT_FIRST", "DSV4_CUDA_PEER_DEVICE", "DSV4_CUDA_PEER_COMPUTE",
            "DSV4_CUDA_BATCHED", "DSV4_CUDA_EXPERT_MIRRORS", "DSV4_CUDA_VRAM_RESERVE_MB", "V4_SERVE_STATS_LOG",
            "V4_LOGIT_LOG", "DSV4_MOE_EXPERT_MAJOR", "DSV4_MM_WARP", "COLI_V4_SAVE_USAGE")

_src = open(os.path.join(RAW, "meta", "bench_decode.py")).read()
PROMPTS = next(ast.literal_eval(n.value) for n in ast.parse(_src).body
               if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "PROMPTS")
PROMPT_HASH = hashlib.sha256(json.dumps(PROMPTS).encode()).hexdigest()
assert PROMPT_HASH == "f9ed5d77f965bc53da65ebb1a85da283e871b6342c867f13bb1cdd4914249202", PROMPT_HASH
NAMES = ["decode_%d" % i for i in range(len(PROMPTS))]

HW = json.load(open(os.path.join(HERE, "hardware.json")))
MODEL = ("DeepSeek V4 Flash REAP 150B (puwaer/DeepSeek-V4-Flash-0731-reap-150b, 132 routed experts, fp4 experts / "
         "fp8 dense), colibri safetensors layout, 79 GiB on disk")
COMMON_ENV = {"COLI_CUDA": "1", "COLI_GPU": "0", "DSV4_CUDA_DEVICE": "0", "CUDA_DENSE": "1", "COLI_CUDA_PIPE": "2",
              "COLI_CUDA_ATTN_BATCH": "1", "COLI_CUDA_MOE_BATCH": "1", "V4_MOE_REFILL_GROUP": "12",
              "V4_LOADER_LANES": "3", "V4_PREFIX_CKPT_DISK": "0",
              "COLI_CUDA_MOE_DOUBLE": "unset (default 0)", "ram_gib": "40", "ctx": "8192",
              "temperature": "0 (greedy engine; request says 0.7, ignored)"}
BUILD = "cd c && make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j8"
RUN = ("python3 c/coli serve --model <REAP dir> --ram 40 --ctx 8192 --host 127.0.0.1 --port 8090, server environment "
       "exactly PATH/HOME/USER/LOGNAME/LANG + the config's env keys (env -i; COLI_CUDA_MOE_DOUBLE unset); one cold server "
       "process per sample, 3 samples per arm interleaved D Z / Z D / D Z (D = dev 7b55a9b3, Z = pr0 a2c8ba7a); "
       "client bench_decode.py sends 6 prompts one after another, max_tokens 300 each, streamed; "
       "tok_s = mean over the 6 prompts of (chunks - 1) / (last chunk time - first chunk time)")


def kv(path):
    out = {}
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                out[k] = v
    return out


def lines(path):
    return open(path, errors="replace").read().splitlines() if os.path.exists(path) else []


def num(s):
    m = re.match(r"\s*([\d.]+)", s)
    return float(m.group(1)) if m else None


def gpu_csv(path):
    out = {}
    for line in lines(path)[1:]:
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 2 and f[0].isdigit():
            out[int(f[0])] = int(num(f[1]))
    return out


def nvsmi_max(path):
    mem, util = {}, {}
    for line in lines(path)[1:]:
        f = [x.strip() for x in line.split(",")]
        if len(f) > 4 and f[1].isdigit():
            i = int(f[1])
            m, u = num(f[2]), num(f[4])
            if m is not None:
                mem[i] = max(mem.get(i, 0), int(m))
            if u is not None:
                util.setdefault(i, []).append(u)
    return {"max_mem_mib": mem, "mean_util_pct": {i: round(statistics.mean(v), 1) for i, v in util.items()},
            "n_samples": len(lines(path)) - 1}


def load(arm):
    out = []
    for d in sorted(glob.glob(os.path.join(RAW, arm, "s*")), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)))):
        meta = kv(os.path.join(d, "sample.txt"))
        env = kv(os.path.join(d, "engine_environ.txt"))
        rows = [json.loads(l) for l in open(os.path.join(d, "bench.jsonl")) if l.strip().startswith("{")] \
            if os.path.exists(os.path.join(d, "bench.jsonl")) else []
        for r in rows:  # the full text is in bench.jsonl; the hash is computed here from it
            r["sha256"] = hashlib.sha256(r["text"].encode()).hexdigest()
        log = lines(os.path.join(d, "serve.log"))
        out.append({"dir": os.path.relpath(d, RAW), "meta": meta, "env": env, "rows": {r["name"]: r for r in rows},
                    "order": [r["name"] for r in rows], "log": log,
                    "logits": [l.strip() for l in log if l.startswith("v4_logit ")],
                    "cpu_moe": [l.strip() for l in log if l.startswith("v4_cpu_moe ")],
                    "ready": gpu_csv(os.path.join(d, "gpu_ready.csv")), "before": gpu_csv(os.path.join(d, "gpu_before.csv")),
                    "end": gpu_csv(os.path.join(d, "gpu_end.csv")), "nvsmi": nvsmi_max(os.path.join(d, "nvsmi.csv"))})
    return out


ARMS = {a: load(a) for a in ("D", "Z", "Dp", "Zp")}
REF = {n: ARMS["D"][0]["rows"][n] for n in NAMES}     # reference: dev, phase 1, sample 1
TREES = open(os.path.join(RAW, "meta", "trees.txt")).read()


def first_diff(a, b):
    return next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None if len(a) == len(b) else min(len(a), len(b)))


def checks(s, arm):
    """the engine's real environment, tree/binary, client exit, prompt set, token count, usage history, failures"""
    p = []
    want = {k: v for k, v in COMMON_ENV.items() if k.isupper() and not v.startswith("unset")}
    want.update(ARM_EXTRA[arm])
    p += ["%s=%r" % (k, s["env"].get(k)) for k, v in want.items() if s["env"].get(k) != v]
    p += ["%s set" % k for k in SWITCHES if k in s["env"] and k not in ARM_EXTRA[arm]]
    if s["meta"].get("tree_head") != ARM_TREE[arm]:
        p.append("tree_head %s" % s["meta"].get("tree_head"))
    if s["meta"].get("binary_sha256") != BIN[arm]:
        p.append("binary %s" % s["meta"].get("binary_sha256"))
    if s["meta"].get("client_rc") != "0":
        p.append("client_rc %s" % s["meta"].get("client_rc"))
    dirty = s["meta"].get("tree_dirty_lines")
    if arm in PROBE_DIFF:
        if dirty != "1" or ("diff_sha256=%s" % PROBE_DIFF[arm]) not in TREES:
            p.append("probe tree dirty=%s / diff hash not in trees.txt" % dirty)
    elif dirty != "0":
        p.append("dirty=%s" % dirty)
    if s["meta"].get("usage_before") != s["meta"].get("usage_after"):
        p.append("usage history changed")
    want_names = NAMES if arm in ("D", "Z") else NAMES[:len(s["order"])]
    if s["order"] != want_names or not s["order"]:
        p.append("prompts %s" % s["order"])
    p += ["%s chunks=%s" % (n, r.get("decode_chunks")) for n, r in s["rows"].items() if r.get("decode_chunks") != 300]
    p += ["fail line: %s" % l[:160] for l in s["log"] if re.search(r"out of memory|FAILED|Traceback|error", l, re.I)][:3]
    if arm in PROBE_DIFF and len(s["logits"]) != 300 * len(s["order"]):
        p.append("logit lines %d for %d prompts" % (len(s["logits"]), len(s["order"])))
    if arm not in PROBE_DIFF and (s["logits"] or s["cpu_moe"]):
        p.append("probe lines in a non-probe build")
    return p


def text_problems(s):
    return ["%s %s sha %s != D s1 %s (first differing char %s)" % (s["dir"], n, r["sha256"][:12], REF[n]["sha256"][:12],
                                                                  first_diff(REF[n]["text"], r["text"]))
            for n, r in s["rows"].items() if r["sha256"] != REF[n]["sha256"]]


def gate(arm):
    probs = []
    for s in ARMS[arm]:
        probs += ["%s: %s" % (s["dir"], x) for x in checks(s, arm)] + text_problems(s)
    return probs


def evidence(arm):
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=buf, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as t:
        for f in sorted(glob.glob(os.path.join(RAW, arm, "s*", "*"))):
            info = t.gettarinfo(f, arcname=os.path.join("drift", os.path.relpath(f, RAW)))
            info.mtime = 0; info.uid = info.gid = 0; info.uname = info.gname = ""
            with open(f, "rb") as fh:
                t.addfile(info, fh)
    data = buf.getvalue()
    os.makedirs(os.path.join(HERE, "evidence"), exist_ok=True)
    path = os.path.join(HERE, "evidence", "drift-arm-%s.tgz" % arm)
    open(path, "wb").write(data)
    return {"uri": os.path.relpath(path, HERE), "sha256": hashlib.sha256(data).hexdigest(),
            "contains": "drift/%s/s*/ (serve.log, bench.jsonl with full texts, sample.txt, engine_environ.txt, "
                        "procs.txt, nvsmi.csv, gpu_*.csv, mode.txt)" % arm,
            "full_run": "drift.tgz (sha256 in drift.tgz.sha256): all arms, meta/ (host, trees, scripts, probe diffs, "
                        "llama before/after), phase1.json, run.log, summary.json; the run.log inside ends at the "
                        "'ALL DONE; packing' line, because run_drift.sh appends its 'packed <sha256>' line only after "
                        "writing the archive (the raw run.log on shadowfax has it)"}


def stat(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": round(statistics.mean(xs), 4), "stdev": round(statistics.stdev(xs), 4) if len(xs) > 1 else None,
            "median": round(statistics.median(xs), 4), "min": min(xs), "max": max(xs), "n": len(xs)}


def sample_tok_s(s):
    return round(statistics.mean([s["rows"][n]["decode_tok_s"] for n in s["order"]]), 4)


EVID = {a: evidence(a) for a in ARMS}
QUALITY = ("greedy: the full output text of each of the 6 prompts in every sample of this arm has the same sha256 as "
           "the same prompt in dev sample 1 (arm D s1), i.e. all 3 cold dev runs, all 3 cold pr0 runs and dev vs pr0 "
           "byte-identical; plus the engine's real environment (/proc/<pid>/environ), tree SHA, clean tree, binary "
           "sha256, client exit code, 300 streamed chunks per prompt, an unchanged .coli_usage and no failure line in "
           "serve.log, checked per sample")


def run(arm, cfg):
    ss = ARMS[arm]
    speeds = [sample_tok_s(s) for s in ss]
    probs = gate(arm)
    return {
        "config": cfg,
        "samples": {"tok_s": speeds,
                    "per_prompt_tok_s": [[s["rows"][n]["decode_tok_s"] for n in NAMES] for s in ss],
                    "text_sha256": [[s["rows"][n]["sha256"] for n in NAMES] for s in ss],
                    "sample_dirs": [s["dir"] for s in ss], "run_order_seq": [int(s["meta"]["seq"]) for s in ss],
                    "gpu_mem_used_mib_at_ready": [s["ready"] for s in ss],
                    "gpu_mem_used_mib_max": [s["nvsmi"]["max_mem_mib"] for s in ss],
                    "tok_s_mean": round(statistics.mean(speeds), 4), "tok_s_stdev": round(statistics.stdev(speeds), 4)},
        "median_tok_s": statistics.median(speeds),
        "engine": {"commit": ARM_TREE[arm], "binary_sha256": BIN[arm]},
        "quality": {"method": QUALITY, "passed": not probs, "problems": probs},
        "evidence": EVID[arm],
    }


# ---------------------------------------------------------------- phase 2: probe builds, per-token logit/hidden hash
def cmp_logits(a, b):
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), None)
    tok = next((i for i in range(n) if a[i].split()[1] != b[i].split()[1]), None)
    return {"compared": n, "len_a": len(a), "len_b": len(b), "identical": a == b, "first_diff_line": first,
            "first_token_diff": tok, "n_diff_lines": sum(1 for i in range(n) if a[i] != b[i]),
            "first_a": a[first] if first is not None else None, "first_b": b[first] if first is not None else None}


PROBES = [(arm, s) for arm in ("Dp", "Zp") for s in ARMS[arm]]
phase2 = {"pairs": {}, "logit_stream_sha256": {}, "samples": []}
for (a, sa), (b, sb) in itertools.combinations(PROBES, 2):
    phase2["pairs"]["%s%s_vs_%s%s" % (a, sa["dir"][-1], b, sb["dir"][-1])] = cmp_logits(sa["logits"], sb["logits"])
for arm, s in PROBES:
    phase2["logit_stream_sha256"]["%s%s" % (arm, s["dir"][-1])] = hashlib.sha256("\n".join(s["logits"]).encode()).hexdigest()
    phase2["samples"].append({"dir": s["dir"], "seq": int(s["meta"]["seq"]), "prompts": s["order"],
                              "logit_lines": len(s["logits"]), "cpu_moe_lines": len(s["cpu_moe"]),
                              "text_identical_to_D_s1": {n: s["rows"][n]["sha256"] == REF[n]["sha256"] for n in s["order"]},
                              "mean_tok_s": sample_tok_s(s), "checks": checks(s, arm)})

# ---------------------------------------------------------------- probe speed on the same two prompts as phase 1
# The probe builds ran only decode_0 and decode_1, so their mean_tok_s is over 2 prompts and phase 1's over 6:
# compare like with like, the mean of decode_0 and decode_1 per sample in every arm.
PROBE_PROMPTS = NAMES[:2]
assert all(s["order"] == PROBE_PROMPTS for a in ("Dp", "Zp") for s in ARMS[a]), "probe prompt set"


def two_prompt_tok_s(s):
    return round(statistics.mean([s["rows"][n]["decode_tok_s"] for n in PROBE_PROMPTS]), 4)


SAME = {a: [two_prompt_tok_s(s) for s in ARMS[a]] for a in ARMS}
SAME_MED = {a: statistics.median(v) for a, v in SAME.items()}
SAME_MEAN = {a: statistics.mean(v) for a, v in SAME.items()}
probe_vs_plain = {"Dp_vs_D_median": round(SAME_MED["Dp"] / SAME_MED["D"], 4), "Zp_vs_Z_median": round(SAME_MED["Zp"] / SAME_MED["Z"], 4),
                  "Dp_vs_D_mean": round(SAME_MEAN["Dp"] / SAME_MEAN["D"], 4), "Zp_vs_Z_mean": round(SAME_MEAN["Zp"] / SAME_MEAN["Z"], 4),
                  "Zp_vs_Dp_median": round(SAME_MED["Zp"] / SAME_MED["Dp"], 4)}
worst = max(abs(v - 1) for k, v in probe_vs_plain.items() if k != "Zp_vs_Dp_median")
PROBE_SPEED_NOTE = (
    "Probe builds ran at parity with the plain builds on the same two prompts (decode_0, decode_1; mean of the two per "
    "sample): D %s, Z %s, Dp %s, Zp %s tok/s; probe/plain %.4fx (dev) and %.4fx (pr0) on medians, within %.1f%% on "
    "means, so the logging did not slow decode. phase2 mean_tok_s is over these 2 prompts and phase-1 tok_s over all "
    "6, so the two are not comparable directly." % (
        "/".join("%g" % v for v in SAME["D"]), "/".join("%g" % v for v in SAME["Z"]),
        "/".join("%g" % v for v in SAME["Dp"]), "/".join("%g" % v for v in SAME["Zp"]),
        probe_vs_plain["Dp_vs_D_median"], probe_vs_plain["Zp_vs_Z_median"], math.ceil(worst * 1000) / 10))
phase2["speed_same_prompts"] = {"prompts": PROBE_PROMPTS, "tok_s_per_sample": SAME,
                                "median": {a: round(v, 4) for a, v in SAME_MED.items()},
                                "mean": {a: round(v, 4) for a, v in SAME_MEAN.items()},
                                "ratios": probe_vs_plain, "note": PROBE_SPEED_NOTE}


# ---------------------------------------------------------------- pr0 vs dev decode speed: how big, how sure
def welch(a, b):
    """Welch's t-test of b against a: t, Welch-Satterthwaite df, two-sided p from Student's t
    (p = 1 - 2 * integral_0^|t| of the t density, composite Simpson with 20000 intervals; no scipy needed)."""
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    t = (statistics.mean(b) - statistics.mean(a)) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    f = lambda x: c * (1 + x * x / df) ** (-(df + 1) / 2)
    n, h = 20000, abs(t) / 20000
    area = (f(0) + f(abs(t)) + sum((4 if i % 2 else 2) * f(i * h) for i in range(1, n))) * h / 3
    return {"t": round(t, 2), "df": round(df, 1), "p_two_sided": round(1 - 2 * area, 3)}


D_TOK, Z_TOK = [sample_tok_s(s) for s in ARMS["D"]], [sample_tok_s(s) for s in ARMS["Z"]]
SPEED = {"dev_tok_s": D_TOK, "pr0_tok_s": Z_TOK, "dev_median": statistics.median(D_TOK), "pr0_median": statistics.median(Z_TOK),
         "median_ratio": round(statistics.median(Z_TOK) / statistics.median(D_TOK), 4),
         "every_pr0_sample_above_every_dev_sample": min(Z_TOK) > max(D_TOK),
         "welch_pr0_vs_dev": welch(D_TOK, Z_TOK), "no_change_threshold": "median ratio within 1% (0.99..1.01)"}
SPEED["note"] = (
    "pr0 median %.4f vs dev %.4f tok/s: %+.2f%%, below the 1%% no-change threshold, so the outcome is no-change. "
    "%s, but with n=3 per arm Welch's t = %.2f (df %.1f), two-sided p = %.2f: not significant. Read: pr0 does not "
    "slow decode here; this is no evidence of a speed-up either." % (
        SPEED["pr0_median"], SPEED["dev_median"], (SPEED["median_ratio"] - 1) * 100,
        "Every pr0 sample is faster than every dev sample" if SPEED["every_pr0_sample_above_every_dev_sample"]
        else "The pr0 and dev samples overlap",
        SPEED["welch_pr0_vs_dev"]["t"], SPEED["welch_pr0_vs_dev"]["df"], SPEED["welch_pr0_vs_dev"]["p_two_sided"]))

# ---------------------------------------------------------------- why no drift was expected here: the known triggers
TTFT = [s["rows"][n]["ttft_s"] for a in ("D", "Z") for s in ARMS[a] for n in NAMES]
for a in ("D", "Z"):  # the gate (checks) already refuses any of these set in the engine's real environment
    for s in ARMS[a]:
        assert not any(k in s["env"] for k in ("COLI_CUDA_MOE_DOUBLE", "DSV4_CUDA_EXPERT_MIRRORS", "DSV4_CUDA_PIN_HOST",
                                               "CUDA_LAUNCH_BLOCKING")), s["dir"]
TRIGGERS = (
    "The run lacked every documented trigger of the race: (1) no long first request (6 one-sentence prompts, TTFT "
    "%.1f-%.1f s; on a development tree with the same copies, runs differed from the first decode step after a long "
    "first request, pr0 commit b2c8ca61); (2) the default mirror budget (DSV4_CUDA_EXPERT_MIRRORS unset; a smaller "
    "budget made the drift show within a few tokens and flip a token in 2 of 3 runs, b2c8ca61); (3) no double bank "
    "(COLI_CUDA_MOE_DOUBLE unset; with it, 10-14 PR #1 arm C diverged in 2 of 3 cold runs before the fix); (4) refills "
    "not forced onto the pageable path (DSV4_CUDA_PIN_HOST unset, so refill sources were page-locked up to the 20 GiB "
    "cap, and dev queues page-locked refills on the compute stream; the PR #0 harness used DSV4_CUDA_PIN_HOST=0; "
    "whether any refill here went past the cap was not logged)." % (min(TTFT), max(TTFT)))


# ---------------------------------------------------------------- Hermes on shadowfax while llama.cpp was down
def window():
    stop = next(re.match(r"\[\S+ (\S+)\]", l).group(1) for l in lines(os.path.join(RAW, "run.log"))
                if "stopping llama-server-active" in l)
    back = kv(os.path.join(RAW, "meta", "llama_after.txt"))["restored_at"][11:19]
    return stop, back


def hermes():
    path = os.path.join(HERE, "evidence", "hermes-window.txt")
    stop, back = window()
    txt = lines(path)
    dog = [m.groups() for m in (re.match(r"(\d\d:\d\d:\d\d) (\S+)\s+(ok|FAILED)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\[.*\]) / (.*)$", l)
                                for l in txt) if m and stop <= m.group(1) <= back]
    cron = [m.groups() for m in (re.match(r"(\w{12}) (\S+)\s+(\w+)\s+started (\S+) finished (\S*) error=(.*) delivery=(\S*)$", l)
                                 for l in txt) if m and stop <= m.group(4) <= back]
    assert dog and all(d[2] == "FAILED" and d[3] == "False" and d[5] == "False" and d[6] == "[]" for d in dog), dog
    starts = [d[0] for d in dog]
    jobs = {}
    for c in cron:
        jobs.setdefault(c[1], []).append("%s (%s%s)" % (c[3], c[2], (": " + c[5].strip("'\"")) if c[5].strip("'\"") else ""))
    note = ("Hermes during the GPU window (llama.cpp stopped %s-%s PDT): the shadowfax-ensure-active-stack watchdog "
            "(OnUnitActiveSec=5min user timer) ran %d times from %s to %s, reported the backend unhealthy every time "
            "and restarted nothing (restarted_units [], recovered false); the Hermes cron jobs that ran in the window "
            "had no LLM backend: %s. Source: evidence/hermes-window.txt." % (
                stop, back, len(dog), starts[0], starts[-1],
                "; ".join("%s at %s" % (j, ", ".join(t)) for j, t in jobs.items()) if jobs else "none ran"))
    return {"llama_down": [stop, back], "watchdog_runs": len(dog), "watchdog_first": starts[0], "watchdog_last": starts[-1],
            "watchdog_restarted_units": sorted({d[6] for d in dog}),
            "cron_runs": [{"job": c[1], "job_id": c[0], "started": c[3], "finished": c[4], "status": c[2],
                           "error": c[5].strip("'\""), "delivery": c[6]} for c in cron],
            "evidence": {"uri": "evidence/hermes-window.txt",
                         "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                         "captured_by": "evidence/hermes_window.py (read-only: journalctl --user, executions.db mode=ro)"},
            "note": note}


HERMES = hermes()

CONCLUSION = (
    "No drift was reproduced: dev 7b55a9b3 and pr0 a2c8ba7a gave byte-identical greedy text across 3 cold runs each "
    "and against each other on all 6 prompts, and the probe builds gave bit-identical per-token logit and final-hidden "
    "hashes (2 dev, 2 pr0 runs x 600 tokens). %s So this neither shows dev race-free nor demonstrates the fix end to "
    "end; the race itself is shown by PR #0's single-GPU harness (a refill in place under a queued fused MoE read the "
    "refilled expert in 50 of 50 and 300 of 300 trials on the unfixed backend, in none with the fix; b2c8ca61) and by "
    "the 10-14 double-bank divergence before the fix. Speed: %s" % (TRIGGERS, SPEED["note"]))

OUT = []


def manifest(name, hypothesis, changed, base_arm, base_val, trial_arm, trial_val):
    bc = dict(COMMON_ENV); bc[changed] = base_val
    tc = dict(COMMON_ENV); tc[changed] = trial_val
    b = run(base_arm, bc); t = run(trial_arm, tc)
    ratio = t["median_tok_s"] / b["median_tok_s"]
    rec = {"version": 1, "hypothesis": hypothesis, "commit": PR0, "model": MODEL,
           "command": "build: %s; run: %s; baseline arm %s, trial arm %s" % (BUILD, RUN, base_arm, trial_arm),
           "prompt_hash": PROMPT_HASH, "hardware": HW, "warmup_runs": 0,
           "changed_variables": [changed], "baseline": b, "trial": t,
           "outcome": "improvement" if ratio > 1.01 else ("regression" if ratio < 0.99 else "no-change"),
           "speedup_median_tok_s": round(ratio, 4),
           "drift_observation": {
               "dev_text_differs_across_cold_runs": any(len({s["rows"][n]["sha256"] for s in ARMS["D"]}) > 1 for n in NAMES),
               "pr0_text_differs_across_cold_runs": any(len({s["rows"][n]["sha256"] for s in ARMS["Z"]}) > 1 for n in NAMES),
               "dev_vs_pr0_text_differs": any(ARMS["Z"][0]["rows"][n]["sha256"] != REF[n]["sha256"] for n in NAMES),
               "probe_logit_streams_identical": all(v["identical"] for v in phase2["pairs"].values()),
               "note": ("dev did not drift in these runs: its 3 cold runs gave byte-identical text on all 6 prompts, and "
                        "the probe builds gave bit-identical per-token logits and final-hidden-row hashes (2 dev, 2 pr0 "
                        "runs x 600 tokens). " + TRIGGERS + " The upload race pr0 fixes is timing-dependent, so this is "
                        "no evidence that dev is race-free, and this manifest makes no claim that pr0 fixes a drift "
                        "observed here. " + PROBE_SPEED_NOTE + " See stats.json phase2.")},
           "speed_comparison": SPEED,
           "conclusion": CONCLUSION,
           "side_effects": HERMES["note"]}
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(rec, f, indent=2); f.write("\n")
    OUT.append(name)
    print("%-26s %-11s %7.4f -> %7.4f tok/s (%.4fx) quality %s/%s" % (
        name, rec["outcome"], b["median_tok_s"], t["median_tok_s"], ratio, b["quality"]["passed"], t["quality"]["passed"]))


manifest("pr0-engine-decode.json",
         "The pr0/dsv4-fixes tip (decode-mirror uploads and in-place refills queued on the compute stream with h2d() "
         "instead of a legacy-stream cudaMemcpy, plus the other pr0 fixes) leaves one-GPU REAP-132 greedy decode output "
         "byte-identical to upstream dev and does not slow decode",
         "engine_commit", "D", DEV, "Z", PR0)

stats = {"reference": "D s1", "prompt_hash": PROMPT_HASH,
         "reference_sha256": {n: REF[n]["sha256"] for n in NAMES},
         "reference_text_chars": {n: len(REF[n]["text"]) for n in NAMES},
         "phase1": {}, "phase2": phase2, "arms": {}}
for arm in ("D", "Z"):
    ss = ARMS[arm]
    stats["phase1"][arm] = {
        n: {"distinct_sha256": sorted({s["rows"][n]["sha256"] for s in ss}),
            "n_samples": len(ss),
            "identical_across_cold_runs": len({s["rows"][n]["sha256"] for s in ss}) == 1,
            "first_diff_char_vs_s1": [first_diff(ss[0]["rows"][n]["text"], s["rows"][n]["text"]) for s in ss[1:]],
            "identical_to_D_s1": all(s["rows"][n]["sha256"] == REF[n]["sha256"] for s in ss)} for n in NAMES}
for arm, ss in ARMS.items():
    a = {"tree": ARM_TREE[arm], "binary_sha256": BIN[arm], "extra_env": ARM_EXTRA[arm], "gate_problems": gate(arm), "samples": []}
    for s in ss:
        a["samples"].append({
            "dir": s["dir"], "seq": int(s["meta"]["seq"]), "start": s["meta"].get("start"),
            "wall_s": int(s["meta"].get("wall_s", 0)), "ready_s": int(s["meta"].get("ready_s", 0)),
            "client_s": int(s["meta"].get("client_s", 0)), "mean_tok_s": sample_tok_s(s),
            "per_prompt": {n: {k: s["rows"][n].get(k) for k in ("decode_tok_s", "ttft_s", "decode_chunks", "decode_s", "total_s", "sha256")}
                           for n in s["order"]},
            "gpu_mem_used_mib": {"before": s["before"], "ready": s["ready"], "end": s["end"], "max": s["nvsmi"]["max_mem_mib"]},
            "gpu_mean_util_pct": s["nvsmi"]["mean_util_pct"],
            "logit_lines": len(s["logits"]), "cpu_moe_lines": len(s["cpu_moe"]),
            "usage_before": s["meta"].get("usage_before"), "usage_after": s["meta"].get("usage_after"),
            "checks": checks(s, arm)})
    a["mean_tok_s"] = stat([x["mean_tok_s"] for x in a["samples"]])
    stats["arms"][arm] = a
stats["pr0_vs_dev_median_tok_s"] = round(stats["arms"]["Z"]["mean_tok_s"]["median"] / stats["arms"]["D"]["mean_tok_s"]["median"], 4)
stats["speed_comparison"] = SPEED
stats["conclusion"] = CONCLUSION
stats["hermes_during_window"] = HERMES
stats["manifests"] = OUT
stats["evidence"] = EVID
with open(os.path.join(HERE, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2); f.write("\n")
for arm, a in stats["arms"].items():
    m = a["mean_tok_s"]
    print("%-2s n=%d tok/s mean %.4f sd %s median %.4f gate_problems=%s" % (arm, m["n"], m["mean"], m["stdev"], m["median"], a["gate_problems"] or "none"))
for k, v in phase2["pairs"].items():
    print("phase2 %s identical=%s compared=%d first_diff_line=%s" % (k, v["identical"], v["compared"], v["first_diff_line"]))
for arm in ("D", "Z"):
    print("phase1 %s identical across cold runs per prompt: %s" % (arm, [stats["phase1"][arm][n]["identical_across_cold_runs"] for n in NAMES]))
