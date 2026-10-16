#!/usr/bin/env python3
"""Build the PR #2 experiment manifests (upstream docs/experiments/manifest.example.json format,
validated by c/experiment_manifest.py) from the final run on shadowfax (run_pr2.sh).

usage: make_manifests_pr2.py RAW_DIR
  RAW_DIR = the extracted pr2.tgz (the directory that holds G1/ X1/ X2/ X3/ P1/ meta/).
Reads, next to this script: hardware.json, and evidence/pr2-host-journal.txt (a filtered excerpt of shadowfax's
user journal for the GPU window; its header gives the exact journalctl/grep command).
Writes, next to this script:
  evidence/pr2-arm-<X>.tgz   deterministic per-arm tarball of RAW_DIR/<X>/ (what a manifest's evidence names)
  pr2-*.json                 the strictly one-variable manifests (c/experiment_manifest.py accepts each):
                             G1 -> X1 (V4_GPU_RESIDENT_FIRST), X1 -> X3 (DSV4_CUDA_PEER_DEVICE), X2 -> X3 (DSV4_CUDA_PEER_COMPUTE)
  supplementary/pr2-peer-device-p2p-read-two-variable.json
                             X1 -> X2 in the same format, but a TWO-variable comparison (X2 sets DSV4_CUDA_PEER_DEVICE=1
                             and DSV4_CUDA_PEER_COMPUTE=0, X1 sets neither), so changed_variables names both and
                             c/experiment_manifest.py rejects it by design; kept for the p2p-read number only
  stats.json                 per-prompt, per-sample and per-arm numbers (mean, stdev, median), VRAM,
                             V4_SERVE_STATS_LOG counters, mode proof, the correctness gate, the serve unit's memory
                             peaks and the host activity in the window (from the journal excerpt), and notes
Metric: per sample, decode tok/s = the mean over the 6 prompts of bench_decode.py's per-prompt
decode_tok_s ((chunks - 1) / (time of the last chunk - time of the first chunk), client side).
"""
import ast, collections, datetime, glob, gzip, hashlib, io, json, os, re, statistics, sys, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(sys.argv[1])
JOURNAL = os.path.join(HERE, "evidence", "pr2-host-journal.txt")
SUPP_DIR = "supplementary"
# earlier single-sample decode numbers (2026-10-13/14, intermediate PR #2 binaries, COLI_CUDA_MOE_DOUBLE=1 set on every
# row per review F43; colibri-claude measurements.md "Decode - PR #2 (6 prompts x 300 tokens, one sample each)")
EARLIER = {"G1": 2.78, "X1": 2.84, "X2": 3.32, "X3": 3.87}
PR1 = "93b741839677ec457868fd71ab668cc1249b9a22"
PR2 = "d0e19de23c3baf314ee932347c42d5f6d49ef4fb"
BIN = {PR1: "a0a65edd3a5041a040bdd6a8abed1fedff1a2226d303a58a177bc87d8fce7c5f",
       PR2: "63a667bac98a95daffc42428f0a82a1fd5051588047b524d8f8b1c564a20b926"}
ARM_TREE = {"G1": PR2, "X1": PR2, "X2": PR2, "X3": PR2, "P1": PR1}
ARM_EXTRA = {"G1": {"V4_SERVE_STATS_LOG": "1"},
             "X1": {"V4_SERVE_STATS_LOG": "1", "V4_GPU_RESIDENT_FIRST": "1"},
             "X2": {"V4_SERVE_STATS_LOG": "1", "V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1", "DSV4_CUDA_PEER_COMPUTE": "0"},
             "X3": {"V4_SERVE_STATS_LOG": "1", "V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1"},
             "P1": {"V4_SERVE_STATS_LOG": "1"}}
SWITCHES = ("COLI_CUDA_MOE_DOUBLE", "V4_GPU_RESIDENT_FIRST", "DSV4_CUDA_PEER_DEVICE", "DSV4_CUDA_PEER_COMPUTE",
            "DSV4_CUDA_BATCHED", "DSV4_CUDA_PEER_RESERVE_MB", "DSV4_CUDA_EXPERT_MIRRORS", "DSV4_CUDA_VRAM_RESERVE_MB")
# mode proof in serve.log: (required patterns, forbidden patterns)
MODE = {"G1": ([r"^v4_serve_turn "], [r"peer-mirrors", r"peer mirror pool", r"peer device"]),
        "X1": ([r"^v4_serve_turn "], [r"peer-mirrors", r"peer mirror pool", r"peer device"]),
        "X2": ([r"^v4_gpu peer-mirrors device=1 .*mode=p2p-read", r"^v4_serve_turn "],
               [r"mode=compute", r"peer mirror pool off", r"peer device \d+ failed"]),
        "X3": ([r"^v4_gpu peer-mirrors device=1 .*mode=compute", r"^v4_serve_turn "],
               [r"mode=p2p-read", r"peer mirror pool off", r"peer device \d+ failed"]),
        "P1": ([], [r"peer-mirrors"])}
VRAM_FIRST = {"G1": "zero", "X1": "positive", "X2": "positive", "X3": "positive", "P1": None}

# the exact prompts of bench_decode.py (read from the copy the run used); prompt_hash = sha256(json.dumps(PROMPTS))
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
              "V4_LOADER_LANES": "3", "V4_PREFIX_CKPT_DISK": "0", "V4_SERVE_STATS_LOG": "1",
              "COLI_CUDA_MOE_DOUBLE": "unset (default 0)", "ram_gib": "40", "ctx": "8192", "temperature": "0 (greedy engine; request says 0.7, ignored)"}
BUILD = "cd c && make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j8"
RUN = ("python3 c/coli serve --model <REAP dir> --ram 40 --ctx 8192 --host 127.0.0.1 --port 8090, server environment "
       "exactly PATH/HOME/USER/LOGNAME/LANG + the config's env keys (env -i; COLI_CUDA_MOE_DOUBLE unset); one cold server "
       "process per sample, 3 samples per arm interleaved with the arm order rotated per round (G1 X1 X2 X3 / X1 X2 X3 G1 / "
       "X2 X3 G1 X1); client bench_decode.py sends 6 prompts one after another, max_tokens 300 each, streamed; "
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
    """{index: memory.used MiB} from a gpu_*.csv (index, memory.used [MiB], memory.total [MiB])"""
    out = {}
    for line in lines(path)[1:]:
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 2 and f[0].isdigit():
            out[int(f[0])] = int(num(f[1]))
    return out


def nvsmi_max(path):
    """max memory.used / utilization.gpu per GPU over the 5 s nvidia-smi samples"""
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
    return {"max_mem_mib": mem,
            "mean_util_pct": {i: round(statistics.mean(v), 1) for i, v in util.items()},
            "n_samples": len(lines(path)) - 1}


def parse_kv_line(l):
    return {k: v for k, v in re.findall(r"(\w+)=([^\s]+)", l)}


def load(arm):
    out = []
    for d in sorted(glob.glob(os.path.join(RAW, arm, "s*")), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)))):
        meta = kv(os.path.join(d, "sample.txt"))
        env = kv(os.path.join(d, "engine_environ.txt"))
        rows = [json.loads(l) for l in open(os.path.join(d, "bench.jsonl")) if l.strip().startswith("{")] \
            if os.path.exists(os.path.join(d, "bench.jsonl")) else []
        for r in rows:  # bench_decode.py stores the full text; the hash is computed here from it
            r["text_sha256"] = hashlib.sha256(r["text"].encode()).hexdigest()
        log = lines(os.path.join(d, "serve.log"))
        out.append({"dir": os.path.relpath(d, RAW), "meta": meta, "env": env, "rows": {r["name"]: r for r in rows},
                    "log": log, "ready": gpu_csv(os.path.join(d, "gpu_ready.csv")),
                    "before": gpu_csv(os.path.join(d, "gpu_before.csv")), "end": gpu_csv(os.path.join(d, "gpu_end.csv")),
                    "nvsmi": nvsmi_max(os.path.join(d, "nvsmi.csv")),
                    "gpu_apps": [l.split(",")[1].strip() for l in lines(os.path.join(d, "gpu_apps.csv"))[1:] if "," in l]})
    return out


ARMS = {a: load(a) for a in ARM_TREE}
# reference: G1 s1; every G1 sample must agree with it, and so must every other arm (and P1)
REF = {n: ARMS["G1"][0]["rows"][n]["text_sha256"] for n in NAMES}


def env_ok(s, arm):
    want = {k: v for k, v in COMMON_ENV.items() if k.isupper() and not v.startswith("unset")}
    want.update(ARM_EXTRA[arm])
    bad = ["%s=%r" % (k, s["env"].get(k)) for k, v in want.items() if s["env"].get(k) != v]
    bad += ["%s set" % k for k in SWITCHES if k in s["env"] and k not in ARM_EXTRA[arm]]
    if s["meta"].get("tree_head") != ARM_TREE[arm] or s["meta"].get("binary_sha256") != BIN[ARM_TREE[arm]]:
        bad.append("tree/binary")
    if s["meta"].get("client_rc") != "0" or s["meta"].get("tree_dirty_lines") != "0":
        bad.append("client_rc/dirty")
    return bad


def serve_stats(s):
    turns = [parse_kv_line(l) for l in s["log"] if l.startswith("v4_serve_turn ")]
    dec = [parse_kv_line(l) for l in s["log"] if l.startswith("v4_dec_moe ")]
    last = dec[-1] if dec else {}
    return {
        "turns": len(turns),
        "engine_tok_s": [float(t["tok_s"]) for t in turns],
        "engine_decode_s": [float(t["decode_s"]) for t in turns],
        "completion": [int(t["completion"]) for t in turns],
        "prompt_tokens": [int(t["prompt"]) for t in turns],
        "hit_pct": [t["hit"] for t in turns],
        "expert_wait_s": [float(t["expert_wait_s"]) for t in turns],
        "expert_matmul_s": [float(t["expert_matmul_s"]) for t in turns],
        "attention_s": [float(t["attention_s"]) for t in turns],
        "vram_first_hits_total": int(turns[-1]["vram_first_hits_total"]) if turns else None,
        "store_fetches_total": int(turns[-1]["store_fetches_total"]) if turns else None,
        "dec_moe_final": {k: (float(v) if "." in v else int(v)) for k, v in last.items()} if last else None,
        "peer_lines": [l for l in s["log"] if re.search(r"peer-mirrors|peer mirror pool|peer device", l)][:4],
        "mirror_lines": [l[:300] for l in s["log"] if re.search(r"mirror-reserve|mirrors=|v4_gpu ", l)][:6],
    }


def mode_problems(s, arm):
    need, forbid = MODE[arm]
    p = ["missing /%s/" % x for x in need if not any(re.search(x, l) for l in s["log"])]
    p += ["forbidden /%s/" % x for x in forbid if any(re.search(x, l) for l in s["log"])]
    vf = serve_stats(s)["vram_first_hits_total"]
    if VRAM_FIRST[arm] == "zero" and vf != 0:
        p.append("vram_first_hits_total=%s want 0" % vf)
    if VRAM_FIRST[arm] == "positive" and not vf:
        p.append("vram_first_hits_total=%s want > 0" % vf)
    p += ["fail line: %s" % l[:160] for l in s["log"] if re.search(r"out of memory|FAILED|Traceback", l)][:3]
    return p


def gate(arm):
    """every prompt of every sample of this arm: full text sha256 == G1 s1's; 6 prompts; env/tree/binary/mode checked"""
    problems = []
    for s in ARMS[arm]:
        problems += ["%s: %s" % (s["dir"], p) for p in env_ok(s, arm) + mode_problems(s, arm)]
        if sorted(s["rows"]) != NAMES:
            problems.append("%s: prompts %s" % (s["dir"], sorted(s["rows"])))
        for n, r in s["rows"].items():
            if r["text_sha256"] != REF[n]:
                problems.append("%s %s sha %s != G1s1 %s" % (s["dir"], n, r["text_sha256"][:12], REF[n][:12]))
    return problems


def evidence(arm):
    buf = io.BytesIO()
    # gzip header mtime 0 and no file name: the tarball is byte-identical on every run
    with gzip.GzipFile(filename="", fileobj=buf, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as t:
        for f in sorted(glob.glob(os.path.join(RAW, arm, "s*", "*"))):
            info = t.gettarinfo(f, arcname=os.path.join("pr2", os.path.relpath(f, RAW)))
            info.mtime = 0; info.uid = info.gid = 0; info.uname = info.gname = ""
            with open(f, "rb") as fh:
                t.addfile(info, fh)
    data = buf.getvalue()
    os.makedirs(os.path.join(HERE, "evidence"), exist_ok=True)
    path = os.path.join(HERE, "evidence", "pr2-arm-%s.tgz" % arm)
    open(path, "wb").write(data)
    return {"uri": os.path.relpath(path, HERE), "sha256": hashlib.sha256(data).hexdigest(),
            "contains": "pr2/%s/s*/ (serve.log with v4_serve_turn/v4_dec_moe stats, bench.jsonl with full texts, sample.txt, "
                        "engine_environ.txt, procs.txt, nvsmi.csv, gpu_*.csv, mode.txt)" % arm,
            "full_run": ("pr2.tgz (sha256 in pr2.tgz.sha256): all arms, meta/ (host, trees, scripts, llama before/after), "
                         "run.log, summary.json; its run.log ends at 'ALL DONE; packing', because run_pr2.sh appends the "
                         "'packed <tgz> <sha256>' line to the live run.log only after packing")}


def stat(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": round(statistics.mean(xs), 4), "stdev": round(statistics.stdev(xs), 4) if len(xs) > 1 else None,
            "median": round(statistics.median(xs), 4), "min": min(xs), "max": max(xs), "n": len(xs)}


def sample_tok_s(s):
    return round(statistics.mean([s["rows"][n]["decode_tok_s"] for n in NAMES]), 4)


def iso(t):
    return datetime.datetime.fromisoformat(t)


def journal():
    """the GPU window's host activity, from the filtered user-journal excerpt (evidence/pr2-host-journal.txt)"""
    L = [l for l in lines(JOURNAL) if l and not l.startswith("#")]
    ts = lambda l: iso(l.split()[0])
    hms = lambda l: l.split()[0][11:19]
    stop = next(ts(l) for l in L if "Stopped llama-server-active" in l)
    start = next(ts(l) for l in L if "Started llama-server-active" in l)
    win = [l for l in L if stop <= ts(l) <= start]
    wd = [hms(l) for l in win if "Starting shadowfax-ensure-active-stack" in l]
    restarted = [l.split('"restarted_units":', 1)[1].strip().rstrip(",") for l in win if '"restarted_units":' in l]
    recovered = [l.split('"recovered":', 1)[1].strip().rstrip(",") for l in win if '"recovered":' in l]
    cron = collections.OrderedDict()
    for l in L:
        m = re.search(r"Started (hermes-worker-cron-[0-9a-f]+)-exec-([0-9a-f]+)\.scope", l)
        if m:
            cron[m.group(2)] = {"job": m.group(1), "started": hms(l), "in_window": stop <= ts(l) <= start}
        m = re.search(r"(hermes-worker-cron-[0-9a-f]+)-exec-([0-9a-f]+)\.scope: Consumed ([\d.]+)s CPU", l)
        if m and m.group(2) in cron:
            cron[m.group(2)].update(ended=hms(l), cpu_s=float(m.group(3)))
    exo_start = [ts(l) for l in win if "Starting exocortex-watchdog" in l]
    exo_end = [ts(l) for l in win if "Finished exocortex-watchdog" in l]
    peaks = [(ts(l), re.search(r"CPU time, ([\d.]+[KMGT]?) memory peak", l).group(1))
             for l in L if "colibri-final-pr2-serve.service: Consumed" in l]
    return {"source": os.path.relpath(JOURNAL, HERE),
            "llama_stopped": stop.isoformat(), "llama_started": start.isoformat(),
            # the last 'Starting llama-server-active' in the window is the restore itself
            "llama_starts_inside_window": sum(1 for l in win if "Starting llama-server-active" in l) - 1,
            "ensure_active_stack": {"runs": len(wd), "first": wd[0], "last": wd[-1],
                                    "backend_unhealthy": sum(1 for l in win if '"error": "backend unhealthy"' in l),
                                    "failed": sum(1 for l in win if "ensure-active-stack.service: Main process exited" in l),
                                    "restarted_units": sorted(set(restarted)), "recovered": sorted(set(recovered)),
                                    "first_run_after_restore_finished_ok": next((hms(l) for l in L if "Finished shadowfax-ensure-active-stack" in l and ts(l) > start), None)},
            "hermes_cron_workers": [dict(v, exec=k) for k, v in cron.items() if v["in_window"]],
            "exocortex_watchdog": {"runs": len(exo_start), "max_duration_s": max((b - a).total_seconds() for a, b in zip(exo_start, exo_end))
                                   if len(exo_end) == len(exo_start) else None},
            "serve_unit_memory_peaks": peaks}


JRN = journal()


def serve_peak(s):
    """the systemd 'memory peak' of this sample's colibri-final-pr2-serve.service run (journal line at its end)"""
    end = iso(s["meta"]["end"])
    near = [p for t, p in JRN["serve_unit_memory_peaks"] if abs((t - end).total_seconds()) <= 5]
    return near[0] if len(near) == 1 else None


def times(ws):
    c = collections.Counter(w["started"] for w in ws)
    out = ["%s%s" % (t, " (two)" if n == 2 else (" (%d)" % n if n > 2 else "")) for t, n in c.items()]
    return ", ".join(out[:-1]) + " and " + out[-1] if len(out) > 1 else out[0]


GPU_APPS_ENGINE_ONLY = all(s["gpu_apps"] and all(a.endswith("/c/deepseek_v4") for a in s["gpu_apps"])
                           for ss in ARMS.values() for s in ss)
GPU1_IDLE_NO_PEER = sorted({s["nvsmi"]["max_mem_mib"].get(1) for a in ("G1", "X1", "P1") for s in ARMS[a]})
W = JRN["ensure_active_stack"]
CRON = JRN["hermes_cron_workers"]
LATE = [w for w in CRON if w.get("ended") and w["ended"] > JRN["llama_started"][11:19]]
HOST_LINE = (
    "Host activity in the llama-down window (%s-%s PDT; %s): the Hermes shadowfax-ensure-active-stack watchdog ran every "
    "5 min (%d runs, %s-%s), reported 'backend unhealthy' each time and restarted nothing (restarted_units %s, recovered %s); "
    "Hermes cron workers started at %s with no LLM backend (%.1f-%.1f s CPU each%s), so those cron jobs ran without a model; "
    "no llama start inside the window, and the first watchdog run after the restore finished ok at %s.%s" % (
        JRN["llama_stopped"][11:19], JRN["llama_started"][11:19], JRN["source"], W["runs"], W["first"], W["last"],
        "/".join(W["restarted_units"]), "/".join(W["recovered"]), times(CRON),
        min(w["cpu_s"] for w in CRON), max(w["cpu_s"] for w in CRON),
        "; the %s one exited at %s, after llama was back" % (LATE[0]["started"], LATE[0]["ended"]) if len(LATE) == 1 else "",
        W["first_run_after_restore_finished_ok"],
        " Nothing else showed up on the GPUs: every sample's gpu_apps.csv (taken at ready) lists only the engine, and GPU1 "
        "peaked at %s MiB in the arms without a peer pool." % "/".join(map(str, GPU1_IDLE_NO_PEER)) if GPU_APPS_ENGINE_ONLY else ""))
assert W["runs"] == W["backend_unhealthy"] == W["failed"] and W["restarted_units"] == ["[]"] and not JRN["llama_starts_inside_window"]

EVID = {}


def run(arm, cfg):
    ss = ARMS[arm]
    speeds = [sample_tok_s(s) for s in ss]
    probs = gate(arm)
    if arm not in EVID:
        EVID[arm] = evidence(arm)
    return {
        "config": cfg,
        "samples": {"tok_s": speeds,
                    "per_prompt_tok_s": [[s["rows"][n]["decode_tok_s"] for n in NAMES] for s in ss],
                    "engine_tok_s": [serve_stats(s)["engine_tok_s"] for s in ss],
                    "text_sha256": [[s["rows"][n]["text_sha256"] for n in NAMES] for s in ss],
                    "sample_dirs": [s["dir"] for s in ss], "run_order_seq": [int(s["meta"]["seq"]) for s in ss],
                    "gpu_mem_used_mib_at_ready": [s["ready"] for s in ss],
                    "gpu_mem_used_mib_max": [s["nvsmi"]["max_mem_mib"] for s in ss],
                    "vram_first_hits_total": [serve_stats(s)["vram_first_hits_total"] for s in ss],
                    "store_fetches_total": [serve_stats(s)["store_fetches_total"] for s in ss],
                    "tok_s_mean": round(statistics.mean(speeds), 4), "tok_s_stdev": round(statistics.stdev(speeds), 4)},
        "median_tok_s": statistics.median(speeds),
        "engine": {"commit": ARM_TREE[arm], "binary_sha256": BIN[ARM_TREE[arm]]},
        "quality": {"method": ("greedy: the full output text of each of the 6 prompts in every sample of this arm has the "
                               "same sha256 as the same prompt in the one-GPU reference (arm G1, sample 1), all 3 G1 samples "
                               "agreeing; plus the engine's real environment (/proc/<pid>/environ), tree SHA, binary sha256, "
                               "client exit code and the mode lines of serve.log (peer-mirrors mode, vram_first_hits_total) "
                               "checked per sample"),
                    "passed": not probs, "problems": probs},
        "evidence": EVID[arm],
    }


OUT, SUPP, SPEEDUP = [], [], {}


def manifest(name, hypothesis, changed, base_arm, base_vals, trial_arm, trial_vals, fixed=None, notes=(), supplementary=None):
    """changed: the variable(s) that differ between the two arms; base_vals/trial_vals: their values, one per variable.
    A record with more than one changed variable is a supplementary comparison, written under supplementary/ and
    rejected by c/experiment_manifest.py by design (it accepts exactly one changed variable)."""
    changed = [changed] if isinstance(changed, str) else list(changed)
    base_vals = [base_vals] if isinstance(base_vals, str) else list(base_vals)
    trial_vals = [trial_vals] if isinstance(trial_vals, str) else list(trial_vals)
    bc = dict(COMMON_ENV); bc.update(fixed or {}); bc.update(zip(changed, base_vals))
    tc = dict(COMMON_ENV); tc.update(fixed or {}); tc.update(zip(changed, trial_vals))
    assert sorted(k for k in set(bc) | set(tc) if bc.get(k) != tc.get(k)) == sorted(changed), name
    assert (len(changed) == 1) == (supplementary is None), name
    b = run(base_arm, bc); t = run(trial_arm, tc)
    path = os.path.join(HERE, SUPP_DIR, name) if supplementary else os.path.join(HERE, name)
    if supplementary:  # evidence uris are relative to the record's own directory
        for r in (b, t):
            r["evidence"] = dict(r["evidence"], uri=os.path.relpath(os.path.join(HERE, r["evidence"]["uri"]), os.path.dirname(path)))
    ratio = t["median_tok_s"] / b["median_tok_s"]
    rec = {"version": 1, "hypothesis": hypothesis, "commit": PR2, "model": MODEL,
           "command": "build: %s; run: %s; baseline arm %s, trial arm %s" % (BUILD, RUN, base_arm, trial_arm),
           "prompt_hash": PROMPT_HASH, "hardware": HW, "warmup_runs": 0,
           "changed_variables": changed, "baseline": b, "trial": t,
           "outcome": "improvement" if ratio > 1.01 else ("regression" if ratio < 0.99 else "no-change"),
           "speedup_median_tok_s": round(ratio, 4)}
    if supplementary:
        rec["supplementary"] = supplementary
    # paths in the notes are relative to the record's own directory, like its evidence uris
    rec["notes"] = [n.replace(JRN["source"], os.path.relpath(JOURNAL, os.path.dirname(path))) for n in notes]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=2); f.write("\n")
    (SUPP if supplementary else OUT).append(os.path.relpath(path, HERE))
    SPEEDUP[os.path.relpath(path, HERE)] = rec["speedup_median_tok_s"]
    print("%-58s %-11s %7.4f -> %7.4f tok/s (%.3fx) quality %s/%s" % (
        os.path.relpath(path, HERE), rec["outcome"], b["median_tok_s"], t["median_tok_s"], ratio,
        b["quality"]["passed"], t["quality"]["passed"]))


ENG = "engine_commit"
UNSET1 = "unset (default 1)"
SUPP_NAME = "pr2-peer-device-p2p-read-two-variable.json"
PEER_COMPUTE_NOTE = (
    "X1 -> X2 is a two-variable comparison: X2 sets DSV4_CUDA_PEER_DEVICE=1 and DSV4_CUDA_PEER_COMPUTE=0, while X1 sets "
    "neither (no X1 engine_environ.txt has DSV4_CUDA_PEER_COMPUTE). Code inspection at d0e19de2 says the second variable "
    "is inert in X1: DSV4_CUDA_PEER_COMPUTE is read only inside peer_runs() (c/backend_cuda_dsv4.cu:1410-1412), which is "
    "called only for an expert mirrored on another device (backend_cuda_dsv4.cu:2107) and, through "
    "dsv4_cuda_peer_compute(), from v4_peer_mirrors_create (c/deepseek_v4.c:11812); neither path exists without "
    "DSV4_CUDA_PEER_DEVICE. So X1 -> X2 very likely measures the p2p-read peer pool alone, but that rests on reading the "
    "code, not on a measured X1 + DSV4_CUDA_PEER_COMPUTE=0 arm. It is therefore kept as a supplementary record (%s/%s, "
    "changed_variables names both; c/experiment_manifest.py rejects it by design), and the strictly one-variable records "
    "for the peer pool are pr2-peer-device-compute.json (X1 -> X3) and pr2-peer-compute.json (X2 -> X3). To make X1 -> X2 "
    "strict, measure an X1 + DSV4_CUDA_PEER_COMPUTE=0 arm (3 samples) in a later GPU window." % (SUPP_DIR, SUPP_NAME))
manifest("pr2-resident-first.json",
         "VRAM-first decode lookups (V4_GPU_RESIDENT_FIRST=1: a routed expert already mirrored in GPU0's VRAM is computed "
         "from the mirror without a RAM/SSD store lookup, and the RAM tier stops duplicating mirrored experts) speed up "
         "one-GPU REAP-132 decode, with outputs unchanged",
         "V4_GPU_RESIDENT_FIRST", "G1", "unset (default 0)", "X1", "1", fixed={ENG: PR2},
         notes=["Strictly one-variable: X1 adds only V4_GPU_RESIDENT_FIRST=1 to G1's environment.", HOST_LINE])
manifest("pr2-peer-device-compute.json",
         "The PR #2 two-GPU default (DSV4_CUDA_PEER_DEVICE=1 with peer compute at its default) speeds up REAP-132 decode "
         "over VRAM-first decode on one GPU, with outputs unchanged (X1 -> X3: the one switch a user sets)",
         "DSV4_CUDA_PEER_DEVICE", "X1", "unset (no peer pool)", "X3", "1",
         fixed={ENG: PR2, "V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_COMPUTE": UNSET1},
         notes=["Strictly one-variable, and one of the two lead PR #2 records with pr2-peer-compute.json: X1 and X3 both "
                "leave DSV4_CUDA_PEER_COMPUTE unset (default 1); X3 adds only DSV4_CUDA_PEER_DEVICE=1. The p2p-read step "
                "X1 -> X2 is only a two-variable supplementary comparison (%s/%s)." % (SUPP_DIR, SUPP_NAME), HOST_LINE])
manifest("pr2-peer-compute.json",
         "Peer compute (DSV4_CUDA_PEER_COMPUTE default 1: GPU1 runs the decode MoE for the experts it mirrors, concurrently "
         "with GPU0) is faster than GPU0 reading GPU1's mirrors in place over NVLink, with outputs unchanged",
         "DSV4_CUDA_PEER_COMPUTE", "X2", "0", "X3", UNSET1,
         fixed={ENG: PR2, "V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1"},
         notes=["Strictly one-variable, and one of the two lead PR #2 records with pr2-peer-device-compute.json: X2 and "
                "X3 both set DSV4_CUDA_PEER_DEVICE=1; X2 adds DSV4_CUDA_PEER_COMPUTE=0, X3 leaves it at its default 1.",
                HOST_LINE])
manifest(SUPP_NAME,
         "SUPPLEMENTARY, TWO-VARIABLE (not a one-variable manifest; c/experiment_manifest.py rejects it by design): a peer "
         "mirror pool on GPU1 (DSV4_CUDA_PEER_DEVICE=1), read in place by GPU0 over NVLink because peer compute is "
         "switched off (DSV4_CUDA_PEER_COMPUTE=0), speeds up REAP-132 decode over VRAM-first decode on one GPU, with "
         "outputs unchanged",
         ["DSV4_CUDA_PEER_DEVICE", "DSV4_CUDA_PEER_COMPUTE"], "X1", ["unset (no peer pool)", UNSET1], "X2", ["1", "0"],
         fixed={ENG: PR2, "V4_GPU_RESIDENT_FIRST": "1"},
         supplementary={"one_variable": False,
                        "validator": "c/experiment_manifest.py rejects this record: changed_variables must name exactly one variable",
                        "strict_records": ["pr2-peer-device-compute.json (X1 -> X3)", "pr2-peer-compute.json (X2 -> X3)"],
                        "note": PEER_COMPUTE_NOTE},
         notes=[HOST_LINE])

# per-arm statistics and the gate, for the report
stats = {"reference": "G1 s1", "reference_sha256": REF, "prompt_hash": PROMPT_HASH, "arms": {}}
for arm, ss in ARMS.items():
    a = {"tree": ARM_TREE[arm], "extra_env": ARM_EXTRA[arm], "samples": [], "gate_problems": gate(arm)}
    for s in ss:
        sv = serve_stats(s)
        a["samples"].append({
            "dir": s["dir"], "seq": int(s["meta"]["seq"]), "start": s["meta"].get("start"),
            "wall_s": int(s["meta"].get("wall_s", 0)), "ready_s": int(s["meta"].get("ready_s", 0)),
            "client_s": int(s["meta"].get("client_s", 0)),
            "mean_tok_s": sample_tok_s(s) if sorted(s["rows"]) == NAMES else None,
            "per_prompt": {n: {k: s["rows"][n].get(k) for k in ("decode_tok_s", "ttft_s", "decode_chunks", "decode_s", "total_s", "text_sha256")}
                           for n in NAMES if n in s["rows"]},
            "per_prompt_identical_to_G1s1": {n: s["rows"][n]["text_sha256"] == REF[n] for n in NAMES if n in s["rows"]},
            "gpu_mem_used_mib": {"before": s["before"], "ready": s["ready"], "end": s["end"], "max": s["nvsmi"]["max_mem_mib"]},
            "gpu_mean_util_pct": s["nvsmi"]["mean_util_pct"],
            "serve_unit_memory_peak": serve_peak(s),
            "serve_stats": sv, "mode_problems": mode_problems(s, arm), "env_problems": env_ok(s, arm)})
    toks = [x["mean_tok_s"] for x in a["samples"] if x["mean_tok_s"] is not None]
    a["mean_tok_s"] = stat(toks)
    a["per_prompt_tok_s"] = {n: stat([s["rows"][n]["decode_tok_s"] for s in ss if n in s["rows"]]) for n in NAMES}
    a["engine_tok_s_mean_of_6"] = stat([statistics.mean(x["serve_stats"]["engine_tok_s"]) for x in a["samples"]
                                        if len(x["serve_stats"]["engine_tok_s"]) == 6])
    a["gpu0_max_mib"] = stat([x["gpu_mem_used_mib"]["max"].get(0) for x in a["samples"]])
    a["gpu1_max_mib"] = stat([x["gpu_mem_used_mib"]["max"].get(1) for x in a["samples"]])
    dm = [x["serve_stats"]["dec_moe_final"] for x in a["samples"] if x["serve_stats"]["dec_moe_final"]]
    # the same statistic (mean over the arm's samples) for every arm's v4_dec_moe totals
    a["dec_moe_final_mean"] = {k: round(statistics.mean(d[k] for d in dm), 2) for k in dm[0]} if dm else None
    a["identical_to_G1s1"] = all(all(x["per_prompt_identical_to_G1s1"].values()) and len(x["per_prompt_identical_to_G1s1"]) == 6
                                 for x in a["samples"])
    stats["arms"][arm] = a
g = stats["arms"]["G1"]["mean_tok_s"]
for arm, a in stats["arms"].items():
    if a["mean_tok_s"] and g:
        a["vs_G1_mean"] = round(a["mean_tok_s"]["mean"] / g["mean"], 4)
        a["vs_G1_median"] = round(a["mean_tok_s"]["median"] / g["median"], 4)
stats["manifests"] = OUT
stats["supplementary"] = SUPP
stats["evidence"] = EVID
stats["peer_compute_note"] = PEER_COMPUTE_NOTE
stats["host_journal"] = dict(JRN, serve_unit_memory_peaks=[[t.isoformat(), p] for t, p in JRN["serve_unit_memory_peaks"]])

# notes: the figures a report quotes, computed here so the wording matches the data
A3 = ("G1", "X1", "X2", "X3")
AR = stats["arms"]
med = lambda a: AR[a]["mean_tok_s"]["median"]
pct = lambda r: 100.0 * (r - 1.0)
earlier = {a: {"earlier_single_sample_tok_s": EARLIER[a], "median_tok_s": med(a), "ratio": round(med(a) / EARLIER[a], 4)} for a in A3}
lo, hi = min(e["ratio"] for e in earlier.values()), max(e["ratio"] for e in earlier.values())
pp = [(AR[a]["per_prompt_tok_s"][n]["stdev"], a, n) for a in A3 for n in NAMES]
others = sorted([x for x in pp if x[2] != "decode_0"], reverse=True)
d0_largest = [a for a in A3 if AR[a]["per_prompt_tok_s"]["decode_0"]["stdev"] == max(x[0] for x in pp if x[1] == a)]
allx = [(a, x) for a in AR for x in AR[a]["samples"]]
total = {int(f[0]): int(num(f[2])) for f in ([y.strip() for y in l.split(",")] for l in lines(os.path.join(RAW, "G1", "s1", "gpu_ready.csv"))[1:]) if f[0].isdigit()}
g0max = collections.Counter(x["gpu_mem_used_mib"]["max"][0] for a, x in allx)
g0top = g0max.most_common(1)[0][0]
g0ready = sorted({x["gpu_mem_used_mib"]["ready"][0] for a, x in allx})
g1 = {a: (sorted({x["gpu_mem_used_mib"]["ready"][1] for x in AR[a]["samples"]}), sorted({x["gpu_mem_used_mib"]["max"][1] for x in AR[a]["samples"]})) for a in AR}
gib = lambda m: m / 1024.0
peaks = collections.Counter(x["serve_unit_memory_peak"] for a, x in allx)
ptop = peaks.most_common(1)[0][0]
odd = sorted([(x["seq"], x["dir"], x["serve_unit_memory_peak"], round(pct(x["mean_tok_s"] / AR[a]["mean_tok_s"]["median"]), 2))
              for a, x in allx if x["serve_unit_memory_peak"] != ptop])
dmm = {a: AR[a]["dec_moe_final_mean"] for a in A3}
runlog = lines(os.path.join(RAW, "run.log"))
stats["notes"] = {
    "manifest_set": (
        "Lead PR #2 with the strictly one-variable records pr2-peer-device-compute.json (X1 -> X3, %.3fx) and "
        "pr2-peer-compute.json (X2 -> X3, %.3fx), with pr2-resident-first.json (G1 -> X1, %.3fx) for the one-GPU step. "
        "X1 -> X2 (p2p-read peer pool, %.3fx) changes two variables, so it is kept only as the supplementary record %s, "
        "which c/experiment_manifest.py rejects by design; see peer_compute_note." % (
            SPEEDUP["pr2-peer-device-compute.json"], SPEEDUP["pr2-peer-compute.json"], SPEEDUP["pr2-resident-first.json"],
            SPEEDUP[os.path.join(SUPP_DIR, SUPP_NAME)], os.path.join(SUPP_DIR, SUPP_NAME))),
    "vs_earlier_single_sample": {
        "arms": earlier,
        "text": ("Medians are %.1f-%.1f%% above the earlier single-sample numbers (%s). Likely cause, not tested in this run: "
                 "the earlier runs had COLI_CUDA_MOE_DOUBLE=1 set (review F43; the review measured the double bank costing "
                 "about 3-5%% of decode) and used intermediate PR #2 binaries; this run isolated neither factor." % (
                     pct(lo), pct(hi), ", ".join("%s %.2f -> %.4f, %+.1f%%" % (a, EARLIER[a], med(a), pct(earlier[a]["ratio"])) for a in A3)))},
    "spread": (
        "Per-arm stdev of the per-sample means (mean of 6 prompts): %s tok/s. Per-prompt stdev over the 3 samples: decode_0 "
        "(each fresh server's first request) is the largest in %s (%s tok/s); every other prompt is %.4f tok/s or less "
        "(largest %s)." % (
            ", ".join("%s %.4f" % (a, AR[a]["mean_tok_s"]["stdev"]) for a in A3),
            "every arm" if len(d0_largest) == len(A3) else ", ".join(d0_largest),
            ", ".join("%s %.4f" % (a, AR[a]["per_prompt_tok_s"]["decode_0"]["stdev"]) for a in A3),
            others[0][0], ", ".join("%s %s %.4f" % (a, n, v) for v, a, n in others[:3]))),
    "vram_mib": (
        "VRAM in MiB (nvidia-smi memory.used; 1 GiB = 1024 MiB), each RTX 3090 has %d MiB (%.1f GiB). GPU0: %s MiB at ready "
        "in all %d samples, peak %d MiB (%.1f GiB) in %d of %d%s. GPU1: %s; %s." % (
            total[0], gib(total[0]), "/".join(map(str, g0ready)), len(allx), g0top, gib(g0top), g0max[g0top], len(allx),
            "".join(" (%d MiB in %s)" % (x["gpu_mem_used_mib"]["max"][0], x["dir"]) for a, x in allx if x["gpu_mem_used_mib"]["max"][0] != g0top),
            "%s MiB peak in every G1, X1 and P1 sample (%s MiB idle before each sample; no peer pool)" % (
                "/".join(map(str, sorted(set(g1["G1"][1] + g1["X1"][1] + g1["P1"][1])))),
                "/".join(map(str, sorted({x["gpu_mem_used_mib"]["before"][1] for a, x in allx})))),
            "X2 and X3 %s MiB at ready and a %s MiB (%.1f GiB) peak in every sample, with the peer mirror pool" % (
                "/".join(map(str, sorted(set(g1["X2"][0] + g1["X3"][0])))), "/".join(map(str, sorted(set(g1["X2"][1] + g1["X3"][1])))),
                gib(max(g1["X3"][1]))))),
    "dec_moe_totals": (
        "v4_dec_moe totals at the end of each run, the mean of the 3 samples for every arm (the same statistic each time; "
        "per-sample values in arms.<arm>.samples[].serve_stats.dec_moe_final): %s." % "; ".join(
            "%s %s" % (k, ", ".join("%s %.2f s" % (a, dmm[a][k]) for a in A3)) for k in ("total_s", "fetch_wait_s", "attach_s", "gpu_s"))),
    "cache_state": (
        "The OS page cache was not dropped between samples, and round 1 was partly cold: the serve unit's systemd memory "
        "peak was %s, against %s for every other sample (%d of %d, including G1/s1, seq 1). That is consistent with those "
        "servers reading model pages that were not cached yet; they are not systematically slower (%s vs their arm medians)." % (
            " and ".join("%s for %s" % (pk, " and ".join("%s (seq %d)" % (d, q) for q, d, p, _ in odd if p == pk))
                         for pk in sorted({p for _, _, p, _ in odd}, key=lambda v: -float(v.rstrip("KMGT")))),
            ptop, peaks[ptop], len(allx), ", ".join("%s %+.2f%%" % (d, r) for q, d, p, r in odd))),
    "host_activity": HOST_LINE,
    "packed_run_log": (
        "pr2.tgz's run.log (%d lines, sha256 %s) ends at %r: run_pr2.sh appends '[2026-10-15 17:33:10] packed "
        "/home/maplemercury/colibri-bench/final/pr2.tgz 290793b8...' to the live run.log only after packing, so the live "
        "copy on shadowfax has that one extra line (sha256 6d100ef326bb...). The other 197 of the 198 files are "
        "byte-identical to the tarball (checked 2026-10-15 from computer4)." % (
            len(runlog), hashlib.sha256(open(os.path.join(RAW, "run.log"), "rb").read()).hexdigest()[:12], runlog[-1][22:]))}
with open(os.path.join(HERE, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2); f.write("\n")
for arm, a in stats["arms"].items():
    m = a["mean_tok_s"]
    print("%s n=%d tok/s mean %.4f sd %s median %.4f vsG1 %s | engine %s | gpu0 max %s gpu1 max %s | identical_to_G1s1=%s" % (
        arm, m["n"], m["mean"], m["stdev"], m["median"], a.get("vs_G1_mean"),
        a["engine_tok_s_mean_of_6"] and a["engine_tok_s_mean_of_6"]["mean"],
        a["gpu0_max_mib"] and a["gpu0_max_mib"]["max"], a["gpu1_max_mib"] and a["gpu1_max_mib"]["max"], a["identical_to_G1s1"]))
    for x in a["samples"]:
        print("   %s seq=%d tok/s=%s per_prompt=%s vf=%s fetch=%s dec_moe=%s" % (
            x["dir"], x["seq"], x["mean_tok_s"], [x["per_prompt"][n]["decode_tok_s"] for n in NAMES if n in x["per_prompt"]],
            x["serve_stats"]["vram_first_hits_total"], x["serve_stats"]["store_fetches_total"], x["serve_stats"]["dec_moe_final"]))
    if a["gate_problems"]:
        print("  GATE PROBLEMS %s: %s" % (arm, a["gate_problems"]))
