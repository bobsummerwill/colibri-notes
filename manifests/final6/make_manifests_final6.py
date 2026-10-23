#!/usr/bin/env python3
"""Build the PR #1 (#2264) experiment manifests (upstream docs/experiments/manifest.example.json format, validated by
c/experiment_manifest.py) from the final6 runs of 2026-10-22, the full 256-expert DeepSeek V4 Flash. Derived from
../final5/make_manifests_final5.py (same conventions: every number read from the evidence, per-sample texts hashed
again from the stored full text, deterministic per-arm evidence tarballs, a notes block that says what is and is not
known).

usage: make_manifests_final6.py final6 RAW_DIR    RAW_DIR = the extracted ./bench.tgz's bench/ (shadowfax)
       make_manifests_final6.py c4ab   RAW_DIR    RAW_DIR = the extracted ../final6-c4ab/bench.tgz's bench/ (computer4)

final6 (shadowfax, 2x RTX 3090, generic CUDA sm_86 build), written here:
  pr1-default-1992.json / -3005.json  A dev 7df8c27c -> B PR #1 b286167e, the DEFAULT configuration (no
                                      COLI_CUDA_MOE_BATCH): JustVugg's question on #2264 (DSV4_MM_WARP is on by default)
  pr1-double-1992.json / -3005.json   BK -> BD (COLI_CUDA_MOE_DOUBLE=1), both PR #1 with COLI_CUDA_MOE_BATCH=1
  No BK/AK manifest: AK (dev with the bank) has 1 sample and lost its bank on prompt 2 (final7 re-measures it).
c4ab (computer4, RTX 5070, DeepGEMM sm_120a build, CUDA 12.8), written to ../final6-c4ab/:
  pr1-deepgemm-1992.json / -3005.json A dev -> B PR #1, the default configuration
Each mode also writes stats.json, evidence/<run>-arm-<X>.tgz; SHA256SUMS and validation.txt are made by the caller
(see generate.log).
"""
import decimal, glob, gzip, hashlib, io, json, os, re, statistics, sys, tarfile

MODE, RAW = sys.argv[1], os.path.abspath(sys.argv[2])
assert MODE in ("final6", "c4ab"), MODE
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = HERE if MODE == "final6" else os.path.join(HERE, "..", "final6-c4ab")
TGZ = os.path.join(OUTDIR, "bench.tgz")
TGZ_SHA = hashlib.sha256(open(TGZ, "rb").read()).hexdigest()
assert open(TGZ + ".sha256").read().split()[0] == TGZ_SHA, "bench.tgz does not match bench.tgz.sha256"

DEV = "7df8c27c95acf7019dd60714ec1c42fa7e36b730"
PR1 = "b286167e614bb208e5a84a25812d7fb29786b230"
REQS = {"long_prefill": 1992, "long_prefill_2": 3005}


def kv(path):
    out = {}
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                out[k.strip()] = v
    return out


def rd(*p):
    return open(os.path.join(RAW, *p), errors="replace").read()


def hx(pat, text):
    m = re.search(pat, text, re.M)
    assert m, pat
    return m.group(1)


# ---------------------------------------------------------------- the prompts (meta/bench_pr1.py)
filler = " ".join(["The quick brown fox jumps over the lazy dog while the committee reviews section %d of the quarterly infrastructure report." % i for i in range(90)])
filler2 = " ".join(["Station %d logged a temperature of %d degrees and a wind speed of %d knots during the overnight survey." % (i, 10 + i % 17, 3 + i % 11) for i in range(130)])
PROMPTS = [filler + "\n\nSummarize the above in one sentence.", filler2 + "\n\nWhich station had the highest wind speed? Answer briefly."]
PROMPT_HASH = hashlib.sha256(json.dumps(PROMPTS).encode()).hexdigest()
assert PROMPT_HASH == "13fc61ddae676a87f55d68c32d7488f90f0c99af634f30dc846eaa8d1707257f", PROMPT_HASH
BENCH_SRC = rd("meta", "bench_pr1.py")
assert all(s in BENCH_SRC for s in (
    '"The quick brown fox jumps over the lazy dog while the committee reviews section %d of the quarterly infrastructure report." % i for i in range(90)',
    '"Station %d logged a temperature of %d degrees and a wind speed of %d knots during the overnight survey." % (i, 10 + i % 17, 3 + i % 11) for i in range(130)',
    '"\\n\\nSummarize the above in one sentence.", 100, 1992', '"\\n\\nWhich station had the highest wind speed? Answer briefly.", 60, 3005')), \
    "meta/bench_pr1.py prompts differ from the ones hashed here"

COMMON = {"COLI_CUDA": "1", "COLI_GPU": "0", "DSV4_CUDA_DEVICE": "0", "CUDA_DENSE": "1", "COLI_CUDA_PIPE": "2",
          "COLI_CUDA_ATTN_BATCH": "1", "V4_MOE_REFILL_GROUP": "12", "V4_LOADER_LANES": "3", "V4_PREFIX_CKPT_DISK": "0"}
CFG_FIXED = {"ram_gib": "40", "ctx": "8192", "temperature": "0"}
RUNLOG = rd("run.log")

# ---------------------------------------------------------------- the run's identity and host
if MODE == "final6":
    TIPS = kv(os.path.join(RAW, "meta", "tips.txt"))
    assert TIPS["dev_sha"] == DEV and TIPS["pr1_sha"] == PR1 and TIPS["dev_is_ancestor_of_pr1"] == "yes", TIPS
    TREES = {}
    for block in rd("meta", "trees.txt").split("tree=")[1:]:
        d = dict(l.strip().split("=", 1) for l in block.splitlines()[1:] if "=" in l)
        TREES[d["head"]] = d
    BIN = {DEV: TREES[DEV]["binary_sha256"], PR1: TREES[PR1]["binary_sha256"]}
    assert all(TREES[c]["status_lines"] == "0" for c in BIN), TREES
    CE = kv(os.path.join(RAW, "meta", "common_env.txt"))
    assert dict(x.split("=", 1) for x in CE["COMMON_ENV"].split()) == COMMON, CE
    assert "--ram 40 --ctx 8192 --host 127.0.0.1 --port 8090" in CE["SERVE_ARGS"], CE
    HOST = rd("meta", "host.txt")
    CPU_MODEL = hx(r"^Model name:\s+(.+?)\s*$", HOST)
    CPUS, CORES = int(hx(r"^CPU\(s\):\s+(\d+)", HOST)), int(hx(r"^Core\(s\) per socket:\s+(\d+)", HOST))
    SOCKETS, MAXMHZ, GOV = int(hx(r"^Socket\(s\):\s+(\d+)", HOST)), float(hx(r"^CPU max MHz:\s+([\d.]+)", HOST)), hx(r"^governor=(\S+)", HOST)
    FREEB = HOST.split("total        used")[1]
    MEM_B, SWAP_B = int(hx(r"^Mem:\s+(\d+)\s", FREEB)), int(hx(r"^Swap:\s+(\d+)\s", FREEB))
    KERNEL, OSNAME = hx(r"^Linux shadowfax (\S+) ", HOST), hx(r'^PRETTY_NAME="([^"]+)"', HOST)
    GCC, NVCC, PY = hx(r"^gcc \(.*\) (\S+)$", HOST), hx(r"Cuda compilation tools, release [\d.]+, V(\S+)", HOST), hx(r"^Python (\S+)", HOST)
    GPUS = [m.groups() for m in re.finditer(r"^\d, (NVIDIA GeForce RTX 3090), GPU-[\w-]+, ([\d.]+), (\d+) MiB, (\d+), (\d+), \d+, \d+,", HOST, re.M)]
    assert len(GPUS) == 2 and GPUS[0] == GPUS[1], GPUS
    GPU_NAME, DRIVER, GPU_MIB, PCIE_GEN_MAX, PCIE_W_MAX = GPUS[0]
    assert re.search(r"^GPU0\s+X\s+NV4", HOST, re.M), "NVLink topology"
    DISK = hx(r"^nvme0n1\s+(WD_BLACK SN850X 4000GB)", HOST)
    ROOTFS = re.search(r"^/\s+/dev/nvme0n1p2 (\S+)\s+(\S+)", HOST, re.M).groups()
    MODEL_KIB = int(hx(r"^total (\d+)$", HOST))
    MEMAVAIL_MIB = int(hx(r"preflight common ok: MemAvailable=(\d+) MiB", RUNLOG))
    LLAMA_DOWN = hx(r"\[(\S+ \S+)\] stopping llama-server-active", RUNLOG)
    LLAMA_UP = hx(r"restore: llama health \S+ \(ok=1\) at (\S+?);", RUNLOG)[:19].replace("T", " ")
    BUILD = "cd c && make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j8"
    assert ("(%s)" % BUILD.split("&& ")[1]) in RUNLOG, "build command differs from run.log"
    GPU_IDX = ("0", "1")
else:
    HOST = rd("meta", "host.txt")
    BIN = {DEV: hx(r"tree=\S+/dg/dev head=%s binary=([0-9a-f]{64})" % DEV, HOST),
           PR1: hx(r"tree=\S+/dg/pr1b head=%s binary=([0-9a-f]{64})" % PR1, HOST)}
    SCRIPT = rd("meta", "run_c4ab.sh")
    assert "ORDER=\"W:1 A:1 B:1 B:2 A:2 A:3 B:3\"" in SCRIPT
    assert re.search(r"COMMON_ENV=\(COLI_CUDA=1 COLI_GPU=0 DSV4_CUDA_DEVICE=0 CUDA_DENSE=1 COLI_CUDA_PIPE=2 COLI_CUDA_ATTN_BATCH=1\s+"
                     r"V4_MOE_REFILL_GROUP=12 V4_LOADER_LANES=3 V4_PREFIX_CKPT_DISK=0\)", SCRIPT), "COMMON_ENV differs"
    assert 'SERVE_ARGS="--model $MODEL --ram 40 --ctx 8192 --host 127.0.0.1 --port $PORT"' in SCRIPT
    CPU_MODEL = hx(r"^Model name:\s+(.+?)\s*$", HOST)
    CPUS = int(hx(r"^CPU\(s\):\s+(\d+)", HOST))
    MEM_G = int(hx(r"^Mem:\s+(\d+)\s", HOST)); SWAP_G = int(hx(r"^Swap:\s+(\d+)\s", HOST))
    AVAIL_G = int(hx(r"^Mem:\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)", HOST))
    KERNEL = hx(r"^Linux computer4 (\S+) ", HOST)
    GPU_NAME, DRIVER, GPU_MIB = re.search(r"^0, (NVIDIA GeForce RTX 5070), ([\d.]+), (\d+) MiB", HOST, re.M).groups()
    VOX = hx(r"voxtype\s+(\d+)MiB", HOST)
    BUILD = ("cd c && make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 DEEPGEMM=1 DEEPGEMM_HOME=<pinned DeepGEMM checkout> "
             "CUDA_HOME=/usr/local/cuda-12.8 NVCC=/usr/local/cuda-12.8/bin/nvcc -j8")
    GPU_IDX = ("0",)

# ---------------------------------------------------------------- per-sample data


def nvsmi(path):
    peak = {}
    for line in open(path):
        p = [x.strip() for x in line.split(",")]
        if len(p) < 3 or p[0].startswith("timestamp") or not p[2].split()[0].isdigit():
            continue
        peak[p[1]] = max(peak.get(p[1], 0), int(p[2].split()[0]))
    return peak


HANDLED_OOM = re.compile(r"^\[DSV4 CUDA\] ([A-Za-z0-9 _-]+) allocation: out of memory$", re.M)


def handled_oom(log):
    out = {}
    for m in HANDLED_OOM.finditer(log):
        out[m.group(1)] = out.get(m.group(1), 0) + 1
    return out


def load(arm):
    out = []
    for d in sorted(glob.glob(os.path.join(RAW, arm, "s*")), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)))):
        meta = kv(os.path.join(d, "sample.txt"))
        if os.path.exists(os.path.join(d, "engine_environ.txt")):   # final6: the engine process's real environment
            env, env_src = kv(os.path.join(d, "engine_environ.txt")), "engine /proc/<pid>/environ"
        else:                                                         # c4ab: the env -i line the server got
            env, env_src = dict(x.split("=", 1) for x in meta["server_env"].split()), "sample.txt server_env (env -i)"
        rows = [json.loads(l) for l in open(os.path.join(d, "bench.jsonl")) if l.strip().startswith("{")]
        for r in rows:
            assert hashlib.sha256(r["text"].encode()).hexdigest() == r["text_sha256"], (d, r["name"])
        log = open(os.path.join(d, "serve.log"), errors="replace").read()
        out.append({"dir": os.path.relpath(d, RAW), "meta": meta, "env": env, "env_src": env_src,
                    "rows": {r["name"]: r for r in rows}, "serve_log": log, "gpu_peak": nvsmi(os.path.join(d, "nvsmi.csv")),
                    "handled_oom": handled_oom(log)})
    return out


if MODE == "final6":
    ARM_TREE = {"A": DEV, "B": PR1, "BK": PR1, "BD": PR1}
    ARM_EXTRA = {"A": {}, "B": {}, "BK": {"COLI_CUDA_MOE_BATCH": "1"}, "BD": {"COLI_CUDA_MOE_BATCH": "1", "COLI_CUDA_MOE_DOUBLE": "1"}}
else:
    ARM_TREE = {"A": DEV, "B": PR1}
    ARM_EXTRA = {"A": {}, "B": {}}
NOT_SET = ("COLI_CUDA_MOE_BATCH", "COLI_CUDA_MOE_DOUBLE", "DSV4_MM_WARP", "DSV4_MM_WARP_VERIFY", "V4_DRAFT", "V4_MTP",
           "DSV4_CUDA_VRAM_RESERVE_MB")
ARMS = {a: load(a) for a in ARM_TREE}
assert all(len(ss) == 3 for ss in ARMS.values()), {a: len(ss) for a, ss in ARMS.items()}
W = load("W")[0]
REF = {n: {s["rows"][n]["text_sha256"] for s in ARMS["A"]} for n in REQS}
assert all(len(v) == 1 for v in REF.values()), REF
REF = {k: next(iter(v)) for k, v in REF.items()}


def env_ok(s, arm):
    want = dict(COMMON); want.update(ARM_EXTRA[arm])
    bad = ["%s=%s missing (%s)" % (k, v, s["env_src"]) for k, v in want.items() if s["env"].get(k) != v]
    # the coli wrapper passes V4_DRAFT=0 / V4_MTP=0 to the engine itself: only a non-zero value is "set"
    bad += ["%s=%s set" % (k, s["env"][k]) for k in NOT_SET if s["env"].get(k, "0") != "0" and k not in ARM_EXTRA[arm]]
    if MODE == "final6" and (s["env"].get("RAM_GB") != "40" or s["env"].get("CTX") != "8192"):
        bad.append("engine RAM_GB/CTX")
    if s["meta"].get("tree_head") != ARM_TREE[arm] or s["meta"].get("binary_sha256") != BIN[ARM_TREE[arm]]:
        bad.append("tree/binary")
    if s["meta"].get("client_rc") != "0" or s["meta"].get("tree_dirty_lines") != "0" or s["meta"].get("ready") != "1":
        bad.append("client_rc/dirty/ready")
    if MODE == "final6" and (s["meta"].get("usage_before") != "missing" or s["meta"].get("usage_after") != "missing"):
        bad.append(".coli_usage appeared")
    log = s["serve_log"]
    if "no usage history says which, so it fills as experts are used" not in log:
        bad.append("serve.log: expert-cache line missing")
    batch = re.findall(r"v4_gpu moe-batch=(\S+)", log)
    if "COLI_CUDA_MOE_BATCH" in ARM_EXTRA[arm]:
        if batch != ["on", "on"] or len(re.findall(r"moe-batch=on bank=256-experts", log)) != 2:
            bad.append("serve.log: moe-batch %s" % batch)
    elif batch:
        bad.append("serve.log: moe-batch line in the default configuration: %s" % batch)
    if ("COLI_CUDA_MOE_DOUBLE" in ARM_EXTRA[arm]) != ("v4_gpu moe-double=on" in log):
        bad.append("serve.log: moe-double state")
    if re.search(r"Traceback|Segmentation fault|Aborted|core dumped", log):
        bad.append("serve.log: crash")
    rest = HANDLED_OOM.sub("", log)
    if re.search(r"out of memory", rest, re.I):
        bad.append("serve.log: unhandled out of memory")
    return bad


def gate(arm):
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


EVID = {}


def evidence(arm):
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=buf, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as t:
        for f in sorted(glob.glob(os.path.join(RAW, arm, "s*", "*"))):
            info = t.gettarinfo(f, arcname=os.path.join(MODE, os.path.relpath(f, RAW)))
            info.mtime = 0; info.uid = info.gid = 0; info.uname = info.gname = ""
            with open(f, "rb") as fh:
                t.addfile(info, fh)
    data = buf.getvalue()
    os.makedirs(os.path.join(OUTDIR, "evidence"), exist_ok=True)
    path = os.path.join(OUTDIR, "evidence", "%s-arm-%s.tgz" % (MODE, arm))
    open(path, "wb").write(data)
    return {"uri": os.path.relpath(path, OUTDIR), "sha256": hashlib.sha256(data).hexdigest(),
            "contains": "%s/%s/s*/ (serve.log, bench.jsonl with full texts, sample.txt, nvsmi.csv%s)" % (
                MODE, arm, ", engine_environ.txt, serve_environ.txt, procs.txt, gpu_*.csv, mode.txt, bench.out/err, error_count.txt" if MODE == "final6" else ", bench.out/err"),
            "full_run": "bench.tgz (sha256 %s, also in bench.tgz.sha256): every arm, meta/, run.log, summary.txt" % TGZ_SHA}


def vals(arm, request, key):
    return [s["rows"][request][key] for s in ARMS[arm]]


def dmean(xs, places=4):
    m = sum(decimal.Decimal(str(x)) for x in xs) / len(xs)
    return float(m.quantize(decimal.Decimal(1).scaleb(-places), rounding=decimal.ROUND_HALF_UP))


def stat(xs):
    return {"mean": round(statistics.mean(xs), 4), "stdev": round(statistics.stdev(xs), 4),
            "cv_pct": round(100 * statistics.stdev(xs) / statistics.mean(xs), 2),
            "median": round(statistics.median(xs), 4), "min": min(xs), "max": max(xs), "n": len(xs)}


def pct(a, b):
    return 100.0 * (a / b - 1)


def hm(ts):
    return ts[11:19]


ORDER_TEXT = " ".join(n for _, n in sorted((int(s["meta"]["seq"]), "%s s%s" % (a, s["meta"]["sample"])) for a, ss in ARMS.items() for s in ss))
assert W["meta"]["seq"] == "1" and W["meta"]["tree_head"] == PR1 and all(W["rows"][n]["text_sha256"] == REF[n] for n in REQS), "W"
N_OUT = {a: sum(len(s["rows"]) for s in ss) for a, ss in ARMS.items()}
ALL_IDENTICAL = all(r["text_sha256"] == REF[n] for ss in ARMS.values() for s in ss for n, r in s["rows"].items())
OOM = {a: [s["handled_oom"] for s in ss] for a, ss in ARMS.items()}
GPU_PEAK = {a: {i: max(s["gpu_peak"].get(i, 0) for s in ss) for i in GPU_IDX} for a, ss in ARMS.items()}
RESERVE = {a: sorted(set(re.findall(r"mirror-reserve=\d+MiB", " ".join(s["serve_log"] for s in ss)))) for a, ss in ARMS.items()}
TT = {(a, n): stat(vals(a, n, "ttft_s")) for a in ARMS for n in REQS}
DECODE = {n: {a: dmean(vals(a, n, "decode_tok_s")) for a in ARMS} for n in REQS}

# ---------------------------------------------------------------- notes
MODEL = ("DeepSeek V4 Flash (deepseek-ai/DeepSeek-V4-Flash-0731, 256 routed experts, fp4 experts / fp8 dense, 48 "
         "safetensors shards), colibri safetensors layout, %s" % (
             "%.1f GiB on disk (meta/host.txt `ls` total)" % (MODEL_KIB / 2 ** 20) if MODE == "final6" else "the same checkpoint copied to computer4"))
ram = [s for s in ARMS["A"][0]["serve_log"].splitlines() if s.startswith("ram_tiers ")][0]
assert ram == "ram_tiers available=40.00GiB dense=resident(6.27GiB) target_slots=59 target_cache=31.59GiB head=resident-bf16 projected=39.79GiB", ram
threads = hx(r"deepseek-v4: (\d+) compute threads", ARMS["A"][0]["serve_log"])
if MODE == "final6":
    HW = {
        "cpu": "%s, %d cores / %d threads, %d socket, governor %s, max %.2f GHz (engine: %s OpenMP compute threads + "
               "3 expert-loader lanes)" % (CPU_MODEL.replace(" 16-Core Processor", ""), CORES * SOCKETS, CPUS, SOCKETS, GOV, MAXMHZ / 1000, threads),
        "ram": "%.1f GiB (free -b; %.1f GiB swap); engine --ram 40 (ram_tiers: 31.59 GiB routed-expert RAM cache = 59 of 256 "
               "experts per layer, 6.27 GiB dense resident); MemAvailable %.1f GiB (%d MiB) at preflight with Hermes's model "
               "server still loaded" % (MEM_B / 2 ** 30, SWAP_B / 2 ** 30, MEMAVAIL_MIB / 1024, MEMAVAIL_MIB),
        "storage": "%s NVMe, %s root filesystem (%s), model read with O_DIRECT; OS page cache not dropped between samples" % (
            DISK.replace("4000GB", "4 TB"), ROOTFS[0], ROOTFS[1]),
        "os": "%s, kernel %s, gcc %s, Python %s, CUDA toolkit %s (generic build CUDA_ARCH=sm_86 NO_TC=1), NVIDIA driver %s; "
              "2x %s %d GB, PCIe gen%s x%s-capable, NVLink NV4 bridge; only GPU0 used (COLI_GPU=0 DSV4_CUDA_DEVICE=0)" % (
                  OSNAME, KERNEL, GCC, PY, NVCC, DRIVER, GPU_NAME, int(GPU_MIB) // 1024, PCIE_GEN_MAX, PCIE_W_MAX),
    }
    RUN_LINE = ("final6, shadowfax, samples %s - %s PDT (Hermes's model server down %s - %s); bench.tgz sha256 %s" % (
        hm(min(s["meta"]["start"] for ss in ARMS.values() for s in ss)), hm(max(s["meta"]["end"] for ss in ARMS.values() for s in ss)),
        LLAMA_DOWN[11:], LLAMA_UP[11:], TGZ_SHA))
else:
    HW = {
        "cpu": "%s, %d threads (meta/host.txt lscpu; engine: %s OpenMP compute threads + 3 expert-loader lanes)" % (CPU_MODEL, CPUS, threads),
        "ram": "%d GiB (free -g; %d GiB swap), %d GiB available at the start of the run; engine --ram 40 (ram_tiers: 31.59 GiB "
               "routed-expert RAM cache = 59 of 256 experts per layer, 6.27 GiB dense resident)" % (MEM_G, SWAP_G, AVAIL_G),
        "storage": "not recorded in this run's meta/host.txt (the model directory's disk and filesystem are not in the evidence); "
                   "OS page cache not dropped between samples",
        "os": "Linux kernel %s (computer4), CUDA toolkit 12.8 (/usr/local/cuda-12.8, DeepGEMM build: -gencode "
              "arch=compute_120a,code=sm_120a -DCOLI_DSV4_DEEPGEMM, pinned DeepGEMM headers), NVIDIA driver %s; 1x %s, "
              "%s MiB, of which the desktop's Voxtype held %s MiB (nvidia-smi at the start)" % (KERNEL, DRIVER, GPU_NAME, GPU_MIB, VOX),
    }
    RUN_LINE = "final6 DeepGEMM, computer4, samples %s - %s PDT (run.log, start of W to the end of the last sample; the tarball was packed before the run.log line ALL SAMPLES DONE); bench.tgz sha256 %s" % (
        hx(r"^\[\S+ (\S+)\] === sample 1:", RUNLOG), re.findall(r"^\[\S+ (\S+)\] --- ", RUNLOG, re.M)[-1], TGZ_SHA)

RUN = ("python3 c/coli serve --model <V4 Flash dir> --ram 40 --ctx 8192 --host 127.0.0.1 --port 8090, server environment "
       "exactly PATH/HOME/USER/LOGNAME/LANG%s + the config's env keys (env -i); one freshly started server process per "
       "sample (OS page cache not dropped between samples); one discarded warm-up sample W on the pr1 tip first, then "
       "%s; client bench_pr1.py sends request 1 (long_prefill: 1992-token prompt, max_tokens 100) then request 2 "
       "(long_prefill_2: 3005-token prompt, max_tokens 60), streamed, greedy; tok_s = prompt_tokens / time-to-first-token "
       "of the named request" % ("" if MODE == "final6" else "/LD_LIBRARY_PATH", "the arms interleaved, run order (seq) " + ORDER_TEXT))

WARMUP = {"long_prefill": (0, "warmup_runs 0: request 1 (long_prefill) is the first request the client sends to each freshly "
                              "started server process. The run's discarded W sample (seq 1, pr1 tip, both requests, texts identical "
                              "to A's) was a separate server process before the first measured sample."),
          "long_prefill_2": (1, "warmup_runs 1: request 2 (long_prefill_2) is served by the same server process right after "
                                "request 1 (the 1992-token prompt, max_tokens 100), its one warm-up here; request 1 is itself "
                                "measured, in the matching -1992 manifest")}
OUTPUT_NOTE = ("%d outputs in %d samples (%s), %s byte-identical (full-text sha256, recomputed from the stored text) to arm A's "
               "per-request text (long_prefill %s, long_prefill_2 %s); the discarded warm-up W also matched.%s" % (
                   sum(N_OUT.values()), sum(len(ss) for ss in ARMS.values()), ", ".join("%s %d" % x for x in N_OUT.items()),
                   "all" if ALL_IDENTICAL else "NOT all", REF["long_prefill"][:12], REF["long_prefill_2"][:12],
                   "" if MODE == "final6" else " The same two texts as the shadowfax final6 run (generic CUDA build, RTX 3090)."))
OOM_NOTE = ("Handled allocation failures (serve.log '[DSV4 CUDA] <what> allocation: out of memory'; the engine falls back "
            "and goes on, the texts are unaffected) per sample: %s." % "; ".join(
                "%s %s" % (a, ", ".join("s%d %s" % (i + 1, o or "none") for i, o in enumerate(v))) for a, v in OOM.items()))
GPU_NOTE = ("Peak memory.used (nvsmi.csv every 5 s, max over the arm's samples): %s. Decode-mirror reserve lines in "
            "serve.log: %s." % ("; ".join("%s %s" % (a, ", ".join("GPU%s %d MiB" % (i, v[i]) for i in GPU_IDX)) for a, v in GPU_PEAK.items()),
                                ", ".join("%s %s" % (a, "/".join(v) or "none printed") for a, v in RESERVE.items())))
DECODE_NOTE = ("Observation (not the hypothesis): decode right after the prefill (bench_pr1.py decode_tok_s, 3-sample means): %s." % (
    "; ".join("%s: %s" % (n, ", ".join("%s %.4f" % (a, v) for a, v in d.items())) for n, d in DECODE.items())))
VAR = lambda request, arms: "TTFT stdev / CV over 3 samples: %s" % "; ".join(
    "%s %.3f s / %.2f%%" % (a, TT[(a, request)]["stdev"], TT[(a, request)]["cv_pct"]) for a in arms)


def verify_note():
    vs = load("V")
    assert len(vs) == 1
    v = vs[0]
    hits = [(int(c), int(m)) for c, m in re.findall(r"mm-warp verify calls=(\d+) mismatches=(\d+)", v["serve_log"])]
    assert hits and hits[-1] == (1500, 0) and all(m == 0 for _, m in hits), hits
    same = all(v["rows"][n]["text_sha256"] == REF[n] for n in REQS)
    assert v["env"].get("DSV4_MM_WARP_VERIFY") == "1" and "COLI_CUDA_MOE_BATCH" not in v["env"]
    return ("Arm V (seq %s, not timed for a manifest: verify runs the old kernel beside the new one): B's configuration plus "
            "DSV4_MM_WARP_VERIFY=1 on the pr1 tip. serve.log counts %d mm_batch_w calls with 0 mismatching bits (last line "
            "'mm-warp verify calls=%d mismatches=%d'); its texts are %s A's. So the warp dense kernel that is on by default "
            "(auto: fp8 shapes with I a multiple of 256 and <= 1024) produced the old kernel's bits on every call it took "
            "in this configuration." % (v["meta"]["seq"], hits[-1][0], hits[-1][0], hits[-1][1], "identical to" if same else "DIFFERENT from"))


OUT = []


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
                    "prompt_tokens": [r["prompt_tokens"] for r in rows], "text_sha256": [r["text_sha256"] for r in rows],
                    "sample_dirs": [s["dir"] for s in ss], "run_order_seq": [int(s["meta"]["seq"]) for s in ss],
                    "handled_allocation_failures": [s["handled_oom"] for s in ss],
                    "tok_s_mean": round(statistics.mean(speeds), 4), "tok_s_stdev": round(statistics.stdev(speeds), 4),
                    "ttft_s_mean": round(statistics.mean([r["ttft_s"] for r in rows]), 3),
                    "ttft_s_stdev": round(statistics.stdev([r["ttft_s"] for r in rows]), 3)},
        "median_tok_s": statistics.median(speeds),
        "engine": {"commit": ARM_TREE[arm], "binary_sha256": BIN[ARM_TREE[arm]]},
        "quality": {"method": ("greedy (the engine ignores temperature): the full output text of both requests in every sample "
                               "of this arm has the same sha256 (recomputed from the stored text) as upstream dev's (arm A, "
                               "itself identical across its 3 samples); per sample also checked: the server's environment (%s), "
                               "tree SHA and binary sha256, the serve.log moe-batch / moe-double lines, no crash and no "
                               "unhandled out-of-memory line" % ss[0]["env_src"]),
                    "passed": not probs, "problems": probs},
        "evidence": EVID[arm],
    }


def outcome(b, t):
    """final5's rule (ratio > 1.01 / < 0.99), and here also: every trial sample beyond every baseline sample"""
    ratio = t["median_tok_s"] / b["median_tok_s"]
    bs, ts = b["samples"]["tok_s"], t["samples"]["tok_s"]
    if ratio > 1.01 and min(ts) > max(bs):
        return "improvement", ratio
    if ratio < 0.99 and max(ts) < min(bs):
        return "regression", ratio
    return "no-change", ratio


def manifest(name, hypothesis, changed, base_arm, base_val, trial_arm, trial_val, request, extra_notes, fixed=None):
    bc = dict(COMMON); bc.update(CFG_FIXED); bc.update(fixed or {}); bc[changed] = base_val
    tc = dict(COMMON); tc.update(CFG_FIXED); tc.update(fixed or {}); tc[changed] = trial_val
    b = run(base_arm, bc, request); t = run(trial_arm, tc, request)
    oc, ratio = outcome(b, t)
    rec = {"version": 1, "hypothesis": hypothesis, "commit": PR1, "model": MODEL,
           "command": "build: %s; run: %s; metric: request '%s' (%d prompt tokens); baseline arm %s, trial arm %s" % (
               BUILD, RUN, request, REQS[request], base_arm, trial_arm),
           "prompt_hash": PROMPT_HASH, "hardware": HW, "warmup_runs": WARMUP[request][0],
           "changed_variables": [changed], "baseline": b, "trial": t, "outcome": oc,
           "speedup_median_tok_s": round(ratio, 4),
           "ttft_ratio_median": round(statistics.median(t["samples"]["ttft_s"]) / statistics.median(b["samples"]["ttft_s"]), 4)}
    rec["notes"] = {"run": RUN_LINE, "warmup": WARMUP[request][1], "outcome_rule": (
        "outcome: 'improvement' when the median tok/s ratio is above 1.01 and every trial sample is faster than every "
        "baseline sample, 'regression' the mirror image, else 'no-change' (final5's 1%% threshold plus a no-overlap "
        "condition). Here: median ratio %.4f; baseline tok/s %s, trial %s." % (ratio, b["samples"]["tok_s"], t["samples"]["tok_s"])),
        "variability": VAR(request, (base_arm, trial_arm)), "outputs": OUTPUT_NOTE, "handled_allocation_failures": OOM_NOTE,
        "gpu": GPU_NOTE, "observation_decode": DECODE_NOTE}
    rec["notes"].update(extra_notes)
    with open(os.path.join(OUTDIR, name), "w") as f:
        json.dump(rec, f, indent=2); f.write("\n")
    OUT.append(name)
    print("%-26s %-11s %8.4f -> %8.4f tok/s (%.4fx, TTFT %.4fx) quality %s/%s" % (
        name, oc, b["median_tok_s"], t["median_tok_s"], ratio, rec["ttft_ratio_median"], b["quality"]["passed"], t["quality"]["passed"]))


ENG = "engine_commit"
QUESTION = ("JustVugg's question on #2264 (2026-10-22): DSV4_MM_WARP is on by default (auto) in PR #1's mm_batch, so it "
            "changes the prefill of every FP8 checkpoint on a generic CUDA build, not only the COLI_CUDA_MOE_BATCH=1 path "
            "the earlier manifests measured; measure the default configuration (no COLI_CUDA_MOE_BATCH) on the full V4 "
            "Flash, dev against the branch, prompt and decode, interleaved. It stays on by default if bit-identical and "
            "not slower. The point of this manifest is therefore 'not slower, bit-identical', not a speed-up: in the "
            "default configuration the prefill MoE runs on the CPU union in both arms, and the warp kernel only takes "
            "the batched attention's short fp8 rows.")
if MODE == "final6":
    base_notes = {"question": QUESTION, "mm_warp_verify": verify_note(),
                  "base": "Arm A is upstream dev %s ('%s'); the pr1 tip %s ('%s', tree %s) has dev merged in (meta/tips.txt "
                          "dev_is_ancestor_of_pr1=yes; %s)." % (DEV, TIPS["dev_subject"], PR1, TIPS["pr1_subject"], TIPS["pr1_tree"],
                                                                hx(r"^ (\d+ files changed.*)$", rd("meta", "tips.txt"))),
                  "cache_state": "The model directory had no .coli_usage before or after any sample (sample.txt usage_before / "
                                 "usage_after = missing), and every serve.log says 'expert cache: 59 of 256 experts per layer "
                                 "fit and no usage history says which, so it fills as experts are used'. OS page cache not dropped.",
                  "host": "Hermes's model server (llama-server-active) was stopped %s and restored %s PDT (run.log, health ok); "
                          "the host journal was not captured, so other host activity during samples cannot be stated." % (LLAMA_DOWN[11:], LLAMA_UP[11:]),
                  "earlier": "First measurement of the full 256-expert checkpoint in this project; the final5 manifests "
                             "(../final5) are on REAP-150B (132 experts) with COLI_CUDA_MOE_BATCH=1 and are not comparable."}
    manifest("pr1-default-1992.json",
             "PR #1 (pr1 tip b286167e, dev 7df8c27c merged in) does not slow down the prefill of the full DeepSeek V4 Flash "
             "in the default configuration (no COLI_CUDA_MOE_BATCH; DSV4_MM_WARP on by default) and leaves the output text "
             "byte-identical to upstream dev's",
             ENG, "A", DEV, "B", PR1, "long_prefill", base_notes)
    manifest("pr1-default-3005.json",
             "PR #1 also does not slow down the default configuration's prefill on the longer prompt (request 2, 3005 tokens, "
             "served right after request 1 by the same server process) and leaves the output text byte-identical to upstream dev's",
             ENG, "A", DEV, "B", PR1, "long_prefill_2", base_notes)
    dbl = {"base": base_notes["base"], "cache_state": base_notes["cache_state"], "host": base_notes["host"],
           "bank": "Both arms run the batched GPU prefill with the 256-slot bank (COLI_CUDA_MOE_BATCH=1; serve.log 'v4_gpu "
                   "moe-batch=on bank=256-experts' for both requests in every sample). Reserve lines: BK %s, BD %s. The "
                   "double bank prefetches the next layer's experts while the current one computes (BD serve.log: %s)." % (
                       "/".join(RESERVE["BK"]), "/".join(RESERVE["BD"]),
                       "; ".join(sorted(set(re.findall(r"v4_gpu moe-double swaps=\d+ prefetched=\d+", " ".join(s["serve_log"] for s in ARMS["BD"])))))),
           "earlier": "final5 (../final5/pr1-moe-double-*.json, REAP-150B, dev 736a0682): 1.1985x / 1.1340x; a different "
                      "model and base, so not a repeat."}
    manifest("pr1-double-1992.json",
             "The double expert bank (COLI_CUDA_MOE_DOUBLE=1) speeds up the batched GPU prefill of the full DeepSeek V4 "
             "Flash (256-slot bank) over the single bank, with the output text unchanged",
             "COLI_CUDA_MOE_DOUBLE", "BK", "unset (default 0)", "BD", "1", "long_prefill", dbl,
             fixed={ENG: PR1, "COLI_CUDA_MOE_BATCH": "1"})
    manifest("pr1-double-3005.json",
             "The double expert bank (COLI_CUDA_MOE_DOUBLE=1) also speeds up the full model's batched prefill on request 2 "
             "(3005 tokens, served right after request 1 by the same server process), with the output text unchanged",
             "COLI_CUDA_MOE_DOUBLE", "BK", "unset (default 0)", "BD", "1", "long_prefill_2", dbl,
             fixed={ENG: PR1, "COLI_CUDA_MOE_BATCH": "1"})
else:
    s2 = ARMS["B"][1]
    assert s2["handled_oom"] == {"weight": 1, "kv comp": 420}, s2["handled_oom"]
    c4_notes = {"question": QUESTION + " On a DeepGEMM build DSV4_MM_WARP and DSV4_MOE_EXPERT_MAJOR are compiled out "
                                       "(#ifndef COLI_DSV4_DEEPGEMM), so this checks that the rest of PR #1 leaves the DeepGEMM "
                                       "path's texts and speed alone.",
                "build": "Binaries from DeepGEMM builds (meta/host.txt binary sha256: dev %s, pr1 %s); the build logs are not in "
                         "this tarball but copied next to it (builds/build-dev.log, builds/build-pr1b.log: nvcc -gencode "
                         "arch=compute_120a,code=sm_120a -DCOLI_DSV4_DEEPGEMM), and `cuobjdump --list-elf` of both binaries "
                         "lists deepseek_v4.1.sm_120a.cubin (checked 2026-10-23, not recorded in the run)." % (BIN[DEV][:16], BIN[PR1][:16]),
                "b_s2": "B s2 (seq %s) alone had handled allocation failures (%s) and a GPU0 peak of %d MiB (the other "
                        "samples ~%d MiB); its request-2 TTFT was %.3f s against %s for the other B samples. GPU memory in use "
                        "at the start of the sample was the same as in the others; the cause is not established. The "
                        "median is unaffected for request 1 and takes B s1's value for request 2." % (
                            s2["meta"]["seq"], s2["handled_oom"], s2["gpu_peak"]["0"],
                            statistics.median([s["gpu_peak"]["0"] for ss in ARMS.values() for s in ss]), s2["rows"]["long_prefill_2"]["ttft_s"],
                            " / ".join("%.3f s" % s["rows"]["long_prefill_2"]["ttft_s"] for s in ARMS["B"] if s is not s2)),
                "cache_state": "Every serve.log says 'expert cache: 59 of 256 experts per layer fit and no usage history says "
                               "which'. OS page cache not dropped between samples.",
                "not_filled": "This run's meta records less than shadowfax's: no engine /proc environ (the env -i line is "
                              "checked instead), no storage, OS release, gcc or Python versions, no host journal."}
    manifest("pr1-deepgemm-1992.json",
             "On a DeepGEMM build (RTX 5070, sm_120a), PR #1 does not slow down the full DeepSeek V4 Flash's prefill in the "
             "default configuration and leaves the output text byte-identical to upstream dev's",
             ENG, "A", DEV, "B", PR1, "long_prefill", c4_notes)
    manifest("pr1-deepgemm-3005.json",
             "On a DeepGEMM build, PR #1 also does not slow down the default configuration's prefill on request 2 (3005 "
             "tokens, served right after request 1 by the same server process), text byte-identical to dev's",
             ENG, "A", DEV, "B", PR1, "long_prefill_2", c4_notes)

stats = {"mode": MODE, "bench_tgz_sha256": TGZ_SHA, "reference_A_sha256": REF, "order": ORDER_TEXT, "arms": {}}
for arm, ss in ARMS.items():
    a = {"samples": [], "requests": {}, "gate_problems": gate(arm)}
    for s in ss:
        a["samples"].append({"dir": s["dir"], "seq": int(s["meta"]["seq"]), "start": s["meta"].get("start"),
                             "wall_s": int(s["meta"]["wall_s"]), "gpu_mem_used_mib_max": s["gpu_peak"],
                             "handled_allocation_failures": s["handled_oom"],
                             "requests": {n: {k: r[k] for k in ("ttft_s", "prefill_tok_s", "decode_tok_s", "decode_chunks", "total_s", "text_sha256")}
                                          for n, r in s["rows"].items()}})
    for name in REQS:
        a["requests"][name] = {k: stat(vals(arm, name, k)) for k in ("ttft_s", "prefill_tok_s", "decode_tok_s")}
        a["requests"][name]["identical_to_A"] = all(r == REF[name] for r in vals(arm, name, "text_sha256"))
    stats["arms"][arm] = a
stats["notes"] = {"outputs": OUTPUT_NOTE, "handled_allocation_failures": OOM_NOTE, "gpu": GPU_NOTE, "observation_decode": DECODE_NOTE,
                  "decode_means": DECODE, "warmup": {n: v[1] for n, v in WARMUP.items()}}
if MODE == "final6":
    stats["notes"]["mm_warp_verify"] = verify_note()
stats["manifests"] = OUT
stats["evidence"] = EVID
with open(os.path.join(OUTDIR, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2); f.write("\n")
for arm, a in stats["arms"].items():
    for name, r in a["requests"].items():
        print("%s %-15s ttft median %8.3f cv %5.2f%% | tok/s median %7.4f | identical_to_A=%s" % (
            arm, name, r["ttft_s"]["median"], r["ttft_s"]["cv_pct"], r["prefill_tok_s"]["median"], r["identical_to_A"]))
    if a["gate_problems"]:
        print("  GATE PROBLEMS %s: %s" % (arm, a["gate_problems"]))
