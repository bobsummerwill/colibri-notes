#!/usr/bin/env python3
"""Build colibri experiment manifests (docs/experiments/manifest.example.json format) for PR #1."""
import glob, hashlib, io, json, os, statistics, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "pr1bench")
DEV = "7b55a9b3792527f52d1a2337fd9e312b9a7983e3"
PR1 = "48767c016c993e59cd74f5511983be7bb65ec791"

filler = " ".join(["The quick brown fox jumps over the lazy dog while the committee reviews section %d of the quarterly infrastructure report." % i for i in range(90)])
filler2 = " ".join(["Station %d logged a temperature of %d degrees and a wind speed of %d knots during the overnight survey." % (i, 10 + i % 17, 3 + i % 11) for i in range(130)])
PROMPTS = [filler + "\n\nSummarize the above in one sentence.", filler2 + "\n\nWhich station had the highest wind speed? Answer briefly."]
PROMPT_HASH = hashlib.sha256(json.dumps(PROMPTS).encode()).hexdigest()

HW = {
    "cpu": "AMD Ryzen 9 9950X3D, 16 cores / 32 threads, 1 socket (engine: 16 OpenMP compute threads + 3 expert-loader lanes)",
    "ram": "62 GiB DDR5; engine --ram 40 (31.6 GiB routed-expert RAM cache, 6.2 GiB dense resident)",
    "storage": "WD_BLACK SN850X 4 TB NVMe (PCIe 4.0), ext4 root filesystem, model read with O_DIRECT (v4_ssd_io mode=direct-aligned)",
    "os": "Ubuntu 24.04.4 LTS, kernel 7.0.0-34-generic, gcc 13.3.0, CUDA toolkit 12.0.140 (generic build CUDA_ARCH=sm_86 NO_TC=1), NVIDIA driver 595.84; GPU0 NVIDIA GeForce RTX 3090 24 GB on PCIe 4.0 x8 (one GPU used)",
}
MODEL = "DeepSeek V4 Flash REAP 150B (puwaer/DeepSeek-V4-Flash-0731-reap-150b, 132 routed experts, fp4 experts / fp8 dense), colibri safetensors layout, 79 GiB on disk"
COMMON_ENV = {"COLI_CUDA": "1", "COLI_GPU": "0", "DSV4_CUDA_DEVICE": "0", "CUDA_DENSE": "1", "COLI_CUDA_PIPE": "2",
              "COLI_CUDA_ATTN_BATCH": "1", "COLI_CUDA_MOE_BATCH": "1", "V4_MOE_REFILL_GROUP": "12",
              "V4_LOADER_LANES": "3", "V4_PREFIX_CKPT_DISK": "0", "ram_gib": "40", "ctx": "8192", "temperature": "0"}
BUILD = "cd c && make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1"
RUN = ("python3 c/coli serve --model <REAP dir> --ram 40 --ctx 8192 --host 127.0.0.1 --port 8090 with the env in config; "
       "one cold server per sample; client bench_pr1.py sends request 1 (1992-token prompt, max_tokens 100) then request 2 "
       "(3005-token prompt, max_tokens 60), streamed, greedy; tok_s = prompt_tokens / time-to-first-token of the named request")


def load(arm):
    rows = {}
    for f in sorted(glob.glob(os.path.join(RAW, arm, "s*", "bench.jsonl"))):
        for l in open(f):
            d = json.loads(l)
            rows.setdefault(d["name"], []).append(d)
    return rows


def evidence(arm):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for f in sorted(glob.glob(os.path.join(RAW, arm, "s*", "*"))):
            info = t.gettarinfo(f, arcname=os.path.relpath(f, HERE)); info.mtime = 0
            with open(f, "rb") as fh:
                t.addfile(info, fh)
    data = buf.getvalue()
    path = os.path.join(HERE, "evidence", f"pr1-arm-{arm}.tgz")
    open(path, "wb").write(data)
    return {"uri": os.path.relpath(path, HERE), "sha256": hashlib.sha256(data).hexdigest()}


REF = load("A")
REF_SHA = {name: {r["text_sha256"] for r in rs} for name, rs in REF.items()}


def run(arm, cfg, request):
    rows = load(arm)
    speeds = [r["prefill_tok_s"] for r in rows[request]]
    ok = all({r["text_sha256"] for r in rs} <= REF_SHA[name] and len(REF_SHA[name]) == 1 for name, rs in rows.items())
    return {
        "config": cfg,
        "samples": {"tok_s": speeds, "ttft_s": [r["ttft_s"] for r in rows[request]],
                    "decode_tok_s": [r["decode_tok_s"] for r in rows[request]]},
        "median_tok_s": statistics.median(speeds),
        "quality": {"method": "greedy (temperature 0): full output text of every request in every sample has the same sha256 as upstream dev (arm A); A itself identical across its 3 samples",
                    "passed": bool(ok)},
        "evidence": evidence(arm),
    }


def manifest(name, hypothesis, changed, base_arm, base_val, trial_arm, trial_val, request, base_extra=None, trial_extra=None):
    bc = dict(COMMON_ENV); bc[changed] = base_val; bc.update(base_extra or {})
    tc = dict(COMMON_ENV); tc[changed] = trial_val; tc.update(trial_extra or {})
    b = run(base_arm, bc, request); t = run(trial_arm, tc, request)
    rec = {"version": 1, "hypothesis": hypothesis, "commit": PR1, "model": MODEL,
           "command": f"build: {BUILD}; run: {RUN}; metric: request '{request}'",
           "prompt_hash": PROMPT_HASH, "hardware": HW, "warmup_runs": 0,
           "changed_variables": [changed], "baseline": b, "trial": t,
           "outcome": "improvement" if t["median_tok_s"] > b["median_tok_s"] * 1.01 else ("regression" if t["median_tok_s"] < b["median_tok_s"] * 0.99 else "no-change")}
    p = os.path.join(HERE, name)
    json.dump(rec, open(p, "w"), indent=2)
    print(name, rec["outcome"], f'{b["median_tok_s"]:.3f} -> {t["median_tok_s"]:.3f} tok/s ({t["median_tok_s"]/b["median_tok_s"]:.2f}x)', "quality", b["quality"]["passed"], t["quality"]["passed"])


manifest("pr1-engine-1992.json",
         "Batched GPU prefill for expert-pruned checkpoints (pr1) speeds up REAP-132 prefill over upstream, which refuses the GPU batch path for !=256 experts and runs the CPU union",
         "engine_commit", "A", DEV, "B", PR1, "long_prefill")
manifest("pr1-engine-3005.json",
         "Same change, second request (3005-token prompt) of the same sessions",
         "engine_commit", "A", DEV, "B", PR1, "long_prefill_2")
manifest("pr1-expert-major-1992.json",
         "The expert-major fp4 MoE kernel (default on) is faster than the per-route kernel in the generic batched prefill MoE",
         "DSV4_MOE_EXPERT_MAJOR", "E", "0", "B", "unset (default 1)", "long_prefill", base_extra={"engine_commit": PR1}, trial_extra={"engine_commit": PR1})
manifest("pr1-mm-warp-1992.json",
         "The warp-per-row dense batch matmul (auto: I<=1024, i.e. attn.wq_b) is faster than mm_batch in batched prefill",
         "DSV4_MM_WARP", "F", "0", "B", "unset (auto)", "long_prefill", base_extra={"engine_commit": PR1}, trial_extra={"engine_commit": PR1})
