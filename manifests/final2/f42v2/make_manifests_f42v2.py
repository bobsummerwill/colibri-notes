#!/usr/bin/env python3
"""F42 v2: upstream-format manifest (docs/experiments/manifest.example.json, validated by
c/experiment_manifest.py) and per-arm evidence tarballs from the run_f42v2.sh run on shadowfax
(2026-10-16 05:43:53-06:18:34 PDT; pr2 tip d0e19de2 + a logging-only probe, no PR #2 switch set,
so the PR #1 code paths).

usage: make_manifests_f42v2.py RAW_DIR
  RAW_DIR = the extracted f42v2.tgz's f42v2/ directory (GB/ U/ GBn/ W/ meta/ run.log ...).
Reads stats.json next to this script (analyze_f42v2_local.py RAW_DIR stats.json) for the decomposition
numbers quoted in the notes. Writes, next to this script:
  evidence/f42v2-arm-<X>.tgz      deterministic per-arm tarball of RAW_DIR/<X>/ (X = GB, U, GBn, W)
  f42v2-prefill-path-24tok.json   the one-variable record GB -> U (COLI_CUDA_MOE_BATCH_MIN), written only
                                  if every GB and U sample's text is byte-identical
"""
import glob, gzip, hashlib, io, json, os, re, statistics as st, sys, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(sys.argv[1])
STATS = json.load(open(os.path.join(HERE, "stats.json")))
PR2 = "d0e19de23c3baf314ee932347c42d5f6d49ef4fb"
PROBE_BIN = "3de5acf4d30720735081153a9d8a9c3ad1a9ec2e7b8cbe81e2aace3f80508b93"
PROBE_DIFF = "a003effa519301065e77ae3aeb9e67d2ad42ea27b3bf66ad5da536e5cd3690c4"
PLAIN_BIN = "63a667bac98a95daffc42428f0a82a1fd5051588047b524d8f8b1c564a20b926"
PROMPT_SHA = "e13a02a377c208b8eac583aa2a52ef82ab5ebff96a0c2e9bbf8c08954e97ca4d"
REF_TEXT = "32534c06abd640d59d2eb39ab6609051f63e9876f533aad1d75dd5cd775835ba"

RUNLOG = open(os.path.join(RAW, "run.log")).read()
LLAMA_DOWN = re.search(r"\[(\S+ \S+)\] stopping llama-server-active", RUNLOG).group(1)
LLAMA_UP = re.search(r"restore: llama health \S+ \(ok=1\) at (\S+?);", RUNLOG).group(1)
WINDOW_S = int(re.search(r"llama-down window (\d+) s", RUNLOG).group(1))
MEMAVAIL = int(re.search(r"preflight ok: MemAvailable=(\d+) MiB", RUNLOG).group(1))
TREES = open(os.path.join(RAW, "meta", "trees.txt")).read()
assert "head=%s" % PR2 in TREES and "binary_sha256=%s" % PROBE_BIN in TREES and "diff_sha256=%s" % PROBE_DIFF in TREES
assert "binary_sha256=%s" % PLAIN_BIN in TREES
assert "ABORT" not in RUNLOG and "NOTE" not in RUNLOG and "budget: skipping" not in RUNLOG


def kv(path):
    return dict(l.rstrip("\n").split("=", 1) for l in open(path) if "=" in l)


def samples(arm):
    out = []
    for d in sorted(glob.glob(os.path.join(RAW, arm, "s*")), key=lambda p: int(p.rsplit("s", 1)[1])):
        meta = kv(os.path.join(d, "sample.txt"))
        rec = json.loads(next(l for l in open(os.path.join(d, "bench.jsonl")) if l.startswith("{")))
        assert hashlib.sha256(rec["text"].encode()).hexdigest() == rec["text_sha256"]
        assert rec["prompt_sha256"] == PROMPT_SHA and meta["client_rc"] == "0" and meta["tree_head"] == PR2
        log = open(os.path.join(d, "serve.log"), errors="replace").read()
        dm = re.findall(r"^v4_dec_moe (.*) \(cumulative\)$", log, re.M)
        out.append({"k": int(meta["sample"]), "seq": int(meta["seq"]), "meta": meta, "rec": rec,
                    "env": dict(x.split("=", 1) for x in meta["server_env"].split()),
                    "dec_moe": {k: float(v) for k, v in (t.split("=") for t in dm[-1].split())} if dm else None,
                    "reserve": re.findall(r"mirror-reserve=(\d+)MiB", log),
                    "bank": len(re.findall(r"moe-batch=on bank=", log)),
                    "trace_start": re.search(r"^v4_dec_trace_start .*$", log, re.M).group(0)
                    if "v4_dec_trace_start" in log else None})
    return out


S = {a: samples(a) for a in ("GB", "U", "GBn", "W")}
assert [len(S[a]) for a in ("GB", "U", "GBn", "W")] == [5, 5, 3, 1]
for a in ("GB", "U", "W"):
    assert all(x["meta"]["binary_sha256"] == PROBE_BIN and x["meta"]["tree_diff_sha256"] == PROBE_DIFF for x in S[a])
assert all(x["meta"]["binary_sha256"] == PLAIN_BIN and x["meta"]["tree_dirty_lines"] == "0" for x in S["GBn"])
assert all(x["reserve"] == ["3100"] for v in S.values() for x in v)
assert all(x["bank"] == 1 for a in ("GB", "GBn", "W") for x in S[a]) and all(x["bank"] == 0 for x in S["U"])
assert all("bank=0 union=688" in x["trace_start"] for x in S["U"])
assert all("bank=688 union=0" in x["trace_start"] for a in ("GB", "W") for x in S[a])

# ---------------------------------------------------------------- text identity (the gate for writing a record)
TEXTS = {a: sorted({x["rec"]["text_sha256"] for x in v}) for a, v in S.items()}
IDENTICAL = TEXTS["GB"] == TEXTS["U"] == TEXTS["GBn"] == [REF_TEXT]
print("texts:", {a: [t[:12] for t in v] for a, v in TEXTS.items()}, "identical GB/U/GBn:", IDENTICAL)

# ---------------------------------------------------------------- environment: exactly one key differs
COMMON_KEYS = ("COLI_CUDA", "COLI_GPU", "DSV4_CUDA_DEVICE", "CUDA_DENSE", "COLI_CUDA_PIPE", "COLI_CUDA_ATTN_BATCH",
               "COLI_CUDA_MOE_BATCH", "V4_MOE_REFILL_GROUP", "V4_LOADER_LANES", "V4_PREFIX_CKPT_DISK",
               "V4_SERVE_STATS_LOG", "DSV4_DECODE_PROF", "V4_DEC_TRACE")
ref_env = S["GB"][0]["env"]
for a in ("GB", "U"):
    for x in S[a]:
        e = dict(x["env"])
        assert e.pop("COLI_CUDA_MOE_BATCH_MIN", None) == ("100000" if a == "U" else None), (a, x["k"])
        assert e == ref_env, (a, x["k"])
COMMON_CFG = {k: ref_env[k] for k in COMMON_KEYS}
COMMON_CFG.update({"COLI_CUDA_MOE_DOUBLE": "unset (default 0)", "DSV4_CUDA_VRAM_RESERVE_MB": "unset (default: bank-sized, 3100 MiB logged in every sample)",
                   "ram_gib": "40", "ctx": "8192", "temperature": "0 (greedy)", "engine_commit": PR2,
                   "engine_build": "d0e19de2 + logging-only probe (probe/r8-pr2-f42probe.diff sha256 %s), binary sha256 %s"
                                   % (PROBE_DIFF, PROBE_BIN)})
ARM_CFG = {"GB": {"COLI_CUDA_MOE_BATCH_MIN": "unset (default 256: the 1,992-token prompt takes the GPU bank)"},
           "U": {"COLI_CUDA_MOE_BATCH_MIN": "100000"}}


# ---------------------------------------------------------------- evidence tarballs (deterministic)
def evidence(arm):
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=buf, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as t:
        for f in sorted(glob.glob(os.path.join(RAW, arm, "s*", "*"))):
            info = t.gettarinfo(f, arcname=os.path.join("f42v2", os.path.relpath(f, RAW)))
            info.mtime = 0; info.uid = info.gid = 0; info.uname = info.gname = ""
            with open(f, "rb") as fh:
                t.addfile(info, fh)
    data = buf.getvalue()
    path = os.path.join(HERE, "evidence", "f42v2-arm-%s.tgz" % arm)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").write(data)
    return {"uri": os.path.relpath(path, HERE), "sha256": hashlib.sha256(data).hexdigest(),
            "contains": "f42v2/%s/s*/ (serve.log with the probe's v4_dec_trace_start / v4_dec_res / v4_dec_trace / "
                        "v4_dec_layer lines and v4_serve_turn / v4_dec_moe / decprof, bench.jsonl with the full text and "
                        "per-chunk times, sample.txt, engine_environ.txt, procs.txt, nvsmi.csv at 1 s, gpu_*.csv, mode.txt)" % arm,
            "full_run": "f42v2.tgz (sha256 in f42v2.tgz.sha256): all arms, meta/ (host, trees, scripts, clients, probe "
                        "diff, build logs, llama before/after), run.log, summary.json, summary.txt"}


EVID = {a: evidence(a) for a in ("GB", "U", "GBn", "W")}
for a, e in EVID.items():
    print("evidence", e["uri"], e["sha256"])

if not IDENTICAL:
    print("texts differ between arms: no manifest written (token-exact quality gate)")
    sys.exit(0)

# ---------------------------------------------------------------- manifest
HOST = open(os.path.join(RAW, "meta", "host.txt")).read()
KERNEL = re.search(r"^Linux shadowfax (\S+)", HOST, re.M).group(1)
HW = {
    "cpu": "AMD Ryzen 9 9950X3D, 16 cores / 32 threads, 1 socket, governor powersave (amd-pstate), max 5.76 GHz "
           "(engine: 16 OpenMP compute threads + 3 expert-loader lanes)",
    "ram": "62 GiB DDR5 (8 GiB swap, 3.1 GiB of it in use by other processes); engine --ram 40 (59 routed-expert RAM "
           "slots per layer = 31.6 GiB cache, 6.2 GiB dense resident); MemAvailable %.1f GiB (%d MiB) at preflight with "
           "llama.cpp still loaded" % (MEMAVAIL / 1024, MEMAVAIL),
    "storage": "WD_BLACK SN850X 4 TB NVMe (PCIe 4.0), ext4 root filesystem, model read with O_DIRECT "
               "(v4_ssd_io mode=direct-aligned); OS page cache not dropped between samples; one discarded warm-up "
               "sample first; serve-unit memory peaks 7.3-9.7 GiB (stats.json serve_unit_memory_peak_gib)",
    "os": "Ubuntu 24.04.4 LTS, kernel %s, gcc 13.3.0, Python 3.12.3, CUDA toolkit 12.0.140 (generic build "
          "CUDA_ARCH=sm_86 NO_TC=1), NVIDIA driver 595.84; 2x NVIDIA GeForce RTX 3090 24 GB on PCIe x8, NVLink bridge; "
          "GPU0 only (COLI_GPU=0 DSV4_CUDA_DEVICE=0), GPU1 idle" % KERNEL}
MODEL = ("DeepSeek V4 Flash REAP 150B (puwaer/DeepSeek-V4-Flash-0731-reap-150b, 132 routed experts, fp4 experts / "
         "fp8 dense), colibri safetensors layout, 79 GiB on disk")
BUILD = ("cd c && make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 "
         "NO_TC=1 -j8 on d0e19de2 + probe/r8-pr2-f42probe.diff (final2/build_r8.sh)")
COMMAND = ("build: %s; run: python3 c/coli serve --model <REAP dir> --ram 40 --ctx 8192 --host 127.0.0.1 --port 8090, "
           "server environment exactly PATH/HOME/USER/LOGNAME/LANG + the config's env keys (env -i); one freshly started "
           "server process per sample; order W(discarded) GB1 U1 U2 GB2 GBn1 GB3 U3 U4 GB4 GBn2 GB5 U5 GBn3; client "
           "bench_f42v2.py sends bench_pr1.py REQUESTS[0] byte for byte (the 1,992-token prompt ending 'Summarize the "
           "above in one sentence.', max_tokens 100), streamed, greedy; the answer is 25 content chunks then EOS; "
           "tok_s = 24 / (last chunk time - first chunk time) (bench_common decode_tok_s); baseline arm GB, trial arm U"
           % BUILD)
QUALITY = ("greedy: the full output text has the same sha256 (%s, the reference answer of the final PR #1 run) in all 5 "
           "GB and all 5 U samples (and in the 3 GBn samples on the unmodified binary and the warm-up); plus per sample "
           "the engine's real environment (/proc/<pid>/environ), tree SHA and diff sha256, binary sha256, client exit "
           "code, exactly one 'mirror-reserve=3100MiB' line, and the prefill path counted by the probe "
           "(v4_dec_trace_start: GB bank=688 union=0, U bank=0 union=688; GB has the 'moe-batch=on bank=132-experts' "
           "and refill lines, U has none)" % REF_TEXT[:16])
D = STATS["decomposition"]
W = STATS["windows"]
CF = STATS["counterfactual_25_pairs"]
RES = STATS["residency"]


def run_block(arm):
    xs = [x["rec"]["decode_tok_s"] for x in S[arm]]
    return {"config": dict(COMMON_CFG, **ARM_CFG[arm]),
            "engine": {"commit": PR2, "binary_sha256": PROBE_BIN, "probe_diff_sha256": PROBE_DIFF},
            "samples": {"tok_s": xs, "tok_s_mean": round(st.mean(xs), 4), "tok_s_stdev": round(st.stdev(xs), 4),
                        "sample_dirs": ["%s/s%d" % (arm, x["k"]) for x in S[arm]],
                        "run_order_seq": [x["seq"] for x in S[arm]],
                        "ttft_s": [x["rec"]["ttft_s"] for x in S[arm]],
                        "text_sha256": [x["rec"]["text_sha256"] for x in S[arm]],
                        "dec_moe": [x["dec_moe"] for x in S[arm]],
                        "store_misses": STATS["misses_per_sample"][arm],
                        "prefill_trace_start": [x["trace_start"].split(" store_total")[0] for x in S[arm]],
                        "serve_unit_memory_peak_gib": [STATS.get("serve_unit_memory_peak_gib", {}).get("%s/s%d" % (arm, x["k"]))
                                                       for x in S[arm]]},
            "median_tok_s": st.median(xs),
            "quality": {"method": QUALITY, "passed": True, "problems": []},
            "evidence": EVID[arm]}


b, t = run_block("GB"), run_block("U")
d = t["median_tok_s"] / b["median_tok_s"] - 1
gbn = [x["rec"]["decode_tok_s"] for x in S["GBn"]]
rec = {
    "version": 1,
    "hypothesis": ("H1 for F42: decode right after a long GPU-bank prefill is slower than after a CPU-union prefill of the "
                   "same prompt because the bank refill looks each routed expert up in the RAM store only once per layer per "
                   "prompt, so at decode start the 59-slot RAM tier of each layer holds late-first-seen experts with usage 1 "
                   "and id-ranked pins, while the CPU union re-looks-up every chunk's experts and leaves the prompt tail's "
                   "experts resident with frequency-ranked pins. Forcing the CPU union with COLI_CUDA_MOE_BATCH_MIN=100000 "
                   "(the mirror reserve stays 3100 MiB) should then decode the identical answer faster, through less "
                   "fetch_wait (fewer RAM misses = O_DIRECT SSD reads) concentrated in the first tokens, with uploads and "
                   "attach unchanged"),
    "commit": PR2, "model": MODEL, "command": COMMAND, "prompt_hash": PROMPT_SHA, "hardware": HW,
    "warmup_runs": 1,
    "changed_variables": ["COLI_CUDA_MOE_BATCH_MIN"], "baseline": b, "trial": t,
    "outcome": "improvement" if d > 0.01 else "regression" if d < -0.01 else "no-change",
    "notes": [
        "Trial vs baseline median: %+.2f%% (U %.4f vs GB %.4f tok/s; means %.4f vs %.4f; Welch t = %.1f on tok/s). Without GB s1 "
        "(the first measured sample, serve-unit memory peak 9.7 GiB, an attach outlier) GB is %.4f (stdev %.4f) and U/GB +%.1f%%."
        % (100 * d, t["median_tok_s"], b["median_tok_s"], t["samples"]["tok_s_mean"], b["samples"]["tok_s_mean"],
           -STATS["speed_GB_vs_U"]["welch_tok_s"]["t"], STATS["robustness_GB_without_s1"]["GB_s2-s5_tok_s"]["median"],
           STATS["robustness_GB_without_s1"]["GB_s2-s5_tok_s"]["stdev"],
           100 * (t["median_tok_s"] / STATS["robustness_GB_without_s1"]["GB_s2-s5_tok_s"]["median"] - 1)),
        "Decomposition, means over 5+5 samples, GB minus U (decode forwards 1-25 on the server): wall %+.3f s, dec_moe total "
        "%+.3f s, fetch_wait %+.3f s, attach %+.3f s (Welch t %.1f: not significant; it comes from GB s1, 3.94 s vs 3.62-3.64 s in "
        "the other GB samples and 3.61-3.66 s in U, and is %+.3f s without it), gpu %+.3f s, "
        "non-MoE (attention etc.) %+.3f s; uploads %+d (GB 2687, U 2723 in every sample), store misses %+.0f (GB %.0f vs U %.0f, "
        "+%.0f%%), %+.2f GiB read from SSD. fetch_wait per extra miss: %.2f ms (one 13.4 MB O_DIRECT read)."
        % (D["server_fwd1_25_s"]["GB-U"], D["dec_moe_total_s"]["GB-U"], D["fetch_wait_s"]["GB-U"], D["attach_s"]["GB-U"],
           D["attach_s"]["welch"]["t"], STATS["robustness_GB_without_s1"]["fwd 1-24 attach_s GB(s2-s5)-U"], D["gpu_s"]["GB-U"],
           D["non_moe_s (server_fwd - dec_moe_total)"]["GB"] - D["non_moe_s (server_fwd - dec_moe_total)"]["U"],
           D["uploads"]["GB-U"], D["store_misses"]["GB-U"], D["store_misses"]["GB"], D["store_misses"]["U"],
           100 * D["store_misses"]["GB-U"] / D["store_misses"]["U"], D["store_GiB_read"]["GB-U"],
           1000 * D["fetch_wait_s"]["GB-U"] / D["store_misses"]["GB-U"]),
        "Token windows (decode forward j produces content token j+1; means, GB minus U): forwards 1-5: time %+.0f ms, fetch_wait "
        "%+.0f ms, misses %+.0f, uploads %+.0f, attach %+.0f ms; 6-10: %+.0f / %+.0f / %+.0f / %+.0f / %+.0f; 11-24: %+.0f / %+.0f "
        "/ %+.0f / %+.0f / %+.0f (the 11-24 attach is GB s1's; without it -8 ms). Per token: %.0f, %.0f and %.0f ms, so "
        "front-loaded and decaying but not gone by token 24 (GB still misses 3-18 more experts per forward at forwards 17-25)."
        % tuple([W[w][k]["GB-U_mean"] * (1000 if k.endswith("_s") else 1) for w in ("1-5", "6-10", "11-24")
                 for k in ("client_ms", "fetch_wait_s", "misses", "uploads", "attach_s")] +
                [W["1-5"]["client_ms"]["GB-U_mean"] / 5, W["6-10"]["client_ms"]["GB-U_mean"] / 5, W["11-24"]["client_ms"]["GB-U_mean"] / 14]),
        "RAM tier at decode start (probe v4_dec_res, read under the store mutex): GB holds 59 experts per layer, all with usage 1 "
        "(the store saw %.1f requests per layer: one per routed expert per prompt), 16 pins per layer all with usage 1, ranked by "
        "id (exactly the 16 lowest resident ids in 27-29 of 43 layers), 7.9 pins active per layer; U holds 59 per layer with "
        "mean usage %.1f (%.1f%% at usage <= 1; %.1f requests per layer), 16 frequency-ranked pins per layer (median usage 9), all "
        "active. Of the experts decode routes in forwards 1-25, %d of %d were resident at decode start in GB vs %d of %d in U; "
        "first forward: %.0f of 258 vs %.0f of 258. Resident-set overlap GB vs U: Jaccard %.2f."
        % (RES["GB"][0]["lreq_mean"], RES["U"][0]["resident_usage_mean"], 100 * RES["U"][0]["resident_usage_le1_frac"], RES["U"][0]["lreq_mean"],
           RES["GB"][0]["decode_needed_resident_at_start"], RES["GB"][0]["decode_needed_distinct"],
           RES["U"][0]["decode_needed_resident_at_start"], RES["U"][0]["decode_needed_distinct"],
           CF["fwd1_routed_resident GB-routing x GB-RAM"]["mean"], CF["fwd1_routed_resident U-routing x U-RAM"]["mean"],
           STATS["residency_set_jaccard"]["GB_s1_vs_U"][0]),
        "Routing caveat: although the text is identical, the two prefill paths leave slightly different KV states, so decode "
        "routing differs in 430 of 1,075 token-layers (527 of 6,450 expert selections, layers 3-42; deterministic within each "
        "arm). This does not drive the gap: swapping routing and RAM sets, GB's routing against U's decode-start RAM finds %.0f "
        "experts resident (U's own: %.0f) and U's routing against GB's RAM finds %.0f (GB's own: %.0f); on the 645 token-layers "
        "with identical routing GB misses %.0f vs U %.0f."
        % (CF["decode_needed_resident_at_start GB-routing x U-RAM"]["mean"], CF["decode_needed_resident_at_start U-routing x U-RAM"]["mean"],
           CF["decode_needed_resident_at_start U-routing x GB-RAM"]["mean"], CF["decode_needed_resident_at_start GB-routing x GB-RAM"]["mean"],
           CF["misses_on_same_routing_token_layers GB"]["mean"], CF["misses_on_same_routing_token_layers U"]["mean"]),
        "Strictly one variable: GB and U run the same probe binary with the same server environment except "
        "COLI_CUDA_MOE_BATCH_MIN=100000 in U (sample.txt server_env, engine_environ.txt). Its only read is the bank's per-call "
        "gate (hint 1,992 fresh tokens < 100000 -> CPU union); the mirror reserve is sized from COLI_CUDA_MOE_BATCH and logs "
        "3100 MiB in every sample; GPU0 peaks %s vs %s MiB." % (STATS["speed"]["GB"]["gpu0_max_mib"]["median"], STATS["speed"]["U"]["gpu0_max_mib"]["median"]),
        "Probe check: GBn (GB's environment on the unmodified d0e19de2 binary %s..., 3 samples) median %.4f tok/s = %.4fx GB, "
        "identical text: the probe changes neither speed nor output." % (PLAIN_BIN[:12], st.median(gbn), st.median(gbn) / b["median_tok_s"]),
        "U's TTFT is %.1f s vs GB's %.1f s: the CPU union is the slow prefill PR #1 replaces; this record is about the decode "
        "after it, not a recommendation to set COLI_CUDA_MOE_BATCH_MIN." % (st.median(t["samples"]["ttft_s"]), st.median(b["samples"]["ttft_s"])),
        "prompt_hash = sha256 of the prompt string of bench_pr1.py REQUESTS[0] (bench_f42v2.py checks it before every request).",
        "warmup_runs: 1 = the discarded W sample (GB's configuration, the first sample after the llama.cpp stop); every "
        "measured sample is the first request to a freshly started server process.",
        "llama.cpp (the Hermes backend) was stopped %s to %s PDT (%d s) and restored healthy; the Hermes "
        "shadowfax-ensure-active-stack watchdog failed every 5 min in the window and restarted nothing "
        "(evidence/f42v2-host-journal.txt)." % (LLAMA_DOWN, LLAMA_UP[:19], WINDOW_S)]}
path = os.path.join(HERE, "f42v2-prefill-path-24tok.json")
json.dump(rec, open(path, "w"), indent=2)
open(path, "a").write("\n")
print("wrote", path, rec["outcome"], "%+.2f%%" % (100 * d))
for n in rec["notes"]:
    print(" -", n)
