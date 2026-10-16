#!/usr/bin/env python3
"""F42 fix: upstream-format manifest (docs/experiments/manifest.example.json, validated by
c/experiment_manifest.py) and per-arm evidence tarballs from the run_f42fix.sh run on shadowfax
(pr1/v5 tip 4feb6d48 with the routing-books fix c7f79815 vs the old pr1 tip 93b74183, both + the
same logging-only probe; CPU-union reference UX on the fixed tree; plain-build check FXn).

usage: make_manifests_f42fix.py RAW_DIR
  RAW_DIR = the extracted f42fix.tgz's f42fix/ directory (FX/ UX/ OX/ FXn/ W/ meta/ run.log ...).
Reads stats.json next to this script (analyze_f42fix_local.py RAW_DIR stats.json). Writes, next to it:
  evidence/f42fix-arm-<X>.tgz       deterministic per-arm tarball of RAW_DIR/<X>/ (X = FX, UX, OX, FXn, W)
  f42fix-routing-books-24tok.json   the one-variable record OX -> FX (engine_commit), written only if every
                                    OX and FX sample's text is byte-identical (the reference answer)
"""
import glob, gzip, hashlib, io, json, os, re, statistics as st, sys, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(sys.argv[1])
STATS = json.load(open(os.path.join(HERE, "stats.json")))
FIX = "4feb6d487973e064abc53486f379032aa402efc6"
OLD = "93b741839677ec457868fd71ab668cc1249b9a22"
FIX_COMMIT = "c7f7981508113c80bee015de6d29fed4e78524bc"
FIXP_BIN = "98ecc0cbe898fadb0bd55060a7e5c8544ae817e096e45b0df166f90fac188b8f"
FIXP_DIFF = "d72e7c57c642aa619146cc141b0672db78a376ac7f3fc713971b7a143b139342"
OLDP_BIN = "f5ef189a1e4b319e317a95688aa2da34d5dba62c199e15e740b7389d52f57993"
OLDP_DIFF = "b500a731e16e913ce3c37ff165fa834aaa4d9327cd49c81d11f7645e0e3979a1"
FIXN_BIN = "a17d0a0269396f3570c35c0514df4b227bc82530e8b8f73ffb152b918cd5ee1c"
PATCH = "2562ff545038b4a16da60f942303a6b4374413f2c839cd9a940d667ad7c1e78a"
PROMPT_SHA = "e13a02a377c208b8eac583aa2a52ef82ab5ebff96a0c2e9bbf8c08954e97ca4d"
REF_TEXT = "32534c06abd640d59d2eb39ab6609051f63e9876f533aad1d75dd5cd775835ba"
EXPECT = {"W": (FIX, FIXP_BIN, FIXP_DIFF), "FX": (FIX, FIXP_BIN, FIXP_DIFF), "UX": (FIX, FIXP_BIN, FIXP_DIFF),
          "OX": (OLD, OLDP_BIN, OLDP_DIFF), "FXn": (FIX, FIXN_BIN, None)}

RUNLOG = open(os.path.join(RAW, "run.log")).read()
LLAMA_DOWN = re.search(r"\[(\S+ \S+)\] stopping llama-server-active", RUNLOG).group(1)
LLAMA_UP = re.search(r"restore: llama health \S+ \(ok=1\) at (\S+?);", RUNLOG).group(1)
WINDOW_S = int(re.search(r"llama-down window (\d+) s", RUNLOG).group(1))
MEMAVAIL = int(re.search(r"preflight ok: MemAvailable=(\d+) MiB", RUNLOG).group(1))
TREES = open(os.path.join(RAW, "meta", "trees.txt")).read()
for sha, b in ((FIX, FIXP_BIN), (OLD, OLDP_BIN), (FIX, FIXN_BIN)):
    assert "head=%s" % sha in TREES and "binary_sha256=%s" % b in TREES
assert "diff_sha256=%s" % FIXP_DIFF in TREES and "diff_sha256=%s" % OLDP_DIFF in TREES
assert "ABORT" not in RUNLOG
SKIPPED = re.findall(r"budget: skipping (\S+ s\d+)", RUNLOG)
NOTES_LOG = re.findall(r"NOTE: (.*)", RUNLOG)


def kv(path):
    return dict(l.rstrip("\n").split("=", 1) for l in open(path) if "=" in l)


def samples(arm):
    out = []
    head, binary, diff = EXPECT[arm]
    for d in sorted(glob.glob(os.path.join(RAW, arm, "s*")), key=lambda p: int(p.rsplit("s", 1)[1])):
        meta = kv(os.path.join(d, "sample.txt"))
        rec = json.loads(next(l for l in open(os.path.join(d, "bench.jsonl")) if l.startswith("{")))
        assert hashlib.sha256(rec["text"].encode()).hexdigest() == rec["text_sha256"]
        assert rec["prompt_sha256"] == PROMPT_SHA and meta["client_rc"] == "0"
        assert meta["tree_head"] == head and meta["binary_sha256"] == binary, (arm, d)
        if diff:
            assert meta["tree_diff_sha256"] == diff, (arm, d)
        else:
            assert meta["tree_dirty_lines"] == "0", (arm, d)
        log = open(os.path.join(d, "serve.log"), errors="replace").read()
        tr = re.findall(r"^v4_dec_trace ord=25 .*$", log, re.M)
        out.append({"k": int(meta["sample"]), "seq": int(meta["seq"]), "meta": meta, "rec": rec,
                    "env": dict(x.split("=", 1) for x in meta["server_env"].split()),
                    "trace25": {k: float(v) for k, v in re.findall(r"(\w+)=(-?[\d.]+)", tr[0])} if tr else None,
                    "reserve": re.findall(r"mirror-reserve=(\d+)MiB", log),
                    "bank": len(re.findall(r"moe-batch=on bank=", log)),
                    "trace_start": re.search(r"^v4_dec_trace_start .*$", log, re.M).group(0)
                    if "v4_dec_trace_start" in log else None})
    return out


S = {a: samples(a) for a in ("FX", "UX", "OX", "FXn", "W")}
print("samples per arm:", {a: len(v) for a, v in S.items()}, "skipped:", SKIPPED, "run-log NOTEs:", NOTES_LOG)
assert len(S["FX"]) == len(S["OX"]) == len(S["UX"]) == 5 and len(S["W"]) == 1
assert all(x["reserve"] == ["3100"] for v in S.values() for x in v)
assert all(x["bank"] == 1 for a in ("FX", "OX", "FXn", "W") for x in S[a]) and all(x["bank"] == 0 for x in S["UX"])
assert all(re.search(r"bank=0 union=[1-9]\d* rows=0", x["trace_start"]) for x in S["UX"])
assert all(re.search(r"bank=[1-9]\d* union=0 rows=0", x["trace_start"]) for a in ("FX", "OX", "W") for x in S[a])

# ---------------------------------------------------------------- text identity (the gate for writing a record)
TEXTS = {a: sorted({x["rec"]["text_sha256"] for x in v}) for a, v in S.items()}
IDENTICAL = TEXTS["FX"] == TEXTS["OX"] == [REF_TEXT]
ALL_IDENTICAL = all(TEXTS[a] == [REF_TEXT] for a in ("FX", "OX", "UX", "FXn"))
print("texts:", {a: [t[:12] for t in v] for a, v in TEXTS.items()}, "identical OX/FX:", IDENTICAL, "all arms:", ALL_IDENTICAL)

# ---------------------------------------------------------------- environment: OX and FX identical; UX adds the MIN
COMMON_KEYS = ("COLI_CUDA", "COLI_GPU", "DSV4_CUDA_DEVICE", "CUDA_DENSE", "COLI_CUDA_PIPE", "COLI_CUDA_ATTN_BATCH",
               "COLI_CUDA_MOE_BATCH", "V4_MOE_REFILL_GROUP", "V4_LOADER_LANES", "V4_PREFIX_CKPT_DISK",
               "DSV4_DECODE_PROF", "V4_DEC_TRACE")
ref_env = S["FX"][0]["env"]
for a in ("FX", "OX", "FXn", "UX"):
    for x in S[a]:
        e = dict(x["env"])
        assert e.pop("COLI_CUDA_MOE_BATCH_MIN", None) == ("100000" if a == "UX" else None), (a, x["k"])
        assert e == ref_env, (a, x["k"])
COMMON_CFG = {k: ref_env[k] for k in COMMON_KEYS}
COMMON_CFG.update({
    "COLI_CUDA_MOE_BATCH_MIN": "unset (default 256: the 1,992-token prompt takes the GPU bank)",
    "COLI_CUDA_MOE_DOUBLE": "unset (default 0)",
    "DSV4_CUDA_VRAM_RESERVE_MB": "unset (default: bank-sized, 3100 MiB logged in every sample)",
    "ram_gib": "40", "ctx": "8192", "temperature": "0 (greedy)",
    "probe": "final2/probe/f42fix_probe_patch.py (sha256 %s), logging only, the same +/- lines on both trees" % PATCH})
ARM_CFG = {"OX": {"engine_commit": OLD}, "FX": {"engine_commit": FIX}}
ENGINE = {"OX": {"commit": OLD, "binary_sha256": OLDP_BIN, "probe_diff_sha256": OLDP_DIFF,
                 "note": "old pr1 tip (PR #1 before the routing-books fix)"},
          "FX": {"commit": FIX, "binary_sha256": FIXP_BIN, "probe_diff_sha256": FIXP_DIFF,
                 "note": "pr1/v5 tip; differs from %s in the engine sources by exactly %s" % (OLD[:8], FIX_COMMIT[:8])}}


# ---------------------------------------------------------------- evidence tarballs (deterministic)
def evidence(arm):
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=buf, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as t:
        for f in sorted(glob.glob(os.path.join(RAW, arm, "s*", "*"))):
            info = t.gettarinfo(f, arcname=os.path.join("f42fix", os.path.relpath(f, RAW)))
            info.mtime = 0; info.uid = info.gid = 0; info.uname = info.gname = ""
            with open(f, "rb") as fh:
                t.addfile(info, fh)
    data = buf.getvalue()
    path = os.path.join(HERE, "evidence", "f42fix-arm-%s.tgz" % arm)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").write(data)
    return {"uri": os.path.relpath(path, HERE), "sha256": hashlib.sha256(data).hexdigest(),
            "contains": "f42fix/%s/s*/ (serve.log with the probe's v4_dec_trace_start / v4_dec_res / v4_dec_trace / "
                        "v4_dec_layer lines and decprof, bench.jsonl with the full text and per-chunk times, sample.txt, "
                        "engine_environ.txt, procs.txt, nvsmi.csv at 1 s, gpu_*.csv, mode.txt)" % arm,
            "full_run": "f42fix.tgz (sha256 in f42fix.tgz.sha256): all arms, meta/ (host, trees, scripts, clients, probe "
                        "patch and diffs, build logs, llama before/after), run.log, summary.json, summary.txt"}


EVID = {a: evidence(a) for a in ("FX", "UX", "OX", "FXn", "W")}
for a, e in EVID.items():
    print("evidence", e["uri"], e["sha256"])

if not IDENTICAL:
    print("texts differ between OX and FX: no manifest written (token-exact quality gate)")
    sys.exit(0)

# ---------------------------------------------------------------- manifest
HOST = open(os.path.join(RAW, "meta", "host.txt")).read()
KERNEL = re.search(r"^Linux shadowfax (\S+)", HOST, re.M).group(1)
HW = {
    "cpu": "AMD Ryzen 9 9950X3D, 16 cores / 32 threads, 1 socket, governor powersave (amd-pstate), max 5.76 GHz "
           "(engine: 16 OpenMP compute threads + 3 expert-loader lanes)",
    "ram": "62 GiB DDR5 (8 GiB swap); engine --ram 40 (59 routed-expert RAM slots per layer = 31.6 GiB cache, 6.2 GiB "
           "dense resident); MemAvailable %.1f GiB (%d MiB) at preflight with llama.cpp still loaded" % (MEMAVAIL / 1024, MEMAVAIL),
    "storage": "WD_BLACK SN850X 4 TB NVMe (PCIe 4.0), ext4 root filesystem, model read with O_DIRECT "
               "(v4_ssd_io mode=direct-aligned); OS page cache not dropped between samples; one discarded warm-up "
               "sample first",
    "os": "Ubuntu 24.04.4 LTS, kernel %s, gcc 13.3.0, Python 3.12.3, CUDA toolkit 12.0.140 (generic build "
          "CUDA_ARCH=sm_86 NO_TC=1), NVIDIA driver 595.84; 2x NVIDIA GeForce RTX 3090 24 GB on PCIe x8, NVLink bridge; "
          "GPU0 only (COLI_GPU=0 DSV4_CUDA_DEVICE=0), GPU1 idle" % KERNEL}
MODEL = ("DeepSeek V4 Flash REAP 150B (puwaer/DeepSeek-V4-Flash-0731-reap-150b, 132 routed experts, fp4 experts / "
         "fp8 dense), colibri safetensors layout, 79 GiB on disk")
BUILD = ("cd c && make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 "
         "NO_TC=1 -j8 on the engine_commit + final2/probe/f42fix_probe_patch.py (final2/build_r9f.sh; 0 warnings)")
ORDER = " ".join("%s%d" % (x[0], x[1]) for x in sorted(((a, x["k"], x["seq"]) for a, v in S.items() for x in v),
                                                         key=lambda t: t[2]))
COMMAND = ("build: %s; run: python3 c/coli serve --model <REAP dir> --ram 40 --ctx 8192 --host 127.0.0.1 --port 8090, "
           "server environment exactly PATH/HOME/USER/LOGNAME/LANG + the config's env keys (env -i); one freshly started "
           "server process per sample; run order %s (W discarded); client bench_f42v2.py sends bench_pr1.py REQUESTS[0] "
           "byte for byte (the 1,992-token prompt ending 'Summarize the above in one sentence.', max_tokens 100), "
           "streamed, greedy; the answer is 25 content chunks then EOS; tok_s = 24 / (last chunk time - first chunk "
           "time) (bench_common decode_tok_s); baseline arm OX, trial arm FX" % (BUILD, ORDER))
QUALITY = ("greedy: the full output text has the same sha256 (%s, the reference answer of the final PR #1 run) in all 5 "
           "OX and all 5 FX samples%s; plus per sample the engine's real environment (/proc/<pid>/environ), tree SHA and "
           "probe diff sha256, binary sha256, client exit code, exactly one 'mirror-reserve=3100MiB' line, and the "
           "prefill path counted by the probe (v4_dec_trace_start: bank>0 union=0 and the 'moe-batch=on bank=' and "
           "refill lines in both arms)"
           % (REF_TEXT[:16], " (and in the 5 UX CPU-union samples, the %d FXn samples on the plain fixed binary and the "
                             "warm-up)" % len(S["FXn"]) if ALL_IDENTICAL else ""))
D = STATS["decomposition"]
W = STATS["windows"]
R = STATS["residency_summary"]
C = STATS["compare"]
SP = STATS["speed"]


def m(x):
    return x["median"] if x else None


def run_block(arm):
    xs = [x["rec"]["decode_tok_s"] for x in S[arm]]
    return {"config": dict(COMMON_CFG, **ARM_CFG[arm]),
            "engine": ENGINE[arm],
            "samples": {"tok_s": xs, "tok_s_mean": round(st.mean(xs), 4), "tok_s_stdev": round(st.stdev(xs), 4),
                        "sample_dirs": ["%s/s%d" % (arm, x["k"]) for x in S[arm]],
                        "run_order_seq": [x["seq"] for x in S[arm]],
                        "ttft_s": [x["rec"]["ttft_s"] for x in S[arm]],
                        "text_sha256": [x["rec"]["text_sha256"] for x in S[arm]],
                        "decode_trace_fwd1_25": [{k: x["trace25"][k] for k in ("t", "total_s", "fetch_wait_s", "attach_s",
                                                                                "req", "hits", "misses", "wait_s", "disk_s")}
                                                 for x in S[arm]],
                        "prefill_trace_start": [x["trace_start"].split(" store_total")[0] for x in S[arm]]},
            "median_tok_s": st.median(xs),
            "quality": {"method": QUALITY, "passed": True, "problems": []},
            "evidence": EVID[arm]}


b, t = run_block("OX"), run_block("FX")
d = t["median_tok_s"] / b["median_tok_s"] - 1
ux = [x["rec"]["decode_tok_s"] for x in S["UX"]]
fxn = [x["rec"]["decode_tok_s"] for x in S["FXn"]]


# claims made in the notes, checked against stats.json
assert all(v["differ_from_FX_s1"] == 0 and v["token_layers"] == 1075 for k, v in STATS["decode_routing"].items()
           if k.split("/")[0] in ("FX", "OX"))
assert all(p["req"] == 4468 and p["misses"] == 4468 and p["bytes"] == 59734228992
           for a in ("FX", "OX") for p in STATS["prefill_store"][a])


def _welch(a, b):
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    return round((st.mean(a) - st.mean(b)) / (va + vb) ** 0.5, 2)


_late = {a: [x["rec"]["decode_tok_s"] for x in S[a] if x["k"] != 1] for a in ("FX", "OX", "UX")}
ROB = {"s1": {a: next(x["rec"]["decode_tok_s"] for x in S[a] if x["k"] == 1) for a in ("FX", "OX", "UX")},
       "mean": {a: round(st.mean(v), 4) for a, v in _late.items()},
       "t": {"FX-OX": _welch(_late["FX"], _late["OX"]), "FX-UX": _welch(_late["FX"], _late["UX"])}}
ROB["recovered"] = (ROB["mean"]["FX"] - ROB["mean"]["OX"]) / (ROB["mean"]["UX"] - ROB["mean"]["OX"])


def wl(w, k, scale=1.0):
    r = W[w][k]
    return (scale * r["OX"]["mean"], scale * r["FX"]["mean"], scale * r["UX"]["mean"], r["recovered"])


rec = {
    "version": 1,
    "hypothesis": ("F42's cause is the RAM expert store's books after a GPU-bank prefill: the bank refill looked each routed "
                   "expert up once per layer per prompt (~104 counted requests per layer vs ~1065 for the CPU union), so "
                   "decode started with flat usage, id-ranked pins, 7-9 of 16 pins active and first-read recency, and "
                   "missed +57%% more experts in RAM in its first ~25 tokens. Commit %s makes the bank keep the CPU union's "
                   "books (uncounted reads, a recency touch of each chunk's union before the refill, each member counted "
                   "once after the chunk's GPU MoE). On the identical answer, decode after the bank prefill should then "
                   "miss fewer experts early and be faster than on the unfixed tree, closing most of the gap to the CPU "
                   "union, with prefill time unchanged" % FIX_COMMIT[:8]),
    "commit": FIX, "model": MODEL, "command": COMMAND, "prompt_hash": PROMPT_SHA, "hardware": HW,
    "warmup_runs": 1,
    "changed_variables": ["engine_commit"], "baseline": b, "trial": t,
    "outcome": "improvement" if d > 0.01 else "regression" if d < -0.01 else "no-change",
    "notes": [
        "Trial vs baseline median: %+.2f%% (FX %.4f vs OX %.4f tok/s; means %.4f vs %.4f; Welch t = %s on tok/s). CPU-union "
        "reference UX (the fixed tree with COLI_CUDA_MOE_BATCH_MIN=100000, same 3100 MiB reserve, not part of this record): "
        "median %.4f tok/s, so OX/UX = %.4f (F42 on this code line), FX/UX = %.4f; the fix recovers %.0f%% of the OX-UX gap "
        "(means; %.0f%% by medians)."
        % (100 * d, t["median_tok_s"], b["median_tok_s"], t["samples"]["tok_s_mean"], b["samples"]["tok_s_mean"],
           C["FX/OX"]["welch_tok_s"]["t"], st.median(ux), C["OX/UX"]["median_ratio"], C["FX/UX"]["median_ratio"],
           100 * STATS["recovered_tok_s"]["means"], 100 * STATS["recovered_tok_s"]["medians"]),
        "Robustness: the first round of samples ran slower in every arm (FX s1 %.4f, OX s1 %.4f, UX s1 %.4f tok/s; seq 2-4 "
        "right after the warm-up). Without each arm's s1: FX %.4f / OX %.4f / UX %.4f tok/s (means), FX/OX %.4f (Welch t = "
        "%.1f), FX/UX %.4f (t = %.1f), recovered %.0f%% of the OX-UX gap."
        % (ROB["s1"]["FX"], ROB["s1"]["OX"], ROB["s1"]["UX"], ROB["mean"]["FX"], ROB["mean"]["OX"], ROB["mean"]["UX"],
           ROB["mean"]["FX"] / ROB["mean"]["OX"], ROB["t"]["FX-OX"], ROB["mean"]["FX"] / ROB["mean"]["UX"], ROB["t"]["FX-UX"],
           100 * ROB["recovered"]),
        "Decomposition, means over 5+5+5 samples (decode forwards 1-25 on the server), OX / FX / UX: wall %.3f / %.3f / %.3f s, "
        "decode MoE %.3f / %.3f / %.3f s, fetch_wait %.3f / %.3f / %.3f s (recovered %.0f%%), attach %.3f / %.3f / %.3f s, "
        "RAM-store misses %.0f / %.0f / %.0f (recovered %.0f%%), %.2f / %.2f / %.2f GiB read from SSD, non-MoE %.3f / %.3f / "
        "%.3f s."
        % (D["server_fwd1_25_s"]["OX"], D["server_fwd1_25_s"]["FX"], D["server_fwd1_25_s"]["UX"],
           D["dec_moe_total_s"]["OX"], D["dec_moe_total_s"]["FX"], D["dec_moe_total_s"]["UX"],
           D["fetch_wait_s"]["OX"], D["fetch_wait_s"]["FX"], D["fetch_wait_s"]["UX"], 100 * D["fetch_wait_s"]["recovered"],
           D["attach_s"]["OX"], D["attach_s"]["FX"], D["attach_s"]["UX"],
           D["store_misses"]["OX"], D["store_misses"]["FX"], D["store_misses"]["UX"], 100 * D["store_misses"]["recovered"],
           D["store_GiB_read"]["OX"], D["store_GiB_read"]["FX"], D["store_GiB_read"]["UX"],
           D["non_moe_s"]["OX"], D["non_moe_s"]["FX"], D["non_moe_s"]["UX"]),
        "Token windows (decode forward j produces content token j+1; means OX / FX / UX, share of the OX-UX gap recovered): "
        "misses fwd 1-5 %.0f / %.0f / %.0f (%s), 6-10 %.0f / %.0f / %.0f (%s), 11-24 %.0f / %.0f / %.0f (%s); fetch_wait ms "
        "fwd 1-5 %.0f / %.0f / %.0f (%s), 6-10 %.0f / %.0f / %.0f (%s), 11-24 %.0f / %.0f / %.0f (%s); client ms fwd 1-5 "
        "%.0f / %.0f / %.0f (%s), 6-10 %.0f / %.0f / %.0f (%s), 11-24 %.0f / %.0f / %.0f (%s)."
        % (wl("1-5", "misses") + wl("6-10", "misses") + wl("11-24", "misses") +
           wl("1-5", "fetch_wait_s", 1000) + wl("6-10", "fetch_wait_s", 1000) + wl("11-24", "fetch_wait_s", 1000) +
           wl("1-5", "client_ms") + wl("6-10", "client_ms") + wl("11-24", "client_ms")),
        "RAM tier at decode start (probe v4_dec_res, read under the store mutex; medians over samples), OX / FX / UX: layer "
        "requests %.0f / %.0f / %.0f per layer, active pins %.2f / %.2f / %.2f of 16 per layer, pin usage median %s / %s / %s, "
        "residents with usage <= 1 %.1f%% / %.1f%% / %.1f%%, layers whose pins are ids 0..15 %s / %s / %s, pins among the "
        "layer's 16 most-used residents %.0f%% / %.0f%% / %.0f%% (usage ties count; every OX resident has usage 1); of the experts decode routes in forwards 1-25, %s / %s / %s "
        "resident at decode start (of %s / %s / %s); first forward %s / %s / %s of 258."
        % (m(R["OX"]["lreq_mean"]), m(R["FX"]["lreq_mean"]), m(R["UX"]["lreq_mean"]),
           m(R["OX"]["pinned_resident_mean"]), m(R["FX"]["pinned_resident_mean"]), m(R["UX"]["pinned_resident_mean"]),
           m(R["OX"]["pin_usage_median"]), m(R["FX"]["pin_usage_median"]), m(R["UX"]["pin_usage_median"]),
           100 * m(R["OX"]["resident_usage_le1_frac"]), 100 * m(R["FX"]["resident_usage_le1_frac"]),
           100 * m(R["UX"]["resident_usage_le1_frac"]),
           m(R["OX"]["layers_pins_ids_0_to_15"]), m(R["FX"]["layers_pins_ids_0_to_15"]), m(R["UX"]["layers_pins_ids_0_to_15"]),
           100 * m(R["OX"]["pins_are_top_usage_frac"]), 100 * m(R["FX"]["pins_are_top_usage_frac"]),
           100 * m(R["UX"]["pins_are_top_usage_frac"]),
           m(R["OX"]["decode_needed_resident_at_start"]), m(R["FX"]["decode_needed_resident_at_start"]),
           m(R["UX"]["decode_needed_resident_at_start"]),
           m(R["OX"]["decode_needed_distinct"]), m(R["FX"]["decode_needed_distinct"]), m(R["UX"]["decode_needed_distinct"]),
           m(R["OX"]["first_fwd_resident_at_start"]), m(R["FX"]["first_fwd_resident_at_start"]),
           m(R["UX"]["first_fwd_resident_at_start"])),
        "What the fix leaves: the books now match the CPU union's, but residency does not fully (the commit's own caveat: "
        "the bank reads an expert once per layer per prompt, so a member the RAM cache evicted earlier in the layer is not "
        "read back as the CPU union reads it). Pins are ranked by usage yet only %.1f of 16 per layer are resident at decode "
        "start (UX 16), and of the first decode forward's 258 routed experts %s are in RAM (UX %s, OX %s): forwards 1-5 still "
        "miss %.0f more experts than UX (%.0f%% of OX's excess recovered) and carry most of the remaining time gap, while "
        "forwards 6-10 recover %.0f%% and forwards 11-24 run slightly faster than UX. Swapping routing and RAM sets, FX's "
        "decode routing against UX's decode-start RAM finds %.0f of its experts resident (its own RAM: %.0f): residency, not "
        "routing, is what is left."
        % (m(R["FX"]["pinned_resident_mean"]), m(R["FX"]["first_fwd_resident_at_start"]),
           m(R["UX"]["first_fwd_resident_at_start"]), m(R["OX"]["first_fwd_resident_at_start"]),
           W["1-5"]["misses"]["FX-UX_mean"], 100 * W["1-5"]["misses"]["recovered"], 100 * W["6-10"]["client_ms"]["recovered"],
           STATS["counterfactual_pairs"]["needed_resident FX-routing x UX-RAM"]["mean"],
           STATS["counterfactual_pairs"]["needed_resident FX-routing x FX-RAM"]["mean"]),
        "Prefill: TTFT median FX %.2f s vs OX %.2f s (ratio %.4f; means %+.2f s, Welch t = %s, not significant): the "
        "bookkeeping costs at most a few tenths of a second of an 82 s prefill%s. UX's TTFT is %.1f s (the CPU-union prefill PR #1 replaces)."
        % (m(SP["FX"]["ttft_s"]), m(SP["OX"]["ttft_s"]), C["FX/OX"]["ttft_median_ratio"], C["FX/OX"]["ttft_welch"]["diff"],
           C["FX/OX"]["ttft_welch"]["t"],
           "" if abs(C["FX/OX"]["ttft_median_ratio"] - 1) < 0.01 else " (NOTE: the ratio differs from 1 by >= 1%)",
           m(SP["UX"]["ttft_s"])),
        "Strictly one variable: OX and FX run the same probe patch (identical +/- lines) with the same server environment "
        "(sample.txt server_env, engine_environ.txt); the engine sources of %s and %s differ by exactly %s's lines "
        "(checked by run_f42fix.sh's preflight: git diff of c/ without c/tests); the binaries differ only through that. "
        "The decode routing is identical in all 10 OX and FX samples (all 1,075 token-layers of forwards 1-25, probe "
        "v4_dec_layer), and so is the prefill's store traffic (4,468 reads, all misses, 59.7 GB): the fix changes only the "
        "store's books."
        % (OLD[:8], FIX[:8], FIX_COMMIT[:8]),
        "Probe check: FXn (FX's environment on the plain %s binary %s..., %d samples) median %.4f tok/s = %.4fx FX%s."
        % (FIX[:8], FIXN_BIN[:12], len(fxn), st.median(fxn), st.median(fxn) / t["median_tok_s"],
           ", identical text" if TEXTS["FXn"] == [REF_TEXT] else ", text DIFFERS"),
        "prompt_hash = sha256 of the prompt string of bench_pr1.py REQUESTS[0] (bench_f42v2.py checks it before every request).",
        "warmup_runs: 1 = the discarded W sample (FX's configuration, the first sample after the llama.cpp stop); every "
        "measured sample is the first request to a freshly started server process.",
        "llama.cpp (the Hermes backend) was stopped %s to %s PDT (%d s) and restored healthy (/health ok)."
        % (LLAMA_DOWN, LLAMA_UP[:19], WINDOW_S)]}
if SKIPPED:
    rec["notes"].append("Budget-skipped samples: %s." % ", ".join(SKIPPED))
if NOTES_LOG:
    rec["notes"].append("Run-log notes: %s." % "; ".join(NOTES_LOG))
path = os.path.join(HERE, "f42fix-routing-books-24tok.json")
json.dump(rec, open(path, "w"), indent=2)
open(path, "a").write("\n")
print("wrote", path, rec["outcome"], "%+.2f%%" % (100 * d))
for n in rec["notes"]:
    print(" -", n)
