#!/usr/bin/env python3
"""F42 (decode right after a long GPU-batch prefill): stats, decomposition and upstream-format manifests
(docs/experiments/manifest.example.json, validated by c/experiment_manifest.py) from the run_f42.sh run on
shadowfax (2026-10-16 01:46-02:30 PDT, pr2 tip d0e19de2 = the PR #1 code paths when no PR #2 switch is set).

usage: make_manifests_f42.py RAW_DIR
  RAW_DIR = the extracted f42.tgz (the directory holding GB/ CU/ GR/ meta/ run.log summary.json).
  Parsing uses the final_tools.py shipped inside the tarball (RAW_DIR/meta), the same code that wrote summary.json.
Also reads evidence/f42-host-journal.txt (capture_host_journal_f42.sh: per-sample serve-unit memory peaks).
Writes, next to this script:
  evidence/f42-arm-<X>.tgz   deterministic per-arm tarball of RAW_DIR/<X>/ (what a manifest's evidence names)
  f42-*.json                 the strictly one-variable manifests the evidence supports (identical texts)
  supplementary/*.json       a two-variable record (rejected by the validator by design)
  stats.json                 per-sample numbers, per-arm mean/stdev/median, the decomposition, the token profile
"""
import glob, gzip, hashlib, io, json, os, re, statistics as st, sys, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.join(RAW, "meta"))
import final_tools as ft      # noqa: E402  (the tarball's copy)
import bench_common as bc     # noqa: E402

PR2 = "d0e19de23c3baf314ee932347c42d5f6d49ef4fb"
BIN = "63a667bac98a95daffc42428f0a82a1fd5051588047b524d8f8b1c564a20b926"
ARMS = ("GB", "CU", "GR")
REQS = ("long_200", "short_200")
MIRROR_MIB = ft.MIRROR_MIB
LONG, SHORT = bc.long_prompt(), bc.load_short_prompt()
PROMPT_HASH = hashlib.sha256(json.dumps([LONG, SHORT]).encode()).hexdigest()
assert PROMPT_HASH == "b4f45ed8908d379e301640f75f59320b5bc60800845d38c57dcf2da29323639d", PROMPT_HASH

RUNLOG = open(os.path.join(RAW, "run.log")).read()
LLAMA_DOWN = re.search(r"\[(\S+ \S+)\] stopping llama-server-active", RUNLOG).group(1)
LLAMA_UP = re.search(r"restore: llama health \S+ \(ok=1\) at (\S+?);", RUNLOG).group(1)
WINDOW_S = int(re.search(r"llama-down window (\d+) s", RUNLOG).group(1))
MEMAVAIL = int(re.search(r"preflight ok: MemAvailable=(\d+) MiB", RUNLOG).group(1))
TREES = open(os.path.join(RAW, "meta", "trees.txt")).read()
assert "head=%s" % PR2 in TREES and "binary_sha256=%s" % BIN in TREES and "status_lines=0" in TREES, TREES

# serve-unit memory peaks (systemd accounting of the unit's cgroup, which counts the page cache it faulted in)
JOURNAL = open(os.path.join(HERE, "evidence", "f42-host-journal.txt")).read().splitlines()
PEAKS, cur = {}, None
for l in JOURNAL:
    m = re.search(r"Started colibri-final2-f42-serve\.service .*/final2/f42/(\w+/s\d)/serve\.log", l)
    if m:
        cur = m.group(1); continue
    m = re.search(r"colibri-final2-f42-serve\.service: Consumed .*?, ([\d.]+)G memory peak", l)
    if m and cur:
        PEAKS[cur] = float(m.group(1)); cur = None


def r4(x):
    return round(x, 4) if isinstance(x, float) else x


def stat(xs):
    xs = [x for x in xs if x is not None]
    return {"mean": r4(st.mean(xs)), "stdev": r4(st.stdev(xs)) if len(xs) > 1 else None,
            "median": r4(st.median(xs)), "min": min(xs), "max": max(xs), "n": len(xs)}


# ---------------------------------------------------------------- per sample
S = {}
for arm in ARMS:
    for s in ft.samples(RAW, arm):
        x = ft.f42_sample(s, arm)
        assert not x["problems"], (arm, s["k"], x["problems"])
        assert x["tree_head"] == PR2 and x["binary_sha256"] == BIN
        assert s["meta"].get("tree_dirty_lines") == "0" and s["meta"].get("client_rc") == "0"
        for r in s["rows"]:
            assert hashlib.sha256(r["text"].encode()).hexdigest() == r["text_sha256"]
        turns = ft.f42_turns(s["log"])
        prof = ft.decprof(s["log"], [int(t["prompt"]) for t in turns])
        x["_prof"], x["_rows"] = prof, {r["name"]: r for r in s["rows"]}
        x["_vram"] = [(t, m) for t, i, m in ft.nvsmi(s["dir"]) if i == 0]
        x["_meta"] = s["meta"]
        S.setdefault(arm, []).append(x)
assert all(len(S[a]) == 3 for a in ARMS)


def q(x, req):
    return x["requests"][req]


def sample_record(arm, x):
    rec = {"dir": "%s/s%d" % (arm, x["k"]), "run_order_seq": int(x["seq"]), "wall_s": int(x["wall_s"]),
           "serve_unit_memory_peak_gib": PEAKS.get("%s/s%d" % (arm, x["k"])), "requests": {}}
    for req in REQS:
        qq = q(x, req); c = qq["client"]; d = qq["dec_moe"]; p = qq["decprof"]
        n = c["n"]
        rec["requests"][req] = {
            "prompt_tokens": int(qq["prompt_tokens"]), "completion_tokens": int(qq["completion"]),
            "ttft_s": c["ttft_s"], "decode_tok_s": c["rate_all"], "decode_tok_s_first24": c["rate_first24"],
            "decode_tok_s_after24": c["rate_after24"], "client_decode_s": round(c["sec_per_tok"] * (n - 1), 3),
            "engine_turn_tok_s": qq["turn"]["tok_s"], "engine_turn_hit_pct_incl_prefill": qq["turn"]["hit"],
            "dec_moe": d,
            "dec_moe_other_s": round(d["total_s"] - d["fetch_wait_s"] - d["attach_s"] - d["gpu_s"], 3),
            "attach_ms_per_upload": round(1000 * d["attach_s"] / d["uploads"], 4),
            "decprof_s": {"attn": p["attn_s"], "moe": p["moe_s"], "hc": p["hc_s"], "tokens": p["n"]},
            "vram0_mib": qq["vram0"], "mirrors_est": qq["mirrors_est"],
            "text_sha256": qq["text_sha256"]}
    return rec


SAMPLES = {arm: [sample_record(arm, x) for x in S[arm]] for arm in ARMS}

# ---------------------------------------------------------------- text identity
TEXT = {}
for req in REQS:
    shas = {arm: sorted({q(x, req)["text_sha256"] for x in S[arm]}) for arm in ARMS}
    assert all(len(v) == 1 for v in shas.values()), shas      # every arm is self-consistent
    ref = S["GB"][0]["_rows"][req]["chunk_text"]
    TEXT[req] = {"sha256_per_arm": {a: v[0] for a, v in shas.items()},
                 "distinct_across_arms": len({v[0] for v in shas.values()}),
                 "first_differing_chunk_vs_GB": {a: ft.first_diff(ref, S[a][0]["_rows"][req]["chunk_text"]) for a in ARMS},
                 "head_120_chars": {a: S[a][0]["_rows"][req]["text"][:120] for a in ARMS}}
assert TEXT["short_200"]["distinct_across_arms"] == 1
assert TEXT["long_200"]["sha256_per_arm"]["GB"] == TEXT["long_200"]["sha256_per_arm"]["GR"]
assert TEXT["long_200"]["first_differing_chunk_vs_GB"]["CU"] == 0

# ---------------------------------------------------------------- per-arm stats
def vals(arm, req, f, ks=None):
    return [f(r["requests"][req]) for x, r in zip(S[arm], SAMPLES[arm]) if ks is None or x["k"] in ks]


FIELDS = {
    "decode_tok_s": lambda r: r["decode_tok_s"], "decode_tok_s_first24": lambda r: r["decode_tok_s_first24"],
    "decode_tok_s_after24": lambda r: r["decode_tok_s_after24"], "ttft_s": lambda r: r["ttft_s"],
    "client_decode_s": lambda r: r["client_decode_s"],
    "dec_moe_total_s": lambda r: r["dec_moe"]["total_s"], "fetch_wait_s": lambda r: r["dec_moe"]["fetch_wait_s"],
    "attach_s": lambda r: r["dec_moe"]["attach_s"], "uploads": lambda r: r["dec_moe"]["uploads"],
    "gpu_s": lambda r: r["dec_moe"]["gpu_s"], "dec_moe_calls": lambda r: r["dec_moe"]["calls"],
    "dec_moe_other_s": lambda r: r["dec_moe_other_s"], "attach_ms_per_upload": lambda r: r["attach_ms_per_upload"],
    "decprof_attn_s": lambda r: r["decprof_s"]["attn"], "decprof_moe_s": lambda r: r["decprof_s"]["moe"],
    "mirrors_est": lambda r: r["mirrors_est"], "vram0_decode_max_mib": lambda r: r["vram0_mib"]["decode_max"],
}
ARMSTATS = {arm: {req: {k: stat(vals(arm, req, f)) for k, f in FIELDS.items()} for req in REQS} for arm in ARMS}

# ---------------------------------------------------------------- decomposition
COMP = ("client_decode_s", "dec_moe_total_s", "fetch_wait_s", "attach_s", "uploads", "gpu_s", "dec_moe_other_s",
        "decprof_attn_s", "decprof_moe_s")


def decomp(stat_key, ks=None):
    out = {}
    for req in REQS:
        m = {arm: {k: (st.median if stat_key == "median" else st.mean)(vals(arm, req, FIELDS[k], ks.get(arm) if ks else None))
                   for k in COMP} for arm in ARMS}
        out[req] = {pair: {k: round(m[a][k] - m[b][k], 3) for k in COMP}
                    for pair, (a, b) in (("GB-GR (reserve 3100->600 MiB, same text)", ("GB", "GR")),
                                         ("GR-CU (GPU bank vs CPU union prefill, same 600 MiB reserve; long: different text)", ("GR", "CU")),
                                         ("GB-CU (total)", ("GB", "CU")))}
        for pair in out[req]:
            d = out[req][pair]
            d["ms_per_token"] = round(1000 * d["client_decode_s"] / 199, 2)
            d["attach_s_predicted_from_uploads"] = round(d["uploads"] * st.mean(vals("GR", req, FIELDS["attach_ms_per_upload"])) / 1000, 3)
    return out


DECOMP = {"mean_all_samples": decomp("mean"), "median_all_samples": decomp("median"),
          "mean_GB_s1_excluded": decomp("mean", {"GB": (2, 3), "CU": (1, 2, 3), "GR": (1, 2, 3)}),
          "round3_only (warmest page cache)": decomp("mean", {a: (3,) for a in ARMS})}

# ---------------------------------------------------------------- token profile (front-loading)
WIN = [(1, 10), (11, 24), (25, 50), (51, 100), (101, 199)]


def window_ms(arm, req, lo, hi, ks=None, what="interval"):
    xs = []
    for x in S[arm]:
        if ks and x["k"] not in ks:
            continue
        if what == "interval":
            iv = q(x, req)["interval_ms"][lo - 1:hi]
        else:
            iv = [r[what] for r in x["_prof"][REQS.index(req)][lo - 1:hi]]
        xs.append(sum(iv) / len(iv))
    return st.mean(xs)


PROFILE = {}
for req in REQS:
    rows = []
    for lo, hi in WIN:
        row = {"tokens": "%d-%d" % (lo, hi)}
        for arm in ARMS:
            row[arm] = {"interval_ms": round(window_ms(arm, req, lo, hi), 1),
                        "decprof_moe_ms": round(window_ms(arm, req, lo, hi, what="moe"), 1),
                        "decprof_attn_ms": round(window_ms(arm, req, lo, hi, what="attn"), 1)}
        gb23 = window_ms("GB", req, lo, hi, ks=(2, 3))
        row["GB-GR_interval_ms"] = round(row["GB"]["interval_ms"] - row["GR"]["interval_ms"], 1)
        row["GB(s2,s3)-GR_interval_ms"] = round(gb23 - row["GR"]["interval_ms"], 1)
        row["GR-CU_interval_ms"] = round(row["GR"]["interval_ms"] - row["CU"]["interval_ms"], 1)
        rows.append(row)
    cum = []
    for n in (10, 24, 50, 100, 150, 199):
        c = {arm: st.mean(sum(q(x, req)["interval_ms"][:n]) / 1000 for x in S[arm]) for arm in ARMS}
        c23 = st.mean(sum(q(x, req)["interval_ms"][:n]) / 1000 for x in S["GB"] if x["k"] in (2, 3))
        cum.append({"first_n_intervals": n, **{a: round(v, 3) for a, v in c.items()},
                    "GB-GR": round(c["GB"] - c["GR"], 3), "GB(s2,s3)-GR": round(c23 - c["GR"], 3),
                    "GR-CU": round(c["GR"] - c["CU"], 3), "GB-CU": round(c["GB"] - c["CU"], 3)})
    PROFILE[req] = {"windows": rows, "cumulative_s": cum}

# ---------------------------------------------------------------- VRAM
VRAM = {}
for arm in ARMS:
    for x in S[arm]:
        r = x["_rows"]["long_200"]; first = r["t0_epoch"] + r["ttft_s"]
        dec = [(t, m) for t, m in x["_vram"] if t >= first]
        plateau = max(m for _, m in dec)
        before = [m for t, m in x["_vram"] if t < first][-1]
        t98 = next(t for t, m in dec if m >= before + 0.98 * (plateau - before)) - first
        VRAM["%s/s%d" % (arm, x["k"])] = {"ready_mib": q(x, "long_200")["vram0"]["ready"],
                                          "prefill_median_mib": q(x, "long_200")["vram0"]["prefill_median"],
                                          "decode_plateau_mib": plateau, "free_at_plateau_mib": 24576 - plateau,
                                          "seconds_first_token_to_98pct_plateau": round(t98, 1),
                                          "mirrors_est": q(x, "long_200")["mirrors_est"]}

# ---------------------------------------------------------------- verdict numbers (cited in the notes)
D = DECOMP["mean_GB_s1_excluded"]
L1, L2, L3 = (D["long_200"][k] for k in D["long_200"])
S1, S2, S3 = (D["short_200"][k] for k in D["short_200"])
med = {a: {r: ARMSTATS[a][r]["decode_tok_s"]["median"] for r in REQS} for a in ARMS}
VERDICT = (
    "H3 (mirror reserve) is supported and H1 (RAM residency after the GPU prefill) is not. "
    "GB is slower than GR on the identical long text (median %.4f vs %.4f tok/s, %.1f%%) and on the identical short text "
    "(%.4f vs %.4f, %.1f%%). The extra time is in attach: GB uploads +%d experts (long) and +%d (short) more, at the same "
    "~1.1 ms per upload. fetch_wait is lower in GB than in GR (%+.2f s long, %+.2f s short), not higher. The VRAM plateau is "
    "2236 MiB lower in GB (about 175 fewer 12.75 MiB mirrors, 1096 vs 1271). On the identical short text, GR is within "
    "0.4%% of CU (%+.3f s over 199 tokens; uploads +51, fetch_wait %+.2f s). So the prefill path itself costs about nothing "
    "once the reserve is equal. On the long text GR-CU is %+.2f s (%+.1f ms/token: attach %+.2f s from +%d uploads, fetch_wait "
    "%+.2f s). That comparison is between DIFFERENT token sequences: the CPU-union prefill picks a different first token "
    "('This' vs 'The'). Both mirror caches start empty and have the same capacity, so the upload difference comes from the "
    "different routing. Only the fetch_wait part, at most %.2f s (about %.1f ms/token, under 1%%), could be RAM residency. "
    "The shape does not match H1 either. The GR-CU gap is not front-loaded (+%.2f s in the first 24 intervals, then -0.5 to "
    "+2.4 s swings in tokens 25-150) and is mostly uploads. The GB-GR reserve gap is not front-loaded either. Without "
    "the cold GB s1 it is +%.1f ms/token in tokens 1-10, while the mirror cache is still filling (about 5 s to its "
    "plateau), and then a steady +5 to +10 ms/token from token 11 on (mean +%.1f ms/token over tokens 25-199). That is "
    "the shape of a capacity limit. Only GB s1, the first and coldest sample, adds a front-loaded +16 ms/token over "
    "tokens 1-24. The only front-loaded component anywhere is GR-CU on the short text: +%.2f s in its first 24 "
    "intervals and about 0 after that (fetch_wait +0.11 s, uploads +51). That is the largest possible residency "
    "transient, about 0.3%% of the answer." % (
        med["GB"]["long_200"], med["GR"]["long_200"], 100 * (med["GB"]["long_200"] / med["GR"]["long_200"] - 1),
        med["GB"]["short_200"], med["GR"]["short_200"], 100 * (med["GB"]["short_200"] / med["GR"]["short_200"] - 1),
        L1["uploads"], S1["uploads"], L1["fetch_wait_s"], S1["fetch_wait_s"], S2["client_decode_s"], S2["fetch_wait_s"],
        L2["client_decode_s"], L2["ms_per_token"], L2["attach_s"], L2["uploads"], L2["fetch_wait_s"], L2["fetch_wait_s"],
        1000 * L2["fetch_wait_s"] / 199, PROFILE["long_200"]["cumulative_s"][1]["GR-CU"],
        PROFILE["long_200"]["windows"][0]["GB(s2,s3)-GR_interval_ms"],
        1000 * (PROFILE["long_200"]["cumulative_s"][5]["GB(s2,s3)-GR"] - PROFILE["long_200"]["cumulative_s"][1]["GB(s2,s3)-GR"]) / 175,
        PROFILE["short_200"]["cumulative_s"][1]["GR-CU"]))

# ---------------------------------------------------------------- evidence tarballs
def evidence(arm):
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=buf, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as t:
        for f in sorted(glob.glob(os.path.join(RAW, arm, "s*", "*"))):
            info = t.gettarinfo(f, arcname=os.path.join("f42", os.path.relpath(f, RAW)))
            info.mtime = 0; info.uid = info.gid = 0; info.uname = info.gname = ""
            with open(f, "rb") as fh:
                t.addfile(info, fh)
    data = buf.getvalue()
    path = os.path.join(HERE, "evidence", "f42-arm-%s.tgz" % arm)
    open(path, "wb").write(data)
    return {"uri": os.path.relpath(path, HERE), "sha256": hashlib.sha256(data).hexdigest(),
            "contains": "f42/%s/s*/ (serve.log with v4_serve_turn / v4_dec_moe / decprof lines, bench.jsonl with full texts "
                        "and per-chunk times, sample.txt, engine_environ.txt, procs.txt, nvsmi.csv at 1 s, gpu_*.csv, mode.txt)" % arm,
            "full_run": "f42.tgz (sha256 in f42.tgz.sha256): all arms, meta/ (host, trees, scripts, clients, llama before/after), "
                        "run.log, summary.json, summary.txt"}


EVID = {arm: evidence(arm) for arm in ARMS}

# ---------------------------------------------------------------- manifests
HW = {
    "cpu": "AMD Ryzen 9 9950X3D, 16 cores / 32 threads, 1 socket, governor powersave (amd-pstate), max 5.76 GHz "
           "(engine: 16 OpenMP compute threads + 3 expert-loader lanes)",
    "ram": "62 GiB DDR5 (8 GiB swap, 3.0 GiB of it in use by other processes); engine --ram 40 (31.6 GiB routed-expert RAM "
           "cache, 6.2 GiB dense resident); MemAvailable %.1f GiB (%d MiB) at preflight with llama.cpp still loaded" % (MEMAVAIL / 1024, MEMAVAIL),
    "storage": "WD_BLACK SN850X 4 TB NVMe (PCIe 4.0), ext4 root filesystem (rw,relatime), model read with O_DIRECT "
               "(v4_ssd_io mode=direct-aligned); OS page cache not dropped between samples. It warmed through the run: "
               "the serve unit's systemd memory peak fell from 14.6G (GB s1, CU s1) to 7.9-8.0G in round 3 "
               "(evidence/f42-host-journal.txt, stats.json samples.*.serve_unit_memory_peak_gib)",
    "os": "Ubuntu 24.04.4 LTS, kernel 7.0.0-34-generic, gcc 13.3.0, Python 3.12.3, CUDA toolkit 12.0.140 (generic build "
          "CUDA_ARCH=sm_86 NO_TC=1), NVIDIA driver 595.84; 2x NVIDIA GeForce RTX 3090 24 GB on PCIe 4.0 x8 (x16-capable), "
          "NVLink NV4 bridge; GPU0 only (COLI_GPU=0 DSV4_CUDA_DEVICE=0), GPU1 idle"}
MODEL = ("DeepSeek V4 Flash REAP 150B (puwaer/DeepSeek-V4-Flash-0731-reap-150b, 132 routed experts, fp4 experts / "
         "fp8 dense), colibri safetensors layout, 79 GiB on disk")
BUILD = "cd c && make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j8"
COMMON_CFG = {"COLI_CUDA": "1", "COLI_GPU": "0", "DSV4_CUDA_DEVICE": "0", "CUDA_DENSE": "1", "COLI_CUDA_PIPE": "2",
              "COLI_CUDA_ATTN_BATCH": "1", "V4_MOE_REFILL_GROUP": "12", "V4_LOADER_LANES": "3", "V4_PREFIX_CKPT_DISK": "0",
              "V4_SERVE_STATS_LOG": "1", "DSV4_DECODE_PROF": "1", "COLI_CUDA_MOE_DOUBLE": "unset (default 0)",
              "ram_gib": "40", "ctx": "8192", "temperature": "0 (greedy)", "engine_commit": PR2}
ARM_CFG = {
    "GB": {"COLI_CUDA_MOE_BATCH": "1", "DSV4_CUDA_VRAM_RESERVE_MB": "unset"},
    "CU": {"COLI_CUDA_MOE_BATCH": "0", "DSV4_CUDA_VRAM_RESERVE_MB": "unset"},
    "GR": {"COLI_CUDA_MOE_BATCH": "1", "DSV4_CUDA_VRAM_RESERVE_MB": "600"}}
# what the engine actually kept free (not a config key: it follows from the two above)
EFFECTIVE_RESERVE = {
    "GB": "3100 MiB: the bank-sized default (serve.log: v4_gpu mirror-reserve=3100MiB (132-slot bank 1683MiB x1 + 600MiB headroom + 64 mirrors))",
    "CU": "600 MiB: no bank wanted, so the default is the headroom only (no mirror-reserve line)",
    "GR": "600 MiB: explicit DSV4_CUDA_VRAM_RESERVE_MB (no mirror-reserve line)"}
# the literal environment of each arm (sample.txt extra_env merged over COMMON_ENV) must match ARM_CFG
for arm in ARMS:
    for x in S[arm]:
        env = dict(kv.split("=", 1) for kv in x["_meta"]["server_env"].split())
        assert env.get("COLI_CUDA_MOE_BATCH") == ARM_CFG[arm]["COLI_CUDA_MOE_BATCH"], (arm, env)
        assert (env.get("DSV4_CUDA_VRAM_RESERVE_MB") == "600") == (arm == "GR"), (arm, env)
        assert all(env.get(k) == v for k, v in COMMON_CFG.items() if k.isupper() and not v.startswith("unset")), (arm, env)
        assert "COLI_CUDA_MOE_DOUBLE" not in env


def run_cmd(base, trial, req):
    return ("build: %s; run: python3 c/coli serve --model <REAP dir> --ram 40 --ctx 8192 --host 127.0.0.1 --port 8090, "
            "server environment exactly PATH/HOME/USER/LOGNAME/LANG + the config's env keys (env -i); one freshly started "
            "server process per sample; 3 samples per arm (GB, CU, GR) interleaved with the order rotated per round "
            "(GB CU GR / CU GR GB / GR GB CU); client bench_f42.py sends long_200 (bench_pr1.py's 1,992-token filler + a "
            "closing question asking for a long answer, 2041 prompt tokens, max_tokens 200) and then short_200 "
            "(bench_decode.py prompt 0, 33 tokens, max_tokens 200) to the same server, streamed, greedy; tok_s = %s decode "
            "rate (chunks - 1) / (last chunk time - first chunk time) = 199 / decode seconds; baseline arm %s, trial arm %s"
            % (BUILD, req, base, trial))


def run_block(arm, req, quality_text):
    xs = [r["requests"][req]["decode_tok_s"] for r in SAMPLES[arm]]
    return {"config": dict(COMMON_CFG, **ARM_CFG[arm]),
            "engine": {"commit": PR2, "binary_sha256": BIN},
            "effective_mirror_reserve": EFFECTIVE_RESERVE[arm],
            "samples": {"tok_s": xs, "tok_s_mean": r4(st.mean(xs)), "tok_s_stdev": r4(st.stdev(xs)),
                        "sample_dirs": [r["dir"] for r in SAMPLES[arm]],
                        "run_order_seq": [r["run_order_seq"] for r in SAMPLES[arm]],
                        "text_sha256": [r["requests"][req]["text_sha256"] for r in SAMPLES[arm]],
                        "dec_moe": [r["requests"][req]["dec_moe"] for r in SAMPLES[arm]],
                        "mirrors_est": [r["requests"][req]["mirrors_est"] for r in SAMPLES[arm]],
                        "vram0_decode_max_mib": [r["requests"][req]["vram0_mib"]["decode_max"] for r in SAMPLES[arm]],
                        "serve_unit_memory_peak_gib": [r["serve_unit_memory_peak_gib"] for r in SAMPLES[arm]]},
            "median_tok_s": st.median(xs),
            "quality": {"method": quality_text, "passed": True, "problems": []},
            "evidence": EVID[arm]}


def outcome(b, t):
    d = t / b - 1
    return "improvement" if d > 0.01 else "regression" if d < -0.01 else "no-change"


def manifest(name, var, base, trial, req, hypothesis, notes, two_var=None):
    ql = ("greedy: the full output text of %s has the same sha256 (%s) in all 3 samples of this arm and of the other arm "
          "of this record; plus the engine's real environment (/proc/<pid>/environ), tree SHA, binary sha256, client exit "
          "code, the mode lines of serve.log (moe-batch=on bank=132-experts / mirror-reserve) and 200 chunks with "
          "finish_reason length checked per sample (final_tools.py f42_sample: no problems)" % (req, TEXT[req]["sha256_per_arm"][base][:16]))
    assert TEXT[req]["sha256_per_arm"][base] == TEXT[req]["sha256_per_arm"][trial]
    b, t = run_block(base, req, ql), run_block(trial, req, ql)
    rec = {"version": 1, "hypothesis": hypothesis, "commit": PR2, "model": MODEL, "command": run_cmd(base, trial, req),
           "prompt_hash": PROMPT_HASH, "hardware": HW,
           "warmup_runs": 0 if req == "long_200" else 1,
           "changed_variables": two_var or [var], "baseline": b, "trial": t,
           "outcome": outcome(b["median_tok_s"], t["median_tok_s"]),
           "notes": notes + [
               "prompt_hash = sha256(json.dumps([long_200 prompt, short_200 prompt])) of bench_common.py (both requests of each sample).",
               "warmup_runs: %s" % ("0: long_200 is the first request to each freshly started server process" if req == "long_200"
                                     else "1: short_200 runs in the same server right after long_200 (2041-token prompt + 200-token answer), its one warm-up"),
               "Trial vs baseline median: %+.2f%%." % (100 * (t["median_tok_s"] / b["median_tok_s"] - 1)),
               "llama.cpp (the Hermes backend) was stopped %s to %s PDT (%d s) and restored healthy; the Hermes "
               "shadowfax-ensure-active-stack watchdog failed every 5 min in the window and restarted nothing "
               "(evidence/f42-host-journal.txt)." % (LLAMA_DOWN, LLAMA_UP[:19], WINDOW_S)]}
    path = os.path.join(HERE, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(rec, open(path, "w"), indent=2); open(path, "a").write("\n")
    return name


RES_NOTE = ("DSV4_CUDA_VRAM_RESERVE_MB is the free VRAM the decode mirror cache leaves on GPU0 while it grows. Unset with "
            "COLI_CUDA_MOE_BATCH=1, it is the bank-sized default (3100 MiB = 600 headroom + 1683 MiB 132-slot bank + 64 "
            "mirrors), so the next long prefill finds its bank. At 600 the cache took 2236 MiB more (VRAM plateau 23558 vs "
            "21322 MiB, about 1271 vs 1096 mirrors of 12.75 MiB). Decode then uploads fewer experts. Not measured here: the "
            "cost on a later long prefill, which would have to find bank VRAM the mirrors hold. short_200 is below "
            "COLI_CUDA_MOE_BATCH_MIN and never takes the bank.")
MANIFESTS = [
    manifest("f42-vram-reserve-long.json", "DSV4_CUDA_VRAM_RESERVE_MB", "GB", "GR", "long_200",
             "The slower decode right after a long GPU-batch prefill (F42) comes from the bank-sized decode-mirror "
             "reserve: lowering DSV4_CUDA_VRAM_RESERVE_MB from its 3100 MiB default to 600 MiB (CU's) gives the mirror "
             "cache ~175 more experts, fewer uploads and faster decode of the same answer",
             [RES_NOTE, "Strictly one-variable: GB and GR both run COLI_CUDA_MOE_BATCH=1 (GPU-bank prefill); GR adds "
                        "DSV4_CUDA_VRAM_RESERVE_MB=600. Texts byte-identical across all 6 samples."]),
    manifest("f42-vram-reserve-short.json", "DSV4_CUDA_VRAM_RESERVE_MB", "GB", "GR", "short_200",
             "The same reserve change also speeds up the next short-prompt decode in the same server, because the mirror "
             "cache stays ~175 experts larger",
             [RES_NOTE, "Strictly one-variable (GB vs GR as in f42-vram-reserve-long.json)."]),
    manifest("f42-moe-batch-short.json", "COLI_CUDA_MOE_BATCH", "CU", "GB", "short_200",
             "The PR #1 default COLI_CUDA_MOE_BATCH=1 (GPU-bank prefill of the preceding long prompt) slows the following "
             "short-prompt decode in the same server",
             ["One literal variable: CU sets COLI_CUDA_MOE_BATCH=0, GB keeps COMMON_ENV's 1, and neither sets "
              "DSV4_CUDA_VRAM_RESERVE_MB. The switch also changes the default mirror reserve (600 -> 3100 MiB), and the "
              "f42-vram-reserve-* records attribute about 88% of this gap to that reserve: GR (bank prefill, 600 MiB "
              "reserve) is within 0.4% of CU on this text. The long_200 record for this variable is not written: the "
              "two prefill paths give different long_200 texts from the first token on.", RES_NOTE]),
]
SUPP = manifest("supplementary/f42-prefill-path-short-two-variable.json", None, "CU", "GR", "short_200",
                "With the mirror reserve held at 600 MiB, the prefill path (GPU bank vs CPU union) of the preceding long "
                "prompt does not change the following short-prompt decode",
                ["Two literal variables (COLI_CUDA_MOE_BATCH 0 -> 1 and DSV4_CUDA_VRAM_RESERVE_MB unset -> 600), so the "
                 "validator rejects it by design. The effective reserve is 600 MiB in both: CU has no bank, so its default "
                 "is the 600 MiB headroom, and the VRAM plateaus are 23562 vs 23558 MiB. This is the cleanest test of the "
                 "prefill path: identical text, equal mirror capacity.", RES_NOTE],
                two_var=["COLI_CUDA_MOE_BATCH", "DSV4_CUDA_VRAM_RESERVE_MB"])

# ---------------------------------------------------------------- stats.json
ORIG = ("Original F42 observation (manifests/final/pr1, 2026-10-15): dev (arm A, COLI_CUDA_MOE_BATCH=1 set but the bank "
        "refused for REAP) vs the pr1 tip (arm B). Its GPU0 decode plateau on request 1 was 22064 MiB for dev vs 21262 MiB "
        "for pr1 (A/s2, B/s2 nvsmi.csv), about 63 more mirrors for dev. That reserve gap is about a third of the GB-vs-GR "
        "gap here (175 mirrors). So the reserve explains part of the original 24-token gap (2.054 vs 2.247 chunks/s, about "
        "1.0 s over 24 tokens), but not all of it. In this run the first 24 intervals of long_200 differ GB(s2,s3)-GR "
        "%+.2f s and GR-CU %+.2f s, on different texts for GR-CU. The original texts were identical but the prompt's "
        "closing question was different (24-token answer), so this run does not reproduce that window like for like." % (
            PROFILE["long_200"]["cumulative_s"][1]["GB(s2,s3)-GR"], PROFILE["long_200"]["cumulative_s"][1]["GR-CU"]))
DIVERGE = ("long_200 texts: GB and GR are byte-identical (sha %s) in all 6 samples, and CU is self-consistent in all 3 (sha %s). "
           "But CU differs from GB/GR from the first token: '%s...' vs '%s...'. Each arm is deterministic, so this is not a "
           "race. The CPU-union prefill (MoE on the CPU, expert_matmul_s 135.7 s) and the GPU-bank prefill (MoE on the GPU) "
           "are different kernels. Their final hidden state differs by rounding, and that flips a near-tie on this prompt's "
           "first token. docs/deepseek-v4.md at d0e19de2 says as much: different kernels are not bit-identical in general. "
           "But the PR #1 docs line 'Answers were identical to the CPU union in every run' holds only for the two benchmark "
           "prompts. short_200 is byte-identical in all 9 samples." % (
               TEXT["long_200"]["sha256_per_arm"]["GB"][:16], TEXT["long_200"]["sha256_per_arm"]["CU"][:16],
               TEXT["long_200"]["head_120_chars"]["CU"][:40], TEXT["long_200"]["head_120_chars"]["GB"][:40]))
STATS = {"experiment": "f42", "commit": PR2, "binary_sha256": BIN, "llama_down": LLAMA_DOWN, "llama_restored": LLAMA_UP,
         "gpu_window_s": WINDOW_S, "prompt_hash": PROMPT_HASH, "samples": SAMPLES, "arm_stats": ARMSTATS,
         "decomposition_s": DECOMP, "token_profile": PROFILE, "vram": VRAM, "text_identity": TEXT,
         "manifests": MANIFESTS, "supplementary": [SUPP],
         "notes": {"verdict": VERDICT, "text_divergence": DIVERGE, "original_f42": ORIG,
                   "cache_state": "GB s1 (the first sample of the window, serve-unit memory peak 14.6G) is the slow outlier "
                                  "of GB: long fetch_wait 15.39 s vs 14.32/14.40 s and dec_moe total 49.21 vs 47.71/47.72 s. "
                                  "Medians and the 'mean_GB_s1_excluded' decomposition are robust to it, and round 3 alone "
                                  "(memory peaks 7.9-8.0G) gives the same picture.",
                   "uploads_deterministic": "uploads per request are identical in all 3 samples of each arm (long "
                                            "18806/15725/16897, short 20189/18103/18154 for GB/CU/GR). They are a "
                                            "deterministic function of routing and mirror capacity, and attach costs the "
                                            "same 1.13 ms (long) / 1.07 ms (short) per upload in every arm (12.75 MiB at "
                                            "~11 GB/s, PCIe 4.0 x8)."}}
json.dump(STATS, open(os.path.join(HERE, "stats.json"), "w"), indent=1)
print(VERDICT)
print(DIVERGE)
print(ORIG)
print(json.dumps(DECOMP["mean_GB_s1_excluded"], indent=1))
print(MANIFESTS, SUPP)
