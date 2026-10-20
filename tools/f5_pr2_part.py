

# ================================================================ PR #2 (final5): decode arms D0 / G1 / R1 / X3
# D0 dev, G1 pr2 tip + V4_SERVE_STATS_LOG=1 (every PR2 switch off), R1 = G1 + V4_GPU_RESIDENT_FIRST=1 (one GPU),
# X3 = R1 + DSV4_CUDA_PEER_DEVICE=1 (peer compute, DSV4_CUDA_PEER_COMPUTE default 1). Client bench_decode.py:
# 6 prompts x 300 tokens, one fresh server process per sample; metric per sample = mean of the 6 decode tok/s.
import ast

DEC_SRC = open(os.path.join(RAW, "meta", "bench_decode.py")).read()
PROMPTS_DEC = ast.literal_eval(re.search(r"^PROMPTS=(\[.*\])$", DEC_SRC, re.M).group(1))
assert len(PROMPTS_DEC) == 6 and 'run("decode_%d"%k, p, 300)' in DEC_SRC
PROMPT_HASH_DEC = hashlib.sha256(json.dumps(PROMPTS_DEC).encode()).hexdigest()
DEC_REQS = ["decode_%d" % k for k in range(6)]
DEC_TREE = {"D0": DEV, "G1": PR2, "R1": PR2, "X3": PR2}
DEC_EXTRA = {"D0": {}, "G1": {"V4_SERVE_STATS_LOG": "1"},
             "R1": {"V4_SERVE_STATS_LOG": "1", "V4_GPU_RESIDENT_FIRST": "1"},
             "X3": {"V4_SERVE_STATS_LOG": "1", "V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1"}}
PR2_SW = ("V4_SERVE_STATS_LOG", "V4_GPU_RESIDENT_FIRST", "DSV4_CUDA_PEER_DEVICE", "DSV4_CUDA_PEER_COMPUTE", "COLI_CUDA_MOE_DOUBLE")
def load_dec(arm):
    """load() for the decode client, whose rows carry the full text but no text_sha256: the hash is computed here"""
    out = []
    for d in sorted(glob.glob(os.path.join(RAW, arm, "s*")), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)))):
        rows = [json.loads(l) for l in open(os.path.join(d, "bench.jsonl")) if l.strip().startswith("{")]
        for r in rows:
            assert "text_sha256" not in r and r["text"].startswith(r["text_head"]), (d, r["name"])
            r["text_sha256"] = hashlib.sha256(r["text"].encode()).hexdigest()
        out.append({"dir": os.path.relpath(d, RAW), "meta": kv(os.path.join(d, "sample.txt")),
                    "env": kv(os.path.join(d, "engine_environ.txt")), "rows": {r["name"]: r for r in rows},
                    "serve_log": open(os.path.join(d, "serve.log"), errors="replace").read(),
                    "gpu": nvsmi(os.path.join(d, "nvsmi.csv"))})
    return out


DARMS = {a: load_dec(a) for a in DEC_TREE}
assert all(len(ss) == 3 for ss in DARMS.values()), {a: len(ss) for a, ss in DARMS.items()}
DREF = {n: {s["rows"][n]["text_sha256"] for s in DARMS["D0"]} for n in DEC_REQS}
assert all(len(v) == 1 for v in DREF.values()), "D0 texts differ between its own samples"
DREF = {k: next(iter(v)) for k, v in DREF.items()}


def vram_first_hits(log):
    m = re.findall(r"vram_first_hits_total=(\d+)", log)
    return int(m[-1]) if m else None


def dgate(arm):
    probs = []
    for s in DARMS[arm]:
        want = {k: v for k, v in COMMON_ENV.items() if k.isupper()}
        want.update(DEC_EXTRA[arm]); want.update({"RAM_GB": "40", "CTX": "8192"})
        bad = [k for k, v in want.items() if s["env"].get(k) != v]
        bad += ["%s set" % k for k in PR2_SW if k in s["env"] and k not in DEC_EXTRA[arm]]
        if s["meta"].get("tree_head") != DEC_TREE[arm] or s["meta"].get("binary_sha256") != BIN[DEC_TREE[arm]]:
            bad.append("tree/binary")
        if s["meta"].get("client_rc") != "0" or s["meta"].get("tree_dirty_lines") != "0":
            bad.append("client_rc/dirty")
        if s["meta"].get("usage_after") != s["meta"].get("usage_before") or not s["meta"].get("usage_before", "").startswith("size=0 "):
            bad.append(".coli_usage not empty/unchanged")
        log = s["serve_log"]
        hits = vram_first_hits(log)
        if arm == "D0" and (hits is not None or "peer-mirrors" in log):
            bad.append("dev printed PR2 lines")
        if arm == "G1" and hits != 0:
            bad.append("G1 vram_first_hits=%s, want 0" % hits)
        if arm in ("R1", "X3") and not hits:
            bad.append("%s vram_first_hits=%s, want > 0" % (arm, hits))
        peer_on = bool(re.search(r"v4_gpu peer-mirrors device=1 .*mode=compute", log))
        if peer_on != (arm == "X3"):
            bad.append("peer-mirrors compute line %s" % ("missing" if arm == "X3" else "present"))
        if re.search(r"peer mirror pool off|peer device \d+ failed|mode=p2p-read", log):
            bad.append("peer pool off/failed/p2p-read")
        for n in DEC_REQS:
            r = s["rows"].get(n)
            if r is None:
                bad.append("%s missing" % n)
            elif r["text_sha256"] != DREF[n]:
                bad.append("%s sha %s != D0 %s" % (n, r["text_sha256"][:12], DREF[n][:12]))
            elif r.get("decode_chunks") != 300:
                bad.append("%s decode_chunks %s" % (n, r.get("decode_chunks")))
        probs += ["%s: %s" % (s["dir"], p) for p in bad]
    return probs


def dspeed(s):
    """per-sample metric: mean over the 6 prompts of decode_tok_s (= (chunks-1)/(t_last - t_first chunk))"""
    return dmean([s["rows"][n]["decode_tok_s"] for n in DEC_REQS])


DSPEED = {a: [dspeed(s) for s in ss] for a, ss in DARMS.items()}
DORDER = " ".join(n for _, n in sorted((int(s["meta"]["seq"]), "%s s%s" % (a, s["meta"]["sample"])) for a, ss in DARMS.items() for s in ss))
assert DORDER == "D0 s1 G1 s1 R1 s1 X3 s1 G1 s2 R1 s2 X3 s2 D0 s2 R1 s3 X3 s3 D0 s3 G1 s3", DORDER
DGPU = {a: {i: max(s["gpu"]["mem_used_mib_max"][i] for s in ss) for i in ("0", "1")} for a, ss in DARMS.items()}
DRESERVE = {a: sorted(set(re.findall(r"mirror-reserve=\d+MiB", " ".join(s["serve_log"] for s in ss)))) for a, ss in DARMS.items()}
DHITS = {a: [vram_first_hits(s["serve_log"]) for s in ss] for a, ss in DARMS.items()}
DPEER = sorted(set(m for ss in DARMS.values() for s in ss for m in re.findall(r"v4_gpu peer-mirrors [^\n]*", s["serve_log"])))
P2P = open(os.path.join(RAW, "meta", "p2p_read.txt")).read() if os.path.exists(os.path.join(RAW, "meta", "p2p_read.txt")) else ""

D0_G1 = round(pct(statistics.median(DSPEED["G1"]), statistics.median(DSPEED["D0"])), 2)
DEC_OBS = (
    "D0 (upstream dev) vs G1 (pr2 tip, every PR2 switch off, V4_SERVE_STATS_LOG=1): median %.4f vs %.4f tok/s (%+.2f%%), "
    "texts identical. Not a manifest: more than one variable differs (the whole PR1+PR2 code, and the stats log). "
    "The likely cause is PR1's batched-prefill bank: with COLI_CUDA_MOE_BATCH=1 (set for every arm) the pr2 tip "
    "reserves VRAM for the 132-slot bank (serve.log %s), while dev refuses the batch path for 132 experts and reserves "
    "none (%s); GPU0 peak memory.used D0 %d MiB vs G1 %d MiB, i.e. ~%d MiB fewer decode mirrors. The reserve can be "
    "lowered with DSV4_CUDA_VRAM_RESERVE_MB; a decode-only server can leave COLI_CUDA_MOE_BATCH unset." % (
        statistics.median(DSPEED["D0"]), statistics.median(DSPEED["G1"]), D0_G1, "/".join(DRESERVE["G1"]),
        "no mirror-reserve line" if not DRESERVE["D0"] else "/".join(DRESERVE["D0"]), DGPU["D0"]["0"], DGPU["G1"]["0"],
        DGPU["D0"]["0"] - DGPU["G1"]["0"]))
DGPU_NOTE = ("Peak memory.used (nvsmi.csv, max over the arm's samples): GPU0 %s MiB; GPU1 %s MiB. Decode-mirror reserve "
             "lines: %s. Peer pool (X3 only): %s. VRAM-first hits per sample (serve.log v4_serve_turn "
             "vram_first_hits_total, last turn): %s." % (
                 ", ".join("%s %d" % (a, v["0"]) for a, v in DGPU.items()), ", ".join("%s %d" % (a, v["1"]) for a, v in DGPU.items()),
                 ", ".join("%s %s" % (a, "/".join(v) or "none") for a, v in DRESERVE.items()), "; ".join(DPEER) or "none",
                 ", ".join("%s %s" % (a, v) for a, v in DHITS.items())))
RUN_DEC = ("python3 c/coli serve --model <REAP dir> --ram 40 --ctx 8192 --host 127.0.0.1 --port 8090, server environment "
           "exactly PATH/HOME/USER/LOGNAME/LANG + the config's env keys (env -i); one freshly started server process per "
           "sample; 3 samples per arm D0/G1/R1/X3 interleaved with the order rotated per round (%s); client bench_decode.py "
           "sends 6 prompts one after another, max_tokens 300 each, streamed (the engine is greedy: temperature is "
           "ignored); per-sample tok_s = mean over the 6 prompts of (chunks - 1) / (last chunk - first chunk)" % DORDER)
HW_DEC = dict(HW)
HW_DEC["os"] = HW["os"].replace("only GPU0 used (COLI_GPU=0 DSV4_CUDA_DEVICE=0)",
                                "GPU0 primary (COLI_GPU=0 DSV4_CUDA_DEVICE=0); GPU1 used only by the trial of pr2-peer-device.json")
assert HW_DEC["os"] != HW["os"]
DEVID = {}


def drun(arm, cfg):
    ss = DARMS[arm]
    probs = dgate(arm)
    if arm not in EVID:
        EVID[arm] = evidence(arm)
    return {
        "config": cfg,
        "samples": {"tok_s": DSPEED[arm], "per_prompt_decode_tok_s": [[s["rows"][n]["decode_tok_s"] for n in DEC_REQS] for s in ss],
                    "text_sha256_per_prompt": {n: [s["rows"][n]["text_sha256"] for s in ss] for n in DEC_REQS},
                    "sample_dirs": [s["dir"] for s in ss], "run_order_seq": [int(s["meta"]["seq"]) for s in ss],
                    "vram_first_hits_total": DHITS[arm],
                    "tok_s_mean": round(statistics.mean(DSPEED[arm]), 4), "tok_s_stdev": round(statistics.stdev(DSPEED[arm]), 4)},
        "median_tok_s": statistics.median(DSPEED[arm]),
        "engine": {"commit": DEC_TREE[arm], "binary_sha256": BIN[DEC_TREE[arm]]},
        "quality": {"method": ("greedy: the full output text of all 6 prompts (300 tokens each) in every sample of this arm "
                               "has the same sha256 (recomputed from the stored text) as upstream dev's (arm D0, itself "
                               "identical across its 3 samples); the engine's real environment, tree SHA, binary sha256, "
                               "empty unchanged .coli_usage, the VRAM-first hit counter and the peer-pool lines checked per sample"),
                    "passed": not probs, "problems": probs},
        "evidence": EVID[arm],
    }


def dmanifest(name, hypothesis, changed, base_arm, trial_arm):
    bc = dict(COMMON_ENV); bc[ENG] = PR2; bc.update(DEC_EXTRA[base_arm])
    tc = dict(COMMON_ENV); tc[ENG] = PR2; tc.update(DEC_EXTRA[trial_arm])
    bc.setdefault(changed, "unset (default 0)" if changed == "V4_GPU_RESIDENT_FIRST" else "unset (no peer pool)")
    assert {k for k in set(bc) | set(tc) if bc.get(k) != tc.get(k)} == {changed}, (bc, tc)
    b = drun(base_arm, bc); t = drun(trial_arm, tc)
    ratio = t["median_tok_s"] / b["median_tok_s"]
    rec = {"version": 1, "hypothesis": hypothesis, "commit": PR2, "model": MODEL,
           "command": "build: %s; run: %s; baseline arm %s, trial arm %s" % (BUILD, RUN_DEC, base_arm, trial_arm),
           "prompt_hash": PROMPT_HASH_DEC, "hardware": HW_DEC, "warmup_runs": 0,
           "changed_variables": [changed], "baseline": b, "trial": t,
           "outcome": "improvement" if ratio > 1.01 else ("regression" if ratio < 0.99 else "no-change"),
           "speedup_median_tok_s": round(ratio, 4)}
    rec["notes"] = {
        "run": "final5, shadowfax, decode samples %s - %s PDT (Hermes's model server down %s - %s); bench.tgz sha256 %s" % (
            hm(min(s["meta"]["start"] for ss in DARMS.values() for s in ss)), hm(max(s["meta"]["end"] for ss in DARMS.values() for s in ss)),
            hm(LLAMA_DOWN), hm(LLAMA_UP), TGZ_SHA),
        "warmup": "warmup_runs 0: each sample is a freshly started server process and decode_0 is its first request; "
                  "the metric averages all 6 prompts of the sample",
        "spread": "per-sample tok/s %s: %s; %s: %s" % (base_arm, DSPEED[base_arm], trial_arm, DSPEED[trial_arm]),
        "outputs": "all %d outputs of the D0/G1/R1/X3 samples (12 samples x 6 prompts) are byte-identical to D0's per-prompt "
                   "text" % sum(len(s["rows"]) for ss in DARMS.values() for s in ss),
        "gpu": DGPU_NOTE,
        "dev_vs_pr2_off": DEC_OBS,
        "earlier": "final4 (2026-10-17, base 62eaa9dc) ran G1 and X3 once each: 3.01 -> 4.34 tok/s (+44%%, a sanity "
                   "check, no manifest); final5 is the first PR2 measurement with 3 samples per arm and one switch per manifest",
        "host": HOST_NOTE,
    }
    if P2P:
        rec["notes"]["p2p"] = "meta/p2p_read.txt: " + " ".join(P2P.split())[:400]
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(rec, f, indent=2); f.write("\n")
    OUT.append(name)
    print("%-26s %-11s %8.4f -> %8.4f tok/s (%.4fx) quality %s/%s" % (
        name, rec["outcome"], b["median_tok_s"], t["median_tok_s"], ratio, b["quality"]["passed"], t["quality"]["passed"]))


dmanifest("pr2-vram-first.json",
          "VRAM-first decode lookups (V4_GPU_RESIDENT_FIRST=1: an expert already mirrored in VRAM is computed from the mirror "
          "without a RAM-store lookup, and the RAM pins skip mirrored experts) speed up one-GPU decode of REAP-132, with "
          "outputs unchanged", "V4_GPU_RESIDENT_FIRST", "G1", "R1")
dmanifest("pr2-peer-device.json",
          "A second mirror pool on GPU 1 (DSV4_CUDA_PEER_DEVICE=1; its experts run on GPU 1 by default, "
          "DSV4_CUDA_PEER_COMPUTE=1, NVLink peer access) speeds up decode of REAP-132 over VRAM-first decode on one GPU, "
          "with outputs unchanged", "DSV4_CUDA_PEER_DEVICE", "R1", "X3")

stats["pr2"] = {"arms": {a: {"tok_s": DSPEED[a], "median": statistics.median(DSPEED[a]), "stat": stat(DSPEED[a]),
                             "gate_problems": dgate(a), "vram_first_hits": DHITS[a], "gpu_peak_mib": DGPU[a],
                             "per_prompt_mean": {n: dmean([s["rows"][n]["decode_tok_s"] for s in DARMS[a]]) for n in DEC_REQS}}
                         for a in DARMS},
                "ratios_median": {"G1/D0": round(statistics.median(DSPEED["G1"]) / statistics.median(DSPEED["D0"]), 4),
                                  "R1/G1": round(statistics.median(DSPEED["R1"]) / statistics.median(DSPEED["G1"]), 4),
                                  "X3/R1": round(statistics.median(DSPEED["X3"]) / statistics.median(DSPEED["R1"]), 4),
                                  "X3/G1": round(statistics.median(DSPEED["X3"]) / statistics.median(DSPEED["G1"]), 4),
                                  "X3/D0": round(statistics.median(DSPEED["X3"]) / statistics.median(DSPEED["D0"]), 4)},
                "order": DORDER, "prompt_hash": PROMPT_HASH_DEC, "dev_vs_pr2_off": DEC_OBS, "gpu": DGPU_NOTE}
if "D0" not in EVID:   # the reference arm of the PR2 quality gate: its evidence ships too
    EVID["D0"] = evidence("D0")
stats["manifests"] = OUT
stats["evidence"] = EVID
with open(os.path.join(HERE, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2); f.write("\n")
for a, v in stats["pr2"]["arms"].items():
    print("%s decode tok/s %s median %.4f gate %s" % (a, v["tok_s"], v["median"], v["gate_problems"] or "ok"))
print("ratios (median):", stats["pr2"]["ratios_median"])
