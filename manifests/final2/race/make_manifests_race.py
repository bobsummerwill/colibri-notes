#!/usr/bin/env python3
"""Upload race, amplified (final2 run_race.sh on shadowfax, 2026-10-16): per-sample numbers, per-arm statistics,
every pairwise text / per-token logit comparison, the pass criteria as observed, and the dev -> H engine_commit
manifest (docs/experiments/manifest.example.json format, validated by c/experiment_manifest.py) when the evidence
supports one.

usage: make_manifests_race.py RAW_DIR
  RAW_DIR = the extracted race.tgz (the directory holding P_*/ D/ H/ Z/ Dp/ Hp/ Zp/ Qp/ Fp/ meta/ run.log summary.json).
  Parsing uses the final_tools.py and bench_common.py shipped inside the tarball (RAW_DIR/meta), the code that wrote
  summary.json; every per-sample check is repeated here from the raw files.
Also reads evidence/race-host-journal.txt (capture_host_journal_race.sh: serve-unit accounting, llama.cpp unit state,
Hermes watchdog runs in the window; read-only journalctl).
Writes, next to this script:
  evidence/race-arm-<X>.tgz   deterministic per-arm tarball of RAW_DIR/<X>/ (what a manifest's evidence names);
                              the pilot settings P_* go into one evidence/race-pilot.tgz
  race-engine-dev-h2d.json    only if the gate below passes: engine_commit dev 7b55a9b3 -> H cc9c9fab
  stats.json                  everything else (and the reason when no manifest is written)

Groups (run_race.sh): P_<setting> = phase-1 pilot on the dev probe build (2 runs each);
  D/H/Z = plain builds dev 7b55a9b3, dev + only cc9c9fab, pr0 tip a2c8ba7a (2 runs each, rounds 3 and 5);
  Dp/Hp/Zp = the same SHAs + the logging-only probe (3 runs each, rounds 1, 2, 4);
  Qp/Fp = phase 3, pr1-line probe builds 48767c01 (before the fix) and a406901b (with it), COLI_CUDA_MOE_DOUBLE=1.
The per-token probe line is "v4_logit tok=<argmax> logit=<max logit, %a> hidden=<FNV-1a of the final hidden row>":
line i is decode token i of the sample (phase 3: the two requests' tokens in order).
"""
import glob, gzip, hashlib, io, itertools, json, math, os, re, statistics as st, sys, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.join(RAW, "meta"))
import final_tools as ft      # noqa: E402  (the tarball's copy)
import bench_common as bc     # noqa: E402

DEV = "7b55a9b3792527f52d1a2337fd9e312b9a7983e3"
H = "cc9c9fab4d7d0e8c962d38a7ffb8c752a6458505"
PR0 = "a2c8ba7a149a31a781e84a992aa980afdb93497d"
P1PRE = "48767c016c993e59cd74f5511983be7bb65ec791"
P1FIX = "a406901b2c256499ed02140c8d10aa34aa00b1eb"
GROUP = {  # group: (role, tree, sha, probe)
    "D": ("dev", "/tmp/cw/r6-dev", DEV, 0), "H": ("dev+h2d", "/tmp/cw/r7-devh2d", H, 0), "Z": ("pr0", "/tmp/cw/r6-pr0", PR0, 0),
    "Dp": ("dev", "/tmp/cw/r6-dev-probe", DEV, 1), "Hp": ("dev+h2d", "/tmp/cw/r7-devh2d-probe", H, 1),
    "Zp": ("pr0", "/tmp/cw/r6-pr0-probe", PR0, 1),
    "Qp": ("pr1-pre-fix", "/tmp/cw/r7-p1pre-probe", P1PRE, 1), "Fp": ("pr1-fix", "/tmp/cw/r7-p1fix-probe", P1FIX, 1)}
BUILD_GROUPS = {"dev": ("D", "Dp"), "dev+h2d": ("H", "Hp"), "pr0": ("Z", "Zp")}

RUNLOG = open(os.path.join(RAW, "run.log")).read()
LLAMA_DOWN = re.search(r"\[(\S+ \S+)\] stopping llama-server-active", RUNLOG).group(1)
LLAMA_UP = re.search(r"restore: llama health \S+ \(ok=1\) at (\S+?);", RUNLOG).group(1)
WINDOW_S = int(re.search(r"llama-down window (\d+) s", RUNLOG).group(1))
MEMAVAIL = int(re.search(r"preflight ok: MemAvailable=(\d+) MiB", RUNLOG).group(1))
PHASE2_LINE = re.search(r"phase 2: setting (\S+) \((\w+), budget (\w+)\), (\d+) tokens per run.*env '([^']*)'", RUNLOG)
CHOSEN, CMODE, CBUDGET, N2, ENV2 = PHASE2_LINE.groups()
N2 = int(N2)
PILOT_LINE = re.search(r"phase 1 done at GPU window \d+s: chosen=\S+ mode=\S+ budget=\S+ result=(\S+) first_diff_line=(\S+)", RUNLOG)
PILOT_RESULT = PILOT_LINE.group(1)


def trees():
    out, cur = {}, None
    for l in open(os.path.join(RAW, "meta", "trees.txt")):
        l = l.rstrip("\n")
        if l.startswith("tree="):
            cur = l[5:]; out[cur] = {}
        elif cur and "=" in l:
            k, v = l.strip().split("=", 1); out[cur][k] = v
    return out


TREES = trees()
for g, (_, t, sha, probe) in GROUP.items():
    assert TREES[t]["head"] == sha, (g, TREES[t])
    assert (TREES[t]["status_lines"] == "1") == bool(probe), (g, TREES[t])
assert TREES["/tmp/cw/r7-devh2d"]["parent"] == DEV and TREES["/tmp/cw/r7-p1fix-probe"]["parent"] == P1PRE
# the probe diff is byte-identical on D and H (and on Q and F): the probe adds no difference between the arms compared
assert TREES["/tmp/cw/r6-dev-probe"]["diff_sha256"] == TREES["/tmp/cw/r7-devh2d-probe"]["diff_sha256"]
assert TREES["/tmp/cw/r7-p1pre-probe"]["diff_sha256"] == TREES["/tmp/cw/r7-p1fix-probe"]["diff_sha256"]

LONG, SHORT = bc.long_prompt(), bc.load_short_prompt()
PROMPT_SHA = {"long": ft.sha(LONG), "short": ft.sha(SHORT)}

# ---------------------------------------------------------------- per sample: the race_sample record + checks
NEVER = ["CUDA_LAUNCH_BLOCKING", "COLI_V4_SAVE_USAGE", "V4_GPU_RESIDENT_FIRST", "DSV4_CUDA_PEER_DEVICE",
         "DSV4_CUDA_PEER_COMPUTE", "DSV4_HYBRID", "DSV4_CUDA_VRAM_RESERVE_MB"]


def want_env(group, meta):
    ex = ft.extra_env({"meta": meta})
    return dict(ft.COMMON, **ex), ex


def check(group, s, x):
    """repeat every per-sample check from the raw files"""
    p = list(x["problems"])
    _, tree, sha, probe = GROUP.get(group, ("dev", "/tmp/cw/r6-dev-probe", DEV, 1))
    m, env = s["meta"], s["env"]
    if m.get("tree") != tree:
        p.append("tree %s != %s" % (m.get("tree"), tree))
    if m.get("tree_head") != sha:
        p.append("tree_head %s" % m.get("tree_head"))
    if m.get("binary_sha256") != TREES[tree]["binary_sha256"]:
        p.append("binary %s != trees.txt %s" % (m.get("binary_sha256"), TREES[tree]["binary_sha256"]))
    if m.get("tree_dirty_lines") != ("1" if probe else "0"):
        p.append("tree_dirty_lines=%s" % m.get("tree_dirty_lines"))
    if probe and m.get("tree_diff_sha256") != TREES[tree].get("diff_sha256"):
        p.append("probe diff %s" % m.get("tree_diff_sha256"))
    if m.get("client_rc") != "0":
        p.append("client_rc=%s" % m.get("client_rc"))
    if m.get("usage_before") != m.get("usage_after"):
        p.append("usage history changed")
    must, ex = want_env(group, m)
    p += ["engine env %s=%r (want %r)" % (k, env.get(k), v) for k, v in must.items() if env.get(k) != v]
    p += ["engine env %s set" % k for k in NEVER if k in env]
    if "COLI_CUDA_MOE_DOUBLE" in env and group not in ("Qp", "Fp"):
        p.append("COLI_CUDA_MOE_DOUBLE set outside phase 3")
    if bool(probe) != (ex.get("V4_LOGIT_LOG") == "1"):
        p.append("V4_LOGIT_LOG does not match the build kind")
    if not probe and x["logit_lines"]:
        p.append("probe lines in a plain build")
    for r in s["rows"]:
        if ft.sha(r["text"]) != r["text_sha256"]:
            p.append("%s text_sha256 does not match its text" % r["name"])
    if group in ("Qp", "Fp"):
        if [r["name"] for r in s["rows"]] != ["long_prefill", "long_prefill_2"]:
            p.append("phase-3 requests %s" % [r["name"] for r in s["rows"]])
        if probe:
            rl = request_lines(s["log"])
            if len(rl) != len(s["rows"]) or sum(rl) != x["logit_lines"] or \
                    any(not 0 <= n - r["decode_chunks"] <= 2 for n, r in zip(rl, s["rows"])):
                p.append("logit lines per request %s for chunks %s" % (rl, [r["decode_chunks"] for r in s["rows"]]))
    else:
        mode = m["client"].split(":")[1]
        ntok = int(m["client"].split(":")[2])
        r = s["rows"][0] if len(s["rows"]) == 1 else None
        if r is None:
            p.append("%d requests" % len(s["rows"]))
        else:
            if r["prompt_sha256"] != PROMPT_SHA[mode]:
                p.append("prompt sha256 %s" % r["prompt_sha256"])
            if r["decode_chunks"] != ntok or r["finish_reason"] != "length":
                p.append("chunks=%s finish=%s (want %d, length)" % (r["decode_chunks"], r["finish_reason"], ntok))
            if probe and x["logit_lines"] != r["decode_chunks"]:
                p.append("%d logit lines for %d chunks" % (x["logit_lines"], r["decode_chunks"]))
    return p


def request_lines(log):
    """logit lines per request: each '[api] ... POST' line opens a request, the v4_logit lines after it belong to it"""
    counts = []
    for l in log:
        if l.startswith("[api] ") and "POST" in l:
            counts.append(0)
        elif l.startswith("v4_logit ") and counts:
            counts[-1] += 1
    return counts


def load(group):
    out = []
    for s in ft.samples(RAW, group):
        x = ft.race_sample(s)
        x["req_lines"] = request_lines(s["log"])
        x["checks"] = check(group, s, x)
        x["dir"] = "%s/s%d" % (group, s["k"])
        x["chunk_text"] = [r.get("chunk_text") for r in s["rows"]]
        x["meta"] = s["meta"]
        out.append(x)
    return out


GROUPS = sorted(os.path.basename(p) for p in glob.glob(os.path.join(RAW, "*")) if os.path.isdir(p) and
                (os.path.basename(p) in GROUP or os.path.basename(p).startswith("P_")))
S = {g: load(g) for g in GROUPS}
SEQ_ALL = sorted(x["seq"] for xs in S.values() for x in xs)
assert SEQ_ALL == list(range(1, len(SEQ_ALL) + 1)), SEQ_ALL


# ---------------------------------------------------------------- comparisons
def split_line(l):
    f = dict(kv.split("=", 1) for kv in l.split()[1:])
    return f.get("tok"), f.get("logit"), f.get("hidden")


def where(x, i):
    if i is None:
        return None
    for name, n in zip(x["names"], x["req_lines"]):
        if i < n:
            return {"request": name, "token_index_in_request": i}
        i -= n
    return {"request": None, "token_index_in_request": i}


def logit_pair(a, b):
    """first differing index of the argmax token, the max logit (exact %a bits) and the final-hidden-row hash"""
    la, lb = a["logits"], b["logits"]
    n = min(len(la), len(lb))
    pa, pb = [split_line(l) for l in la[:n]], [split_line(l) for l in lb[:n]]
    first = lambda j: next((i for i in range(n) if pa[i][j] != pb[i][j]), None)
    c = {"compared": n, "len_a": len(la), "len_b": len(lb), "identical": la == lb,
         "first_diff_line": next((i for i in range(n) if la[i] != lb[i]), None if len(la) == len(lb) else n),
         "first_token_diff": first(0), "first_logit_diff": first(1), "first_hidden_diff": first(2),
         "n_diff_lines": sum(1 for i in range(n) if la[i] != lb[i])}
    if c["first_diff_line"] is not None and c["first_diff_line"] < n:
        c["first_a"], c["first_b"] = la[c["first_diff_line"]], lb[c["first_diff_line"]]
    for k in ("first_diff_line", "first_token_diff"):
        c[k.replace("first_", "where_first_")] = where(a, c[k])
    c["req_lines_a"], c["req_lines_b"] = a["req_lines"], b["req_lines"]
    return c


def text_pair(a, b):
    out = []
    for i, (ta, tb) in enumerate(zip(a["texts"], b["texts"])):
        ca, cb = a["chunk_text"][i], b["chunk_text"][i]
        tokd = None
        if ca is not None and cb is not None:
            tokd = next((j for j in range(min(len(ca), len(cb))) if ca[j] != cb[j]), None if len(ca) == len(cb) else min(len(ca), len(cb)))
        out.append({"request": a["names"][i], "identical": ta == tb, "first_diff_char": ft.first_diff(ta, tb),
                    "first_diff_chunk": tokd})
    return out


def label(x):
    return x["dir"].replace("/s", "#")


def compare_set(xs):
    """all pairs of a set of samples: texts always, logits where both are probe runs"""
    pairs = {}
    for a, b in itertools.combinations(xs, 2):
        c = {"text": text_pair(a, b)}
        if a["probe"] and b["probe"]:
            c["logits"] = logit_pair(a, b)
        pairs["%s_vs_%s" % (label(a), label(b))] = c
    texts = {}
    for x in xs:
        texts.setdefault(tuple(x["text_sha256"]), []).append(label(x))
    lseq = {}
    for x in xs:
        if x["probe"]:
            lseq.setdefault(x["logit_seq_sha256"], []).append(label(x))
    any_text = len(texts) > 1
    any_logit = any(not c["logits"]["identical"] for c in pairs.values() if "logits" in c)
    return {"n": len(xs), "n_probe": sum(1 for x in xs if x["probe"]),
            "distinct_text_sha256": [{"sha256": list(k), "samples": v} for k, v in texts.items()],
            "n_distinct_texts": len(texts),
            "distinct_logit_seq_sha256": [{"sha256": k, "samples": v} for k, v in lseq.items()],
            "n_distinct_logit_seqs": len(lseq),
            "any_text_diff": any_text, "any_logit_diff": any_logit, "bit_identical": not any_text and not any_logit,
            "pairs": pairs}


def r4(v):
    return round(v, 4) if isinstance(v, float) else v


def stat(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    if not xs:
        return None
    return {"mean": r4(st.mean(xs)), "stdev": r4(st.stdev(xs)) if len(xs) > 1 else None, "median": r4(st.median(xs)),
            "min": min(xs), "max": max(xs), "n": len(xs)}


def sample_row(x):
    return {"dir": x["dir"], "seq": x["seq"], "probe": x["probe"], "tree_head": x["tree_head"],
            "binary_sha256": x["binary_sha256"], "tree_diff_sha256": x["tree_diff_sha256"] if x["probe"] else None,
            "extra_env": x["extra_env"], "start": x["meta"].get("start"), "ready_s": x["ready_s"], "client_s": x["client_s"],
            "wall_s": x["wall_s"], "requests": x["names"], "ttft_s": x["ttft_s"], "decode_tok_s": x["decode_tok_s"],
            "decode_s": x["decode_s"], "chunks": x["chunks"], "finish": x["finish"], "text_sha256": x["text_sha256"],
            "text_chars": [len(t) for t in x["texts"]], "logit_lines": x["logit_lines"],
            "logit_seq_sha256": x["logit_seq_sha256"], "logit_lines_per_request": x["req_lines"],
            "cpu_moe_lines": x["cpu_moe_lines"],
            "gpu0_ready_mib": x["gpu0_ready_mib"], "gpu0_max_mib": x["gpu0_max_mib"], "gpu1_max_mib": x["gpu1_max_mib"],
            "mode_lines": x["mode_lines"], "usage_before": x["meta"].get("usage_before"),
            "usage_after": x["meta"].get("usage_after"), "checks": x["checks"]}


def group_stats(xs):
    return {"tok_s": stat([x["decode_tok_s"][0] for x in xs]), "ttft_s": stat([x["ttft_s"][0] for x in xs]),
            "wall_s": stat([x["wall_s"] for x in xs]), "gpu0_max_mib": stat([x["gpu0_max_mib"] for x in xs])}


stats = {"run": {"llama_down": LLAMA_DOWN, "llama_restored": LLAMA_UP, "gpu_window_s": WINDOW_S,
                 "preflight_mem_available_mib": MEMAVAIL, "pilot_result": PILOT_RESULT, "phase2_setting": CHOSEN,
                 "phase2_mode": CMODE, "phase2_budget": CBUDGET, "phase2_tokens": N2, "phase2_env": ENV2,
                 "trees": {g: {"tree": GROUP[g][1], **TREES[GROUP[g][1]]} for g in GROUP}},
         "pilot": {}, "phase2": {}, "phase3": {}}

# phase 1: every pilot setting, and all pilot runs of one mode against each other (with >= 8 mirrors every expert
# is computed on the GPU, so the budget alone does not change the numbers)
for g in [g for g in GROUPS if g.startswith("P_")]:
    xs = S[g]
    stats["pilot"][g[2:]] = {"samples": [sample_row(x) for x in xs], "compare": compare_set(xs), **group_stats(xs)}
for mode, tag in (("long", "L"), ("short", "S")):
    xs = [x for g in GROUPS if g.startswith("P_" + tag) for x in S[g]]
    if xs:
        c = compare_set(xs)
        stats["pilot"]["all_%s_runs" % mode] = {k: c[k] for k in c if k != "pairs"}
        stats["pilot"]["all_%s_runs" % mode]["logit_pairs_differing"] = sorted(
            k for k, v in c["pairs"].items() if "logits" in v and not v["logits"]["identical"])

# phase 2: per group, per build (plain + probe together for texts, probe pairs for logits), across builds
for g in ("D", "H", "Z", "Dp", "Hp", "Zp"):
    if S.get(g):
        stats["phase2"][g] = {"samples": [sample_row(x) for x in S[g]], "compare": compare_set(S[g]), **group_stats(S[g])}
verdict = {}
for role, (plain, probe) in BUILD_GROUPS.items():
    xs = sorted(S.get(plain, []) + S.get(probe, []), key=lambda x: x["seq"])
    if xs:
        verdict[role] = compare_set(xs)
        verdict[role].update(group_stats(xs))
cross = {}
for a, b in (("Dp", "Hp"), ("Hp", "Zp"), ("Dp", "Zp"), ("D", "H"), ("H", "Z"), ("D", "Z")):
    if S.get(a) and S.get(b):
        c = {}
        for x, y in itertools.product(S[a], S[b]):
            cc = {"text": text_pair(x, y)}
            if x["probe"] and y["probe"]:
                cc["logits"] = logit_pair(x, y)
            c["%s_vs_%s" % (label(x), label(y))] = cc
        cross["%s_vs_%s" % (a, b)] = {
            "all_texts_identical": all(t["identical"] for v in c.values() for t in v["text"]),
            "all_logits_identical": (all(v["logits"]["identical"] for v in c.values() if "logits" in v)
                                     if any("logits" in v for v in c.values()) else None),
            "pairs": c}
pilot_long = [x for g in GROUPS if g.startswith("P_L") for x in S[g]]
if pilot_long and any(S.get(g) for g in ("Dp", "Hp", "Zp")):
    ref = pilot_long[0]; n = len(ref["logits"])
    pref = {}
    for g in ("Dp", "Hp", "Zp"):
        for x in S.get(g, []):
            pref[label(x)] = {"compared": min(n, len(x["logits"])), "identical_prefix": x["logits"][:n] == ref["logits"],
                              "text_prefix_identical": x["texts"][0].startswith(ref["texts"][0])}
    cross["probe_prefix_vs_pilot"] = {"reference": label(ref), "pilot_long_runs_identical_to_reference":
                                      all(y["logits"] == ref["logits"] for y in pilot_long), "per_sample": pref}
dev_differs = bool(verdict.get("dev") and not verdict["dev"]["bit_identical"])
h_same = bool(verdict.get("dev+h2d") and verdict["dev+h2d"]["bit_identical"])
z_same = bool(verdict.get("pr0") and verdict["pr0"]["bit_identical"])
stats["phase2_verdict"] = {
    "per_build": verdict, "cross_build": cross,
    "dev_differs_among_its_runs": dev_differs, "h_bit_identical_across_its_runs": h_same,
    "z_bit_identical_across_its_runs": z_same, "PASS": dev_differs and h_same and z_same,
    "criterion": "dev differs among its own runs (text or per-token logit line) while H and Z are bit-identical across all "
                 "of theirs (all texts equal, all probe logit lines equal)"}

# phase 3
REF_PR1 = ft.REF_PR1
for g in ("Qp", "Fp"):
    if S.get(g):
        stats["phase3"][g] = {"what": ft.PHASE3[g], "samples": [sample_row(x) for x in S[g]], "compare": compare_set(S[g]),
                              "texts_vs_dev_reference": [{n: sha_ == REF_PR1.get(n) for n, sha_ in zip(x["names"], x["text_sha256"])}
                                                         for x in S[g]],
                              "ttft_s": {n: stat([x["ttft_s"][i] for x in S[g]]) for i, n in enumerate(S[g][0]["names"])},
                              "decode_tok_s": {n: stat([x["decode_tok_s"][i] for x in S[g]]) for i, n in enumerate(S[g][0]["names"])},
                              "mode_lines": sorted({l for x in S[g] for l in ft.grep(ft.lines(os.path.join(RAW, x["dir"], "serve.log")),
                                                                                   r"moe-double=\w+|moe-batch=on bank=\d+-experts")})}
if S.get("Qp") and S.get("Fp"):
    c = {}
    for x, y in itertools.product(S["Qp"], S["Fp"]):
        c["%s_vs_%s" % (label(x), label(y))] = {"text": text_pair(x, y), "logits": logit_pair(x, y)}
    stats["phase3"]["cross_Qp_vs_Fp"] = c
    q, f = stats["phase3"]["Qp"]["compare"], stats["phase3"]["Fp"]["compare"]
    stats["phase3_verdict"] = {"pre_fix_differs_among_its_runs": not q["bit_identical"],
                               "fix_bit_identical_across_its_runs": f["bit_identical"],
                               "PASS": (not q["bit_identical"]) and f["bit_identical"],
                               "criterion": "pre-fix 48767c01 differs among its runs while fix a406901b is bit-identical across all of its runs"}

ALL_CHECKS = {x["dir"]: x["checks"] for xs in S.values() for x in xs if x["checks"]}
stats["check_problems"] = ALL_CHECKS


# ---------------------------------------------------------------- evidence tarballs
def tarball(name, groups):
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=buf, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as t:
        for g in groups:
            for f in sorted(glob.glob(os.path.join(RAW, g, "s*", "*"))):
                info = t.gettarinfo(f, arcname=os.path.join("race", os.path.relpath(f, RAW)))
                info.mtime = 0; info.uid = info.gid = 0; info.uname = info.gname = ""
                with open(f, "rb") as fh:
                    t.addfile(info, fh)
    data = buf.getvalue()
    os.makedirs(os.path.join(HERE, "evidence"), exist_ok=True)
    path = os.path.join(HERE, "evidence", name)
    open(path, "wb").write(data)
    return {"uri": os.path.relpath(path, HERE), "sha256": hashlib.sha256(data).hexdigest(),
            "contains": "%s (serve.log with the v4_logit lines on probe builds, bench.jsonl with full texts and per-chunk "
                        "times, sample.txt, engine_environ.txt, procs.txt, nvsmi.csv, gpu_*.csv, mode.txt)"
                        % ", ".join("race/%s/s*/" % g for g in groups),
            "full_run": "race.tgz (sha256 in race.tgz.sha256): all groups, meta/ (host, trees, scripts, clients, probe "
                        "diffs, build logs, llama before/after), run.log, summary.json, summary.txt; the run.log inside "
                        "ends at the 'ALL DONE; packing' line (the 'packed <sha256>' line is appended after the archive)"}


EVID = {g: tarball("race-arm-%s.tgz" % g, [g]) for g in GROUPS if not g.startswith("P_")}
EVID["pilot"] = tarball("race-pilot.tgz", [g for g in GROUPS if g.startswith("P_")])
stats["evidence"] = EVID


# ---------------------------------------------------------------- Hermes / host journal in the window
def journal():
    path = os.path.join(HERE, "evidence", "race-host-journal.txt")
    if not os.path.exists(path):
        return None
    txt = open(path).read().splitlines()
    ts = lambda l: l[11:19]
    up = LLAMA_UP[11:19]
    starts_all = [ts(l) for l in txt if "Starting shadowfax-ensure-active-stack" in l]
    starts = [t for t in starts_all if LLAMA_DOWN[11:19] <= t <= up]
    after = [t for t in starts_all if t > up]
    restarted = sorted({l.split('"restarted_units":')[1].strip(" ,") for l in txt if '"restarted_units":' in l})
    failed = [ts(l) for l in txt if "Failed to start shadowfax-ensure-active-stack" in l]
    finished = [ts(l) for l in txt if "Finished shadowfax-ensure-active-stack" in l]
    recovered = [l.split('"recovered":')[1].strip(" ,") for l in txt if '"recovered":' in l]
    peaks = re.findall(r"colibri-final2-race-serve\.service: Consumed .*?, ([\d.]+[GM]) memory peak", "\n".join(txt))
    return {"uri": "evidence/race-host-journal.txt", "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "watchdog_runs_in_window": len(starts), "watchdog_runs_after_restore": after,
            "restarted_units_values": restarted, "watchdog_first": starts[0] if starts else None,
            "watchdog_last": starts[-1] if starts else None, "watchdog_failed": len(failed),
            "watchdog_finished_ok": finished, "watchdog_recovered_values": sorted(set(recovered)),
            "serve_unit_memory_peaks": peaks}


stats["host_journal"] = journal()


# ---------------------------------------------------------------- the dev -> H manifest (engine_commit)
def welch(a, b):
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    if va + vb == 0:
        return {"t": None, "df": None, "p_two_sided": None}
    t = (st.mean(b) - st.mean(a)) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / ((va ** 2 / (len(a) - 1) if va else 0) + (vb ** 2 / (len(b) - 1) if vb else 0))
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    f = lambda x: c * (1 + x * x / df) ** (-(df + 1) / 2)
    n, h = 20000, abs(t) / 20000
    area = (f(0) + f(abs(t)) + sum((4 if i % 2 else 2) * f(i * h) for i in range(1, n))) * h / 3
    return {"t": round(t, 2), "df": round(df, 1), "p_two_sided": round(1 - 2 * area, 5)}


MODEL = ("DeepSeek V4 Flash REAP 150B (puwaer/DeepSeek-V4-Flash-0731-reap-150b, 132 routed experts, fp4 experts / "
         "fp8 dense), colibri safetensors layout, 79 GiB on disk")
BUILD = ("cd c && make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j8, "
         "after applying the logging-only probe (final2/probe/probe_patch.py; the same diff, sha256 %s, on both trees)"
         % TREES["/tmp/cw/r6-dev-probe"]["diff_sha256"])
HW = {
    "cpu": "AMD Ryzen 9 9950X3D, 16 cores / 32 threads, 1 socket, governor powersave (amd-pstate), max 5.76 GHz "
           "(engine: 16 OpenMP compute threads + 3 expert-loader lanes)",
    "ram": "62 GiB DDR5 (8 GiB swap); engine --ram 40 (31.6 GiB routed-expert RAM cache, 6.2 GiB dense resident); "
           "MemAvailable %.1f GiB (%d MiB) at preflight with llama.cpp still loaded" % (MEMAVAIL / 1024, MEMAVAIL),
    "storage": "WD_BLACK SN850X 4 TB NVMe (PCIe 4.0), ext4 root filesystem (rw,relatime), model read with O_DIRECT "
               "(v4_ssd_io mode=direct-aligned); OS page cache not dropped between samples",
    "os": "Ubuntu 24.04.4 LTS, kernel 7.0.0-34-generic, gcc 13.3.0, Python 3.12.3, CUDA toolkit 12.0.140 (generic build "
          "CUDA_ARCH=sm_86 NO_TC=1), NVIDIA driver 595.84; 2x NVIDIA GeForce RTX 3090 24 GB on PCIe 4.0 x8 (x16-capable), "
          "NVLink NV4 bridge; GPU0 only (COLI_GPU=0 DSV4_CUDA_DEVICE=0), GPU1 idle"}
COMMON_CFG = dict(ft.COMMON, **{"COLI_CUDA_MOE_DOUBLE": "unset (default 0)", "ram_gib": "40", "ctx": "8192",
                                "temperature": "0 (greedy)"})
for kv_ in ENV2.split():
    k, v = kv_.split("=", 1); COMMON_CFG[k] = v
COMMON_CFG["V4_LOGIT_LOG"] = "1"
COMMON_CFG["probe_diff_sha256"] = TREES["/tmp/cw/r6-dev-probe"]["diff_sha256"]
MANIFEST = "race-engine-dev-h2d.json"


def manifest_gate():
    """the record needs >= 3 samples per arm, clean per-sample checks, exactly one changed variable, and a quality gate
    both arms pass; with the greedy engine the gate is token-exact text identity, which a drifting dev cannot pass"""
    why = []
    if len(S.get("Dp", [])) < 3 or len(S.get("Hp", [])) < 3:
        why.append("fewer than 3 probe samples per arm (Dp %d, Hp %d)" % (len(S.get("Dp", [])), len(S.get("Hp", []))))
    for g in ("D", "H", "Dp", "Hp"):
        for x in S.get(g, []):
            if x["checks"]:
                why.append("%s: %s" % (x["dir"], "; ".join(x["checks"])))
    if dev_differs:
        why.append("dev differs among its own runs, so no token-exact quality gate holds for the baseline arm; the race is "
                   "then reported in stats.json phase2_verdict, not as a one-variable speed record")
    if not h_same:
        why.append("H is not bit-identical across its runs")
    if verdict.get("dev") and verdict.get("dev+h2d") and not cross["Dp_vs_Hp"]["all_texts_identical"]:
        why.append("dev and H texts differ")
    return why


def run_block(g, quality):
    xs = S[g]
    tok = [x["decode_tok_s"][0] for x in xs]
    cfg = dict(COMMON_CFG, engine_commit=GROUP[g][2])
    return {"config": cfg, "engine": {"commit": GROUP[g][2], "binary_sha256": xs[0]["binary_sha256"],
                                      "tree": GROUP[g][1], "probe_diff_sha256": xs[0]["tree_diff_sha256"]},
            "samples": {"tok_s": tok, "tok_s_mean": r4(st.mean(tok)), "tok_s_stdev": r4(st.stdev(tok)),
                        "ttft_s": [x["ttft_s"][0] for x in xs], "decode_s": [x["decode_s"][0] for x in xs],
                        "chunks": [x["chunks"][0] for x in xs], "text_sha256": [x["text_sha256"][0] for x in xs],
                        "logit_seq_sha256": [x["logit_seq_sha256"] for x in xs],
                        "sample_dirs": [x["dir"] for x in xs], "run_order_seq": [x["seq"] for x in xs],
                        "gpu0_ready_mib": [x["gpu0_ready_mib"] for x in xs], "gpu0_max_mib": [x["gpu0_max_mib"] for x in xs],
                        "gpu1_max_mib": [x["gpu1_max_mib"] for x in xs]},
            "median_tok_s": st.median(tok),
            "quality": {"method": quality, "passed": True, "problems": []},
            "evidence": EVID[g]}


WHY = manifest_gate()
stats["manifest"] = {"name": MANIFEST if not WHY else None, "not_written_because": WHY}
if not WHY:
    ref = S["Dp"][0]
    QUALITY = (
        "greedy, token-exact: the full %d-token output text has the same sha256 (%s) in every run of this record's two arms "
        "and in the 2 plain-build runs of each (D, H: the unmodified binaries), and the per-token probe lines (argmax token, "
        "max logit bits, FNV-1a of the final hidden row) are bit-identical in all %d probe runs of dev and H (logit stream "
        "sha256 %s); plus, per sample, the engine's real environment (/proc/<pid>/environ: the config keys, no "
        "CUDA_LAUNCH_BLOCKING, no COLI_CUDA_MOE_DOUBLE), tree SHA, probe-diff sha256, binary sha256, client exit code, "
        "%d streamed chunks with finish_reason length, one probe line per token, an unchanged .coli_usage and no failure "
        "line in serve.log" % (N2, ref["text_sha256"][0][:16], len(S["Dp"]) + len(S["Hp"]), (ref["logit_seq_sha256"] or "")[:16], N2))
    b, t = run_block("Dp", QUALITY), run_block("Hp", QUALITY)
    ratio = t["median_tok_s"] / b["median_tok_s"]
    D_TOK, H_TOK = b["samples"]["tok_s"], t["samples"]["tok_s"]
    plain = {g: [x["decode_tok_s"][0] for x in S.get(g, [])] for g in ("D", "H", "Z", "Zp")}
    speed = {"dev_tok_s": D_TOK, "h_tok_s": H_TOK, "dev_median": b["median_tok_s"], "h_median": t["median_tok_s"],
             "median_ratio": round(ratio, 4), "welch_h_vs_dev": welch(D_TOK, H_TOK),
             "plain_build_tok_s": plain, "no_change_threshold": "median ratio within 1% (0.99..1.01)"}
    outcome = "improvement" if ratio > 1.01 else ("regression" if ratio < 0.99 else "no-change")
    drift_note = (
        "dev did not drift in this run: %s. Phase 1 (pilot on the dev probe build, 2 cold runs per setting, 120 tokens): "
        "%s; all %d long-mode pilot runs gave one logit stream, all %d short-mode runs another (the mirror budget alone does "
        "not change the numbers). Phase 2 then ran the fallback setting %s (no pilot setting diverged): DSV4_CUDA_PIN_HOST=0 "
        "(every refill and upload from pageable memory, so on dev every one goes through the legacy-stream cudaMemcpy the "
        "commit replaces), DSV4_CUDA_EXPERT_MIRRORS=%s (in-place recycling of nearly every expert of every token-layer; "
        "0 v4_cpu_moe lines: every decode token-layer ran on the fused GPU path) and a long first request (2041-token "
        "prompt, CPU-union prefill, decode right after it). The race is timing-dependent, so this is no evidence that dev is "
        "race-free, and this record makes no claim that cc9c9fab fixes a drift observed here. Across phases: the first "
        "120 per-token probe lines of all 9 phase-2 probe runs (dev, H, pr0) equal those of all 6 long-mode pilot runs on "
        "dev (%s), so 9 cold dev runs of this request (6 pilot, 3 phase-2 probe) agree bit for bit on those tokens."
        % ("its %d cold runs (%d probe + %d plain) gave byte-identical text and its probe runs bit-identical per-token "
           "logit lines" % (verdict["dev"]["n"], verdict["dev"]["n_probe"], verdict["dev"]["n"] - verdict["dev"]["n_probe"]),
           ", ".join("%s %s" % (k, "identical" if v["compare"]["bit_identical"] else "DIFFERENT")
                     for k, v in stats["pilot"].items() if not k.startswith("all_")),
           stats["pilot"].get("all_long_runs", {}).get("n", 0), stats["pilot"].get("all_short_runs", {}).get("n", 0),
           CHOSEN, CBUDGET,
           "identical" if cross.get("probe_prefix_vs_pilot", {}).get("pilot_long_runs_identical_to_reference") and
           all(v["identical_prefix"] for v in cross.get("probe_prefix_vs_pilot", {}).get("per_sample", {}).values())
           else "NOT identical"))
    p3 = stats.get("phase3_verdict")
    p3_note = None
    if p3:
        qx, fx = stats["phase3"]["Qp"]["compare"], stats["phase3"]["Fp"]["compare"]
        div = [k for k, v in qx["pairs"].items() if not v["logits"]["identical"]]
        c0 = next(iter(stats["phase3"]["cross_Qp_vs_Fp"].values()))["logits"]
        p3_note = (
            "Not part of this record (another tree pair, another configuration): phase 3 of the same run, the pr1 line just "
            "before the same upload-ordering change (48767c01) and with it (a406901b), GPU-bank prefill + "
            "COLI_CUDA_MOE_DOUBLE=1, bench_pr1.py's two requests (1,992 / 3,005 prompt tokens), 3 cold probe runs each: "
            "pre-fix %d distinct texts and %d distinct logit streams in 3 runs (differing pairs %s; first differing logit "
            "line %s = %s token %s, the first decode step after the 3,005-token prefill), fix %d text and %d logit stream "
            "in 3 runs; every pre-fix run differs from every fix run in the logits from that same step. stats.json phase3."
            % (qx["n_distinct_texts"], qx["n_distinct_logit_seqs"], div, c0["first_diff_line"],
               c0["where_first_diff_line"]["request"], c0["where_first_diff_line"]["token_index_in_request"],
               fx["n_distinct_texts"], fx["n_distinct_logit_seqs"]))
    rec = {"version": 1,
           "hypothesis": ("Upstream dev + only the upload-ordering commit cc9c9fab (decode-mirror uploads and in-place refills "
                          "queued on the compute stream with h2d() instead of a legacy-stream cudaMemcpy) leaves one-GPU "
                          "REAP-132 greedy decode byte-identical to dev under the race's documented triggers (pageable "
                          "refills, smallest live mirror budget, long first request), and does not slow that decode"),
           "commit": H, "model": MODEL,
           "command": ("build: %s; run: python3 c/coli serve --model <REAP dir> --ram 40 --ctx 8192 --host 127.0.0.1 --port "
                       "8090, server environment exactly PATH/HOME/USER/LOGNAME/LANG + the config's env keys (env -i); one "
                       "freshly started server process per sample; phase-2 rounds interleaved with the build order rotated "
                       "per round (D H Z / H Z D / Z D H / D H Z / H Z D; rounds 1, 2, 4 on the probe builds, 3 and 5 on "
                       "the plain builds); client bench_race.py RACE_MODE=%s RACE_NTOK=%d: one streamed greedy request, "
                       "bench_pr1.py's 1,992-token filler + a closing question asking for a long answer (2041 prompt "
                       "tokens), max_tokens %d; tok_s = (chunks - 1) / (last chunk time - first chunk time); baseline arm "
                       "Dp (dev 7b55a9b3 + probe), trial arm Hp (cc9c9fab + the same probe)" % (BUILD, CMODE, N2, N2)),
           "prompt_hash": PROMPT_SHA[CMODE], "hardware": HW, "warmup_runs": 0,
           "changed_variables": ["engine_commit"], "baseline": b, "trial": t,
           "outcome": outcome, "speedup_median_tok_s": round(ratio, 4),
           "speed_comparison": speed,
           "drift_observation": {"dev_differs_among_its_runs": dev_differs, "h_bit_identical": h_same,
                                 "pr0_bit_identical": z_same, "pass_criterion_met": stats["phase2_verdict"]["PASS"],
                                 "note": drift_note, "phase3_same_run": p3_note},
           "notes": ["prompt_hash = sha256 of the request's prompt text (bench_common.long_prompt(); bench.jsonl prompt_sha256).",
                     "warmup_runs 0: the request is the first one to each freshly started server process.",
                     "The probe builds carry the logging-only probe (one v4_logit line per token); the same diff is on both "
                     "arms, so engine_commit is the only difference. The plain builds' runs (D, H: 2 each) are in stats.json "
                     "and agree with the probe texts.",
                     "Trial vs baseline median: %+.2f%%; Welch t = %s (df %s), two-sided p = %s; every H sample is %s "
                     "every dev sample (plain builds: D %s, H %s tok/s)." % (
                         100 * (ratio - 1), speed["welch_h_vs_dev"]["t"], speed["welch_h_vs_dev"]["df"],
                         speed["welch_h_vs_dev"]["p_two_sided"],
                         "above" if min(H_TOK) > max(D_TOK) else ("below" if max(H_TOK) < min(D_TOK) else "not separated from"),
                         plain["D"], plain["H"]),
                     "The speed difference is measured in a stress configuration chosen to provoke the race, not in a "
                     "default one: every refill and upload pageable (DSV4_CUDA_PIN_HOST=0) and DSV4_CUDA_EXPERT_MIRRORS=8, "
                     "so nearly every expert of every token-layer is refilled in place (dev decodes at %.2f tok/s here vs "
                     "%.2f with the default budget in the pilot, P_Ldef). It says nothing about decode speed at the defaults "
                     "(there the final drift run measured pr0 vs dev +0.31%%, no-change)." % (
                         b["median_tok_s"], st.median([x["decode_tok_s"][0] for x in S.get("P_Ldef", [])] or [float("nan")])),
                     "llama.cpp (the Hermes backend) was stopped %s to %s PDT (%d s) and restored healthy%s; the Hermes "
                     "cron database was not opened, so which Hermes jobs ran without a backend is not recorded." % (
                         LLAMA_DOWN, LLAMA_UP[11:19], WINDOW_S,
                         "; the shadowfax-ensure-active-stack watchdog ran %d times in the window (%s to %s), failed every "
                         "time with backend unhealthy and restarted nothing (restarted_units %s, recovered %s; "
                         "evidence/race-host-journal.txt)" % (
                             stats["host_journal"]["watchdog_runs_in_window"], stats["host_journal"]["watchdog_first"],
                             stats["host_journal"]["watchdog_last"], stats["host_journal"]["restarted_units_values"],
                             stats["host_journal"]["watchdog_recovered_values"]) if stats["host_journal"] else "")]}
    with open(os.path.join(HERE, MANIFEST), "w") as fh:
        json.dump(rec, fh, indent=2); fh.write("\n")
    stats["manifest"]["outcome"] = outcome
    stats["manifest"]["speed_comparison"] = speed
    print("%s %s %.4f -> %.4f tok/s (%.4fx)" % (MANIFEST, outcome, b["median_tok_s"], t["median_tok_s"], ratio))
else:
    print("no manifest: " + " | ".join(WHY))

with open(os.path.join(HERE, "stats.json"), "w") as fh:
    json.dump(stats, fh, indent=2); fh.write("\n")

# ---------------------------------------------------------------- readable summary
print("pilot result %s, phase 2 setting %s (%s, budget %s, %d tokens, env '%s')" % (PILOT_RESULT, CHOSEN, CMODE, CBUDGET, N2, ENV2))
for k, v in stats["pilot"].items():
    if k.startswith("all_"):
        print("  pilot %-15s n=%d distinct_texts=%d distinct_logit_seqs=%d differing_pairs=%s" % (
            k, v["n"], v["n_distinct_texts"], v["n_distinct_logit_seqs"], v["logit_pairs_differing"]))
    else:
        c = v["compare"]
        print("  pilot %-6s n=%d distinct_texts=%d distinct_logit_seqs=%d tok/s=%s" % (
            k, c["n"], c["n_distinct_texts"], c["n_distinct_logit_seqs"], [s["decode_tok_s"][0] for s in v["samples"]]))
for role, v in verdict.items():
    print("  phase2 %-8s n=%d (probe %d) distinct_texts=%d distinct_logit_seqs=%d tok/s mean %s sd %s" % (
        role, v["n"], v["n_probe"], v["n_distinct_texts"], v["n_distinct_logit_seqs"], v["tok_s"]["mean"], v["tok_s"]["stdev"]))
    for pk, pc in v["pairs"].items():
        lg = pc.get("logits")
        print("    %-14s text_identical=%s first_diff_chunk=%s%s" % (
            pk, all(t["identical"] for t in pc["text"]), [t["first_diff_chunk"] for t in pc["text"]],
            (" logits identical=%s first_token=%s first_logit=%s first_hidden=%s" % (
                lg["identical"], lg["first_token_diff"], lg["first_logit_diff"], lg["first_hidden_diff"])) if lg else ""))
for k, v in cross.items():
    if k == "probe_prefix_vs_pilot":
        print("  phase-2 probe runs vs pilot %s (first %d tokens): pilot long runs all identical=%s, prefix identical %s" % (
            v["reference"], min(p_["compared"] for p_ in v["per_sample"].values()), v["pilot_long_runs_identical_to_reference"],
            {k2: p_["identical_prefix"] for k2, p_ in v["per_sample"].items()}))
        continue
    print("  cross %-10s all_texts_identical=%s all_logits_identical=%s" % (k, v["all_texts_identical"], v["all_logits_identical"]))
print("  phase2 PASS=%s (dev differs=%s, H identical=%s, Z identical=%s)" % (
    stats["phase2_verdict"]["PASS"], dev_differs, h_same, z_same))
for g in ("Qp", "Fp"):
    if g in stats["phase3"]:
        v = stats["phase3"][g]; c = v["compare"]
        print("  phase3 %s n=%d distinct_texts=%d distinct_logit_seqs=%d vs_dev_ref=%s mode=%s" % (
            g, c["n"], c["n_distinct_texts"], c["n_distinct_logit_seqs"], v["texts_vs_dev_reference"], v["mode_lines"]))
        for pk, pc in c["pairs"].items():
            lg = pc["logits"]
            print("    %-14s text_identical=%s first_diff_char=%s logits identical=%s first_token=%s first_logit=%s first_hidden=%s n_diff=%d" % (
                pk, [t["identical"] for t in pc["text"]], [t["first_diff_char"] for t in pc["text"]], lg["identical"],
                lg["first_token_diff"], lg["first_logit_diff"], lg["first_hidden_diff"], lg["n_diff_lines"]))
if "phase3_verdict" in stats:
    print("  phase3 PASS=%s (pre-fix differs=%s, fix identical=%s)" % (
        stats["phase3_verdict"]["PASS"], stats["phase3_verdict"]["pre_fix_differs_among_its_runs"],
        stats["phase3_verdict"]["fix_bit_identical_across_its_runs"]))
print("  check problems: %s" % (ALL_CHECKS or "none"))
