#!/usr/bin/env python3
"""Result tools for the final2 colibri GPU runs (run_f42.sh, run_race.sh). CPU only.

Reads <exp>/<arm>/s<k>/{sample.txt,bench.jsonl,serve.log,engine_environ.txt,nvsmi.csv,gpu_ready.csv}.
Derived from final/final_tools.py (same sample layout and helpers).

  final_tools.py brief BENCH.jsonl             one line per request (logged after each sample)
  final_tools.py racepilot RACE_DIR            phase-1 decision of run_race.sh; prints KEY=VALUE lines
  final_tools.py summarize f42|race DIR        writes DIR/summary.json, prints a readable summary
  final_tools.py selftest                      runs both summaries on synthetic samples (preflight)
"""
import glob, hashlib, json, os, re, statistics, sys, tempfile, time
from datetime import datetime


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path, errors="replace"):
        line = line.strip()
        if line.startswith("{"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            r.setdefault("text", r.get("text_head", ""))
            r.setdefault("text_sha256", sha(r["text"]))
            rows.append(r)
    return rows


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


def samples(base, arm):
    out = []
    for d in sorted(glob.glob(os.path.join(base, arm, "s*")),
                    key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0)):
        out.append({"dir": d, "arm": arm, "k": int(re.sub(r"\D", "", os.path.basename(d)) or 0),
                    "meta": kv(os.path.join(d, "sample.txt")), "rows": read_jsonl(os.path.join(d, "bench.jsonl")),
                    "log": lines(os.path.join(d, "serve.log")), "env": kv(os.path.join(d, "engine_environ.txt"))})
    return out


def usage_changed(base):
    bad = []
    for st in glob.glob(os.path.join(base, "*", "s*", "sample.txt")):
        m = kv(st)
        if m.get("usage_before") and m.get("usage_after") and m["usage_before"] != m["usage_after"]:
            bad.append("%s: %s -> %s" % (os.path.relpath(os.path.dirname(st), base), m["usage_before"], m["usage_after"]))
    return bad


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return round(statistics.median(xs), 4) if xs else None


def ratio(a, b):
    return round(a / b, 4) if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b else None


def diff(a, b):
    return round(a - b, 4) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None


def grep(log, pat):
    r = re.compile(pat)
    return [l for l in log if r.search(l)]


def fields(line):
    """name=number pairs of an engine stats line"""
    return {k: float(v) for k, v in re.findall(r"(\w+)=(-?[\d.]+)", line)}


def brief(path):
    for r in read_jsonl(path):
        print("   %-16s ttft=%-8s decode_tok_s=%-7s chunks=%-4s finish=%-6s sha=%s" % (
            r.get("name"), r.get("ttft_s"), r.get("decode_tok_s"), r.get("decode_chunks"),
            r.get("finish_reason"), r["text_sha256"][:12]))


def first_diff(a, b):
    return next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None if len(a) == len(b) else min(len(a), len(b)))


COMMON = {"COLI_CUDA": "1", "COLI_GPU": "0", "DSV4_CUDA_DEVICE": "0", "CUDA_DENSE": "1", "COLI_CUDA_PIPE": "2",
          "COLI_CUDA_ATTN_BATCH": "1", "COLI_CUDA_MOE_BATCH": "1", "V4_MOE_REFILL_GROUP": "12",
          "V4_LOADER_LANES": "3", "V4_PREFIX_CKPT_DISK": "0"}
NEVER = ["CUDA_LAUNCH_BLOCKING", "COLI_V4_SAVE_USAGE", "V4_GPU_RESIDENT_FIRST", "DSV4_CUDA_PEER_DEVICE",
         "DSV4_CUDA_PEER_COMPUTE", "DSV4_HYBRID"]


def extra_env(s):
    """the arm's KEY=VAL extras as the runner passed them (sample.txt extra_env)"""
    out = {}
    for tok in s["meta"].get("extra_env", "").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def env_problems(s, must, must_not):
    env = s["env"]
    if not env:
        return ["engine_environ.txt missing"]
    bad = ["%s=%r (want %r)" % (k, env.get(k), v) for k, v in must.items() if env.get(k) != v]
    bad += ["%s set (=%r) but must be unset" % (k, env[k]) for k in must_not if k in env]
    return bad


def nvsmi(d):
    """[(epoch, index, used_mib)] from the sample's nvidia-smi log (local-time timestamps)"""
    out = []
    for line in lines(os.path.join(d, "nvsmi.csv"))[1:]:
        f = [x.strip() for x in line.split(",")]
        if len(f) < 3:
            continue
        try:
            t = datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            m = re.match(r"(\d+)", f[2])
            out.append((t, int(f[1]), int(m.group(1))))
        except (ValueError, AttributeError):
            continue
    return out


def ready_mib(d, index=0):
    for line in lines(os.path.join(d, "gpu_ready.csv"))[1:]:
        f = [x.strip() for x in line.split(",")]
        if len(f) > 1 and f[0] == str(index):
            m = re.match(r"(\d+)", f[1])
            if m:
                return int(m.group(1))
    return None


def gpu_max(d, index=0):
    xs = [m for _, i, m in nvsmi(d) if i == index]
    return max(xs) if xs else None


def window(trace, t0, t1, index=0):
    return [m for t, i, m in trace if i == index and t0 <= t <= t1]


# ---------------------------------------------------------------- per-chunk timing
def chunk_metrics(r, head=24, win=25):
    """client-side decode timing of one request from its per-chunk offsets"""
    t = r.get("chunk_t") or []
    n = len(t)
    out = {"n": n, "ttft_s": r.get("ttft_s"), "finish": r.get("finish_reason"),
           "rate_all": round((n - 1) / (t[-1] - t[0]), 4) if n > 1 and t[-1] > t[0] else None,
           "rate_first%d" % head: round(head / (t[head] - t[0]), 4) if n > head and t[head] > t[0] else None,
           "rate_after%d" % head: round((n - 1 - head) / (t[-1] - t[head]), 4) if n > 2 * head and t[-1] > t[head] else None}
    dt = [t[i] - t[i - 1] for i in range(1, n)]
    out["interval_ms_median"] = round(1000 * statistics.median(dt), 1) if dt else None
    out["interval_ms"] = [round(1000 * x, 1) for x in dt]
    out["window_rates"] = [round(len(dt[i:i + win]) / sum(dt[i:i + win]), 4) for i in range(0, len(dt), win)
                           if sum(dt[i:i + win]) > 0]
    out["sec_per_tok"] = round((t[-1] - t[0]) / (n - 1), 5) if n > 1 else None
    return out


# ---------------------------------------------------------------- F42
F42_ARMS = {
    # arm: (env beyond COMMON, env that must be unset, required log patterns, forbidden log patterns, bank MiB)
    "GB": ({"V4_SERVE_STATS_LOG": "1", "DSV4_DECODE_PROF": "1"}, ["DSV4_CUDA_VRAM_RESERVE_MB"],
           [r"moe-batch=on bank=132-experts", r"mirror-reserve=3100MiB"], [r"moe-batch=off"], 1683),
    "CU": ({"V4_SERVE_STATS_LOG": "1", "DSV4_DECODE_PROF": "1", "COLI_CUDA_MOE_BATCH": "0"}, ["DSV4_CUDA_VRAM_RESERVE_MB"],
           [], [r"moe-batch=on", r"mirror-reserve="], 0),
    "GR": ({"V4_SERVE_STATS_LOG": "1", "DSV4_DECODE_PROF": "1", "DSV4_CUDA_VRAM_RESERVE_MB": "600"}, [],
           [r"moe-batch=on bank=132-experts"], [r"mirror-reserve=", r"moe-batch=off"], 1683),
}
F42_UNSET = ["COLI_CUDA_MOE_DOUBLE", "DSV4_CUDA_EXPERT_MIRRORS", "DSV4_CUDA_PIN_HOST", "V4_LOGIT_LOG"] + NEVER
MIRROR_MIB = 12.75
DEC_KEYS = ("calls", "total_s", "fetch_wait_s", "attach_s", "uploads", "gpu_s")


def f42_turns(log):
    """per turn: v4_serve_turn fields and the v4_dec_moe deltas (the line is cumulative)"""
    turns = [fields(l) for l in log if l.startswith("v4_serve_turn ")]
    decs = [fields(l) for l in log if l.startswith("v4_dec_moe ")]
    prev = {k: 0.0 for k in DEC_KEYS}
    for i, t in enumerate(turns):
        cur = decs[i] if i < len(decs) else None
        t["dec"] = {k: round(cur.get(k, 0) - prev.get(k, 0), 4) for k in DEC_KEYS} if cur else None
        if cur:
            prev = cur
    return turns


def decprof(log, prompts):
    """decprof lines split by turn: a line belongs to the last turn whose decode it can be
    (position >= that turn's prompt length - 1); the rest (per-token prefill, if any) to 'other'"""
    out = [[] for _ in prompts] + [[]]
    for l in log:
        if not l.startswith("decprof "):
            continue
        f = fields(l)
        pos = int(f.get("position", -1))
        idx = len(prompts)
        for i in sorted(range(len(prompts)), key=lambda j: -prompts[j]):
            if pos >= prompts[i] - 1:
                idx = i
                break
        out[idx].append(f)
    return out


def prof_sums(rows, head=24):
    def s(rs, k):
        return round(sum(r.get(k, 0) for r in rs) / 1000.0, 3)
    first, rest = rows[:head], rows[head:]
    return {"n": len(rows), "attn_s": s(rows, "attn"), "moe_s": s(rows, "moe"), "hc_s": s(rows, "hc"),
            "first_attn_s": s(first, "attn"), "first_moe_s": s(first, "moe"),
            "rest_attn_s": s(rest, "attn"), "rest_moe_s": s(rest, "moe"),
            "moe_ms_per_tok": round(sum(r.get("moe", 0) for r in rows) / len(rows), 1) if rows else None,
            "attn_ms_per_tok": round(sum(r.get("attn", 0) for r in rows) / len(rows), 1) if rows else None,
            "first_moe_ms_per_tok": round(sum(r.get("moe", 0) for r in first) / len(first), 1) if first else None,
            "rest_moe_ms_per_tok": round(sum(r.get("moe", 0) for r in rest) / len(rest), 1) if rest else None}


def f42_sample(s, arm):
    extra, unset, need, forbid, bank_mib = F42_ARMS[arm]
    problems = env_problems(s, dict(COMMON, **extra), unset + F42_UNSET)
    problems += ["missing log line /%s/" % p for p in need if not grep(s["log"], p)]
    problems += ["forbidden log line /%s/: %s" % (p, grep(s["log"], p)[0][:160]) for p in forbid if grep(s["log"], p)]
    if s["meta"].get("client_rc") != "0":
        problems.append("client_rc=%s" % s["meta"].get("client_rc"))
    if len(grep(s["log"], r"out of memory|FAILED|Traceback")):
        problems.append("%d OOM/FAILED/Traceback lines" % len(grep(s["log"], r"out of memory|FAILED|Traceback")))
    turns = f42_turns(s["log"])
    rows = {r["name"]: r for r in s["rows"]}
    if len(turns) != len(s["rows"]):
        problems.append("%d v4_serve_turn lines for %d requests" % (len(turns), len(s["rows"])))
    prompts = [int(t.get("prompt", 0)) for t in turns]
    prof = decprof(s["log"], prompts) if prompts else [[]]
    trace = nvsmi(s["dir"])
    ready = ready_mib(s["dir"])
    reqs = {}
    base_mib = None
    for i, r in enumerate(s["rows"]):
        cm = chunk_metrics(r)
        if r.get("finish_reason") != "length" or cm["n"] != r.get("max_tokens"):
            problems.append("%s: %s chunks, finish=%s (want %s, length)" % (r["name"], cm["n"], r.get("finish_reason"), r.get("max_tokens")))
        t = turns[i] if i < len(turns) else {}
        t0 = r.get("t0_epoch") or 0
        first = t0 + (r.get("ttft_s") or 0)
        end = t0 + (r.get("total_s") or 0)
        pre = window(trace, t0, first)
        dec = window(trace, first, end + 1)
        ps = prof_sums(prof[i]) if i < len(prof) else None
        comp = int(t.get("completion", 0)) or None
        dec_d = t.get("dec") or {}
        per_tok = {k: round(dec_d[k] / comp, 5) for k in DEC_KEYS if comp and k in dec_d}
        if i == 0 and pre:
            base_mib = statistics.median(pre) - bank_mib     # VRAM before any decode mirror (bank excluded)
        reqs[r["name"]] = {
            "prompt_tokens": t.get("prompt"), "completion": t.get("completion"), "text_sha256": r["text_sha256"],
            "client": {k: v for k, v in cm.items() if k != "interval_ms"}, "interval_ms": cm["interval_ms"],
            "turn": {k: t.get(k) for k in ("decode_s", "tok_s", "hit", "expert_disk_s", "expert_wait_s",
                                           "expert_matmul_s", "attention_s", "rss_gb", "store_fetches_total")},
            "dec_moe": dec_d, "dec_moe_per_tok": per_tok, "decprof": ps,
            "vram0": {"ready": ready, "prefill_median": statistics.median(pre) if pre else None,
                      "prefill_max": max(pre) if pre else None, "decode_max": max(dec) if dec else None,
                      "decode_end": dec[-1] if dec else None, "samples_prefill": len(pre), "samples_decode": len(dec)}}
    for name, q in reqs.items():
        v = q["vram0"]
        q["mirrors_est"] = round((v["decode_max"] - base_mib) / MIRROR_MIB, 1) if v["decode_max"] and base_mib else None
    return {"k": s["k"], "seq": s["meta"].get("seq"), "tree_head": s["meta"].get("tree_head"),
            "binary_sha256": s["meta"].get("binary_sha256"), "ready_s": s["meta"].get("ready_s"),
            "wall_s": s["meta"].get("wall_s"), "gpu0_max_mib": gpu_max(s["dir"]), "vram_base_mib": base_mib,
            "mode_lines": grep(s["log"], r"moe-batch=on|moe-batch=off|mirror-reserve=|mirror-cache:")[:4],
            "requests": reqs, "problems": problems}


F42_METRICS = [  # (label, request, path)
    ("long: tok/s first 24", "long_200", ("client", "rate_first24")),
    ("long: tok/s after 24", "long_200", ("client", "rate_after24")),
    ("long: tok/s all", "long_200", ("client", "rate_all")),
    ("short: tok/s all", "short_200", ("client", "rate_all")),
    ("long: ttft s", "long_200", ("client", "ttft_s")),
    ("long: s/tok", "long_200", ("client", "sec_per_tok")),
    ("long: dec_moe total s/tok", "long_200", ("dec_moe_per_tok", "total_s")),
    ("long: fetch_wait s/tok", "long_200", ("dec_moe_per_tok", "fetch_wait_s")),
    ("long: attach s/tok", "long_200", ("dec_moe_per_tok", "attach_s")),
    ("long: uploads/tok", "long_200", ("dec_moe_per_tok", "uploads")),
    ("long: gpu s/tok", "long_200", ("dec_moe_per_tok", "gpu_s")),
    ("long: decprof moe ms/tok", "long_200", ("decprof", "moe_ms_per_tok")),
    ("long: decprof attn ms/tok", "long_200", ("decprof", "attn_ms_per_tok")),
    ("long: decprof moe ms/tok first 24", "long_200", ("decprof", "first_moe_ms_per_tok")),
    ("long: decprof moe ms/tok after 24", "long_200", ("decprof", "rest_moe_ms_per_tok")),
    ("long: turn hit %", "long_200", ("turn", "hit")),
    ("long: mirrors (VRAM est.)", "long_200", ("mirrors_est",)),
    ("short: s/tok", "short_200", ("client", "sec_per_tok")),
    ("short: fetch_wait s/tok", "short_200", ("dec_moe_per_tok", "fetch_wait_s")),
    ("short: attach s/tok", "short_200", ("dec_moe_per_tok", "attach_s")),
    ("short: uploads/tok", "short_200", ("dec_moe_per_tok", "uploads")),
    ("short: turn hit %", "short_200", ("turn", "hit")),
    ("short: mirrors (VRAM est.)", "short_200", ("mirrors_est",)),
]


def pick(sample, req, path):
    v = sample["requests"].get(req)
    for p in path:
        v = v.get(p) if isinstance(v, dict) else None
    return v


def summarize_f42(base):
    out = {"kind": "f42", "arms": {}}
    for arm in F42_ARMS:
        ss = samples(base, arm)
        if ss:
            out["arms"][arm] = {"samples": [f42_sample(s, arm) for s in ss]}
    texts = {}
    for arm, a in out["arms"].items():
        for x in a["samples"]:
            for name, q in x["requests"].items():
                texts.setdefault(name, set()).add(q["text_sha256"])
    out["text_identity"] = {name: {"distinct": len(v), "sha256s": sorted(x[:12] for x in v)} for name, v in texts.items()}
    table = {}
    for label, req, path in F42_METRICS:
        row = {}
        for arm, a in out["arms"].items():
            vals = [pick(x, req, path) for x in a["samples"]]
            row[arm] = {"median": med(vals), "samples": vals}
        m = {arm: row[arm]["median"] for arm in row}
        row["GB/CU"] = ratio(m.get("GB"), m.get("CU")); row["GR/CU"] = ratio(m.get("GR"), m.get("CU"))
        row["GB/GR"] = ratio(m.get("GB"), m.get("GR"))
        row["GB-CU"] = diff(m.get("GB"), m.get("CU")); row["GR-CU"] = diff(m.get("GR"), m.get("CU"))
        row["GB-GR"] = diff(m.get("GB"), m.get("GR"))
        table[label] = row
    out["table"] = table
    # where the long answer's extra decode time goes (per token, medians): total gap GB-CU =
    # reserve part GB-GR + prefill-path part GR-CU (the two arms that differ in one variable each)
    att = {}
    for label in ("long: s/tok", "long: dec_moe total s/tok", "long: fetch_wait s/tok", "long: attach s/tok",
                  "long: gpu s/tok", "long: decprof moe ms/tok", "long: decprof attn ms/tok"):
        r = table[label]
        att[label] = {"GB-CU (total)": r["GB-CU"], "GB-GR (reserve 3100->600)": r["GB-GR"],
                      "GR-CU (GPU bank prefill vs CPU union, same 600 MiB reserve)": r["GR-CU"]}
    out["attribution"] = att
    L = ["F42 decode after a long prefill: %s" % base,
         "text identity per request (all arms, all samples): %s" % {k: v["distinct"] for k, v in out["text_identity"].items()}]
    L.append("%-36s %10s %10s %10s %8s %8s %8s" % ("metric (median of samples)", "GB", "CU", "GR", "GB/CU", "GR/CU", "GB/GR"))
    for label, r in table.items():
        L.append("%-36s %10s %10s %10s %8s %8s %8s" % (label, r.get("GB", {}).get("median"), r.get("CU", {}).get("median"),
                                                     r.get("GR", {}).get("median"), r["GB/CU"], r["GR/CU"], r["GB/GR"]))
    L.append("attribution of the long answer's per-token time (GB-CU = (GB-GR) + (GR-CU)):")
    for label, r in att.items():
        L.append("  %-30s %s" % (label, r))
    for arm, a in out["arms"].items():
        for x in a["samples"]:
            w = x["requests"].get("long_200", {}).get("client", {}).get("window_rates")
            L.append("  %s s%d seq=%s gpu0_max=%s base=%s mirrors(long,short)=(%s,%s) long window tok/s=%s %s" % (
                arm, x["k"], x["seq"], x["gpu0_max_mib"], x["vram_base_mib"],
                pick(x, "long_200", ("mirrors_est",)), pick(x, "short_200", ("mirrors_est",)), w,
                ("PROBLEMS: " + "; ".join(x["problems"])[:400]) if x["problems"] else "ok"))
    return out, L


# ---------------------------------------------------------------- upload race
RACE_GROUPS = {  # phase 2: group -> (tree role, probe?)
    "D": ("dev", 0), "H": ("dev+h2d", 0), "Z": ("pr0", 0), "Dp": ("dev", 1), "Hp": ("dev+h2d", 1), "Zp": ("pr0", 1)}
PHASE3 = {"Qp": "pr1-line before the fix (48767c01)", "Fp": "pr1-line with the fix (a406901b)"}
REF_PR1 = {"long_prefill": "32534c06abd640d59d2eb39ab6609051f63e9876f533aad1d75dd5cd775835ba",
           "long_prefill_2": "1f0192038c425512912a1df7d56dca6670c50f01935e664f0219589b802cdc8d"}


def logit_lines(log):
    return [l.strip() for l in log if l.startswith("v4_logit ")]


def cmp_logits(a, b):
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), None)
    tok = next((i for i in range(n) if a[i].split()[1] != b[i].split()[1]), None)
    if first is None and len(a) != len(b):
        first = n
    return {"compared": n, "len_a": len(a), "len_b": len(b), "first_diff_line": first, "first_token_diff": tok,
            "n_diff_lines": sum(1 for i in range(n) if a[i] != b[i]),
            "first_a": a[first] if first is not None and first < len(a) else None,
            "first_b": b[first] if first is not None and first < len(b) else None}


def race_sample(s):
    ll = logit_lines(s["log"])
    rows = s["rows"]
    ex = extra_env(s)
    must = dict(COMMON, **ex)
    problems = env_problems(s, must, NEVER + [k for k in ("COLI_CUDA_MOE_DOUBLE", "DSV4_CUDA_VRAM_RESERVE_MB",
                                                          "DSV4_CUDA_EXPERT_MIRRORS", "DSV4_CUDA_PIN_HOST", "V4_LOGIT_LOG")
                                               if k not in ex])
    if s["meta"].get("client_rc") != "0":
        problems.append("client_rc=%s" % s["meta"].get("client_rc"))
    if grep(s["log"], r"out of memory|FAILED|Traceback"):
        problems.append("OOM/FAILED/Traceback lines in serve.log")
    probe = ex.get("V4_LOGIT_LOG") == "1"
    toks = sum(r.get("decode_chunks") or 0 for r in rows)
    if probe and abs(len(ll) - toks) > len(rows):
        problems.append("%d logit lines for %d streamed chunks" % (len(ll), toks))
    dec = [chunk_metrics(r) for r in rows]
    return {"arm": s["arm"], "k": s["k"], "seq": int(s["meta"].get("seq", 0) or 0), "probe": probe,
            "tree_head": s["meta"].get("tree_head"), "tree_diff_sha256": s["meta"].get("tree_diff_sha256"),
            "binary_sha256": s["meta"].get("binary_sha256"), "extra_env": ex,
            "ready_s": int(s["meta"].get("ready_s", 0) or 0), "client_s": int(s["meta"].get("client_s", 0) or 0),
            "wall_s": int(s["meta"].get("wall_s", 0) or 0), "gpu0_ready_mib": ready_mib(s["dir"]),
            "gpu0_max_mib": gpu_max(s["dir"]), "gpu1_max_mib": gpu_max(s["dir"], 1),
            "names": [r["name"] for r in rows], "text_sha256": [r["text_sha256"] for r in rows],
            "prompt_sha256": [r.get("prompt_sha256") for r in rows], "decode_s": [r.get("decode_s") for r in rows],
            "texts": [r["text"] for r in rows], "chunks": [r.get("decode_chunks") for r in rows],
            "finish": [r.get("finish_reason") for r in rows], "ttft_s": [r.get("ttft_s") for r in rows],
            "decode_tok_s": [r.get("decode_tok_s") for r in rows], "sec_per_tok": [d["sec_per_tok"] for d in dec],
            "logits": ll, "logit_lines": len(ll), "logit_seq_sha256": sha("\n".join(ll)) if ll else None,
            "cpu_moe_lines": len(grep(s["log"], r"^v4_cpu_moe ")),
            "mode_lines": grep(s["log"], r"moe-batch=on|moe-batch=off|moe-double|mirror-reserve=|mirror-cache:")[:4],
            "problems": problems}


def group_compare(xs):
    """within one group of samples: text and (probe) logit identity, every pair"""
    texts = {tuple(x["text_sha256"]) for x in xs}
    pairs = {}
    px = [x for x in xs if x["probe"]]
    for i in range(len(px)):
        for j in range(i + 1, len(px)):
            c = cmp_logits(px[i]["logits"], px[j]["logits"])
            pairs["s%d_vs_s%d" % (px[i]["k"], px[j]["k"])] = c
    text_first = []
    for x in xs[1:]:
        text_first.append([first_diff(a, b) for a, b in zip(xs[0]["texts"], x["texts"])])
    return {"n": len(xs), "n_probe": len(px), "distinct_texts": len(texts),
            "distinct_logit_seqs": len({x["logit_seq_sha256"] for x in px}),
            "logit_pairs": pairs, "text_first_diff_vs_first": text_first,
            "any_logit_diff": any(c["first_diff_line"] is not None for c in pairs.values()),
            "any_text_diff": len(texts) > 1,
            "earliest_logit_diff": min((c["first_diff_line"] for c in pairs.values() if c["first_diff_line"] is not None), default=None),
            "earliest_token_diff": min((c["first_token_diff"] for c in pairs.values() if c["first_token_diff"] is not None), default=None),
            "max_n_diff_lines": max((c["n_diff_lines"] for c in pairs.values()), default=0)}


def pilot(base):
    """phase 1: settings P_<mode><budget> (mode L = long first request, S = short; budget =
    DSV4_CUDA_EXPERT_MIRRORS or 'def'), two runs each on the dev probe build"""
    arms = sorted(os.path.basename(p) for p in glob.glob(os.path.join(base, "P_*")))
    sets = {}
    for arm in arms:
        xs = [race_sample(s) for s in samples(base, arm)]
        if not xs:
            continue
        g = group_compare(xs)
        name = arm[2:]
        m = re.match(r"([LS])(\w+)$", name)
        sets[name] = {"mode": "long" if m and m.group(1) == "L" else "short", "budget": m.group(2) if m else "?",
                      "runs": len(xs), "compare": g, "problems": [p for x in xs for p in x["problems"]],
                      "ttft_s": med([x["ttft_s"][0] for x in xs if x["ttft_s"]]),
                      "sec_per_tok": med([x["sec_per_tok"][0] for x in xs if x["sec_per_tok"]]),
                      "overhead_s": med([x["wall_s"] - x["client_s"] for x in xs]),
                      "ready_s": med([x["ready_s"] for x in xs]), "cpu_moe_lines": [x["cpu_moe_lines"] for x in xs],
                      "logit_seq": [x["logit_seq_sha256"][:12] if x["logit_seq_sha256"] else None for x in xs],
                      "seqs": [x["seq"] for x in xs], "_x": xs}
    # cross-setting check: with >= 8 mirrors every expert is computed on the GPU whatever the budget,
    # so all runs of one mode should be identical; a run that differs from that mode's majority
    # sequence diverged even when its own pair did not
    for mode in ("long", "short"):
        xs = [x for v in sets.values() if v["mode"] == mode for x in v["_x"]]
        cnt = {}
        for x in xs:
            cnt[x["logit_seq_sha256"]] = cnt.get(x["logit_seq_sha256"], 0) + 1
        maj = max(cnt.items(), key=lambda kv_: kv_[1]) if cnt else (None, 0)
        ref = next((x for x in xs if x["logit_seq_sha256"] == maj[0]), None)
        for v in sets.values():
            if v["mode"] != mode:
                continue
            v["vs_mode_majority"] = []
            for x in v["_x"]:
                c = cmp_logits(ref["logits"], x["logits"]) if ref and maj[1] >= 2 else None
                v["vs_mode_majority"].append(None if c is None else c["first_diff_line"])
            v["mode_majority_count"] = maj[1]
    cands = []
    for name, v in sets.items():
        g = v["compare"]
        firsts = [f for f in ([g["earliest_logit_diff"]] + v.get("vs_mode_majority", [])) if f is not None]
        if firsts:
            cands.append((min(firsts), -g["max_n_diff_lines"], 0 if v["mode"] == "short" else 1, name))
    if cands:
        cands.sort()
        chosen = cands[0][3]
        g = sets[chosen]["compare"]
        result = "DIFF_TEXT" if g["any_text_diff"] or g["earliest_token_diff"] is not None else "DIFF_LOGIT"
        first = cands[0][0]
    else:
        # the fallback: the smallest budget with a long first request (L8 even when the pilot had
        # no time to run a long setting; its timings then default in run_race.sh)
        lb = sorted((n for n, v in sets.items() if v["mode"] == "long" and v["budget"].isdigit()), key=lambda n: int(sets[n]["budget"]))
        chosen = lb[0] if lb else "L8"
        result, first = "NODIFF", -1
    for v in sets.values():
        v.pop("_x", None)
    c = sets.get(chosen, {})
    dec = {"CHOSEN": chosen, "CHOSEN_MODE": c.get("mode", "long" if chosen.startswith("L") else "short"),
           "CHOSEN_BUDGET": c.get("budget", chosen[1:] or "8"),
           "PILOT_RESULT": result, "FIRST_DIFF_LINE": first,
           "TTFT_S": int(round(c.get("ttft_s") or 230)), "SPT_MS": int(round(1000 * (c.get("sec_per_tok") or 1.0))),
           "OVERHEAD_S": int(round(c.get("overhead_s") or 30)), "READY_S": int(round(c.get("ready_s") or 10)),
           "CANDIDATES": " ".join("%s:%s" % (cnd[3], cnd[0]) for cnd in cands) or "none"}
    return {"settings": sets, "decision": dec}


def summarize_race(base):
    out = {"kind": "race", "pilot": pilot(base), "phase2": {}, "phase3": {}}
    for g in RACE_GROUPS:
        xs = [race_sample(s) for s in samples(base, g)]
        if xs:
            out["phase2"][g] = {"samples": xs}
    p2 = out["phase2"]
    verdict = {}
    for role, plain, probe in (("dev", "D", "Dp"), ("dev+h2d", "H", "Hp"), ("pr0", "Z", "Zp")):
        xs = p2.get(plain, {}).get("samples", []) + p2.get(probe, {}).get("samples", [])
        if not xs:
            continue
        g = group_compare(sorted(xs, key=lambda x: x["seq"]))
        verdict[role] = g
    cross = {}
    firsts = {g: p2[g]["samples"][0] for g in ("Dp", "Hp", "Zp") if p2.get(g, {}).get("samples")}
    for a, b in (("Dp", "Hp"), ("Hp", "Zp"), ("Dp", "Zp")):
        if a in firsts and b in firsts:
            cross["%s1_vs_%s1" % (a, b)] = cmp_logits(firsts[a]["logits"], firsts[b]["logits"])
    tfirst = {g: p2[g]["samples"][0]["text_sha256"] for g in ("D", "H", "Z") if p2.get(g, {}).get("samples")}
    dev_differs = bool(verdict.get("dev") and (verdict["dev"]["any_text_diff"] or verdict["dev"]["any_logit_diff"]))
    h_same = bool(verdict.get("dev+h2d") and not verdict["dev+h2d"]["any_text_diff"] and not verdict["dev+h2d"]["any_logit_diff"])
    z_same = bool(verdict.get("pr0") and not verdict["pr0"]["any_text_diff"] and not verdict["pr0"]["any_logit_diff"])
    out["phase2_verdict"] = {"per_build": verdict, "cross_build_first_probe_runs": cross,
                             "plain_first_texts_equal_across_builds": len({tuple(v) for v in tfirst.values()}) <= 1 if tfirst else None,
                             "dev_differs_among_itself": dev_differs, "h_bit_identical": h_same, "z_bit_identical": z_same,
                             "PASS": dev_differs and h_same and z_same}
    # manifest inputs: one record per sample of each group (upstream format: >= 3 samples/arm,
    # one changed variable engine_commit dev -> H); the probe groups Dp/Hp carry the same probe diff
    man = {}
    for g, a in p2.items():
        man[g] = [{"k": x["k"], "seq": x["seq"], "tree_head": x["tree_head"], "tree_diff_sha256": x["tree_diff_sha256"],
                   "binary_sha256": x["binary_sha256"], "extra_env": x["extra_env"], "tok_s": x["decode_tok_s"][0] if x["decode_tok_s"] else None,
                   "ttft_s": x["ttft_s"][0] if x["ttft_s"] else None, "text_sha256": x["text_sha256"],
                   "prompt_sha256": x["prompt_sha256"], "decode_s": x["decode_s"],
                   "logit_seq_sha256": x["logit_seq_sha256"], "chunks": x["chunks"], "finish": x["finish"],
                   "gpu0_ready_mib": x["gpu0_ready_mib"], "gpu0_max_mib": x["gpu0_max_mib"], "gpu1_max_mib": x["gpu1_max_mib"],
                   "ready_s": x["ready_s"], "wall_s": x["wall_s"], "cpu_moe_lines": x["cpu_moe_lines"],
                   "sample_dir": "%s/s%d" % (g, x["k"]), "problems": x["problems"]} for x in a["samples"]]
    out["manifest_inputs"] = man
    for g in PHASE3:
        xs = [race_sample(s) for s in samples(base, g)]
        if xs:
            cmpx = group_compare(xs)
            out["phase3"][g] = {"what": PHASE3[g], "compare": cmpx, "samples": xs,
                                "texts_vs_dev_reference": [[sha_ == REF_PR1.get(n) for n, sha_ in zip(x["names"], x["text_sha256"])] for x in xs]}
    if out["phase3"]:
        q = out["phase3"].get("Qp", {}).get("compare", {})
        f = out["phase3"].get("Fp", {}).get("compare", {})
        out["phase3_verdict"] = {"prefix_differs": bool(q.get("any_text_diff") or q.get("any_logit_diff")),
                                 "fixed_bit_identical": bool(f) and not f.get("any_text_diff") and not f.get("any_logit_diff"),
                                 "PASS": bool(q.get("any_text_diff") or q.get("any_logit_diff")) and bool(f) and not f.get("any_text_diff") and not f.get("any_logit_diff")}
    d = out["pilot"]["decision"]
    L = ["Upload race (amplified): %s" % base,
         "phase 1 pilot: %s" % " ".join("%s=%s" % kv_ for kv_ in d.items())]
    for name, v in out["pilot"]["settings"].items():
        g = v["compare"]
        L.append("  %-6s runs=%d logit_seqs=%s first_logit_diff=%s first_token_diff=%s n_diff=%s distinct_texts=%d vs_majority=%s "
                 "ttft=%s s/tok=%s cpu_moe=%s%s" % (name, v["runs"], v["logit_seq"], g["earliest_logit_diff"], g["earliest_token_diff"],
                                                     g["max_n_diff_lines"], g["distinct_texts"], v.get("vs_mode_majority"), v["ttft_s"],
                                                     v["sec_per_tok"], v["cpu_moe_lines"], (" PROBLEMS: " + "; ".join(v["problems"])[:300]) if v["problems"] else ""))
    pv = out["phase2_verdict"]
    if p2:
        L.append("phase 2: PASS=%s (dev differs among itself=%s, H bit-identical=%s, Z bit-identical=%s)" % (
            pv["PASS"], pv["dev_differs_among_itself"], pv["h_bit_identical"], pv["z_bit_identical"]))
        for role, g in pv["per_build"].items():
            L.append("  %-8s runs=%d (probe %d) distinct_texts=%d distinct_logit_seqs=%d earliest_logit_diff=%s earliest_token_diff=%s pairs=%s" % (
                role, g["n"], g["n_probe"], g["distinct_texts"], g["distinct_logit_seqs"], g["earliest_logit_diff"], g["earliest_token_diff"],
                {k: (c["first_diff_line"], c["n_diff_lines"]) for k, c in g["logit_pairs"].items()}))
        for k, c in pv["cross_build_first_probe_runs"].items():
            L.append("  cross %s: first_diff_line=%s first_token_diff=%s n_diff=%d" % (k, c["first_diff_line"], c["first_token_diff"], c["n_diff_lines"]))
        for g, rs in man.items():
            L.append("  %-3s tok/s=%s seqs=%s problems=%s" % (g, [r["tok_s"] for r in rs], [r["seq"] for r in rs],
                                                           sum(len(r["problems"]) for r in rs)))
    else:
        L.append("phase 2: not run")
    if out["phase3"]:
        L.append("phase 3 (pr1 line, GPU-bank prefill + COLI_CUDA_MOE_DOUBLE=1): %s" % out["phase3_verdict"])
        for g, v in out["phase3"].items():
            c = v["compare"]
            L.append("  %s %s: runs=%d distinct_texts=%d distinct_logit_seqs=%d earliest_logit_diff=%s texts_vs_dev_ref=%s" % (
                g, v["what"], c["n"], c["distinct_texts"], c["distinct_logit_seqs"], c["earliest_logit_diff"], v["texts_vs_dev_reference"]))
    return out, L


def strip_big(o):
    if isinstance(o, dict):
        return {k: strip_big(v) for k, v in o.items() if k not in ("texts", "logits")}
    if isinstance(o, list):
        return [strip_big(v) for v in o]
    return o


# ---------------------------------------------------------------- self test (synthetic samples)
def _fake_sample(base, arm, k, seq, extra, rows, log, nv_used, ready=7220):
    d = os.path.join(base, arm, "s%d" % k)
    os.makedirs(d)
    with open(os.path.join(d, "sample.txt"), "w") as f:
        f.write("arm=%s\nsample=%d\nseq=%d\ntree_head=x\ntree_diff_sha256=y\nbinary_sha256=z\nextra_env=%s\n"
                "ready=1\nready_s=5\nclient_s=100\nclient_rc=0\nwall_s=120\nusage_before=size=0\nusage_after=size=0\n"
                % (arm, k, seq, " ".join("%s=%s" % kv_ for kv_ in extra.items())))
    with open(os.path.join(d, "engine_environ.txt"), "w") as f:
        for kk, vv in sorted(dict(COMMON, **extra).items()):
            f.write("%s=%s\n" % (kk, vv))
    with open(os.path.join(d, "bench.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(d, "serve.log"), "w") as f:
        f.write("\n".join(log) + "\n")
    with open(os.path.join(d, "gpu_ready.csv"), "w") as f:
        f.write("index, memory.used [MiB], memory.total [MiB]\n0, %d MiB, 24576 MiB\n1, 83 MiB, 24576 MiB\n" % ready)
    with open(os.path.join(d, "nvsmi.csv"), "w") as f:
        f.write("timestamp, index, memory.used [MiB]\n")
        for t, m in nv_used:
            f.write("%s, 0, %d MiB, 24576 MiB\n" % (datetime.fromtimestamp(t).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3], m))


def _fake_req(name, t0, ttft, n, spt, text):
    ct = [round(ttft + i * spt, 4) for i in range(n)]
    return {"name": name, "t0_epoch": t0, "ttft_s": ttft, "decode_chunks": n, "max_tokens": n, "finish_reason": "length",
            "total_s": ct[-1] + 0.01, "decode_tok_s": round((n - 1) / (ct[-1] - ct[0]), 4), "chunk_t": ct,
            "text": text, "text_sha256": sha(text)}


def selftest():
    tmp = tempfile.mkdtemp(prefix="final2-selftest-")
    t0 = time.time() - 3600
    f42 = os.path.join(tmp, "f42")
    seq = 0
    for arm, (extra, _, need, _, bank) in F42_ARMS.items():
        for k in (1, 2):
            seq += 1
            spt = {"GB": 0.48, "CU": 0.44, "GR": 0.45}[arm]
            r1 = _fake_req("long_200", t0, 80.0, 200, spt, "A" * 50)
            r2 = _fake_req("short_200", t0 + 200, 2.0, 200, 0.35, "B" * 50)
            log = (["v4_gpu mirror-reserve=3100MiB (132-slot bank)"] if arm == "GB" else []) + \
                  (["v4_gpu moe-batch=on bank=132-experts"] if bank else [])
            log += ["decprof position=%d layers=43 hc=5 attn=100 moe=%d ms" % (2010 + i, 300 + (i < 24) * 50) for i in range(199)]
            log += ["v4_dec_moe calls=8557 total_s=60.00 fetch_wait_s=20.00 attach_s=10.00 uploads=900 gpu_s=5.00 (cumulative)",
                    "v4_serve_turn prompt=2011 completion=200 decode_s=95.00 tok_s=2.100 hit=60.0% expert_disk_s=1 expert_wait_s=1 "
                    "expert_matmul_s=1 attention_s=1 rss_gb=40.0 vram_first_hits_total=0 store_fetches_total=1"]
            log += ["decprof position=%d layers=43 hc=5 attn=90 moe=250 ms" % (30 + i) for i in range(199)]
            log += ["v4_dec_moe calls=17114 total_s=110.00 fetch_wait_s=35.00 attach_s=18.00 uploads=1500 gpu_s=10.00 (cumulative)",
                    "v4_serve_turn prompt=31 completion=200 decode_s=70.00 tok_s=2.800 hit=70.0% expert_disk_s=1 expert_wait_s=1 "
                    "expert_matmul_s=1 attention_s=1 rss_gb=40.0 vram_first_hits_total=0 store_fetches_total=2"]
            nv = [(t0 + i, 7400 + bank) for i in range(1, 80)] + [(t0 + 81 + i, 7400 + min(i * 100, 14000)) for i in range(400)]
            _fake_sample(f42, arm, k, seq, dict(extra), [r1, r2], log, nv)
    out, L = summarize_f42(f42)
    assert out["text_identity"]["long_200"]["distinct"] == 1, out["text_identity"]
    assert all(not x["problems"] for a in out["arms"].values() for x in a["samples"]), \
        [x["problems"] for a in out["arms"].values() for x in a["samples"]]
    assert out["table"]["long: tok/s all"]["GB/CU"] < 1
    assert out["table"]["long: decprof moe ms/tok first 24"]["GB"]["median"] == 350.0, out["table"]["long: decprof moe ms/tok first 24"]
    assert out["table"]["long: mirrors (VRAM est.)"]["GB"]["median"] is not None
    race = os.path.join(tmp, "race")
    seq = 0
    good = ["v4_logit tok=%d logit=0x1p+5 hidden=%016x" % (i, i) for i in range(120)]
    bad = good[:40] + ["v4_logit tok=%d logit=0x1.1p+5 hidden=%016x" % (i, i + 7) for i in range(40, 120)]
    for name, run_logs in (("P_L8", (good, bad)), ("P_S8", (good, good)), ("P_S48", (good, good))):
        for k, lg in enumerate(run_logs, 1):
            seq += 1
            ex = {"DSV4_CUDA_PIN_HOST": "0", "V4_LOGIT_LOG": "1", "DSV4_CUDA_EXPERT_MIRRORS": re.sub(r"\D", "", name)}
            _fake_sample(race, name, k, seq, ex, [_fake_req("race", t0, 230.0, 120, 0.8, "T")], lg, [(t0 + 1, 7400)])
    p = pilot(race)
    dd = p["decision"]
    assert dd["CHOSEN"] == "L8" and dd["PILOT_RESULT"] == "DIFF_LOGIT" and dd["FIRST_DIFF_LINE"] == 40, dd
    for g, logs in (("Dp", (good, bad, good)), ("Hp", (good, good, good)), ("Zp", (good, good, good)),
                    ("D", (None, None)), ("H", (None, None)), ("Z", (None, None))):
        for k, lg in enumerate(logs, 1):
            seq += 1
            ex = {"DSV4_CUDA_PIN_HOST": "0", "DSV4_CUDA_EXPERT_MIRRORS": "8"}
            if lg is not None:
                ex["V4_LOGIT_LOG"] = "1"
            _fake_sample(race, g, k, seq, ex, [_fake_req("race", t0, 230.0, 120, 0.8, "T")], lg or [], [(t0 + 1, 7400)])
    out, L = summarize_race(race)
    v = out["phase2_verdict"]
    assert v["PASS"] and v["dev_differs_among_itself"] and v["h_bit_identical"] and v["z_bit_identical"], v
    assert len(out["manifest_inputs"]["Hp"]) == 3
    json.dumps(strip_big(out))
    print("selftest ok (%s)" % tmp)
    return 0


def main(argv):
    if len(argv) >= 2 and argv[0] == "brief":
        brief(argv[1]); return 0
    if len(argv) >= 1 and argv[0] == "selftest":
        return selftest()
    if len(argv) >= 2 and argv[0] == "racepilot":
        p = pilot(argv[1])
        json.dump(strip_big(p), open(os.path.join(argv[1], "pilot.json"), "w"), indent=1)
        for k, v in p["decision"].items():
            print("%s=%s" % (k, v))
        return 0
    if len(argv) >= 3 and argv[0] == "summarize":
        fn = {"f42": summarize_f42, "race": summarize_race}[argv[1]]
        out, L = fn(argv[2])
        out["usage_history_changed"] = usage_changed(argv[2])
        L.append("usage history (.coli_usage) changed during: %s" % (out["usage_history_changed"] or "no sample"))
        json.dump(strip_big(out), open(os.path.join(argv[2], "summary.json"), "w"), indent=1)
        print("\n".join(L))
        return 0
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
