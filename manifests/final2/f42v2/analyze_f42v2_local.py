#!/usr/bin/env python3
"""F42 v2 local analysis (computer4) of the extracted f42v2.tgz: text identity, decode tok/s per arm,
per-forward GB-U curves in token windows 1-5 / 6-10 / 11-24 (+ the EOS forward 25), RAM residency at
decode start, routing identity, a routing/residency swap counterfactual, attach per upload by layer.
Self-contained parsing (sample.txt, bench.jsonl, serve.log, nvsmi.csv); CPU only, reads nothing else.

usage: analyze_f42v2_local.py RAW_DIR [OUT_JSON]     (RAW_DIR = the f42v2/ directory of f42v2.tgz)
  e.g. tar -C /tmp/x -xzf f42v2.tgz && ./analyze_f42v2_local.py /tmp/x/f42v2 stats.json > analysis.txt
Decode forward j (1..25) consumes emitted token ordinal j and produces ordinal j+1; ordinal 1 is the
prefill's token. The client sees 25 content chunks (ordinals 1..25); forward 25 produces EOS (tok=1).
Client decode tok/s = 24 / (chunk_t[24] - chunk_t[0]) (bench_common: decode_tok_s).
"""
import glob, json, math, os, re, statistics as st, sys

RAW = os.path.abspath(sys.argv[1])
OUTJ = sys.argv[2] if len(sys.argv) > 2 else None
ARMS = ("W", "GB", "U", "GBn")
REAL = ("GB", "U", "GBn")
KEYS = ("calls", "total_s", "fetch_wait_s", "attach_s", "uploads", "gpu_s", "store_fetches",
        "req", "hits", "misses", "bytes", "disk_s", "wait_s")
WINDOWS = ((1, 5), (6, 10), (11, 24), (25, 25), (1, 24), (1, 25))
NUM = re.compile(r"^-?\d+(\.\d+)?$")


def fields(line):
    out = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            if NUM.match(v):
                out[k] = float(v) if "." in v else int(v)
            else:
                out[k] = v
    return out


def kv(path):
    d = {}
    for l in open(path):
        if "=" in l:
            k, v = l.rstrip("\n").split("=", 1)
            d[k] = v
    return d


def load(arm, sdir):
    meta = kv(os.path.join(sdir, "sample.txt"))
    rec = json.loads(next(l for l in open(os.path.join(sdir, "bench.jsonl")) if l.startswith("{")))
    log = open(os.path.join(sdir, "serve.log"), errors="replace").read().splitlines()
    trace, layers, res, start = {}, {}, [], None
    for l in log:
        if l.startswith("v4_dec_trace_start "):
            start = fields(l)
        elif l.startswith("v4_dec_trace "):
            f = fields(l)
            trace[f["ord"]] = f
        elif l.startswith("v4_dec_layer "):
            f = fields(l)
            m = re.search(r"routed=([\d,]*)", l)
            f["routed"] = [int(x) for x in m.group(1).split(",") if x]
            layers[(f["ord"], f["layer"])] = f
        elif l.startswith("v4_dec_res "):
            m = re.search(r" layer=(\d+) lreq=(\d+) clock=(\d+) res=(\S*) n=(\d+) pinned_resident=(\d+) pins=(\S*)", l)
            if m:
                r = {}
                for item in filter(None, m.group(4).split(",")):
                    p = item.endswith("p")
                    e, u, age = item.rstrip("p").split(":")
                    r[int(e)] = (int(u), int(age), p)
                res.append({"layer": int(m.group(1)), "lreq": int(m.group(2)), "n": int(m.group(5)),
                            "pinned_resident": int(m.group(6)), "res": r,
                            "pins": [int(x) for x in m.group(7).split(",") if x not in ("", "-1")]})
    turn = next((fields(l) for l in log if l.startswith("v4_serve_turn ")), {})
    decmoe = next((fields(l) for l in log if l.startswith("v4_dec_moe ")), {})
    reserve = re.findall(r"mirror-reserve=(\d+)MiB", "\n".join(log))
    bank = sum(1 for l in log if "moe-batch=on bank=" in l)
    refill = sum(1 for l in log if "moe-batch refill " in l)
    gpu0 = []
    for l in open(os.path.join(sdir, "nvsmi.csv")):
        p = [x.strip() for x in l.split(",")]
        if len(p) > 3 and p[1] == "0" and p[2].endswith("MiB"):
            gpu0.append(int(p[2].split()[0]))
    ct = rec["chunk_t"]
    fw = []
    for j in sorted(trace):
        if j + 1 in trace:
            a, b = trace[j], trace[j + 1]
            d = {k: b.get(k, 0) - a.get(k, 0) for k in KEYS}
            d["t_ms"] = 1000 * (b["t"] - a["t"])
            d["fwd"] = j
            d["client_ms"] = 1000 * (ct[j] - ct[j - 1]) if j < len(ct) else None
            fw.append(d)
    return {"arm": arm, "k": int(meta["sample"]), "seq": int(meta["seq"]), "meta": meta, "rec": rec,
            "text_sha": rec["text_sha256"], "ttft": rec["ttft_s"], "tok_s": rec["decode_tok_s"],
            "decode_s": rec["decode_s"], "chunks": rec["decode_chunks"], "finish": rec["finish_reason"],
            "trace": trace, "fw": fw, "layers": layers, "res": res, "start": start, "turn": turn,
            "decmoe": decmoe, "reserve": reserve, "bank_lines": bank, "refill_lines": refill,
            "gpu0_max": max(gpu0) if gpu0 else None, "binary": meta["binary_sha256"],
            "env": dict(x.split("=", 1) for x in meta["server_env"].split())}


S = {a: [load(a, d) for d in sorted(glob.glob(os.path.join(RAW, a, "s*")), key=lambda p: int(p.rsplit("s", 1)[1]))]
     for a in ARMS}
S = {a: v for a, v in S.items() if v}


def stat(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"n": len(xs), "mean": round(st.mean(xs), 4), "stdev": round(st.stdev(xs), 4) if len(xs) > 1 else None,
            "median": round(st.median(xs), 4), "min": round(min(xs), 4), "max": round(max(xs), 4)}


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    if va + vb == 0:
        return {"diff": st.mean(a) - st.mean(b), "t": math.inf if st.mean(a) != st.mean(b) else 0.0}
    t = (st.mean(a) - st.mean(b)) / math.sqrt(va + vb)
    return {"diff": round(st.mean(a) - st.mean(b), 5), "t": round(t, 2)}


out = {"raw": RAW, "samples": {}, "problems": []}
# ---------------------------------------------------------------- per-sample sanity
for a, v in S.items():
    for x in v:
        p = []
        if x["meta"].get("client_rc") != "0":
            p.append("client_rc")
        if x["reserve"] != ["3100"]:
            p.append("reserve %s" % x["reserve"])
        if a == "U":
            if x["bank_lines"] or x["refill_lines"]:
                p.append("bank used in U")
            if not x["start"] or x["start"].get("bank") != 0 or not x["start"].get("union"):
                p.append("U prefill paths %s" % (x["start"] and {k: x["start"].get(k) for k in ("bank", "union", "rows")}))
            if x["env"].get("COLI_CUDA_MOE_BATCH_MIN") != "100000":
                p.append("U without MIN")
        else:
            if not x["bank_lines"] or not x["refill_lines"]:
                p.append("bank not used")
            if "COLI_CUDA_MOE_BATCH_MIN" in x["env"]:
                p.append("MIN set")
        if a in ("W", "GB", "U") and len(x["trace"]) < 2:
            p.append("no trace")
        if a == "GBn" and (x["trace"] or x["layers"] or x["res"]):
            p.append("probe lines in GBn")
        if p:
            out["problems"].append("%s s%d: %s" % (a, x["k"], "; ".join(p)))

# ---------------------------------------------------------------- text identity
TEXT = {a: sorted({x["text_sha"] for x in v}) for a, v in S.items()}
all_real = sorted({x["text_sha"] for a in REAL if a in S for x in S[a]})
out["text_identity"] = {"per_arm": TEXT, "distinct_GB_U_GBn": len(all_real), "shas": all_real,
                        "chunks": {a: sorted({x["chunks"] for x in v}) for a, v in S.items()},
                        "finish": {a: sorted({x["finish"] for x in v}) for a, v in S.items()},
                        "text": S["GB"][0]["rec"]["text"]}

# ---------------------------------------------------------------- per-arm speed
SPEED = {}
for a, v in S.items():
    SPEED[a] = {"client_decode_tok_s": stat([x["tok_s"] for x in v]),
                "client_decode_s": stat([x["decode_s"] for x in v]),
                "ttft_s": stat([x["ttft"] for x in v]),
                "server_decode_s_fwd1_25": stat([x["trace"][max(x["trace"])]["t"] for x in v]) if v[0]["trace"] else None,
                "engine_turn_tok_s": stat([x["turn"].get("tok_s") for x in v]),
                "dec_moe_total_s": stat([x["decmoe"].get("total_s") for x in v]),
                "fetch_wait_s": stat([x["decmoe"].get("fetch_wait_s") for x in v]),
                "attach_s": stat([x["decmoe"].get("attach_s") for x in v]),
                "uploads": stat([x["decmoe"].get("uploads") for x in v]),
                "gpu_s": stat([x["decmoe"].get("gpu_s") for x in v]),
                "gpu0_max_mib": stat([x["gpu0_max"] for x in v]),
                "per_sample": [{"k": x["k"], "seq": x["seq"], "tok_s": x["tok_s"], "ttft": x["ttft"],
                                "decode_s": x["decode_s"], "dec_moe": x["decmoe"], "gpu0_max": x["gpu0_max"],
                                "text": x["text_sha"][:12]} for x in v]}
out["speed"] = SPEED
g, u = [x["tok_s"] for x in S["GB"]], [x["tok_s"] for x in S["U"]]
out["speed_GB_vs_U"] = {"median_ratio_GB/U": round(st.median(g) / st.median(u), 4),
                        "mean_ratio_GB/U": round(st.mean(g) / st.mean(u), 4), "welch_tok_s": welch(g, u)}
if "GBn" in S:
    gn = [x["tok_s"] for x in S["GBn"]]
    out["speed_GBn_vs_GB"] = {"median_ratio_GBn/GB": round(st.median(gn) / st.median(g), 4),
                              "mean_ratio_GBn/GB": round(st.mean(gn) / st.mean(g), 4), "welch_tok_s": welch(gn, g)}

# ---------------------------------------------------------------- windows (per forward deltas)
WK = ("t_ms", "client_ms", "fetch_wait_s", "attach_s", "uploads", "misses", "hits", "bytes", "gpu_s", "total_s", "disk_s", "wait_s")


def wsum(x, lo, hi, k):
    sel = [f for f in x["fw"] if lo <= f["fwd"] <= hi]
    if not sel or any(f.get(k) is None for f in sel):
        return None
    return sum(f[k] for f in sel)


WIN = {}
for lo, hi in WINDOWS:
    w = "%d-%d" % (lo, hi)
    WIN[w] = {}
    for k in WK:
        row = {}
        for a in ("W", "GB", "U"):
            if a in S:
                row[a] = stat([wsum(x, lo, hi, k) for x in S[a]])
        gb = [wsum(x, lo, hi, k) for x in S["GB"]]
        uu = [wsum(x, lo, hi, k) for x in S["U"]]
        if None not in gb and None not in uu:
            row["GB-U_mean"] = round(st.mean(gb) - st.mean(uu), 4)
            row["GB-U_median"] = round(st.median(gb) - st.median(uu), 4)
            row["welch"] = welch(gb, uu)
        WIN[w][k] = row
out["windows"] = WIN

# cumulative at the end of each window
CUM = []
for n in (1, 2, 3, 5, 10, 15, 20, 24, 25):
    row = {"through_fwd": n}
    for k in ("t_ms", "client_ms", "fetch_wait_s", "attach_s", "uploads", "misses", "bytes"):
        gb = [wsum(x, 1, n, k) for x in S["GB"]]
        uu = [wsum(x, 1, n, k) for x in S["U"]]
        if None not in gb and None not in uu:
            row[k] = {"GB": round(st.mean(gb), 4), "U": round(st.mean(uu), 4), "GB-U": round(st.mean(gb) - st.mean(uu), 4)}
    CUM.append(row)
out["cumulative"] = CUM

# per-forward curve
CURVE = []
for j in range(1, 26):
    row = {"fwd": j}
    for a in ("GB", "U"):
        xs = [f for x in S[a] for f in x["fw"] if f["fwd"] == j]
        row[a] = {k: round(st.mean(f[k] for f in xs), 4) for k in ("t_ms", "fetch_wait_s", "attach_s", "uploads", "misses", "hits", "gpu_s")}
        row[a + "_misses_range"] = [min(f["misses"] for f in xs), max(f["misses"] for f in xs)]
    row["GB-U"] = {k: round(row["GB"][k] - row["U"][k], 4) for k in row["GB"]}
    CURVE.append(row)
out["per_forward"] = CURVE

# ---------------------------------------------------------------- routing identity (decode)
def routing(x):
    return {(o, l): tuple(v["routed"]) for (o, l), v in x["layers"].items()}


ref = routing(S["GB"][0])
rdiff = {}
for a in ("W", "GB", "U"):
    for x in S[a]:
        r = routing(x)
        nd = sum(1 for key in set(ref) | set(r) if ref.get(key) != r.get(key))
        rdiff["%s/s%d" % (a, x["k"])] = {"token_layers": len(r), "differ_from_GB_s1": nd}
out["decode_routing"] = rdiff
# uploads per sample (deterministic?)
out["uploads_per_sample"] = {a: [x["decmoe"].get("uploads") for x in v] for a, v in S.items()}
out["misses_per_sample"] = {a: [x["trace"][max(x["trace"])]["misses"] if x["trace"] else None for x in v] for a, v in S.items()}

# ---------------------------------------------------------------- residency at decode start
def resid(x):
    if not x["res"]:
        return None
    L = x["res"]
    used = {}   # all experts routed in decode forwards 1..25 per layer
    first = {}
    for (o, l), v in x["layers"].items():
        used.setdefault(l, set()).update(v["routed"])
        if o == 1:
            first[l] = set(v["routed"])
    # first-use order: which experts are needed in decode and were resident at start
    needed = sum(len(s) for s in used.values())
    resident_needed = sum(len(set(r["res"]) & used.get(r["layer"], set())) for r in L)
    first_needed = sum(len(s) for s in first.values())
    first_resident = sum(len(set(r["res"]) & first.get(r["layer"], set())) for r in L)
    usage = [u for r in L for (u, _, _) in r["res"].values()]
    pin_usage = [r["res"][e][0] for r in L for e in r["pins"] if e in r["res"]]
    lowest = sum(1 for r in L if r["pins"] and r["pins"] == list(range(len(r["pins"]))))
    # pins equal to the 16 lowest RESIDENT ids (ties of usage=1 broken by id)
    low_res = sum(1 for r in L if r["pins"] and r["pins"] == sorted(r["res"])[:len(r["pins"])])
    return {"layers": len(L), "resident_mean": round(st.mean(r["n"] for r in L), 2),
            "resident_total": sum(r["n"] for r in L),
            "pinned_resident_mean": round(st.mean(r["pinned_resident"] for r in L), 2),
            "pins_per_layer": round(st.mean(len(r["pins"]) for r in L), 2),
            "resident_usage_le1_frac": round(sum(1 for q in usage if q <= 1) / len(usage), 4),
            "resident_usage_mean": round(st.mean(usage), 3), "resident_usage_max": max(usage),
            "pin_usage_median": st.median(pin_usage) if pin_usage else None, "pin_usage_max": max(pin_usage) if pin_usage else None,
            "layers_pins_ids_0_to_15": lowest, "layers_pins_lowest_resident_ids": low_res,
            "lreq_mean": round(st.mean(r["lreq"] for r in L), 1),
            "decode_needed_distinct": needed, "decode_needed_resident_at_start": resident_needed,
            "first_fwd_needed": first_needed, "first_fwd_resident_at_start": first_resident,
            "_sets": {r["layer"]: set(r["res"]) for r in L}}


RES = {}
for a in ("W", "GB", "U"):
    RES[a] = [resid(x) for x in S[a]]
# set identity across samples of an arm and between arms
def jacc(s1, s2):
    tot = inter = 0
    for l in s1:
        inter += len(s1[l] & s2.get(l, set()))
        tot += len(s1[l] | s2.get(l, set()))
    return round(inter / tot, 4) if tot else None


out["residency"] = {a: [{k: v for k, v in r.items() if k != "_sets"} for r in v if r] for a, v in RES.items()}
out["residency_set_jaccard"] = {
    "GB_s1_vs_GB_others": [jacc(RES["GB"][0]["_sets"], r["_sets"]) for r in RES["GB"][1:]],
    "U_s1_vs_U_others": [jacc(RES["U"][0]["_sets"], r["_sets"]) for r in RES["U"][1:]],
    "GB_s1_vs_U": [jacc(RES["GB"][0]["_sets"], r["_sets"]) for r in RES["U"]]}
out["prefill_store"] = {a: [{k: x["start"].get(k) for k in ("bank", "union", "rows", "req", "hits", "misses", "bytes", "disk_s", "wait_s")}
                            for x in S[a] if x["start"]] for a in ("W", "GB", "U")}

# ---------------------------------------------------------------- decomposition of the GB-U decode time
def m(a, f):
    return st.mean(f(x) for x in S[a])


dec = {}
for k, f in (("client_decode_s(24 intervals)", lambda x: x["decode_s"]),
             ("server_fwd1_25_s", lambda x: x["trace"][max(x["trace"])]["t"]),
             ("dec_moe_total_s", lambda x: x["decmoe"]["total_s"]), ("fetch_wait_s", lambda x: x["decmoe"]["fetch_wait_s"]),
             ("attach_s", lambda x: x["decmoe"]["attach_s"]), ("gpu_s", lambda x: x["decmoe"]["gpu_s"]),
             ("uploads", lambda x: x["decmoe"]["uploads"]),
             ("store_misses", lambda x: x["trace"][max(x["trace"])]["misses"]),
             ("store_GiB_read", lambda x: x["trace"][max(x["trace"])]["bytes"] / 2 ** 30),
             ("store_disk_s", lambda x: x["trace"][max(x["trace"])]["disk_s"]),
             ("store_wait_s", lambda x: x["trace"][max(x["trace"])]["wait_s"])):
    dec[k] = {"GB": round(m("GB", f), 4), "U": round(m("U", f), 4), "GB-U": round(m("GB", f) - m("U", f), 4),
              "welch": welch([f(x) for x in S["GB"]], [f(x) for x in S["U"]])}
dec["non_moe_s (server_fwd - dec_moe_total)"] = {
    a: round(m(a, lambda x: x["trace"][max(x["trace"])]["t"] - x["decmoe"]["total_s"]), 4) for a in ("GB", "U")}
out["decomposition"] = dec


# ---------------------------------------------------------------- routing / residency swap counterfactual
def needed(x):
    n = {}
    for (o, l), v in x["layers"].items():
        n.setdefault(l, set()).update(v["routed"])
    return n


def first_routed(x):
    return {l: set(v["routed"]) for (o, l), v in x["layers"].items() if o == 1}


CF = {}
for g in S["GB"]:
    for uu in S["U"]:
        gs, us = RES["GB"][S["GB"].index(g)]["_sets"], RES["U"][S["U"].index(uu)]["_sets"]
        nG, nU, fG, fU = needed(g), needed(uu), first_routed(g), first_routed(uu)
        rows = {
            "decode_needed_resident_at_start GB-routing x GB-RAM": sum(len(nG[l] & gs[l]) for l in nG),
            "decode_needed_resident_at_start GB-routing x U-RAM": sum(len(nG[l] & us[l]) for l in nG),
            "decode_needed_resident_at_start U-routing x U-RAM": sum(len(nU[l] & us[l]) for l in nU),
            "decode_needed_resident_at_start U-routing x GB-RAM": sum(len(nU[l] & gs[l]) for l in nU),
            "fwd1_routed_resident GB-routing x GB-RAM": sum(len(fG[l] & gs[l]) for l in fG),
            "fwd1_routed_resident GB-routing x U-RAM": sum(len(fG[l] & us[l]) for l in fG),
            "fwd1_routed_resident U-routing x U-RAM": sum(len(fU[l] & us[l]) for l in fU),
            "fwd1_routed_resident U-routing x GB-RAM": sum(len(fU[l] & gs[l]) for l in fU),
        }
        same = [k for k in g["layers"] if set(g["layers"][k]["routed"]) == set(uu["layers"][k]["routed"])]
        sset = set(same)
        rows["token_layers_same_routing"] = len(same)
        rows["misses_on_same_routing_token_layers GB"] = sum(g["layers"][k]["misses"] for k in same)
        rows["misses_on_same_routing_token_layers U"] = sum(uu["layers"][k]["misses"] for k in same)
        rows["misses_on_differing_token_layers GB"] = sum(v["misses"] for k, v in g["layers"].items() if k not in sset)
        rows["misses_on_differing_token_layers U"] = sum(v["misses"] for k, v in uu["layers"].items() if k not in sset)
        rows["expert_selections_differing (of 6450)"] = sum(len(set(g["layers"][k]["routed"]) - set(uu["layers"][k]["routed"]))
                                                         for k in g["layers"])
        rows["layers_with_any_routing_difference"] = len({k[1] for k in g["layers"] if k not in sset})
        rows["first_layer_with_routing_difference"] = min((k[1] for k in g["layers"] if k not in sset), default=None)
        for k, v in rows.items():
            CF.setdefault(k, []).append(v)
out["counterfactual_25_pairs"] = {k: {"min": min(v), "max": max(v), "mean": round(st.mean(v), 2)} for k, v in CF.items()}
# first-touch misses: decode-needed (layer, expert) not resident at start (lower bound of a sample's misses)
out["first_touch"] = {a: [{"needed": r["decode_needed_distinct"], "not_resident_at_start":
                           r["decode_needed_distinct"] - r["decode_needed_resident_at_start"],
                           "misses": x["trace"][max(x["trace"])]["misses"]}
                          for x, r in zip(S[a], RES[a]) if r] for a in ("GB", "U")}


# ---------------------------------------------------------------- attach per upload by layer group / forward
def apu(x, f0, f1, l0, l1):
    sel = [v for (o, l), v in x["layers"].items() if f0 <= o <= f1 and l0 <= l <= l1]
    up = sum(v["uploads"] for v in sel)
    return round(sum(v["attach_ms"] for v in sel) / up, 3) if up else None


out["attach_ms_per_upload"] = {"%s/s%d" % (a, x["k"]): {"fwd1-7 all layers": apu(x, 1, 7, 0, 42),
                                                        "fwd8-25 L0-31": apu(x, 8, 25, 0, 31),
                                                        "fwd8-25 L32-41": apu(x, 8, 25, 32, 41),
                                                        "fwd8-25 L42": apu(x, 8, 25, 42, 42)}
                               for a in ("W", "GB", "U") for x in S[a]}

# ---------------------------------------------------------------- robustness: GB without its first sample (seq 2)
GB_LATE = [x for x in S["GB"] if x["k"] != 1]
rob = {"GB_s2-s5_tok_s": stat([x["tok_s"] for x in GB_LATE]),
       "median_ratio_GB(s2-s5)/U": round(st.median([x["tok_s"] for x in GB_LATE]) / st.median(u), 4)}
for lo, hi in ((1, 5), (6, 10), (11, 24), (1, 24)):
    for k in ("client_ms", "fetch_wait_s", "attach_s", "misses", "uploads"):
        rob["fwd %d-%d %s GB(s2-s5)-U" % (lo, hi, k)] = round(st.mean(wsum(x, lo, hi, k) for x in GB_LATE) -
                                                              st.mean(wsum(x, lo, hi, k) for x in S["U"]), 4)
out["robustness_GB_without_s1"] = rob

# ---------------------------------------------------------------- serve-unit memory peaks (host journal, optional)
JOUR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence", "f42v2-host-journal.txt")
if os.path.exists(JOUR):
    # the journal's Started lines are cut before the serve.log path, so the i-th Started/Consumed pair is
    # the sample with seq i (one serve unit per sample, started in run order); checked against sample.txt start=
    pairs, started = [], None
    for l in open(JOUR):
        mm = re.match(r"(\S+) \S+ systemd\[\d+\]: Started colibri-final2-f42v2-serve\.service", l)
        if mm:
            started = mm.group(1); continue
        mm = re.search(r"colibri-final2-f42v2-serve\.service: Consumed .*?, ([\d.]+)G memory peak", l)
        if mm and started:
            pairs.append((started, float(mm.group(1)))); started = None
    by_seq = {x["seq"]: x for v in S.values() for x in v}
    peaks = {}
    for i, (t, g) in enumerate(pairs, 1):
        x = by_seq.get(i)
        if x and x["meta"]["start"][:16] == t[:16]:
            peaks["%s/s%d" % (x["arm"], x["k"])] = g
        else:
            out["problems"].append("journal pair %d (%s) does not match sample seq %d" % (i, t, i))
    out["serve_unit_memory_peak_gib"] = peaks


def strip(o):
    if isinstance(o, dict):
        return {str(k): strip(v) for k, v in o.items() if k != "_sets"}
    if isinstance(o, (list, tuple)):
        return [strip(v) for v in o]
    if isinstance(o, set):
        return sorted(o)
    return o


if OUTJ:
    json.dump(strip(out), open(OUTJ, "w"), indent=1)

# ---------------------------------------------------------------- print
P = print
P("problems:", out["problems"] or "none")
P("texts:", {a: [t[:12] for t in v] for a, v in TEXT.items()}, "distinct over GB/U/GBn:", len(all_real))
P("chunks:", out["text_identity"]["chunks"], "finish:", out["text_identity"]["finish"])
for a in S:
    s = SPEED[a]["client_decode_tok_s"]
    P("%-3s tok/s n=%d mean=%.4f stdev=%s median=%.4f min=%.4f max=%.4f  ttft median %.2f  gpu0max %s" % (
        a, s["n"], s["mean"], s["stdev"], s["median"], s["min"], s["max"], SPEED[a]["ttft_s"]["median"],
        SPEED[a]["gpu0_max_mib"]["median"]))
    P("     per sample:", [(p["k"], p["seq"], p["tok_s"]) for p in SPEED[a]["per_sample"]])
P("GB vs U:", out["speed_GB_vs_U"]); P("GBn vs GB:", out.get("speed_GBn_vs_GB"))
P("decomposition (means):")
for k, v in dec.items():
    P("   %-42s %s" % (k, v))
P("windows (forwards), mean over samples: GB, U, GB-U (welch t)")
for w, d in WIN.items():
    P("  fwd %-6s" % w + "  ".join("%s: %s/%s/%+.4g(t=%s)" % (k, d[k]["GB"]["mean"], d[k]["U"]["mean"], d[k]["GB-U_mean"],
                                                           (d[k]["welch"] or {}).get("t"))
                                      for k in ("client_ms", "t_ms", "fetch_wait_s", "misses", "uploads", "attach_s") if "GB-U_mean" in d[k]))
P("per forward GB-U: t_ms fetch_ms misses uploads attach_ms | GB misses U misses")
for r in CURVE:
    d = r["GB-U"]
    P("  %2d %+8.1f %+8.1f %+6.1f %+6.1f %+7.1f | %6.1f %6.1f  ranges %s %s" % (
        r["fwd"], d["t_ms"], 1000 * d["fetch_wait_s"], d["misses"], d["uploads"], 1000 * d["attach_s"],
        r["GB"]["misses"], r["U"]["misses"], r["GB_misses_range"], r["U_misses_range"]))
P("decode routing vs GB s1:", rdiff)
P("uploads per sample:", out["uploads_per_sample"]); P("misses per sample:", out["misses_per_sample"])
for a, v in out["residency"].items():
    for r in v:
        P("res %-2s" % a, r)
P("residency jaccard:", out["residency_set_jaccard"])
P("prefill store:", out["prefill_store"])
P("counterfactual (25 GB x U pairs):")
for k, v in out["counterfactual_25_pairs"].items():
    P("   %-62s %s" % (k, v))
P("first touch:", out["first_touch"])
P("attach ms per upload:")
for k, v in out["attach_ms_per_upload"].items():
    P("   %-7s %s" % (k, v))
P("robustness (GB without s1):", json.dumps(out["robustness_GB_without_s1"]))
P("serve-unit memory peaks GiB:", out.get("serve_unit_memory_peak_gib"))
