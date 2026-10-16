#!/usr/bin/env python3
"""F42 fix local analysis (computer4) of the extracted f42fix.tgz (run_f42fix.sh on shadowfax): text
identity, decode tok/s per arm, FX vs OX vs UX ratios, how much of the OX-UX gap the fix recovers,
per-forward misses / fetch wait in token windows 1-5 / 6-10 / 11-24 (+ the EOS forward 25), RAM tier
and pins at decode start, routing identity, a routing/residency swap, prefill TTFT FX vs OX.
Self-contained parsing (sample.txt, bench.jsonl, serve.log, nvsmi.csv); CPU only. Adapted from
analyze_f42v2_local.py (f42v2: arms GB/U/GBn on the pr2 tip; here FX/UX/OX/FXn on the PR #1 line).

Arms: FX = pr1/v5 tip 4feb6d48 (routing-books fix c7f79815) + probe, GPU-bank prefill; UX = FX +
COLI_CUDA_MOE_BATCH_MIN=100000 (CPU union, same 3100 MiB reserve); OX = old pr1 tip 93b74183 (no
fix) + the same probe, GPU-bank prefill; FXn = FX's env on the plain 4feb6d48 build; W = warm-up.

usage: analyze_f42fix_local.py RAW_DIR [OUT_JSON]     (RAW_DIR = the f42fix/ directory of f42fix.tgz)
Decode forward j (1..25) consumes emitted token ordinal j and produces ordinal j+1; ordinal 1 is the
prefill's token. Client decode tok/s = 24 / (chunk_t[24] - chunk_t[0]) (bench_common: decode_tok_s).
The PR #1 probe (f42fix_probe_patch.py) prints calls / total_s / fetch_wait_s / attach_s (its own
counters, measured as pr2's v4_dec_moe) and the store's req / hits / misses / bytes / disk_s / wait_s;
no upload counter.
"""
import glob, json, math, os, re, statistics as st, sys

RAW = os.path.abspath(sys.argv[1])
OUTJ = sys.argv[2] if len(sys.argv) > 2 else None
ARMS = ("W", "FX", "UX", "OX", "FXn")
REAL = ("FX", "UX", "OX", "FXn")
PROBE = ("W", "FX", "UX", "OX")
KEYS = ("calls", "total_s", "fetch_wait_s", "attach_s", "req", "hits", "misses", "bytes", "disk_s", "wait_s")
WINDOWS = ((1, 5), (6, 10), (11, 24), (25, 25), (1, 24), (1, 25))
PAIRS = (("OX", "UX"), ("FX", "UX"), ("FX", "OX"))
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
            if f["ord"] == 1 and trace:
                break
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
                res.append({"layer": int(m.group(1)), "lreq": int(m.group(2)), "clock": int(m.group(3)), "n": int(m.group(5)),
                            "pinned_resident": int(m.group(6)), "res": r,
                            "pins": [int(x) for x in m.group(7).split(",") if x not in ("", "-1")]})
    prof = [fields(l) for l in log if l.startswith("decprof ")]
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
            "trace": trace, "fw": fw, "layers": layers, "res": res, "start": start, "prof": prof,
            "reserve": reserve, "bank_lines": bank, "refill_lines": refill,
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
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    d = st.mean(a) - st.mean(b)
    if va + vb == 0:
        return {"diff": round(d, 5), "t": 0.0 if d == 0 else (math.inf if d > 0 else -math.inf)}
    return {"diff": round(d, 5), "t": round(d / math.sqrt(va + vb), 2)}


def recovered(fx, ox, ux):
    """share of the OX-UX gap FX closes: (FX - OX) / (UX - OX); 1 = all of it, 0 = none"""
    if None in (fx, ox, ux) or ux == ox:
        return None
    return round((fx - ox) / (ux - ox), 3)


out = {"raw": RAW, "samples": {}, "problems": []}
# ---------------------------------------------------------------- per-sample sanity
for a, v in S.items():
    for x in v:
        p = []
        if x["meta"].get("client_rc") != "0":
            p.append("client_rc")
        if x["reserve"] != ["3100"]:
            p.append("reserve %s" % x["reserve"])
        if a == "UX":
            if x["bank_lines"] or x["refill_lines"]:
                p.append("bank used in UX")
            if not x["start"] or x["start"].get("bank") != 0 or not x["start"].get("union"):
                p.append("UX prefill paths %s" % (x["start"] and {k: x["start"].get(k) for k in ("bank", "union", "rows")}))
            if x["env"].get("COLI_CUDA_MOE_BATCH_MIN") != "100000":
                p.append("UX without MIN")
        else:
            if not x["bank_lines"] or not x["refill_lines"]:
                p.append("bank not used")
            if "COLI_CUDA_MOE_BATCH_MIN" in x["env"]:
                p.append("MIN set")
            if a in PROBE and (not x["start"] or x["start"].get("union") != 0 or not x["start"].get("bank")):
                p.append("%s prefill paths %s" % (a, x["start"] and {k: x["start"].get(k) for k in ("bank", "union", "rows")}))
        if a in PROBE and len(x["trace"]) < 2:
            p.append("no trace")
        if a == "FXn" and (x["trace"] or x["layers"] or x["res"]):
            p.append("probe lines in FXn")
        if p:
            out["problems"].append("%s s%d: %s" % (a, x["k"], "; ".join(p)))

# ---------------------------------------------------------------- text identity
TEXT = {a: sorted({x["text_sha"] for x in v}) for a, v in S.items()}
all_real = sorted({x["text_sha"] for a in REAL if a in S for x in S[a]})
out["text_identity"] = {"per_arm": TEXT, "distinct_FX_UX_OX_FXn": len(all_real), "shas": all_real,
                        "chunks": {a: sorted({x["chunks"] for x in v}) for a, v in S.items()},
                        "finish": {a: sorted({x["finish"] for x in v}) for a, v in S.items()},
                        "text": S["FX"][0]["rec"]["text"]}


def last(x, k):
    return x["trace"][max(x["trace"])].get(k) if x["trace"] else None


# ---------------------------------------------------------------- per-arm speed
SPEED = {}
for a, v in S.items():
    SPEED[a] = {"client_decode_tok_s": stat([x["tok_s"] for x in v]),
                "client_decode_s": stat([x["decode_s"] for x in v]),
                "ttft_s": stat([x["ttft"] for x in v]),
                "server_decode_s_fwd1_25": stat([last(x, "t") for x in v]),
                "dec_moe_total_s": stat([last(x, "total_s") for x in v]),
                "fetch_wait_s": stat([last(x, "fetch_wait_s") for x in v]),
                "store_wait_s": stat([last(x, "wait_s") for x in v]),
                "attach_s": stat([last(x, "attach_s") for x in v]),
                "store_misses": stat([last(x, "misses") for x in v]),
                "gpu0_max_mib": stat([x["gpu0_max"] for x in v]),
                "decprof_moe_s": stat([sum(p.get("moe", 0) for p in x["prof"]) / 1000 for x in v]),
                "decprof_attn_s": stat([sum(p.get("attn", 0) for p in x["prof"]) / 1000 for x in v]),
                "per_sample": [{"k": x["k"], "seq": x["seq"], "tok_s": x["tok_s"], "ttft": x["ttft"],
                                "decode_s": x["decode_s"], "misses": last(x, "misses"), "fetch_wait_s": last(x, "fetch_wait_s"),
                                "gpu0_max": x["gpu0_max"], "text": x["text_sha"][:12]} for x in v]}
out["speed"] = SPEED
tok = {a: [x["tok_s"] for x in S[a]] for a in S}
mean = {a: st.mean(v) for a, v in tok.items()}
med = {a: st.median(v) for a, v in tok.items()}
CMP = {}
for a, b in PAIRS:
    CMP["%s/%s" % (a, b)] = {"median_ratio": round(med[a] / med[b], 4), "mean_ratio": round(mean[a] / mean[b], 4),
                             "welch_tok_s": welch(tok[a], tok[b]),
                             "ttft_welch": welch([x["ttft"] for x in S[a]], [x["ttft"] for x in S[b]]),
                             "ttft_median_ratio": round(st.median([x["ttft"] for x in S[a]]) / st.median([x["ttft"] for x in S[b]]), 4)}
if "FXn" in S:
    CMP["FXn/FX"] = {"median_ratio": round(med["FXn"] / med["FX"], 4), "mean_ratio": round(mean["FXn"] / mean["FX"], 4),
                     "welch_tok_s": welch(tok["FXn"], tok["FX"]),
                     "ttft_welch": welch([x["ttft"] for x in S["FXn"]], [x["ttft"] for x in S["FX"]])}
out["compare"] = CMP
out["recovered_tok_s"] = {"means": recovered(mean["FX"], mean["OX"], mean["UX"]),
                          "medians": recovered(med["FX"], med["OX"], med["UX"])}

# ---------------------------------------------------------------- windows (per forward deltas)
WK = ("t_ms", "client_ms", "fetch_wait_s", "wait_s", "attach_s", "misses", "hits", "bytes", "total_s", "disk_s")


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
        row = {a: stat([wsum(x, lo, hi, k) for x in S[a]]) for a in PROBE if a in S}
        vals = {a: [wsum(x, lo, hi, k) for x in S[a]] for a in ("FX", "UX", "OX")}
        if all(None not in v for v in vals.values()):
            m_ = {a: st.mean(v) for a, v in vals.items()}
            for a, b in PAIRS:
                row["%s-%s_mean" % (a, b)] = round(m_[a] - m_[b], 4)
                row["welch %s-%s" % (a, b)] = welch(vals[a], vals[b])
            row["recovered"] = recovered(m_["FX"], m_["OX"], m_["UX"])
        WIN[w][k] = row
out["windows"] = WIN

# per-forward curve
CURVE = []
for j in range(1, 26):
    row = {"fwd": j}
    for a in ("FX", "UX", "OX"):
        xs = [f for x in S[a] for f in x["fw"] if f["fwd"] == j]
        row[a] = {k: round(st.mean(f[k] for f in xs), 4) for k in ("t_ms", "fetch_wait_s", "attach_s", "misses", "hits")}
        row[a + "_misses_range"] = [min(f["misses"] for f in xs), max(f["misses"] for f in xs)]
    CURVE.append(row)
out["per_forward"] = CURVE


# ---------------------------------------------------------------- routing identity (decode)
def routing(x):
    return {(o, l): tuple(v["routed"]) for (o, l), v in x["layers"].items()}


ref = routing(S["FX"][0])
rdiff = {}
for a in PROBE:
    for x in S[a]:
        r = routing(x)
        nd = sum(1 for key in set(ref) | set(r) if ref.get(key) != r.get(key))
        rdiff["%s/s%d" % (a, x["k"])] = {"token_layers": len(r), "differ_from_FX_s1": nd}
out["decode_routing"] = rdiff
out["misses_per_sample"] = {a: [last(x, "misses") for x in v] for a, v in S.items()}


# ---------------------------------------------------------------- residency at decode start
def resid(x):
    if not x["res"]:
        return None
    L = x["res"]
    used, first = {}, {}
    for (o, l), v in x["layers"].items():
        used.setdefault(l, set()).update(v["routed"])
        if o == 1:
            first[l] = set(v["routed"])
    needed = sum(len(s) for s in used.values())
    resident_needed = sum(len(set(r["res"]) & used.get(r["layer"], set())) for r in L)
    first_needed = sum(len(s) for s in first.values())
    first_resident = sum(len(set(r["res"]) & first.get(r["layer"], set())) for r in L)
    usage = [u for r in L for (u, _, _) in r["res"].values()]
    pin_usage = [r["res"][e][0] for r in L for e in r["pins"] if e in r["res"]]
    lowest = sum(1 for r in L if r["pins"] and r["pins"] == list(range(len(r["pins"]))))
    low_res = sum(1 for r in L if r["pins"] and r["pins"] == sorted(r["res"])[:len(r["pins"])])
    # the 16 pins vs the 16 most-used residents of the layer (usage ties count as matches)
    top_match = []
    for r in L:
        if r["pins"]:
            us = sorted((u for (u, _, _) in r["res"].values()), reverse=True)
            cut = us[min(len(r["pins"]), len(us)) - 1] if us else 0
            top_match.append(sum(1 for e in r["pins"] if e in r["res"] and r["res"][e][0] >= cut) / len(r["pins"]))
    # recency: rank correlation between age at decode start and first decode use is not well defined
    # for 25 tokens; report the mean age of decode-needed vs not-needed residents instead
    age_needed = [r["res"][e][1] for r in L for e in r["res"] if e in used.get(r["layer"], set())]
    age_other = [r["res"][e][1] for r in L for e in r["res"] if e not in used.get(r["layer"], set())]
    return {"layers": len(L), "resident_mean": round(st.mean(r["n"] for r in L), 2),
            "resident_total": sum(r["n"] for r in L),
            "pinned_resident_mean": round(st.mean(r["pinned_resident"] for r in L), 2),
            "pins_per_layer": round(st.mean(len(r["pins"]) for r in L), 2),
            "resident_usage_le1_frac": round(sum(1 for q in usage if q <= 1) / len(usage), 4),
            "resident_usage_mean": round(st.mean(usage), 3), "resident_usage_max": max(usage),
            "pin_usage_median": st.median(pin_usage) if pin_usage else None, "pin_usage_max": max(pin_usage) if pin_usage else None,
            "pins_are_top_usage_frac": round(st.mean(top_match), 4) if top_match else None,
            "layers_pins_ids_0_to_15": lowest, "layers_pins_lowest_resident_ids": low_res,
            "lreq_mean": round(st.mean(r["lreq"] for r in L), 1),
            "decode_needed_distinct": needed, "decode_needed_resident_at_start": resident_needed,
            "first_fwd_needed": first_needed, "first_fwd_resident_at_start": first_resident,
            "age_mean_decode_needed_residents": round(st.mean(age_needed), 1) if age_needed else None,
            "age_mean_other_residents": round(st.mean(age_other), 1) if age_other else None,
            "_sets": {r["layer"]: set(r["res"]) for r in L}}


RES = {a: [resid(x) for x in S[a]] for a in PROBE if a in S}


def jacc(s1, s2):
    tot = inter = 0
    for l in s1:
        inter += len(s1[l] & s2.get(l, set()))
        tot += len(s1[l] | s2.get(l, set()))
    return round(inter / tot, 4) if tot else None


out["residency"] = {a: [{k: v for k, v in r.items() if k != "_sets"} for r in v if r] for a, v in RES.items()}
RES_KEYS = ("resident_mean", "pinned_resident_mean", "pins_per_layer", "resident_usage_le1_frac", "resident_usage_mean",
            "pin_usage_median", "pins_are_top_usage_frac", "layers_pins_ids_0_to_15", "layers_pins_lowest_resident_ids",
            "lreq_mean", "decode_needed_distinct", "decode_needed_resident_at_start", "first_fwd_needed",
            "first_fwd_resident_at_start", "age_mean_decode_needed_residents", "age_mean_other_residents")
out["residency_summary"] = {a: {k: stat([r[k] for r in v if r and r[k] is not None]) for k in RES_KEYS}
                            for a, v in RES.items()}
out["residency_set_jaccard"] = {
    "FX_s1_vs_FX_others": [jacc(RES["FX"][0]["_sets"], r["_sets"]) for r in RES["FX"][1:]],
    "UX_s1_vs_UX_others": [jacc(RES["UX"][0]["_sets"], r["_sets"]) for r in RES["UX"][1:]],
    "OX_s1_vs_OX_others": [jacc(RES["OX"][0]["_sets"], r["_sets"]) for r in RES["OX"][1:]],
    "FX_s1_vs_UX": [jacc(RES["FX"][0]["_sets"], r["_sets"]) for r in RES["UX"]],
    "OX_s1_vs_UX": [jacc(RES["OX"][0]["_sets"], r["_sets"]) for r in RES["UX"]],
    "FX_s1_vs_OX": [jacc(RES["FX"][0]["_sets"], r["_sets"]) for r in RES["OX"]]}
out["prefill_store"] = {a: [{k: x["start"].get(k) for k in ("bank", "union", "rows", "req", "hits", "misses", "bytes", "disk_s", "wait_s")}
                            for x in S[a] if x["start"]] for a in PROBE if a in S}


# ---------------------------------------------------------------- decomposition of the decode time
def mm(a, f):
    return st.mean(f(x) for x in S[a])


dec = {}
for k, f in (("client_decode_s(24 intervals)", lambda x: x["decode_s"]),
             ("server_fwd1_25_s", lambda x: last(x, "t")),
             ("dec_moe_total_s", lambda x: last(x, "total_s")), ("fetch_wait_s", lambda x: last(x, "fetch_wait_s")),
             ("store_wait_s", lambda x: last(x, "wait_s")), ("attach_s", lambda x: last(x, "attach_s")),
             ("store_misses", lambda x: last(x, "misses")),
             ("store_GiB_read", lambda x: last(x, "bytes") / 2 ** 30),
             ("store_disk_s", lambda x: last(x, "disk_s")),
             ("non_moe_s", lambda x: last(x, "t") - last(x, "total_s"))):
    row = {a: round(mm(a, f), 4) for a in ("FX", "UX", "OX")}
    for a, b in PAIRS:
        row["%s-%s" % (a, b)] = round(row[a] - row[b], 4)
        row["welch %s-%s" % (a, b)] = welch([f(x) for x in S[a]], [f(x) for x in S[b]])
    row["recovered"] = recovered(row["FX"], row["OX"], row["UX"])
    dec[k] = row
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
for arm in ("FX", "OX"):
    for i, g in enumerate(S[arm]):
        for j, uu in enumerate(S["UX"]):
            gs, us = RES[arm][i]["_sets"], RES["UX"][j]["_sets"]
            nG, nU, fG, fU = needed(g), needed(uu), first_routed(g), first_routed(uu)
            rows = {
                "needed_resident %s-routing x %s-RAM" % (arm, arm): sum(len(nG[l] & gs[l]) for l in nG),
                "needed_resident %s-routing x UX-RAM" % arm: sum(len(nG[l] & us[l]) for l in nG),
                "needed_resident UX-routing x UX-RAM (pairs with %s)" % arm: sum(len(nU[l] & us[l]) for l in nU),
                "needed_resident UX-routing x %s-RAM" % arm: sum(len(nU[l] & gs[l]) for l in nU),
                "fwd1_resident %s-routing x %s-RAM" % (arm, arm): sum(len(fG[l] & gs[l]) for l in fG),
                "fwd1_resident %s-routing x UX-RAM" % arm: sum(len(fG[l] & us[l]) for l in fG),
            }
            same = [k for k in g["layers"] if set(g["layers"][k]["routed"]) == set(uu["layers"][k]["routed"])]
            sset = set(same)
            rows["token_layers_same_routing %s vs UX" % arm] = len(same)
            rows["misses_on_same_routing_token_layers %s" % arm] = sum(g["layers"][k]["misses"] for k in same)
            rows["misses_on_same_routing_token_layers UX (pairs with %s)" % arm] = sum(uu["layers"][k]["misses"] for k in same)
            for k, v in rows.items():
                CF.setdefault(k, []).append(v)
out["counterfactual_pairs"] = {k: {"min": min(v), "max": max(v), "mean": round(st.mean(v), 2), "n": len(v)} for k, v in CF.items()}
out["first_touch"] = {a: [{"needed": r["decode_needed_distinct"], "not_resident_at_start":
                           r["decode_needed_distinct"] - r["decode_needed_resident_at_start"],
                           "misses": last(x, "misses")}
                          for x, r in zip(S[a], RES[a]) if r] for a in ("FX", "UX", "OX")}


def strip(o):
    if isinstance(o, dict):
        return {str(k): strip(v) for k, v in o.items() if k != "_sets"}
    if isinstance(o, (list, tuple)):
        return [strip(v) for v in o]
    if isinstance(o, set):
        return sorted(o)
    if isinstance(o, float) and math.isinf(o):
        return "inf" if o > 0 else "-inf"
    return o


if OUTJ:
    json.dump(strip(out), open(OUTJ, "w"), indent=1)

# ---------------------------------------------------------------- print
P = print
P("problems:", out["problems"] or "none")
P("texts:", {a: [t[:12] for t in v] for a, v in TEXT.items()}, "distinct over FX/UX/OX/FXn:", len(all_real))
P("chunks:", out["text_identity"]["chunks"], "finish:", out["text_identity"]["finish"])
for a in S:
    s = SPEED[a]["client_decode_tok_s"]
    P("%-3s tok/s n=%d mean=%.4f stdev=%s median=%.4f min=%.4f max=%.4f  ttft median %.2f (mean %.2f)  gpu0max %s" % (
        a, s["n"], s["mean"], s["stdev"], s["median"], s["min"], s["max"], SPEED[a]["ttft_s"]["median"],
        SPEED[a]["ttft_s"]["mean"], SPEED[a]["gpu0_max_mib"]["median"]))
    P("     per sample (k, seq, tok/s, ttft):", [(p["k"], p["seq"], p["tok_s"], p["ttft"]) for p in SPEED[a]["per_sample"]])
for k, v in CMP.items():
    P("%-7s %s" % (k, v))
P("recovered share of the OX-UX tok/s gap:", out["recovered_tok_s"])
P("decomposition (means): FX / UX / OX, pair differences, recovered = (FX-OX)/(UX-OX)")
for k, v in dec.items():
    P("   %-30s FX %s UX %s OX %s | OX-UX %+.4f (t=%s) FX-UX %+.4f (t=%s) FX-OX %+.4f (t=%s) | recovered %s" % (
        k, v["FX"], v["UX"], v["OX"], v["OX-UX"], (v["welch OX-UX"] or {}).get("t"), v["FX-UX"], (v["welch FX-UX"] or {}).get("t"),
        v["FX-OX"], (v["welch FX-OX"] or {}).get("t"), v["recovered"]))
P("windows (decode forwards), means FX / UX / OX, OX-UX, FX-UX, recovered")
for w, d in WIN.items():
    for k in ("client_ms", "t_ms", "misses", "fetch_wait_s", "wait_s", "attach_s"):
        if "recovered" in d[k]:
            P("  fwd %-6s %-13s FX %10.4g UX %10.4g OX %10.4g | OX-UX %+9.4g (t=%s) FX-UX %+9.4g (t=%s) | recovered %s" % (
                w, k, d[k]["FX"]["mean"], d[k]["UX"]["mean"], d[k]["OX"]["mean"], d[k]["OX-UX_mean"],
                (d[k]["welch OX-UX"] or {}).get("t"), d[k]["FX-UX_mean"], (d[k]["welch FX-UX"] or {}).get("t"), d[k]["recovered"]))
P("per forward: misses FX / UX / OX (ranges) | fetch_wait ms FX / UX / OX | t_ms FX / UX / OX")
for r in CURVE:
    P("  %2d  %6.1f %6.1f %6.1f  %s %s %s | %6.1f %6.1f %6.1f | %6.1f %6.1f %6.1f" % (
        r["fwd"], r["FX"]["misses"], r["UX"]["misses"], r["OX"]["misses"], r["FX_misses_range"], r["UX_misses_range"],
        r["OX_misses_range"], 1000 * r["FX"]["fetch_wait_s"], 1000 * r["UX"]["fetch_wait_s"], 1000 * r["OX"]["fetch_wait_s"],
        r["FX"]["t_ms"], r["UX"]["t_ms"], r["OX"]["t_ms"]))
P("decode routing vs FX s1:", rdiff)
P("misses per sample:", out["misses_per_sample"])
for a, v in out["residency_summary"].items():
    P("res %-2s" % a, {k: (s["median"] if s else None) for k, s in v.items()})
P("residency jaccard:", out["residency_set_jaccard"])
P("prefill store:", out["prefill_store"])
P("counterfactual (arm x UX pairs):")
for k, v in out["counterfactual_pairs"].items():
    P("   %-58s %s" % (k, v))
P("first touch:", out["first_touch"])
