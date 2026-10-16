#!/usr/bin/env python3
"""Summary of run_f42fix.sh (F42 fix: decode after a GPU-bank prefill with the routing-books fix
c7f79815, pr1/v5 tip 4feb6d48, vs the unfixed old pr1 tip 93b74183 and vs the CPU union, on
IDENTICAL text, equal 3100 MiB mirror reserve). CPU only; reads final2/f42fix/<arm>/s<k>/.

  analyze_f42fix.py summarize DIR   writes DIR/summary.json, prints a readable summary
  analyze_f42fix.py selftest        runs the summary on synthetic samples (run_f42fix.sh preflight)

Arms: W (warm-up, FX config, discarded), FX (fixed probe build r9-pr1v5-probe, GPU-bank prefill),
UX (FX + COLI_CUDA_MOE_BATCH_MIN=100000: CPU union, same reserve), OX (unfixed probe build
r9-pr1-probe, GPU-bank prefill), FXn (FX's env on the plain fixed build r9-pr1v5: the probe
changes neither speed nor text). Probe lines (f42fix_probe_patch.py, the PR #1 adaptation of
f42v2_probe_patch.py): v4_dec_trace_start, v4_dec_res, v4_dec_trace (one per emitted token,
cumulative from ordinal 1), v4_dec_layer. Reuses analyze_f42v2.py's parsers (trace_rows,
layer_rows, residency) and final_tools.py.
Fix prediction: FX's decode-start books (usage, request clock, active pins) look like UX's, RAM
misses in the first decode tokens drop towards UX's, and FX recovers part or all of OX-UX.
"""
import json, math, os, re, statistics, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_tools as ft
import analyze_f42v2 as v2

REF_TEXT = ft.REF_PR1["long_prefill"]
STATS_ENV = {"DSV4_DECODE_PROF": "1", "V4_DEC_TRACE": "2"}
ARMS = {
    # arm: (env beyond COMMON, env that must be unset, probe build?, prefill path wanted)
    "W": (dict(STATS_ENV), ["COLI_CUDA_MOE_BATCH_MIN"], True, "bank"),
    "FX": (dict(STATS_ENV), ["COLI_CUDA_MOE_BATCH_MIN"], True, "bank"),
    "UX": (dict(STATS_ENV, COLI_CUDA_MOE_BATCH_MIN="100000"), [], True, "union"),
    "OX": (dict(STATS_ENV), ["COLI_CUDA_MOE_BATCH_MIN"], True, "bank"),
    "FXn": (dict(STATS_ENV), ["COLI_CUDA_MOE_BATCH_MIN"], False, "bank"),
}
UNSET = ["DSV4_CUDA_VRAM_RESERVE_MB", "COLI_CUDA_MOE_DOUBLE", "DSV4_CUDA_EXPERT_MIRRORS", "DSV4_CUDA_PIN_HOST",
         "V4_LOGIT_LOG", "V4_EXPERT_UNION", "V4_SERVE_STATS_LOG", "COLI_V4_PREWARM", "V4_MOE_BANK_FULL"] + ft.NEVER
TRACE_KEYS = ("calls", "total_s", "fetch_wait_s", "attach_s", "req", "hits", "misses", "bytes", "disk_s", "wait_s")
WINDOWS = ((1, 5), (6, 10), (11, 24), (25, 25), (1, 24), (1, 25))   # decode forwards (1 = the first)
WIN_KEYS = ("t_ms", "fetch_wait_s", "wait_s", "attach_s", "misses", "hits", "bytes", "total_s")


def per_forward(rows):
    """per decode forward j: the counters it added (cum[ord j+1] - cum[ord j]) and its time"""
    out = []
    for j in sorted(rows):
        if j + 1 in rows:
            a, b = rows[j], rows[j + 1]
            d = {k: round(b.get(k, 0) - a.get(k, 0), 6) for k in TRACE_KEYS}
            d["t_ms"] = round(1000 * (b.get("t", 0) - a.get("t", 0)), 2)
            d["fwd"] = j
            out.append(d)
    return out


def res_extra(res, lay):
    """decode-start residency against the decode's own routing (as analyze_f42v2_local.py)"""
    if not res:
        return {}
    sets = res["_sets"]
    used, first = {}, {}
    for (o, l), v in lay.items():
        used.setdefault(l, set()).update(v["routed"])
        if o == 1:
            first[l] = set(v["routed"])
    return {"decode_needed_distinct": sum(len(s) for s in used.values()),
            "decode_needed_resident_at_start": sum(len(set(sets.get(l, [])) & s) for l, s in used.items()),
            "first_fwd_needed": sum(len(s) for s in first.values()),
            "first_fwd_resident_at_start": sum(len(set(sets.get(l, [])) & s) for l, s in first.items())}


def lreq_mean(log):
    xs = [int(m.group(1)) for l in log if l.startswith("v4_dec_res ") for m in [re.search(r" lreq=(\d+) ", l)] if m]
    return round(statistics.mean(xs), 1) if xs else None


def sample(s, arm):
    extra, unset, probe, path = ARMS[arm]
    probs = ft.env_problems(s, dict(ft.COMMON, **extra), unset + UNSET)
    log = s["log"]
    if s["meta"].get("client_rc") != "0":
        probs.append("client_rc=%s" % s["meta"].get("client_rc"))
    if ft.grep(log, r"out of memory|FAILED|Traceback"):
        probs.append("%d OOM/FAILED/Traceback lines" % len(ft.grep(log, r"out of memory|FAILED|Traceback")))
    reserve = [m for l in log for m in re.findall(r"mirror-reserve=(\d+)MiB", l)]
    bank_on = len(ft.grep(log, r"moe-batch=on bank="))
    refills = len(ft.grep(log, r"moe-batch refill "))
    if reserve != ["3100"]:
        probs.append("mirror-reserve lines %s (want exactly one, 3100)" % reserve)
    if ft.grep(log, r"moe-batch=off"):
        probs.append("moe-batch=off line: %s" % ft.grep(log, r"moe-batch=off")[0][:120])
    start = next((ft.fields(l) for l in log if l.startswith("v4_dec_trace_start ")), None)
    if path == "bank" and (not bank_on or not refills):
        probs.append("GPU bank not used (moe-batch=on lines %d, refill lines %d)" % (bank_on, refills))
    if path == "union" and (bank_on or refills):
        probs.append("GPU bank used in the CPU-union arm (moe-batch=on lines %d, refill lines %d)" % (bank_on, refills))
    if probe:
        if not start:
            probs.append("no v4_dec_trace_start line")
        elif path == "bank" and not (start.get("bank", 0) > 0 and start.get("union", 0) == 0):
            probs.append("prefill chunk-layers bank=%s union=%s (want bank>0, union=0)" % (start.get("bank"), start.get("union")))
        elif path == "union" and not (start.get("bank", 0) == 0 and start.get("union", 0) > 0):
            probs.append("prefill chunk-layers bank=%s union=%s (want bank=0, union>0)" % (start.get("bank"), start.get("union")))
    elif ft.grep(log, r"^v4_dec_(trace|trace_start|layer|res) "):
        probs.append("probe lines in a non-probe sample")
    r = s["rows"][0] if s["rows"] else {}
    if r.get("text_sha256") != REF_TEXT:
        probs.append("text %s is not the reference %s" % (str(r.get("text_sha256"))[:12], REF_TEXT[:12]))
    cm = ft.chunk_metrics(r) if r else {"n": 0, "interval_ms": []}
    rows = v2.trace_rows(log)
    fw = per_forward(rows)
    lay = v2.layer_rows(log)
    first = sorted([v for (o, _), v in lay.items() if o == 1], key=lambda v: v.get("layer", 0))
    win = {}
    for a, b in WINDOWS:
        sel = [x for x in fw if a <= x["fwd"] <= b]
        win["%d-%d" % (a, b)] = {k: round(sum(x[k] for x in sel), 4) for k in WIN_KEYS} if sel else None
    res = v2.residency(log)
    if probe and not res:
        probs.append("no v4_dec_res lines")
    rx = res_extra(res, lay) if res else {}
    cum = {k: round(rows[max(rows)].get(k, 0), 4) for k in TRACE_KEYS} if rows else {}
    prof = [ft.fields(l) for l in log if l.startswith("decprof ")]
    return {
        "k": s["k"], "seq": s["meta"].get("seq"), "binary_sha256": s["meta"].get("binary_sha256"),
        "wall_s": s["meta"].get("wall_s"), "ttft_s": r.get("ttft_s"), "chunks": cm.get("n"),
        "finish": r.get("finish_reason"), "text_sha256": r.get("text_sha256"),
        "client_tok_s": cm.get("rate_all"), "client_decode_s": round(sum(cm["interval_ms"]) / 1000, 3),
        "interval_ms": cm["interval_ms"], "reserve": reserve, "bank_on_lines": bank_on, "refill_lines": refills,
        "prefill_paths": {k: start.get(k) for k in ("bank", "union", "rows")} if start else None,
        "prefill_store": {k: start.get(k) for k in ("req", "hits", "misses", "bytes", "disk_s", "wait_s")} if start else None,
        "trace_total": cum, "forwards": fw, "windows": win,
        "server_decode_s": round(rows[max(rows)].get("t", 0), 4) if rows else None,
        "first_forward": {"hits": sum(v.get("hits", 0) for v in first), "misses": sum(v.get("misses", 0) for v in first),
                          "fetch_ms": round(sum(v.get("fetch_ms", 0) for v in first), 2),
                          "wait_ms": round(sum(v.get("wait_ms", 0) for v in first), 2),
                          "layers": len(first)} if first else None,
        "first_routed": {int(v["layer"]): v["routed"] for v in first},
        "decprof_moe_ms": [round(x.get("moe", 0)) for x in prof],
        "residency": dict(res, lreq_mean=lreq_mean(log), **rx) if res else None,
        "gpu0_max_mib": ft.gpu_max(s["dir"]), "problems": probs}


def welch(a, b):
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    d = statistics.mean(a) - statistics.mean(b)
    if va + vb == 0:
        return {"diff": round(d, 5), "t": None if d == 0 else (math.inf if d > 0 else -math.inf)}
    return {"diff": round(d, 5), "t": round(d / math.sqrt(va + vb), 2)}


def recovered(fx, ox, ux):
    """share of the OX-UX gap the fix closes: (FX - OX) / (UX - OX), on means"""
    if None in (fx, ox, ux) or ux == ox:
        return None
    return round((fx - ox) / (ux - ox), 3)


def summarize(base):
    out = {"kind": "f42fix", "arms": {}}
    for arm in ARMS:
        ss = ft.samples(base, arm)
        if ss:
            out["arms"][arm] = [sample(s, arm) for s in ss]
    real = {a: v for a, v in out["arms"].items() if a != "W"}
    out["text_identity"] = sorted({x["text_sha256"] for v in real.values() for x in v if x["text_sha256"]})
    metric = {
        "client tok/s": lambda x: x["client_tok_s"],
        "client decode s": lambda x: x["client_decode_s"],
        "ttft s": lambda x: x["ttft_s"],
        "server decode s (trace)": lambda x: x["server_decode_s"],
        "dec moe total s": lambda x: (x["trace_total"] or {}).get("total_s"),
        "fetch_wait s": lambda x: (x["trace_total"] or {}).get("fetch_wait_s"),
        "store wait s": lambda x: (x["trace_total"] or {}).get("wait_s"),
        "attach s": lambda x: (x["trace_total"] or {}).get("attach_s"),
        "store misses (SSD reads)": lambda x: (x["trace_total"] or {}).get("misses"),
        "store hits": lambda x: (x["trace_total"] or {}).get("hits"),
        "store GiB read": lambda x: round(x["trace_total"]["bytes"] / 2 ** 30, 3) if x["trace_total"] else None,
        "first fwd RAM hits (of 258)": lambda x: (x["first_forward"] or {}).get("hits"),
        "first fwd misses": lambda x: (x["first_forward"] or {}).get("misses"),
        "decode-start resident/layer": lambda x: (x["residency"] or {}).get("resident_mean"),
        "decode-start pinned residents/layer": lambda x: (x["residency"] or {}).get("pinned_resident_mean"),
        "decode-start pins/layer": lambda x: (x["residency"] or {}).get("pins_mean"),
        "decode-start pin usage median": lambda x: (x["residency"] or {}).get("pin_usage_median"),
        "decode-start layers pins=ids 0..15": lambda x: (x["residency"] or {}).get("layers_pins_are_lowest_ids"),
        "decode-start residents usage<=1/layer": lambda x: (x["residency"] or {}).get("resident_usage<=1_mean"),
        "decode-start layer requests mean": lambda x: (x["residency"] or {}).get("lreq_mean"),
        "decode-needed resident at start": lambda x: (x["residency"] or {}).get("decode_needed_resident_at_start"),
        "decode-needed distinct": lambda x: (x["residency"] or {}).get("decode_needed_distinct"),
        "prefill store requests": lambda x: (x["prefill_store"] or {}).get("req"),
        "prefill store misses": lambda x: (x["prefill_store"] or {}).get("misses"),
        "gpu0 max MiB": lambda x: x["gpu0_max_mib"],
    }
    for a, b in WINDOWS:
        w = "%d-%d" % (a, b)
        for k in ("t_ms", "fetch_wait_s", "wait_s", "misses", "attach_s"):
            metric["fwd %s %s" % (w, k)] = (lambda x, w=w, k=k: (x["windows"].get(w) or {}).get(k))
    table = {}
    for label, f in metric.items():
        row = {arm: {"median": ft.med([f(x) for x in v]), "samples": [f(x) for x in v],
                     "mean": (round(statistics.mean([f(x) for x in v if f(x) is not None]), 4)
                              if any(f(x) is not None for x in v) else None)} for arm, v in real.items()}
        md = {arm: row[arm]["median"] for arm in row}
        mn = {arm: row[arm]["mean"] for arm in row}
        row["FX/OX"] = ft.ratio(md.get("FX"), md.get("OX"))
        row["FX/UX"] = ft.ratio(md.get("FX"), md.get("UX"))
        row["OX/UX"] = ft.ratio(md.get("OX"), md.get("UX"))
        row["FXn/FX"] = ft.ratio(md.get("FXn"), md.get("FX"))
        row["recovered (means)"] = recovered(mn.get("FX"), mn.get("OX"), mn.get("UX"))
        row["welch FX-OX"] = welch(row.get("FX", {}).get("samples", []), row.get("OX", {}).get("samples", []))
        row["welch FX-UX"] = welch(row.get("FX", {}).get("samples", []), row.get("UX", {}).get("samples", []))
        table[label] = row
    out["table"] = table
    curve = []
    for j in range(1, 26):
        row = {"fwd": j}
        for arm in ("FX", "UX", "OX"):
            xs = [f for x in real.get(arm, []) for f in x["forwards"] if f["fwd"] == j]
            if xs:
                row[arm] = {k: round(statistics.mean(f[k] for f in xs), 4) for k in ("t_ms", "fetch_wait_s", "attach_s", "misses", "hits")}
        curve.append(row)
    out["per_forward"] = curve
    routes = {json.dumps(x["first_routed"], sort_keys=True) for arm in ("FX", "UX", "OX") for x in real.get(arm, []) if x["first_routed"]}
    out["first_forward_routing_distinct"] = len(routes)
    L = ["F42 fix: decode after a GPU-bank prefill, fixed (FX) vs unfixed (OX) vs CPU union (UX), identical text: %s" % base,
         "texts (all arms but W): %s  reference %s" % ([t[:12] for t in out["text_identity"]], REF_TEXT[:12]),
         "first-forward routing distinct across FX/UX/OX probe samples: %s" % out["first_forward_routing_distinct"],
         "%-38s %9s %9s %9s %9s %7s %7s %7s %7s %7s" % ("metric (median of samples)", "FX", "UX", "OX", "FXn",
                                                        "FX/OX", "FX/UX", "OX/UX", "FXn/FX", "recov")]
    for label, r in table.items():
        L.append("%-38s %9s %9s %9s %9s %7s %7s %7s %7s %7s" % (
            label, r.get("FX", {}).get("median"), r.get("UX", {}).get("median"), r.get("OX", {}).get("median"),
            r.get("FXn", {}).get("median"), r["FX/OX"], r["FX/UX"], r["OX/UX"], r["FXn/FX"], r["recovered (means)"]))
    L.append("welch client tok/s FX-OX %s  FX-UX %s" % (table["client tok/s"]["welch FX-OX"], table["client tok/s"]["welch FX-UX"]))
    L.append("per decode forward, mean over samples: misses FX / UX / OX | fetch_wait ms FX / UX / OX")
    for row in curve:
        if all(a in row for a in ("FX", "UX", "OX")):
            L.append("  fwd %2d  %6.1f %6.1f %6.1f | %7.1f %7.1f %7.1f" % (
                row["fwd"], row["FX"]["misses"], row["UX"]["misses"], row["OX"]["misses"],
                1000 * row["FX"]["fetch_wait_s"], 1000 * row["UX"]["fetch_wait_s"], 1000 * row["OX"]["fetch_wait_s"]))
    for arm, v in out["arms"].items():
        for x in v:
            L.append("  %-3s s%d seq=%s ttft=%s chunks=%s tok/s=%s reserve=%s paths=%s first_fwd=%s %s" % (
                arm, x["k"], x["seq"], x["ttft_s"], x["chunks"], x["client_tok_s"], x["reserve"], x["prefill_paths"],
                x["first_forward"], ("PROBLEMS: " + "; ".join(x["problems"])[:400]) if x["problems"] else "ok"))
    return out, L


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in ("_sets",)}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


# ---------------------------------------------------------------- self test (synthetic samples)
def _fake_log(arm, extra_miss, fixed_books):
    """serve.log lines in the probe's exact formats (f42fix_probe_patch.py)"""
    probe = ARMS[arm][2]
    bank = ARMS[arm][3] == "bank"
    log = ["v4_hot_policy pin_slots_per_layer=16 repin_interval=6 mode=resident-ram rows16=hot-pins",
           "v4_gpu mirror-reserve=3100MiB (132-slot bank 1683MiB x1 + 600MiB headroom + 64 mirrors; DSV4_CUDA_VRAM_RESERVE_MB overrides)"]
    if bank:
        log += ["v4_gpu moe-batch=on bank=132-experts", "v4_gpu moe-batch refill layer=0 missing=85 lookup=0.02s upload=0.16s"]
    log += ["decprof position=%d layers=43 hc=5 attn=122 moe=%d ms" % (2005 + i, 250 + (i < 5) * 80) for i in range(24)]
    if probe:
        log.append("v4_dec_trace_start ord=1 pos=2005 wall=1792140000.0000 prefill_moe_chunk_layers bank=%d union=%d rows=0 "
                   "store_total req=9000 hits=5000 misses=4000 bytes=40000000000 prefetched=0 prefetch_hits=0 "
                   "resident_bytes=1 capacity_bytes=2 disk_s=10.000 wait_s=5.000 dec_moe_total calls=0 (process totals)"
                   % ((688, 0) if bank else (0, 688)))
        for layer in range(43):
            if bank and not fixed_books:
                res = ",".join("%d:1:%d%s" % (e, 900 - e, "p" if e < 8 else "") for e in range(59))
                pins = ",".join(str(e) for e in range(16))
                lreq = 104
            else:
                res = ",".join("%d:%d:%d%s" % (e, 9 - e % 9, 900 - e, "p" if e % 4 == 0 and e < 94 else "") for e in range(30, 89))
                pins = ",".join(str(e) for e in range(32, 96, 4))
                lreq = 1065
            log.append("v4_dec_res tag=decode-start layer=%d lreq=%d clock=99999 res=%s n=59 pinned_resident=16 pins=%s"
                       % (layer, lreq, res, pins))
        cum = dict.fromkeys(TRACE_KEYS, 0.0)
        for o in range(1, 26):
            if o > 1:
                f = o - 1
                miss = 60 + (extra_miss if f <= 3 else 0)
                cum["calls"] += 43; cum["total_s"] += 0.25 + miss * 0.001; cum["fetch_wait_s"] += miss * 0.001
                cum["attach_s"] += 0.08
                cum["req"] += 258; cum["hits"] += 258 - miss; cum["misses"] += miss; cum["bytes"] += miss * 13369344
                cum["disk_s"] += miss * 0.002; cum["wait_s"] += miss * 0.001
                for layer in range(43):
                    lm = miss // 43 + (1 if layer < miss % 43 else 0)
                    log.append("v4_dec_layer ord=%d layer=%d routed=%s req=6 hits=%d misses=%d bytes=%d "
                               "fetch_ms=%.3f attach_ms=1.000 wait_ms=%.3f total_ms=6.000"
                               % (f, layer, ",".join(str(e) for e in (31, 35, 40, 50, 70, 100)), 6 - lm, lm, lm * 13369344,
                                  lm * 1.0, lm * 1.0))
            log.append("v4_dec_trace ord=%d pos=%d tok=%d t=%.4f wall=%.4f calls=%d total_s=%.4f fetch_wait_s=%.4f "
                       "attach_s=%.4f req=%d hits=%d misses=%d bytes=%d prefetched=0 disk_s=%.4f wait_s=%.4f"
                       % (o, 2004 + o, 100 + o, cum["total_s"] + 0.12 * (o - 1), 1792140000.0 + o, cum["calls"], cum["total_s"],
                          cum["fetch_wait_s"], cum["attach_s"], cum["req"], cum["hits"], cum["misses"], cum["bytes"],
                          cum["disk_s"], cum["wait_s"]))
    return log


def selftest():
    tmp = tempfile.mkdtemp(prefix="f42fix-selftest-")
    t0 = time.time() - 3600
    seq = 0
    spt = {"W": 0.45, "FX": 0.44, "FXn": 0.44, "UX": 0.435, "OX": 0.47}
    for arm in ("W", "FX", "UX", "OX", "FXn"):
        for k in (1, 2):
            seq += 1
            r = ft._fake_req("long_prefill", t0, 82.0 if arm != "UX" else 219.0, 25, spt[arm] + 0.002 * k, "REF")
            r["finish_reason"] = "stop"
            r["text_sha256"] = REF_TEXT
            nv = [(t0 + i, 9000) for i in range(1, 80)] + [(t0 + 81 + i, 7400 + min(i * 100, 13900)) for i in range(30)]
            ft._fake_sample(tmp, arm, k, seq, dict(ARMS[arm][0]), [r],
                            _fake_log(arm, {"OX": 40, "FX": 10, "W": 10, "FXn": 10, "UX": 0}[arm], arm != "OX"), nv)
    out, L = summarize(tmp)
    probs = [(a, x["k"], x["problems"]) for a, v in out["arms"].items() for x in v if x["problems"]]
    assert not probs, probs
    t = out["table"]
    assert t["fwd 1-5 misses"]["FX"]["median"] - t["fwd 1-5 misses"]["UX"]["median"] == 30.0, t["fwd 1-5 misses"]
    assert t["fwd 1-5 misses"]["OX"]["median"] - t["fwd 1-5 misses"]["UX"]["median"] == 120.0, t["fwd 1-5 misses"]
    assert t["fwd 1-5 misses"]["recovered (means)"] == 0.75, t["fwd 1-5 misses"]
    assert t["decode-start layers pins=ids 0..15"]["OX"]["median"] == 43, t["decode-start layers pins=ids 0..15"]
    assert t["decode-start layers pins=ids 0..15"]["FX"]["median"] == 0
    assert t["decode-start layer requests mean"]["OX"]["median"] == 104
    assert t["decode-start layer requests mean"]["FX"]["median"] == 1065
    assert t["first fwd misses"]["OX"]["median"] - t["first fwd misses"]["FX"]["median"] == 30.0, t["first fwd misses"]
    assert out["first_forward_routing_distinct"] == 1
    assert out["text_identity"] == [REF_TEXT]
    assert t["client tok/s"]["FX/OX"] > 1 and t["client tok/s"]["OX/UX"] < 1
    assert t["client tok/s"]["FXn/FX"] is not None
    # a UX sample that used the bank, and an FX sample with a different reserve, must be flagged
    bad = os.path.join(tmp, "bad")
    ft._fake_sample(bad, "UX", 1, 1, dict(ARMS["UX"][0]), [ft._fake_req("long_prefill", t0, 82.0, 25, 0.47, "x")],
                    _fake_log("FX", 0, True), [(t0 + 1, 9000)])
    lg = [l.replace("mirror-reserve=3100MiB", "mirror-reserve=2800MiB") for l in _fake_log("FX", 0, True)]
    ft._fake_sample(bad, "FX", 1, 2, dict(ARMS["FX"][0]), [ft._fake_req("long_prefill", t0, 82.0, 25, 0.47, "x")], lg,
                    [(t0 + 1, 9000)])
    ft._fake_sample(bad, "FXn", 1, 3, dict(ARMS["FXn"][0]), [ft._fake_req("long_prefill", t0, 82.0, 25, 0.47, "x")],
                    _fake_log("FX", 0, True), [(t0 + 1, 9000)])
    o2, _ = summarize(bad)
    pu = " ".join(o2["arms"]["UX"][0]["problems"])
    pg = " ".join(o2["arms"]["FX"][0]["problems"])
    pn = " ".join(o2["arms"]["FXn"][0]["problems"])
    assert "GPU bank used in the CPU-union arm" in pu and "want bank=0" in pu and "not the reference" in pu, pu
    assert "mirror-reserve lines ['2800']" in pg, pg
    assert "probe lines in a non-probe sample" in pn, pn
    json.dumps(strip(out))
    print("selftest ok (%s)" % tmp)
    return 0


def main(argv):
    if argv[:1] == ["selftest"]:
        return selftest()
    if len(argv) == 2 and argv[0] == "summarize":
        out, L = summarize(argv[1])
        json.dump(strip(out), open(os.path.join(argv[1], "summary.json"), "w"), indent=1, default=str)
        print("\n".join(L))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
