#!/usr/bin/env python3
# usage: analyze_f42.py RAW_DIR (the extracted f42.tgz f42/ directory); prints the per-sample / per-arm / token-window tables
"""Deeper F42 analysis on top of final_tools.py summarize f42: per-sample tables, mean/stdev,
per-token-window profile (front-loading), attach cost per upload, VRAM plateau, text identity."""
import json, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(sys.argv[1], "meta"))
import final_tools as ft  # the copy inside f42.tgz

base = sys.argv[1]
ARMS = ["GB", "CU", "GR"]
WIN = [(1, 10), (11, 24), (25, 50), (51, 100), (101, 199)]   # decode-token index windows (interval i = chunk i-1 -> i)

def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return "n/a"
    m = st.mean(xs); s = st.stdev(xs) if len(xs) > 1 else 0.0
    return "%.4f +- %.4f" % (m, s)

def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None

data = {}
for arm in ARMS:
    for s in ft.samples(base, arm):
        x = ft.f42_sample(s, arm)
        turns = ft.f42_turns(s["log"])
        prompts = [int(t.get("prompt", 0)) for t in turns]
        prof = ft.decprof(s["log"], prompts)
        x["_prof"] = prof
        x["_rows"] = {r["name"]: r for r in s["rows"]}
        x["_turns"] = turns
        tr = [(t, m) for t, i, m in ft.nvsmi(s["dir"]) if i == 0]
        x["_vram"] = tr
        data.setdefault(arm, []).append(x)

out = {}
P = print
P("== text identity")
for req in ("long_200", "short_200"):
    shas = {arm: [x["requests"][req]["text_sha256"][:12] for x in data.get(arm, [])] for arm in ARMS}
    P("  %s: %s" % (req, shas))
ref = data["GB"][0]["_rows"]["long_200"]["chunk_text"]
for arm in ARMS:
    for x in data.get(arm, []):
        c = x["_rows"]["long_200"]["chunk_text"]
        P("  long_200 %s s%d first chunk differing from GB s1: %s" % (arm, x["k"], ft.first_diff(ref, c)))

P("\n== per-sample decode (client chunk rates; v4_serve_turn tok_s) and dec_moe deltas per request")
hdr = "%-4s %-3s %-9s %8s %8s %8s %8s | %6s %7s %7s %7s %6s %6s | %7s %6s %7s"
P(hdr % ("arm", "s", "req", "tok/s", "first24", "after24", "turn_tps", "calls", "total", "fetchw", "attach", "upl", "gpu", "ms/upl", "hit%", "mirrors"))
for arm in ARMS:
    for x in data.get(arm, []):
        for req in ("long_200", "short_200"):
            q = x["requests"][req]; c = q["client"]; d = q["dec_moe"]; t = q["turn"]
            P("%-4s s%-2d %-9s %8.4f %8s %8s %8.3f | %6d %7.2f %7.2f %7.2f %6d %6.2f | %7.3f %6.1f %7.1f" % (
                arm, x["k"], req, c["rate_all"], c.get("rate_first24"), c.get("rate_after24"), t["tok_s"],
                d["calls"], d["total_s"], d["fetch_wait_s"], d["attach_s"], d["uploads"], d["gpu_s"],
                1000 * d["attach_s"] / d["uploads"], t["hit"], q["mirrors_est"]))

P("\n== per-arm mean +- stdev (3 samples)")
metrics = [
    ("long tok/s (all 199 intervals)", "long_200", lambda q: q["client"]["rate_all"]),
    ("long tok/s first 24", "long_200", lambda q: q["client"]["rate_first24"]),
    ("long tok/s after 24", "long_200", lambda q: q["client"]["rate_after24"]),
    ("long ttft s", "long_200", lambda q: q["client"]["ttft_s"]),
    ("long s/tok", "long_200", lambda q: q["client"]["sec_per_tok"]),
    ("long dec_moe total s", "long_200", lambda q: q["dec_moe"]["total_s"]),
    ("long fetch_wait s", "long_200", lambda q: q["dec_moe"]["fetch_wait_s"]),
    ("long attach s", "long_200", lambda q: q["dec_moe"]["attach_s"]),
    ("long uploads", "long_200", lambda q: q["dec_moe"]["uploads"]),
    ("long gpu s", "long_200", lambda q: q["dec_moe"]["gpu_s"]),
    ("long dec_moe other s (total-fw-att-gpu)", "long_200", lambda q: q["dec_moe"]["total_s"] - q["dec_moe"]["fetch_wait_s"] - q["dec_moe"]["attach_s"] - q["dec_moe"]["gpu_s"]),
    ("long attach ms/upload", "long_200", lambda q: 1000 * q["dec_moe"]["attach_s"] / q["dec_moe"]["uploads"]),
    ("long decprof attn s", "long_200", lambda q: q["decprof"]["attn_s"]),
    ("long decprof moe s", "long_200", lambda q: q["decprof"]["moe_s"]),
    ("long decprof hc s", "long_200", lambda q: q["decprof"]["hc_s"]),
    ("long client decode s", "long_200", lambda q: q["client"]["sec_per_tok"] * (q["client"]["n"] - 1)),
    ("long mirrors (VRAM est)", "long_200", lambda q: q["mirrors_est"]),
    ("long VRAM0 decode max MiB", "long_200", lambda q: q["vram0"]["decode_max"]),
    ("short tok/s", "short_200", lambda q: q["client"]["rate_all"]),
    ("short s/tok", "short_200", lambda q: q["client"]["sec_per_tok"]),
    ("short dec_moe total s", "short_200", lambda q: q["dec_moe"]["total_s"]),
    ("short fetch_wait s", "short_200", lambda q: q["dec_moe"]["fetch_wait_s"]),
    ("short attach s", "short_200", lambda q: q["dec_moe"]["attach_s"]),
    ("short uploads", "short_200", lambda q: q["dec_moe"]["uploads"]),
    ("short gpu s", "short_200", lambda q: q["dec_moe"]["gpu_s"]),
    ("short attach ms/upload", "short_200", lambda q: 1000 * q["dec_moe"]["attach_s"] / q["dec_moe"]["uploads"]),
    ("short decprof attn s", "short_200", lambda q: q["decprof"]["attn_s"]),
    ("short decprof moe s", "short_200", lambda q: q["decprof"]["moe_s"]),
]
tab = {}
P("%-42s %22s %22s %22s %9s %9s %9s" % ("metric", "GB", "CU", "GR", "GB-GR", "GR-CU", "GB-CU"))
for label, req, f in metrics:
    vals = {arm: [f(x["requests"][req]) for x in data.get(arm, [])] for arm in ARMS}
    m = {arm: mean(vals[arm]) for arm in ARMS}
    tab[label] = {"samples": vals, "mean": m}
    dd = lambda a, b: "%9.4f" % (m[a] - m[b]) if m[a] is not None and m[b] is not None else "      n/a"
    P("%-42s %22s %22s %22s %s %s %s" % (label, ms(vals["GB"]), ms(vals["CU"]), ms(vals["GR"]), dd("GB", "GR"), dd("GR", "CU"), dd("GB", "CU")))
out["table"] = tab

P("\n== per-token-window profile of the long answer (mean over samples): client interval ms / decprof moe ms / decprof attn ms")
P("   window = decode token index (interval k = chunk k-1 -> chunk k; decprof row k = the forward producing chunk k)")
prof_tab = {}
for arm in ARMS:
    for lo, hi in WIN:
        iv, mo, at = [], [], []
        for x in data.get(arm, []):
            ivs = x["requests"]["long_200"]["interval_ms"]
            pr = x["_prof"][0]
            iv.append(sum(ivs[lo - 1:hi]) / (hi - lo + 1))
            mo.append(sum(r["moe"] for r in pr[lo - 1:hi]) / len(pr[lo - 1:hi]))
            at.append(sum(r["attn"] for r in pr[lo - 1:hi]) / len(pr[lo - 1:hi]))
        prof_tab[(arm, lo)] = (mean(iv), mean(mo), mean(at))
for lo, hi in WIN:
    row = "  tok %3d-%3d " % (lo, hi)
    for arm in ARMS:
        iv, mo, at = prof_tab[(arm, lo)]
        row += " %s %6.1f/%6.1f/%6.1f |" % (arm, iv, mo, at)
    g1 = prof_tab[("GB", lo)][0] - prof_tab[("GR", lo)][0]
    g2 = prof_tab[("GR", lo)][0] - prof_tab[("CU", lo)][0]
    row += " GB-GR %+6.1f  GR-CU %+6.1f ms/tok (interval)" % (g1, g2)
    P(row)

P("\n== cumulative gap of the long answer (client time, mean over samples), s")
def cum(arm, n):
    return mean([sum(x["requests"]["long_200"]["interval_ms"][:n]) / 1000 for x in data.get(arm, [])])
for n in (10, 24, 50, 100, 150, 199):
    a, b, c = cum("GB", n), cum("GR", n), cum("CU", n)
    P("  first %3d intervals: GB %7.2f GR %7.2f CU %7.2f | GB-GR %+6.2f  GR-CU %+6.2f  GB-CU %+6.2f" % (n, a, b, c, a - b, b - c, a - c))
P("  same for decprof moe (s):")
def cumm(arm, n, key="moe"):
    return mean([sum(r[key] for r in x["_prof"][0][:n]) / 1000 for x in data.get(arm, [])])
for n in (10, 24, 50, 100, 150, 199):
    a, b, c = cumm("GB", n), cumm("GR", n), cumm("CU", n)
    P("  first %3d tokens: GB %7.2f GR %7.2f CU %7.2f | GB-GR %+6.2f  GR-CU %+6.2f  GB-CU %+6.2f" % (n, a, b, c, a - b, b - c, a - c))

P("\n== short answer profile (identical text in every arm)")
for lo, hi in WIN:
    row = "  tok %3d-%3d " % (lo, hi)
    v = {}
    for arm in ARMS:
        iv = mean([sum(x["requests"]["short_200"]["interval_ms"][lo - 1:hi]) / (hi - lo + 1) for x in data.get(arm, [])])
        mo = mean([sum(r["moe"] for r in x["_prof"][1][lo - 1:hi]) / len(x["_prof"][1][lo - 1:hi]) for x in data.get(arm, [])])
        v[arm] = iv
        row += " %s %6.1f/%6.1f |" % (arm, iv, mo)
    row += " GB-GR %+6.1f  GR-CU %+6.1f" % (v["GB"] - v["GR"], v["GR"] - v["CU"])
    P(row)

P("\n== VRAM GPU0: ready / prefill median / decode plateau / seconds from first token to 98% of plateau")
for arm in ARMS:
    for x in data.get(arm, []):
        r = x["_rows"]["long_200"]; tr = x["_vram"]
        first = r["t0_epoch"] + r["ttft_s"]
        dec = [(t, m) for t, m in tr if t >= first]
        plateau = max(m for t, m in dec)
        base0 = [m for t, m in tr if t < first][-1]
        t98 = next(t for t, m in dec if m >= base0 + 0.98 * (plateau - base0)) - first
        v = x["requests"]["long_200"]["vram0"]
        P("  %s s%d ready=%s prefill_med=%s prefill_max=%s last_before_decode=%d plateau=%d t98=%.1fs mirrors_est=%s modes=%s" % (
            arm, x["k"], v["ready"], v["prefill_median"], v["prefill_max"], base0, plateau, t98, x["requests"]["long_200"]["mirrors_est"],
            [l[:90] for l in x["mode_lines"]]))
P("\n== problems")
for arm in ARMS:
    for x in data.get(arm, []):
        P("  %s s%d seq=%s: %s" % (arm, x["k"], x["seq"], x["problems"] or "ok"))
