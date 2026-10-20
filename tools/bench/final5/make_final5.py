"""Create ~/colibri-bench/final5 from final4 (shadowfax): the PR #1/#2 benchmark on upstream dev 736a0682
after PR #0 merged (2026-10-19/20). Copies final4's harness and changes only what this run needs:
paths/units, the base and tips, two new decode arms (D0 = dev, R1 = VRAM-first only), 3 samples for every
decode arm, a 5 h cap, a GPU-exclusivity check that accepts every process of llama-server-active's cgroup
(Tabby runs its model in a child process), and a final5 summary that compares every arm with this run's
own dev texts (the fixed references are from older bases)."""
import os, re, shutil

B = os.path.expanduser("~/colibri-bench")
F4, F5 = B + "/final4", B + "/final5"
os.makedirs(F5 + "/clients", exist_ok=True)
for f in ("bench_pr1.py", "bench_decode.py"):
    shutil.copy2(F4 + "/clients/" + f, F5 + "/clients/" + f)


def edit(s, old, new, count=1):
    n = s.count(old)
    assert n == count, (old[:80], n)
    return s.replace(old, new)


# ---------------- final_common.sh
s = open(F4 + "/final_common.sh").read()
s = edit(s, "# final_common.sh (final4: copy of final3 for the rebased stack, 2026-10-17)",
         "# final_common.sh (final5: copy of final4 for the stack on dev 736a0682, 2026-10-19)")
s = edit(s, "FINAL=/home/maplemercury/colibri-bench/final4\n", "FINAL=/home/maplemercury/colibri-bench/final5\n")
s = edit(s, "FINAL2=/home/maplemercury/colibri-bench/final3     # final4: the final3 lock is taken too",
         "FINAL2=/home/maplemercury/colibri-bench/final4     # final5: the final4 lock is taken too")
s = edit(s, "EXP_UNIT=colibri-final4 ", "EXP_UNIT=colibri-final5 ")
s = edit(s, "SERVE_UNIT=colibri-final4-serve", "SERVE_UNIT=colibri-final5-serve")
s = edit(s, "SUMMARY_KIND=final4", "SUMMARY_KIND=final5")
assert "final4" not in re.sub(r"#.*", "", s).replace("FINAL2=/home/maplemercury/colibri-bench/final4", ""), \
    [l for l in s.splitlines() if "final4" in re.sub(r"#.*", "", l)]
open(F5 + "/final_common.sh", "w").write(s)

# ---------------- final_tools.py
t = open(F4 + "/final_tools.py").read()
t = edit(t, 'DEV_SHA = "62eaa9dc249b67567b9cbc83e43fbc875a8b94c1"   # final4: the new base (the 10-15 references below are for 7b55a9b3)',
         'DEV_SHA = "736a0682d38f0a1c6592f34b1c6749fea221da42"   # final5: upstream dev after PR #0 merged (references below: older bases)\n'
         'DEV_SHA_FINAL4 = "62eaa9dc249b67567b9cbc83e43fbc875a8b94c1"')
t = edit(t, '    "X3": ("pr2", 6, {"V4_SERVE_STATS_LOG": "1", "V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1"},\n'
            '           ["COLI_CUDA_MOE_DOUBLE", "DSV4_CUDA_PEER_COMPUTE"], [r"v4_gpu peer-mirrors device=1 .*mode=compute", r"^v4_serve_turn "],\n'
            '           [r"mode=p2p-read", r"peer mirror pool off", r"peer device \\d+ failed"]),\n}\n',
         '    "X3": ("pr2", 6, {"V4_SERVE_STATS_LOG": "1", "V4_GPU_RESIDENT_FIRST": "1", "DSV4_CUDA_PEER_DEVICE": "1"},\n'
         '           ["COLI_CUDA_MOE_DOUBLE", "DSV4_CUDA_PEER_COMPUTE"], [r"v4_gpu peer-mirrors device=1 .*mode=compute", r"^v4_serve_turn "],\n'
         '           [r"mode=p2p-read", r"peer mirror pool off", r"peer device \\d+ failed"]),\n'
         '    # final5: R1 = G1 + VRAM-first only (one GPU); D0 = upstream dev on the decode client (no PR2 code at all)\n'
         '    "R1": ("pr2", 6, {"V4_SERVE_STATS_LOG": "1", "V4_GPU_RESIDENT_FIRST": "1"},\n'
         '           ["COLI_CUDA_MOE_DOUBLE", "DSV4_CUDA_PEER_DEVICE", "DSV4_CUDA_PEER_COMPUTE"], [r"^v4_serve_turn "], [r"peer-mirrors"]),\n'
         '    "D0": ("dev", 6, {}, ["COLI_CUDA_MOE_DOUBLE"] + PR2_SWITCHES, [], [r"peer-mirrors", r"^v4_serve_turn "]),\n}\n')
t = edit(t, '    if arm == "X3" and not vram_first(log):',
         '    if arm in ("X3", "R1") and not vram_first(log):')
t = edit(t, '    refs = REF_G1 if key == "pr2" else REF_PR1\n',
         '    refs = REF_G1 if arm in ("G1", "X3", "R1", "D0") else REF_PR1\n')

f5 = r'''

# ---------------------------------------------------------------- final5 summary
PLAN5 = {"W": 1, "A": 3, "B": 3, "C": 3, "D0": 3, "G1": 3, "R1": 3, "X3": 3}
PR1_ARMS5, PR2_ARMS5 = ("W", "A", "B", "C"), ("D0", "G1", "R1", "X3")
DEC_REQS = tuple(sorted(REF_G1))


def summarize_final5(base):
    """Like summarize_final4, plus D0/R1 and 3 samples per decode arm. Texts are compared with THIS run's
    dev arm (A s1 for the prefill client, D0 s1 for the decode client): a difference is a gate problem.
    A difference from the older fixed references is only a note (the base moved by 574 commits)."""
    tp = tips(base)
    out = {"kind": "final5", "tips": tp["raw"], "arms": {}, "comparisons": {}, "gates": {"problems": [], "notes": []}}
    P, N = out["gates"]["problems"], out["gates"]["notes"]
    ref_run = {}
    for dev_arm in ("A", "D0"):
        ss = samples(base, dev_arm)
        if ss:
            ref_run.update({r.get("name"): r["text_sha256"] for r in ss[0]["rows"]})
    out["run_reference_texts"] = ref_run
    for arm in PR1_ARMS5 + PR2_ARMS5:
        ss = samples(base, arm)
        if len(ss) != PLAN5[arm]:
            P.append("%s: %d of %d samples ran" % (arm, len(ss), PLAN5[arm]))
        if not ss:
            continue
        a = {"samples": []}
        for s in ss:
            bad, notes = premise(arm, s, tp)
            x = {"k": s["k"], "seq": s["meta"].get("seq"), "start": s["meta"].get("start"), "wall_s": s["meta"].get("wall_s"),
                 "ready_s": s["meta"].get("ready_s"), "tree_head": s["meta"].get("tree_head"),
                 "binary_sha256": s["meta"].get("binary_sha256"), "client_rc": s["meta"].get("client_rc"),
                 "premise_problems": bad, "notes_vs_old_references": notes,
                 "gpu0_max_mib": gpu_max_mem(s["dir"], 0), "gpu1_max_mib": gpu_max_mem(s["dir"], 1),
                 "mirror_reserve": re.findall(r"mirror-reserve=\d+MiB", " ".join(grep(s["log"], r"mirror-reserve="))),
                 "requests": {}}
            for r in s["rows"]:
                n = r.get("name")
                x["requests"][n] = {k: r.get(k) for k in ("ttft_s", "prefill_tok_s", "decode_tok_s", "decode_chunks", "total_s")}
                x["requests"][n]["text_sha256"] = r["text_sha256"]
                x["requests"][n]["identical_to_run_dev"] = (r["text_sha256"] == ref_run[n]) if n in ref_run else None
                if n in ref_run and r["text_sha256"] != ref_run[n]:
                    (P if arm != "W" else N).append("%s s%d: %s text %s differs from this run's dev text %s" % (
                        arm, s["k"], n, r["text_sha256"][:12], ref_run[n][:12]))
            if arm in PR2_ARMS5:
                toks = [r.get("decode_tok_s") for r in s["rows"] if r.get("decode_tok_s")]
                x["mean_tok_s"] = round(statistics.mean(toks), 4) if toks else None
                x["vram_first_hits_total"] = vram_first(s["log"])
                x["peer_lines"] = [l[:200] for l in grep(s["log"], r"peer-mirrors|peer mirror pool|peer device")[:3]]
            a["samples"].append(x)
            P.extend("%s s%d: %s" % (arm, s["k"], p) for p in bad)
            N.extend("%s s%d: %s" % (arm, s["k"], n) for n in notes)
        bins = {x["binary_sha256"] for x in a["samples"]}
        if len(bins) > 1:
            P.append("%s: samples ran different binaries %s" % (arm, sorted(b[:12] for b in bins if b)))
        if arm in PR2_ARMS5:
            a["mean_tok_s"] = stats([x["mean_tok_s"] for x in a["samples"]])
            a["per_prompt_tok_s"] = {n: stats([x["requests"].get(n, {}).get("decode_tok_s") for x in a["samples"]]) for n in DEC_REQS}
        else:
            a["requests"] = {n: {k: stats([x["requests"].get(n, {}).get(k) for x in a["samples"]])
                                 for k in ("ttft_s", "prefill_tok_s", "decode_tok_s")} for n in PR1_REQS}
        a["all_texts_identical_to_run_dev"] = all(v.get("identical_to_run_dev") is not False
                                                  for x in a["samples"] for v in x["requests"].values())
        out["arms"][arm] = a

    def st(arm, n, k, f):
        v = out["arms"].get(arm, {}).get("requests", {}).get(n, {}).get(k)
        return v[f] if v else None
    C = out["comparisons"]
    for n in PR1_REQS:
        c = {}
        for t2, b2 in (("B", "A"), ("C", "B"), ("C", "A")):
            for f in ("mean", "median"):
                c["ttft_%s_over_%s_%s" % (b2, t2, f)] = ratio(st(b2, n, "ttft_s", f), st(t2, n, "ttft_s", f))
                c["prefill_%s_vs_%s_%s" % (t2, b2, f)] = ratio(st(t2, n, "prefill_tok_s", f), st(b2, n, "prefill_tok_s", f))
            c["decode_%s_vs_%s_mean" % (t2, b2)] = ratio(st(t2, n, "decode_tok_s", "mean"), st(b2, n, "decode_tok_s", "mean"))
        C[n] = c

    def dm(arm, f="mean"):
        v = out["arms"].get(arm, {}).get("mean_tok_s")
        return v[f] if v else None
    C["decode"] = {arm + "_mean_tok_s": dm(arm) for arm in PR2_ARMS5}
    C["decode"].update({arm + "_median_tok_s": dm(arm, "median") for arm in PR2_ARMS5})
    for t2, b2 in (("G1", "D0"), ("R1", "G1"), ("X3", "R1"), ("X3", "G1"), ("X3", "D0")):
        C["decode"]["%s_vs_%s_mean" % (t2, b2)] = ratio(dm(t2), dm(b2))
        C["decode"]["%s_vs_%s_median" % (t2, b2)] = ratio(dm(t2, "median"), dm(b2, "median"))
    lb, la = kv(os.path.join(base, "meta", "llama_before.txt")), kv(os.path.join(base, "meta", "llama_after.txt"))
    out["llama"] = {"stop_at": lb.get("llama_stop_at"), "restored_at": la.get("restored_at"), "health_ok": la.get("ok"),
                    "down_s": la.get("gpu_window_s"), "script_status": la.get("script_status")}
    if la and la.get("ok") != "1":
        P.append("llama not healthy after the restore (%s)" % la.get("health"))
    out["usage_history_changed"] = usage_changed(base)
    out["gates"]["pass"] = not P

    L = ["final5 benchmark: %s" % base,
         "dev %s | pr1 %s | pr2 %s" % (tp["dev"], tp["pr1"], tp["pr2"]),
         "gates: %s (%d problems, %d notes)" % ("PASS" if out["gates"]["pass"] else "FAIL", len(P), len(N))]
    L += ["  problem: " + p for p in P[:40]]
    L.append("")
    L.append("Prefill client (bench_pr1.py): mean [median] over samples; texts vs this run's dev (A s1)")
    for arm in PR1_ARMS5:
        a = out["arms"].get(arm)
        if not a:
            continue
        for n in PR1_REQS:
            r = a["requests"][n]
            L.append("  %-2s %-15s ttft %8s s [%8s]  prefill %7s tok/s [%7s]  decode %6s tok/s  n=%s  texts_same=%s%s" % (
                arm, n, (r["ttft_s"] or {}).get("mean"), (r["ttft_s"] or {}).get("median"),
                (r["prefill_tok_s"] or {}).get("mean"), (r["prefill_tok_s"] or {}).get("median"),
                (r["decode_tok_s"] or {}).get("mean"), (r["ttft_s"] or {}).get("n"), a["all_texts_identical_to_run_dev"],
                "  (warm-up, discarded)" if arm == "W" else ""))
    for n in PR1_REQS:
        c = C[n]
        L.append("  %s: TTFT speedup B vs A %sx (median %sx), C vs A %sx (median %sx), C vs B %sx; decode B/A %s C/B %s" % (
            n, c["ttft_A_over_B_mean"], c["ttft_A_over_B_median"], c["ttft_A_over_C_mean"], c["ttft_A_over_C_median"],
            c["ttft_B_over_C_mean"], c["decode_B_vs_A_mean"], c["decode_C_vs_B_mean"]))
    L.append("")
    L.append("Decode client (bench_decode.py, 6 x 300 tokens): per-sample mean over prompts; texts vs this run's dev (D0 s1)")
    for arm in PR2_ARMS5:
        a = out["arms"].get(arm)
        if not a:
            continue
        L.append("  %-2s mean %s tok/s [median %s, cv %s%%, n=%s] samples %s  vram_first_hits %s  texts_same=%s" % (
            arm, (a["mean_tok_s"] or {}).get("mean"), (a["mean_tok_s"] or {}).get("median"), (a["mean_tok_s"] or {}).get("cv_pct"),
            (a["mean_tok_s"] or {}).get("n"), [x["mean_tok_s"] for x in a["samples"]],
            [x["vram_first_hits_total"] for x in a["samples"]], a["all_texts_identical_to_run_dev"]))
    d = C["decode"]
    L.append("  G1/D0 %s (PR2 switches off vs dev)  R1/G1 %s (VRAM-first)  X3/R1 %s (second GPU)  X3/G1 %s  X3/D0 %s" % (
        d["G1_vs_D0_mean"], d["R1_vs_G1_mean"], d["X3_vs_R1_mean"], d["X3_vs_G1_mean"], d["X3_vs_D0_mean"]))
    L.append("")
    L.append("llama: stopped %s, restored %s, healthy=%s, down %s s, script status %s" % (
        out["llama"]["stop_at"], out["llama"]["restored_at"], out["llama"]["health_ok"], out["llama"]["down_s"], out["llama"]["script_status"]))
    if N:
        L.append("notes (%d): %s" % (len(N), "; ".join(N[:8])))
    return out, L
'''
t = edit(t, "\n\ndef main(", f5 + "\n\ndef main(") if "\n\ndef main(" in t else t + f5
t = edit(t, '    if len(argv) >= 3 and argv[0] == "summarize" and argv[1] == "final4":\n',
         '    if len(argv) >= 3 and argv[0] == "summarize" and argv[1] == "final5":\n'
         '        out, L = summarize_final5(argv[2])\n'
         '        json.dump(out, open(os.path.join(argv[2], "summary.json"), "w"), indent=2)\n'
         '        print("\\n".join(L))\n'
         '        return 0 if out["gates"]["pass"] else 2\n'
         '    if len(argv) >= 3 and argv[0] == "summarize" and argv[1] == "final4":\n')
open(F5 + "/final_tools.py", "w").write(t)

# ---------------- run_final5.sh
r = open(F4 + "/run_final4.sh").read()
body = r[r.index("\nEXP=bench\n"):]
hdr = """#!/bin/bash
# run_final5.sh (2026-10-19) -- run_final4.sh adapted to the stack on upstream dev 736a0682 (PR #0 merged):
#   A/D0 = upstream dev 736a0682, B/C = refresh/pr1-reap-prefill, G1/R1/X3 = refresh/pr2-two-gpu-decode,
#   all from /tmp/colibri, built into /tmp/cw5/r15-*. STOP_AT is given at launch.
# Arms (every one 3 samples except the warm-up; order rotated per round):
#   W   warm-up, B's config, discarded                                           pr1  prefill client
#   A   dev (no GPU batch for 132 experts: CPU union)                            dev  prefill client
#   B   pr1 tip (COLI_CUDA_MOE_BATCH=1 takes the 132-expert checkpoint)           pr1  prefill client
#   C   B + COLI_CUDA_MOE_DOUBLE=1                                               pr1  prefill client
#   D0  dev on the decode client (no PR2 code)                                   dev  decode client
#   G1  pr2 tip, V4_SERVE_STATS_LOG=1 only (every PR2 switch off)                 pr2  decode client
#   R1  G1 + V4_GPU_RESIDENT_FIRST=1 (one GPU)                                   pr2  decode client
#   X3  R1 + DSV4_CUDA_PEER_DEVICE=1 (peer compute on GPU 1)                     pr2  decode client
# One changed variable per comparison: A->B (the PR1 engine), B->C (double bank), G1->R1 (VRAM-first),
# R1->X3 (second GPU); D0->G1 checks that PR2 with its switches off behaves like dev.
# Everything else (locks, builds, preflight, the llama stop/EXIT-trap restore, watchdog, evidence,
# CUDA tests at the end) is final4's; see final4/run_final4.sh for the full description.
# final5: the GPU-exclusivity check accepts every process in llama-server-active's cgroup (Tabby).
"""
body = edit(body, "PR1=/tmp/cw-rb/r14-pr1\nPR2=/tmp/cw-rb/r14-pr2\n", "PR1=/tmp/cw5/r15-pr1\nPR2=/tmp/cw5/r15-pr2\n")
body = edit(body, "BUILDS=/home/maplemercury/colibri-bench/final4/builds-dry", "BUILDS=/home/maplemercury/colibri-bench/final5/builds-dry")
body = edit(body, "    PR1=/tmp/cw-rb/f4dry-pr1\n    PR2=/tmp/cw-rb/f4dry-pr2\n    DEV_DRY=/tmp/cw-rb/f4dry-dev\n",
            "    PR1=/tmp/cw5/f5dry-pr1\n    PR2=/tmp/cw5/f5dry-pr2\n    DEV_DRY=/tmp/cw5/f5dry-dev\n")
body = edit(body, "MAX_GPU_S=$((110 * 60))     # final4: hard cap on the llama-down window (samples + CUDA tests)",
            "MAX_GPU_S=$((300 * 60))     # final5: hard cap on the llama-down window (samples + CUDA tests)")
body = edit(body, "source /home/maplemercury/colibri-bench/final4/final_common.sh", "source /home/maplemercury/colibri-bench/final5/final_common.sh")
body = edit(body, "REPO=/tmp/cw-rb/base\nPR1_BRANCH=rb/pr1/reap-prefill\nPR2_BRANCH=rb/pr2/two-gpu-decode\n",
            "REPO=/tmp/colibri\nPR1_BRANCH=refresh/pr1-reap-prefill\nPR2_BRANCH=refresh/pr2-two-gpu-decode\n")
body = edit(body, "DEV_SHA=62eaa9dc249b67567b9cbc83e43fbc875a8b94c1   # final4: upstream dev, the rebased base\nDEV=/tmp/cw-rb/r14-dev\n",
            "DEV_SHA=736a0682d38f0a1c6592f34b1c6749fea221da42   # final5: upstream dev after PR #0 merged\nDEV=/tmp/cw5/r15-dev\n")
body = edit(body, 'TREE[X3]=$PR2; CLIENT[X3]=decode;   TMO[X3]=1500; XENV[X3]="V4_SERVE_STATS_LOG=1 V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1"; EST[X3]=520\n'
                  'ORDER="W:1 A:1 B:1 C:1 B:2 C:2 A:2 C:3 A:3 B:3 G1:1 X3:1"\n',
            'TREE[X3]=$PR2; CLIENT[X3]=decode;   TMO[X3]=1500; XENV[X3]="V4_SERVE_STATS_LOG=1 V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1"; EST[X3]=520\n'
            'TREE[R1]=$PR2; CLIENT[R1]=decode;   TMO[R1]=1500; XENV[R1]="V4_SERVE_STATS_LOG=1 V4_GPU_RESIDENT_FIRST=1";                         EST[R1]=650\n'
            'TREE[D0]=$DEV; CLIENT[D0]=decode;   TMO[D0]=1500; XENV[D0]="";                                                        EST[D0]=720\n'
            'ORDER="W:1 A:1 B:1 C:1 B:2 C:2 A:2 C:3 A:3 B:3 D0:1 G1:1 R1:1 X3:1 G1:2 R1:2 X3:2 D0:2 R1:3 X3:3 D0:3 G1:3"\n')
body = edit(body, """foreign_gpu_pids() {
    local main
    main=$(systemctl --user show -p MainPID --value llama-server-active 2>/dev/null)
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u \\
        | grep -vx "${main:-0}" | grep -v '^$'
}""", """foreign_gpu_pids() {   # final5: every process of llama-server-active's cgroup is llama's (Tabby's model runs in a child)
    local cg allowed
    cg=$(systemctl --user show -p ControlGroup --value llama-server-active 2>/dev/null)
    allowed=$(cat "/sys/fs/cgroup$cg/cgroup.procs" 2>/dev/null | tr '\\n' ' ')
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u | grep -v '^$' \\
        | while read -r p; do case " $allowed " in *" $p "*) ;; *) echo "$p" ;; esac; done
}""")
assert "cw-rb" not in body, [l for l in body.splitlines() if "cw-rb" in l]
open(F5 + "/run_final5.sh", "w").write(hdr + body)
os.makedirs("/tmp/cw5", exist_ok=True)
print("final5 written")
