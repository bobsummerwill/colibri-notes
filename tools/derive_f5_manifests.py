"""Derive manifests/final5/make_manifests_final5.py from final4's generator (colibri-claude, computer4).
The PR #1 part keeps every final4 check; only the identity, paths and wording change. A PR #2 decode part
(two manifests, one switch each) is appended."""
import os

R = os.path.expanduser("~/Projects/colibri-claude/manifests")
s = open(R + "/final4/make_manifests_final4.py").read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:90], n)
    s = s.replace(old, new)


rep('''"""Build the PR #1 experiment manifests (upstream docs/experiments/manifest.example.json format,
validated by c/experiment_manifest.py) from the final4 run on shadowfax (2026-10-17, run_final4.sh).
Adapted from ../final/pr1/make_manifests_pr1.py.''',
    '''"""Build the PR #1 and PR #2 experiment manifests (upstream docs/experiments/manifest.example.json format,
validated by c/experiment_manifest.py) from the final5 run on shadowfax (2026-10-20, run_final5.sh), the
stack on upstream dev 736a0682 after PR #0 (#2212-#2217) merged. Derived from ../final4/make_manifests_final4.py
by ../../tools/derive_f5_manifests.py: the PR #1 part keeps every final4 check; the PR #2 part is new
(D0/G1/R1/X3 now have 3 samples each).''')
rep('''PR #2 arms G1/X3 ran one sample each: no manifest (the validator needs >= 3 samples per arm); see pr2-NOTE.md.''',
    '''PR #2: pr2-vram-first.json (G1 -> R1, V4_GPU_RESIDENT_FIRST) and pr2-peer-device.json (R1 -> X3,
DSV4_CUDA_PEER_DEVICE); D0 (dev) vs G1 is recorded in stats.json as an observation (not one changed variable).''')
rep('''DEV = "62eaa9dc249b67567b9cbc83e43fbc875a8b94c1"
PR1 = "7e9d2d20816398c25dc979e952ce924cc1012fc3"
PR1_AMENDED = "4412cc1b60371424dbbdd479ca9fde054ec55186"  # rb/pr1/reap-prefill after the run (docs-only amend)
OLD_DEV = "7b55a9b3792527f52d1a2337fd9e312b9a7983e3"   # the 2026-10-15 manifests' base''',
    '''DEV = "736a0682d38f0a1c6592f34b1c6749fea221da42"
PR1 = "34fe3f808d348f28e0813b02f42eeea14c2414d3"
PR2 = "4558700fa51606cc4f4cadac0f9e2ca59a852eb9"
OLD_DEV = "62eaa9dc249b67567b9cbc83e43fbc875a8b94c1"   # the final4 manifests' base (2026-10-17)''')
rep('''assert TIPS["dev_sha"] == DEV and TIPS["pr1_sha"] == PR1 and TIPS["pr1_branch"] == "rb/pr1/reap-prefill", TIPS
assert TIPS["dev_is_ancestor_of_pr1"] == "yes", TIPS''',
    '''assert TIPS["dev_sha"] == DEV and TIPS["pr1_sha"] == PR1 and TIPS["pr1_branch"] == "refresh/pr1-reap-prefill", TIPS
assert TIPS["pr2_sha"] == PR2 and TIPS["pr2_branch"] == "refresh/pr2-two-gpu-decode", TIPS
assert TIPS["dev_is_ancestor_of_pr1"] == "yes" and TIPS["pr1_is_ancestor_of_pr2"] == "yes", TIPS''')
rep('''BIN = {DEV: TREES[DEV]["binary_sha256"], PR1: TREES[PR1]["binary_sha256"]}''',
    '''BIN = {DEV: TREES[DEV]["binary_sha256"], PR1: TREES[PR1]["binary_sha256"], PR2: TREES[PR2]["binary_sha256"]}''')
rep('''LLAMA_DOWN = hx(r"\\[(\\S+ \\S+)\\] stopping llama-server-active", RUNLOG).replace(" ", "T")''',
    '''LLAMA_DOWN = hx(r"\\[(\\S+ \\S+)\\] stopping llama-server-active", RUNLOG).replace(" ", "T")  # Hermes's model server (TabbyAPI)''')
rep('''    "New base: arm A is upstream dev %s ('%s'); the 2026-10-15 manifests (../final/pr1) used dev %s. The pr1 tip %s "
    "(rb/pr1/reap-prefill, tree %s) is %s commits on top of this base (meta/tips.txt: dev_is_ancestor_of_pr1=yes). "''',
    '''    "New base: arm A is upstream dev %s ('%s'), after PR #0 (#2212-#2217) merged; the final4 manifests (../final4) "
    "used dev %s. The pr1 tip %s (refresh/pr1-reap-prefill, tree %s) is %s commits on top of this base (meta/tips.txt: "
    "dev_is_ancestor_of_pr1=yes). "''')
rep('''    OLDM[f] = json.load(open(os.path.join(HERE, "..", "final", "pr1", f)))''',
    '''    OLDM[f] = json.load(open(os.path.join(HERE, "..", "final4", f)))''')
rep('''REPRO = {k: {"median_tok_s_2026_10_15": v, "median_tok_s_final4": statistics.median(vals(k[0], k[1], "prefill_tok_s")),
             "delta_pct": round(pct(statistics.median(vals(k[0], k[1], "prefill_tok_s")), v), 2),
             "commit_2026_10_15": OLD_DEV if k[0] == "A" else OLD_PR1, "commit_final4": ARM_TREE[k[0]]}''',
    '''REPRO = {k: {"median_tok_s_final4": v, "median_tok_s_final5": statistics.median(vals(k[0], k[1], "prefill_tok_s")),
             "delta_pct": round(pct(statistics.median(vals(k[0], k[1], "prefill_tok_s")), v), 2),
             "commit_final4": OLD_DEV if k[0] == "A" else OLD_PR1, "commit_final5": ARM_TREE[k[0]]}''')
rep('''            info = t.gettarinfo(f, arcname=os.path.join("final4", os.path.relpath(f, RAW)))''',
    '''            info = t.gettarinfo(f, arcname=os.path.join("final5", os.path.relpath(f, RAW)))''')
rep('''    path = os.path.join(HERE, "evidence", "final4-arm-%s.tgz" % arm)''',
    '''    path = os.path.join(HERE, "evidence", "final5-arm-%s.tgz" % arm)''')
rep('''            "contains": "final4/%s/s*/''', '''            "contains": "final5/%s/s*/''')
rep('''"full_run": "bench.tgz (sha256 %s, also in bench.tgz.sha256): all arms incl. W/G1/X3,''',
    '''"full_run": "bench.tgz (sha256 %s, also in bench.tgz.sha256): all arms incl. W/D0/G1/R1/X3,''')
rep('''        "earlier_run_2026_10_15": "; ".join("%s median %.4f tok/s on 2026-10-15 (%s) -> %.4f here (%s, %+.2f%%)" % (
            a, REPRO[(a, request)]["median_tok_s_2026_10_15"], REPRO[(a, request)]["commit_2026_10_15"][:8],
            REPRO[(a, request)]["median_tok_s_final4"], REPRO[(a, request)]["commit_final4"][:8], REPRO[(a, request)]["delta_pct"])
            for a in (base_arm, trial_arm)) + "; median ratio then %.4f (../final/pr1/%s), now %.4f. Both arms changed commit "
            "(new dev base; pr1 %s then), so this is a comparison across commits, not a repeat." % (''',
    '''        "earlier_run_final4": "; ".join("%s median %.4f tok/s in final4, 2026-10-17 (%s) -> %.4f here (%s, %+.2f%%)" % (
            a, REPRO[(a, request)]["median_tok_s_final4"], REPRO[(a, request)]["commit_final4"][:8],
            REPRO[(a, request)]["median_tok_s_final5"], REPRO[(a, request)]["commit_final5"][:8], REPRO[(a, request)]["delta_pct"])
            for a in (base_arm, trial_arm)) + "; median ratio then %.4f (../final4/%s), now %.4f. Both arms changed commit "
            "(new dev base; pr1 %s then, the same 8 patches rebased), so this is a comparison across commits, not a repeat." % (''')
rep('''    "llama.cpp (llama-server-active) was stopped %s and restored %s PDT (health ok, run.log), so the PR arms ran "
    "without it on the GPUs.''',
    '''    "Hermes's model server (llama-server-active, now TabbyAPI) was stopped %s and restored %s PDT (health ok, "
    "run.log), so the arms ran without it on the GPUs.''')
rep('''if ENGINE_REPO:''', '''if False and ENGINE_REPO:   # final5: no post-run amend to check (final4's amend logic kept for reference)''')
rep('''        "run": "final4, shadowfax, samples %s - %s PDT (llama.cpp down %s - %s); bench.tgz sha256 %s" % (''',
    '''        "run": "final5, shadowfax, samples %s - %s PDT (Hermes's model server down %s - %s); bench.tgz sha256 %s" % (''')
rep('''experts per layer, 6.23 GiB dense resident); MemAvailable %.1f GiB (%d MiB) at preflight with llama.cpp "
           "still loaded"''',
    '''experts per layer, 6.23 GiB dense resident); MemAvailable %.1f GiB (%d MiB) at preflight with Hermes's "
           "model server still loaded"''')
rep('''stats = {"bench_tgz_sha256": TGZ_SHA, "reference_A_sha256": REF, "reference_2026_10_15_sha256": REF_1015,''',
    '''stats = {"bench_tgz_sha256": TGZ_SHA, "reference_A_sha256": REF, "reference_2026_10_15_sha256": REF_1015,''')
rep('''                  "earlier_run_2026_10_15_table": {"%s %s" % k: v for k, v in REPRO.items()},
                  "earlier_run_2026_10_15_ratios": OLD_RATIO,''',
    '''                  "earlier_run_final4_table": {"%s %s" % k: v for k, v in REPRO.items()},
                  "earlier_run_final4_ratios": OLD_RATIO,''')
rep('''                  "pr2": "G1/X3 ran one sample each; no manifest (see pr2-NOTE.md)"}''',
    '''                  "pr2": "see stats['pr2'] and pr2-*.json"}''')
rep('''with open(os.path.join(HERE, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2); f.write("\\n")''',
    '''# (stats.json is written after the PR #2 part below)''')

s += open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "f5_pr2_part.py")).read()
open(R + "/final5/make_manifests_final5.py", "w").write(s)
print("written", R + "/final5/make_manifests_final5.py")
