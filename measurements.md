# Measurements

> **Naming key.** These notes use internal names for the upstream pull requests. **PR #0** is the six
> fixes opened as JustVugg/colibri [#2212](https://github.com/JustVugg/colibri/pull/2212) (0a), [#2213](https://github.com/JustVugg/colibri/pull/2213) (0b), [#2214](https://github.com/JustVugg/colibri/pull/2214) (0c), [#2215](https://github.com/JustVugg/colibri/pull/2215) (0d),
> [#2216](https://github.com/JustVugg/colibri/pull/2216) (0e) and [#2217](https://github.com/JustVugg/colibri/pull/2217) (0f). **PR #1** (batched GPU prefill for expert-pruned checkpoints) is
> [#2264](https://github.com/JustVugg/colibri/pull/2264), opened 2026-10-20; **PR #2** (two-GPU decode) is not opened yet. Documents dated before 2026-10-17 describe earlier
> versions of these changes, on the old base 7b55a9b3.

Up to final5: all on shadowfax (see environment.md), DeepSeek V4 Flash REAP 150B, `--ram 40 --ctx 8192`, llama.cpp stopped for the window, one fresh server per sample. The engine is greedy, so "identical" means byte-identical output text (sha256 compared). final6 adds the full 256-expert V4 Flash and computer4 (DeepGEMM).

## FINAL6 (2026-10-22 17:46-23:56 PDT, shadowfax, `manifests/final6/`) — full DeepSeek V4 Flash (256 experts) + REAP-150B; dev 7df8c27c, pr1 b286167e, pr2 0523e8e0
Harness `tools/bench/final6/` ([TEST_PLAN_20261022.md](TEST_PLAN_20261022.md)); base environment = the canonical serve
line **without** `COLI_CUDA_MOE_BATCH`, `--ram 40`, fresh server per sample, W discarded. 3 samples for A, B, BK, BD and
the PR #2 decode arms; 1 for the rest. **"QUESTION (A vs B, V) ANSWERED CLEANLY: True"; GATES: FAIL** (side arms, below).
Evidence: `manifests/final6/bench.tgz` (sha256 6b9b1d36…), `pilot.tgz`, `summary.txt`.

**JustVugg's question (default configuration, full model, interleaved W A B B A A B):**

| arm | 1,992-tok TTFT | 3,005-tok TTFT | decode (2k / 3k) | text |
|---|---|---|---|---|
| A dev 7df8c27c | 160.9 s | 264.6 s | 2.009 / 1.897 | reference |
| B PR #1 | 158.2 s (0.983x) | 261.5 s (0.989x) | 2.010 / 1.900 (1.002x mean) | identical, all samples |
| V B + `DSV4_MM_WARP_VERIFY=1` | 164.3 s | 276.0 s | | identical; **1,500 calls, 0 mismatches** |

=> `DSV4_MM_WARP` stays on by default (the plan's rule: bit-identical and not slower).

**The rest (PR #1 vs dev unless noted; texts identical within each group unless noted):**

| comparison | TTFT ratio | decode ratio | note |
|---|---|---|---|
| BK/AK 256-expert bank | 0.665 (85.9 vs 129.1 s) / 0.481 | 0.973 | AK: 1 sample; dev's bank allocation failed on prompt 2 (`moe-batch=off`: dev's 2800 MiB default reserve is below the bank once decode mirrors grow, as dev's comment says), so **only the 2k ratio is like-for-like**; AS, KA, MA send one request and kept it |
| BD/BK double bank | 0.775 / 0.822 | 0.974 | 66.6 s / 101.4 s |
| LB/LA ~8k prompt, default | 0.982 | 1.000 | 832 vs 847 s |
| KB/KA ~16k prompt, bank, `--ctx 32768` | **0.690** (795 vs 1151 s) | 0.997 | like-for-like (dev's bank on) |
| MB/MA MTP drafting, bank | 0.665 | 0.997 | |
| AS/BS speculative, bank | 129.0 / 85.7 s | | texts = A's; pass |
| CC two requests at once | | | each text = its sequential one; pass |
| VB/VA Vulkan+CUDA build, bank | 0.587 / 0.535 | 1.008 | **texts differ from RB (CUDA only)**; VA = VB on prompt 1, not prompt 2 |
| QB/QA pure Vulkan | 0.979 | | **QB text differs; QB killed by the host OOM killer** after prompt 1 (see investigations §6) |
| G1/D0 PR #2 switches off vs dev (REAP) | ~1.00 | 0.999 | 3.011 vs 3.014 tok/s |
| R1/G1 VRAM-first | | 1.021-1.030 | 3.096 tok/s |
| X3/R1 second GPU | | **1.36-1.48** | 4.343 tok/s (+44% vs dev) |
| P0 (`DSV4_CUDA_PEER_COMPUTE=0`), NB (`DSV4_CUDA_BATCHED=0`) | | | 3.66 / 3.12 tok/s; texts = D0's; pass |
| HY (`DSV4_HYBRID=1`, PR #2 all on) | | | 2.91 tok/s; **decode_0 text differs from D0** (also differed in final5) |
| SK soak, 6-turn chat x3, PR #2 all on | | | every repeat = the first; pass |

- **Handled allocation failures** (`[DSV4 CUDA] kv comp allocation: out of memory`, the engine falls back and goes on)
  in **both** arms of the default configuration: 252 per sample in W, A, B and CC, 420 in V, 316 (`buffer`) in LA/LB.
  Pre-existing on dev at 24 GB with the full model, not PR #1's.
- **F18 probe** (PR #2, `DSV4_F18_TRAP`, a real `__trap()` on the peer GPU): both cases CAUGHT-REFUSED (no hang, no silent
  wrong output), **but the next all-on-primary call on device 0 also fails** ("MoE activation upload: unspecified launch
  failure"): the sticky fault takes the whole process's CUDA, so PR #2's log line "decode continues on device 0" is wrong
  (the engine falls back to host copies). The probe's verdict ignored that line. See investigations §6.
- GPU tests in the pilot: `dsv4-cuda-test` rc=0 at pr1 and pr2 (0 warning lines), `cuda-test` rc=0 at pr2.

## FINAL6 DeepGEMM (2026-10-22 17:43-19:01 PDT, computer4, `manifests/final6-c4ab/`) — RTX 5070 12 GB (sm_120a), CUDA 12.8
First DeepGEMM build in this project (`DSV4_MM_WARP` and `DSV4_MOE_EXPERT_MAJOR` are compiled out there). dev, pr1, pr2
build and pass their GPU tests (two-GPU and >9 GB tests skip). Full V4 Flash, default configuration, W A B B A A B,
`run_c4ab.sh`. **GATES: PASS.** The ~10 GB bank row does not fit beside Voxtype's 1.5 GB and was not run.

| arm | 1,992-tok TTFT | 3,005-tok TTFT | decode (2k / 3k) | text |
|---|---|---|---|---|
| A dev 7df8c27c | 242.6 s | 373.1 s | 1.372 / 1.305 | reference |
| B PR #1 b286167e | 242.3 s (0.999x) | 374.4 s (1.004x) | 1.375 / 1.294 (1.002x mean) | identical, all samples |

- B s2 alone: 421 handled allocation failures (1 weight, 420 kv comp), GPU0 peak 11,747 vs ~11,255 MiB, its 3k TTFT
  409 s (vs ~373). Same baseline VRAM before the sample; cause not found. Medians unaffected.

## FINAL5 (2026-10-20 00:15-03:00 PDT, `manifests/final5/`) — on upstream dev 736a0682 (PR #0 merged); pr1 34fe3f80, pr2 4558700f (= refresh/* tips; the docs5/* branches to submit differ only in docs/)
Harness `run_final5.sh` = final4's plus decode arms D0 (dev) and R1 (VRAM-first only) and 3 samples for every decode arm; Hermes's model server is now TabbyAPI (GPU check covers its whole cgroup, health accepts "healthy"). 3 cold samples per arm (+1 discarded warm-up), rotated order, medians; `.coli_usage` empty. GATES: PASS; all 92 outputs identical to this run's dev text (and to the older fixed references). Six manifests, all valid.

| arm | 1,992-tok TTFT | 3,005-tok TTFT | decode after (2k / 3k, medians) | text |
|---|---|---|---|---|
| A upstream dev 736a0682 (CPU union) | 130.2 s (15.30 tok/s) | 205.9 s (14.59) | 2.278 / 2.940 | reference |
| B PR #1 | 76.8 s (25.94), 1.70x | 111.0 s (27.08), 1.86x | 2.205 / 2.893 | identical |
| C B + `COLI_CUDA_MOE_DOUBLE=1` | 64.1 s (31.09), 2.03x | 97.8 s (30.71), 2.10x | 2.087 / 2.835 | identical |

| decode arm (6 x 300-token answers; per-sample mean, median of 3) | tok/s | vs row above | VRAM-first hits |
|---|---|---|---|
| D0 upstream dev | 3.011 | | n/a |
| G1 PR #2, every switch off (`V4_SERVE_STATS_LOG=1`) | 2.983 | -0.9% | 0 |
| R1 G1 + `V4_GPU_RESIDENT_FIRST=1` | 3.072 | +3.0% | 310,011 |
| X3 R1 + `DSV4_CUDA_PEER_DEVICE=1` (peer compute) | 4.287 | +39.5% | 418,453 |

- CVs: decode arms <= 0.1%; TTFT <= 1%. X3 vs G1 +43.7%, vs dev +42.4%.
- G1 < D0 and B/C decode-after < A: with `COLI_CUDA_MOE_BATCH=1` the pr tips reserve 3100 MiB for the 132-slot bank (`v4_gpu mirror-reserve=3100MiB`), dev reserves none for this model; GPU0 peak 21,192 vs 22,008 MiB, i.e. fewer decode mirrors. Not PR #2 code.
- Speedups vs final4 (62eaa9dc): PR #1 1.69/1.86 -> 1.70/1.86 (double 2.01/2.09 -> 2.03/2.10); PR #2 one-sample +44% -> 3-sample +43.7%.
- CUDA tests in the window: `dsv4-cuda-test` rc=0 at both tips (0 warnings), `cuda-test` rc=0 at pr2. `make check` (computer4): pr1 2612, pr2 2619 OK (284 skipped); docs5/pr2 2619 OK. Per-commit: 16/16 commits 0 diagnostics (`percommit.txt`).

## REBASED (2026-10-17 22:34-23:38 PDT, `manifests/final4/`) — on upstream dev 62eaa9dc; pr1 7e9d2d20, pr2 bffe2aed (code trees identical to the final rebased/* tips)
Same harness as final3 (`run_final4.sh`), 3 cold samples per arm (+1 discarded warm-up), medians; `.coli_usage` empty, so upstream's serve prewarm (#2109) reads nothing at `--ram 40`. GATES: PASS. GPUs exclusive (no llama start attempts in the window; an earlier 21:34 attempt was contended by repeated llama-server-active starts and discarded: `bench.contended-2134`).

| arm | 1,992-tok TTFT | 3,005-tok TTFT | decode after (2k / 3k) | text |
|---|---|---|---|---|
| A upstream dev 62eaa9dc (CPU union) | 128.9 s (15.45 tok/s) | 204.4 s (14.70) | 2.290 / 2.938 | reference |
| B PR #1 | 76.5 s (26.03), 1.69x | 110.1 s (27.29), 1.86x | 2.221 / 2.917 | identical |
| C B + `COLI_CUDA_MOE_DOUBLE=1` | 64.1 s (31.09), 2.01x | 98.0 s (30.65), 2.09x | 2.094 / 2.840 | identical |

- Upstream dev itself prefills ~1.7x faster than 7b55a9b3 (219 s -> 129 s on 2k), most likely #2069 (faster CPU decode kernels); PR #1's relative gain shrinks accordingly.
- PR #2 (one sample each, six 300-token answers, DOUBLE unset): G1 one GPU 3.011 tok/s, X3 VRAM-first + peer compute 4.341 tok/s (+44%); all six texts identical to the 10-15 reference.
- CUDA tests in the same window: `make dsv4-cuda-test` rc=0 at the pr1 and pr2 tips, `make -C c cuda-test` rc=0 at pr2 (0 warnings).
- `make check` (computer4): upstream dev 2560 tests OK; pr0 2572; pr1 2579; pr2 2586 (277 skipped each).

## FINAL CODE (2026-10-17 02:00-03:16 PDT, `manifests/final3/`) — pr1/v6 9222a65a (with the F42 fix), pr2/v5 ef3ade98
3 cold samples per PR #1 arm after a discarded warm-up; all gates passed; every output byte-identical to the dev reference.
| Arm | 1992-tok TTFT | tok/s | answer decode | 3005-tok TTFT | tok/s | answer decode |
|---|---|---|---|---|---|---|
| A dev | 219.4 s | 9.08 | 2.207 | 335.2 s | 8.96 | 2.820 |
| B pr1 | 82.2 s | 24.25 (**2.67x**) | 2.170 (**0.983x A**, was 0.914x before the fix) | 119.1 s | 25.24 (**2.82x**) | 2.794 (0.991x) |
| C B + DOUBLE | 69.8 s | 28.52 (**3.14x**) | 2.049 | 106.4 s | 28.24 (3.15x) | 2.732 |
- F42 essentially closed against dev: decode right after the long GPU prefill is now 1.7% (2k) / 1.0% (3k) behind dev, vs 8.6% / 2.4% before the fix.
- PR #2 sanity on the final tip (1 sample each, 6x300): G1 2.896, X3 4.084 tok/s (+41%), all 6 texts identical to the 10-15 reference; both within ~1% of the 10-15 medians.
- Prefill numbers match 10-15 within 0.5%. Hermes: llama.cpp down 02:00:02-03:15:39 PDT.

## FINAL (2026-10-15, on the final stack tips) — evidence and validated manifests in `manifests/final/`
Builds: dev 7b55a9b3, pr0/dsv4-fixes a2c8ba7a, pr1/reap-prefill 93b74183, pr2/two-gpu-decode d0e19de2 (CUDA 12.0, sm_86, NO_TC=1). 3 samples per arm unless noted; samples interleaved across arms.

### PR #1 prefill (`manifests/final/pr1/`, 6 one-variable manifests, all valid)
| Arm | 1992-tok TTFT (median) | tok/s | 3005-tok TTFT | tok/s | Output |
|---|---|---|---|---|---|
| A dev | 219.1 s | 9.09 | 334.7 s | 8.98 | reference (same hashes as 10-14) |
| B pr1 tip | 82.1 s | 24.27 (**2.67x**) | 119.4 s | 25.17 (**2.80x**) | identical |
| C B + `COLI_CUDA_MOE_DOUBLE=1` | 69.9 s | 28.51 (**3.14x**) | 106.4 s | 28.26 (3.15x) | identical (3/3; the pre-fix divergence did not recur) |
| E B, expert-major off | 123.7 s | 16.10 (B is 1.51x) | — | — | identical |
| F B, warp dense off | 85.4 s | 23.32 (B is 1.04x) | — | — | identical |
- Verify run (both VERIFY modes): 0 mismatches (mm-warp >= 500 calls, expert-major >= 600 calls), output identical.
- Observation: decode right after the GPU-batch prefill is ~8.6% slower on request 1 and ~2.4% on request 2 than after dev's CPU-union prefill (seen in both runs; documented in the PR docs, F42).
- Round 1 of samples was partly page-cache cold (serve-unit memory peaks 7.9-14.6 GiB vs 7.3 GiB in all later samples); within-arm TTFT CV <= 0.32% except A request 1 (1.36%).

### PR #2 decode (`manifests/final/pr2/`, 6 prompts x 300 tokens, `COLI_CUDA_MOE_DOUBLE` unset everywhere)
| Arm | Decode tok/s (median of 3) | vs G1 |
|---|---|---|
| G1 one GPU, no PR #2 switch | 2.866 | 1.00 |
| X1 `V4_GPU_RESIDENT_FIRST=1` | 2.943 | 1.027 |
| X2 X1 + `DSV4_CUDA_PEER_DEVICE=1` `DSV4_CUDA_PEER_COMPUTE=0` (peer reads over NVLink) | 3.441 | 1.200 |
| X3 X1 + `DSV4_CUDA_PEER_DEVICE=1` (peer compute) | **4.044** | **1.411 (+41%)** |
| P1 pr1 tip, same env as G1 (1 sample, sanity) | 2.864 | 0.999 (no regression) |
- All 13 samples x 6 prompts byte-identical to G1. Stdev of the per-sample means <= 0.005 tok/s per arm. GPU0 peaks at 21214 MiB (one X2 reading 21478 MiB); the peer pool takes GPU1 to 23137 MiB of 24576 (X2/X3); GPU1 otherwise unused.
- 3-4.5% above the earlier single-sample numbers (2.78/2.84/3.32/3.87), likely because those ran with DOUBLE set, on intermediate binaries.
- X1->X2 changes two env variables (PEER_DEVICE=1 and explicit PEER_COMPUTE=0); the strictly one-variable manifests are X1->X3 and X2->X3.

### PR #0 (#2212) upload race on the GPU (`manifests/final/drift/`)
- dev 7b55a9b3 vs pr0 a2c8ba7a, 3 cold runs each of the 6-prompt decode bench, then 2+2 runs of a logging-only per-token logit/hidden-hash probe.
- **dev did not drift in these runs**: all runs byte-identical, all per-token logits bit-identical, and pr0 identical to dev. The known triggers were absent (long first request, smaller mirror budget, double bank) and refills were not forced onto the pageable path (whether any refill passed the 20 GiB page-lock cap was not logged), so this neither shows dev race-free nor demonstrates the fix end to end. The race itself is shown by the tiny CUDA harness in PR #0 (h2d_race WAR: 50/50 and 300/300 mismatches on dev, 0 with the fix) and by the 10-14 arm-C divergence (2 of 3) before the fix.
- pr0 vs dev decode speed: 2.892 vs 2.883 tok/s median (0.3%, below the 1% no-change threshold; manifest outcome no-change).

### F42 diagnosis, first run (2026-10-16, `manifests/final2/f42/`; pr2 tip d0e19de2, no PR #2 switches, 3 cold samples per arm)
Arms: GB = GPU-bank prefill (default reserve 3100 MiB), CU = `COLI_CUDA_MOE_BATCH=0` (CPU-union prefill, reserve 600), GR = GB + `DSV4_CUDA_VRAM_RESERVE_MB=600`. Client: 2041-token prompt + 200-token answer, then a 33-token prompt + 200-token answer.
- **Mirror reserve (H3) is a real, general decode cost**: on identical text GB is 2.2% (long) / 2.5% (short) slower than GR: the bank-sized reserve leaves ~175 fewer decode mirrors (14%), so ~11% more uploads at ~1.1 ms each. Zero while the cache fills (first ~10 tokens), steady afterwards. Versus dev the reserve gap is much smaller (dev also reserves 2800 MiB when `COLI_CUDA_MOE_BATCH=1` is set: ~63 mirrors), so it explains the ~0.6% short-prompt delta vs dev but only ~5% of F42's 24-token gap.
- **F42's own cause is still open.** RAM residency after the GPU prefill (H1) was neither confirmed nor excluded: on this prompt the CPU-union prefill picks a different first token ("This text" vs "The text"), so the long answers differ between CU and GB/GR and cannot isolate the prefill path. Next: re-run on the original prompt pair (identical text on both paths), forcing the CPU union with `COLI_CUDA_MOE_BATCH_MIN` so the reserve stays equal, with per-token fetch-wait logging.
- **Finding:** GPU-batch prefill and CPU-union prefill are not bit-identical in general (a near-tie argmax flip at token 0 here, deterministic within each arm). The PR #1 bit-identity guarantees cover routing and the new kernels vs the old GPU kernels, not GPU vs CPU expert arithmetic; the earlier "identical to dev" results hold for the benchmark prompts.

### F42 diagnosis, targeted run (2026-10-16, `manifests/final2/f42v2/`) — **cause confirmed: RAM-tier state after the GPU-bank prefill (H1)**
Same binary (pr2 tip + logging-only per-token trace probe; the probe has no measurable cost), same 3100 MiB reserve, the original bench_pr1 1992-token request; all 14 samples byte-identical text (the PR #1 reference). GB = GPU-bank prefill, U = GB + `COLI_CUDA_MOE_BATCH_MIN=100000` (forces the CPU union); 5 cold samples each after a discarded warm-up.
- Decode of the 24-token answer: GB 2.132 vs U 2.303 tok/s (median), **GB 7.4% slower** (t = -12.7): the same size as F42's original gap (8.6% by medians, 7.4% by means), so this is F42.
- 92% of the extra +0.89 s is fetch_wait, from **+608 RAM misses (+57%, +7.6 GiB of SSD reads)**; uploads and attach equal. Front-loaded: 69 ms/token in tokens 1-5, 48 in 6-10, 21 in 11-24, still a few extra misses per token at token 24.
- Why: the bank refill looks each routed expert up once per layer per prompt (~104 store requests per layer vs ~1065 for the CPU union), so at decode start every RAM resident has usage 1, pins are ranked by id, the resident set is the late-first-seen experts (Jaccard 0.32 vs U), and the pin ramp (4 + layer_requests/24) leaves only 7-9 of 16 pins active. Only ~40% of the experts decode needs were resident vs ~66% after the CPU union. A routing-swap control shows routing differences are not the driver.
- Fix direction: during the bank prefill, keep the store's bookkeeping (usage, request counts / pin ramp, recency) in step with what the CPU union would do, without extra I/O; optionally warm the prompt tail's experts. Being implemented in PR #1.
- Manifest `f42v2-prefill-path-24tok.json` (one variable, valid).

### F42 fix measured (2026-10-16, `manifests/final2/f42fix/`) — commit c7f79815 "GPU-bank prefill keeps the expert store's routing books" (in pr1/v5)
Same request, identical text in all 19 samples; probe builds (logging only); OX = unfixed pr1 tip, FX = fixed, UX = fixed + forced CPU union; 5 cold samples each.
- Decode of the 24-token answer: OX 2.131, **FX 2.264 (+6.3%)**, UX 2.301 tok/s (medians). The fix recovers 67-79% of the gap; FX is 1.5-2.5% behind the CPU union.
- RAM misses over the answer: 1684 / 1244 / 1072; fetch_wait 2.75 / 2.18 / 1.94 s; attach equal. Prefill TTFT FX vs OX +0.2% (not significant), identical prefill I/O.
- Decode-start books now like the CPU union's (~1068 requests per layer, pin usage median 9; 16 pins active, ~12 of them resident vs 16 after the CPU union). The remaining gap is residency in decode forwards 1-5 (the bank reads each expert once per layer, so some prompt-tail experts are not in RAM); from forward 6 on FX matches the CPU union. Closing it would need extra reads during prefill (SSD or copied back from the bank) — left as possible follow-up.
- Known label nits in that record (to tidy): "active pins 7.88/11.95/16" means active-and-resident (active: 7.88/16/16); the manifest field decode_trace_fwd1_25 holds forwards 1-24; forward 1 alone is ~39 ms slower in FX than OX despite fewer misses.

### Upload race, amplified (2026-10-16, `manifests/final2/race/`)
- **dev never drifted**: 17 cold dev runs (pilot: mirror budgets 8/48/192/default, long and short first requests; main: budget 8 + long first request), all with `DSV4_CUDA_PIN_HOST=0` (every refill pageable): identical text per request, and the 15 probe runs bit-identical in every per-token logit and hidden hash. dev + only the upload-ordering commit (cc9c9fab) and the PR #0 tip were identical to dev and to themselves. So on this hardware the race does not change dev's output even under stress; PR #0's GPU evidence remains the micro harness.
- **The race is reproducible on the PR #1 bank path** (the 10-14 configuration, double bank on): pre-fix 48767c01 diverged in 1 of 3 cold runs (token 43 of the second answer), and all 3 pre-fix runs' logits differ from the fixed tree from the first decode step after the 3,005-token bank prefill; the fixed a406901b was bit-identical 3/3 and matched dev's reference texts. Consistent with the fix removing the drift (n=3; the fix only reorders/syncs copies and changes no arithmetic).
- Side result: in the stress setting the upload-ordering commit decodes ~3.8% faster than dev (all 5 runs per build; every H run faster than every dev run). Manifest `race-engine-dev-h2d.json` (speed, identical output; no drift-fix claim).
- Hermes: llama.cpp down 01:46-02:30 and 02:38-04:57 PDT 2026-10-16.

### Side effect on Hermes during the GPU windows (14:12-15:21, 15:24-17:33, 17:36-18:59 PDT, 2026-10-15)
llama.cpp was restored after each window. During them:
- Hermes cron runs had no LLM backend: `shadowfax-autonomous-heartbeat` (*/30) at 14:30, 15:00, 15:30, 16:00, 16:30, 17:00, 17:30, 18:00 and 18:30 (the logged ones failed with "Connection error"; the 18:00 failure notice was delivered), plus one run of job d9d6abf6a201 at 17:00.
- The openclaw-team watchdog "Ensure the active Shadowfax local LLM stack is healthy" (`shadowfax-ensure-active-stack`) failed every 5 min with "backend unhealthy" and restarted nothing, as intended.
Details per window: `manifests/final/*/stats.json`.

## Earlier measurements (2026-10-13/14, intermediate binaries) — superseded by FINAL above
### Prefill — PR #1 (3 cold samples per arm, pr1 48767c01)
Each sample is a fresh server; request 1 = 1992-token prompt, request 2 = 3005-token prompt. Tok/s = prompt tokens / TTFT.

| Arm | Build / switch | 1992-tok TTFT | 3005-tok TTFT | Output |
|---|---|---|---|---|
| A | dev 7b55a9b3 (reference) | 219.0 s (9.10 tok/s) | 333.6 s (9.01) | reference |
| B | pr1 48767c01 | 81.9 s (24.32) | 118.9 s (25.28) | identical to A (all samples) |
| C | B + `COLI_CUDA_MOE_DOUBLE=1` | 69.7 s (28.56) | 106.2 s | 2 of 3 diverged **before** the upload-race fix (a406901b); identical after |
| E | B with expert-major kernel off | 123.6 s (16.12) | — | identical |
| F | B with warp dense kernel off | 85.1 s (23.40) | — | identical |

- Speed-up of B over A: 2.67x (2k) / 2.81x (3k); C: 3.14x (2k).
- E vs B isolates the expert-major MoE kernel: worth ~42 s on 2k tokens. F vs B isolates the warp dense kernel (short rows only, I <= 1024): ~3 s.
- Verify modes (`DSV4_MOE_EXPERT_MAJOR_VERIFY=1`, `DSV4_MM_WARP_VERIFY=1`): 0 bit mismatches over a full prefill.
- Bank reserve log: 4783 MiB with DOUBLE (was a fixed 6100 MiB), 3100 MiB without.
- Manifests: `manifests/pr1-engine-1992.json`, `pr1-engine-3005.json`, `pr1-expert-major-1992.json`, `pr1-mm-warp-1992.json` (all pass `c/experiment_manifest.py`); raw logs in `manifests/evidence/`.
- **Still to do:** re-run B/C on the final PR #1 tip (`tools/bench/run_pr1_final.sh`, arms Bf/Cf) and add a DOUBLE manifest.
- Earlier overnight numbers (prefill on intermediate branches): 2k prompt 220.5 s -> 69.1 s.

### Decode — PR #2 (6 prompts x 300 tokens, one sample each; `tools/bench/bench_decode.py`)
| Step | Decode tok/s (mean of 6) |
|---|---|
| Original, one GPU | 2.78 |
| X1: experts already in VRAM skip the RAM/SSD fetch (`V4_GPU_RESIDENT_FIRST=1`) | 2.84 |
| X2: GPU 1 holds more experts as a peer mirror pool; GPU 0 reads them over NVLink | 3.32 |
| X3: GPU 1 computes its own experts concurrently (`DSV4_CUDA_PEER_DEVICE=1`) | **3.87** (+39%) |

- Per prompt X1 -> X3: 2.61->3.51, 2.73->3.88, 2.93->3.91, 2.69->3.72, 3.22->4.27, 2.86->3.92. Text identical in every case.
- **Caveat (review F43-F45):** these ran with the DOUBLE env set and on intermediate binaries, single sample. Re-measure on the final PR #2 binary, >=3 samples per arm, full text, environment stated.
- Decode profile (one GPU): per token MoE ~249 ms, attention ~107 ms. Over 6x300 tokens: expert attach 193 s, fetch wait 102 s, GPU compute 71 s. After X3: `v4_dec_moe total_s=203 fetch_wait_s=70 attach_s=58`.
- `--ram 48` instead of 40: decode +5% only, so a larger RAM cache alone does little.

### Correctness checks
- `make check` on upstream dev: 2366 OK, 253 skipped (`/tmp/cw/check-devbase.log` on shadowfax). PR #2 final (fe28d80a): 2384 OK.
- Upload-race fix verified with per-token logit and hidden-state hashes (`evidence/determinism/`): all configurations bit-identical at every token after a406901b.

## Reference: what Hermes uses today
Qwen3.8-27B Q4_K_M + MTP draft on llama.cpp, same box, both GPUs: ~72 tok/s decode, ~760-940 tok/s prefill.
