# Measurements

All on shadowfax (see environment.md), DeepSeek V4 Flash REAP 150B, `--ram 40 --ctx 8192`, llama.cpp stopped for the window, one fresh server per sample. The engine is greedy, so "identical" means byte-identical output text (sha256 compared).

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

### PR #0 upload race on the GPU (`manifests/final/drift/`)
- dev 7b55a9b3 vs pr0 a2c8ba7a, 3 cold runs each of the 6-prompt decode bench, then 2+2 runs of a logging-only per-token logit/hidden-hash probe.
- **dev did not drift in these runs**: all runs byte-identical, all per-token logits bit-identical, and pr0 identical to dev. The known triggers were absent (long first request, smaller mirror budget, double bank) and refills were not forced onto the pageable path (whether any refill passed the 20 GiB page-lock cap was not logged), so this neither shows dev race-free nor demonstrates the fix end to end. The race itself is shown by the tiny CUDA harness in PR #0 (h2d_race WAR: 50/50 and 300/300 mismatches on dev, 0 with the fix) and by the 10-14 arm-C divergence (2 of 3) before the fix.
- pr0 vs dev decode speed: 2.892 vs 2.883 tok/s median (0.3%, below the 1% no-change threshold; manifest outcome no-change).

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
