# Measurements

All on shadowfax (see environment.md), DeepSeek V4 Flash REAP 150B, `--ram 40 --ctx 8192`, llama.cpp stopped for the window. The engine is greedy, so "identical" means byte-identical output text (sha256 compared).

## Prefill — PR #1 (final benchmark, 3 cold samples per arm)
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

## Decode — PR #2 (6 prompts x 300 tokens, one sample each; `tools/bench/bench_decode.py`)
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

## Correctness checks
- `make check` on upstream dev: 2366 OK, 253 skipped (`/tmp/cw/check-devbase.log` on shadowfax). PR #2 final (fe28d80a): 2384 OK.
- Upload-race fix verified with per-token logit and hidden-state hashes (`evidence/determinism/`): all configurations bit-identical at every token after a406901b.

## Reference: what Hermes uses today
Qwen3.8-27B Q4_K_M + MTP draft on llama.cpp, same box, both GPUs: ~72 tok/s decode, ~760-940 tok/s prefill.
