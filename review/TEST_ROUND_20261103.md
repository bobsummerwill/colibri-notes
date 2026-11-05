# Test / profiling round, 6-PR stack (2026-11-03)

Executed on shadowfax (2×RTX 3090, sm_86, CUDA 12.0, `NO_TC=1`, 32 cores) and
computer4 (RTX 5070, sm_120, CUDA 12.8). llama/TabbyAPI and hermes were stopped
for the round; REAP checkpoint (`deepseek-v4-flash-reap`, 85 GB) used for the
benchmarks with `--ram 48 --ngen 32`.

## Result

All six PRs and the integration build and pass their tests. No merge was done
(as instructed). Seven PRs carry the results as comments; each branch was fixed
where the refreshed `dev` or the merge exposed a break.

## Builds

| host | arch | result |
|---|---|---|
| shadowfax | sm_86, CUDA 12.0, `NO_TC=1` | all 7 branches clean (~30–60 s each) |
| computer4 | sm_120, CUDA 12.8 | all 7 branches clean (~30–40 s each) |

The `NO_TC=1` requirement is real: CUDA 12.0 lacks the cuBLASLt block-scaling
APIs that the optional TC path uses.

## Tests

| suite | where | result |
|---|---|---|
| `test_deepseek_v4` (CPU unit) | both | ok |
| `test_deepseek_v4_cuda_arch_flags.py` | both | ok |
| `dsv4-cuda-tests` (nine CUDA tests) | shadowfax 2×3090 | 9/9 on PR 4, PR 6, integration |
| `test_v4_gpu_placement` (mock, Python driver) | shadowfax | ok on PR 2, 3, 4, 6 |
| `test_dsv4_multigpu_cuda 0,1` | shadowfax | ok |
| `test_dsv4_resident_route 0,1` | shadowfax | ok (normal + hash routes) |
| `test_dsv4_peer_init_cuda`, `test_dsv4_peer_moe_cuda` | shadowfax | ok, incl. sticky/lost fault injection |
| `test_v4_mirror_guard_source.py`, `test_v4_peer_init_source.py`, `test_dsv4_cuda_lost_source.py` | shadowfax | ok |

## Benchmarks (REAP, 32 tokens, single run each)

| config | TTFT | steady decode |
|---|---:|---:|
| `dev`, one 3090 | 8.45 s | 2.55 tok/s |
| integration, one 3090 | 3.99 s | 2.84 tok/s |
| integration, `DSV4_CUDA_DEVICES=0,1` | 4.04 s | 3.25 tok/s |
| integration, `DSV4_CUDA_PEER_DEVICE=1` | 3.99 s | 3.31 tok/s |
| PR 3 (resident), `RESIDENT_EXPERTS=0` | 7.05 s | 2.61 tok/s |
| PR 3 (resident), `auto` (budget fallback) | 3.97 s | 2.58 tok/s |
| PR 4 (head off / on) | 3.98 / 3.96 s | 2.71 / 2.77 tok/s |

`generated_text` sha256 prefix `2ae4a2c472b2ddf7` in dev, single, placement and
peer configurations — identical output. Resident `auto` on REAP falls back to
the LRU cache with the expected `resident-experts=cache-fallback (…exceed device
0 budget…)` line; 132 experts do not fit 24 GB.

## Profiling

`nsys profile --trace=cuda` + a separate `nsys stats` on a decode run. Top GPU
kernels by time: `mv_fp4_grouped` 28 %, `compressor_project_exact` 28 %,
`mv<8>` variants 25 %, `mm_batch`/`mm_batch_w` 8 %, the rest spread across
attention/indexer/head. Decode is dominated by expert matvec and compressor
work, not by the exact head.

## Defects found and fixed during the round

| PR | fix |
|---|---|
| 2 | `unified` guard state left function-static after the rebase; restored to base behaviour (per-cache state is PR 5) |
| 2 | `dsv4_cuda_init` re-init refusal (`||ng`) dropped by refreshed dev; restored, matching the multigpu contract |
| 2 | placement mock: missing expert-bank/activation/set stubs, missing FP4 format on fake views; free-VRAM assertion relaxed until per-cache guard |
| 3 | `v4_gpu_resident_reserve_mb` used before definition in the head upload; moved above |
| 3 | resident-route test hard-coded devices `{0,5}`; reads `DSV4_TEST_DEVICES` (later the shared header) |
| 6 (integration) | merge dropped PR 4's matvec format guard (fmt 4/8/9/16/32); restored |
| 6 (integration) | duplicated struct members and a lost closing brace from applying the integration diff onto the new merge |
| 6 (integration) | `shared_missing` mock lost, duplicate stubs; deduped/restored |
| 6 (integration) | `test_v4_peer_init_source.py` updated for mutual exclusion + three init call sites |

## Caveats

- Single-run timings on a warm page cache; not a statistical benchmark. TTFT
  varies with page-cache state (visible in the resident runs).
- The full (non-REAP) checkpoint was not run.
- `V4_GPU_RESIDENT_FIRST` requires `COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER`;
  the default build returns 0, so it could not be A/B'd here.
- `nsys` on shadowfax works through the user-level shim (see
  `ops/NSYS_SHADOWFAX.md`); in-process `--stats=true` is broken on this build.
- computer4's role was builds + sm_120 unit tests; the 12 GB 5070 cannot run the
  V4 model meaningfully.

## State after the round

- PRs #4, #5, #6, #7, #8, #9, #11 open, green builds/tests, no merges.
- llama/TabbyAPI restarted by request; hermes left stopped (it was stopped by
  the user before the run).
- All results also posted as comments on each PR.

## Long round (2026-11-04)

128-token decode, 3 repeats per cell, median; REAP unless stated; same prompt
everywhere; llama/TabbyAPI and hermes stopped. Output text identical in every
REAP cell (sha `bdda375698df`) and in both full-checkpoint cells.

| config | TTFT | steady |
|---|---:|---:|
| `dev` | 9.5 s | 3.54 tok/s |
| integration single | 5.6 s | 3.93 tok/s |
| integration, `DSV4_CUDA_DEVICES=0,1` | 5.7 s | 4.88 tok/s |
| integration, peer compute | 5.6 s | 4.98 tok/s |
| full checkpoint single | 4.35 s | 2.61 tok/s |
| full checkpoint placement | 4.06 s | 3.09 tok/s |

Per feature (branch-local):

| cell | value |
|---|---|
| resident `0` / `auto` | 3.59 / 3.60 tok/s (auto falls back on budget) |
| `V4_GPU_RESIDENT_FIRST` 0 / 1 | 3.57 / 3.62 tok/s |
| mirrors 512 / 2048 / 4096 | 2.99 / 3.57 / 3.57 tok/s |
| peer off / P2P / compute | 3.58 / 3.86 / 4.27 tok/s |
| head off / on | 3.72 / 3.93 tok/s |
| batch-head, wo, compressor, indexer-prep | within noise (±1.5 %) |

Other numbers:

- prefill-heavy 512-token prompt: TTFT 53.2 s single / 53.3 s placement;
  post-prefill decode 3.30 / 3.32 tok/s. Placement does not help prefill.
- soak: 15 consecutive runs, identical, 3.94 tok/s mean.
- peer fault injection: peer latched off, decode continued on the primary,
  same output.
- nsys decode: `mv_fp4_grouped` 31.7 %, `mv<8>` 28.1 %, `compressor` 22.9 %.
- nsys prefill: `compressor_project_exact` 63.8 %, `mm_batch*` 27.2 %.

Corrections to the smoke round: the dual expert loader is on by default, so
`V4_GPU_RESIDENT_FIRST` is measurable (the earlier comment on #8 said
otherwise and has been corrected).

## Expanded round (2026-11-04, second pass)

Interleaved (5 rounds × 3 reps, 128 tokens, dev→single→placement→peer):

| config | mean |
|---|---:|
| dev | 3.60 tok/s |
| single | 3.95 |
| placement | 4.96 |
| peer | 5.03 |

Context / generation:

| cell | result |
|---|---|
| 1k-token context (4k-char prompt) | TTFT 41.9 / 42.0 s (single/placement), steady 3.21 / 3.24, identical output |
| 256-token generation, single | 3.77 overall, TTFT 5.5 s, steady 4.09 |
| CPU-only, 8 tokens | 2.02 tok/s |

Soak (1024 tokens × 3, VRAM sampled every 5 s): single 4.27 tok/s peak
23 564 MiB; placement 5.75 tok/s peak 23 560 MiB both cards; peer 5.74 tok/s
peak 23 564 / 23 136 MiB. Stable, identical output, ~96 % VRAM at peak.

Feature cells:

| cell | value |
|---|---|
| `INDEXER_HEADS` 0/1 | 3.93 / 3.95 |
| `RESIDENT_INDEXER` 0/1 | 3.93 / 3.95 |
| `COLI_CUDA_MOE_BATCH` 0/1 | 3.92 / 3.79 |
| `COLI_CUDA_ATTN_BATCH` 0/1 | 3.89 / 3.75 (text changes: `65e321b385ef`) |
| `PIN_CAP_GB` 0/1 | 3.68 / 3.67 |
| `VRAM_RESERVE_MB` 600/2000 | 3.58 / 3.52 |
| peer mirrors 512/4096 | 3.89 / 4.30 |
| peer reserve 512/2048 | 4.30 / 4.27 |
| full model peer compute | 3.21 |

Refusals: `RESIDENT_EXPERTS=1` over budget exits 1 with the budget line;
`DSV4_CUDA_DEVICES>1` + `PEER_DEVICE` exits 1 with the mutual-exclusion line.

Lane profiles: decode ≈ 32 % `mv_fp4_grouped`, 28 % `mv<8>`, 23 % compressor;
prefill 64 % compressor, 27 % `mm_batch*`; peer decode GPU memops 98.7 % HtoD
(2.07 s, 18 379 copies), 1.3 % DtoH, 0.04 % PtoP.

sm_120 runtime: `test_dsv4_quant_cuda` passes on the 5070 (35 769 cases, 0
mismatches). The remaining CUDA tests require two distinct devices by
construction; they ran on the 2×3090.

Caveat: `COLI_CUDA_ATTN_BATCH=1` changes greedy text (documented
summation-order difference, not a break).

## PR descriptions rewritten (2026-11-04)

Every PR body was rewritten to be free-standing: the three source streams
(upstream #1766, upstream #2405, the head-followups fixes), the recomposition
(the 31 commits + merge -> 15 commits in six feature PRs plus integration),
the colibri-notes planning/test documents, the stack table, the full testing
approach and results for that PR, and its own known deviations. Per-PR bodies
are 4.7–7.6 kB; the comments keep the raw round-by-round numbers.

## PR descriptions now hold everything (2026-11-04)

The round-by-round comments were folded into the PR bodies and deleted, so each
description is the single source: history, stack, testing method, per-PR
results (smoke + long + expanded, including the full checkpoint, soaks, stats,
prefill/context numbers and nsys), defects found while testing, and known
deviations. Comments are now zero on #4–#9 and #11.

## Descriptions restructured, profiling added everywhere (2026-11-04)

Each PR body now leads with the history, the testing method and the stack,
then what the PR is, then tests, benchmarks, **profiling** and defects. Two
missing captures were taken: placement decode (`DSV4_CUDA_DEVICES=0,1`) and
resident decode (`auto`, fallback path), so every PR except #4 has its own
nsys numbers; #4 states explicitly that no profile belongs to it.
