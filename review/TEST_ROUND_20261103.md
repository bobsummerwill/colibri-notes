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
