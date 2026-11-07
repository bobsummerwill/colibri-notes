# Rebase and verification round — 6-PR DSV4 stack (2026-11-07)

Base: upstream `dev` at `e24cd51a` (2026-11-06). Previous split base was
`6d907b56`. All seven branches are in `bobsummerwill/colibri` as
`split-20261107/*`; PRs are #2664–#2670 against `JustVugg/colibri`.

This file is the full record for the round. The PR descriptions are deliberately
short and point here.

## Branch map

| # | PR | branch | tip | commits ahead of dev | diff vs dev |
|---:|---|---|---|---:|---|
| 1 | #2664 | `split-20261107/dsv4-build-fixes` | `60986bba` | 2 | 2 files, +14/−1 |
| 2 | #2665 | `split-20261107/dsv4-multigpu-layers` | `1d152bee` | 6 | 12 files, +462/−59 |
| 3 | #2666 | `split-20261107/dsv4-resident` | `e893e544` | 8 | 16 files, +1230/−61 |
| 4 | #2667 | `split-20261107/dsv4-cuda-kernels` | `c34d3a92` | 13 | 30 files, +2996/−135 |
| 5 | #2668 | `split-20261107/dsv4-mirror-vram-first` | `4891a34d` | 2 | 7 files, +821/−192 |
| 6 | #2669 | `split-20261107/dsv4-peer-pool` | `74bfa093` | 6 | 15 files, +2200/−264 |
| — | #2670 | `split-20261107/dsv4-cplus-integration` | `4c468115` | 24 | 37 files, +5290/−384 |

Commit counts include prerequisite branches and the test/build fix commits
listed below. Feature commit content and authorship are preserved.

## Rebase

The six feature branches were replayed onto `e24cd51a` with `git cherry-pick`,
keeping original authors and messages. Conflicts resolved:

- `c/deepseek_v4.c` — route-scratch block vs mirror-cache declaration; indexer
  GPU prep vs the CPU fast path; output-projection helpers; engine close.
- `c/backend_cuda_dsv4.cu` — `dev_release` loops; fused MoE vs peer `moe_once`;
  duplicate RMSNorm.
- `c/dsv4.def` — `proj_batch*` plus the resident indexer exports.
- `c/deepseek_v4_internal.h` — resident MoE entry-point declarations.
- `docs/deepseek-v4.md` — placement, mirror and peer sections.

The integration branch is the previous integration tree merged with `e24cd51a`,
then repaired (see defects below). It is a draft until #2667 and #2669 land.

## Environment

- **shadowfax:** 2×RTX 3090, sm_86, CUDA 12.0, `NO_TC=1`, 32 cores.
- **computer4:** RTX 5070, sm_120, CUDA 12.8.
- `llama-server-active` and `camofox-browser` stopped; `hermes-gateway`
  frozen with `SIGSTOP` for the round (its system unit needs root). All restored
  afterwards; llama `/health` healthy.

## Builds

| host | arch | result |
|---|---|---|
| shadowfax | sm_86, CUDA 12.0, `NO_TC=1` | all seven branches clean |
| computer4 | sm_120, CUDA 12.8 | all six feature branches clean |

`NO_TC=1` is required on CUDA 12.0: the optional cuBLASLt block-scaling APIs
are absent. Final integration and cuda-kernels rebuilds after the last fixes
are clean.

## Tests

### Full `make -j16 check` (computer4)

`make check` is `clean` + `portable` + `test`; it builds the whole project and
runs the C and Python suites.

| PR | tests | result |
|---|---:|---|
| #2664 | 3020 | OK, 379 skipped |
| #2665 | 3021 | OK, 375 skipped |
| #2666 | 3021 | OK, 375 skipped |
| #2667 | 3022 | OK, 375 skipped (after the export fix) |
| #2668 | 3022 | OK, 375 skipped |
| #2669 | 3032 | OK, 375 skipped |
| #2670 | 3034 | OK, 375 skipped |

### Targeted CUDA and source tests (shadowfax, 2×3090)

- `dsv4-cuda-tests` 9/9 on #2667 and #2670.
- `test_dsv4_multigpu_cuda 0,1`, `test_dsv4_resident_route 0,1`,
  `test_dsv4_resident_indexer`: pass.
- `test_dsv4_peer_init_cuda` and `test_dsv4_peer_moe_cuda`: pass on #2669 and
  #2670, including the sticky/lost fault injection paths.
- `test_v4_gpu_placement.py` (mock, ELF section GC): pass on #2665, #2666,
  #2667, #2670.
- `test_v4_mirror_guard_source.py`, `test_v4_peer_init_source.py`,
  `test_dsv4_cuda_lost_source.py`, `test_deepseek_v4_cuda_arch_flags.py`: pass
  where present.
- `tests/test_deepseek_v4` (resource plan, sparse attention, core): pass on all
  seven branches.

### Refusals and fault behaviour

- `DSV4_CUDA_DEVICES=0,1` + `DSV4_CUDA_PEER_DEVICE=1`: exits 1 with the
  mutual-exclusion line.
- `DSV4_CUDA_RESIDENT_EXPERTS=1` on REAP (experts exceed device 0 budget):
  exits 1 with the budget line. `auto` keeps the LRU cache and serves.
- Peer fault injection: peer latched off, decode continues on the primary;
  sticky faults latch CUDA off process-wide and the host paths continue.

## Benchmarks

REAP checkpoint `deepseek-v4-flash-reap` (132 experts, 85 GB),
`--ram 40 --ctx 8192`; two prompts × 128 generated tokens × 3 repetitions;
median; greedy. Identical output text in every REAP cell for a given prompt
(sha `3846c4863341…` and `d740de876936…`).

| config | TTFT (s) | decode tok/s (short_0 / short_1) | mean |
|---|---:|---:|---:|
| `dev` single | 3.83 | 2.763 / 2.775 | 2.77 |
| #2664 build-fixes single | 3.84 | 3.279 / 3.289 | 3.28 |
| #2665 multigpu-layers single | 3.69 | 3.205 / 3.289 | 3.25 |
| #2666 resident single | 3.69 | 3.300 / 3.304 | 3.30 |
| #2667 cuda-kernels single | 3.73 | 3.295 / 3.293 | 3.29 |
| #2667 cuda-kernels, `DSV4_CUDA_DEVICES=0,1` | 3.75 | 4.384 / 4.275 | 4.33 |
| #2668 mirror-vram-first single | 3.71 | 3.287 / 3.283 | 3.29 |
| #2669 peer-pool, `DSV4_CUDA_PEER_DEVICE=1` | 4.05 | 5.607 / 5.536 | 5.57 |
| #2670 integration single | 3.72 | 3.293 / 3.292 | 3.29 |
| #2670 integration, `DSV4_CUDA_DEVICES=0,1` | 3.75 | 4.382 / 4.290 | 4.34 |
| #2670 integration, `DSV4_CUDA_PEER_DEVICE=1` | 4.04 | 5.610 / 5.542 | 5.58 |

Additional cells:

- **Full checkpoint** (156 GB, 64 generated tokens, one run): single
  2.795 / 2.998 tok/s; peer compute 3.424 / 4.029 tok/s. Same text in both
  (`2cd0553c…` / `de8d67d6…`).
- **Prefill** (integration, 1992-token prompt): TTFT 66.0 s, prefill
  30.17 tok/s, post-prefill decode 2.505 tok/s.
- **nsys decode** (integration single): `mv_fp4_grouped` 32.0 %,
  `mv<8>` 25.0 %, `compressor_project_exact` 14.1 %,
  `fp8_ref_matmul_rows8` 13.8 %, `indexer_head_project` 3.5 %,
  `head_scores_exact` 1.3 %.
- Earlier smoke/long numbers from the previous base are in
  `review/TEST_ROUND_20261103.md`; they are not comparable across bases.

## Defects found and fixed in this round

| PR | fix | commit |
|---|---|---|
| #2665, #2666, #2667, #2670 | placement mock needed `dsv4_cuda_proj_batch_release` from current dev | `83c2a28e` / cherry-picks |
| #2669, #2670 | `dev_release` freed only `ng` slots after a failed init; now all `n` touched devices | `74bfa093`, `3d9b77eb` |
| #2670 | resident MoE entry points used without declarations; added to `deepseek_v4_internal.h` | `a523128c` |
| #2665, #2666, #2667, #2670 | hand-listed headers on `-MMD` test rules (`test_v4_gpu_devices`, `test_v4_bank_pair`); removed | `034e9a3c`, `1d152bee` + cherry-picks |
| #2667, #2670 | `dsv4_cuda_wo_decode` declared/resolved but missing from `dsv4.def`; added | `c34d3a92`, `f7e682a1` |
| #2670 | integration's optional peer best-effort lookup used a `peer_best_effort` slot for `dsv4_cuda_set_peer_best_effort`; renamed | `4c468115` |

The full suite found the last three; the smaller CUDA/source suites did not.
All branches pass the full suite after the fixes.

## CI state

At the end of the round, all 116 checks per new PR were still queued
(0 completed, 0 failed). #2405's 116 checks were green but the PR was
conflicting; it was closed as superseded by #2668/#2669/#2670.

## Services and cleanup

- `llama-server-active`, `camofox-browser`, `hermes-gateway` restored.
- Fork PRs #1–#11 and #3 closed; 28 obsolete branches deleted, plus
  `release/2.0.0`.
- Open upstream PRs from the fork: #2664–#2670; #2405 closed.

## Reproduction pointers

- Shadowfax worktree: `/tmp/colibri-split`; integration: `/tmp/colibri-int`;
  peer: `/tmp/colibri-peer`.
- Benchmark client: `/tmp/bench_short.py` (2 prompts, `REPS`, `MAX_TOKENS`).
- Scripts: `/tmp/bench-split.sh`, `/tmp/bench-extra.sh`, `/tmp/bench-prefill.sh`,
  `/tmp/bench-int128.sh`, `/tmp/nsys-int.sh`.
- computer4: `/tmp/check-split4.sh` (full check), `/tmp/build-split120.sh`.
- nsys shim details: `ops/NSYS_SHADOWFAX.md`.
