# Test / profiling round estimate for the 6+1 PRs (2026-11-03)

Assumes shadowfax (2×RTX 3090 + NVLink) and computer4 (RTX 5070, sm_120) are
free, the DeepSeek V4 Flash checkpoints are already on local disk, and the
Hermes/llama services on shadowfax are paused for the GPU windows.

## What "complete" means per PR

1. **Build**: CPU-only engine + the CUDA backend for the PR's arch
   (sm_86 on shadowfax, sm_120 on computer4). The V4 backend is a monolithic
   `.cu`; a cold nvcc build is the single most expensive step per branch.
2. **Unit tests**: the nine CUDA binaries (`dsv4-cuda-tests`), the Python
   source/guard tests, and the placement/service tests the PR touches.
3. **Benchmark**: A/B vs `dev` on the PR's feature, same prompt and model,
   identical-output check plus decode/prefill tok/s.
4. **Profiling**: built-in `DECODE`/`ATTN`/`MOE` lines and `V4_SERVE_STATS_LOG`;
   `nsys` is unavailable on shadowfax, `ncu` is not usable on these cards for
   the long runs.
5. **Notes**: numbers and verdicts written back into the PR.

## Cost per PR

| step | shadowfax (2×3090) | computer4 (5070) |
|---|---:|---:|
| cold CPU build | 10 min | 10 min |
| cold CUDA build (one arch) | 20–30 min | 15–25 min |
| nine unit tests | 10 min | 10 min |
| targeted A/B benchmark | 45–90 min | 30–60 min |
| analysis/notes | 20 min | 20 min |
| **per PR, wall, both boxes in parallel** | **~2–3 h** | (parallel) |

The two machines can build/test the same branch simultaneously, so per-PR wall
clock is the slower box.

## Totals

| scope | wall clock |
|---|---:|
| PRs 1–2 (build fixes, placement): light, no model bench except one regression | 3–4 h |
| PRs 3–4 (resident, kernels): heaviest unit + A/B work | 6–8 h |
| PRs 5–6 (mirror/VRAM, peer/fault): A/B + fault injection | 5–7 h |
| PR 7 integration: full matrix on the merged tree (placement vs peer, resident 0/auto/1, VRAM-first, fault paths) | 6–8 h |
| buffer for rebuilds/regressions (25–30 %) | 5–7 h |
| **total, complete round** | **~25–34 h ≈ 3.5–4 working days** |

If only the aggregate (PR 7) gets the full matrix and each feature PR gets its
targeted tests + one A/B, the round drops to **~15–20 h (~2–2.5 days)**, with
the integration still fully covered.

## What parallelises

- shadowfax and computer4 are independent: per PR, both can run at once.
- Builds are CPU-bound on each box; benchmarks are GPU-bound. Do not build and
  bench the same box at once.
- The 2405 lane (PRs 5–6) can be tested while the 1766 lane (3–4) is still in
  review; only PR 7 needs both.
- Nightly/detached runs are fine for builds, unit tests and the benchmark
  loops; the analysis and go/no-go remain manual.

## Risks that move the estimate

- Cold nvcc builds of the monolithic V4 backend can take longer than 30 min on
  the 3090 box if the machine is loaded.
- Full-checkpoint runs (not REAP) have long prefill and 48 GB RAM pressure;
  each adds 15–30 min per configuration.
- Any regression found in a feature PR re-runs its build + bench (~2–3 h).
