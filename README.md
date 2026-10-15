# colibri-notes

Project notebook for Bob Summerwill's work on [colibri](https://github.com/JustVugg/colibri) (pure-C/CUDA engine that runs huge MoE models by streaming experts from NVMe and caching hot experts in RAM/VRAM), done with Claude Code on 2026-10-12..14. Not branch-specific; the code lives in the fork [bobsummerwill/colibri](https://github.com/bobsummerwill/colibri).

**Rule: no PRs and no pushes to upstream JustVugg/colibri.** Branches go to the fork only; Bob decides on submission.

## Status (2026-10-15)
- Model under test: **DeepSeek V4 Flash REAP 150B** (`puwaer/DeepSeek-V4-Flash-0731-reap-150b`, 132 routed experts/layer instead of 256, fp4 experts, 79 GiB) on **shadowfax** (2x RTX 3090 + NVLink, 62 GB RAM) — see [environment.md](environment.md).
- Results ([measurements.md](measurements.md)): prefill 9 -> 24-28 tok/s (2.7-3.1x); decode 2.78 -> 3.87 tok/s with two GPUs (+39%); all greedy outputs identical to upstream.
- Practical verdict ([decisions.md](decisions.md)): great upstream contribution, not a replacement for Qwen3.8-27B on llama.cpp (~72 tok/s) for Hermes.
- Planned upstream PRs (target `dev`, base `7b55a9b3`):
  - **PR #0 DeepSeek V4 CUDA bug fixes** — 12 standalone upstream fixes (incl. a real decode race, F26 loader jobs, F2 speculative batches, F27 CUDA last-error, F4 mirror-slot rollback, the `make dsv4-cuda-test` stub stop) + doc corrections.
  - **PR #1 GPU batched prefill for expert-pruned checkpoints** + expert-major MoE kernel + warp dense kernel.
  - **PR #2 Two-GPU decode** (VRAM-first lookups, peer mirror pool over NVLink, peer compute).

## Branch map (fork)
Compare against `dev` (upstream's PR target), never `main`.

| Branch | What |
|---|---|
| `pr0/dsv4-fixes` | **PR #0 final**: 12 upstream bug fixes + docs, 13 commits on dev 7b55a9b3 (a2c8ba7a) — [compare](https://github.com/bobsummerwill/colibri/compare/dev...pr0/dsv4-fixes) |
| `pr1/reap-prefill` | **PR #1 final**: REAP batched prefill + kernels, 7 commits on PR #0 (93b74183) — [compare](https://github.com/bobsummerwill/colibri/compare/pr0/dsv4-fixes...pr1/reap-prefill) |
| `pr2/two-gpu-decode` | **PR #2 final**: two-GPU decode, 8 commits on PR #1 (d0e19de2) — [compare](https://github.com/bobsummerwill/colibri/compare/pr1/reap-prefill...pr2/two-gpu-decode) |
| `pr1/fixed`, `pr2/fixed` | The benchmarked/validated code (a406901b, fe28d80a), straight on dev; reference only. Final PR #1/#2 code = these + the PR #0 fixes they lacked + text fixes (see review/STACK_CHECK.md "code differences") |
| `pr0/v3`, `pr1/v3`, `pr0/dsv4-fixes-v1`, `-v2`, `pr1/reap-prefill-v1`, `-v2`, `pr2/two-gpu-decode-v1`, `-v2`, `pr2/split`, `pr2/ref`, `fix1/*`, `fix2/*`, `reap-gpu-prefill`, `vram-first` | Earlier versions / history |

Final stack check: ready, 0 blockers/majors — `review/STACK_CHECK.md`; draft PR summaries — `review/PR_SUMMARIES_DRAFT.md`.

## Contents
- [environment.md](environment.md) — hardware, services, safe GPU-downtime procedure, build/run commands, paths.
- [measurements.md](measurements.md) — every number with config and evidence.
- [investigations.md](investigations.md) — how the bottlenecks and bugs were found (incl. the upload race).
- [decisions.md](decisions.md) — choices and rationale; token/time cost of the work.
- [open-problems.md](open-problems.md) — what is untested / next.
- `review/` — the 214-agent adversarial review (REVIEW_REPORT.md: F1..F60), its briefing, an old PR draft.
- `manifests/` — colibri experiment manifests (validated with `c/experiment_manifest.py`) + raw evidence tarballs.
- `tools/` — benchmark clients/runners (`bench/`), determinism probes and debug patches (`probes/`), kernel microbenchmarks (`microbench/`), workflow scripts used to orchestrate agents (`workflows/`), usage analysis (`analysis/`).
- `evidence/` — per-token logit/hidden-hash logs of the determinism probes; misc run logs.
