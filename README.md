# colibri-notes

> **Naming key.** These notes use internal names for the upstream pull requests. **PR #0** is the six
> fixes opened as JustVugg/colibri [#2212](https://github.com/JustVugg/colibri/pull/2212) (0a), [#2213](https://github.com/JustVugg/colibri/pull/2213) (0b), [#2214](https://github.com/JustVugg/colibri/pull/2214) (0c), [#2215](https://github.com/JustVugg/colibri/pull/2215) (0d),
> [#2216](https://github.com/JustVugg/colibri/pull/2216) (0e) and [#2217](https://github.com/JustVugg/colibri/pull/2217) (0f). **PR #1** (batched GPU prefill for expert-pruned checkpoints) is
> [#2264](https://github.com/JustVugg/colibri/pull/2264), opened 2026-10-20; **PR #2** (two-GPU decode) is not opened yet. Documents dated before 2026-10-17 describe earlier
> versions of these changes, on the old base 7b55a9b3.

Project notebook for Bob Summerwill's work on [colibri](https://github.com/JustVugg/colibri) (pure-C/CUDA engine that runs huge MoE models by streaming experts from NVMe and caching hot experts in RAM/VRAM), done with Claude Code on 2026-10-12..14. Not branch-specific; the code lives in the fork [bobsummerwill/colibri](https://github.com/bobsummerwill/colibri).

**Rule: no PRs and no pushes to upstream JustVugg/colibri.** Branches go to the fork only; Bob decides on submission (PR #0 and PR #1 were opened at his request).

## Status (2026-10-22)
- **PR #1 is [#2264](https://github.com/JustVugg/colibri/pull/2264)** (opened 2026-10-20). JustVugg's review (2026-10-22, [comment](https://github.com/JustVugg/colibri/pull/2264#issuecomment-6335156265)) left it out of the day's batch (#2284) for three points: (1) merge dev (a one-line conflict in `coli_v4_gpu_engine_open`); (2) `DSV4_MM_WARP` is on by default (auto), so measure the **default configuration (no `COLI_CUDA_MOE_BATCH`) on the full V4 Flash**, dev against the branch, prompt and decode, interleaved; it stays on by default if bit-identical and not slower; (3) say in the description why `test_v4_moe_batch_speculative_source.py` is deleted (its checks moved into `test_v4_moe_batch_lifecycle_source.py`).
- **Done 2026-10-22:** dev 7df8c27c merged into `docs5/pr1-reap-prefill` and, through it, `docs5/pr2-two-gpu-decode` (merge commits, no force-push; PR #2 had two more conflicts with dev's CUDA expert tier `cu_tier.c`, resolved keeping both sides, see [decisions.md](decisions.md)); the PR description says why the test file went; a reply on #2264 says the full-model measurement is coming.
- **Next:** the full DeepSeek V4 Flash (`deepseek-ai/DeepSeek-V4-Flash-0731`, 256 experts, 167 GB) downloaded to shadowfax `~/colibri-models/deepseek-v4-flash`; a one-sample pilot (memory, timing, CUDA tests), then the overnight A/B `final6` ([tools/bench/final6/](tools/bench/final6/)).

## Status (2026-10-17)
- Model under test: **DeepSeek V4 Flash REAP 150B** (`puwaer/DeepSeek-V4-Flash-0731-reap-150b`, 132 routed experts/layer instead of 256, fp4 experts, 79 GiB) on **shadowfax** (2x RTX 3090 + NVLink, 62 GB RAM) — see [environment.md](environment.md).
- **Rebased onto upstream dev 62eaa9dc** (298 commits past the old base 7b55a9b3) on 2026-10-17 and re-validated there: `make check` passes at every tip, every commit compiles with 0 diagnostics, CUDA builds clean, `dsv4-cuda-test` / `cuda-test` pass on the GPUs, and the final GPU benchmark (`manifests/final4/`) passed every gate. Details: [review/REBASE_20261017.md](review/REBASE_20261017.md).
- Results on the new base ([measurements.md](measurements.md)): prefill 1.69x / 1.86x over upstream dev (2.01x / 2.09x with the double bank) — upstream's own prefill got ~1.7x faster since 7b55a9b3, so the old 2.7-3.1x no longer applies; two-GPU decode 3.01 -> 4.34 tok/s (+44%); all greedy outputs identical to upstream.
- Practical verdict ([decisions.md](decisions.md)): great upstream contribution, not a replacement for Qwen3.8-27B on llama.cpp for Hermes. A blind 12-prompt quality comparison (2026-10-16, [quality/REPORT.md](quality/REPORT.md)) found REAP-150B **not better** than Qwen3.8-27B while ~21x slower and needing Hermes stopped.
- Upstream PRs (target `dev`, base `62eaa9dc`), drafts in [review/PR_DESCRIPTIONS_20261017.md](review/PR_DESCRIPTIONS_20261017.md): six independent PR #0 fixes (0a-0f), then PR #1 (REAP prefill) and PR #2 (two-GPU decode), which depend on all six. **PR #0 opened 2026-10-18 at Bob's request** (from the `rebased/*` branches): [#2212](https://github.com/JustVugg/colibri/pull/2212) 0a, [#2213](https://github.com/JustVugg/colibri/pull/2213) 0b, [#2214](https://github.com/JustVugg/colibri/pull/2214) 0c, [#2215](https://github.com/JustVugg/colibri/pull/2215) 0d, [#2216](https://github.com/JustVugg/colibri/pull/2216) 0e, [#2217](https://github.com/JustVugg/colibri/pull/2217) 0f. PR #1/#2 not opened yet.

## Branch map (fork)
Compare against upstream `dev`, never `main`. **The rebased branches are on the fork as `rebased/<name>`** (force-pushing over the old names needs Bob's go-ahead); the old names still hold the 2026-10-15 stack on 7b55a9b3.

| Branch (fork) | What |
|---|---|
| `rebased/pr0a/dsv4-upload-ordering` | [#2212](https://github.com/JustVugg/colibri/pull/2212) (0a): h2d upload ordering + CUDA error hygiene + shutdown zeroing (4 commits) |
| `rebased/pr0b/dsv4-prefill-bank-lifecycle` | [#2213](https://github.com/JustVugg/colibri/pull/2213) (0b): prefill bank lifecycle, speculative batches, missing-bias guard (3) |
| `rebased/pr0c/dsv4-decode-error-paths` | [#2214](https://github.com/JustVugg/colibri/pull/2214) (0c): mirror slot rollback, loader lanes on every MoE exit (2) |
| `rebased/pr0d/dsv4-dll-export` | [#2215](https://github.com/JustVugg/colibri/pull/2215) (0d): DLL export + parity test (1) |
| `rebased/pr0e/dsv4-cuda-test-target` | [#2216](https://github.com/JustVugg/colibri/pull/2216) (0e): dsv4-cuda-test passes on generic builds (2) |
| `rebased/pr0f/dsv4-docs-sizes` | [#2217](https://github.com/JustVugg/colibri/pull/2217) (0f): real bank/mirror sizes in docs (1) |
| `rebased/pr0/dsv4-fixes` | the six combined (13 commits, #2212–#2217) — base of PR #1 |
| `rebased/pr1/reap-prefill` | PR #1 final (8 commits on pr0) |
| `rebased/pr2/two-gpu-decode` | PR #2 final (8 commits on PR #1) |
| `refresh/pr1-reap-prefill` | **PR #1 on upstream dev 736a0682** (2026-10-19, after PR #0 merged): the same 8 commits directly on dev |
| `refresh/pr2-two-gpu-decode` | **PR #2 on upstream dev 736a0682** (2026-10-19): the same 8 commits on refresh/pr1 |
| `docs5/pr1-reap-prefill` | **PR #1, opened as #2264** (2026-10-20): refresh/pr1 with its docs commit carrying the final5 numbers; upstream dev 7df8c27c merged in on 2026-10-22 |
| `docs5/pr2-two-gpu-decode` | **PR #2 to submit** (2026-10-20): refresh/pr2 on docs5/pr1, its docs commit carrying the final5 numbers; docs5/pr1's dev merge merged in on 2026-10-22 |
| `pr0*/…`, `pr1/reap-prefill`, `pr2/two-gpu-decode`, `pr1/v6`, `pr1/v7`, `pr2/v5`, … | the 2026-10-15/16 stack on 7b55a9b3 (history) |

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
