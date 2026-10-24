# colibri-notes

> **Naming key.** These notes use internal names for the upstream pull requests. **PR #0** is the six
> fixes opened as JustVugg/colibri [#2212](https://github.com/JustVugg/colibri/pull/2212) (0a), [#2213](https://github.com/JustVugg/colibri/pull/2213) (0b), [#2214](https://github.com/JustVugg/colibri/pull/2214) (0c), [#2215](https://github.com/JustVugg/colibri/pull/2215) (0d),
> [#2216](https://github.com/JustVugg/colibri/pull/2216) (0e) and [#2217](https://github.com/JustVugg/colibri/pull/2217) (0f). **PR #1** (batched GPU prefill for expert-pruned checkpoints) is
> [#2264](https://github.com/JustVugg/colibri/pull/2264), opened 2026-10-20; **PR #2** (two-GPU decode) is not opened yet. Documents dated before 2026-10-17 describe earlier
> versions of these changes, on the old base 7b55a9b3.

Project notebook for Bob Summerwill's work on [colibri](https://github.com/JustVugg/colibri) (pure-C/CUDA engine that runs huge MoE models by streaming experts from NVMe and caching hot experts in RAM/VRAM), done with Claude Code on 2026-10-12..14. Not branch-specific; the code lives in the fork [bobsummerwill/colibri](https://github.com/bobsummerwill/colibri).

**Rule: no PRs and no pushes to upstream JustVugg/colibri.** Branches go to the fork only; Bob decides on submission (PR #0 and PR #1 were opened at his request).

## Status (2026-10-24, overnight; log: [OVERNIGHT_20261024.md](OVERNIGHT_20261024.md))
- **#2264 (PR #1):** nothing to rebase (upstream dev still 61646cbd at 05:05; the PR is clean, no new comments). The
  Vulkan + CUDA prompt-2 text is explained (final9/9b/11): the Vulkan dense hook is used identically by dev's union and
  the bank, and the difference is CUDA vs CPU expert sums on a near-tie. A corrected follow-up is drafted, **not
  posted**: [review/REPLY_2264_FOLLOWUP_20261024_DRAFT.md](review/REPLY_2264_FOLLOWUP_20261024_DRAFT.md).
- **Two upstream fixes on fork branches off dev 61646cbd, no issues or PRs** ([review/UPSTREAM_FIXES_20261024.md](review/UPSTREAM_FIXES_20261024.md)):
  **fix A** `fix/dsv4-mirror-guard-overshoot` bd7103fd (the decode mirror cache no longer overshoots its VRAM reserve:
  0 kv comp fallbacks instead of ~420 a sample on the full model, texts identical, prompt-2 TTFT 0.946x on the 3090
  and 0.918x on the 5070); **fix B** `fix/vk-host-image-alloc` 72aa325a (the uploader's host images no longer pile up in
  glibc's arenas: anon -5 GiB, pure Vulkan with the default host pool no longer OOM-killed, prompt-1 TTFT 0.957x).
- **PR #2:** merged with #2264's tip (14d67f05); **F18 fixed** (6628f5b1 latch, 8eedc83b stage names, 693204ed the
  attention's `wo_a` CPU fallback): after a sticky peer fault every later request completes on the host. Retest on the
  new tip (final11/13): all texts = dev's, X3 1.437x dev's decode. Pushed to the fork; still not opened upstream.
- **Optimisation lead:** staged uploads (`COLI_VK_UP_IMPORT=0`) beat host-page imports on this RTX 3090 (6.6 vs 3.5-4.0
  GB/s, prompt-1 TTFT 0.931x on dev); final16 re-measures it on top of fix B.

## Status (2026-10-23, evening)
- **#2264 refreshed (2026-10-23 18:30):** upstream dev 61646cbd merged into `docs5/pr1-reap-prefill` (47c8ec51, no
  conflicts; `make check` 2,709 tests OK, 292 skipped; generic CUDA build on shadowfax clean; GPU tests not re-run,
  dev's 77 new commits leave `c/backend_cuda_dsv4.cu` alone, but change `deepseek_v4.c`'s Windows memory bound and the
  Vulkan tier's host-pool sizing (5c0469ae), so the Vulkan numbers are pre-merge only), pushed; the earlier CI runs on #2264 were all *cancelled*
  (not failed); the fresh run on 47c8ec51 is green (CI + check, 116 of 116 pass, 2026-10-24 03:2x UTC). Reply posted 2026-10-23 ([comment](https://github.com/JustVugg/colibri/pull/2264#issuecomment-6380394089), `review/REPLY_2264_20261024.md`: everything in final6-8
  incl. every kv comp fallback count, sample counts, evidence links, the notes repo up front); it replaces a shorter first
  reply (deleted). PR description updated the same day (merges, 2,709 tests, summary of the full-model results, notes repo). PR #2 not refreshed (it waits for #2264).
- **final7 (10:00-12:57) and final8 (13:05-14:19) done on shadowfax; final8 GATES: PASS** ([measurements.md](measurements.md),
  [investigations.md](investigations.md) §7). Bank like for like: PR #1 0.665 / 0.659x dev's TTFT on the full model with
  dev given the same reserve. Every side-arm difference from final6 is explained and none is PR #1's: pure Vulkan is
  reproducible (dev = PR #1 = CUDA text) once the host pool cannot OOM; Vulkan + CUDA PR #1 = dev with the same reserve
  (final7's VU had kv comp fallbacks); HY is dev's hybrid. computer4 repeats: B/A 0.998 / 0.995.
- **Open:** F18 in a server (PR #2: later requests fail until restart), upstream issue
  drafts ([open-problems.md](open-problems.md)).

## Status (2026-10-23, morning)
- **final6 ran overnight** (shadowfax 17:46-23:56, computer4 DeepGEMM 17:43-19:01; [measurements.md](measurements.md)
  FINAL6). **JustVugg's question is answered:** default configuration, full V4 Flash, PR #1 vs dev: texts byte-identical,
  TTFT 0.983x / 0.989x, decode 1.002x; `DSV4_MM_WARP_VERIFY` 1,500 calls, 0 mismatches; DeepGEMM (RTX 5070): 0.999x /
  1.004x, identical. So `DSV4_MM_WARP` stays on by default. Also: bank prefill 0.67x TTFT (1-sample dev arm), ~16k prompt
  0.69x, PR #2 second GPU +36-48 % decode, soak and speculative decoding pass.
- **The overall gate failed on side arms**, looked into in [investigations.md](investigations.md) §6: dev loses its bank on a
  second prompt (AK; its default reserve is below the bank, as dev documents), the pure-Vulkan QB was a host OOM kill (upstream pool sizing), Vulkan texts differ (probably run-time CPU/GPU
  split), HY differs (probably dev's hybrid path), and F18's fault leaves CUDA unusable for the process (PR #2's message
  is wrong). Follow-ups: **[TEST_PLAN_20261023.md](TEST_PLAN_20261023.md)** (final7).
- **Pushed 2026-10-23:** `docs5/pr1-reap-prefill` b286167e (two unused variables in the DeepGEMM kernels, upstream code;
  found by the first DeepGEMM build) and `docs5/pr2-two-gpu-decode` 0523e8e0 (its merge), after `make check` at both
  tips (2,688 / 2,695 OK, 292 skipped). PR #2 stays on the fork (Bob: PR #1 first; PR #2 is rebased, retested and
  opened only after #2264 merges).
- **final6 manifests done** (`manifests/final6/`, `manifests/final6-c4ab/`, all valid). **final7 running** on shadowfax
  since 10:00 ([TEST_PLAN_20261023.md](TEST_PLAN_20261023.md)).
- **Not done yet:** the reply on #2264 (draft first, after final7).

## Status (2026-10-22)
- **PR #1 is [#2264](https://github.com/JustVugg/colibri/pull/2264)** (opened 2026-10-20). JustVugg's review (2026-10-22, [comment](https://github.com/JustVugg/colibri/pull/2264#issuecomment-6335156265)) left it out of the day's batch (#2284) for three points: (1) merge dev (a one-line conflict in `coli_v4_gpu_engine_open`); (2) `DSV4_MM_WARP` is on by default (auto), so measure the **default configuration (no `COLI_CUDA_MOE_BATCH`) on the full V4 Flash**, dev against the branch, prompt and decode, interleaved; it stays on by default if bit-identical and not slower; (3) say in the description why `test_v4_moe_batch_speculative_source.py` is deleted (its checks moved into `test_v4_moe_batch_lifecycle_source.py`).
- **Done 2026-10-22:** dev 7df8c27c merged into `docs5/pr1-reap-prefill` and, through it, `docs5/pr2-two-gpu-decode` (merge commits, no force-push; PR #2 had two more conflicts with dev's CUDA expert tier `cu_tier.c`, resolved keeping both sides, see [decisions.md](decisions.md)); the PR description says why the test file went; a reply on #2264 says the full-model measurement is coming.
- **Test plan and rationale: [TEST_PLAN_20261022.md](TEST_PLAN_20261022.md)** (the requested A/B plus every open item this hardware can test; shadowfax overnight, DeepGEMM on computer4, the F18 fault probe).
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
| `docs5/pr1-reap-prefill` | **PR #1, opened as #2264** (2026-10-20): refresh/pr1 with its docs commit carrying the final5 numbers; upstream dev 7df8c27c merged in on 2026-10-22 (ebe82123); b286167e (DeepGEMM unused variables) on 2026-10-22, pushed 2026-10-23; dev 61646cbd merged in 2026-10-23 (47c8ec51), pushed |
| `docs5/pr2-two-gpu-decode` | **PR #2 to submit** (2026-10-20): refresh/pr2 on docs5/pr1, its docs commit carrying the final5 numbers; docs5/pr1's dev merge merged in on 2026-10-22 (073f18d2), then b286167e (0523e8e0); docs5/pr1 47c8ec51 merged in 2026-10-24 (14d67f05), then the F18 fix 6628f5b1 + 8eedc83b + 693204ed, pushed 2026-10-24 |
| `fix/dsv4-mirror-guard-overshoot` | **upstream fix A** (2026-10-24): bd7103fd on dev 61646cbd, the decode mirror cache guard (no issue / PR) |
| `fix/vk-host-image-alloc` | **upstream fix B** (2026-10-24): 72aa325a on dev 61646cbd, the Vulkan uploader's host images (no issue / PR) |
| `f18-probe` (local only, never pushed) | b62d1068: PR #2 + the F18 probe, debug build only (`-DCOLI_F18_PROBE`) |
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
