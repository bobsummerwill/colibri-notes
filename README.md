# colibri-notes

> **Naming key.** These notes use internal names for the upstream pull requests. **PR #0** is the six
> fixes opened as JustVugg/colibri [#2212](https://github.com/JustVugg/colibri/pull/2212) (0a), [#2213](https://github.com/JustVugg/colibri/pull/2213) (0b), [#2214](https://github.com/JustVugg/colibri/pull/2214) (0c), [#2215](https://github.com/JustVugg/colibri/pull/2215) (0d),
> [#2216](https://github.com/JustVugg/colibri/pull/2216) (0e) and [#2217](https://github.com/JustVugg/colibri/pull/2217) (0f). **PR #1** (batched GPU prefill for expert-pruned checkpoints) is
> [#2264](https://github.com/JustVugg/colibri/pull/2264), opened 2026-10-20, **merged 2026-10-24** in JustVugg's batch [#2360](https://github.com/JustVugg/colibri/pull/2360) (merge commit 60a73ec; dev 561966e).
> **PR #2** (two-GPU decode) is [#2405](https://github.com/JustVugg/colibri/pull/2405), opened as a draft 2026-10-26, **marked ready 2026-10-29**. Documents dated before 2026-10-17 describe earlier
> versions of these changes, on the old base 7b55a9b3.

Project notebook for Bob Summerwill's work on [colibri](https://github.com/JustVugg/colibri) (pure-C/CUDA engine that runs huge MoE models by streaming experts from NVMe and caching hot experts in RAM/VRAM), done with Claude Code on 2026-10-12..14. Not branch-specific; the code lives in the fork [bobsummerwill/colibri](https://github.com/bobsummerwill/colibri).

**Rule: nothing goes to upstream JustVugg/colibri unless Bob asks for that specific submission.** Branches go to the
fork (`--force-with-lease`); Bob decides what is opened and when, and comments on other people's upstream PRs need his
OK too. Opened at his request: PR #0 (#2212-#2217), PR #1 (#2264, merged); 2026-10-25: issues #2399 (fix A) and
#2400 (fix B), PRs #2401 (fix A), #2402 (fix B), #2403 (test_vk_chain initializers), #2404 (`COLI_VK_SUBGROUP=n`);
2026-10-26: draft PR #2405 (PR #2, on #2401); 2026-10-29: #2405 marked ready, with a comment.

> **Pick up here: [HANDOFF_20261026.md](HANDOFF_20261026.md)**, top section "2026-10-29" (#2405 merged with dev d68e16b6,
> re-checked, re-benchmarked (final36: same 1.03x/1.44x/1.02x/1.27x), **marked ready** with a merge-order comment).
> Earlier (2026-10-26 evening: the same-base runs of #1766,
> PR #2 and C+ (#1766 + PR #2's peer tier) are done. C+ is fastest: 2.01x dev on REAP and 1.62x on the full model, with
> two 3090s. The #1766 comment is posted and #2405's description is rewritten (22:51, Bob's OK); #2405 is draft, CI green.
> C+ with the `f4` fix (e2933955) is being measured, and both texts get its numbers after.)
> Plan and results: [review/PLAN_SAME_BASE_20261026.md](review/PLAN_SAME_BASE_20261026.md). Before that:
> [HANDOFF_20261025.md](HANDOFF_20261025.md), [HANDOFF_20261024.md](HANDOFF_20261024.md).

## Status (2026-10-29, early morning; computer4 + shadowfax)
- **#2405 (PR #2) is ready for review.** `docs5/pr2-two-gpu-decode` = **f1e56b00**: upstream dev d68e16b6 merged into
  11c40038, **no conflicts**. Pushed to the fork (fast-forward). Description rewritten for the new base:
  [review/POSTED_2405_BODY_20261029.md](review/POSTED_2405_BODY_20261029.md); comment:
  [review/POSTED_2405_COMMENT_20261029.md](review/POSTED_2405_COMMENT_20261029.md).
- **Checks on f1e56b00 (computer4, gcc 13.3.0):** `make check` 2,795 OK (305 skipped); `-Wall -Wextra` on `deepseek_v4.c`
  (GPU tier, and `-DCOLI_VULKAN`) and `test_deepseek_v4.c`: 0 diagnostics. shadowfax CUDA build (sm_86, NO_TC): 0 warnings.
- **final36 (shadowfax, 2x 3090, power limits now 500/450 W, final27-34 ran at 300 W), GATES: PASS, all texts identical:**
  REAP P1/B1 1.028x, P2/B1 1.436x (4.329 vs 3.016); full fdec DP1F 1.022x, DP2F 1.268x (3.348 vs 2.642); long-prompt
  TTFT 1.00x; S7 sticky fault: all 6 answers complete; `dsv4-cuda-test` and `cuda-test` rc=0; tabby restored healthy.
  Same as the c1c06ac6 runs within 0.3%, so the higher power limit changes nothing here.
  [manifests/shadowfax-20261028/final36](manifests/shadowfax-20261028/final36) (bench.tgz on shadowfax,
  sha256 463e29ae...).
- **Merge order vs #1766:** independent (no shared commits); textual conflicts in 5 files (`c/Makefile`,
  `c/backend_cuda_dsv4.cu`, `c/deepseek_v4.c`, `c/deepseek_v4_internal.h`, `docs/deepseek-v4.md`) for whichever lands
  second. #1766 (ca55947c) already conflicts with dev. Resolution reference: `samebase/cplus` 66982895 (on c1c06ac6).
  Bob's 2026-10-28 #1766 comment ([6462594190](https://github.com/JustVugg/colibri/pull/1766#issuecomment-6462594190))
  offers the cherry-pick or a combined series.
- shadowfax's `llama-server-active` now runs TabbyAPI; the harness stops and restores it unchanged.

## Status (2026-10-24, night; counter)
- **All three branches are on dev 561966e**, done on counter (`~/Projects/colibri`) and
  **pushed to the fork** (not upstream; nothing opened):
  - **PR #2** `docs5/pr2-two-gpu-decode` 090d8160: dev merged in, **no conflicts**. dev 61646cbd..561966e changes no
    V4/dsv4 GPU code outside #2264 (its other GPU changes are Qwen3.6's CUDA chain, and the Makefile only gains a
    Qwen3.6 test target), so final11/13 still describe the tree. `make check` 2,741 OK (312 skipped). Per commit: all
    11 + the merge build, `deepseek_v4.c` (GPU tier, and with `-DCOLI_VULKAN`) and `test_deepseek_v4.c` compile with
    `-Wall -Wextra`, 0 diagnostics ([manifests/counter-20261024/pr2-percommit.txt](manifests/counter-20261024/pr2-percommit.txt)).
    **Bisect nit:** `test_v4_moe_batch_books_source.py` (added in fb0332a3) needs the second `v4_views_fetch_host`
    that b8295605 adds, so it fails at the 4 commits in between and passes at every tip; fixing that needs a
    force-push.
  - **Fix A** f30c7b23: dev merged in **with a conflict**. #2264 replaced the fixed 2800/600 MiB reserve with
    `v4_gpu_mirror_reserve_mb()`, so the margin is now `v4_gpu_mirror_reserve_margin_mb()`: the 64 mirrors the
    bank-sized default carries, 0 for an explicit reserve or the 600 MiB headroom. Source test updated. `make check`
    2,733 OK (312 skipped).
  - **Fix B** 97b26c0a: dev merged in, clean (dev doesn't touch the Vulkan uploader). `make check` 2,732 OK (312 skipped).
- **Fix A vs PR #2:** PR #2's 6b7fc6c0 already *contains* fix A (per-cache form). The trial merge (local
  `scratch/fixA-x-pr2` c89353e4) keeps PR #2's guard and docs and ports fix A's test; `make check` 2,745 OK.
  Recommended order: fix A first. See [review/UPSTREAM_FIX_PRS_20261024_DRAFT.md](review/UPSTREAM_FIX_PRS_20261024_DRAFT.md).
- **Drafts for Bob:** [review/PR2_DESCRIPTION_20261024_DRAFT.md](review/PR2_DESCRIPTION_20261024_DRAFT.md) (final6/11/13
  numbers, the F18 fix, the "continues on device 0" correction) and
  [review/UPSTREAM_FIX_PRS_20261024_DRAFT.md](review/UPSTREAM_FIX_PRS_20261024_DRAFT.md) (fixes A and B; fix A absorbs the
  2026-10-23 kv comp issue draft).
- **Not done on counter:** CUDA/DeepGEMM builds (no nvcc) and the GPU re-check (shadowfax); triage says the re-check
  isn't needed before opening PR #2.

## Status (2026-10-24, evening)
- **#2264 (PR #1) merged** into dev through batch [#2360](https://github.com/JustVugg/colibri/pull/2360) (merge commits
  60a73ec for #2264, 561966e for the batch; 116 checks passed). A real merge, not a squash: the branch's commits,
  including 47c8ec51, are now ancestors of dev. JustVugg's own check was V4 Flash UD-IQ1_S on an RTX PRO 4000, 2 rounds:
  Vulkan and CUDA decode equal to dev, prompt the same.
- The follow-up comment ([comment](https://github.com/JustVugg/colibri/pull/2264#issuecomment-6387132721)) was posted
  before the merge; JustVugg did not answer its offer of fixes A and B.
- **shadowfax and computer4 unreachable.** Only counter is available for building and testing. See
  [open-problems.md](open-problems.md) for what that allows.
- **Next:** PR #2 (merge dev 561966e, re-check, refresh the description, open at Bob's decision), then fixes A and B.

## Status (2026-10-24, overnight; log: [OVERNIGHT_20261024.md](OVERNIGHT_20261024.md))
- **#2264 (PR #1):** nothing to rebase (upstream dev still 61646cbd at 05:05; the PR is clean, no new comments). The
  Vulkan + CUDA prompt-2 text is explained (final9/9b/11): the Vulkan dense hook is used identically by dev's union and
  the bank, and the difference is CUDA vs CPU expert sums on a near-tie. A corrected follow-up was drafted and then
  posted 2026-10-24 ([comment](https://github.com/JustVugg/colibri/pull/2264#issuecomment-6387132721)).
- **Two upstream fixes on fork branches off dev 61646cbd, no issues or PRs** ([review/UPSTREAM_FIXES_20261024.md](review/UPSTREAM_FIXES_20261024.md)):
  **fix A** `fix/dsv4-mirror-guard-overshoot` bd7103fd (the decode mirror cache no longer overshoots its VRAM reserve:
  0 kv comp fallbacks instead of ~420 a sample on the full model, texts identical, prompt-2 TTFT 0.946x on the 3090
  and 0.918x on the 5070); **fix B** `fix/vk-host-image-alloc` 72aa325a (the uploader's host images no longer pile up in
  glibc's arenas: anon -5 GiB, pure Vulkan with the default host pool no longer OOM-killed, prompt-1 TTFT 0.957x).
- **PR #2:** merged with #2264's tip (14d67f05); **F18 fixed** (6628f5b1 latch, 8eedc83b stage names, 693204ed the
  attention's `wo_a` CPU fallback): after a sticky peer fault every later request completes on the host. Retest on the
  new tip (final11/13): all texts = dev's, X3 1.437x dev's decode. Pushed to the fork; still not opened upstream.
- **Optimisation lead:** staged uploads (`COLI_VK_UP_IMPORT=0`) beat host-page imports on this RTX 3090 (6.6 vs 3.5-4.0
  GB/s, prompt-1 TTFT 0.931x on dev; on top of fix B 0.980x and decode +2.6%, final16). Device dependent: a note, not a fix.

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
| `docs5/pr1-reap-prefill` | **PR #1, merged upstream 2026-10-24** as [#2264](https://github.com/JustVugg/colibri/pull/2264) (60a73ec, in batch #2360 / dev 561966e). History: refresh/pr1 with its docs commit carrying the final5 numbers; dev 7df8c27c merged in 2026-10-22 (ebe82123); b286167e (DeepGEMM unused variables); dev 61646cbd merged in 2026-10-23 (47c8ec51). No further work. |
| `docs5/pr2-two-gpu-decode` | **PR #2 to submit**, at **3ce988da**: 11 linear commits on dev 561966e (rewritten from 090d8160 and force-pushed 2026-10-24, Bob's OK; every commit passes its tests). See [HANDOFF_20261024.md](HANDOFF_20261024.md) for the old→new hashes. |
| `fix/dsv4-mirror-guard-overshoot` | **upstream fix A** (2026-10-24): bd7103fd on dev 61646cbd, the decode mirror cache guard (no issue / PR); dev 561966e merged in on counter (f30c7b23, conflict resolved, pushed) |
| `fix/vk-host-image-alloc` | **upstream fix B** (2026-10-24): 72aa325a on dev 61646cbd, the Vulkan uploader's host images (no issue / PR); dev 561966e merged in on counter (97b26c0a, clean, pushed) |
| `rewrite/pr2-linear` | 3ce988da, the same commit as `docs5/pr2-two-gpu-decode` now; can be deleted |
| `split/pr2-1-guard` → `split/pr2-2-settings` → `split/pr2-3-vram-first` → `split/pr2-4-two-gpu` | **trial split of PR #2** (2026-10-24 night): a 4-part stack on dev 561966e, every part `make check` OK, part 4's tree = 3ce988da; see [review/PR2_SPLIT_OPTIONS_20261024.md](review/PR2_SPLIT_OPTIONS_20261024.md) |
| `fix/vk-subgroup-size-env` | d7879d19 on dev 561966e: `COLI_VK_SUBGROUP=n` pins the chain at n (ANV's > 38 min SIMD32 compile of `chain_moe.spv`; see [review/VK_CHAIN_HANG_20261024.md](review/VK_CHAIN_HANG_20261024.md)) |
| `fix/vk-chain-test-initializers` | 597b49a4 on dev 561966e: the 13 `-Wmissing-field-initializers` in `tests/test_vk_chain.c`; `test_vk_chain` passes on Lavapipe (pushed 2026-10-24 night) |
| `scratch/fixA-x-pr2` (counter, local only) | c89353e4: trial merge of fix A into PR #2 (item 4 of open-problems); not for pushing |
| `f18-probe` (local only, never pushed) | b62d1068: PR #2 + the F18 probe, debug build only (`-DCOLI_F18_PROBE`) |
| `pr0*/…`, `pr1/reap-prefill`, `pr2/two-gpu-decode`, `pr1/v6`, `pr1/v7`, `pr2/v5`, … | the 2026-10-15/16 stack on 7b55a9b3 (history) |

The `rebased/*`, `refresh/*` and 7b55a9b3-era branches are history only. Bob gave the go-ahead to delete them (2026-10-24 night), but the bulk delete was blocked in-session; the list, tips and the command are in [HANDOFF_20261024.md](HANDOFF_20261024.md).

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
