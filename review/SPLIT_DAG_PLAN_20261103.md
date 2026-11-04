# Restructure plan: 18 small PRs in three review lanes + integration (2026-11-03)

Status: **planned, not executed.** The 8-PR linear stack (#4–#12) is built and
mergeable; this document is the smaller, parallel-restructured replacement to run
in the next free window. Nothing here has been applied yet.

## Why restructure (measured, not a guess)

The linear stack serialises review and moved integration work into feature PRs.
I tested the raw commits against the refreshed `dev` (`6d907b56`) in throwaway
worktrees, no builds:

| replay | result |
|---|---|
| `4293d4c9 → 0d6454f6 → 897ff79e → 99a37137 → 93ec3778` on `dev` | **0 conflicts** |
| `6928922a`, `83850cf9` on the above | **0 conflicts** |
| `14339ca2` on the above | 1 conflict, `docs/deepseek-v4.md` only |
| `0ce6e5dc`, `11c40038` after resolving that doc | **0 conflicts** |
| `split/dsv4-multigpu-layers` rebased off PR0 onto `dev` | **0 conflicts** |
| `split/dsv4-resident`, then `split/dsv4-cuda-kernels` on top | **0 conflicts** |

So the 1766 line and the 2405 line are independent feature lines. The current
chain forced me to resolve their overlap inside PR 4 (per-device cache dispatch
in `peek`/`attach`/`vram_held`, the `upload_failed` growth condition,
`v4_peer_pool()`), which is exactly what an integration PR should own.

## The 18 PRs

Lanes A, B and C are independent and reviewable in parallel. Lane D is the
late-binding integration and polish.

### Lane A — base (2 PRs, both on `dev`)

| # | branch | commits | size | content |
|---:|---|---|---:|---|
| 1 | `split/dsv4-build-fixes` | `aa1b4f52`, `28b35786` | +8/−1 | refused-device error + stale-error clear; GGUF helpers C-only |
| 2 | `split/dsv4-multigpu-layers` | `bfaee5b2` + KV guard from `9bcbc95b` | ~+450 | `DSV4_CUDA_DEVICES`, contiguous layer ranges, per-device mirror caches, KV/dense ownership, KV-owner guard, multigpu/devices tests, placement docs |

### Lane B — #1766 (6 PRs, serial on #2)

| # | branch | base | commits | size | content |
|---:|---|---|---|---:|---|
| 3 | `split/dsv4-resident` | #2 | `460f21cb`, `919c58cd`, auto half of `96d2000e` | ~+420 | skip host expert loads, immutable resident tables, budgeted `auto` |
| 4 | `split/dsv4-device-route` | #3 | routing half of `96d2000e` | ~+320 | `DSV4_CUDA_RESIDENT_ROUTE`, resident route/MoE on one stream, route test |
| 5 | `split/dsv4-kernels-indexer` | #4 | `1c42795b`, `7411d188`, `2c675407` | ~+340 | resident indexer weights, native quant, `f4` table, `DSV4_TEST_DEVICES` |
| 6 | `split/dsv4-kernels-compressor` | #5 | `7019d0af` | ~+520 | compressor projection + indexer prepare on CUDA |
| 7 | `split/dsv4-kernels-wo-decode` | #6 | `371ed4e3` | ~+170 | decode output-projection intermediates stay on the card |
| 8 | `split/dsv4-kernels-head` | #7 | `81c6cb39`, `97a47941`, `35544b2d` | ~+690 | exact head, batched speculative head, validated batch heads by default |

PR 3/4 are the only hunk-level split (one commit, two features). If it proves
fiddly, fall back to a single resident PR (~+740, still cohesive).
Sizes exclude `docs/experiments/` reports, which stay out.

### Lane C — #2405 (6 PRs, serial; #9 on `dev`)

| # | branch | base | commits | size | content |
|---:|---|---|---|---:|---|
| 9 | `split/dsv4-mirror-guard` | `dev` | `99a37137`, `93ec3778`, `4293d4c9` | ~+210/−160 | per-cache growth guard, one LRU clock, keep reserve, drop failed uploads, pin cap |
| 10 | `split/dsv4-vram-first` | #9 | `0d6454f6`, `897ff79e` | ~+640 | `V4_GPU_RESIDENT_FIRST`, `views/fetch/leased`, `V4_SERVE_STATS_LOG` |
| 11 | `split/dsv4-peer-pool` | #10 | `6928922a` | ~+720 | `DSV4_CUDA_PEER_DEVICE`, peer mirrors/reserve, init rollback, peer tests |
| 12 | `split/dsv4-peer-compute` | #11 | `83850cf9` | ~+370 | `DSV4_CUDA_PEER_COMPUTE`, peer-side MoE, latched-peer refusal |
| 13 | `split/dsv4-cuda-fault` | #12 | `14339ca2` | ~+180 | sticky CUDA fault latch, peer fault injection, health checks |
| 14 | `split/dsv4-decode-failures` | #13 | `0ce6e5dc`, `11c40038` | ~+60 | stage-named decode failure; `wo_a` CPU fallback |

### Lane D — integration and polish (4 PRs)

| # | branch | base | commits | size | content |
|---:|---|---|---|---:|---|
| 15 | `split/dsv4-cplus-integration` | `split/integration-base` (temp merge of #8+#14) | merge resolution | ~+200 | mutual exclusion of layer split vs peer pool, per-device cache + peer dispatch, `experts_init` wrapper, strict peer init / best-effort layer split, upload-failed growth condition, docs merge |
| 16 | `split/dsv4-head-followups` | #15 | `07dc142a`, `f4586dca` | ~+85 | `auto` hard-failure handling, `DSV4_CUDA_RESIDENT_RESERVE_MB`, head placement + `DSV4_CUDA_HEAD_DEVICE` |
| 17 | `split/dsv4-test-target` | #16 | `6b2efd31`, `0956307c`, `7ad5319a` | ~+35 | `dsv4-cuda-tests` target, generated-header deps, shared test-device header |
| 18 | `split/dsv4-docs` | #17 | `eaf63b1a` | ~+50 | env rows: resident reserve/experts, head switches, indexer/wo/compressor, routing/batched |

## Graph

```text
dev ── 1 build fixes                      (independent, land first)
dev ── 2 multi-GPU layers ── 3 resident ── 4 device route ── 5 indexer ──
        6 compressor ── 7 wo_decode ── 8 exact head
dev ── 9 mirror guard ── 10 vram-first ── 11 peer pool ── 12 peer compute ──
        13 fault latch ── 14 decode failures

        #8 + #14 ── 15 integration ── 16 head fixes ── 17 test target ── 18 docs
```

- Three parallel review lanes: A (2), B (6), C (6).
- Only #15 is late-binding; its base is a temporary `split/integration-base`
  branch that is the merge of #8 and #14 (the same shape as `64fd75e3`), because
  a GitHub PR has one base but this change needs two branches.
- #16–#18 are small and land after #15.

## What moves into #15

The merge resolutions currently baked into PR 4–#9 of the linear stack:

- per-device cache lookup (`v4_gpu_expert_cache(store, layer)`) in
  `v4_gpu_expert_vram_held`, `coli_v4_gpu_expert_peek`, the tiered attach and
  resident route (instead of casting `store->gpu`);
- the `upload_failed ||` term in the growth-guard condition and the frozen-cache
  comment;
- `v4_peer_pool()` health gating and the peer dispatch in attach/peek;
- `coli_v4_gpu_experts_init()` wrapper and the `DSV4_CUDA_RESIDENT_RESERVE_MB`
  split introduced by the merge;
- mutual exclusion of `DSV4_CUDA_DEVICES` (>1) and `DSV4_CUDA_PEER_DEVICE`;
- verify peer access after best-effort init before creating the pool;
- head-device ordering/override glue and the docs merge of both sides.

## Execution checklist (for the next window)

1. Freeze: do not merge #4–#12; close them as replaced once the new set exists
   (or reuse #4 as-is for #1, since it is already correct).
2. Lane A: keep `split/dsv4-build-fixes`; rebase
   `split/dsv4-multigpu-layers` onto `dev` (measured clean) and fold the KV
   guard hunk from `9bcbc95b` into it.
3. Lane B: recreate #3–#8 from the listed commits (measured clean); hunk-split
   `96d2000e` for #3/#4; delete the old `split/dsv4-cuda-kernels` name.
4. Lane C: recreate #9–#14 from the listed commits on `dev`; resolve the one
   docs conflict at `14339ca2`; keep `docs/experiments/` out.
5. Lane D: create `split/integration-base` = merge #8 + #14; commit the
   integration resolution as #15; then cherry-pick #16, #17, #18.
6. Verify per branch: no conflict markers, `dsv4.def` exports all bound,
   `deepseek_v4.c` function set matches the merged reference, `goto moe_done`
   has its label (the checks used for the linear stack).
7. Open the 18 PRs with the bodies already written, updated bases, and a note
   that #9–#14 are independent of #2–#8.
8. Update `colibri-notes` with the final PR links; keep this document as the
   mapping from old PRs to new ones.

Estimated wall clock: ~3–3.5 h, no builds, no GPU time.

## Risk and rollback

- The old branches stay on `origin` until the new set is verified, so nothing is
  lost; the new set can be deleted and the linear stack kept if the restructure
  stalls.
- Biggest risk is the `96d2000e` split (#3/#4). Fallback: one resident PR.
- Second risk is the merge resolution in #15; the reference trees are
  `64fd75e3` (integration) and `8a30a985` (strict peer), both still local, and
  the current linear branches carry the same resolved code to diff against.

## Count

**18 PRs:** 2 base, 6 for #1766, 6 for #2405, 4 integration/polish. Fifteen of them
can be reviewed concurrently. If 18 feels like too many to shepherd, the cheap
folds are #3+#4 (resident + routing) and #12+#13 (peer compute + fault), giving
**16** without losing a review boundary.
