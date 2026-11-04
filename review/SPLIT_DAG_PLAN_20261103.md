# Restructure plan: 7 PRs in two review lanes + integration (2026-11-03)

Status: **planned, not executed.** Supersedes the 18-PR granularity in the
previous draft of this file. The 8-PR linear stack (#4–#12) is still open and
mergeable; this is the replacement to run in the next free window.

## How many commits are we actually splitting?

Measured against the refreshed `dev` (`6d907b56`) history:

| stream | commits | of which code/test |
|---|---:|---:|
| #1766 branch (`3b953308..aabd99d4`) | 26 (20 non-merge) | **14** — the other 6 are experiment reports, a docs-bench commit and two "samebase" harness commits, all dropped |
| your fixes (`aabd99d4..f4586dca`) | **7** | 7 |
| #2405 (`c1c06ac6..11c40038`) | **11** | 10 — one is a docs commit, folded into the docs PR |
| **relevant total** | **31** | plus the one merge resolution (`64fd75e3`) and one repair (`73c05be2`) |

So: **31 source commits + 1 integration merge**. That is the thing being split —
not 18 things. A PR can carry several commits; the review cost tracks the diff,
not the commit count.

## Recommended: 7 PRs

Two independent lanes plus integration. Sizes exclude `docs/experiments/`.

| # | branch | base | commits | size | content |
|---:|---|---|---|---:|---|
| 1 | `split/dsv4-build-fixes` | `dev` | `aa1b4f52`, `28b35786` | +8/−1 | refused-device error, GGUF helpers C-only |
| 2 | `split/dsv4-multigpu-resident` | #1 | `bfaee5b2`, `460f21cb`, `919c58cd`, `96d2000e` | ~+1,150 | `DSV4_CUDA_DEVICES` + contiguous layers + per-device caches + KV owner; immutable resident tables; budgeted `auto`; device routing |
| 3 | `split/dsv4-cuda-kernels` | #2 | `1c42795b`, `7411d188`, `7019d0af`, `371ed4e3`, `81c6cb39`, `97a47941`, `35544b2d`, `2c675407` | ~+1,700 | resident indexer + quant, `f4`, test devices; compressor/indexer prepare; `wo_decode`; exact head (single + batched + default) |
| 4 | `split/dsv4-mirror-vram-first` | `dev` | `4293d4c9`, `0d6454f6`, `897ff79e`, `99a37137`, `93ec3778` | ~+850 | pin cap; serve stats; VRAM-first; per-cache guard; reserve/rollback |
| 5 | `split/dsv4-peer-fault` | #4 | `6928922a`, `83850cf9`, `14339ca2`, `0ce6e5dc`, `11c40038` | ~+1,450 | peer pool; peer compute; sticky fault latch; stage-named failures; `wo_a` fallback |
| 6 | `split/dsv4-cplus-integration` | `split/integration-base` = merge of #3 + #5 | merge resolution | ~+200 | mutual exclusion, per-device cache + peer dispatch, `experts_init`, strict peer init / best-effort split, `upload_failed` guard, docs merge |
| 7 | `split/dsv4-followups` | #6 | `9bcbc95b` (KV guard half only), `07dc142a`, `f4586dca`, `6b2efd31`, `0956307c`, `7ad5319a`, `eaf63b1a` | ~+170 | KV-owner guard, `auto` hard-failure handling, resident reserve split, head placement + `DSV4_CUDA_HEAD_DEVICE`, test target + header deps, docs |

Graph:

```text
dev ── 1 build fixes
dev ── 2 multi-GPU + resident ── 3 CUDA kernels
dev ── 4 mirror + VRAM-first ── 5 peer + fault
                 #3 + #5 ── 6 integration ── 7 followups
```

- **#2,#3 and #4,#5 are two independent lanes**: #4 does not depend on #1–#3.
- **#6 is the only cross-lane PR**; its base is a temporary branch that merges
  #3 and #5 (a GitHub PR has one base but this change needs two branches).
- **#7 is small** and lands after #6.

### Answering "does D depend on all prior PRs?"

Directly, no: the integration PR depends on **two branches only** — the tip of
the 1766 lane (#3) and the tip of the 2405 lane (#5). Transitively it contains
their commits (that is why it can only be reviewed once those lanes are stable),
and it does not depend on #1 at all. The followups (#7) depend only on #6.

## Ladder, if 7 is still too coarse or too fine

Same commits, three groupings; only the branch boundaries change.

| grouping | PRs | when |
|---|---:|---|
| **7** | 1 + 4 + 1 | recommended: smallest count with a real review boundary between placement/resident, kernels, mirror/VRAM, peer/fault, integration |
| **9** | 1 + 5 + 1 + 2 | split kernels into "head" and "rest"; split peer from fault |
| 18 | 1 + 6 + 6 + 4 | too fine: serialises review, adds base juggling, no extra safety |

The earlier 18-PR draft is kept in git history but should not be used.

## Measured rebase results (no builds)

Tested in throwaway worktrees against refreshed `dev`:

| replay | result |
|---|---|
| `4293d4c9 → 0d6454f6 → 897ff79e → 99a37137 → 93ec3778` | 0 conflicts |
| `6928922a`, `83850cf9` on top | 0 conflicts |
| `14339ca2` on top | 1 docs conflict |
| `0ce6e5dc`, `11c40038` after that | 0 conflicts |
| 1766 line rebased off PR0 onto `dev` | 0 conflicts |

## What moves into the integration PR (#6)

Currently baked into the linear stack's PR 4–#9, and belonging to the merge:

- per-device cache lookup in `v4_gpu_expert_vram_held`, `peek`, tiered attach and
  resident route;
- `upload_failed ||` in the growth-guard condition and the frozen-cache comment;
- `v4_peer_pool()` health gating and peer dispatch in attach/peek;
- `coli_v4_gpu_experts_init()` and the `DSV4_CUDA_RESIDENT_RESERVE_MB` split;
- mutual exclusion of `DSV4_CUDA_DEVICES` (>1) and `DSV4_CUDA_PEER_DEVICE`;
- verify peer access after best-effort init before creating the pool;
- head-device ordering glue and the docs merge of both sides.

## Execution checklist (next window)

1. Freeze #4–#12; keep them on `origin` until the new set is verified, then close
   as replaced (reuse #4 as-is for new #1).
2. New #1: unchanged (`split/dsv4-build-fixes`).
3. New #2: branch `split/dsv4-multigpu-resident` from #1; replay `bfaee5b2`
   (measured clean off PR0), then `460f21cb`, `919c58cd`, `96d2000e` (measured
   clean as a lane). Fold the KV-owner guard hunk from `9bcbc95b` here.
4. New #3: `split/dsv4-cuda-kernels` on #2; replay the eight kernel commits.
5. New #4: `split/dsv4-mirror-vram-first` from `dev`; replay the five 2405
   commits in author order (clean; keep `docs/experiments/` out).
6. New #5: `split/dsv4-peer-fault` on #4; replay peer pool, peer compute, fault,
   failures (one docs conflict at `14339ca2`).
7. New #6: create `split/integration-base` = merge of #3 and #5; commit the
   resolution; new #7 on top with the seven followup commits.
8. Verify each tip: no conflict markers, `dsv4.def` exports all bound,
   `deepseek_v4.c` function set matches `8a30a985`, `goto moe_done` has its label.
9. Open the 7 PRs, update the notes with links, close the old set.

Estimated wall clock: ~2.5–3 h, no builds, no GPU time.

## Rollback

Old branches stay on `origin`; if the restructure stalls, the linear stack is
still intact and mergeable. The reference trees (`64fd75e3`, `8a30a985`) and the
current linear branches carry the resolved code, so the integration resolution
can be diffed rather than re-derived.
