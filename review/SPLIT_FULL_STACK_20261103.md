# Full-stack split: #1766 + head-followups + #2405/C+ (2026-11-03)

Combined scope: **37 files, +5,280/−378** against the #1766 dev base
(`3b953308`). This is the plan for splitting **everything** into reviewable
PRs inside `bobsummerwill/colibri`.

## 9 PRs

| # | branch | base | content |
|---:|---|---|---|
| 0 | `split/dsv4-build-fixes` | `dev` | refused-device error, table `f4`, `DSV4_TEST_DEVICES`, GGUF C-only |
| S | `split/dsv4-shared-expert` | `dev` | shared-expert up/down binding fix (already #1812) |
| 1 | `split/dsv4-cuda-kernels` | `dev` | exact head, batched head, compressor/indexer, `wo_decode`, resident indexer, quant |
| 2 | `split/dsv4-multigpu-layers` | `dev` | device-list parsing, contiguous layer placement, per-device caches, KV/dense ownership, multigpu test |
| 3 | `split/dsv4-resident` | integration of 1+2 | preload immutable tables, skip host loads, budgeted `auto`, resident route/MoE, auto/reserve/head fixes |
| 4 | `split/dsv4-mirror-vram-first` | 2 | per-cache mirror guard/reserve/rollback, VRAM-first, pin cap, stats log |
| 5 | `split/dsv4-peer-pool` | 4 | `DSV4_CUDA_PEER_DEVICE`, peer compute, peer mirrors/reserve, peer tests |
| 6 | `split/dsv4-cuda-fault` | 5 | sticky CUDA fault latch, stage-named decode failure, F18 `wo_a` fallback |
| 7 | `split/dsv4-cplus-integration` | integration of 3+5+6 | mutual exclusion of layer split vs peer pool, head-on-primary, strict peer init / best-effort layer split |

## Dependencies

```text
dev
├─ 0 build fixes
├─ S shared-expert
├─ 1 #1766 kernels ─┐
├─ 2 #1766 layers ──┴─► 3 resident ─┐
├─ 4 mirror + VRAM-first ─► 5 peer ─┼─► 7 C+ integration
└─ 6 fault/F18 ─────────────────────┘
```

- 0, S, 1, 2 are independent and base on `dev`.
- 3 bases on a temporary integration branch of 1+2.
- 4 bases on 2 so it reuses the per-device cache structure; it can be rebased
  to `dev` later at the cost of resolving the cache hunks again.
- 5 bases on 4.
- 6 bases on 5 (the F18 half is independent; it can be split out if desired).
- 7 bases on an integration branch of 3+5+6.

## Commit/source map

- 0: `aa1b4f52`, `7411d188`, `2c675407`, `28b35786`.
- S: `8c094e52`.
- 1: `1c42795b`, `7019d0af`, `371ed4e3`, `81c6cb39`, `97a47941`, `35544b2d`.
- 2: `bfaee5b2`; add the KV-owner guard and layer-split best-effort P2P from
  `9bcbc95b`.
- 3: `919c58cd`, `460f21cb`, `96d2000e`; add `07dc142a` auto/reserve/head and
  `f4586dca` head override.
- 4: `99a37137`, `93ec3778`, `897ff79e`, `0d6454f6`, `4293d4c9`.
- 5: `6928922a`, `83850cf9`.
- 6: `14339ca2`, `0ce6e5dc`, `11c40038`.
- 7: the C+ merge resolution (`64fd75e3`) and strict peer fix (`8a30a985`).

## Conflict hotspots

`c/deepseek_v4.c` (`coli_v4_gpu_engine_open`, mirror-cache struct,
attach/peek/guard, head), `c/backend_cuda_dsv4.cu` (init, KV, head, quant),
`c/Makefile.deepseek-v4`, `c/backend_loader_dsv4.c`, `c/dsv4.def`,
`docs/deepseek-v4.md`.

The source commits were written on older dev; direct cherry-picks conflict in
the same five files. The split is hunk-level extraction from the resolved
`f4586dca`/`8a30a985` trees, not a raw replay.

## Testing

No build/test/profiling is part of the split pass. Each PR body states that the
code is split-only and names the tests that will be run after the split is
reviewed: CUDA unit tests, 3090 placement/peer/resident A/B, 5070 sm_120, and
the built-in/nsys profiling round.
