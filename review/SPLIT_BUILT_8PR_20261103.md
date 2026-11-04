> **Superseded for the next window:** this 8-PR linear stack is to be replaced
> by the 18-PR, three-lane structure in
> [SPLIT_DAG_PLAN_20261103.md](SPLIT_DAG_PLAN_20261103.md). Keep this document
> as the record of what was built and why the linear order was chosen.

# Full-stack split, as built: #1766 + head-followups + #2405/C+ (2026-11-03)

This supersedes the 9-PR plan in `SPLIT_FULL_STACK_20261103.md`. The split was
built and is open as **8 PRs in a linear stack** in `bobsummerwill/colibri`,
all based on a refreshed `dev` (`6d907b56`, from `JustVugg/colibri:dev`).

No build, test or profiling was run in the split pass; every PR body states that
and carries its deferred test plan.

## The stack

| # | PR | branch | base | content |
|---:|---:|---|---|---|
| 0 | [#4](https://github.com/bobsummerwill/colibri/pull/4) | `split/dsv4-build-fixes` | `dev` | device-refusal fix, GGUF C-only helpers |
| 1 | [#5](https://github.com/bobsummerwill/colibri/pull/5) | `split/dsv4-multigpu-layers` | #4 | `DSV4_CUDA_DEVICES`, contiguous layer ranges, per-device mirror caches, KV owner |
| 2 | [#6](https://github.com/bobsummerwill/colibri/pull/6) | `split/dsv4-resident` | #5 | immutable resident expert tables, skip host loads, budgeted `auto`, device routing |
| 3 | [#7](https://github.com/bobsummerwill/colibri/pull/7) | `split/dsv4-cuda-kernels` | #6 | resident indexer + quant, compressor/indexer prepare, `wo_decode`, exact head (single + batched), `f4` table, `DSV4_TEST_DEVICES` |
| 4 | [#8](https://github.com/bobsummerwill/colibri/pull/8) | `split/dsv4-mirror-vram-first` | #7 | per-cache growth guard, reserve/rollback, VRAM-first decode, serve stats, pin cap |
| 5 | [#9](https://github.com/bobsummerwill/colibri/pull/9) | `split/dsv4-peer-pool` | #8 | `DSV4_CUDA_PEER_DEVICE`, peer mirrors/reserve, peer compute |
| 6 | [#10](https://github.com/bobsummerwill/colibri/pull/10) | `split/dsv4-cuda-fault` | #9 | sticky CUDA fault latch, stage-named failures, `wo_a` fallback, docs catch-up |
| 7 | [#11](https://github.com/bobsummerwill/colibri/pull/11) | `split/dsv4-cplus-integration` | #10 | strict peer init, best-effort layer-split init, test mock |
| 8 | [#12](https://github.com/bobsummerwill/colibri/pull/12) | `split/dsv4-head-followups` | #11 | `auto` hard-failure handling, `DSV4_CUDA_RESIDENT_RESERVE_MB`, head placement + `DSV4_CUDA_HEAD_DEVICE`, KV owner guard, test target, docs |

```
dev
└─ 0 build fixes
   └─ 1 multi-GPU layers
      └─ 2 resident
         └─ 3 kernels
            └─ 4 mirror + VRAM-first
               └─ 5 peer pool
                  └─ 6 fault/F18
                     └─ 7 C+ integration
                        └─ 8 head-followups fixes
```

## Why linear, not the 9-PR DAG

The author's commits are children of one another, so the split had to follow the
parent order:

`bfaee5b2 → 460f21cb → 919c58cd → 96d2000e → 1c42795b → 7019d0af → 371ed4e3 →
81c6cb39 → 97a47941 → 35544b2d`

and the #2405 layers in theirs:

`4293d4c9 → 0d6454f6 → 897ff79e → 99a37137 → 93ec3778 → 6928922a → 83850cf9 →
14339ca2 → 0ce6e5dc → 11c40038`

The first attempt (kernels → resident) failed because the kernel commits modify
resident-path code (`dsv4_cuda_head_batch`, resident matmul anchors). Multi-GPU
placement must also come first: the resident table is per owning device.

## What changed from the 9-PR plan

- **Shared-expert PR dropped**: `8c094e52` is already in refreshed `dev`; the
  cherry-pick was empty.
- **Build-fixes PR trimmed**: `7411d188` (`f4` table) and `2c675407`
  (`DSV4_TEST_DEVICES`) only touch files the kernels PR creates, so they live in
  PR 3. PR 0 has `aa1b4f52` and `28b35786`.
- **PR 4's author order corrected**: the stats commit (`0d6454f6`) precedes
  VRAM-first (`897ff79e`), which precedes the per-cache guard (`99a37137`).
  Applying them out of order re-adds already-present hunks; only the missing
  `g_v4_dec_counting` scope lines were taken from the stats commit.
- **PR 4 gained `73c05be2`**: the VRAM-first merge rebuilt `moe_token_pipeline`
  from the #2405 side, which does not know the resident routing hop, and dropped
  the `moe_done` label the resident MoE block jumps to. The commit restores both
  and adds `upload_failed` to the growth-guard condition.
- **PR 6 carries the docs catch-up**: the author's fault commit also documents
  the peer pool, VRAM-first, pin cap and the fault test knob.
- **PR 8 exists** because the seven head-followups commits were written against
  the merged C+ tree (`07dc142a` assumes `coli_v4_gpu_experts_init`, added by the
  merge), so they form their own fixes PR at the top instead of being
  distributed.

## Differences kept from the author's branch

- Everything stays on **per-device caches**: `v4_gpu_expert_cache(store, layer)`
  replaces casts of `store->gpu` in `vram_held`, `peek`, `attach` and the
  resident route.
- Mutual exclusion of `DSV4_CUDA_DEVICES` (>1) and `DSV4_CUDA_PEER_DEVICE` is in
  PR 5 (needed for correctness), not PR 7.
- `dsv4_cuda_init` keeps the PR 0 up-front validation (refused ordinal,
  duplicates, no stale error) on top of every later rollback/peer change.
- `dsv4_cuda_shutdown` frees the per-layer KV rings before `dev_release`.
- `docs/experiments/` reports stay out of the split, as in the author's final
  branch; docs keep the links.

## Validated by inspection only

The split pass did not build. What was checked:

- No conflict markers in `c/` or `docs/` on any branch tip.
- `c/dsv4.def` symbols all exist in `c/backend_cuda_dsv4.cu` and are bound in
  `c/backend_loader_dsv4.c` (100 exports).
- Function sets of `c/deepseek_v4.c` match the merged reference (`8a30a985`);
  the remaining text differences are a comment, a blank line and statement
  order.
- `goto moe_done` has its label; the resident route call is back.

## Next steps

1. Build and test each PR bottom-up: `make -C c`, then
   `make -C c -f Makefile.deepseek-v4 dsv4-cuda-tests CUDA=1 CUDA_ARCH=sm_86`
   and `sm_120`.
2. Run the deferred GPU round: 2×3090 placement/resident/peer A/B, 5070
   sm_120, `V4_SERVE_STATS_LOG` runs, then the profiling pass (`nsys` is
   unavailable on shadowfax).
3. Update PR bodies with results and merge bottom-up.
