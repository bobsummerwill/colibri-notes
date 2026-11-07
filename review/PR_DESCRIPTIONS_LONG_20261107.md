# Long-form PR descriptions and verification comment, 2026-11-07

These are the pre-tersening descriptions and the consolidated verification
comment, kept for the record. The posted descriptions are in
`review/PR_BODIES_20261107.md`; the full round record is
`review/TEST_ROUND_20261107.md`.

## #2664 — long description

> **Rebase (2026-11-07):** this stack now targets upstream `dev` at `e24cd51a`, not the previous `6d907b56`. Branches are `split-20261107/*`; feature commit content and authorship are unchanged apart from conflict resolutions. Small test/build fix commits were added where the new base or the full suite exposed a break; they are listed in the verification comment.

## Where this comes from

This PR is one slice of a body of DeepSeek V4 work that arrived in three streams:

- **Upstream PR #1766** — *feat(dsv4): add automatic multi-GPU residency and faster CUDA execution* (ZacharyZcR, [JustVugg/colibri#1766](https://github.com/JustVugg/colibri/pull/1766)): multi-GPU layer placement, resident expert tables, device routing and a set of CUDA kernels.
- **Upstream PR #2405** — *deepseek-v4: two-GPU decode and VRAM-first expert lookups* (Bob Summerwill, [JustVugg/colibri#2405](https://github.com/JustVugg/colibri/pull/2405)): peer mirror pool, peer compute, VRAM-first decode, mirror-budget work and the sticky CUDA fault latch.
- **Head-followups fixes**: resident reserve split, `auto` hard-failure handling, head placement and `DSV4_CUDA_HEAD_DEVICE`, KV-owner guard, CUDA test target, docs.

### Who built it, and how long it has been iterated

- **ZacharyZcR** designed and implemented the core of #1766 — device placement, resident expert tables, device routing and the CUDA kernels — in commits dated 27 September to 26 October 2026, and has iterated that PR since it opened on **27 September 2026**.
- **Bob Summerwill** wrote #2405 (peer pool, peer compute, VRAM-first lookups, mirror budget, fault latch; branch work 14–25 October, PR opened 26 October), the head-followups fixes (3 November), the C+ reconciliation, and this recomposed split together with its test round.

The work has been iterated and retested for **about five weeks, 27 September – 4 November 2026**. That span covers the authors' own correctness passes, the C+ merge reconciliation on 3 November, and **three test/profiling passes over this stack on 3–4 November**: a smoke pass, a 3-repeat benchmark pass, then an expanded pass with 1024-token soaks, prefill and 4k-context runs, the full checkpoint, refusal paths and `nsys` captures. Every defect found in those passes was fixed on the owning branch, the branches above it were rebased and rebuilt, and the affected branches were force-pushed. The planning, per-branch manifests and full logs are in colibri-notes.

Together: **31 source commits plus one merge**, 37 files, +5,280/−378 against refreshed `dev`. Replayed directly they conflict repeatedly in the same five files, and the combined PR in this repository ([#3](https://github.com/bobsummerwill/colibri/pull/3)) was too large to review. This stack splits the aggregate into **six stand-alone feature PRs plus one integration PR**, *recomposed rather than replayed*: add-then-fix pairs folded into their feature, merge artefacts moved to where they belong, late docs follow their feature. Two lines are independent (1766 and 2405) and meet in the integration PR.

## Testing method

- **Builds:** shadowfax 2×RTX 3090, CUDA 12.0 → sm_86 with `NO_TC=1` (the optional cuBLASLt block-scaling path needs 12.8); computer4 RTX 5070, CUDA 12.8 → sm_120. Cold build ~30–60 s per branch on 32 cores.
- **Unit tests:** the nine `dsv4-cuda-tests`, the placement mock (`test_v4_gpu_placement.py`, ELF section GC), `test_dsv4_multigpu_cuda`, `test_dsv4_resident_route`, `test_dsv4_peer_init_cuda` / `test_dsv4_peer_moe_cuda`, the source-guard tests, CPU `test_deepseek_v4` and the arch-flags test.
- **Benchmarks:** DeepSeek-V4-Flash **REAP** (132 experts, 85 GB) unless stated, `--ram 48`, three repeats per cell, in three passes:
  1. **smoke** — 32 tokens, one run per cell (early validation);
  2. **long** — 128 tokens, 3 repeats, medians; headline cells interleaved 5 rounds × 3 to cancel page-cache/thermal drift;
  3. **expanded** — 1024-token soaks with VRAM sampled every 5 s, 512-token prefill, 4k-char context, 256-token generation, full checkpoint (256 experts, 156 GB), CPU-only, refusal paths and feature sweeps.
- **Profiling:** `nsys profile --trace=cuda` decode and prefill captures plus a separate `nsys stats` (the Ubuntu host/target packaging split is fixed by a user-level shim documented in the notes), and the engine's built-in counters.
- **Services:** llama/TabbyAPI and hermes stopped for the rounds, llama restarted afterwards.

## The stack

| # | PR | branch | base |
|---:|---|---|---|
| 1 | #2664 build fixes | `split-20261107/dsv4-build-fixes` | `dev` |
| 2 | #2665 multi-GPU placement | `split-20261107/dsv4-multigpu-layers` | #2664 |
| 3 | #2666 resident experts + routing | `split-20261107/dsv4-resident` | #2665 |
| 4 | #2667 CUDA kernels | `split-20261107/dsv4-cuda-kernels` | #2666 |
| 5 | #2668 mirror budget + VRAM-first | `split-20261107/dsv4-mirror-vram-first` | `dev` |
| 6 | #2669 peer pool + fault handling | `split-20261107/dsv4-peer-pool` | #2668 |
| — | #2670 integration (draft) | `split-20261107/dsv4-cplus-integration` | `split-20261107/integration-base` (merge of #2667 and #2669) |

Planning, per-branch manifests, the mess ledger and the complete logs are in **[colibri-notes](https://github.com/bobsummerwill/colibri-notes)**: `review/SPLIT_BUILT_6PR_20261103.md`, `review/TEST_ROUND_20261103.md`, `ops/NSYS_SHADOWFAX.md`.

## What this PR is

Two device/backend fixes that every other PR builds on. They are independent of all features and land first:

- **`dsv4_cuda_init`: a refused device leaves no stale CUDA error** (`402e51c2`). Validation used to return with `cudaErrorInvalidDevice` still pending, so a retry with a valid list succeeded and then failed its next checked call on the old error. It now clears the error, names the refused ordinal when several devices were requested, and refuses `NULL` lists and duplicate ordinals up front.
- **`native_quant.h`: the GGUF helpers are C only** (`994fa920`) — so the new C++/CUDA translation units that include the header link.

## Results

- **Builds:** sm_86 (CUDA 12.0, `NO_TC=1`) and sm_120 (CUDA 12.8) clean.
- **Tests:** `tests/test_deepseek_v4` passes (resource plan, sparse attention, core); `tests/test_deepseek_v4_cuda_arch_flags.py` passes.
- No feature benchmark belongs here; every branch above was built and tested on top of it.

## Profiling

None belongs to this PR: it changes device initialisation and header linkage only. Every `nsys` capture in the other PRs was taken on a branch built on this one.

## Why separate

These are the only changes in the stack that touch code no feature owns. Keeping them here lets the feature PRs rebase without dragging them along, and isolates a failure in device initialisation.





## #2665 — long description

> **Rebase (2026-11-07):** this stack now targets upstream `dev` at `e24cd51a`, not the previous `6d907b56`. Branches are `split-20261107/*`; feature commit content and authorship are unchanged apart from conflict resolutions. Small test/build fix commits were added where the new base or the full suite exposed a break; they are listed in the verification comment.

## Where this comes from

This PR is one slice of a body of DeepSeek V4 work that arrived in three streams:

- **Upstream PR #1766** — *feat(dsv4): add automatic multi-GPU residency and faster CUDA execution* (ZacharyZcR, [JustVugg/colibri#1766](https://github.com/JustVugg/colibri/pull/1766)): multi-GPU layer placement, resident expert tables, device routing and a set of CUDA kernels.
- **Upstream PR #2405** — *deepseek-v4: two-GPU decode and VRAM-first expert lookups* (Bob Summerwill, [JustVugg/colibri#2405](https://github.com/JustVugg/colibri/pull/2405)): peer mirror pool, peer compute, VRAM-first decode, mirror-budget work and the sticky CUDA fault latch.
- **Head-followups fixes**: resident reserve split, `auto` hard-failure handling, head placement and `DSV4_CUDA_HEAD_DEVICE`, KV-owner guard, CUDA test target, docs.

### Who built it, and how long it has been iterated

- **ZacharyZcR** designed and implemented the core of #1766 — device placement, resident expert tables, device routing and the CUDA kernels — in commits dated 27 September to 26 October 2026, and has iterated that PR since it opened on **27 September 2026**.
- **Bob Summerwill** wrote #2405 (peer pool, peer compute, VRAM-first lookups, mirror budget, fault latch; branch work 14–25 October, PR opened 26 October), the head-followups fixes (3 November), the C+ reconciliation, and this recomposed split together with its test round.

The work has been iterated and retested for **about five weeks, 27 September – 4 November 2026**. That span covers the authors' own correctness passes, the C+ merge reconciliation on 3 November, and **three test/profiling passes over this stack on 3–4 November**: a smoke pass, a 3-repeat benchmark pass, then an expanded pass with 1024-token soaks, prefill and 4k-context runs, the full checkpoint, refusal paths and `nsys` captures. Every defect found in those passes was fixed on the owning branch, the branches above it were rebased and rebuilt, and the affected branches were force-pushed. The planning, per-branch manifests and full logs are in colibri-notes.

Together: **31 source commits plus one merge**, 37 files, +5,280/−378 against refreshed `dev`. Replayed directly they conflict repeatedly in the same five files, and the combined PR in this repository ([#3](https://github.com/bobsummerwill/colibri/pull/3)) was too large to review. This stack splits the aggregate into **six stand-alone feature PRs plus one integration PR**, *recomposed rather than replayed*: add-then-fix pairs folded into their feature, merge artefacts moved to where they belong, late docs follow their feature. Two lines are independent (1766 and 2405) and meet in the integration PR.

## Testing method

- **Builds:** shadowfax 2×RTX 3090, CUDA 12.0 → sm_86 with `NO_TC=1` (the optional cuBLASLt block-scaling path needs 12.8); computer4 RTX 5070, CUDA 12.8 → sm_120. Cold build ~30–60 s per branch on 32 cores.
- **Unit tests:** the nine `dsv4-cuda-tests`, the placement mock (`test_v4_gpu_placement.py`, ELF section GC), `test_dsv4_multigpu_cuda`, `test_dsv4_resident_route`, `test_dsv4_peer_init_cuda` / `test_dsv4_peer_moe_cuda`, the source-guard tests, CPU `test_deepseek_v4` and the arch-flags test.
- **Benchmarks:** DeepSeek-V4-Flash **REAP** (132 experts, 85 GB) unless stated, `--ram 48`, three repeats per cell, in three passes:
  1. **smoke** — 32 tokens, one run per cell (early validation);
  2. **long** — 128 tokens, 3 repeats, medians; headline cells interleaved 5 rounds × 3 to cancel page-cache/thermal drift;
  3. **expanded** — 1024-token soaks with VRAM sampled every 5 s, 512-token prefill, 4k-char context, 256-token generation, full checkpoint (256 experts, 156 GB), CPU-only, refusal paths and feature sweeps.
- **Profiling:** `nsys profile --trace=cuda` decode and prefill captures plus a separate `nsys stats` (the Ubuntu host/target packaging split is fixed by a user-level shim documented in the notes), and the engine's built-in counters.
- **Services:** llama/TabbyAPI and hermes stopped for the rounds, llama restarted afterwards.

## The stack

| # | PR | branch | base |
|---:|---|---|---|
| 1 | #2664 build fixes | `split-20261107/dsv4-build-fixes` | `dev` |
| 2 | #2665 multi-GPU placement | `split-20261107/dsv4-multigpu-layers` | #2664 |
| 3 | #2666 resident experts + routing | `split-20261107/dsv4-resident` | #2665 |
| 4 | #2667 CUDA kernels | `split-20261107/dsv4-cuda-kernels` | #2666 |
| 5 | #2668 mirror budget + VRAM-first | `split-20261107/dsv4-mirror-vram-first` | `dev` |
| 6 | #2669 peer pool + fault handling | `split-20261107/dsv4-peer-pool` | #2668 |
| — | #2670 integration (draft) | `split-20261107/dsv4-cplus-integration` | `split-20261107/integration-base` (merge of #2667 and #2669) |

Planning, per-branch manifests, the mess ledger and the complete logs are in **[colibri-notes](https://github.com/bobsummerwill/colibri-notes)**: `review/SPLIT_BUILT_6PR_20261103.md`, `review/TEST_ROUND_20261103.md`, `ops/NSYS_SHADOWFAX.md`.

## What this PR is

**Multi-GPU layer placement** for the DeepSeek V4 CUDA tier: decode layers are spread over contiguous per-device ranges, and each layer's dense mirrors, attention KV and routed-expert mirrors live on the owning device.

### Commit

`fc6f3c57` **distribute CUDA layers and expert caches across devices** —
`DSV4_CUDA_DEVICES` (ordered, distinct ordinals; `DSV4_CUDA_DEVICE` remains the single-device default), contiguous ranges (43 layers / 6 cards → 0–6, 7–13, …, 35–42), one mirror cache per device, dense/KV/mirror creation on the owner, and the KV-owner guard (`9bcbc95b`): a KV ring/compressor append addressed to a device that does not own the buffer is refused instead of writing through a foreign pointer.

This raises usable cache capacity; it is **not** tensor or pipeline parallelism. Layers still execute sequentially with CPU-canonical activations, and startup prints each assignment.

## Tests

On 2×3090:

- `test_v4_gpu_devices` — list parsing: ok.
- `test_v4_gpu_placement` (mock, via its Python driver) — ranges, ownership, cleanup: ok.
- `test_dsv4_multigpu_cuda 0,1` — every listed device exercised, duplicate list refused, active contexts survive a second init, reopen with a different KV owner: ok.
- `test_deepseek_v4` (CPU unit): ok.

## Benchmarks

**Smoke (32 tokens, one run):** `dev` 2.55 tok/s (TTFT 8.45 s) → one device 2.84 (3.99 s) → `DSV4_CUDA_DEVICES=0,1` 3.25 (4.04 s).

**Long (128 tokens, 3 repeats, medians):** `dev` 3.54 → one device 3.93 → placement **4.88** tok/s.

**Interleaved (5 rounds × 3, drift-cancelling):** `dev` 3.60 → single 3.95 → placement **4.96** tok/s.

**Other configurations:**

| cell | result |
|---|---|
| full checkpoint (256 experts), 32 tokens | single 2.61 → placement **3.09** tok/s |
| 512-token prefill | TTFT 53.2 s single / 53.3 s placement (placement does not accelerate prefill); post-prefill 3.30/3.32 |
| 4k-char context, 16 tokens | TTFT 41.9 / 42.0 s; steady 3.21 / 3.24; identical output |
| 256-token generation, single | 3.77 tok/s overall, steady 4.09 |
| soak 1024 tokens × 3, placement | **5.75** tok/s, peak 23,560 MiB on each card, stable, identical output |

All REAP cells produce byte-identical text (`bdda375698df`).

## Profiling

`nsys` decode capture with `DSV4_CUDA_DEVICES=0,1` (16 tokens):

| kernel | time |
|---|---:|
| `mv_fp4_grouped` 512×12 | 24.5 % |
| `mv` 4096×1 | 13.8 % |
| `mv_fp4_grouped` 1024×6 | 12.7 % |
| rest | `mv<8>` and compressor |

GPU memops are **98.7 % HtoD** (2.07 s / 18,055 copies), 1.3 % DtoH, no PtoP. Splitting layers does not by itself reduce host→device traffic; it adds cache room, and that is what the 24 % decode gain reflects.

## Defects found and fixed while testing (on this branch)

- the mirror guard's `unified` state was left function-static after the rebase; restored to the base behaviour (per-cache state is PR 5).
- `dsv4_cuda_init` had lost its re-init refusal (`||ng`); restored, matching the multigpu contract.
- the placement mock lacked the expert-bank/activation/set stubs and the FP4 format on its fake views; the free-VRAM probe assertion is relaxed to "at least one device probed" until the per-cache guard arrives in PR 5.




## #2666 — long description

> **Rebase (2026-11-07):** this stack now targets upstream `dev` at `e24cd51a`, not the previous `6d907b56`. Branches are `split-20261107/*`; feature commit content and authorship are unchanged apart from conflict resolutions. Small test/build fix commits were added where the new base or the full suite exposed a break; they are listed in the verification comment.

## Where this comes from

This PR is one slice of a body of DeepSeek V4 work that arrived in three streams:

- **Upstream PR #1766** — *feat(dsv4): add automatic multi-GPU residency and faster CUDA execution* (ZacharyZcR, [JustVugg/colibri#1766](https://github.com/JustVugg/colibri/pull/1766)): multi-GPU layer placement, resident expert tables, device routing and a set of CUDA kernels.
- **Upstream PR #2405** — *deepseek-v4: two-GPU decode and VRAM-first expert lookups* (Bob Summerwill, [JustVugg/colibri#2405](https://github.com/JustVugg/colibri/pull/2405)): peer mirror pool, peer compute, VRAM-first decode, mirror-budget work and the sticky CUDA fault latch.
- **Head-followups fixes**: resident reserve split, `auto` hard-failure handling, head placement and `DSV4_CUDA_HEAD_DEVICE`, KV-owner guard, CUDA test target, docs.

### Who built it, and how long it has been iterated

- **ZacharyZcR** designed and implemented the core of #1766 — device placement, resident expert tables, device routing and the CUDA kernels — in commits dated 27 September to 26 October 2026, and has iterated that PR since it opened on **27 September 2026**.
- **Bob Summerwill** wrote #2405 (peer pool, peer compute, VRAM-first lookups, mirror budget, fault latch; branch work 14–25 October, PR opened 26 October), the head-followups fixes (3 November), the C+ reconciliation, and this recomposed split together with its test round.

The work has been iterated and retested for **about five weeks, 27 September – 4 November 2026**. That span covers the authors' own correctness passes, the C+ merge reconciliation on 3 November, and **three test/profiling passes over this stack on 3–4 November**: a smoke pass, a 3-repeat benchmark pass, then an expanded pass with 1024-token soaks, prefill and 4k-context runs, the full checkpoint, refusal paths and `nsys` captures. Every defect found in those passes was fixed on the owning branch, the branches above it were rebased and rebuilt, and the affected branches were force-pushed. The planning, per-branch manifests and full logs are in colibri-notes.

Together: **31 source commits plus one merge**, 37 files, +5,280/−378 against refreshed `dev`. Replayed directly they conflict repeatedly in the same five files, and the combined PR in this repository ([#3](https://github.com/bobsummerwill/colibri/pull/3)) was too large to review. This stack splits the aggregate into **six stand-alone feature PRs plus one integration PR**, *recomposed rather than replayed*: add-then-fix pairs folded into their feature, merge artefacts moved to where they belong, late docs follow their feature. Two lines are independent (1766 and 2405) and meet in the integration PR.

## Testing method

- **Builds:** shadowfax 2×RTX 3090, CUDA 12.0 → sm_86 with `NO_TC=1` (the optional cuBLASLt block-scaling path needs 12.8); computer4 RTX 5070, CUDA 12.8 → sm_120. Cold build ~30–60 s per branch on 32 cores.
- **Unit tests:** the nine `dsv4-cuda-tests`, the placement mock (`test_v4_gpu_placement.py`, ELF section GC), `test_dsv4_multigpu_cuda`, `test_dsv4_resident_route`, `test_dsv4_peer_init_cuda` / `test_dsv4_peer_moe_cuda`, the source-guard tests, CPU `test_deepseek_v4` and the arch-flags test.
- **Benchmarks:** DeepSeek-V4-Flash **REAP** (132 experts, 85 GB) unless stated, `--ram 48`, three repeats per cell, in three passes:
  1. **smoke** — 32 tokens, one run per cell (early validation);
  2. **long** — 128 tokens, 3 repeats, medians; headline cells interleaved 5 rounds × 3 to cancel page-cache/thermal drift;
  3. **expanded** — 1024-token soaks with VRAM sampled every 5 s, 512-token prefill, 4k-char context, 256-token generation, full checkpoint (256 experts, 156 GB), CPU-only, refusal paths and feature sweeps.
- **Profiling:** `nsys profile --trace=cuda` decode and prefill captures plus a separate `nsys stats` (the Ubuntu host/target packaging split is fixed by a user-level shim documented in the notes), and the engine's built-in counters.
- **Services:** llama/TabbyAPI and hermes stopped for the rounds, llama restarted afterwards.

## The stack

| # | PR | branch | base |
|---:|---|---|---|
| 1 | #2664 build fixes | `split-20261107/dsv4-build-fixes` | `dev` |
| 2 | #2665 multi-GPU placement | `split-20261107/dsv4-multigpu-layers` | #2664 |
| 3 | #2666 resident experts + routing | `split-20261107/dsv4-resident` | #2665 |
| 4 | #2667 CUDA kernels | `split-20261107/dsv4-cuda-kernels` | #2666 |
| 5 | #2668 mirror budget + VRAM-first | `split-20261107/dsv4-mirror-vram-first` | `dev` |
| 6 | #2669 peer pool + fault handling | `split-20261107/dsv4-peer-pool` | #2668 |
| — | #2670 integration (draft) | `split-20261107/dsv4-cplus-integration` | `split-20261107/integration-base` (merge of #2667 and #2669) |

Planning, per-branch manifests, the mess ledger and the complete logs are in **[colibri-notes](https://github.com/bobsummerwill/colibri-notes)**: `review/SPLIT_BUILT_6PR_20261103.md`, `review/TEST_ROUND_20261103.md`, `ops/NSYS_SHADOWFAX.md`.

## What this PR is

**Resident routed experts** and **device routing**: target experts are preloaded into immutable per-layer device tables, host expert loads are skipped when a layer is fully resident, and the default is a budgeted `auto` mode.

### Commits

- `893607e7` **resident routed experts with a budgeted auto mode** — skip host loaders when every routed expert and the shared expert of a layer are mirrored (`460f21cb`); preload immutable tables published only after every card succeeds (`919c58cd`); `DSV4_CUDA_RESIDENT_EXPERTS=auto` (default) keeps the LRU cache on a **budget refusal** but rejects a **hard load failure**, `1` requires residency, `0` keeps the cache; `coli_v4_gpu_experts_init()` owns that decision; `DSV4_CUDA_RESIDENT_RESERVE_MB` is separate from the mirror reserve; NULL guard in `v4_gpu_expert_find`.
- `39685b0f` **route and run resident MoE on the device** — a per-layer borrowed-weight descriptor table; routing, ascending expert-ID ordering, descriptor selection and MoE on one device stream; `DSV4_CUDA_RESIDENT_ROUTE=0` keeps host routing; an older DLL that declines the optional entry point keeps the old path.

Ordered after placement because the author's commits are its children: the resident table is per owning device.

## Tests

On 2×3090:

- `test_v4_gpu_placement` — full/partial/incomplete residency, `auto` hard-failure vs budget fallback, backend decline, locking through compute: ok.
- `test_dsv4_resident_route 0,1` — exact parity on two devices, normal and hash routes: ok.
- `test_deepseek_v4`: ok.

## Benchmarks

| cell | steady |
|---|---:|
| `DSV4_CUDA_RESIDENT_EXPERTS=0` | 3.59 tok/s |
| `auto` | 3.60 tok/s |

`auto` logs `resident-experts=cache-fallback (resident experts exceed device 0 budget …)` and keeps the cache — the intended behaviour. `DSV4_CUDA_RESIDENT_EXPERTS=1` **refuses engine open** (exit 1) with the same budget line. A 32-token smoke pair showed `0` 2.61 vs `auto` 2.58 tok/s (TTFT 7.05 vs 3.97 s; the difference is page-cache order, not the feature).

REAP has 132 experts and the full model 256; neither fits 24 GB × 43 layers, so **full residency and the routing path cannot be measured on this hardware**. The budget-refusal/fallback path is what the defaults take here, and it is stable in the integration soak (4.27 tok/s single, 5.75 placement, 5.74 peer at 1024 tokens).

## Profiling

`nsys` decode capture on this branch with `auto` (16 tokens, the fallback path): 24.6 % `mv_fp4_grouped` (512×12), 13.6 % `mv` (4096×1), 12.8 % `mv_fp4_grouped` (1024×6), then `mv<8>` and compressor. Because residency falls back on this model, the profile is the cache path by construction; a resident-table profile needs a checkpoint that fits 24 GB (or a 6-GPU box).

## Defect found while testing

`v4_gpu_resident_reserve_mb` was used before its definition (fixed in PR 4's branch, where the head upload consumes it); the resident-route test hard-coded devices `{0,5}` and now reads `DSV4_TEST_DEVICES` (later the shared header).





## #2667 — long description

> **Rebase (2026-11-07):** this stack now targets upstream `dev` at `e24cd51a`, not the previous `6d907b56`. Branches are `split-20261107/*`; feature commit content and authorship are unchanged apart from conflict resolutions. Small test/build fix commits were added where the new base or the full suite exposed a break; they are listed in the verification comment.

## Where this comes from

This PR is one slice of a body of DeepSeek V4 work that arrived in three streams:

- **Upstream PR #1766** — *feat(dsv4): add automatic multi-GPU residency and faster CUDA execution* (ZacharyZcR, [JustVugg/colibri#1766](https://github.com/JustVugg/colibri/pull/1766)): multi-GPU layer placement, resident expert tables, device routing and a set of CUDA kernels.
- **Upstream PR #2405** — *deepseek-v4: two-GPU decode and VRAM-first expert lookups* (Bob Summerwill, [JustVugg/colibri#2405](https://github.com/JustVugg/colibri/pull/2405)): peer mirror pool, peer compute, VRAM-first decode, mirror-budget work and the sticky CUDA fault latch.
- **Head-followups fixes**: resident reserve split, `auto` hard-failure handling, head placement and `DSV4_CUDA_HEAD_DEVICE`, KV-owner guard, CUDA test target, docs.

### Who built it, and how long it has been iterated

- **ZacharyZcR** designed and implemented the core of #1766 — device placement, resident expert tables, device routing and the CUDA kernels — in commits dated 27 September to 26 October 2026, and has iterated that PR since it opened on **27 September 2026**.
- **Bob Summerwill** wrote #2405 (peer pool, peer compute, VRAM-first lookups, mirror budget, fault latch; branch work 14–25 October, PR opened 26 October), the head-followups fixes (3 November), the C+ reconciliation, and this recomposed split together with its test round.

The work has been iterated and retested for **about five weeks, 27 September – 4 November 2026**. That span covers the authors' own correctness passes, the C+ merge reconciliation on 3 November, and **three test/profiling passes over this stack on 3–4 November**: a smoke pass, a 3-repeat benchmark pass, then an expanded pass with 1024-token soaks, prefill and 4k-context runs, the full checkpoint, refusal paths and `nsys` captures. Every defect found in those passes was fixed on the owning branch, the branches above it were rebased and rebuilt, and the affected branches were force-pushed. The planning, per-branch manifests and full logs are in colibri-notes.

Together: **31 source commits plus one merge**, 37 files, +5,280/−378 against refreshed `dev`. Replayed directly they conflict repeatedly in the same five files, and the combined PR in this repository ([#3](https://github.com/bobsummerwill/colibri/pull/3)) was too large to review. This stack splits the aggregate into **six stand-alone feature PRs plus one integration PR**, *recomposed rather than replayed*: add-then-fix pairs folded into their feature, merge artefacts moved to where they belong, late docs follow their feature. Two lines are independent (1766 and 2405) and meet in the integration PR.

## Testing method

- **Builds:** shadowfax 2×RTX 3090, CUDA 12.0 → sm_86 with `NO_TC=1` (the optional cuBLASLt block-scaling path needs 12.8); computer4 RTX 5070, CUDA 12.8 → sm_120. Cold build ~30–60 s per branch on 32 cores.
- **Unit tests:** the nine `dsv4-cuda-tests`, the placement mock (`test_v4_gpu_placement.py`, ELF section GC), `test_dsv4_multigpu_cuda`, `test_dsv4_resident_route`, `test_dsv4_peer_init_cuda` / `test_dsv4_peer_moe_cuda`, the source-guard tests, CPU `test_deepseek_v4` and the arch-flags test.
- **Benchmarks:** DeepSeek-V4-Flash **REAP** (132 experts, 85 GB) unless stated, `--ram 48`, three repeats per cell, in three passes:
  1. **smoke** — 32 tokens, one run per cell (early validation);
  2. **long** — 128 tokens, 3 repeats, medians; headline cells interleaved 5 rounds × 3 to cancel page-cache/thermal drift;
  3. **expanded** — 1024-token soaks with VRAM sampled every 5 s, 512-token prefill, 4k-char context, 256-token generation, full checkpoint (256 experts, 156 GB), CPU-only, refusal paths and feature sweeps.
- **Profiling:** `nsys profile --trace=cuda` decode and prefill captures plus a separate `nsys stats` (the Ubuntu host/target packaging split is fixed by a user-level shim documented in the notes), and the engine's built-in counters.
- **Services:** llama/TabbyAPI and hermes stopped for the rounds, llama restarted afterwards.

## The stack

| # | PR | branch | base |
|---:|---|---|---|
| 1 | #2664 build fixes | `split-20261107/dsv4-build-fixes` | `dev` |
| 2 | #2665 multi-GPU placement | `split-20261107/dsv4-multigpu-layers` | #2664 |
| 3 | #2666 resident experts + routing | `split-20261107/dsv4-resident` | #2665 |
| 4 | #2667 CUDA kernels | `split-20261107/dsv4-cuda-kernels` | #2666 |
| 5 | #2668 mirror budget + VRAM-first | `split-20261107/dsv4-mirror-vram-first` | `dev` |
| 6 | #2669 peer pool + fault handling | `split-20261107/dsv4-peer-pool` | #2668 |
| — | #2670 integration (draft) | `split-20261107/dsv4-cplus-integration` | `split-20261107/integration-base` (merge of #2667 and #2669) |

Planning, per-branch manifests, the mess ledger and the complete logs are in **[colibri-notes](https://github.com/bobsummerwill/colibri-notes)**: `review/SPLIT_BUILT_6PR_20261103.md`, `review/TEST_ROUND_20261103.md`, `ops/NSYS_SHADOWFAX.md`.

## What this PR is

The **single-device CUDA kernels**: resident indexer weights and native quantisation, compressor and indexer preparation on the device, output-projection decode intermediates on the card, and the exact/batched vocabulary head — plus the CUDA test target.

### Commits

- `cc1882f4` **keep the indexer resident and decode FP4 from a table** (`1c42795b`, `7411d188`, `2c675407`) — `weights_proj` uploaded once per layer; native FP4/MXFP4 decode in `backend_cuda_dsv4_quant.cuh`; the 256-entry table (the bit path lost prefill); `DSV4_TEST_DEVICES`.
- `98cd5331` **compressor and indexer preparation on CUDA** (`7019d0af`) — bf16 `wkv`/`wgate` projection for decode with the `ape` bias where the CPU adds it; indexer query/head prep on device; GGUF and Vulkan order preserved.
- `a4d2d40f` **decode output projection on CUDA** (`371ed4e3`) — grouped `wo_a` + `wo_b` on the card, final vector only copied back.
- `b4e67e0a` **exact and batched vocabulary head on CUDA** (`81c6cb39`, `97a47941`, `35544b2d`, head placement from `07dc142a`, `f4586dca`, `6b2efd31`, `0956307c`) — all rows on the owning device (then any device with room, then CPU), `DSV4_HEAD_VERIFY` recompute, batched speculative head with device-side candidate reduction, validated batch heads on by default, `DSV4_CUDA_HEAD_DEVICE`, and the `dsv4-cuda-tests` target.

## Tests

All nine `dsv4-cuda-tests` pass on 2×3090 (`DSV4_TEST_DEVICES=0,1`): head, wo_decode, compressor, compressor_state, indexer_prepare, multigpu, quant, resident_route, resident_indexer. The placement mock and CPU unit tests pass. On computer4/sm_120 `test_dsv4_quant_cuda` passes (35,769 cases, 0 mismatches); the remaining tests require two distinct devices by construction.

## Benchmarks

**Smoke (32 tokens):** head off 2.71 → on 2.77 tok/s.

**Long (128 tokens, 3 repeats, medians):**

| toggle | off | on |
|---|---:|---:|
| exact head `DSV4_CUDA_HEAD` | 3.72 | **3.93** (+5.6 %) |
| batched head | 3.92 | 3.92 |
| `wo_decode` | 3.91 | 3.93 |
| compressor | 3.98 | 3.92 |
| indexer preparation | 3.89 | 3.94 |
| indexer heads | 3.93 | 3.95 |
| resident indexer | 3.93 | 3.95 |
| `COLI_CUDA_MOE_BATCH` | 3.92 | 3.79 |
| `COLI_CUDA_ATTN_BATCH` | 3.89 | 3.75 (text changes) |

## Profiling

`nsys` on this branch, decode: 21.0 % `mv_fp4_grouped` (512×12) + 10.9 % (1024×6), then `mv<8>` and `compressor_project_exact`. Aggregate decode: 32 % `mv_fp4_grouped`, 28 % `mv<8>`, 23 % compressor. Prefill: 64 % compressor, 27 % `mm_batch*` — prefill is compressor/dense-bound, decode is expert-matvec-bound.

**Caveat:** `COLI_CUDA_ATTN_BATCH=1` changes greedy text (`65e321b385ef` vs `bdda375698df`). That is the documented summation-order difference of the batched attention block, not a break, but it is not token-identical to the CPU reference and there is no `..._VERIFY` for it yet.




## #2668 — long description

> **Rebase (2026-11-07):** this stack now targets upstream `dev` at `e24cd51a`, not the previous `6d907b56`. Branches are `split-20261107/*`; feature commit content and authorship are unchanged apart from conflict resolutions. Small test/build fix commits were added where the new base or the full suite exposed a break; they are listed in the verification comment.

## Where this comes from

This PR is one slice of a body of DeepSeek V4 work that arrived in three streams:

- **Upstream PR #1766** — *feat(dsv4): add automatic multi-GPU residency and faster CUDA execution* (ZacharyZcR, [JustVugg/colibri#1766](https://github.com/JustVugg/colibri/pull/1766)): multi-GPU layer placement, resident expert tables, device routing and a set of CUDA kernels.
- **Upstream PR #2405** — *deepseek-v4: two-GPU decode and VRAM-first expert lookups* (Bob Summerwill, [JustVugg/colibri#2405](https://github.com/JustVugg/colibri/pull/2405)): peer mirror pool, peer compute, VRAM-first decode, mirror-budget work and the sticky CUDA fault latch.
- **Head-followups fixes**: resident reserve split, `auto` hard-failure handling, head placement and `DSV4_CUDA_HEAD_DEVICE`, KV-owner guard, CUDA test target, docs.

### Who built it, and how long it has been iterated

- **ZacharyZcR** designed and implemented the core of #1766 — device placement, resident expert tables, device routing and the CUDA kernels — in commits dated 27 September to 26 October 2026, and has iterated that PR since it opened on **27 September 2026**.
- **Bob Summerwill** wrote #2405 (peer pool, peer compute, VRAM-first lookups, mirror budget, fault latch; branch work 14–25 October, PR opened 26 October), the head-followups fixes (3 November), the C+ reconciliation, and this recomposed split together with its test round.

The work has been iterated and retested for **about five weeks, 27 September – 4 November 2026**. That span covers the authors' own correctness passes, the C+ merge reconciliation on 3 November, and **three test/profiling passes over this stack on 3–4 November**: a smoke pass, a 3-repeat benchmark pass, then an expanded pass with 1024-token soaks, prefill and 4k-context runs, the full checkpoint, refusal paths and `nsys` captures. Every defect found in those passes was fixed on the owning branch, the branches above it were rebased and rebuilt, and the affected branches were force-pushed. The planning, per-branch manifests and full logs are in colibri-notes.

Together: **31 source commits plus one merge**, 37 files, +5,280/−378 against refreshed `dev`. Replayed directly they conflict repeatedly in the same five files, and the combined PR in this repository ([#3](https://github.com/bobsummerwill/colibri/pull/3)) was too large to review. This stack splits the aggregate into **six stand-alone feature PRs plus one integration PR**, *recomposed rather than replayed*: add-then-fix pairs folded into their feature, merge artefacts moved to where they belong, late docs follow their feature. Two lines are independent (1766 and 2405) and meet in the integration PR.

## Testing method

- **Builds:** shadowfax 2×RTX 3090, CUDA 12.0 → sm_86 with `NO_TC=1` (the optional cuBLASLt block-scaling path needs 12.8); computer4 RTX 5070, CUDA 12.8 → sm_120. Cold build ~30–60 s per branch on 32 cores.
- **Unit tests:** the nine `dsv4-cuda-tests`, the placement mock (`test_v4_gpu_placement.py`, ELF section GC), `test_dsv4_multigpu_cuda`, `test_dsv4_resident_route`, `test_dsv4_peer_init_cuda` / `test_dsv4_peer_moe_cuda`, the source-guard tests, CPU `test_deepseek_v4` and the arch-flags test.
- **Benchmarks:** DeepSeek-V4-Flash **REAP** (132 experts, 85 GB) unless stated, `--ram 48`, three repeats per cell, in three passes:
  1. **smoke** — 32 tokens, one run per cell (early validation);
  2. **long** — 128 tokens, 3 repeats, medians; headline cells interleaved 5 rounds × 3 to cancel page-cache/thermal drift;
  3. **expanded** — 1024-token soaks with VRAM sampled every 5 s, 512-token prefill, 4k-char context, 256-token generation, full checkpoint (256 experts, 156 GB), CPU-only, refusal paths and feature sweeps.
- **Profiling:** `nsys profile --trace=cuda` decode and prefill captures plus a separate `nsys stats` (the Ubuntu host/target packaging split is fixed by a user-level shim documented in the notes), and the engine's built-in counters.
- **Services:** llama/TabbyAPI and hermes stopped for the rounds, llama restarted afterwards.

## The stack

| # | PR | branch | base |
|---:|---|---|---|
| 1 | #2664 build fixes | `split-20261107/dsv4-build-fixes` | `dev` |
| 2 | #2665 multi-GPU placement | `split-20261107/dsv4-multigpu-layers` | #2664 |
| 3 | #2666 resident experts + routing | `split-20261107/dsv4-resident` | #2665 |
| 4 | #2667 CUDA kernels | `split-20261107/dsv4-cuda-kernels` | #2666 |
| 5 | #2668 mirror budget + VRAM-first | `split-20261107/dsv4-mirror-vram-first` | `dev` |
| 6 | #2669 peer pool + fault handling | `split-20261107/dsv4-peer-pool` | #2668 |
| — | #2670 integration (draft) | `split-20261107/dsv4-cplus-integration` | `split-20261107/integration-base` (merge of #2667 and #2669) |

Planning, per-branch manifests, the mess ledger and the complete logs are in **[colibri-notes](https://github.com/bobsummerwill/colibri-notes)**: `review/SPLIT_BUILT_6PR_20261103.md`, `review/TEST_ROUND_20261103.md`, `ops/NSYS_SHADOWFAX.md`.

## What this PR is

The **mirror-cache budget** lane from #2405, independent of the 1766 stack: per-cache growth guard, VRAM-first decode, per-turn stats and the page-locked slab cap.

### Commits

- `0ec4c82b` **keep decode lookups in VRAM and log per-turn stats** (`0d6454f6`, `897ff79e`) — `V4_GPU_RESIDENT_FIRST=1` computes an expert already mirrored in VRAM without asking the store; the store still records routing and its RAM pin ranking leaves mirrored experts out, so VRAM adds to the RAM tier instead of duplicating it. `views[]`/`fetch[]`/`leased[]` separate "experts to compute" from "experts the store must provide"; prefetch asks only for the latter. `V4_SERVE_STATS_LOG=1` prints the turn's decode seconds/tok/s/hit rate and the decode-MoE totals, including VRAM-first hits and store fetches.
- `7b7d1764` **per-cache mirror guard, reserve and pin cap** (`99a37137`, `93ec3778`, `4293d4c9`) — guard state per cache (a peer pool has its own free VRAM and reserve), one LRU clock for cross-cache victim comparison, an explicit reserve kept as given, a failed upload drops its mirror instead of pinning a broken one, `grown_bytes` accounts for mirrors since the last probe, and `DSV4_CUDA_PIN_CAP_GB` caps page-locked expert slabs.

### Why a separate lane

These are mirror/budget mechanics, not placement or residency; they apply to `dev` directly and are reviewed in parallel with the 1766 stack. The integration PR reconciles them with per-device caches.

## Tests

`test_v4_mirror_guard_source.py` passes. The guard is exercised end-to-end by every benchmark run: growth while the reserve is free, recycling the older LRU entry, dropping a failed upload, and the per-cache reserve.

## Benchmarks (128 tokens, 3 repeats)

| cell | steady |
|---|---:|
| `V4_GPU_RESIDENT_FIRST=0` / `1` | 3.57 / **3.62** tok/s |
| mirrors 512 / 2048 / 4096 | 2.99 / 3.57 / 3.57 |
| `DSV4_CUDA_VRAM_RESERVE_MB=600` / 2000 | 3.58 / 3.52 |
| `DSV4_CUDA_PIN_CAP_GB=0` / 1 | 3.68 / 3.67 |

A 512-mirror cap costs 16 %; a larger reserve costs ~1.5 %; VRAM-first is a small consistent gain on a 3090. `V4_SERVE_STATS_LOG` lines (`v4_serve_turn`, `v4_dec_moe`) were verified in every cell.

## Profiling

`nsys` with `V4_GPU_RESIDENT_FIRST=1`: 24.4 % `mv_fp4_grouped` (512×12), 13.8 % `mv` (4096×1), then the compressor and `mv<8>` variants; no abnormal host stalls beyond normal uploads. Soak at 1024 tokens on this lane's integration: 4.27 tok/s single, peak 23.56 GB.

## Correction

`COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER` **is** on by default, so `V4_GPU_RESIDENT_FIRST` is measurable (an earlier note in this PR's history said otherwise).

**Docs:** the environment rows for VRAM-first, the pin cap and the stats line travel with PR 6 because the author's fault commit carried them; moving them here is a docs-only rebase.




## #2669 — long description

> **Rebase (2026-11-07):** this stack now targets upstream `dev` at `e24cd51a`, not the previous `6d907b56`. Branches are `split-20261107/*`; feature commit content and authorship are unchanged apart from conflict resolutions. Small test/build fix commits were added where the new base or the full suite exposed a break; they are listed in the verification comment.

## Where this comes from

This PR is one slice of a body of DeepSeek V4 work that arrived in three streams:

- **Upstream PR #1766** — *feat(dsv4): add automatic multi-GPU residency and faster CUDA execution* (ZacharyZcR, [JustVugg/colibri#1766](https://github.com/JustVugg/colibri/pull/1766)): multi-GPU layer placement, resident expert tables, device routing and a set of CUDA kernels.
- **Upstream PR #2405** — *deepseek-v4: two-GPU decode and VRAM-first expert lookups* (Bob Summerwill, [JustVugg/colibri#2405](https://github.com/JustVugg/colibri/pull/2405)): peer mirror pool, peer compute, VRAM-first decode, mirror-budget work and the sticky CUDA fault latch.
- **Head-followups fixes**: resident reserve split, `auto` hard-failure handling, head placement and `DSV4_CUDA_HEAD_DEVICE`, KV-owner guard, CUDA test target, docs.

### Who built it, and how long it has been iterated

- **ZacharyZcR** designed and implemented the core of #1766 — device placement, resident expert tables, device routing and the CUDA kernels — in commits dated 27 September to 26 October 2026, and has iterated that PR since it opened on **27 September 2026**.
- **Bob Summerwill** wrote #2405 (peer pool, peer compute, VRAM-first lookups, mirror budget, fault latch; branch work 14–25 October, PR opened 26 October), the head-followups fixes (3 November), the C+ reconciliation, and this recomposed split together with its test round.

The work has been iterated and retested for **about five weeks, 27 September – 4 November 2026**. That span covers the authors' own correctness passes, the C+ merge reconciliation on 3 November, and **three test/profiling passes over this stack on 3–4 November**: a smoke pass, a 3-repeat benchmark pass, then an expanded pass with 1024-token soaks, prefill and 4k-context runs, the full checkpoint, refusal paths and `nsys` captures. Every defect found in those passes was fixed on the owning branch, the branches above it were rebased and rebuilt, and the affected branches were force-pushed. The planning, per-branch manifests and full logs are in colibri-notes.

Together: **31 source commits plus one merge**, 37 files, +5,280/−378 against refreshed `dev`. Replayed directly they conflict repeatedly in the same five files, and the combined PR in this repository ([#3](https://github.com/bobsummerwill/colibri/pull/3)) was too large to review. This stack splits the aggregate into **six stand-alone feature PRs plus one integration PR**, *recomposed rather than replayed*: add-then-fix pairs folded into their feature, merge artefacts moved to where they belong, late docs follow their feature. Two lines are independent (1766 and 2405) and meet in the integration PR.

## Testing method

- **Builds:** shadowfax 2×RTX 3090, CUDA 12.0 → sm_86 with `NO_TC=1` (the optional cuBLASLt block-scaling path needs 12.8); computer4 RTX 5070, CUDA 12.8 → sm_120. Cold build ~30–60 s per branch on 32 cores.
- **Unit tests:** the nine `dsv4-cuda-tests`, the placement mock (`test_v4_gpu_placement.py`, ELF section GC), `test_dsv4_multigpu_cuda`, `test_dsv4_resident_route`, `test_dsv4_peer_init_cuda` / `test_dsv4_peer_moe_cuda`, the source-guard tests, CPU `test_deepseek_v4` and the arch-flags test.
- **Benchmarks:** DeepSeek-V4-Flash **REAP** (132 experts, 85 GB) unless stated, `--ram 48`, three repeats per cell, in three passes:
  1. **smoke** — 32 tokens, one run per cell (early validation);
  2. **long** — 128 tokens, 3 repeats, medians; headline cells interleaved 5 rounds × 3 to cancel page-cache/thermal drift;
  3. **expanded** — 1024-token soaks with VRAM sampled every 5 s, 512-token prefill, 4k-char context, 256-token generation, full checkpoint (256 experts, 156 GB), CPU-only, refusal paths and feature sweeps.
- **Profiling:** `nsys profile --trace=cuda` decode and prefill captures plus a separate `nsys stats` (the Ubuntu host/target packaging split is fixed by a user-level shim documented in the notes), and the engine's built-in counters.
- **Services:** llama/TabbyAPI and hermes stopped for the rounds, llama restarted afterwards.

## The stack

| # | PR | branch | base |
|---:|---|---|---|
| 1 | #2664 build fixes | `split-20261107/dsv4-build-fixes` | `dev` |
| 2 | #2665 multi-GPU placement | `split-20261107/dsv4-multigpu-layers` | #2664 |
| 3 | #2666 resident experts + routing | `split-20261107/dsv4-resident` | #2665 |
| 4 | #2667 CUDA kernels | `split-20261107/dsv4-cuda-kernels` | #2666 |
| 5 | #2668 mirror budget + VRAM-first | `split-20261107/dsv4-mirror-vram-first` | `dev` |
| 6 | #2669 peer pool + fault handling | `split-20261107/dsv4-peer-pool` | #2668 |
| — | #2670 integration (draft) | `split-20261107/dsv4-cplus-integration` | `split-20261107/integration-base` (merge of #2667 and #2669) |

Planning, per-branch manifests, the mess ledger and the complete logs are in **[colibri-notes](https://github.com/bobsummerwill/colibri-notes)**: `review/SPLIT_BUILT_6PR_20261103.md`, `review/TEST_ROUND_20261103.md`, `ops/NSYS_SHADOWFAX.md`.

## What this PR is

The **peer mirror pool** and **CUDA fault handling** lane from #2405, on top of the mirror budget.

### Commits

- `1bf47730` **peer mirror pool on a second GPU** (`6928922a`) — `DSV4_CUDA_PEER_DEVICE`, peer mirrors and reserve, `dsv4_cuda_peer_probe()` before any init (a refused peer is never initialised), init rollback that releases every touched device and drops the peer's context, tiered attach (grow primary → grow peer → recycle the older LRU), peek with primary-then-peer.
- `9575ff47` **peer compute on the second GPU** (`83850cf9`) — `DSV4_CUDA_PEER_COMPUTE=1` runs the peer's experts **on the peer** and reduces into the primary's buffer; a latched-off peer is refused before any CUDA call; `v4_peer_pool()` hides a lost peer.
- `7196995d` **latch a CUDA fault off and name decode failures** (`14339ca2`, `0ce6e5dc`, `11c40038`) — sticky process-wide latch for kernel faults/illegal address/lost context, per-peer latch, `DSV4_CUDA_PEER_FAULT` test knob, stage-named decode failure that keeps the inner error, and the `wo_a` CPU fallback when the backend refuses.

## Tests

On 2×3090:

- `test_dsv4_peer_init_cuda 0,1` — probe before init, refused peer untouched, failed init rolled back and the primary retried: ok.
- `test_dsv4_peer_moe_cuda 0,1` — peer compute and P2P parity; a survivable peer error latches the peer off and recomputes on the primary; a sticky fault refuses every later call and leaves CUDA off for the process: ok.
- `test_v4_peer_init_source.py`, `test_dsv4_cuda_lost_source.py`: ok.

## Benchmarks (128 tokens, 3 repeats)

| config | steady |
|---|---:|
| pool off | 3.58 tok/s |
| `PEER_COMPUTE=0` (P2P read) | 3.86 |
| `PEER_COMPUTE=1` (compute) | **4.27** |
| peer mirrors 512 / 4096 | 3.89 / 4.30 |
| peer reserve 512 / 2048 | 4.30 / 4.27 |
| full checkpoint (256 experts), peer compute | 3.21 |

Smoke (32 tokens): pool off 2.84 → compute 3.31 tok/s.

Injected `DSV4_CUDA_PEER_FAULT=1` during decode: `peer device 1 failed in the decode MoE (join: too many resources requested for launch); latched off, decode continues on device 0` — the run completes with the same output.

## Profiling

With peer compute, GPU memop time is **98.7 % HtoD** (2.07 s / 18,379 copies), 1.3 % DtoH and 0.04 % PtoP. The peer path is bounded by host→device traffic, not P2P copies — the next target if peer compute is pursued.

**Docs:** the peer rows and the fault knob are documented by this PR's fault commit (the author's catch-up); VRAM-first/pin/stats rows also travel here and can move to PR 5 if preferred.




## #2670 — long description

> **Rebase (2026-11-07):** integration branch refreshed onto upstream `dev` at `e24cd51a`. This is a merge of the two lanes; open as a draft until #2667 and #2669 land. Small test/build fix commits were added where the new base or the full suite exposed a break; they are listed in the verification comment.

## Where this comes from

This PR is one slice of a body of DeepSeek V4 work that arrived in three streams:

- **Upstream PR #1766** — *feat(dsv4): add automatic multi-GPU residency and faster CUDA execution* (ZacharyZcR, [JustVugg/colibri#1766](https://github.com/JustVugg/colibri/pull/1766)): multi-GPU layer placement, resident expert tables, device routing and a set of CUDA kernels.
- **Upstream PR #2405** — *deepseek-v4: two-GPU decode and VRAM-first expert lookups* (Bob Summerwill, [JustVugg/colibri#2405](https://github.com/JustVugg/colibri/pull/2405)): peer mirror pool, peer compute, VRAM-first decode, mirror-budget work and the sticky CUDA fault latch.
- **Head-followups fixes**: resident reserve split, `auto` hard-failure handling, head placement and `DSV4_CUDA_HEAD_DEVICE`, KV-owner guard, CUDA test target, docs.

### Who built it, and how long it has been iterated

- **ZacharyZcR** designed and implemented the core of #1766 — device placement, resident expert tables, device routing and the CUDA kernels — in commits dated 27 September to 26 October 2026, and has iterated that PR since it opened on **27 September 2026**.
- **Bob Summerwill** wrote #2405 (peer pool, peer compute, VRAM-first lookups, mirror budget, fault latch; branch work 14–25 October, PR opened 26 October), the head-followups fixes (3 November), the C+ reconciliation, and this recomposed split together with its test round.

The work has been iterated and retested for **about five weeks, 27 September – 4 November 2026**. That span covers the authors' own correctness passes, the C+ merge reconciliation on 3 November, and **three test/profiling passes over this stack on 3–4 November**: a smoke pass, a 3-repeat benchmark pass, then an expanded pass with 1024-token soaks, prefill and 4k-context runs, the full checkpoint, refusal paths and `nsys` captures. Every defect found in those passes was fixed on the owning branch, the branches above it were rebased and rebuilt, and the affected branches were force-pushed. The planning, per-branch manifests and full logs are in colibri-notes.

Together: **31 source commits plus one merge**, 37 files, +5,280/−378 against refreshed `dev`. Replayed directly they conflict repeatedly in the same five files, and the combined PR in this repository ([#3](https://github.com/bobsummerwill/colibri/pull/3)) was too large to review. This stack splits the aggregate into **six stand-alone feature PRs plus one integration PR**, *recomposed rather than replayed*: add-then-fix pairs folded into their feature, merge artefacts moved to where they belong, late docs follow their feature. Two lines are independent (1766 and 2405) and meet in the integration PR.

## Testing method

- **Builds:** shadowfax 2×RTX 3090, CUDA 12.0 → sm_86 with `NO_TC=1` (the optional cuBLASLt block-scaling path needs 12.8); computer4 RTX 5070, CUDA 12.8 → sm_120. Cold build ~30–60 s per branch on 32 cores.
- **Unit tests:** the nine `dsv4-cuda-tests`, the placement mock (`test_v4_gpu_placement.py`, ELF section GC), `test_dsv4_multigpu_cuda`, `test_dsv4_resident_route`, `test_dsv4_peer_init_cuda` / `test_dsv4_peer_moe_cuda`, the source-guard tests, CPU `test_deepseek_v4` and the arch-flags test.
- **Benchmarks:** DeepSeek-V4-Flash **REAP** (132 experts, 85 GB) unless stated, `--ram 48`, three repeats per cell, in three passes:
  1. **smoke** — 32 tokens, one run per cell (early validation);
  2. **long** — 128 tokens, 3 repeats, medians; headline cells interleaved 5 rounds × 3 to cancel page-cache/thermal drift;
  3. **expanded** — 1024-token soaks with VRAM sampled every 5 s, 512-token prefill, 4k-char context, 256-token generation, full checkpoint (256 experts, 156 GB), CPU-only, refusal paths and feature sweeps.
- **Profiling:** `nsys profile --trace=cuda` decode and prefill captures plus a separate `nsys stats` (the Ubuntu host/target packaging split is fixed by a user-level shim documented in the notes), and the engine's built-in counters.
- **Services:** llama/TabbyAPI and hermes stopped for the rounds, llama restarted afterwards.

## The stack

| # | PR | branch | base |
|---:|---|---|---|
| 1 | #2664 build fixes | `split-20261107/dsv4-build-fixes` | `dev` |
| 2 | #2665 multi-GPU placement | `split-20261107/dsv4-multigpu-layers` | #2664 |
| 3 | #2666 resident experts + routing | `split-20261107/dsv4-resident` | #2665 |
| 4 | #2667 CUDA kernels | `split-20261107/dsv4-cuda-kernels` | #2666 |
| 5 | #2668 mirror budget + VRAM-first | `split-20261107/dsv4-mirror-vram-first` | `dev` |
| 6 | #2669 peer pool + fault handling | `split-20261107/dsv4-peer-pool` | #2668 |
| — | #2670 integration | `split-20261107/dsv4-cplus-integration` | `split-20261107/integration-base` (merge of #2667 and #2669) |

Planning, per-branch manifests, the mess ledger and the complete logs are in **[colibri-notes](https://github.com/bobsummerwill/colibri-notes)**: `review/SPLIT_BUILT_6PR_20261103.md`, `review/TEST_ROUND_20261103.md`, `ops/NSYS_SHADOWFAX.md`.

## What this PR is

The **integration** of the two independent lanes: 1766 (placement → resident → routing → kernels) and 2405 (mirror budget → VRAM-first → peer pool → fault handling). It is based on `split-20261107/integration-base`, a temporary branch merging #2667 and #2669, because the change needs two branches and a GitHub PR has one base.

### Commits

- `74e76d51` (base) merge of #2667 and #2669.
- `2ced5ed3` **integration: layer placement and the peer pool coexist** —
  - `DSV4_CUDA_DEVICES` with more than one device and `DSV4_CUDA_PEER_DEVICE` are **mutually exclusive** and refuse at open (both claim the other cards);
  - per-device mirror caches stay the primary path; the VRAM-held scan, peek and tiered attach look in the layer's cache **and** in the healthy peer pool;
  - peer access is verified after the best-effort layer-split init before the pool is created;
  - `dsv4_cuda_init` is strict by default again (the peer tier depends on it), with `dsv4_cuda_set_peer_best_effort(1)` only around the multi-device split init; the `||ng` re-init refusal is kept;
  - a failed mirror upload counts as a full card in the growth guard; the resident routing hop and its `moe_done` label stay intact in the merged pipeline;
  - the matvec format guard (fmt 4/8/9/16/32) lost by the merge is restored;
  - the placement mock's `shared_missing` and duplicate stubs are repaired, and `test_v4_peer_init_source.py` is updated for the integrated engine-open shape (mutual exclusion, three init call sites).

## Tests (2×3090)

All nine `dsv4-cuda-tests`, the placement mock, `test_dsv4_multigpu_cuda`, `test_dsv4_peer_init_cuda`, `test_dsv4_peer_moe_cuda` (incl. sticky/lost injection), the mirror/peer/lost source tests, `test_deepseek_v4` and the arch-flags test pass. On computer4/sm_120 `test_dsv4_quant_cuda` passes (35,769 cases, 0 mismatches); the remaining tests require two distinct devices.

## Benchmarks

| config | TTFT | steady |
|---|---:|---:|
| `dev` | 9.5 s | 3.60 tok/s |
| integration, one 3090 | 5.6 s | 3.95 |
| integration, placement | 5.7 s | 4.96 |
| integration, peer compute | 5.6 s | **5.03** |
| full checkpoint, single / placement | 4.35 / 4.06 s | 2.61 / 3.09 |

- **Smoke (32 tokens):** `dev` 2.55, single 2.84, placement 3.25, peer 3.31.
- **Long (128 tokens, 3 repeats):** 3.54 / 3.93 / 4.88 / 4.98.
- **Interleaved (5 rounds × 3):** 3.60 / 3.95 / 4.96 / 5.03.
- **Soak** 1024 tokens × 3: single 4.27, placement 5.75, peer 5.74 tok/s; peak 23.56 GB per card (~96 %), stable, identical output.
- **Prefill:** 512-token prompt TTFT 53.2 s single / 53.3 s placement (placement does not accelerate prefill); post-prefill 3.30/3.32.
- **Context/generation:** 4k-char prompt TTFT 41.9/42.0 s, steady 3.21/3.24; 256-token generation 3.77 overall, steady 4.09.
- **Full model:** single 2.61 (32 tokens) / 3.21 steady (256 tokens), placement 3.09, peer 3.21.
- **CPU-only** (`DSV4_CUDA=0`, 8 tokens): 2.02 tok/s.
- **Refusals:** `RESIDENT_EXPERTS=1` over budget exits 1; placement+peer exits 1 with the mutual-exclusion line.
- **Output parity:** every default REAP cell byte-identical (`bdda375698df`); both full-checkpoint cells identical.

## Profiling

- **Decode:** 32 % `mv_fp4_grouped`, 28 % `mv<8>`, 23 % `compressor_project_exact`, rest spread.
- **Prefill:** 64 % `compressor_project_exact`, 27 % `mm_batch`/`mm_batch_w`; prefill is compressor/dense-bound.
- **Peer decode:** 98.7 % of GPU memop time is HtoD (2.07 s / 18,379 copies), 1.3 % DtoH, 0.04 % PtoP.
- **Placement decode:** 24.5 % `mv_fp4_grouped` (512×12), 13.8 % `mv` (4096×1), 12.7 % `mv_fp4_grouped` (1024×6); memops 98.7 % HtoD.

## Defects found and fixed while testing

- merge dropped PR 4's matvec format guard; restored.
- duplicated struct members and a lost closing brace from applying the integration diff onto the new merge; repaired.
- `shared_missing` mock lost and duplicate stubs introduced; repaired.
- `test_v4_peer_init_source.py` updated for the integrated engine-open shape.
- `dsv4_cuda_init` keeps the `||ng` refusal alongside `g_cuda_lost`.

## Known deviations and limits

- `COLI_CUDA_ATTN_BATCH=1` changes greedy text — documented floating-point reassociation, opt-in; no `..._VERIFY` for it yet.
- Full expert residency and device routing are unmeasurable on 24 GB cards (132/256 experts × 43 layers); only the budget-fallback path runs here.
- sm_120 coverage is build + quant; the two-device tests ran on the 2×3090.
- Benchmarks are medians of three (headline cells 15) runs on a warm page cache, not a formal benchmark suite.

## What is folded in, not a separate PR

The seven head-followups commits are **not** a separate PR: the KV guard is in #2665, `auto`/reserve/mocks in #2666, head placement/override and the test target in #2667, docs distributed. PRs #10 (`split-20261107/dsv4-cuda-fault`) and #12 (`split-20261107/dsv4-head-followups`) were closed as folded.




## Verification comment (posted on all seven PRs, then folded into the descriptions)

## Verification round — 2026-11-07 (rebase onto `dev` `e24cd51a`)

**Environment.** shadowfax: 2×RTX 3090 (sm_86, CUDA 12.0, `NO_TC=1`, 32 cores). computer4: RTX 5070 (sm_120, CUDA 12.8). `llama-server-active` and `camofox-browser` were stopped and `hermes-gateway` frozen for the round; all were restored afterwards.

**Branches.** All six feature branches and the integration branch now sit directly on upstream `dev` `e24cd51a`. Commit content and authorship are unchanged apart from the rebase resolutions and the test/build fixes listed below.

| # | branch | tip |
|---:|---|---|
| 1 | `split-20261107/dsv4-build-fixes` | `60986bba` |
| 2 | `split-20261107/dsv4-multigpu-layers` | `1d152bee` |
| 3 | `split-20261107/dsv4-resident` | `e893e544` |
| 4 | `split-20261107/dsv4-cuda-kernels` | `c34d3a92` |
| 5 | `split-20261107/dsv4-mirror-vram-first` | `4891a34d` |
| 6 | `split-20261107/dsv4-peer-pool` | `74bfa093` |
| — | `split-20261107/dsv4-cplus-integration` | `4c468115` |

### Builds

| host | arch | result |
|---|---|---|
| shadowfax | sm_86, CUDA 12.0, `NO_TC=1` | all seven branches clean |
| computer4 | sm_120, CUDA 12.8 | all six feature branches clean |

`NO_TC=1` is still required on CUDA 12.0 (the cuBLASLt block-scaling APIs are absent).

### Tests

| suite | where | result |
|---|---|---|
| `tests/test_deepseek_v4` (resource plan, sparse attention, core) | shadowfax | pass on all seven branches |
| `dsv4-cuda-tests` (nine CUDA tests) | shadowfax 2×3090 | 9/9 on #2667 and #2670 |
| `test_dsv4_peer_init_cuda`, `test_dsv4_peer_moe_cuda` | shadowfax 2×3090 | pass on #2669 and #2670, including sticky/lost fault injection |
| `test_v4_gpu_placement.py` (mock, no CUDA) | shadowfax | pass on #2665, #2666, #2667, #2670 |
| `test_v4_mirror_guard_source.py`, `test_v4_peer_init_source.py`, `test_dsv4_cuda_lost_source.py`, `test_deepseek_v4_cuda_arch_flags.py` | shadowfax | pass where present |
| `make -j16 check` (full C + Python suite) | computer4 | #2664 3020 pass (379 skipped); #2665 3021 pass; #2666 3021 pass; #2667 3022 pass; #2668 3022 pass; #2669 3032 pass; #2670 3034 pass (375 skipped unless noted) |

The full suite found four defects that the smaller suites missed: two hand-listed headers on `-MMD` test rules reintroduced by the rebase (#2665–#2667, #2670), the missing `dsv4_cuda_wo_decode` export in `dsv4.def` (#2667, #2670), and a misnamed optional-loader slot in the integration's peer best-effort path (#2670). All are fixed on the branches; the full check passes on every branch after the fixes.

### Benchmarks

REAP checkpoint (`deepseek-v4-flash-reap`, 85 GB), `--ram 40 --ctx 8192`; two prompts × 128 generated tokens × 3 repetitions, median; greedy. Every REAP cell produced the same text for a given prompt (`3846c4863341…` and `d740de876936…`).

| config | TTFT (s) | decode (tok/s) |
|---|---:|---:|
| `dev` single | 3.83 | 2.77 |
| #2664 build-fixes single | 3.84 | 3.28 |
| #2665 multigpu-layers single | 3.69 | 3.25 |
| #2666 resident single | 3.69 | 3.30 |
| #2667 cuda-kernels single | 3.73 | 3.29 |
| #2667 cuda-kernels, `DSV4_CUDA_DEVICES=0,1` | 3.75 | 4.33 |
| #2668 mirror-vram-first single | 3.71 | 3.29 |
| #2669 peer-pool, `DSV4_CUDA_PEER_DEVICE=1` | 4.05 | 5.57 |
| #2670 integration single | 3.72 | 3.29 |
| #2670 integration, `DSV4_CUDA_DEVICES=0,1` | 3.75 | 4.34 |
| #2670 integration, `DSV4_CUDA_PEER_DEVICE=1` | 4.04 | 5.58 |

Additional cells:

- **Full checkpoint** (156 GB, 64 generated tokens, one run): single 2.80 / 3.00 tok/s; peer compute 3.42 / 4.03 tok/s; same text in both (`2cd0553c…` / `de8d67d6…`).
- **Prefill** (integration, 1992-token prompt): TTFT 66.0 s, prefill 30.2 tok/s, post-prefill decode 2.50 tok/s.
- **nsys decode** (integration single): `mv_fp4_grouped` 32.0 %, `mv<8>` 25.0 %, `compressor_project_exact` 14.1 %, `fp8_ref_matmul_rows8` 13.8 %, `indexer_head_project` 3.5 %, `head_scores_exact` 1.3 %.

### Defects found and fixed during the round

| PR | fix |
|---|---|
| #2665, #2666, #2667, #2670 | placement mock needed the new `dsv4_cuda_proj_batch_release` stub from current `dev` |
| #2669, #2670 | `dev_release` freed only `ng` slots after a failed init; it now frees all `n` touched devices |
| #2670 | resident MoE entry points (`coli_v4_gpu_moe_resident`, `coli_v4_gpu_resident_route`) were used without declarations; added to `deepseek_v4_internal.h` |
| #2665, #2666, #2667, #2670 | two hand-listed headers reintroduced on `-MMD` test rules (`test_v4_gpu_devices`, `test_v4_bank_pair`); removed |
| #2667, #2670 | `dsv4_cuda_wo_decode` was declared in the header and resolved by the loader but missing from `dsv4.def`; added |
| #2670 | the integration's optional peer best-effort lookup stored `dsv4_cuda_set_peer_best_effort` in a `peer_best_effort` slot; slot renamed to match the symbol |
| #2665–#2667, #2670 | test-only stub and build fixes added as separate commits; feature commit content is unchanged |

Behavioral refusals on #2670 (integration):

- `DSV4_CUDA_DEVICES=0,1` together with `DSV4_CUDA_PEER_DEVICE=1` exits 1 with the mutual-exclusion line.
- `DSV4_CUDA_RESIDENT_EXPERTS=1` on REAP (experts exceed the device 0 budget) exits 1 with the budget line; `auto` keeps the LRU cache and serves.

### CI

At the time of writing, all 116 checks per PR are queued (0 completed, 0 failed). This is queueing, not a failure signal; the local results above are the verification of record.

### State

No merges were performed. #2670 remains a draft until #2667 and #2669 land. `llama-server-active`, `camofox-browser` and `hermes-gateway` are back in service.

