# PR #2: what's in it, and could it be split? (2026-10-24, late night) — analysis only, nothing decided

From a side question during the session. Commits are on the fork's `docs5/pr2-two-gpu-decode` at 3ce988da (the
linear rewrite). **No trial split has been done**, so how cleanly the split points separate in the code is unchecked.

PR #2 adds **two-GPU decode** for DeepSeek V4 on CUDA: experts are mirrored on a second GPU and computed there.
Around that it bundles a single-GPU speedup, a VRAM-guard fix, instrumentation, and fault handling.

| # | Commit | What it does | Kind |
|---|---|---|---|
| 1 | d84ab940 | `DSV4_CUDA_PIN_CAP_GB` caps the page-locked expert memory (default unchanged, 20 GiB) | small standalone setting |
| 2 | d796d4cc | `V4_SERVE_STATS_LOG=1` prints per-turn decode stats and MoE counters | instrumentation, used for the measurements |
| 3 | d75d9639 | `V4_GPU_RESIDENT_FIRST=1`: an expert already in VRAM skips the RAM-store lookup | single-GPU feature, about +2–3% decode |
| 4 | a28ba9d5 | Each mirror cache keeps its own growth-guard state, with one shared LRU clock | refactor needed for a second cache |
| 5 | b9b1f0ac | Mirror caches keep their VRAM reserve between probes and drop failed uploads | **bug fix, active with every switch off** (the same fix as fix A) |
| 6 | 84d901fd | `DSV4_CUDA_PEER_DEVICE=N`: a second mirror pool on GPU N | two-GPU, part 1 |
| 7 | 002fcadd | `DSV4_CUDA_PEER_COMPUTE`: experts mirrored on GPU N are computed there, bit-identically | two-GPU, part 2: the +40% |
| 8 | 40693de7 | Docs for the two-GPU decode and VRAM-first lookups | docs |
| 9 | 9c3f633c | A sticky CUDA fault turns CUDA off for the process; the engine continues on the host (F18) | fault handling |
| 10 | a9f2b075 | A failed decode block names its stage and keeps the inner error | error messages |
| 11 | 3ce988da | The attention's grouped `wo_a` matvec falls back to the CPU when the backend refuses it (F18) | fault handling |

## Are they all related and necessary?
Not all of them are needed for two-GPU decode.
- **Needed for it:** 4 → 6 → 7 are the core chain. 9 and 11 belong with it, because a fault on the peer GPU is what
  made F18 matter. 8 documents it.
- **Not needed for it:**
  - 1 and 2 stand alone.
  - 3 is a separate single-GPU feature, although commit 7 builds on its code in `moe_token_pipeline`. That's the
    test coupling the rewrite hit (the VRAM-first test expected the second `v4_views_fetch_host` that 7 adds), so 3
    has to land before 7.
  - 5 is a general bug fix that fix A duplicates.
  - 10 is a general improvement.
  - 9's latch also helps with any sticky fault, not only peer faults.

## Could it be split?
Yes, and upstream's "keep changes focused" guidance favours it; PR #0 was split into six PRs for that reason
([decisions.md](../decisions.md)). One possible stack:

1. **Fix A**, or commits 4 and 5 as the guard PR. Fix A is the smaller form; if it goes first, PR #2 keeps 4 and only
   the "drop failed uploads" part of 5.
2. **`PIN_CAP_GB`** (1) and **`SERVE_STATS_LOG`** (2), together or as two tiny PRs.
3. **VRAM-first lookups** (3). The measurements already isolate it: G1 → R1, about 1.02–1.03x (final6).
4. **Two-GPU decode** (6, 7, 8) plus F18 (9, 11) and 10. The peer step is also measured on its own: R1 → X3, about
   1.36–1.48x (final6).

**Costs:** stacked PRs have to merge in order, each part needs its own description, and each split point has to be
checked to build and test on its own. Commit 3's test shows the parts touch the same functions.

## To decide / to do
- **Bob:** split or not. If split, which stack.
- **If split:** trial split on counter (cherry-pick each part onto dev in its own worktree; per-commit check and
  `make check` at each part's tip; then check that the parts stacked in order reproduce 3ce988da's tree). No GPU
  needed for that. The GPU runs then belong to the parts that touch CUDA behaviour (fix A / guard, two-GPU).
