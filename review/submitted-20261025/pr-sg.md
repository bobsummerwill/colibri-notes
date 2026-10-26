> **Update (2026-10-26):** a decode test with a real model (OLMoE-1B-7B Q4_K_M, `coli serve`, Meteor Lake) found more than the compile time. At the reported 32 the chain **hangs the GPU at run time**: i915 fence timeouts, `the device is lost`, and decode falls back to the CPU at 22.7 tok/s. At `COLI_VK_SUBGROUP=16` the chain runs at 28.5 tok/s and never lost the device. In `coli serve` the compile at 32 took ~21 s, not 38 minutes. Numbers in the [comment below](https://github.com/JustVugg/colibri/pull/2404#issuecomment-6411097645).

## Summary

Since #1954 and #2046, every pipeline that takes a required subgroup size is pinned at the size the device reports: the dense chain's, `build_pipeline`'s (the GEMVs) and the expert batch's, on both devices. On Intel Meteor Lake with Mesa 26.2 (ANV, sizes 8 to 32, reports 32), that size makes `chain_moe.spv` take over 38 minutes to compile in `vkCreateComputePipelines`. It looks as though `test_vk_chain` hangs in its first `test_moe`.

The three possible settings compare like this:

| `COLI_VK_SUBGROUP` | `chain_moe.spv` compile | `test_vk_chain` |
|---|---|---|
| unset (pins the reported 32) | over 38 min | doesn't finish |
| `0` (unpinned) | 52 s | fails: the chain's GEMVs come out wrong (the Iris Xe problem the pin fixes) |
| `16` (this PR) | 72-195 s | passes |

`COLI_VK_SUBGROUP=n` now pins `n`, if it's a power of two within the device's range, on every pipeline that takes the pinned size. Any other value prints a `[VK]` line and keeps the reported size. Unset and `0` behave as before, and no default changes. The change is one hunk in `vk_pin_sg`, plus a row each in `docs/ENVIRONMENT.md` and `docs/vulkan.md`.

**Related:** #1847 (open) predates #1954/#2046's pinning. It makes the subgroup GEMVs use `ALLOW_VARYING_SUBGROUP_SIZE` (for MoltenVK). The Vulkan spec doesn't allow those flags together with a required size, so the two can't apply to the same pipeline. This PR only changes the pinned value and doesn't touch the pipelines #1847 changes.

## Validation
- [x] `make -C c check`: 2742 tests, OK (320 skipped)
- [x] `make VK=1`: no new warnings. `vk-memtype-check` and `vk-tier-check` pass, and `tests/test_vk_chain` passes on Lavapipe (41 s) and on the Meteor Lake iGPU with `COLI_VK_SUBGROUP=16` (240 s, Mesa 26.2.5)

## Compatibility
- [x] The default CPU build remains dependency-free
- [x] No model files, generated binaries, or benchmark artifacts are included

Developed with AI assistance (Claude Code; see the `Co-Authored-By` trailer).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
