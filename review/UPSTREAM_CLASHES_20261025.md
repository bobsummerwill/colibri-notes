# Open upstream PRs that touch the same code as ours (2026-10-25)

Checked against JustVugg/colibri dev **77530b49** (2026-10-25) and the 73 open upstream PRs. For each open PR, the
files it changes were compared with the files our five branches change (PR #2, fix A, fix B, the initializers, the
subgroup env). Most hits are only `c/Makefile` or `docs/ENVIRONMENT.md` rows. Those merge line by line and are left
out here. Four PRs overlap in substance. All four are **stale**: 1,000 or more commits behind dev and already
conflicting with dev itself, so a textual trial merge says little. What follows is how each relates to our branches.
Evidence: `gh pr view N --comments`, and the PR heads fetched locally as `uppr/N`
(1847 86fee2ee, 1766 ca55947c, 1815 9008c336, 1818 ff50593b).

## #1847 (crichalchemist): "vulkan: let the subgroup GEMVs read the device's real subgroup size" vs `fix/vk-subgroup-size-env`

- **#1847's change:** it adds `ALLOW_VARYING_SUBGROUP_SIZE | REQUIRE_FULL_SUBGROUPS` to qmatmul, gate_up,
  attention_absorb, the expert batch and the chain's GEMVs/MLA, and turns on `computeFullSubgroups`. The driver then
  reports the width it really uses. The aim is MoltenVK on AMD GCN, which reports 32 where Metal's SIMD-group is 64.
- **Dev went the other way after it:** #1954 (41cf9397) and #2046 (8ff0691e) **pin** the subgroup size the device
  reports.
  - `vk_pin_sg` is at backend_vulkan.c:506. It is applied through `VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT`
    in the chain's `make_pipe` (vk_chain.c:413), in `build_pipeline` (`pin_info`, :873/:905) and in the expert batch
    (:4228), on both devices.
  - `COLI_VK_SUBGROUP` is **dev's existing knob** (`0` = the driver's choice).
- **Our branch:** only lets `COLI_VK_SUBGROUP=n` choose the pinned value. It is one hunk inside `vk_pin_sg`, a function
  #1847 doesn't touch.
- **Do they interact:** the two approaches can't both apply to one pipeline, because the Vulkan spec forbids
  `ALLOW_VARYING` together with a required size. A rebased #1847 would have to vary only where nothing is pinned. Ours
  changes only the pinned value, so it neither replaces #1847 nor is replaced by it.
- **Status:** JustVugg said on 10-03 that #1847 could merge once the chain was extended. The author did that on 10-05.
  Since then there has been no activity, it conflicts, and dev's pinning (10-06, 10-13) partly supersedes it.
- **Action taken:** our commit title, its comment and both docs rows said "the chain's pipelines". Since #2046 the pin
  covers every pipeline that takes it, so the wording was corrected in the rebased commit.
- **For the PR description:** "Related: #1847 (ALLOW_VARYING, for MoltenVK) predates #1954/#2046's pinning. Its flags
  can't coexist with a required size; this PR only changes the pinned value."

## #1766 / #1815 / #1818 (ZacharyZcR): DeepSeek V4 multi-GPU vs PR #2 (and fix A)

### #1766 vs PR #2
- **Different features.**
  - #1766 places contiguous layer ranges on N GPUs, one owner per layer (`DSV4_CUDA_DEVICES=0,1,…`).
  - By default it makes every expert resident (`DSV4_CUDA_RESIDENT_EXPERTS=auto`) when the cards together hold the
    model. Its target is 6× RTX 5090.
  - PR #2 keeps one primary GPU. It adds the second GPU as a spill tier of expert mirrors that computes its own experts
    over NVLink (`DSV4_CUDA_PEER_DEVICE`), for a model that doesn't fit (2× 3090).
- **Env vars:** no new names collide. There are two semantic clashes to settle if both ever land:
  - `DSV4_CUDA_DEVICES=0,1` and `DSV4_CUDA_PEER_DEVICE=1` would both claim GPU 1. They should be mutually exclusive.
  - #1766 reads `DSV4_CUDA_VRAM_RESERVE_MB` with a flat default of 2800 MiB, for preload. On dev and in PR #2 it is the
    mirror-growth reserve, sized from the bank or 600 MiB (#2264).
- **Code:** both rewrite `coli_v4_gpu_engine_open`. #1766 also makes the same move of the guard's statics into each
  cache as PR #2's a28ba9d5 (now 81d9e2fb; uppr/1766 c/deepseek_v4.c:11609-11626).

### #1815 and #1818
- **#1815:** speculative execution on the device and `V4_DEVICE_TARGET`. It is stacked on #1766 and doesn't touch the
  guard or the reserve.
- **#1818:** `V4_GPU_LAYERS` and a CPU tail cache. It edits code near `hot_oldest_pool_victim` and `hot_repin_locked`,
  which PR #2's VRAM-first commit also edits, so expect a textual conflict and a mild overlap in meaning. Its CPU
  kernels already landed as #2069.

### Status and verdict
- **Maintainer status:** for #1766, JustVugg said "With that table it can go in" (10-01): a same-machine dev-vs-branch
  tok/s table on 1 and N GPUs. On 10-14 the author said they don't have the GPUs for it. Nothing has happened since,
  so it is stalled, not rejected. #1815 "lands after #1766".
- **Fix A:** no logical conflict with any of the three.
- **Order and split:** neither changes. There is no reason to wait for #1766. If PR #2 is split, landing part 1 (the
  guard) first gives #1766 the per-cache state it would otherwise add itself.
- **For PR #2's description** (a "Related" paragraph):
  > #1766 (open) spreads layers over N GPUs with every expert resident when the cards hold the model
  > (`DSV4_CUDA_DEVICES`). This PR keeps one primary GPU and adds a second as an expert-mirror tier that computes its
  > own experts over NVLink, for a model that doesn't fit (`DSV4_CUDA_PEER_DEVICE`). The names don't overlap; if both
  > land they should be mutually exclusive. The per-cache guard state here is the same move #1766 makes.
