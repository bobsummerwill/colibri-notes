# Counter iGPU / CPU sanity for the head-followups branch (2026-11-03)

- counter: Intel Core Ultra 7 165U, Intel Graphics (Meteor Lake), Mesa 26.2.5, 30 GB RAM.
- Branch `fix/dsv4-head-followups` at `f4586dca`, in `~/colibri-wt/headfix` (worktree; the
  main `~/Projects/colibri` has unrelated shader edits and was not touched).
- The changed code is CUDA-specific, so the iGPU cannot exercise it. The useful check is the
  non-CUDA build and the V4 tiny token-exact oracle; the iGPU itself is a sanity check that the
  Vulkan build still starts and produces the CPU's tokens.

## CPU-only build + tiny V4 oracle

`make -C c -f Makefile.deepseek-v4 deepseek-v4 -j8` succeeded. The generated tiny fixture
(`c/tools/make_deepseek_v4_tiny.py`, Transformers 5.14.1 + torch 2.13 CPU) was written to
`~/colibri-bench/v4tiny-headfix` because the safetensors are intentionally not committed.

`test_deepseek_v4_tiny.py --binary c/deepseek_v4 --fixture ~/colibri-bench/v4tiny-headfix`:
**all checks pass** (`rc=0`):

- short/compressed/long teacher-forcing and greedy token-exact
- truncated-prefix rejection
- three short sessions + one long session exact
- CLI TUNE line and 768-token context
- persistent serve protocol token-exact, PROF phases filled
- `SKIP mtp draft` twice: the tiny fixture has no 3-stage MTP profile (not a failure)

Raw log: [v4tiny-test.log](v4tiny-test.log).

## Vulkan build + iGPU run

`make -C c -f Makefile.deepseek-v4 deepseek-v4 VK=1 -j8` succeeded (Intel ANV/Mesa at run time).
A 4-token raw-prompt run with `COLI_VULKAN=1` initialized the MTL device, put 66/66 routed experts
on the device, and produced the same tokens as the CPU build:

```text
CPU:   <t024><t025><t026><t027>
iGPU:  <t024><t025><t026><t027>
```

Only the TUNE timing differs (0.018 s CPU vs 0.028 s iGPU for 4 tokens). No device loss or Vulkan
error. Raw output: [v4tiny-cpu.out](v4tiny-cpu.out), [v4tiny-vk.out](v4tiny-vk.out).

Caveats: this is a tiny fixture, not a V4 Flash performance run (the full model does not fit the
30 GB host). It does not exercise the CUDA-only changes; it confirms the branch still builds and
runs the non-CUDA paths, and that the iGPU Vulkan path is healthy.
