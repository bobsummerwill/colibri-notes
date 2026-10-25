# `test_vk_chain` "hang" on counter's Intel iGPU: a SIMD32 shader compile (2026-10-24, late night)

**Short version:** it isn't a hang in colibri. On the Intel Meteor Lake iGPU (Mesa ANV), the chain pins every pipeline
to the subgroup size the device reports (32). Compiling `chain_moe.spv` at SIMD32 takes ANV **more than 38 minutes**
(never seen to finish). Pinned at 16 the same build takes 72–195 s and the whole suite **passes**. Unpinned (driver's
choice), it builds in 52 s but the results are wrong (the reason the pin exists). Pre-existing on dev; nothing to do
with fix B. Evidence: [manifests/counter-20261024/vk-chain-hang/](../manifests/counter-20261024/vk-chain-hang/).

## Machine
counter: Intel(R) Graphics (MTL), Mesa 26.2.5 (after the 2026-10-24 full upgrade; the first stalls were on 26.2.2),
LLVM 23.1.3. `subgroupSize = 32`, `minSubgroupSize 8`, `maxSubgroupSize 32`, `subgroupSizeControl = true`.
llvmpipe: one size (8), so it is never pinned.

## What the earlier notes got wrong
The handoff said the chain "stalls right after its `test-small-cap` stage". That line was the last **stderr** line;
the test's progress lines go to stdout, which is fully buffered into a log file. Run line-buffered (`stdbuf -oL`),
the test gets much further: through matmul, norm, rope, chunk rows, attention, KV, ... `ares`, `route`, and stops
inside the first `test_moe` (`"fmt 1/1"`, tests/test_vk_chain.c:4097).

## The stall
Under gdb (as the parent, since `ptrace_scope=1`), interrupted after 60 s in the stuck stage
([gdb-pin32-backtrace.log](../manifests/counter-20261024/vk-chain-hang/gdb-pin32-backtrace.log)):

```
#0-#7  libvulkan_intel.so                      (ANV: shader compile)
#8     make_pipe            vk_chain.c:427      vkCreateComputePipelines
#9     moe_pipe_f(bd=0, n=256, fmt=0)  vk_chain.c:2058   chain_moe.spv, every format's pipeline
#10    vkc_moe              vk_chain.c:2299
#11    test_moe("fmt 1/1", D 256, I 88, ...)  tests/test_vk_chain.c:1815
```

The main thread at 100% CPU and no GPU wait was the driver's compiler.

## A/B on the iGPU (shader cache off: `MESA_SHADER_CACHE_DISABLE=true`)
| Run | Pinned size | `chain_moe.spv` builds | Result |
|---|---|---|---|
| dev + initializers (597b49a4) | 32 (default) | first build not done after ~38 min | killed by the 2400 s timeout (rc 124), 0 FAILs before it |
| same, `COLI_VK_SUBGROUP=0` | none (driver) | 52.3 s | 206 FAILs from the first matmul on (the Iris Xe width bug the pin fixes) |
| scratch patch, `COLI_VK_SUBGROUP=16` | 16 | 74.3, 71.6, 5.8, 194.8, 5.0 s | **PASS**, 0 FAILs, 523 s total |

The scratch patch is one line in `vk_pin_sg` (backend_vulkan.c) that takes a size from `COLI_VK_SUBGROUP`
([pin-n-scratch.patch](../manifests/counter-20261024/vk-chain-hang/pin-n-scratch.patch)). **Now a proper branch:** `fix/vk-subgroup-size-env` (d7879d19, on the fork) validates `n` against the device's range, warns on a bad value, and documents the variable in `docs/ENVIRONMENT.md` and `docs/vulkan.md`. The
host side reads the pinned size only in `make_pipe`; `vk_chain.c:560` (chain_gemv2) needs `subgroupSize >= 16`,
which 16 meets.

## Lavapipe (llvmpipe, what upstream CI uses)
| Tree | Result |
|---|---|
| dev 561966e | PASS, 43 s |
| fix B 97b26c0a | PASS, 46 s |
| `fix/vk-chain-test-initializers` 597b49a4 | PASS (first run, cold cache) |

All three print the same totals: 6325 frames, 7074 ops, 437 matmuls (12 GEMM), 20392 barriers. So fix B and the
initializers branch are clean in the full chain suite too.

## Possible upstream report / fix (Bob's call; nothing opened)
- Report: on Intel ANV (MTL, Mesa 26.2.x) the chain's pin to the reported subgroup size (32) makes
  `chain_moe.spv` take > 38 min to compile. That hits `vk-chain-check`, and presumably any engine whose MoE runs on
  the chain on Intel iGPUs. (The engine path wasn't tried.)
- Fix options, for discussion: pin to 16 on Intel (or to `min(reported, 16)` where the device allows it), or let
  `COLI_VK_SUBGROUP=n` choose a size, as in the scratch patch. Both need checking on the Iris Xe that motivated the
  pin, and SIMD16 vs SIMD32 decode speed should be measured before changing a default.
- The 195 s variant at SIMD16 is still slow. The disk shader cache hides it after the first run.
