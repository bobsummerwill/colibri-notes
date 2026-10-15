# deepseek-v4: GPU batched prefill for expert-pruned (REAP) checkpoints + faster generic CUDA kernels

## Problem

Expert-pruned DeepSeek V4 checkpoints such as `puwaer/DeepSeek-V4-Flash-0731-reap-150b`
(132 routed experts) never use the batched GPU prefill path:

```
v4_gpu moe-batch=off (unsupported model geometry)
```

`coli_v4_gpu_moe_batch_union` refuses anything but exactly 256 routed experts, so prefill
falls back to the CPU union (~9 tok/s on a 9950X3D). On non-Blackwell cards (generic kernels)
the batched path was also bound by the per-route fp4 matvec and the shared-memory reduction
tree rather than by memory bandwidth.

## Changes

**Pruned checkpoints (`c/deepseek_v4.c`)**
- moe-batch accepts 1..256 routed experts. Counts other than 256 route on the CPU through
  the existing bf16 router (`coli_v4_route_bf16`), with the chunk's tokens in parallel
  (OpenMP). Each token's dot products keep their serial order, so routing is bit-identical
  to the CPU union and to decode. The GPU top-6 kernel still handles 256-expert models.
- The f32 gate mirror (256-only) is required only when routing on the GPU.
- Loops that assumed experts 0..255 use `n_routed_experts`.

**Bank sizing (`backend_cuda_dsv4.{cu,h}`, `backend_loader_dsv4.c`)**
- New `dsv4_cuda_expert_bank_slots(experts)`: generic builds size the bank to the model
  (132 slots ≈ 1.6 GB instead of 3.2 GB), DeepGEMM builds keep 256. The Windows loader
  resolves it as an optional symbol (older DLLs → 256).

**Generic kernels (bit-identical)**
- `mv_fp4_expert_major` (`DSV4_MOE_EXPERT_MAJOR`, default on): one warp per expert row
  walks every route of the chunk that picked that expert, so expert weights are read once
  per chunk instead of once per route. Lane l holds the 256-wide tree's partials
  j = l + 32q, so every output has `mv_fp4_grouped`'s operands in its order.
- `mm_batch_w` (`DSV4_MM_WARP`, default auto = I ≤ 1024): the same warp-per-row
  layout for the dense batch matmul, plus an exact e4m3 lookup table. It is only enabled
  where it measured faster on real prefill (`attn.wq_b`: 6.2 s → 3.6 s per 2k prompt);
  slower on the 2048/4096-row shapes, so those keep `mm_batch`.
- `DSV4_MOE_EXPERT_MAJOR_VERIFY=1` / `DSV4_MM_WARP_VERIFY=1` run old and new side by side
  and count mismatching bits.

**Fixes**
- With `COLI_CUDA_MOE_DOUBLE=1`, the default `DSV4_CUDA_VRAM_RESERVE_MB` now also reserves
  the prefetch bank. Without this, the mirrors grown during decode leave no room and the
  second bank never allocates after a short first request.
- A failed bank `cudaMalloc` left the runtime's sticky error set, so the next
  `cudaGetLastError()` launch check failed an unrelated chunk
  (`generic batched MoE launch: out of memory`). The error is now cleared.

**Diagnostics / docs**
- `V4_MOE_BATCH_PROF=1`: per chunk-layer `moebprof` route / refill / compute ms.
- `docs/deepseek-v4.md`: new subsection on pruned checkpoints and generic kernels, env table
  rows, measurements.

## Results

2× RTX 3090 (one used), sm_86, CUDA 12.0 `NO_TC=1`, Ryzen 9 9950X3D, 62 GB RAM (`--ram 40`),
one SN850X, REAP 150B, greedy, `--ctx 8192`, cold server per run:

| prompt | CPU union (before) | GPU batch, per-route kernels | + expert-major + warp dense | + `COLI_CUDA_MOE_DOUBLE=1` |
|---|---|---|---|---|
| 1,992 tokens | 220.5 s | 125.4 s | 79.7 s | **69.1 s** |
| ~3,000 tokens | 334.4 s | 183.5 s | 118.2 s | **105.8 s** |

- **Correctness:** answers were identical to the CPU union in every run (greedy).
  - The verify modes reported 0 mismatching bits over 600+ chunk-layers × 2 runs for the MoE
    kernel and 10,000 dense calls for `mm_batch_w`.
- **Decode:** unchanged, 2.2–2.9 tok/s (disk-bound). With `COLI_CUDA_MOE_DOUBLE=1` it is a few
  percent lower because of the larger reserve.
- **Kernel time** (Nsight Systems, one 2k prefill): the MoE fp4 kernels went from ~65 s
  (per-route) to ~22 s.

## Testing

- Linux CUDA build (`CUDA=1 CUDA_ARCH=sm_86 NO_TC=1`, CUDA 12.0) and CPU-only build both compile
  without warnings. `deepseek-v4-test-objs` builds and `deepseek-v4-test-registry` passes.
- **Not tested:**
  - DeepGEMM builds (no sm_120 card here). The DeepGEMM path keeps a 256-slot bank and its
    own kernels; only the shared host-side changes apply.
  - Windows DLL loader (no MinGW here). The change mirrors the existing optional-symbol
    pattern (`expert_bank_upload_aux`).
  - A 256-expert checkpoint. Its routing path is unchanged; the generic kernel rewrites are
    bit-identical by construction and verified on the 132-expert model.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
