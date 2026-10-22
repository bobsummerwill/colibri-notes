# Investigations

> **Naming key.** These notes use internal names for the upstream pull requests. **PR #0** is the six
> fixes opened as JustVugg/colibri [#2212](https://github.com/JustVugg/colibri/pull/2212) (0a), [#2213](https://github.com/JustVugg/colibri/pull/2213) (0b), [#2214](https://github.com/JustVugg/colibri/pull/2214) (0c), [#2215](https://github.com/JustVugg/colibri/pull/2215) (0d),
> [#2216](https://github.com/JustVugg/colibri/pull/2216) (0e) and [#2217](https://github.com/JustVugg/colibri/pull/2217) (0f). **PR #1** (batched GPU prefill for expert-pruned checkpoints) is
> [#2264](https://github.com/JustVugg/colibri/pull/2264), opened 2026-10-20; **PR #2** (two-GPU decode) is not opened yet. Documents dated before 2026-10-17 describe earlier
> versions of these changes, on the old base 7b55a9b3.

## 1. Why prefill was slow on REAP (-> PR #1)
- The REAP checkpoint has 132 routed experts per layer instead of 256. The upstream GPU batched-prefill path assumed the 256-expert layout, so REAP fell back to slower per-token MoE work (~9 tok/s prefill).
- PR #1 makes batched GPU prefill (`coli_v4_gpu_moe_batch_union`, expert bank) work for expert-pruned checkpoints, with route-aware refill.
- With `V4_MOE_BATCH_PROF=1` the remaining time was split between expert refill (host->GPU uploads, PCIe x8 ~12 GB/s) and MoE compute. That led to:
  - **Expert-major MoE kernel**, warp-per-row. Lane l holds partials j = l + 32q, reduced with in-lane tree steps (128/64/32) and then `__shfl_down_sync`, which reproduces the reference summation order bit for bit. The first version was reduction-bound (`tools/microbench/`). The v3 warp-per-row design fixed that. Shared tokens are staged in shared memory once per block.
  - **Warp dense kernel** for fp8 matmuls. It won on microbenchmarks but nsys showed it was slower on the 2048/4096-row shapes in production (21.8 s vs 20.7 s), so it is limited to short rows (I <= 1024, e.g. `attn.wq_b`).
  - **Double bank** (`COLI_CUDA_MOE_DOUBLE=1`): a prefetch worker on the aux stream fills the next bank while the current one computes. Problems found and fixed along the way:
    - the reserve was too small, so it ran out of memory;
    - the sticky `cudaMalloc` error was not cleared, so it broke the next launch (fixed with `cudaGetLastError()`);
    - it caused nondeterminism (see 3).

## 2. Why decode was slow, and making GPU 1 useful (-> PR #2)
- The DeepSeek V4 engine used one GPU. GPU 1 sat idle (24 GB unused) while llama.cpp was down.
- Decode profile: per token MoE ~249 ms and attention ~107 ms. Inside MoE the time was mostly expert attach/upload and fetch wait, not compute.
- `--ram 48` gave only +5%, so the extra VRAM had to act as a real cache tier rather than more RAM. The RAM tier covers ~44% of experts; two GPUs' VRAM can cover ~90%.
- Steps:
  - VRAM-first lookups (X1);
  - a peer mirror pool on GPU 1 read in place over NVLink (X2);
  - peer compute, so GPU 1 runs its own experts concurrently and returns partial outputs (X3).
- The NVLink bridge (P2P OK) makes this practical. Without it, 3090s have no P2P.

## 3. The upload race (upstream bug -> PR #0, commit a406901b)
- Symptom: with the double bank, 2 of 3 cold runs diverged from the reference text at some token.
- Method:
  - debug builds printing per-token logit hashes and hidden-state hashes (`tools/probes/dbg_patch*.py`, `cmp_logits.py`, logs in `evidence/determinism/`);
  - bisecting the divergence point;
  - `CUDA_LAUNCH_BLOCKING=1` made the divergence disappear, which pointed at an ordering race.
- Root cause:
  - upstream used legacy-stream `cudaMemcpy` from **pageable** host memory for several uploads (expert table, hash router, KV RoPE tables, tensor refill);
  - that call can return before the DMA has finished;
  - it is not ordered against the engine's non-blocking compute stream;
  - so kernels could read half-uploaded data.
- Fix: a helper `h2d()` that does `cudaMemcpyAsync(..., c->stream)` and then `cudaStreamSynchronize`, used for every upload. The race also exists on upstream dev for decode, so it is a standalone PR #0 fix.
- Verification: all configurations bit-identical at every token. Still to do: show the drift on plain dev and that PR #0 removes it.

## 4. Adversarial review
- 214 agents found 60 findings (F1-F60), each voted on by independent verifiers. Report: `review/REVIEW_REPORT.md`, briefing: `review/REVIEW_CONTEXT.md`.
- All PR #1 and PR #2 findings were fixed in the `fix1/*` and `fix2/*` groups, then folded into `pr1/fixed` and `pr2/fixed`.
- Seven of the findings turned out to be bugs already on upstream dev. They became PR #0:
  1. h2d ordering (the upload race);
  2. sticky error after a failed bank allocation;
  3. bank not released on every `generate()` exit or at engine close (F10/F11);
  4. missing bias mirror, so the GPU union is refused on dev (F23);
  5. shutdown leaves dangling pointers (F28);
  6. `dsv4.def` missing `dsv4_cuda_device_unified`, plus the parity test;
  7. loader jobs not finished on every MoE exit (F26).

  PR #0 also carries a doc fix (mirror size 12.75 MiB, bank sizes).

## 5. Side questions answered
- GLM-5.2 (744B, 429 GB) was judged not worth downloading for practical use. colibri's GLM engine can use both GPUs (`COLI_GPUS=0,1`), but at that size it would be far slower than llama.cpp + Qwen.
- Logprobs are not supported for the dsv4 engine, so determinism probes use debug fprintf builds instead.
