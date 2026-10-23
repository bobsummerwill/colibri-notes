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

## 6. final6 failures, looked into (2026-10-23)
- **QB (pure Vulkan, PR #1) did not crash: the kernel's OOM killer killed it** (`journalctl`: "colibri-final6-serve.service:
  A process of this unit has been killed by the OOM killer", global OOM, `deepseek_v4` anon-rss 44.1 GB + file-rss 17.1 GB
  on the 62 GB box). The server log's last line is the gateway's "colibri engine is shutting down", so prompt 2 got
  "connection refused". Cause, as read in upstream's `c/vk_tier.c` `host_init` (unchanged by PR #1, which touches no
  Vulkan file): the Vulkan tier's host pool is sized from **`MemAvailable` at start, less 4 GiB**, beside the engine's
  `--ram 40` cache. QA (dev) started with less free RAM and got 898 experts / 13.25 GiB; QB, an hour later, 1,058 /
  15.75 GiB, and the two together no longer fit. So QB's failure is a property of when it started, not of its code;
  to be confirmed by running dev with QB's pool (`COLI_VK_TIER_HOST_GB=15.75`) — final7.
- **Vulkan texts may not be reproducible at all, dev against dev.** The tier chooses per step which cold experts go to
  the device and which stay on the CPU from timings it measures at run time ("cold experts with N rows or more go to the
  device (an upload of 15.0 MiB at 3.2 GB/s takes 4.98 ms, the CPU 0.60 ms a row ...)"): QA's threshold was 23-30 rows,
  QB's 16-19. CPU and GPU sum in different orders, so a different split gives different bits and a different greedy text.
  The same applies to VA/VB (Vulkan + CUDA). Hypothesis, to be tested in final7: repeats of one arm differ, and with the
  split pinned (`COLI_VK_TIER_STREAM_ROWS`, `COLI_VK_TIER_BALANCE=0`, a fixed `COLI_VK_TIER_HOST_GB`) dev = PR #1.
- **AK (dev with the 256-expert bank) failed once, not always.** `[DSV4 CUDA] expert FC2 bank allocation: out of memory`
  -> `moe-batch=off` came during prompt 2 of AK s1; prompt 1 had the bank (TTFT 129.1 s, as AS's and MA's). AS, KA and MA
  ran dev's bank in the same configuration without a failure. dev reserves no VRAM for the bank, so whether it fits
  depends on how many decode mirrors are already up: marginal at 24 GB, which is what PR #1's reserve (4,681 MiB here:
  256-slot bank 3,264 MiB + 600 MiB headroom + 64 mirrors) is for. AK s2/s3 were skipped by the runner's rule; BK/AK on
  prompt 1 (0.665) and KB/KA (0.690) are like-for-like, prompt 2's 0.481 is not.
- **F18: the fault is caught, but device 0 does not survive it.** `test_f18_trap_cuda` (debug build) runs a `__trap()`
  on the peer in pass 2 of a decode MoE, before or after the down-projection. Both times `dsv4_cuda_moe` returned 0
  (refused), the peer latched off, no hang. Then the probe's own check of the primary ("AFTER all-on-primary call")
  **also returned 0** with "MoE activation upload: unspecified launch failure": a launch failure is sticky for the whole
  process's CUDA, not one device's context. So after a peer fault the engine falls back to host copies for the rest of
  the process; it does not "continue on device 0" as PR #2's log line says. The probe printed "every fault handled"
  because its verdict ignores the AFTER line. Open: what a real server does then (final7: a fault mid-request).
- **HY** (`DSV4_HYBRID=1` on PR #2 with everything on) differs from D0 on decode_0, as in final5. `DSV4_HYBRID` is dev's
  own switch (`deepseek_v4.c`, also at 7df8c27c), so the likely cause is dev's hybrid path, not PR #2; untested — final7
  runs it on dev too.
