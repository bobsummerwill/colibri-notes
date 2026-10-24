# Draft v2: follow-up comment on #2264 (NOT posted; Bob decides)

> Supersedes REPLY_2264_FOLLOWUP_20261024_DRAFT.md: same correction, plus final12-16 (fix B measured, the imports
> note, pure-Vulkan text variance). Evidence links pinned to colibri-notes d7726561. Sections 2-3 are outward-facing
> offers about upstream code outside this PR; keep or drop them.

---

A follow-up from an overnight run on the merged tips (dev 61646cbd, this branch 47c8ec51). The first section corrects
my last comment; the rest is about dev, not this branch, so ignore it if it isn't useful.

## 1. Correction: why the Vulkan + CUDA build's prompt-2 text differs

In section 3 of my [last comment](https://github.com/JustVugg/colibri/pull/2264#issuecomment-6380394089) I guessed that
dev's union prefill "runs some experts through Vulkan" while the bank runs them all on CUDA. That is wrong. In a
Vulkan + CUDA build the Vulkan expert tier stays off ("the CUDA expert tier is on and wins"). What does run on the
Vulkan device is the dense matmul hook (the router's gate logits and the compressor and indexer projections), and it
runs for the union and the bank alike. REAP-150B, 2× RTX 3090, `DSV4_CUDA_VRAM_RESERVE_MB=3100`, one sample per arm:

| arm | prompt 1 | prompt 2 |
|---|---|---|
| dev, CPU union | `6ec1e4b7…` | `1bdc5d70…` |
| this branch, bank | `6ec1e4b7…` | `1f019203…` |
| dev, CPU union, `COLI_VK_DENSE=0` | `32534c06…` | `1f019203…` |
| this branch, bank, `COLI_VK_DENSE=0` | `32534c06…` | `1f019203…` |
| dev's Vulkan build, `COLI_VULKAN=0` | `32534c06…` | `1f019203…` |

`32534c06…` / `1f019203…` are the CUDA-only build's texts. With the dense hook off, the union and the bank agree with
each other and with the CUDA build on both prompts. With it on, prompt 1 changes the same way for both. A build that
logs every hook call showed the union and the bank making the same 1,720 prefill-sized calls on the device, with none
declined and no upload refused. A bigger reserve (`6000`) does not change the bank's prompt-2 text either. So this is
not a fallback and not a VRAM effect.

What is left is the expert arithmetic. The bank computes the routed experts on CUDA and the union computes them on the
CPU, and the two are not bit-identical in general. This branch's identity claims cover its new kernels against the GPU
kernels they replace, not CUDA expert sums against CPU ones. On these prompts the CUDA build happens to agree with the
CPU. Under the hook's slightly different gate logits and projections, prompt 2 lands on a near-tie, and the two expert
paths pick different tokens. dev refuses the bank on this checkpoint, so I have no dev bank arm to compare.
Evidence: [final9](https://github.com/bobsummerwill/colibri-notes/tree/d7726561b1765c6c43d9c9c8ddb5772c058322a7/manifests/final9),
[final9b](https://github.com/bobsummerwill/colibri-notes/tree/d7726561b1765c6c43d9c9c8ddb5772c058322a7/manifests/final9b),
[final11](https://github.com/bobsummerwill/colibri-notes/tree/d7726561b1765c6c43d9c9c8ddb5772c058322a7/manifests/final11).

## 2. Two problems on dev 61646cbd, each with a one-commit fix on a branch

Neither involves this PR. I can open them as issues or PRs if you want them.

**A. The compressed-KV fallbacks in the default configuration** (the ones in section 1 of my last comment). The decode
mirror cache's growth guard re-reads free VRAM only every 64 misses. Between probes the cache can grow up to 64 mirrors
(~816 MiB) past its reserve, which fills the card, and then `dsv4_cuda_kv_comp_append` cannot grow and that layer's
attention runs on the CPU. The fix counts the mirrors appended since the last probe
([`fix/dsv4-mirror-guard-overshoot`](https://github.com/bobsummerwill/colibri/commit/bd7103fd), +32/−3 in
`deepseek_v4.c`, plus a source test). Full V4 Flash, default configuration:

| | handled alloc failures / sample | texts | 3,005-token TTFT | answer-2 decode |
|---|---|---|---|---|
| RTX 3090 24 GB, x3 | 421 → **0** | identical | **0.946×** | +5.2% |
| RTX 5070 12 GB (DeepGEMM), x5 | 399-420 → **0** | identical | **0.918×** | +6.6% |

Prompt 1 is unchanged (1.000× / 1.004×). The bank-sized default reserve already has a 64-mirror margin, so its
decisions don't change.

**B. Pure Vulkan with the default host pool is still OOM-killed** on a 62 GB box (RTX 3090, REAP-150B, `--ram 40`),
after 0f786bfa / 5c0469ae. Every default-pool run died after prompt 1. Even with the pool off, anon memory came to
45.6 GiB against the engine's own plan of 39.6 GiB. The extra ~5 GiB is the uploader's host images when host-page
imports are on. They come from `posix_memalign` + `memset`, and glibc keeps those pages in its arenas. A fixed
`mmap_threshold` removes the growth too, but it slows uploads, so it only confirmed the cause. The fix aligns the
image by hand inside a `calloc`'d block, the same way the no-import path allocates
([`fix/vk-host-image-alloc`](https://github.com/bobsummerwill/colibri/commit/72aa325a), +14/−7 in `backend_vulkan.c`,
plus a source test; Windows keeps `_aligned_malloc`). Pool 12 GiB, x3: anon 45.4 → **40.3 GiB**, MemAvailable min
0.5-1.0 → 4.8 GiB, prompt-1 TTFT **0.957×**. With the default pool, both prompts complete (2 of 2; dev: killed every
time). The default pool still leaves only ~1 GiB free on this box, so the 4 GiB margin is tight, but nothing uncounted
is left.

## 3. Two notes, not fixes

- **`COLI_VK_UP_IMPORT`:** on this RTX 3090, staged uploads beat imports: 6.2-6.6 vs 4.3 GB/s. With fix B applied,
  prompt-1 TTFT is 0.980× and decode +2.6%, with the same memory and the same texts (x3). The best default probably
  depends on the device, so I'm only reporting it.
- **Pure-Vulkan greedy texts vary run to run on dev itself.** Three dev runs gave three different prompt-2 texts. The
  tier's per-expert device/CPU split follows the measured upload rate, so the arithmetic changes with it. Text identity
  can't be a pass/fail check for pure Vulkan, and I didn't use it as one above.

Evidence for sections 2-3: [write-up](https://github.com/bobsummerwill/colibri-notes/blob/d7726561b1765c6c43d9c9c8ddb5772c058322a7/review/UPSTREAM_FIXES_20261024.md),
[final10](https://github.com/bobsummerwill/colibri-notes/tree/d7726561b1765c6c43d9c9c8ddb5772c058322a7/manifests/final10),
[c4fa](https://github.com/bobsummerwill/colibri-notes/tree/d7726561b1765c6c43d9c9c8ddb5772c058322a7/manifests/c4fa),
[final15](https://github.com/bobsummerwill/colibri-notes/tree/d7726561b1765c6c43d9c9c8ddb5772c058322a7/manifests/final15),
[final16](https://github.com/bobsummerwill/colibri-notes/tree/d7726561b1765c6c43d9c9c8ddb5772c058322a7/manifests/final16).
