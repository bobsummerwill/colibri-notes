# Fix A and fix B: PR descriptions (drafts, 2026-10-24 evening) — NOT submitted; Bob decides (issue or PR)

> From [UPSTREAM_FIXES_20261024.md](UPSTREAM_FIXES_20261024.md) (evidence on dev 61646cbd). Fix A's description absorbs
> issue 1 of [UPSTREAM_ISSUES_20261023_DRAFT.md](UPSTREAM_ISSUES_20261023_DRAFT.md) (the kv comp fallback), which it
> supersedes. Both branches now have dev 561966ec merged in (counter, 2026-10-24; pushed to the fork):
> - fix A `fix/dsv4-mirror-guard-overshoot` **f30c7b23** (bd7103fd + a merge that resolves a conflict with #2264's
>   bank-sized reserve, below)
> - fix B `fix/vk-host-image-alloc` **97b26c0a** (72aa325a + a clean merge)
>
> JustVugg did not answer the offer of both on #2264. Fix B is independent of everything else; fix A overlaps PR #2
> (its b9b1f0ac (was 6b7fc6c0) contains the same guard fix), see the note at the end.

---

## Fix A

**Title:** deepseek-v4: the decode mirror cache keeps its VRAM reserve between free-VRAM probes

### Summary
With the default serve line (no `COLI_CUDA_MOE_BATCH`) on the full DeepSeek V4 Flash, the decode mirror cache grows
past its VRAM reserve. The compressed-KV buffer then cannot grow, those layers' attention falls back to the CPU, and
greedy texts can change from run to run.

**Cause.** The mirror cache's growth guard re-reads free VRAM only every 64 misses and reuses the last answer in
between. Measured against that stale answer alone, the cache grows up to 64 mirrors (816 MiB) past an explicit
`DSV4_CUDA_VRAM_RESERVE_MB` or past the 600 MiB default without the bank, which is past the card's free VRAM. Once the
first answer's decode has grown the cache, `dsv4_cuda_kv_comp_append` can no longer double the compressed-KV buffer
(`[DSV4 CUDA] kv comp allocation: out of memory`, ~420 times per two-prompt run), `coli_v4_gpu_kv_cache_sync`
returns 0, and that layer's attention runs on the CPU. CPU and CUDA attention do not give the same bits (with
`COLI_CUDA_ATTN_BATCH=0` both prompts' texts differ from the CUDA ones), so when the fallback starts during a
prefill the greedy text changes. How often it fires varies from run to run: on an RTX 5070, 0 to 1,066 fallbacks per
sample in one configuration, and 2 of 7 samples changed the second prompt's answer. It also costs time, since that
attention runs on the CPU.

**Fix.** The guard also counts the mirrors appended since the last probe (a recycled slot reuses its VRAM and adds
nothing; a failed upload clears the count along with the probe). From 8 live mirrors on, it stops growing once the
last probe less those mirrors would fall below the reserve less its margin. The margin
(`v4_gpu_mirror_reserve_margin_mb`) is the 64 mirrors that the bank-sized default (`v4_gpu_mirror_reserve_mb`)
already carries, so that default's decisions do not change; an explicit reserve and the 600 MiB headroom-only default
carry none, and now stop at the reserve. Below 8 live mirrors only the old test applies, so a reserve just under the
card's free VRAM still leaves the cache a set it can recycle.

Workaround without this change: `DSV4_CUDA_VRAM_RESERVE_MB=2000` (0 fallbacks in 4 samples, reference texts, normal
TTFT).

### Measurements
Full DeepSeek-V4-Flash-0731, default configuration, two fixed prompts (1,992 and 3,005 tokens), greedy, `--ram 40
--ctx 8192`, a fresh server per sample, interleaved. Measured on dev 61646cbd and bd7103fd (this branch before the
merge of dev; the no-bank path it measures is unchanged by the merge).

RTX 3090 24 GB (W A F F A A F):

| | dev (x3) | this PR (x3) |
|---|---|---|
| handled allocation failures per sample | 421 (420 kv comp, 1 weight), every sample | **0** |
| texts | | identical to dev's |
| 1,992-token TTFT | 161.15 s | 161.10 s (0.9997x) |
| 3,005-token TTFT | 271.05 s | **256.47 s (0.946x)** |
| answer 2 decode | 1.850 tok/s | **1.946 tok/s (+5.2%)** |
| GPU 0 peak | 24,012 MiB | 23,608 MiB |

RTX 5070 12 GB, DeepGEMM build (W + 5 interleaved pairs, medians):

| | dev (x5) | this PR (x5) |
|---|---|---|
| handled allocation failures per sample | 399-420 kv comp + 1 weight, every sample | **0** |
| texts | | identical to dev's |
| 1,992-token TTFT | 243.79 s | 244.88 s (1.004x) |
| 3,005-token TTFT | 411.41 s | **377.88 s (0.918x)** |
| answer 2 decode | 1.227 tok/s | **1.307 tok/s (+6.6%)** |
| GPU 0 peak | 11,745 MiB | 11,239 MiB |

Prompt 2 is faster because its attention now stays on the GPU. In these runs dev's fallbacks happened not to change
the texts (earlier, on the 5070, 2 of 7 runs did). One 5070 sample of this PR overlapped a `make check` on the same
machine and is left out of the medians.

### How this was tested
- New `tests/test_v4_mirror_guard_source.py` pins the guard at source level (no fixture reaches a full card).
- On the merged tip f30c7b23 (counter, gcc 16.2.1): `make -C c check` 2,733 tests OK (312 skipped); `deepseek_v4.c` compiles with
  `-Wall -Wextra -DCOLI_V4_GPU_TIER` (and `-DCOLI_VULKAN`), 0 diagnostics. GPU runs as above, on 61646cbd.

### Validation
- [x] `make -C c check`: 2733 tests, OK (312 skipped)
- [x] Performance claims include hardware, commands and repeatable measurements
- [x] Experiment manifests with raw evidence ([final10](https://github.com/bobsummerwill/colibri-notes/tree/70b7d97/manifests/final10), [c4fa](https://github.com/bobsummerwill/colibri-notes/tree/70b7d97/manifests/c4fa))

---

## Fix B

**Title:** vulkan: the uploader's host images align inside a calloc'd block, not posix_memalign

### Summary
In pure Vulkan with the default host pool, the server is OOM-killed after the first request on a 62 GB box (RTX 3090,
REAP-150B, `--ram 40`), every run. The engine's anonymous memory overshoots its own plan (`projected=39.61GiB`) by
~5-6 GiB, and the host pool, sized from MemAvailable at start, has no room for that.

**Cause.** With upload imports on (`COLI_VK_UP_IMPORT`, the default), the uploader allocates each tier tensor's host
image with `posix_memalign` + `memset`; without imports it uses `calloc`. The aligned allocations pile up in glibc's
arenas: with imports off, anon peak drops from 45.3 to 40.3 GiB, and a fixed malloc mmap threshold
(`GLIBC_TUNABLES=glibc.malloc.mmap_threshold=4194304`, every image its own mmap) also brings it to 40.1 GiB and lets
the default pool complete, but slows uploads and decode.

**Fix.** Off Windows, the image is aligned by hand inside a `calloc`'d block (the block's address sits just before the
image), as the path without imports allocates, and `up_img_free` frees that block. Windows keeps `_aligned_malloc`.

### Measurements
RTX 3090, pure Vulkan (`COLI_VULKAN=1 DSV4_CUDA=0 COLI_CUDA=0`), imports on, REAP-150B, 62 GB host, dev 61646cbd vs
the same change on it (x3 each unless noted):

| | dev | this PR |
|---|---|---|
| pool 12 GiB: anon peak | 45.4-45.6 GiB | **40.3-40.4 GiB** |
| pool 12 GiB: MemAvailable min | 0.53-0.98 GiB | 4.8-4.9 GiB |
| pool 12 GiB: 1,992-token TTFT | 103.22 s | **98.80 s (0.957x)** |
| pool 12 GiB: 3,005-token TTFT | 139.36 s | 137.26 s (0.985x) |
| pool 12 GiB: decode | 4.347 tok/s | 4.375 (1.007x) |
| **default pool** (16.3-16.7 GiB) | OOM-killed after prompt 1, every run | **both prompts, 2 of 2** (anon 40.3 GiB, MemAvailable min 1.16 GiB) |

Texts: prompt 1 is the CUDA build's in every pool-12 sample of both. Prompt 2 varies run to run on dev itself in pure
Vulkan (the tier's device/CPU split follows the measured upload rate), so texts are not a criterion here. The
default-pool runs still leave only ~1 GiB of MemAvailable on this box, so the pool's 4 GiB margin is tight, but the
5 GiB nobody counted is gone.

### How this was tested
- New `tests/test_vk_host_image_alloc_source.py` pins the allocation and the matching free.
- On the merged tip 97b26c0a (counter, gcc 16.2.1): `make -C c check` 2,732 tests OK (312 skipped). Built with `VK=1`
  (0 warnings in `backend_vulkan.c`); on an Intel Meteor Lake iGPU (Mesa 26.2.2) `vk-memtype-check` and
  `vk-tier-check` pass, the tier suite with the import upload path on, i.e. through this change's allocator.
  (`vk-chain-check` hangs on that iGPU on dev 561966e too, so it says nothing about this change.) GPU runs as above, on
  61646cbd (final15; dev..561966ec does not touch `backend_vulkan.c`'s uploader).

### Validation
- [x] `make -C c check`: 2732 tests, OK (312 skipped)
- [x] Performance claims include hardware, commands and repeatable measurements
- [x] Experiment manifests with raw evidence ([final15](https://github.com/bobsummerwill/colibri-notes/tree/70b7d97/manifests/final15))

---

### Notes for Bob (not part of the descriptions)
- **Fix A vs PR #2, recommended order: fix A first.** It's small (one function, measured on two cards in the default
  configuration) and fixes dev on its own; PR #2's b9b1f0ac (was 6b7fc6c0) contains the same fix in per-cache form. The trial merge
  (`scratch/fixA-x-pr2`, c89353e4; `make check` 2,745 OK, 312 skipped) resolves by keeping PR #2's guard and docs and porting fix A's source test to the
  per-cache fields. One behavioural difference: PR #2 gives the bank-sized default its 64-mirror margin also below 32
  live mirrors (where that default is only the 600 MiB headroom), fix A gives it none there. That only matters on a
  card too small for the bank, and PR #2's comment already says so; whichever lands second takes PR #2's rule. If
  both are open, each description should mention the other.
- The 2026-10-23 issue draft's "possible fixes" (allocate the compressed-KV buffer at its final size; give a mirror
  slot back on a kv comp failure) are alternatives, not done; this fix removes the cause on every card measured.
- **Also checked:** dev 61646cbd..561966ec does not touch the Vulkan uploader (`git diff --stat` on `c/backend_vulkan.c`).
