## Summary

Fixes #2400.

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

## Measurements
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

## How this was tested
- New `tests/test_vk_host_image_alloc_source.py` pins the allocation and the matching free.
- On the tip 78dda872 (counter, gcc 16.2.1): `make -C c check` 2,745 tests OK (320 skipped). Built with `VK=1`
  (0 warnings in `backend_vulkan.c`); on an Intel Meteor Lake iGPU (Mesa 26.2.5) `vk-memtype-check` and
  `vk-tier-check` pass, the tier suite with the import upload path on, i.e. through this change's allocator.
  `tests/test_vk_chain` passes on Lavapipe, as on dev. GPU runs as above, on 61646cbd (final15; 61646cbd..77530b49 does not
  touch `backend_vulkan.c` at all).

## Validation
- [x] `make -C c check`: 2745 tests, OK (320 skipped)
- [x] Performance claims include hardware, commands and repeatable measurements
- [x] Experiment manifests with raw evidence ([final15](https://github.com/bobsummerwill/colibri-notes/tree/70b7d97/manifests/final15))

## Compatibility
- [x] The default CPU build remains dependency-free
- [x] No model files, generated binaries, or benchmark artifacts are included

Developed with AI assistance (Claude Code; see the `Co-Authored-By` trailer); the change was reviewed and tested as described here.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
