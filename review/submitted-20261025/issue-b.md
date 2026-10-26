### Commit or version

61646cbd (measured); `c/backend_vulkan.c` is unchanged from there to dev 77530b49

### Environment

Ubuntu 24.04, 62 GB RAM, RTX 3090 24 GB, pure Vulkan (`COLI_VULKAN=1 DSV4_CUDA=0 COLI_CUDA=0`), gcc 13.3, glibc 2.39. Model: DeepSeek V4 Flash REAP-150B.

### Reproduction steps

1. Serve the REAP-150B checkpoint in pure Vulkan with `--ram 40` and the defaults: upload imports on (`COLI_VK_UP_IMPORT`), and the tier's host pool sized from MemAvailable at start (16.3-16.7 GiB here).
2. Send a 1,992-token prompt, then a 3,005-token one.
3. Watch the engine's anonymous RSS and MemAvailable.

### Expected behavior

The engine's anonymous memory stays near its own plan (`projected=39.61GiB` in the log), and both prompts complete.

### Actual behavior and logs

The server is OOM-killed after the first request, in every run. Its anonymous memory peaks at 45.3-45.6 GiB, 5-6 GiB over the plan, and the host pool (sized at start) leaves no room for that.

**Cause:**
- With upload imports on, the uploader allocates each tier tensor's host image with `posix_memalign` plus `memset`.
- Those aligned allocations pile up in glibc's arenas.
- Two experiments confirm it:
  - With imports off (where the image comes from `calloc`) the anon peak is 40.3 GiB.
  - `GLIBC_TUNABLES=glibc.malloc.mmap_threshold=4194304`, which gives every image its own mmap, brings it to 40.1 GiB and lets the default pool complete, but uploads and decode get slower.

**Workaround:**
- `COLI_VK_TIER_HOST_GB=12`: the runs complete, but MemAvailable still dips to 0.5-1 GiB.
- Or `COLI_VK_UP_IMPORT=0`.

A fix follows in a PR.

Evidence (raw logs): [final15](https://github.com/bobsummerwill/colibri-notes/tree/70b7d97/manifests/final15)
