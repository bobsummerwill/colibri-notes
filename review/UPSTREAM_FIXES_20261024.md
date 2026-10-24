# Upstream fixes, 2026-10-24 (branches only: no issues, no PRs; Bob decides)

The two upstream problems from [UPSTREAM_ISSUES_20261023_DRAFT.md](UPSTREAM_ISSUES_20261023_DRAFT.md), each on its own
branch off upstream dev 61646cbd in the fork (bobsummerwill/colibri), plus what was learned on the way.

## Fix A: the decode mirror cache overshoots its VRAM reserve, and default-config texts change

**Branch** `fix/dsv4-mirror-guard-overshoot`, one commit **bd7103fd** on dev 61646cbd (pushed to the fork).
`c/deepseek_v4.c` (+32 / -3), `docs/deepseek-v4.md`, new `c/tests/test_v4_mirror_guard_source.py`. `make check` OK.

**Cause.** The decode mirror cache's growth guard re-reads free VRAM only every 64 misses and reuses the last answer
in between. Measured against that stale answer alone, the cache grows up to 64 mirrors (816 MiB) past an explicit
`DSV4_CUDA_VRAM_RESERVE_MB` or past the 600 MiB default without the bank. That puts it past the card's free VRAM. On
the full V4 Flash at 24 GB, once the first answer's decode has grown the cache, the compressed-KV buffer cannot grow
(`[DSV4 CUDA] kv comp allocation: out of memory`), `coli_v4_gpu_kv_cache_sync` returns 0, and those layers' attention
runs on the CPU. CPU and CUDA attention do not give the same bits, so a greedy text can change (on the RTX 5070, 2 of
7 runs changed prompt 2's answer). The same defect is fixed inside PR #2's 6b7fc6c0 (F16), but nowhere on dev.

**Fix.** The guard also counts the mirrors appended since the last probe. From 8 live mirrors on, it stops growing
once the last probe less those mirrors would fall below the reserve less its probe margin. The bank-sized default
(2800 MiB) already carries a 64-mirror margin, so its decisions do not change. Below 8 live mirrors only the old test
applies.

**Measured** (final10, shadowfax RTX 3090 24 GB, full DeepSeek-V4-Flash-0731, default configuration, W A F F A A F;
[manifests/final10](../manifests/final10/summary.txt)):

| | dev 61646cbd (A, x3) | fix A bd7103fd (F, x3) |
|---|---|---|
| handled allocation failures per sample | 421 (420 kv comp, 1 weight), every sample | **0** |
| texts | `a397814f` / `8b3f437b` (final6's) | identical |
| 1,992-token TTFT | 161.15 s | 161.10 s (0.9997x) |
| 3,005-token TTFT | 271.05 s | **256.47 s (0.946x)** |
| answer 2 decode | 1.850 tok/s | **1.946 tok/s (+5.2%)** |
| GPU 0 peak | 24,012 MiB | 23,608 MiB |

Prompt 2 is faster because its attention now stays on the GPU. On this box dev's fallbacks did not change the texts.

**Measured on the RTX 5070 12 GB** (c4fa, computer4, DeepGEMM builds of dev 61646cbd and bd7103fd, full model, default
configuration, W + 5 interleaved pairs; [manifests/c4fa](../manifests/c4fa/summary.txt); the summary's arm labels are
final6's tool: A = dev, B = fix A):

| | dev (A, x5) | fix A (B, x5) |
|---|---|---|
| handled allocation failures per sample | 399-420 kv comp + 1 weight, every sample | **0** |
| texts | `a397814f` / `8b3f437b` | identical |
| 1,992-token TTFT (median) | 243.79 s | 244.88 s (1.004x) |
| 3,005-token TTFT (median) | 411.41 s | **377.88 s (0.918x)** |
| answer 2 decode (median) | 1.227 tok/s | **1.307 tok/s (+6.6%)** |
| GPU 0 peak | 11,745 MiB | 11,239 MiB |

B s5's prompt 2 (553 s) overlapped a `make check -j8` on the same machine (03:07-03:12) and is an outlier; the
medians leave it out. In this run dev's fallbacks did not change the texts either (earlier, on this card, 2 of 7
runs changed prompt 2's).

## Fix B: the pure-Vulkan host pool OOM-kills the server

**Branch** `fix/vk-host-image-alloc`, one commit **72aa325a** on dev 61646cbd (pushed to the fork).
`c/backend_vulkan.c` (+14 / -7), new `c/tests/test_vk_host_image_alloc_source.py`. `make check` 2,702 OK (292 skipped).
(final15 measured 8c1f0381, the same code with an earlier comment and no test.)

**Symptom.** On dev 61646cbd every pure-Vulkan sample with the default host pool was OOM-killed after prompt 1
(final9 x7, final10 M, final11 M3), on a 62 GB box (RTX 3090, REAP-150B, `--ram 40`). 0f786bfa / 5c0469ae (the
pool's new sizing) did not fix it.

**How it was found.**
- final10/11: file RSS = the pool (`/dev/nvidiactl` mappings); anon was 45.6 GiB with the pool off (M4) against
  the engine's own plan `projected=39.61GiB`. So the engine overshoots its plan by ~6 GiB in pure Vulkan, and the
  pool's sizing (MemAvailable - 4 GiB - ram_later - reserve) has no room for that.
- final12 T1: an LD_PRELOAD tracer of every allocation >= 32 MiB ([tools/probes/bigalloc](../tools/probes/bigalloc))
  found nothing big that explains it, so the growth is in smaller blocks.
- final12 imports A/B (pool 12 GiB): `COLI_VK_UP_IMPORT=0` took anon from 45.3 to 40.3 GiB. The one allocation
  that differs is the uploader's host image of a tier tensor (`up_img_alloc`): `posix_memalign` + `memset` with
  imports, `calloc` without.
- final14: a fixed malloc mmap threshold (`GLIBC_TUNABLES=glibc.malloc.mmap_threshold=4194304`, so every image is
  its own mmap) took anon to 40.1 GiB, and **the default pool completed** both prompts with imports on or off. But
  it slowed uploads and decode (an mmap + munmap per image), so it is a diagnosis, not the fix.

**Fix.** Off Windows the image is aligned by hand inside a `calloc`'d block (its address sits just before the image),
as the path without imports allocates, and `up_img_free` frees that block. Windows keeps `_aligned_malloc`.

**Measured** (final15, shadowfax, pure Vulkan, imports on, REAP-150B; [manifests/final15](../manifests/final15/summary.txt)):

| | dev 61646cbd (C1, x3) | fix B (J2, x3) |
|---|---|---|
| pool 12 GiB: anon peak | 45.4-45.6 GiB | **40.3-40.4 GiB** |
| pool 12 GiB: MemAvailable min | 0.53-0.98 GiB | 4.8-4.9 GiB |
| pool 12 GiB: 1,992-token TTFT | 103.22 s | **98.80 s (0.957x)** |
| pool 12 GiB: 3,005-token TTFT | 139.36 s | 137.26 s (0.985x) |
| pool 12 GiB: decode (mean of answers) | 4.347 tok/s | 4.375 (1.007x) |
| **default pool** (16.3-16.7 GiB) | OOM-killed after prompt 1, every run (final9-11) | **both prompts, 2 of 2** (J1; anon 40.3, MemAvailable min 1.16 GiB) |

Texts: prompt 1 is `32534c06` (the CUDA build's) in all pool-12 samples, both arms. Prompt 2 varies run to run on
dev itself (C1: `94a31d7b`, `69cfde39`, `a683dba1`; J2 gave `a683dba1` x3), because the tier's device/CPU split per
expert follows the measured upload rate ("cold experts with 16 rows or more" vs 23). So texts are not a criterion
for pure Vulkan. The default-pool runs leave ~1 GiB of MemAvailable, so the pool's 4 GiB margin is still tight on
this box, but it no longer overshoots by the 5 GiB nobody counted.

**Also found:** `COLI_VK_UP_IMPORT=0` is faster on this box (uploads 6.2-6.7 GB/s against 3.5-4.0 with imports,
p1 TTFT 0.931x in final12 on dev). On top of fix B the gap is smaller but still there (final16, x3: staged 6.2-6.6
vs imported 4.3 GB/s, p1 TTFT 0.980x, decode +2.6%, same memory, same texts). That is a device-dependent default (the import path was presumably measured on another
device), so it is a note for upstream rather than a fix.

## PR #2 F18 (not upstream; on `docs5/pr2-two-gpu-decode`)

The probe (final10, [f18-probe.log](../manifests/final10/f18-probe.log)) ran a real `__trap()` on the peer GPU in the
decode MoE, before and after its down-projection, with device 0's stream or the host waiting for the peer's join.
In all four cases device 0's next call failed too. Fix **6628f5b1**: a sticky error latches CUDA off for the whole
process, and the engine runs on the host from then on. The messages say so, and a `DSV4_CUDA_PEER_FAULT=n:sticky`
test mode and a sticky scenario were added to the peer test. `make check` 2,720 OK. final11 runs it on hardware.
