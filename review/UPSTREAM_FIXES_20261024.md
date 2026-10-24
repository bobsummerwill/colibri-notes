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

## Fix B: the pure-Vulkan host pool OOM-kills the server (in progress)

Still on dev 61646cbd: every pure-Vulkan sample with the default host pool was OOM-killed after prompt 1 (final9 x7,
final10 M), on a 62 GB box, REAP-150B, `--ram 40`. What final10 measured:
- file RSS = the pool (`/dev/nvidiactl` mappings, 16.3 GiB); anon = the expert cache (37.5 GiB planned) plus
  ~3-4 GiB the plan does not count; MemFree reaches ~0.5 GiB;
- `COLI_VK_UP_IMPORT=0` (uploads staged instead of imported) completed both prompts, just: anon 40.2 + file 17.0,
  MemFree 0.5. It also ran the uploads at 6.6 GB/s instead of 3.1 (TTFT 96.1 vs 102.6 s).
- The pool's sizing (`host_init`, vk_tier.c) leaves a fixed 4 GiB for everything the engine allocates once it serves,
  beyond `ram_later`; deepseek_v4's `v4_state_later` counts 630 MiB. smaps at 55 GiB shows ~4 GiB in anon regions of
  768-row-step sizes.

final11 M3 (smaps every 15 s) and M4 (pool off) locate the growth; the fix follows from that.

## PR #2 F18 (not upstream; on `docs5/pr2-two-gpu-decode`)

The probe (final10, [f18-probe.log](../manifests/final10/f18-probe.log)) ran a real `__trap()` on the peer GPU in the
decode MoE, before and after its down-projection, with device 0's stream or the host waiting for the peer's join.
In all four cases device 0's next call failed too. Fix **6628f5b1**: a sticky error latches CUDA off for the whole
process, and the engine runs on the host from then on. The messages say so, and a `DSV4_CUDA_PEER_FAULT=n:sticky`
test mode and a sticky scenario were added to the peer test. `make check` 2,720 OK. final11 runs it on hardware.
