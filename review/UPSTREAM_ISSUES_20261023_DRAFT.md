# Upstream issue drafts (2026-10-23) — NOT filed; Bob decides

> **Status (2026-10-23 evening).** Not ready to file; no fixes attempted. Upstream dev moved 77 commits past 7df8c27c
> (to 61646cbd). #1 still stands (no DeepSeek V4 CUDA changes upstream); re-check on the dev tip before filing; candidate
> fix: allocate the compressed-KV buffer at its final size once ctx is known (a separate PR, needs Bob's go-ahead).
> #2 may be fixed upstream by 0f786bfa (pool within the driver heap) and 5c0469ae / #2318 (pool leaves the KV and
> state allocated later, deepseek_v4 included; from #2314): re-run QO on the dev tip, then drop or file as a follow-up.
> #3 is withdrawn: final8 QF/QG (pool 12 GiB, split free) gave 5 of 5 identical texts.

All on upstream dev 7df8c27c (unchanged by #2264). Evidence: colibri-notes `manifests/final6`, `final7`,
`final8-c4ab`; reading in investigations.md §6-7.

## 1. deepseek-v4: default-configuration text depends on VRAM headroom (kv comp allocation falls back to CPU attention)
**Setup.** Full DeepSeek-V4-Flash-0731, default serve line (no `COLI_CUDA_MOE_BATCH`, `COLI_CUDA_ATTN_BATCH=1`),
`--ram 40 --ctx 8192`, RTX 3090 24 GB or RTX 5070 12 GB. Two fixed prompts (1,992 and 3,005 tokens), greedy.

**What happens.** Without the bank, the mirror growth guard keeps a 600 MiB reserve. Once decode mirrors have grown,
`dsv4_cuda_kv_comp_append` cannot double the compressed-KV buffer (`[DSV4 CUDA] kv comp allocation: out of memory`,
hundreds of times per request), `coli_v4_gpu_kv_cache_sync` returns 0 and that layer's attention runs on the CPU.
CPU and CUDA attention do not give the same bits: with `COLI_CUDA_ATTN_BATCH=0` both prompts' texts differ from the CUDA
ones. So when the fallback starts during a prefill, the greedy text changes; how often it fires varies from run to run
(RTX 5070: 0, 378, 420, 458, 1,066 per sample in the same configuration; 2 of 7 samples changed prompt 2's answer). The
fallback also costs time (3k-token TTFT ~410 s instead of ~373 s on the 5070).

**Workaround.** `DSV4_CUDA_VRAM_RESERVE_MB=2000`: 0 fallbacks in 4 samples, reference texts, normal TTFT.

**Possible fixes.** Count the compressed-KV buffer's growth (its doubling to `ctx` rows × layers) in the reserve, or
allocate it at its final size when the context is known; or have the mirror cache give a slot back on a kv comp
allocation failure instead of falling back.

## 2. vulkan tier: host pool sized from MemAvailable at start can get the server OOM-killed (pure Vulkan, DeepSeek V4)
**Setup.** REAP-150B, `COLI_VULKAN=1 DSV4_CUDA=0 COLI_CUDA=0`, `--ram 40`, 62 GB host, RTX 3090.

**What happens.** `host_init` (vk_tier.c) sizes the pool as `MemAvailable - 4 GiB` (less the RAM cache's floor) at
start. With more free RAM at start (page cache dropped, another model server stopped) it took 15.75-16.75 GiB, and the
kernel OOM-killed `deepseek_v4` during the first request (anon-rss ~43 GB + file-rss ~17-18 GB) in 3 of 3 runs. With 13.25
GiB it survived; with `COLI_VK_TIER_HOST_GB=12` 6 of 6 runs completed, MemAvailable still dipping to 1.5 GB.

**Note.** final6's unit memory peak (cgroup) read 15.1 GB for a process the kernel counted at 44 GB anon RSS, so a cgroup
`MemoryMax` does not contain it (the pool's pinned pages appear not to be charged).

## 3. ~~vulkan tier: texts depend on run-time timings~~ (withdrawn after final8: reproducible once the pool is safe)
With the tier's choices pinned (`COLI_VK_TIER_HOST_GB=12 COLI_VK_TIER_STREAM_ROWS=16 COLI_VK_TIER_BALANCE=0`) runs are
byte-identical across samples and builds; unpinned, the stream threshold (`cold experts with N rows or more go to the
device`) comes from measured upload bandwidth and CPU row time, so the CPU/GPU split, and with it the text, can vary.
final8's QF (pool fixed, split free, x3) says whether it does in practice.
