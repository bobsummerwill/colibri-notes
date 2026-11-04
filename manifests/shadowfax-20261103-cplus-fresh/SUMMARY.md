# Fresh C+ on `feat/dsv4-cplus-fresh` (2026-11-03)

Branch: `feat/dsv4-cplus-fresh` on `bobsummerwill/colibri`, built from the #1766
head-followups branch (`f4586dca`) plus a merge of PR #2405's 11 commits
(`11c40038`):

- `64fd75e3` — merge of #2405 into the #1766 + fixes base, C+ conflict
  resolution (mutually exclusive layer split / peer pool, per-device caches,
  peer dispatch, reserve split).
- `8a30a985` — **peer-init policy fix:** `dsv4_cuda_init` is strict again by
  default (2405's rollback test), and the engine sets best-effort only around
  the #1766 layer-split init. This was found by the 2405 peer-init test during
  this campaign.

Performance was measured at `64fd75e3`; tests/soak/5070/counter at
`8a30a985`. The strict fix only changes the peer-enable failure path, not the
normal two-GPU path.

## 2×3090: fresh C+ vs dev

2× RTX 3090 24 GB NVLink, 500/450 W, cold server, `--ram 40 --ctx 8192`,
`COLI_CUDA_MOE_BATCH=1`, 2 samples. C+ 1 GPU is `V4_GPU_RESIDENT_FIRST=1`;
C+ 2 GPUs adds `DSV4_CUDA_PEER_DEVICE=1`.

| arm | GPUs | decode tok/s | ratio |
|---|---:|---:|---:|
| dev REAP | 1 | 2.9811 | — |
| #1766 REAP | 1 | 3.6469 | 1.223x |
| C+ REAP | 1 | 3.7714 | **1.265x vs dev** |
| C+ REAP | 2 | 5.7778 | **1.938x vs dev**, 1.532x vs C+ 1 GPU |
| dev full fdec | 1 | 2.6385 | — |
| #1766 full fdec | 1 | 3.1573 | 1.197x |
| C+ full fdec | 1 | 3.2202 | **1.220x vs dev** |
| C+ full fdec | 2 | 4.1852 | **1.586x vs dev**, 1.300x vs C+ 1 GPU |

This matches the notes' earlier C+ (1.96x REAP, 1.59x full) within run noise.
All full-model hashes identical: `3d0a5275`, `f0e57dde`, `d27222f1`; all REAP
texts identical across dev/#1766/C+1/C+2.

### C+ one-GPU ablation

| arm | decode tok/s | vs C+ 1 GPU |
|---|---:|---:|
| `V4_GPU_RESIDENT_FIRST=0` | 3.1337 | 0.976x |
| C+ default | 3.2097 | — |
| head off | 3.0567 | 0.952x |
| resident indexer off | 3.0770 | 0.959x |
| indexer heads off | 3.1517 | 0.982x |
| indexer prep off | 3.1900 | 0.994x |
| `WO_DECODE=0` | 3.1987 | 0.997x |
| compressor off | 3.1937 | 0.995x |
| resident route off | 3.2097 | 1.000x |
| `BATCHED=0` | 2.8400 | 0.885x |

VRAM-first adds ~2.4%; the head and resident indexer are the main kernel
wins; batched routing is worth ~13% on this model.

- **Peer compute off** (`DSV4_CUDA_PEER_COMPUTE=0`, 2 GPUs): 3.5303 tok/s,
  0.844x of peer-on 4.1852 — peer compute is real, not just extra cache.
- **Sticky peer fault** (`DSV4_CUDA_PEER_FAULT=300:sticky`, REAP, 6 prompts):
  6/6 rows complete, mean 2.8845 tok/s (host after the fault), no hang. Texts
  differ from the non-fault run, as expected for the host fallback.
- **Built-in profiles:** C+1 3.2087, C+2 4.1653 tok/s with
  `DSV4_ATTN_PROF=1 V4_MOE_BATCH_PROF=1 DSV4_DECODE_PROF=1`; logs in the
  campaign artifacts.

## Tests, soak, profiling

At `8a30a985`, with `DSV4_TEST_DEVICES=0,1`:

- **Nine head-followups CUDA tests: 9/9 pass** (`NINE_RC=0`).
- **PR #2405 peer/batch tests: pass** (`PEER_RC=0`), including the peer-init
  rollback test that had caught the best-effort policy bug.
- **Soak:** one C+2 server, three repeats of 6 prompts × 300 tokens; all 18
  request texts identical to the first repeat.
- **`nsys`:** this shadowfax install cannot produce a `.nsys-rep` — the
  profiler writes `.qdstrm` and reports a missing importer. Built-in
  attention/decode/MoE profiles ran; no external timeline could be captured.

## RTX 5070 (computer4)

Full V4 Flash, 6 prompts × 300 tokens, 2 samples each, `DSV4_CUDA_VRAM_RESERVE_MB=2000`
and `DSV4_CUDA_RESIDENT_RESERVE_MB=2000`, C+ 1 GPU with VRAM-first.

| arm | samples | mean |
|---|---:|---:|
| dev `3b953308` | 1.6138, 1.6058 | 1.6098 |
| C+ `8a30a985` | 1.9580, 1.9518 | **1.9549** |

**1.214x**, all texts identical. Both runs logged `head=resident device=0`.
The first 5070 attempt used the default resident reserve (2800); the head fell
back to the CPU (`need=2020`, `free=4369`) and the gain was only ~1.12x. With
the explicit 2000 the head stays resident and the gain matches the 1766-era
5070 result.

## Counter (Intel MTL iGPU)

At `8a30a985`:

- CPU-only build succeeds; the V4 tiny token-exact suite passes (`TINY_RC=0`).
- `VK=1` build succeeds; a 4-token iGPU run returns the CPU's exact tokens
  (`<t024><t025><t026><t027>`), `VK_RUN_RC=0`.

As before, this does not exercise the CUDA changes; it is a non-CUDA
build/run sanity.

## Artifacts

- [campaign.log](campaign.log) — 2×3090 arms, summary, ablations
- [tests-soak.log](tests-soak.log) — nine tests, peer/batch tests, soak
- [5070.log](5070.log)
- [counter.log](counter.log)

Raw directories on shadowfax: `~/colibri-bench/final40-cplus/`
(`post2.log`, `soak2/`, `nsys2/`). On computer4:
`~/colibri-bench/c4-cplus-5070/`. On counter: `~/colibri-bench/cplus-counter.log`.

## Caveats

- 2×3090 numbers are 2 samples; the 5070 is 2 samples; the ablation is 1.
- No 6×5090 run; residency still falls back on 24 GB and 12 GB.
- `nsys` is unavailable on shadowfax; only built-in profiles.
- The 5070 needs `DSV4_CUDA_RESIDENT_RESERVE_MB=2000` to keep the 1.0 GiB head
  resident on 12 GB; the 2800 default is too conservative there.
- The C+ branch is a merge of #2405's 11 commits, not a rebase onto current dev.
