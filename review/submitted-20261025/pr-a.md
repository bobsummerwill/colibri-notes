## Summary

Fixes #2399.

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

## Measurements
Full DeepSeek-V4-Flash-0731, default configuration, two fixed prompts (1,992 and 3,005 tokens), greedy, `--ram 40
--ctx 8192`, a fresh server per sample, interleaved. Measured on dev 61646cbd and bd7103fd (this change on
61646cbd), then re-run on this tip (below).

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

**Re-run on this tip** (2026-10-25, dd03a9de vs dev 77530b49, x2 each, interleaved):
- RTX 3090: dev had **1,139 kv comp fallbacks per sample** (GPU 0 at 24,119 MiB), and this time they changed dev's
  answer 2 (`964826e1…`, ≠ the earlier reference `8b3f437b…`). This PR: 0 fallbacks, 23,605 MiB, both answers = the
  reference; 3,005-token TTFT **0.953x**, answer 1 equal (TTFT 0.991x, decode 1.005x).
- RTX 5070 (DeepGEMM): neither had fallbacks this time; texts = reference on both; 0.996x mean decode, 0.987x TTFT.
- With `COLI_CUDA_MOE_BATCH=1` on the 3090 (the bank in use, x1 each): everything 1.00x, texts equal. On the 5070 the
  3.4 GiB bank doesn't fit on dev or here (handled: the batch stays off).
- `dsv4-cuda-test` / `cuda-test` pass on this tip (3090, CUDA 12.0; 5070, DeepGEMM sm_120a).
  [final21](https://github.com/bobsummerwill/colibri-notes/blob/471769a/manifests/shadowfax-20261025/final21-run.log),
  [c4f21](https://github.com/bobsummerwill/colibri-notes/blob/471769a/manifests/computer4-20261025/c4f21-run.log).

## How this was tested
- New `tests/test_v4_mirror_guard_source.py` pins the guard at source level (no fixture reaches a full card).
- On the tip dd03a9de (counter, gcc 16.2.1): `make -C c check` 2,746 tests OK (320 skipped); `deepseek_v4.c` compiles with
  `-Wall -Wextra -DCOLI_V4_GPU_TIER` (and `-DCOLI_VULKAN`), 0 diagnostics. GPU runs as above, on 61646cbd, and
  repeated on dd03a9de vs dev 77530b49 (2026-10-25, above).

## Validation
- [x] `make -C c check`: 2746 tests, OK (320 skipped)
- [x] CUDA: `make -C c dsv4-cuda-test` and `make -C c cuda-test` pass on this tip (RTX 3090, CUDA 12.0 with `-DCOLI_DSV4_NO_TC`); `dsv4-cuda-test` also on an RTX 5070 (DeepGEMM sm_120a)
- [x] Performance claims include hardware, commands and repeatable measurements
- [x] Experiment manifests with raw evidence ([final10](https://github.com/bobsummerwill/colibri-notes/tree/70b7d97/manifests/final10), [c4fa](https://github.com/bobsummerwill/colibri-notes/tree/70b7d97/manifests/c4fa))

## Compatibility
- [x] The default CPU build remains dependency-free
- [x] No model files, generated binaries, or benchmark artifacts are included

Developed with AI assistance (Claude Code; see the `Co-Authored-By` trailer); the change was reviewed and tested as described here.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
