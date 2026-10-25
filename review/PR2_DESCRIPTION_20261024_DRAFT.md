# PR #2 description (draft, 2026-10-24 evening) — NOT submitted; Bob opens

> Supersedes the PR #2 section of [PR_DESCRIPTIONS_20261020.md](PR_DESCRIPTIONS_20261020.md) (final5 numbers, base
> 736a0682, "Depends on PR #1"). #2264 (PR #1) is merged, so this PR now carries only its own 11 commits.

Target: upstream `JustVugg/colibri` branch `dev`, from the fork's **`docs5/pr2-two-gpu-decode`** at **3ce988da**:
11 linear commits on dev 561966ec, no merges (rewritten from the merged 090d8160 and force-pushed to the fork
2026-10-24, Bob's OK; same tree but the one test fix below). Ready to open from.

Measured trees vs these commits: final11 ran c42ad524 and final13 ran 693204ed, the pre-rewrite F18 commits on dev
61646cbd; their code is the code of 9c3f633c..3ce988da (the rewrite only re-based it; dev's changes since don't touch
DeepSeek V4).

---

**Title:** deepseek-v4: two-GPU decode and VRAM-first expert lookups

### Summary
11 commits on dev. #2264 (batched GPU prefill), which this was built on, is merged, so the diff is this PR's own.

All new behaviour is opt-in:
- `V4_GPU_RESIDENT_FIRST=1`: an expert already mirrored in VRAM is computed from the mirror
  without a RAM-store lookup; the store still records the routing (history, HITS, heat) and its
  RAM pin ranking leaves mirrored experts out, so RAM holds experts that are not in VRAM. Useful on one GPU.
- `DSV4_CUDA_PEER_DEVICE=N`: a second mirror pool on GPU N (needs peer access both ways, e.g.
  NVLink; without it the pool stays off with a warning and decode runs on the primary GPU).
  `DSV4_CUDA_PEER_COMPUTE` (default 1) runs the experts mirrored there on that GPU,
  bit-identically. GPU N is probed before it is touched and a failed init is rolled back.
- `DSV4_CUDA_PIN_CAP_GB` caps the page-locked expert slabs (default unchanged, 20 GiB).
- `V4_SERVE_STATS_LOG=1` prints per-turn decode stats and decode-MoE counters.

Two changes apply with every switch off:
- **Mirror caches keep their VRAM reserve between probes and drop failed uploads.** Free VRAM is re-read every 64
  misses; in between, the cache also counts the mirrors it grew since the last reading, so an explicit
  `DSV4_CUDA_VRAM_RESERVE_MB` or the 600 MiB default without the bank is kept once 8 mirrors are live. Before, the
  cache could grow up to 64 mirrors (816 MiB) past it, past the card's free VRAM; on the full V4 Flash at 24 GB the
  compressed-KV buffer then could not grow (`[DSV4 CUDA] kv comp allocation: out of memory`, ~420 per two-prompt run)
  and those layers' attention ran on the CPU. The bank-sized default already carries those 64 mirrors.
- **A sticky CUDA fault no longer fails every later request.** A launch failure (e.g. a trap on the peer GPU) is
  sticky for the process's whole CUDA context, device 0 included. The engine now latches CUDA off for the process
  (`dsv4_cuda_lost()`): mirror lookups miss, the decode MoE takes host copies, the attention's grouped `wo_a` matvec
  falls back to the CPU, and a failed decode block names its stage and keeps the inner error. The faulted request
  and every later one complete on the host, at host speed, until restart. **Correction:** earlier versions of this PR
  said that after a peer fault "decode continues on device 0". That was wrong (a real-trap probe showed device 0's
  next call fails too), and the log line now says the process's CUDA is lost and decode continues on the host.

### Measurements
2× RTX 3090 with an NVLink bridge, REAP-150B, `--ram 40`, `COLI_CUDA_MOE_BATCH=1`, greedy; six
300-token answers per sample, a fresh server per sample, interleaved arms; metric = the sample's mean decode tok/s
over the six answers; medians.

| configuration | final6 (x3; dev 7df8c27c) | final11 (x3; dev 61646cbd) | vs dev (final11) |
|---|---|---|---|
| upstream dev | 3.014 | 3.021 | |
| this PR, every new switch off (`V4_SERVE_STATS_LOG=1` only) | 3.011 | 3.017 | 0.999x |
| + `V4_GPU_RESIDENT_FIRST=1` (one GPU) | 3.096 | — | (final6: 1.021-1.030x over switches off) |
| + `DSV4_CUDA_PEER_DEVICE=1` (peer compute on GPU 1) | 4.343 | **4.342** | **1.437x** |

Every answer in every sample is byte-identical to upstream dev's. final13 (x1, 693204ed = this PR's 3ce988da before the rebase)
repeated the all-on arm: 4.332 tok/s, the same six texts.

**Fault test** (final13, 693204ed): `DSV4_CUDA_PEER_FAULT=n:sticky` injects a sticky fault on the peer GPU in a server
answering the same six prompts. All six requests complete with 300 tokens each (on the host after the fault, 2.94
tok/s mean decode), with no hang; before the fix every request after the fault failed until restart.

Manifests with hashed raw evidence: [final6](https://github.com/bobsummerwill/colibri-notes/tree/70b7d97/manifests/final6),
[final11](https://github.com/bobsummerwill/colibri-notes/tree/70b7d97/manifests/final11),
[final13](https://github.com/bobsummerwill/colibri-notes/tree/70b7d97/manifests/final13).

### How this was tested
- New tests: `test_dsv4_peer_init_cuda.cu` and `test_dsv4_peer_moe_cuda.c` (two-GPU init, rollback, peer compute
  bit-identical to one GPU, peer-fault latch incl. a sticky scenario; they skip without two peer-capable GPUs),
  `test_v4_peer_init_source.py`, `test_dsv4_cuda_lost_source.py` (the F18 latch), `test_v4_moe_batch_books_source.py`,
  and a VRAM-first store test in `test_deepseek_v4.c`.
- On the tip 3ce988da (counter, gcc 16.2.1, no GPU): `make -C c check` 2,741 tests OK (312 skipped).
- Every commit: the CPU engine builds, and `deepseek_v4.c` (with `-DCOLI_V4_GPU_TIER
  -DCOLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER`, and again with `-DCOLI_VULKAN`) and `tests/test_deepseek_v4.c` (with
  `-DCOLI_V4_TEST_HOOKS`) compile with `-Wall -Wextra`: 0 diagnostics; the dsv4 parity and V4 source tests pass.
- GPU (final11, shadowfax, 2× RTX 3090 + NVLink, c42ad524 on dev 61646cbd: the F18 latch before its last two
  commits, which touch only `deepseek_v4.c`'s CPU fallbacks and error messages): `make -C c dsv4-cuda-test` and
  `make -C c cuda-test` pass. dev 61646cbd..561966ec changes no DeepSeek V4 or shared CUDA code outside #2264 (the
  other GPU changes are Qwen3.6's CUDA chain; the Makefile change only adds a Qwen3.6 test target), so these results
  carry over; the GPU tests were not re-run on the merged tip.
- **Everything else**: [colibri-notes](https://github.com/bobsummerwill/colibri-notes/blob/70b7d97/README.md). Developed
  with AI assistance (Claude Code; see the `Co-Authored-By` trailers); every change was reviewed and tested as
  described here.

### Validation
- [x] `make -C c check`: 2741 tests, OK (312 skipped)
- [x] CUDA: `make -C c dsv4-cuda-test` and `make -C c cuda-test` pass on two RTX 3090s with NVLink (on dev 61646cbd);
  the peer tests skip without two peer-capable GPUs
- [x] Performance claims include hardware, commands and repeatable measurements
- [x] Experiment manifests with raw evidence

### Compatibility
- [x] The default CPU build remains dependency-free
- [x] No model files, generated binaries, or benchmark artifacts are included

---

### Notes for Bob (not part of the description)
- **Fix A overlap:** `fix/dsv4-mirror-guard-overshoot` is the same guard fix as this PR's b9b1f0ac (was 6b7fc6c0),
  on its own. If fix A is opened and merged first, this PR takes its own side of the guard and ports fix A's source
  test to the per-cache fields (trial merge against the old tip: `scratch/fixA-x-pr2` c89353e4 on counter; redo it
  against 3ce988da). Mention fix A in this description if both are open.
- **Bisect nit: fixed by the rewrite.** The VRAM-first commit (d75d9639) now asserts at least one
  `v4_views_fetch_host` call instead of exactly two; every commit passes its tests.
- The manifest links are pinned to colibri-notes 70b7d97.
