# PR descriptions (drafts, 2026-10-20) — NOT submitted

> **Naming key.** **PR #0** was the six fixes JustVugg/colibri [#2212](https://github.com/JustVugg/colibri/pull/2212)–[#2217](https://github.com/JustVugg/colibri/pull/2217), **all merged** into `dev`.
> **PR #1** (batched GPU prefill for expert-pruned checkpoints) and **PR #2** (two-GPU decode) are not opened yet.
> This file supersedes [PR_DESCRIPTIONS_20261017.md](PR_DESCRIPTIONS_20261017.md) (base 62eaa9dc, PR #0 still open).

Target: upstream `JustVugg/colibri` branch `dev`, base **736a0682**. Branches on the fork `bobsummerwill/colibri`
to open the PRs from: **`docs5/pr1-reap-prefill`** (833c6762, 8 commits on dev) and **`docs5/pr2-two-gpu-decode`**
(bb08dc4b, 8 commits on PR #1). They equal the benchmarked `refresh/*` trees outside `docs/`; only the two docs commits
changed, to carry the final5 numbers (and PR #2's docs commit message to match). Bob submits; never us.

Validation common to both (2026-10-19/20, on the refresh branches):
- `make -C c check` (computer4, Ubuntu 24.04, gcc 13.3): see each PR.
- Every one of the 16 commits: the CPU engine builds, and `deepseek_v4.c` (with `-DCOLI_V4_GPU_TIER
  -DCOLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER`) and `tests/test_deepseek_v4.c` (with `-DCOLI_V4_TEST_HOOKS`)
  compile with `-Wall -Wextra`: 0 diagnostics ([percommit.txt](https://github.com/bobsummerwill/colibri-notes/blob/80adc1c/manifests/final5/percommit.txt)).
- CUDA engine `make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_ARCH=sm_86 NO_TC=1` (CUDA 12.0): 0 warnings at
  dev and both tips.
- GPU (2026-10-20 00:15–03:00, `final5`, 2× RTX 3090 + NVLink): the benchmarks below, every gate passed;
  then `make -C c dsv4-cuda-test` passes at both tips (0 warnings) and `make -C c cuda-test` at the PR #2 tip.
  (CUDA 12.0 needs `-DCOLI_DSV4_NO_TC` in `NVCCFLAGS` for these test targets: the cuBLASLt block-scaling API is 12.8+.)

---

## PR #1 — `pr1/reap-prefill`: deepseek-v4: batched GPU prefill for expert-pruned (REAP) checkpoints, faster generic kernels

8 commits on `dev`.

### Summary
The batched GPU prefill MoE (`COLI_CUDA_MOE_BATCH=1`, off by default) only took 256-expert
checkpoints. Expert-pruned ones such as `puwaer/DeepSeek-V4-Flash-0731-reap-150b` (132 routed
experts) fell back to the CPU union, ~15 tok/s prefill on an RTX 3090 box.

- The bank is sized by `dsv4_cuda_expert_bank_slots`: 132 slots / 1683 MiB for REAP-150B on
  generic builds; DeepGEMM builds keep 256.
- Pruned, hash and missing-bias-mirror layers route on the CPU bf16 router inside the batched
  path (the GPU top-6 kernel assumes 256 logits). Routing is bit-identical to the CPU union.
- The default decode-mirror VRAM reserve with the bank on is sized from the model at engine
  open (bank ×1 or ×2 + 600 MiB headroom + 64 mirrors: 3100 MiB for 132 experts, 4681 MiB for
  256). A GGUF checkpoint, which never gets a bank, reserves none for it.
- The bank keeps the RAM expert store's routing books (usage, pin ramp, repins, HITS/heat,
  `.coli_usage`) as the CPU union's lookups do, without extra reads (otherwise decode after a
  bank prefill was 7.4 % slower because the store saw almost no requests; this recovers 67–79 % of that gap).
- Two bit-identical generic kernels: expert-major fp4 for the batched MoE
  (`DSV4_MOE_EXPERT_MAJOR`, default on) and warp-per-row for short fp8 dense batch matmuls
  (`DSV4_MM_WARP`, auto = I ≤ 1024). `_VERIFY=1` cross-checks old against new.
- `V4_MOE_BATCH_PROF=1` prints a per-chunk time split.
- Docs: REAP prefill, the kernels, and the measurements below.

### Measurements
2× RTX 3090 (one used; CUDA 12.0, sm_86, `NO_TC=1`), Ryzen 9 9950X3D, 62 GB RAM, SN850X;
REAP-150B, `--ram 40 --ctx 8192`, greedy; a fresh server per sample, 3 samples per arm
interleaved with the order rotated per round (plus one discarded warm-up), medians;
`.coli_usage` empty.

| prompt | upstream dev 736a0682 (CPU union) | this PR | this PR + `COLI_CUDA_MOE_DOUBLE=1` |
|---|---|---|---|
| 1,992 tokens, time to first token | 130.2 s (15.3 tok/s) | 76.8 s (25.9 tok/s), **1.70×** | 64.1 s (31.1 tok/s), **2.03×** |
| 3,005 tokens, time to first token | 205.9 s (14.6 tok/s) | 111.0 s (27.1 tok/s), **1.86×** | 97.8 s (30.7 tok/s), **2.10×** |
| decode after the prompt (means) | 2.27 / 2.94 tok/s | 2.20 / 2.89 tok/s | 2.07 / 2.83 tok/s |

Answer texts were identical to upstream dev's in every sample (both requests, all arms). TTFT CV ≤ 1 %.

Decode right after the prompt is 1.6–2.8 % slower than dev with the bank, and the double bank
costs another 2.3–5.9 % against the single bank (means; the texts are identical). The
likely cause is the bank's VRAM reserve: with `COLI_CUDA_MOE_BATCH=1` the engine reserves 3100 MiB
for the 132-slot bank, which dev (which refuses the batched path for this model) does not, so
fewer decode mirrors fit (GPU0 peak 21,192 vs 22,008 MiB on the decode client). The bank is only
used with `COLI_CUDA_MOE_BATCH=1`; `DSV4_CUDA_VRAM_RESERVE_MB` overrides the reserve.

On the previous base 62eaa9dc the same comparison gave 1.69× / 1.86× (2.01× / 2.09× with the
double bank): the speedups carried over unchanged.

Manifests (`c/experiment_manifest.py` format, all validate): engine A→B and
`COLI_CUDA_MOE_DOUBLE` B→C for both prompts, with hashed raw evidence:
[manifests/final5](https://github.com/bobsummerwill/colibri-notes/tree/80adc1c/manifests/final5).

### How this was tested
- New tests: `test_dsv4_generic_bitwise_cuda.c` (old vs new kernels bit-identical on the GPU),
  `test_dsv4_bank_bytes_cuda.c` (bank sizing), `test_v4_moe_batch_books_source.py`, and two store
  cases in `test_deepseek_v4.c` (the bank's routing books match the CPU union's after every chunk,
  and the saved `.coli_usage` files are byte-identical); `make check` 2,612 tests pass (284 skipped);
  `dsv4-cuda-test` passes on an RTX 3090.
- Bit-identity of the two new kernels: `_VERIFY=1` modes report 0 mismatching bits over full
  prefills; an earlier review re-checked them at runtime over tens of millions of outputs.
- **Review:** an earlier version of this series went through a multi-agent adversarial code
  review that raised 60 findings ([F1–F60](https://github.com/bobsummerwill/colibri-notes/blob/80adc1c/review/REVIEW_REPORT.md)); the commit messages name
  the finding each change resolves, and a stack check approved the series with no serious
  findings ([STACK_CHECK](https://github.com/bobsummerwill/colibri-notes/blob/80adc1c/review/STACK_CHECK.md)). Rebased onto 736a0682 with no conflicts; `git range-diff` shows all 8
  patches unchanged.
- **Everything else** (environment, every measurement, investigations, decisions):
  [colibri-notes](https://github.com/bobsummerwill/colibri-notes/blob/80adc1c/README.md). Developed with AI assistance (Claude Code; see the
  `Co-Authored-By` trailers); every change was reviewed and tested as described here.

### Validation
- [x] `make -C c check`: 2612 tests, OK (284 skipped)
- [x] CUDA: build sm_86 0 warnings; `make -C c dsv4-cuda-test` passes on an RTX 3090 (0 warnings)
- [x] Performance claims include hardware, commands and repeatable measurements
- [x] Experiment manifests with raw evidence (validated with `c/experiment_manifest.py`)

### Compatibility
- [x] The default CPU build remains dependency-free (everything except the bit-identical kernels
  is behind `COLI_CUDA_MOE_BATCH=1`)
- [x] No model files, generated binaries, or benchmark artifacts are included

---

## PR #2 — `pr2/two-gpu-decode`: deepseek-v4: two-GPU decode and VRAM-first expert lookups

**Depends on PR #1.** 8 commits on top.

### Summary
All new behaviour is opt-in:
- `V4_GPU_RESIDENT_FIRST=1`: an expert already mirrored in VRAM is computed from the mirror
  without a RAM-store lookup; the store still records the routing (history, HITS, heat) and its
  RAM pin ranking leaves mirrored experts out, so RAM holds experts that are not in VRAM. Useful on one GPU.
- `DSV4_CUDA_PEER_DEVICE=N`: a second mirror pool on GPU N (needs peer access both ways, e.g.
  NVLink; without it the pool stays off with a warning and decode runs on the primary GPU).
  `DSV4_CUDA_PEER_COMPUTE` (default 1) runs the experts mirrored there on that GPU,
  bit-identically; a failing peer is latched off and the token recomputed. GPU N is probed
  before it is touched and a failed init is rolled back.
- `DSV4_CUDA_PIN_CAP_GB` caps the page-locked expert slabs (default unchanged, 20 GiB).
- `V4_SERVE_STATS_LOG=1` prints per-turn decode stats and decode-MoE counters.
- Mirror caches keep their VRAM reserve between probes and drop failed uploads. This is the one
  change with every switch off: an explicit or no-bank reserve is now kept once 8 mirrors are
  live.

### Measurements
2× RTX 3090 with an NVLink bridge, REAP-150B, `--ram 40`, `COLI_CUDA_MOE_BATCH=1`, greedy; six
300-token answers per sample, a fresh server per sample, 3 samples per arm interleaved with the
order rotated per round; metric = the sample's mean decode tok/s over the six answers; medians.

| configuration | decode tok/s | vs the row above |
|---|---|---|
| upstream dev 736a0682 | 3.011 | |
| this PR, every new switch off (`V4_SERVE_STATS_LOG=1` only) | 2.983 | −0.9 % (see below) |
| + `V4_GPU_RESIDENT_FIRST=1` (one GPU) | 3.072 | **+3.0 %** |
| + `DSV4_CUDA_PEER_DEVICE=1` (peer compute on GPU 1) | 4.287 | **+39.5 %** |

Overall **+43.7 %** over the switches-off build (+42.4 % over dev). All 72 answers (12 samples × 6)
are byte-identical to upstream dev's. The spread is tiny (CV ≤ 0.1 % per arm).

The −0.9 % row is not PR #2 code: with `COLI_CUDA_MOE_BATCH=1`, PR #1's bank reserves 3100 MiB of
VRAM that dev does not reserve for this model (GPU0 peak 21,192 vs 22,008 MiB), so fewer decode
mirrors fit. A decode-only server can leave `COLI_CUDA_MOE_BATCH` unset or lower
`DSV4_CUDA_VRAM_RESERVE_MB`.

Manifests (one switch each, validated): `V4_GPU_RESIDENT_FIRST` (G1→R1, 1.03×) and
`DSV4_CUDA_PEER_DEVICE` (R1→X3, 1.40×), with hashed raw evidence:
[manifests/final5](https://github.com/bobsummerwill/colibri-notes/tree/80adc1c/manifests/final5).
(The previous base's single-sample sanity check gave 3.01 → 4.34 tok/s, +44 %.)

### How this was tested
- New tests: `test_dsv4_peer_init_cuda.cu` and `test_dsv4_peer_moe_cuda.c` (two-GPU init,
  rollback, peer compute bit-identical to one GPU, peer-fault latch; they skip without two
  peer-capable GPUs), `test_v4_peer_init_source.py`, and a VRAM-first store test in
  `test_deepseek_v4.c`; `make check` 2,619 tests pass (284 skipped); `dsv4-cuda-test` and
  `cuda-test` pass on 2× RTX 3090 + NVLink.
- **Review:** as PR #1 (findings F1–F60, stack check). Rebased onto 736a0682 with no conflicts;
  `git range-diff`: 7 of 8 patches unchanged, the 8th differs only in one context line (dev's new
  GGUF-to-Vulkan hand-off sits above the unchanged peer init).
- **Everything else**: [colibri-notes](https://github.com/bobsummerwill/colibri-notes/blob/80adc1c/README.md). Developed with AI assistance (Claude Code; see the
  `Co-Authored-By` trailers); every change was reviewed and tested as described here.

### Validation
- [x] `make -C c check`: 2619 tests, OK (284 skipped)
- [x] CUDA: build sm_86 0 warnings; `make -C c dsv4-cuda-test` and `make -C c cuda-test` pass on
  two RTX 3090s with NVLink; the peer tests skip without two peer-capable GPUs
- [x] Performance claims include hardware, commands and repeatable measurements
- [x] Experiment manifests with raw evidence (validated with `c/experiment_manifest.py`)

### Compatibility
- [x] The default CPU build remains dependency-free
- [x] No model files, generated binaries, or benchmark artifacts are included
