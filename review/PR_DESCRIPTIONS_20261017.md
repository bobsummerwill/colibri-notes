# PR descriptions (drafts, 2026-10-17) — NOT submitted

> **Naming key.** These notes use internal names for the upstream pull requests. **PR #0** is the six
> fixes opened as JustVugg/colibri [#2212](https://github.com/JustVugg/colibri/pull/2212) (0a), [#2213](https://github.com/JustVugg/colibri/pull/2213) (0b), [#2214](https://github.com/JustVugg/colibri/pull/2214) (0c), [#2215](https://github.com/JustVugg/colibri/pull/2215) (0d),
> [#2216](https://github.com/JustVugg/colibri/pull/2216) (0e) and [#2217](https://github.com/JustVugg/colibri/pull/2217) (0f). **PR #1** (batched GPU prefill for expert-pruned checkpoints) and
> **PR #2** (two-GPU decode) are not opened yet. Documents dated before 2026-10-17 describe earlier
> versions of these changes, on the old base 7b55a9b3.

Target: upstream `JustVugg/colibri` branch `dev`, base **62eaa9dc**. Branches are on the fork
`bobsummerwill/colibri` as `rebased/<name>` (e.g. `rebased/pr0a/dsv4-upload-ordering`); the
names in the headings below are the intended PR branch names. Bob submits; never us. Eight PRs: six independent PR #0 fixes
(0a–0f), then PR #1 and PR #2, which build on all six.

Validation common to all (2026-10-17, on the rebased branches):
- `make check` (computer4, Ubuntu 24.04, gcc 13.3): see each PR.
- CUDA engine build `make -C c -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_ARCH=sm_86 NO_TC=1`
  (CUDA 12.0, RTX 3090): 0 warnings, 0 errors at the PR #0, #1 and #2 tips.
- Every commit compiles with `-Wall -Wextra` (`deepseek_v4.c` with `-DCOLI_V4_GPU_TIER
  -DCOLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER`, `tests/test_deepseek_v4.c` with test hooks): 0
  diagnostics.
- GPU (2026-10-17 22:34–23:38, `final4`): benchmarks below; `make -C c dsv4-cuda-test` passes at the
  PR #1 and #2 tips and `make -C c cuda-test` at the PR #2 tip (2× RTX 3090, 0 warnings).

---

## PR 0a — `pr0a/dsv4-upload-ordering`: deepseek-v4: order CUDA uploads on the compute stream, clear stale CUDA errors

### Summary
Host-to-device copies in the DeepSeek V4 CUDA backend (decode expert mirrors, in-place refills,
expert and hash tables, RoPE tables, the TP2 bank) ran on the legacy default stream, while the
fused decode MoE runs on the backend's compute stream. A refill could therefore overwrite a
mirror slot the previous token's kernel was still reading (write-after-read). A new `h2d()`
helper issues every upload with `cudaMemcpyAsync` on the compute stream and synchronizes.

Three smaller fixes in the same file, each its own commit:
- a failed expert-bank allocation no longer leaves a sticky CUDA error that fails the next,
  unrelated launch;
- the same for failed buffer and tensor allocations (`alloc_ok()` covers `buf()`, `upload()`,
  activation creation and the KV ring / compressed-KV appends);
- `dsv4_cuda_shutdown` zeroes each device slot it tears down, so a re-init cannot reuse freed
  buffers.

4 commits, `c/backend_cuda_dsv4.cu` only (+52/−31). No new switches; behaviour changes only in
which stream copies run on and on failure paths.

### Evidence
Three small single-GPU harnesses, linked against `backend_cuda_dsv4.cu`, reproduce each bug on
upstream dev and pass with this PR (RTX 3090, 2026-10-18; sources, build/run commands and logs:
[evidence/pr0-harness](https://github.com/bobsummerwill/colibri-notes/tree/e19cfc03500f0e8ff7e27e4c49b50d841b3515db/evidence/pr0-harness)):

| harness | upstream dev 62eaa9dc | this PR |
|---|---|---|
| `h2d_race 300 20` (pageable refill; WAR / RAW / UPLOAD mismatched) | **300/300** / 0/300 / 0/300 | 0 / 0 / 0 |
| `bank_lasterr` (an unrelated launch after a failed bank allocation) | **fails** (stale OOM) | passes |
| `shutdown_reinit` (matvec after shutdown + init in one process) | **fails** | passes |

Only the write-after-read case races: an in-place refill landing under a queued kernel. In
full-engine runs, plain dev decode did not change its output (17 stressed cold runs on an RTX
3090), so we do not claim a visible end-to-end fix there. On the batched-prefill path with the
double bank (PR #1) it did: the pre-fix build diverged in 1 of 3 cold runs, with per-token logits
differing from the first decode step after a 3,005-token prefill; the fixed build was
bit-identical in 3/3.

### How this was tested
- `make check` at the combined PR #0 tip: 2,572 tests pass (upstream dev: 2,560); every commit
  compiles with `-Wall -Wextra`, 0 diagnostics; CUDA build (sm_86) 0 warnings; `make -C c
  dsv4-cuda-test` and `make -C c cuda-test` pass on 2× RTX 3090 at the PR #1/#2 tips, which
  contain this change.
- How the race was found and fixed: [investigations §3](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/investigations.md); the
  double-bank divergence before/after the fix, with per-token logit hashes:
  [manifests/final2/race](https://github.com/bobsummerwill/colibri-notes/tree/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/manifests/final2/race), [evidence/determinism](https://github.com/bobsummerwill/colibri-notes/tree/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/evidence/determinism).
- No regression (measured on the previous base 7b55a9b3): dev vs this change on the 6-prompt
  decode benchmark, identical text, 0.3 % difference, a validated no-change manifest:
  [manifests/final/drift](https://github.com/bobsummerwill/colibri-notes/tree/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/manifests/final/drift).
- **Review:** an earlier version of this series went through a multi-agent adversarial code
  review that raised 60 findings ([F1–F60](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/REVIEW_REPORT.md)); the commit messages name
  the finding each change resolves, and a final stack check approved every PR with no serious
  findings ([STACK_CHECK](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/STACK_CHECK.md)). After rebasing onto 62eaa9dc the series was
  reviewed again for interactions with upstream's newer code (GGUF DeepSeek V4 loading, the serve
  prewarm) and re-tested ([rebase notes](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/REBASE_20261017.md)).
- **Everything else** (environment, every measurement, investigations, decisions):
  [colibri-notes](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/README.md). Developed with AI assistance (Claude Code; see the
  `Co-Authored-By` trailers); every change was reviewed and tested as described here.

### Validation
- [x] `make -C c check` (combined PR #0 tip: 2572 tests, OK, 277 skipped)
- [x] CUDA changes tested with `make -C c dsv4-cuda-test` (RTX 3090, sm_86, `-DCOLI_DSV4_NO_TC`)
- [ ] Performance claims — none (in a stressed setting the change decoded ~3.8% faster than dev;
  not claimed)

### Compatibility
- [x] The default CPU build remains dependency-free
- [x] No model files, generated binaries, or benchmark artifacts are included

---

## PR 0b — `pr0b/dsv4-prefill-bank-lifecycle`: deepseek-v4: GPU prefill bank lifecycle and speculative batches

### Summary
Three fixes to the batched GPU prefill (`COLI_CUDA_MOE_BATCH=1`):
- `coli_v4_gpu_moe_batch_release()` ran only after a finished prefill loop; every other exit of
  `coli_v4_session_generate` (cancelled between segments, an embedding or layer failure, the
  logprob read-out) skipped it. The bank is now freed on every exit and at engine close.
- Speculative verify and replay batches (`batch > 1`) reached the batched GPU MoE through the
  prompt-size hint that `generate()` sets once per request and never cleared, so on a 256-expert
  checkpoint with a 256+ token prompt and speculative decoding on, draft steps after the
  prefill re-created the bank. Those batches now stay off it.
- On a score-router layer whose router bias has no f32 mirror on the device, the GPU top-6
  kernel routed every chunk as if there were no bias (wrong experts, silently). The GPU prefill
  now refuses such a layer, as decode already does.

3 commits (+244/−11, mostly tests): `c/deepseek_v4.c`, `c/deepseek_v4_internal.h`, and source
tests for the bank lifecycle and the speculative guard
(`c/tests/test_v4_moe_batch_lifecycle_source.py`, `c/tests/test_v4_moe_batch_speculative_source.py`).

### How this was tested
- New source tests: `test_v4_moe_batch_lifecycle_source.py` (the bank is released on every
  `generate()` exit and at close) and `test_v4_moe_batch_speculative_source.py` (verify/replay
  batches never reach the bank); `make check` 2,572 tests pass; CUDA build 0 warnings.
- **Review:** an earlier version of this series went through a multi-agent adversarial code
  review that raised 60 findings ([F1–F60](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/REVIEW_REPORT.md)); the commit messages name
  the finding each change resolves, and a final stack check approved every PR with no serious
  findings ([STACK_CHECK](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/STACK_CHECK.md)). After rebasing onto 62eaa9dc the series was
  reviewed again for interactions with upstream's newer code (GGUF DeepSeek V4 loading, the serve
  prewarm) and re-tested ([rebase notes](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/REBASE_20261017.md)).
- **Everything else** (environment, every measurement, investigations, decisions):
  [colibri-notes](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/README.md). Developed with AI assistance (Claude Code; see the
  `Co-Authored-By` trailers); every change was reviewed and tested as described here.

### Validation
- [x] `make -C c check`
- [x] CUDA build sm_86, 0 warnings; `make -C c dsv4-cuda-test`
- [ ] Performance claims — none

### Compatibility
- [x] The default CPU build remains dependency-free
- [x] No model files, generated binaries, or benchmark artifacts are included

---

## PR 0c — `pr0c/dsv4-decode-error-paths`: deepseek-v4: decode mirror rollback and loader-lane cleanup

### Summary
- A decode mirror slot whose upload failed is rolled back, and the card counts as full until the
  next probe. Before, the dead entry stayed in the cache and later lookups pushed GPU decode back
  onto the CPU.
- The dual expert loader's in-flight loads are finished on every MoE exit, removing a stack
  use-after-return and lease leaks on error paths.

2 commits, `c/deepseek_v4.c` (+51/−16).

Related to 0a but independent: without 0a a failed mirror upload's stale CUDA error makes the
next GPU check fail once; 0c's rollback is correct either way, so the two can merge in either
order.

### How this was tested
- `make check` 2,572 tests pass; every commit compiles with `-Wall -Wextra`, 0 diagnostics; CUDA
  build 0 warnings; exercised by the GPU runs of PR #1/#2 (outputs identical to upstream).
- **Review:** an earlier version of this series went through a multi-agent adversarial code
  review that raised 60 findings ([F1–F60](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/REVIEW_REPORT.md)); the commit messages name
  the finding each change resolves, and a final stack check approved every PR with no serious
  findings ([STACK_CHECK](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/STACK_CHECK.md)). After rebasing onto 62eaa9dc the series was
  reviewed again for interactions with upstream's newer code (GGUF DeepSeek V4 loading, the serve
  prewarm) and re-tested ([rebase notes](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/REBASE_20261017.md)).
- **Everything else** (environment, every measurement, investigations, decisions):
  [colibri-notes](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/README.md). Developed with AI assistance (Claude Code; see the
  `Co-Authored-By` trailers); every change was reviewed and tested as described here.

### Validation
- [x] `make -C c check`
- [x] CUDA build sm_86, 0 warnings
- [ ] Performance claims — none

### Compatibility
- [x] The default CPU build remains dependency-free
- [x] No model files, generated binaries, or benchmark artifacts are included

---

## PR 0d — `pr0d/dsv4-dll-export`: deepseek-v4: export `dsv4_cuda_device_unified` from the Windows DLL

### Summary
The Windows `coli_cuda_dsv4*.dll` exports exactly the names `dsv4.def` lists, and
`dsv4_cuda_device_unified` (looked up optionally by the loader) was never listed. Every freshly
built DLL took the "older DLL" fallback and reported every device as discrete, so on a
unified-memory device the decode mirror cache's VRAM reserve guard stayed on and could freeze the
cache at a handful of entries (#1538). Output was never wrong, only placement. `dsv4.def` now
lists it, and a parity test checks that every `dsv4_cuda_*` symbol the loader resolves is
exported.

1 commit (+128, mostly the test).

### How this was tested
- New `test_backend_loader_dsv4_parity.py`: every `dsv4_cuda_*` symbol the loader resolves must be
  in `dsv4.def`. On upstream dev 62eaa9dc two of its seven tests fail, both naming exactly
  `dsv4_cuda_device_unified`; with this change all seven pass. `make check` 2,572 tests pass.
- **Review:** an earlier version of this series went through a multi-agent adversarial code
  review that raised 60 findings ([F1–F60](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/REVIEW_REPORT.md)); the commit messages name
  the finding each change resolves, and a final stack check approved every PR with no serious
  findings ([STACK_CHECK](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/STACK_CHECK.md)). After rebasing onto 62eaa9dc the series was
  reviewed again for interactions with upstream's newer code (GGUF DeepSeek V4 loading, the serve
  prewarm) and re-tested ([rebase notes](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/REBASE_20261017.md)).
- **Everything else** (environment, every measurement, investigations, decisions):
  [colibri-notes](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/README.md). Developed with AI assistance (Claude Code; see the
  `Co-Authored-By` trailers); every change was reviewed and tested as described here.

### Validation
- [x] `make -C c check` (the parity test runs there; Linux only — no Windows build was available)
- [ ] CUDA — n/a
- [ ] Performance claims — none

### Compatibility
- [x] The default CPU build remains dependency-free
- [x] No model files, generated binaries, or benchmark artifacts are included

---

## PR 0e — `pr0e/dsv4-cuda-test-target`: test(dsv4): make `dsv4-cuda-test` pass on generic builds

### Summary
`make -C c dsv4-cuda-test` always failed on generic (non-DeepGEMM) builds:
- the sparse-batch attention test exited 4 because `dsv4_cuda_attention_sparse_batch` is still a
  `return 0` stub; it now prints a SKIP for that step and still runs the output projection;
- the batched-MoE test then needed a 3.2 GiB bank a generic build cannot use; it now skips
  before allocating it.

These follow the four 0a backend commits in the commit text but do not depend on them.

2 commits, the two tests only (`c/tests/test_dsv4_attention_batch_cuda.c`, `c/tests/test_dsv4_moe_batch_cuda.c`; +35/−13).

### How this was tested
- `make -C c dsv4-cuda-test` passes on RTX 3090s (at the PR #1/#2 tips; on dev it stops at the
  attention stub on every generic build);
  `make check` 2,572 tests pass.
- **Review:** an earlier version of this series went through a multi-agent adversarial code
  review that raised 60 findings ([F1–F60](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/REVIEW_REPORT.md)); the commit messages name
  the finding each change resolves, and a final stack check approved every PR with no serious
  findings ([STACK_CHECK](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/STACK_CHECK.md)). After rebasing onto 62eaa9dc the series was
  reviewed again for interactions with upstream's newer code (GGUF DeepSeek V4 loading, the serve
  prewarm) and re-tested ([rebase notes](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/REBASE_20261017.md)).
- **Everything else** (environment, every measurement, investigations, decisions):
  [colibri-notes](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/README.md). Developed with AI assistance (Claude Code; see the
  `Co-Authored-By` trailers); every change was reviewed and tested as described here.

### Validation
- [x] `make -C c dsv4-cuda-test` on an RTX 3090 (sm_86): dense batch passes, attention prints the
  sparse-batch SKIP and passes the output projection, MoE prints its generic-build SKIP
- [x] `make -C c check`

### Compatibility
- [x] The default CPU build remains dependency-free
- [x] No model files, generated binaries, or benchmark artifacts are included

---

## PR 0f — `pr0f/dsv4-docs-sizes`: deepseek-v4: real expert-bank and mirror sizes in the docs

### Summary
Docs and comments gave wrong sizes; they now give the computed ones: 12.75 MiB per decode mirror
and a 3264 MiB expert bank (256 experts). Docs (`docs/deepseek-v4.md`) and comments in `c/deepseek_v4.c` only (+64/−38).

### How this was tested
- Sizes recomputed from the code (12.75 MiB per mirror, 3264 MiB for a 256-slot bank);
  `make check` 2,572 tests pass.
- **Review:** an earlier version of this series went through a multi-agent adversarial code
  review that raised 60 findings ([F1–F60](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/REVIEW_REPORT.md)); the commit messages name
  the finding each change resolves, and a final stack check approved every PR with no serious
  findings ([STACK_CHECK](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/STACK_CHECK.md)). After rebasing onto 62eaa9dc the series was
  reviewed again for interactions with upstream's newer code (GGUF DeepSeek V4 loading, the serve
  prewarm) and re-tested ([rebase notes](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/REBASE_20261017.md)).
- **Everything else** (environment, every measurement, investigations, decisions):
  [colibri-notes](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/README.md). Developed with AI assistance (Claude Code; see the
  `Co-Authored-By` trailers); every change was reviewed and tested as described here.

### Validation
- [x] `make -C c check`

### Compatibility
- [x] The default CPU build remains dependency-free
- [x] No model files, generated binaries, or benchmark artifacts are included

---

## PR #1 — `pr1/reap-prefill`: deepseek-v4: batched GPU prefill for expert-pruned (REAP) checkpoints, faster generic kernels

**Depends on PR 0a–0f** (this branch sits on all six, combined: `pr0/dsv4-fixes`). 8 commits on top.

### Summary
The batched GPU prefill MoE (`COLI_CUDA_MOE_BATCH=1`, off by default) only took 256-expert
checkpoints. Expert-pruned ones such as `puwaer/DeepSeek-V4-Flash-0731-reap-150b` (132 routed
experts) fell back to the CPU union, ~9 tok/s prefill on an RTX 3090 box.

- The bank is sized by `dsv4_cuda_expert_bank_slots`: 132 slots / 1683 MiB for REAP-150B on
  generic builds; DeepGEMM builds keep 256.
- Pruned, hash and missing-bias-mirror layers route on the CPU bf16 router inside the batched
  path (the GPU top-6 kernel assumes 256 logits). Routing is bit-identical to the CPU union.
- The default decode-mirror VRAM reserve with the bank on is sized from the model at engine
  open (bank ×1 or ×2 + 600 MiB headroom + 64 mirrors: 3100 MiB for 132 experts, 4681 MiB for
  256). A GGUF checkpoint, which never gets a bank, reserves none
  for it.
- The bank keeps the RAM expert store's routing books (usage, pin ramp, repins, HITS/heat,
  `.coli_usage`) as the CPU union's lookups do, without extra reads (review finding F42: decode
  after a bank prefill was 7.4 % slower because the store saw almost no requests; this recovers
  67–79 % of that gap).
- Two bit-identical generic kernels: expert-major fp4 for the batched MoE
  (`DSV4_MOE_EXPERT_MAJOR`, default on) and warp-per-row for short fp8 dense batch matmuls
  (`DSV4_MM_WARP`, auto = I ≤ 1024). `_VERIFY=1` cross-checks old against new.
- `V4_MOE_BATCH_PROF=1` prints a per-chunk time split.
- Docs: REAP prefill, the kernels, and the measurements below.

### Measurements
2× RTX 3090 (one used; CUDA 12.0, sm_86, `NO_TC=1`), Ryzen 9 9950X3D, 62 GB RAM, SN850X;
REAP-150B, `--ram 40 --ctx 8192`, greedy; a fresh server per sample, 3 samples per arm
interleaved (plus one discarded warm-up), medians; `.coli_usage` empty (the serve prewarm reads
nothing at this `--ram`).

| prompt | upstream dev 62eaa9dc (CPU union) | this PR | this PR + `COLI_CUDA_MOE_DOUBLE=1` |
|---|---|---|---|
| 1,992 tokens, time to first token | 128.9 s (15.4 tok/s) | 76.5 s (26.0 tok/s), **1.69×** | 64.1 s (31.1 tok/s), **2.01×** |
| 3,005 tokens, time to first token | 204.4 s (14.7 tok/s) | 110.1 s (27.3 tok/s), **1.86×** | 98.0 s (30.6 tok/s), **2.09×** |
| decode after the prompt | 2.29 / 2.94 tok/s | 2.22 / 2.92 tok/s | 2.09 / 2.84 tok/s |

Answer texts were identical to upstream dev's in every sample (both requests, all arms).
Manifests (`c/experiment_manifest.py` format): engine A→B and `COLI_CUDA_MOE_DOUBLE` B→C for both
prompts, with hashed raw evidence.

For reference, on the previous base 7b55a9b3 the speedups were 2.67×/2.82× (3.14×/3.15× with the
double bank); upstream's own prefill has since become ~1.6–1.7× faster (#2069).

### How this was tested
- New tests: `test_dsv4_generic_bitwise_cuda.c` (old vs new kernels bit-identical on the GPU),
  `test_dsv4_bank_bytes_cuda.c` (bank sizing), `test_v4_moe_batch_books_source.py`, and two store
  cases in `test_deepseek_v4.c` (the bank's routing books match the CPU union's after every chunk,
  and the saved `.coli_usage` files are byte-identical); `make check` 2,579 tests pass;
  `dsv4-cuda-test` passes on an RTX 3090.
- Bit-identity of the two new kernels: `_VERIFY=1` modes report 0 mismatching bits over full
  prefills; the review re-checked them at runtime over tens of millions of outputs.
- Benchmarks: four manifests validated with `c/experiment_manifest.py`, raw evidence hashed:
  [manifests/final4](https://github.com/bobsummerwill/colibri-notes/tree/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/manifests/final4); all measurements and the earlier base:
  [measurements.md](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/measurements.md). The F42 decode-after-prefill investigation (cause,
  fix, +6.3 %): [manifests/final2](https://github.com/bobsummerwill/colibri-notes/tree/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/manifests/final2).
- **Review:** an earlier version of this series went through a multi-agent adversarial code
  review that raised 60 findings ([F1–F60](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/REVIEW_REPORT.md)); the commit messages name
  the finding each change resolves, and a final stack check approved every PR with no serious
  findings ([STACK_CHECK](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/STACK_CHECK.md)). After rebasing onto 62eaa9dc the series was
  reviewed again for interactions with upstream's newer code (GGUF DeepSeek V4 loading, the serve
  prewarm) and re-tested ([rebase notes](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/REBASE_20261017.md)).
- **Everything else** (environment, every measurement, investigations, decisions):
  [colibri-notes](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/README.md). Developed with AI assistance (Claude Code; see the
  `Co-Authored-By` trailers); every change was reviewed and tested as described here.

### Validation
- [x] `make -C c check`: 2579 tests, OK (277 skipped)
- [x] CUDA: build sm_86 0 warnings; `make -C c dsv4-cuda-test` passes on an RTX 3090 (0 warnings)
- [x] Performance claims include hardware, commands and repeatable measurements
- [x] Experiment manifests with raw evidence (validated with `c/experiment_manifest.py`)

### Compatibility
- [x] The default CPU build remains dependency-free (everything except the bit-identical kernels
  is behind `COLI_CUDA_MOE_BATCH=1`)
- [x] No model files, generated binaries, or benchmark artifacts are included

---

## PR #2 — `pr2/two-gpu-decode`: deepseek-v4: two-GPU decode and VRAM-first expert lookups

**Depends on PR #1** (and so on 0a–0f). 8 commits on top.

### Summary
All new behaviour is opt-in:
- `V4_GPU_RESIDENT_FIRST=1`: an expert already mirrored in VRAM is computed from the mirror
  without a RAM-store lookup; the store still records the routing (history, HITS, heat) and its
  RAM pin ranking leaves mirrored experts out, so RAM holds experts that are not in VRAM.
- `DSV4_CUDA_PEER_DEVICE=N`: a second mirror pool on GPU N (needs peer access, e.g. NVLink).
  `DSV4_CUDA_PEER_COMPUTE` (default 1) runs the experts mirrored there on that GPU,
  bit-identically; a failing peer is latched off and the token recomputed. GPU N is probed
  before it is touched and a failed init is rolled back.
- `DSV4_CUDA_PIN_CAP_GB` caps the page-locked expert slabs (default unchanged, 20 GiB).
- `V4_SERVE_STATS_LOG=1` prints per-turn decode stats and decode-MoE counters.
- Mirror caches keep their VRAM reserve between probes and drop failed uploads. This is the one
  change with every switch off: an explicit or no-bank reserve is now kept once 8 mirrors are
  live.

### Measurements
2× RTX 3090 with an NVLink bridge, REAP-150B, `--ram 40`, six 300-token answers, greedy,
`COLI_CUDA_MOE_BATCH=1`: one GPU **3.01 tok/s** → `V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1`
(peer compute) **4.34 tok/s (+44 %)**; all six answers identical. One sample per configuration
(a sanity check, not a manifest); on the previous base the same comparison gave 2.90 → 4.08
(+41 %).

### How this was tested
- New tests: `test_dsv4_peer_init_cuda.cu` and `test_dsv4_peer_moe_cuda.c` (two-GPU init,
  rollback, peer compute bit-identical to one GPU; they skip without two peer-capable GPUs),
  `test_v4_peer_init_source.py`, and a VRAM-first store test in `test_deepseek_v4.c`;
  `make check` 2,586 tests pass; `dsv4-cuda-test` and `cuda-test` pass on 2× RTX 3090 + NVLink.
- Earlier per-switch measurements with validated manifests:
  [manifests/final/pr2](https://github.com/bobsummerwill/colibri-notes/tree/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/manifests/final/pr2); today's re-check on the new base:
  [manifests/final4/pr2-NOTE.md](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/manifests/final4/pr2-NOTE.md).
- **Review:** an earlier version of this series went through a multi-agent adversarial code
  review that raised 60 findings ([F1–F60](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/REVIEW_REPORT.md)); the commit messages name
  the finding each change resolves, and a final stack check approved every PR with no serious
  findings ([STACK_CHECK](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/STACK_CHECK.md)). After rebasing onto 62eaa9dc the series was
  reviewed again for interactions with upstream's newer code (GGUF DeepSeek V4 loading, the serve
  prewarm) and re-tested ([rebase notes](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/review/REBASE_20261017.md)).
- **Everything else** (environment, every measurement, investigations, decisions):
  [colibri-notes](https://github.com/bobsummerwill/colibri-notes/blob/2c8e5e61a63d3f6771bb7405bc7e08c37d23d8f9/README.md). Developed with AI assistance (Claude Code; see the
  `Co-Authored-By` trailers); every change was reviewed and tested as described here.

### Validation
- [x] `make -C c check`: 2586 tests, OK (277 skipped)
- [x] CUDA: build sm_86 0 warnings; `make -C c dsv4-cuda-test` and `make -C c cuda-test` pass on
  two RTX 3090s with NVLink; the peer tests skip without two peer-capable GPUs
- [x] Performance claims include hardware, commands and repeatable measurements

### Compatibility
- [x] The default CPU build remains dependency-free
- [x] No model files, generated binaries, or benchmark artifacts are included
