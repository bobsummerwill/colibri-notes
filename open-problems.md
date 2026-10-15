# Open problems / next steps

## To finish before the PRs are submitted
The restructure is done (2026-10-15): pr0/dsv4-fixes a2c8ba7a -> pr1/reap-prefill 93b74183 -> pr2/two-gpu-decode d0e19de2, all reviewed, `make check` passes at each tip (2378 / 2379 / 2384), CUDA builds clean. Remaining:
1. **PR #0 GPU validation on dev**: show with the logit-hash probe (`tools/probes/`) that dev 7b55a9b3 drifts on decode (upload race) and pr0/dsv4-fixes does not. Needs a GPU window (llama stopped).
2. **Final PR #1 benchmark** on 93b74183 (update `tools/bench/run_pr1_final.sh` to build that SHA), arms B and C (DOUBLE), 3 cold samples each, plus manifests. Its code differs from the benchmarked a406901b only by carried PR #0 fixes (F26, F4/F37, F27) and text.
3. **PR #2 decode re-measurement** on d0e19de2: >=3 samples per arm (one-GPU, X1, X2, X3), full text compared, DOUBLE unset, environment stated (F43-F45).
4. **PR descriptions** from `review/PR_SUMMARIES_DRAFT.md` (drafts only, never submitted by us).
5. **Rebase onto current upstream dev before submitting.** On 2026-10-15 upstream dev = 11dbb2c6, 31 commits past our base; `git merge-tree` is clean for all three tips and the merged trees build (CPU) — re-run make check + CUDA build after rebasing. Optional nit: two messages cite upstream '#1538' by number.

GPU-test caution: `c/tests/test_dsv4_decode_cuda.c` (dsv4-cuda-loader-test) allocates ~3.2 GiB on the GPU — never run it on shadowfax while llama.cpp is up (a verifier did once on 2026-10-15; llama stayed healthy).

## Untested / known gaps
- **F18:** the peer-compute success path never checks the peer context for asynchronous faults. A sticky peer fault could let a wrong token through or hang. The hang case has not been tested.
- Windows build (the loader and `dsv4.def` parity test passes on Linux only), DeepGEMM / tensor-core builds (sm_86 here uses `NO_TC=1`), and the full 256-expert DeepSeek V4 checkpoint have not been tested.
- Upstream test `test_dsv4_attention_batch_cuda` uses a stub that exits 4. This is pre-existing and not caused by this work.

## Performance ideas not pursued
- Decode is now bound by attention (~107 ms/token) and the remaining expert fetch/attach. Next steps would be attention kernels and smarter expert placement across the two GPUs.
- The PCIe 4.0 x8 links (~12 GB/s) cap expert refill. Pinned staging buffers are already used where possible.
