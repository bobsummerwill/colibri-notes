# Open problems / next steps

## To finish before the PRs are submitted (2026-10-17)
1. Bob: review `review/PR_DESCRIPTIONS_20261017.md` and the `rebased/*` branches on the fork; decide whether to force-push them over the old names (or open the PRs from the `rebased/*` names).
2. Rebase again if upstream dev moves before submission (it moved 298 commits in 6 days): re-run `make check`, the CUDA build and, for code conflicts, `run_final4.sh`.
3. Attach `manifests/final4/` (4 validated manifests, `evidence/`, `bench.tgz`) to PR #1; PR #2's numbers are a one-sample sanity check (a manifest would need >=3 samples per arm).

## Untested / known gaps
- **F18:** the peer-compute success path never checks the peer context for asynchronous faults. A sticky peer fault could let a wrong token through or hang. The hang case has not been tested.
- Windows build (the loader and `dsv4.def` parity test passes on Linux only), DeepGEMM / tensor-core builds (sm_86 here uses `NO_TC=1`), and the full 256-expert DeepSeek V4 checkpoint have not been tested.
- Upstream test `test_dsv4_attention_batch_cuda` uses a stub that exits 4. This is pre-existing and not caused by this work.

## Performance ideas not pursued
- Decode is now bound by attention (~107 ms/token) and the remaining expert fetch/attach. Next steps would be attention kernels and smarter expert placement across the two GPUs.
- The PCIe 4.0 x8 links (~12 GB/s) cap expert refill. Pinned staging buffers are already used where possible.
