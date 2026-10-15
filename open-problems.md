# Open problems / next steps

## To finish before the PRs are "ready"
1. **Three-PR restructure.** Branches: `pr0/dsv4-fixes`, `pr1/reap-prefill-v2`, `pr2/two-gpu-decode`. When the workflow finishes:
   - check that the PR #1 tip tree == a406901b and the PR #2 tip tree == fe28d80a;
   - run `make check` on each tip;
   - force-push to the fork.
2. **PR #0 GPU validation on dev.** Using the logit-hash probe (`tools/probes/`), show that plain dev 7b55a9b3 drifts on decode (from the upload race) and that dev + PR #0 does not.
3. **Final PR #1 benchmark.** Run it on the final PR #1 tip SHA:
   - update `tools/bench/run_pr1_final.sh` to build that SHA (it currently builds a406901b in `/tmp/cw/bench-pr1f`);
   - add a manifest for C (DOUBLE).
4. **PR #2 decode re-measurement.** Use the final binary and >=3 samples per arm. Arms: one-GPU, X1, X2, X3. Compare full text and state the environment. Do not set DOUBLE in the decode arms (F43-F45).
5. **PR descriptions.** Draft one for each of the three PRs (`review/PR1_DESCRIPTION_DRAFT_v1.md` is outdated). Drafts only — never submitted.

## Untested / known gaps
- **F18:** the peer-compute success path never checks the peer context for asynchronous faults. A sticky peer fault could let a wrong token through or hang. The hang case has not been tested.
- Windows build (the loader and `dsv4.def` parity test passes on Linux only), DeepGEMM / tensor-core builds (sm_86 here uses `NO_TC=1`), and the full 256-expert DeepSeek V4 checkpoint have not been tested.
- Upstream test `test_dsv4_attention_batch_cuda` uses a stub that exits 4. This is pre-existing and not caused by this work.

## Performance ideas not pursued
- Decode is now bound by attention (~107 ms/token) and the remaining expert fetch/attach. Next steps would be attention kernels and smarter expert placement across the two GPUs.
- The PCIe 4.0 x8 links (~12 GB/s) cap expert refill. Pinned staging buffers are already used where possible.
