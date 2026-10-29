# final36: #2405 (PR #2) f1e56b00 vs dev d68e16b6, 2x RTX 3090 (2026-10-28 23:32 .. 2026-10-29 01:30 PDT)

- Trees: base = dev d68e16b6 (bin b3845277), pr2 = f1e56b00 (11c40038 + merge of d68e16b6; bin d506cd1f). Builds:
  `builds/status` (CUDA sm_86 NO_TC, 0 warnings each).
- Harness: final30's (`final_common.sh`, `final36_tools.py`, `run_final36.sh`). The first line of `bench/summary.txt`
  still says "FINAL30 SUMMARY (same base c1c06ac6 ...)": that is the copied title string, not the trees. The SHAs above
  and in `bench/run.log` are the real ones.
- Arms: B1/P1/P2 REAP decode (6x300, 2 samples); DB1F/DP1F/DP2F full fdec (3x300, 2 samples); B1F/P1F/P2F full long
  prompts (TTFT, 1 sample); S7 = P2 + `DSV4_CUDA_PEER_FAULT=3000:sticky` (note-only, must log the latch).
  P* = `V4_GPU_RESIDENT_FIRST=1`; *2 adds `DSV4_CUDA_PEER_DEVICE=1`.
- GPU power limits 500 W / 450 W (final27-34 ran at 300 W).
- Result: GATES: PASS; 0 non-identical texts. P1/B1 1.028x, P2/B1 1.436x, DP1F/DB1F 1.022x, DP2F/DB1F 1.268x, long-prompt
  TTFT 1.00x. S7: 6/6 answers, 300 tokens each. `bench/meta/`: `dsv4-cuda-test` and `cuda-test` on pr2, rc=0.
- `f1e56b00-*.log`: computer4 `make -C c check` (2,795 OK, 305 skipped) and the `-Wall -Wextra` syntax checks (empty =
  no diagnostics).
- Full bench.tgz stays on shadowfax (`~/colibri-bench/final36/bench.tgz`, sha256 in `bench.tgz.sha256`).
