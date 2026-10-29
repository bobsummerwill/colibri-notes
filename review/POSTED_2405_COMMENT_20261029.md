Marking this ready for review. What changed since the draft:

**Merged current dev (d68e16b6) in: f1e56b00, no conflicts.** The 11 patches are unchanged. dev's V4 changes since c1c06ac6 don't touch the mirror or peer code.

**Re-checked on f1e56b00:**
- `make -C c check` (gcc 13.3.0): 2,795 tests OK (305 skipped).
- `deepseek_v4.c` (GPU tier, and with `-DCOLI_VULKAN`) and `tests/test_deepseek_v4.c` compile with `-Wall -Wextra` and no diagnostics. The CUDA build (sm_86) has no warnings.
- On the 2x RTX 3090: `make -C c dsv4-cuda-test` (peer MoE OK, sticky latch OK) and `make -C c cuda-test` pass.

**Re-ran the benchmark against dev d68e16b6** (same harness, fresh server per sample, interleaved arms):

| | REAP-150B, 1 GPU | REAP-150B, 2 GPUs | V4 Flash, 1 GPU | V4 Flash, 2 GPUs |
|---|---|---|---|---|
| dev d68e16b6 | 3.016 tok/s | | 2.642 tok/s | |
| this PR | 3.101 (1.03x) | 4.329 (**1.44x**) | 2.699 (1.02x) | 3.348 (**1.27x**) |
| previous run, on c1c06ac6 | 1.03x | 1.44x | 1.02x | 1.27x |

- Every answer is byte-identical to dev's.
- Time to first token on long prompts (1,992 and 3,005 tokens) is 1.00x dev's, one GPU or two.
- The fault test (`DSV4_CUDA_PEER_FAULT=3000:sticky`) on this tip: all six answers finish with 300 tokens, on the host after the fault.
- Evidence: [final36](https://github.com/bobsummerwill/colibri-notes/tree/NOTESSHA/manifests/shadowfax-20261028/final36). The description is updated with these numbers.

**Merge order with #1766.** The two are independent: no shared commits, and each applies to dev on its own. They do overlap textually in five files (`c/Makefile`, `c/backend_cuda_dsv4.cu`, `c/deepseek_v4.c`, `c/deepseek_v4_internal.h`, `docs/deepseek-v4.md`), so whichever lands second needs a conflict resolution.
- [`samebase/cplus`](https://github.com/bobsummerwill/colibri/commits/66982895) on my fork is that resolution, done on c1c06ac6. It is the fastest combination we measured: REAP 1.96x dev with two GPUs, 1.27x with one.
- The two multi-GPU modes are alternatives: both claim GPU 1, so `DSV4_CUDA_DEVICES` (#1766) and `DSV4_CUDA_PEER_DEVICE` (this PR) shouldn't be set together.
- #1766 conflicts with current dev already. On #1766 I've offered to help ([comment](https://github.com/JustVugg/colibri/pull/1766#issuecomment-6462594190)): its author can cherry-pick the fixes from `samebase/m1766-fixes`, or I can put together the combined series.
- If this PR goes first, I'm happy to rebase the combined branch onto the result. If #1766 goes first, I'll merge it into this branch and re-run the same checks.
