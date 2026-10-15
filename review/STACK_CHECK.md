# Final stack check (2026-10-15)

Stack: dev 7b55a9b3 -> pr0/dsv4-fixes a2c8ba7a (13) -> pr1/reap-prefill 93b74183 (7) -> pr2/two-gpu-decode d0e19de2 (8). Verdict: **ready**.

Workflow log:
- pr0-polish round 0: approve/approve serious=0 minor=1
- pr2 round 0: approve/approve serious=0 minor=4
- pr0-polish round 1: approve/approve serious=0 minor=0
- pr2 round 1: approve/approve serious=0 minor=1
- pr1-polish round 0: changes_requested/approve serious=0 minor=2
- pr1-polish round 1: approve/approve serious=0 minor=0
- pr2-join round 0: approve serious=0 minor=0
- stack: ready; serious=0

## Issues
- **minor** — My verification run on shadowfax, GPU0, at pr0/v4. This is not a defect in the stack.: I broke the memory limit once. I added c/tests/test_dsv4_decode_cuda.c (the source of dsv4-cuda-loader-test, not part of dsv4-cuda-test) to the tiny-test list without checking its size first. Its MoE step uploads 256 decode experts (about 3.2 GiB). My guard sampled +3558 MiB on GPU0 before it killed the test, so I went past the 1.5 GB cap. GPU0 had 4198 MiB free beforehand. llama.cpp stayed healthy (/health ok), and GPU0 was back to 4198 MiB free afterwards. I did not run it at pr1/v4 or pr2/v4, and none of the results below depend on it. Fix: Nothing to change in the stack. Treat test_dsv4_decode_cuda as a 3.2 GiB test and leave it out of runs on shadowfax. Optionally give its header a size note, as the other dsv4 CUDA tests have.
- **nit** — Commit messages a6e30d2c (export dsv4_cuda_device_unified) and 2f80c63d (mirror caches keep their VRAM reserve): Both messages refer to the upstream merged PR '#1538' by number. That is upstream's change 78a9ee94, 'deepseek_v4: mirror cache on unified memory and the frozen-cache recycle (#1538)'. The rule bans PR numbers; its examples are about this stack, and dev's own code comments already cite #1538, so this is borderline. Fix: Optional: cite the upstream commit subject instead of '#1538', or leave it as it is.
- **nit** — Base of the stack: upstream/dev has moved to 11dbb2c6, 31 commits past 7b55a9b3. All three tips still merge cleanly (see upstream_merge). Fix: Optional: rebase onto 11dbb2c6 before submitting, then re-run make check and the CUDA build.
- **nit** — c/Makefile dsv4-cuda-test (an existing problem, outside the stack): c/Makefile has no NO_TC switch. On CUDA 12.0 (shadowfax) make dsv4-cuda-test only builds with an NVCCFLAGS override that adds -DCOLI_DSV4_NO_TC. Commit 3675850c already says so. Fix: None needed for this stack. A NO_TC=1 switch in c/Makefile would be a separate change.

## Ancestry, hygiene, and where each PR0 fix lives

ITEM 1: ANCESTRY AND COMMIT HYGIENE (checked with git on computer4 after fetching shadowfax and upstream)
- Ancestry holds: dev 7b55a9b3 < pr0/v4 a2c8ba7a (13 commits) < pr1/v4 93b74183 (7) < pr2/v4 d0e19de2 (8). That is 28 commits, 0 merge commits (git rev-list --merges is empty). The shadowfax branches pr0/v4, pr1/v4 and pr2/v4 point at exactly these SHAs. Tree checks: pr1/v4 equals ref/pr1-target-v4 b92b2f4d, and pr2/v4 equals ref/pr2-target-v4 2e2d66a8.
- Authorship: all 28 commits have author and committer Bob Summerwill <bob@summerwill.net>. Each body ends with 'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>' exactly once.
- Subjects are 59 to 72 characters. Four are exactly 72: 9bfc6df7, d8414e1c, c3f40909, f15d2748. Every subject starts with 'deepseek-v4: ' or 'test(dsv4): '.
- No 'PR #n', PR0/1/2, pr1/-style refs, wip, fixup, squash or integration commits. Every hit on 'draft' means speculative-decoding drafts. The only numeric refs are upstream '#1538' (nit) and the nvcc warning '#177-D'.
- Commits that rework an earlier fix name it by subject: ec10ca38 (F2, F23), 63f1fec1 (the docs commit), 32b39f7f (F26), 296d8f7f and 2f80c63d (F4).
- The PR2 series' separate F26 commit (2f7c7544) is gone. PR0 now has 13 commits: pr0/v3's 12 plus the new 3675850c 'test(dsv4): skip the sparse-batch attention step while it is a stub'.

ITEM 2: EACH PR0 FIX AND WHERE IT IS AT EACH TIP (pr0 / pr1 / pr2)
1. Upload ordering (b2c8ca61). The h2d() helper, upload() weight and scale through h2d, refill always cudaMemcpyAsync on c->stream, the TP2 bank upload async plus drain, and RoPE, expert table and hash router through h2d.
   - Present and identical at all three tips.
   - The only legacy-stream cudaMemcpy calls left at every tip are the two device-to-host debug dumps, which run after an explicit sync.
   - pr2 changes only the h2d comment (it now names DSV4_CUDA_PIN_CAP_GB).
2. Bank-create clears the last error (9bfc6df7). (void)cudaGetLastError() on the failure path is identical at all tips.
3. F27 alloc_ok (13c0aea2). Covers buf, upload (weight, scale, DeepGEMM scale), dsv4_cuda_activation_create, kv_ring_append and kv_comp_append.
   - All 7 uses are present at all tips.
   - pr1 and pr2 add 2 more uses (the expert-major and MM_WARP verify buffers), 9 in total.
   - The 20 cudaMalloc calls still using plain ok() are the same set at all three tips: device init, KV cache create, expert set and table, the bank (which clears its own error), and the hash router.
4. F4/F37 (a9f4d388).
   - pr1: identical. The found scan requires tensors, the guard state is hoisted into statics with upload_failed, a failed slot is rolled back with last_free_mb=0, and the frozen warning is suppressed after a failed upload.
   - pr2: replaced by the validated fe28d80a final form (296d8f7f and 2f80c63d). State is per cache, the rollback uses an 'appended' flag and clears grown_bytes, the guard covers an empty cache whenever last_free_mb >= 0, the per-cache v4_mirror_warn_frozen is skipped while upload_failed is set, and a miss without host data is refused before a victim is freed. This form is stronger and also fixes F16.
5. F10/F11 (d8414e1c). session_generate_impl plus a release in the public coli_v4_session_generate, and release first in coli_v4_gpu_engine_close. Identical at all tips. The lifecycle source test passes at every commit.
6. F2 (1b00a4c7). coli_v4_gpu_moe_batch_prefill(1/0) is set around target_batch_impl and reset to -1 after. The union refusal 'if (!v4_gpu_moe_batch_in_prefill) return -1;' comes before bank create and the bank2 pthread_create.
   - Identical at all tips; in pr1/pr2 the refusal still comes before the geometry checks.
   - In ec10ca38 its source checks moved from test_v4_moe_batch_speculative_source.py, which was deleted, into test_v4_moe_batch_lifecycle_source.py. The three checks are the same; the generate check now reads session_generate_impl.
7. F23 (c3f40909).
   - pr0: the union refuses such a layer to the CPU union.
   - pr1/pr2: replaced by the ec10ca38 form. A score layer that has a bias but no bias mirror routes inside the batched path on the CPU bf16 router with the host bias ('hashed || n_routed_experts != 256 || (raw_bias && !bias)'). It gives the same guarantee (the layer is never routed without its bias) and keeps the bank in use, so it is at least as strong.
8. F26 (866b695e).
   - pr1: identical (v4_lanes_finish, calloc'd views, the 'views &&' guard on the release loop).
   - pr2: the fe28d80a form from 32b39f7f. v4_lanes_finish runs after the fetch pass and on every exit of v4_views_fetch_host. Only views marked in the calloc'd leased[] array are released, behind 'if (views)' and 'leased &&'. This is stronger: a view without a lease is never released.
9. F28 (879ced85).
   - pr1: the same memset in dsv4_cuda_shutdown.
   - pr2: the memset is in dev_release(), which both shutdown and init_fail use, so it also covers a failed init. Stronger.
10. F1 (a6e30d2c).
   - dsv4_cuda_device_unified is in dsv4.def at all tips.
   - pr1 adds expert_bank_slots and slot_bytes; pr2 adds peer_probe, peer_access, peer_compute and peer_healthy. Each is in dsv4.def and resolved by the loader.
   - The parity test passes at every commit from a6e30d2c on.
11. F49 (2b631992). The generic-build skip comes before the bank at all tips, and it printed SKIP on the GPU at all three tips.
12. Attention stub skip (3675850c). Identical at all tips. It ran at all three: sparse-batch SKIP, output projection OK.
13. Docs sizes F40/F41 (a2c8ba7a). 12.75 MiB and 3264 MiB appear at every tip, and no '2.2 GB' or '~8 MB' is left. In pr1, 63f1fec1 replaces the manual DSV4_CUDA_VRAM_RESERVE_MB=4700 advice with the bank-sized default, as intended.

No PR0 fix is missing or weakened at either tip. No fixes were applied: this was a read-only verification.

## Code differences vs the validated pr1/fixed and pr2/fixed

PR1/V4 vs a406901b (10 files, 187+/90-; tree equals ref/pr1-target-v4)
Code:
1. backend_cuda_dsv4.cu, kv_ring_append and kv_comp_append: ok -> alloc_ok. Carried fix (F27, 13c0aea2).
2. backend_cuda_dsv4.cu, dsv4_cuda_activation_create: ok -> alloc_ok. Carried fix (F27; this change is also new in pr0/v4 compared with pr0/v3).
3. deepseek_v4.c, moe_token_pipeline: v4_lanes_finish helper and call, calloc'd views, 'views &&' guard on the release loop. Carried fix (F26, 866b695e).
4. deepseek_v4.c, v4_gpu_expert_attach_cached_ex: found scan requires tensors, static guard state hoisted with upload_failed, frozen warning guarded, slot rollback. Carried fix (F4/F37, a9f4d388).
5. deepseek_v4.c, v4_bank_double_on: the dead extern forward declaration is gone and the function is static again, as on dev. Carried review nit F56: it is identical to the validated fe28d80a, which made the same fix in bc446c98. Linkage only; git grep finds no other reference, and every build links.
Text:
- c/Makefile comment above dsv4-cuda-test (CI builds none of these tests).
- run_mm_batch DSV4_MM_WARP comment (I a multiple of 256, F51).
- h2d comment ('20 GiB page-lock cap, or DSV4_CUDA_PIN_HOST=0'; PIN_CAP_GB only arrives in PR2).
- docs/deepseek-v4.md: 8 GB row ~0.9 GiB and ~25 mirrors; the N x 1024 MiB card note; MM_WARP wording; 'one run per cell'; the REAP footnote (F46/F47); per-prompt decode (F42).
- Parity-test docstring, test_dsv4_generic_bitwise_cuda.c header comment, test_dsv4_moe_batch_cuda.c comment (GB -> GiB).
Tests:
- test_dsv4_attention_batch_cuda.c: carried from 3675850c.
- test_v4_moe_batch_lifecycle_source.py: find() helper and failure messages, carried from pr0/v4.
- test_v4_bank_pair.c: one new assertion, 256-slot DeepGEMM double bank = 8329 MiB (600 + 6912 + 816), the figure 63f1fec1's message states.
Unexpected: none.

PR2/V4 vs fe28d80a (11 files, 150+/85-; tree equals ref/pr2-target-v4; c/deepseek_v4.c is byte-identical)
Code:
1. backend kv ring, kv comp and activation_create: alloc_ok. Carried fix (F27).
2. c/Makefile dsv4-cuda-test: the peer_moe build and run move from last to right after peer_init. This only changes test order, and b1fefd77 states it ('The two peer tests run first ... ahead of the single-GPU kernel tests').
Text:
- Makefile comment, MM_WARP comment, h2d comment ('past DSV4_CUDA_PIN_CAP_GB, or all of them with DSV4_CUDA_PIN_HOST=0').
- test_dsv4_peer_moe_cuda.c SKIP message now also names DSV4_CUDA_BATCHED=0.
- test_v4_peer_init_source.py docstring: "exactly PR #1's single init" becomes "exactly one init, dsv4_cuda_init(&device, 1)", which removes a PR-number reference.
- docs: the same corrections as pr1, plus the 'To be re-measured' note reworded.
Tests: the same six test-file diffs as pr1 (the two diffs have the same md5).
Unexpected: none.

For reference, pr0/v4 vs pr0/v3 c0c756fa: the activation_create alloc_ok, the new attention-test skip, GB -> GiB in a comment, and the find() helper in the two source tests.

## Tests run

All runs below were done in this session: computer4 for git and CPU work, shadowfax for CUDA, using /tmp/cw/r5-v4ver-* worktrees. The worktrees are removed now; logs are kept in shadowfax:/tmp/cw/r5-v4ver-{pr0,pr1,pr2,walk}-logs and the computer4 scratchpad.

MAKE CHECK (computer4, from each worktree root, the three run in parallel)
- pr0/v4: rc=0, 2378 tests OK, 253 skipped (847 s). Matches pr0/v3's 2378.
- pr1/v4: rc=0, 2379 OK, 253 skipped (852 s). Matches pr1/v3's 2379.
- pr2/v4: rc=0, 2384 OK, 253 skipped (851 s). Matches fe28d80a's 2384.
- All three are above dev's 2366, so nothing that passes on the references fails.
- The only 'warning' lines are lto-wrapper notes and expected test output. The 'make: *** backend_cuda.o Error 1' lines are expected output of setup tests that simulate a failing build. These lines are the same at all three tips.

PER-COMMIT CPU CHECKS (computer4, dev plus all 28 commits)
- Clean deepseek-v4-clean, then a CPU-only engine build -j8: rc=0 and 0 warnings at every commit.
- All 33 engine units with gcc -fsyntax-only, the project's CFLAGS (-Wall -Wextra) and -DCOLI_V4_GPU_TIER, and the GPU unit again with -DCOLI_VULKAN: 0 failures, 0 warnings at every commit.
- The dsv4 loader parity test passes at every commit from a6e30d2c on.
- The test_v4_*_source.py tests that exist at each commit (lifecycle, speculative, moe_reason, peer_init) pass at every commit.

CUDA ENGINE BUILDS (shadowfax, CUDA 12.0, make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j8 after a clean)
- All three tips: rc=0, 0 warnings, 0 errors (lto notes excluded).
- Per commit: the 23 commits that touch .cu, .h, deepseek_v4.c or a Makefile all build with 0 warnings and 0 errors. Skipped as docs, tests or .def only: a6e30d2c, 2b631992, 3675850c, 93b74183, d0e19de2.

GPU TESTS (shadowfax, generic sm_86 NO_TC build)
Conditions:
- Before every test GPU0 had 4198 MiB free and GPU1 1409 MiB free. They were the same afterwards, and llama.cpp /health stayed ok.
- Peaks come from nvidia-smi samples every ~50-150 ms, so for short tests they are lower bounds.
- Single-GPU tests ran with CUDA_VISIBLE_DEVICES=0 and a 1430 MiB guard. Two-GPU tests ran with 0,1 and a 286 MiB per-card guard.
Tiny tests, linked against the engine's backend object:
- test_backend_cuda_dsv4 (all tips): pass, worst_abs=0.
- test_dsv4_dense_batch_cuda (all tips): OK.
- test_dsv4_attention_batch_cuda (all tips): 'sparse batch SKIP' (the stub returns 0 without writing output), then 'OK (output projection; sparse batch skipped)'.
- test_dsv4_sparse_attn_batch_cuda (all tips): OK, max |diff| 4.88e-4, bf16 relative diff 1.19e-5, peak +282 MiB.
- test_dsv4_moe_batch_cuda (all tips): SKIP on the generic build before init, nothing allocated.
- test_dsv4_bank_bytes_cuda (pr1, pr2): OK. Slot 13369392 bytes, the 16-slot bank took 204 MiB, mirror 13369344 bytes.
- test_dsv4_generic_bitwise_cuda (pr1, pr2): OK. All 45 outputs bit-identical across the 13 settings (EXPERT_MAJOR 0/1, EM_TB 2/3/4/6, VERIFY, MM_WARP 0/1/VERIFY); peak +468 MiB.
Two-GPU tests (pr2 only):
- test_dsv4_peer_init_cuda: 'probe, init rollback and retry: OK (2 devices)'; peaks +264 / +263 MiB.
- test_dsv4_peer_moe_cuda: OK. In both compute and inplace modes all 64 on-peer masks were bit-identical to all-on-primary; the soft fault was recomputed bit-identically and the lost fault was refused cleanly, each latched once; peaks +266 / +261 MiB.
make dsv4-cuda-test (CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NVCCFLAGS='-O3 -std=c++17 -ftz=false -arch=sm_86 -Xcompiler=-Wall,-Wextra -DCOLI_DSV4_NO_TC', CUDA_VISIBLE_DEVICES=0,1): rc=0 at all three tips, 0 compiler warnings.
- pr0: dense OK, attention OK (sparse SKIP), MoE SKIP (generic build).
- pr1: bank_bytes OK, dense OK, attention OK (SKIP), MoE SKIP, generic_bitwise OK. Peaks +468 MiB on GPU0, +0 on GPU1.
- pr2: peer_init OK, peer_moe OK, then the same five as pr1. Peaks +468 / +263 MiB.
Not run:
- The 3.2 GiB bank tests: the DeepGEMM half of test_dsv4_moe_batch_cuda skips on generic builds anyway.
- DeepGEMM builds, which need sm_120.
- The model itself.
Incident: test_dsv4_decode_cuda.c at pr0 went to +3558 MiB on GPU0 before it was killed. It is not part of dsv4-cuda-test and I did not run it again (see issues).

## Trial merge with current upstream dev

I ran git fetch upstream first. upstream/dev is now 11dbb2c61e7b, 31 commits past 7b55a9b3. Of the stack's files it touches only two:
- c/deepseek_v4.c: the Vulkan tier's warm-start line now goes through vkt_warm_report.
- c/Makefile: qwen36 and Vulkan chain targets, and a TEST_EXCLUDE entry.

git merge-tree --write-tree upstream/dev <tip> (git 2.43):
- pr0/v4: clean, rc=0, tree df5b8923, c/deepseek_v4.c auto-merged.
- pr1/v4: clean, rc=0, tree 5cf18c81, c/Makefile and c/deepseek_v4.c auto-merged.
- pr2/v4: clean, rc=0, tree 2afd250c, c/Makefile and c/deepseek_v4.c auto-merged.
No file conflicts at any tip.

I also built each merged tree in a private worktree:
- CPU engine build: rc=0, 0 warnings.
- All 33 units with GPU-tier -fsyntax-only, the GPU unit also with -DCOLI_VULKAN: 0 warnings.
- dsv4 parity test and the v4 source tests: pass.

Not run on the merged trees: make check and the CUDA build.
