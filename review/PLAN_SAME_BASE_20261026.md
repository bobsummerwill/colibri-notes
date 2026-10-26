# Same-base comparison of #1766, PR #2 and a combined "C+" (plan, 2026-10-26)

Status: **planned and waiting for Bob's go** (written 10:00). **10:10, Bob: C+ (and the Phase 0 ablation that
informs it) comes after the B/M/P run.** **10:15, Bob: GO. Don't gate Stage 2 on Stage 1 (do whatever is faster in
aggregate); shadowfax may go down (daytime is fine); draft PRs inside the fork are fine.** C+ = M merged into P (base B shared,
so only the PR #2 vs #1766 conflicts are left). See "Revised timeline" at the end. Nothing here is upstream. Trees stay local or on the fork.
Results go to `manifests/*-2026102[67]-samebase/` and a handoff.

## Why (what final23/24 taught us)
Data: [manifests/shadowfax-20261026/final23](../manifests/shadowfax-20261026/final23), [final24](../manifests/shadowfax-20261026/final24),
[computer4 #1766](../manifests/computer4-20261026-m1766/SUMMARY.md). Summary in [HANDOFF_20261026.md](../HANDOFF_20261026.md).

- End to end on 2x RTX 3090, REAP, #1766 on two GPUs (4.60 tok/s) beat PR #2 (4.34) by 6%. **That isn't a like-for-like
  result.** The two sit on different dev bases (88f25a60 vs 77530b49 + #2401). The bases differ by 5% in REAP decode
  and a lot in full-model prefill: TTFT 138 / 200 s on the old base vs 86 / 122 s on the new.
- **Where #1766's gain comes from.** Its docs (docs/deepseek-v4.md on the branch) and our runs show two separate parts:
  1. **Single-GPU kernel work, on by default:**
     - native E4M3/E2M1 conversion (it replaces a search over 255 codes for every activation);
     - indexer query weights uploaded once (`DSV4_CUDA_RESIDENT_INDEXER`);
     - the CUDA compressor and indexer prepare, WO decode, and the CUDA head and head batch;
     - device routing (`DSV4_CUDA_RESIDENT_ROUTE`), only when the full resident table exists.

     This alone gives **1.24x on one GPU** (REAP) and 1.16x on the full model, against its own base.
  2. **Layer placement (`DSV4_CUDA_DEVICES`).** Contiguous layer ranges, each with its own dense mirrors, KV and expert
     cache. Its own docs say: "This increases usable cache capacity, not parallel compute within a layer … layers
     still execute sequentially … there is no six-way TP, EP, or overlapped pipeline." Measured, a second GPU adds
     **1.29x** (REAP) and **1.04x** (full model) over #1766 on one GPU.
- **PR #2's second GPU adds more:** **1.43x** (REAP) and **1.15x** (full model) over its base. Its peer GPU holds expert
  mirrors and **computes them within the same layer** (peer compute over NVLink, reduce on the primary). So the second
  card works at the same time as the first, instead of taking turns.
- #1766's full residency (`DSV4_CUDA_RESIDENT_EXPERTS=auto`) **never engaged on our machines.** It fell back to the
  cache every time: 3090 free 16.7 / 20.3 GiB, 5070 free 2.6 GiB, all short of every expert. On a box where the model
  fits (the author's 6x RTX 5090: 6.16 vs 2.15 tok/s) it is a different mode, and we can't measure it here.
- #1766's full-model "bank loss" at the default 2800 MiB reserve is its base's bug, which #2401 fixes. With 4700 MiB on
  both sides, #1766 is 1.16x (one GPU) and 1.21x (two) against its base, with reference texts.

## Merge feasibility (counter, trial merges 2026-10-26 09:45, aborted)
- #1766 (ca55947c) today: 17 commits, 3 of them merges, 28 files, +2,855 / -144. It's much smaller than the "+10k" of
  September. Its base 88f25a60 is 1,047 commits behind dev 77530b49.
- **#1766 into dev 77530b49:** 13 conflict hunks, about 200 lines:
  - `deepseek_v4.c`: 8 hunks, 176 lines
  - `backend_cuda_dsv4.cu`: 2 hunks, 13 lines
  - `Makefile`, `deepseek_v4_internal.h`, `docs/deepseek-v4.md`: 1 hunk each
- **#1766 into PR #2 (5b3f4601):** 29 hunks, about 730 lines:
  - `deepseek_v4.c`: 18 hunks, 535 lines
  - `backend_cuda_dsv4.cu`: 5 hunks, 144 lines. Mostly both PRs adding multi-device code at the same spot (PR #2's peer
    probe and access vs #1766's `dsv4_cuda_init(dev, n)` over up to 16 devices).
  - the internal header (2 hunks), docs (3), Makefile (1).
  - The real integration risk is the two models of multi-device state in the CUDA backend.

## Trees (all on dd03a9de = dev 77530b49 + #2401)
#2401 goes into every tree. Otherwise the full model loses its bank and the comparison breaks. #2401 is in the
maintainer's next batch anyway.

| | Tree | How |
|---|---|---|
| **B** | dd03a9de | as is |
| **M** | #1766 merged onto B | resolve the 13 hunks; keep #1766's behaviour; drop nothing |
| **P** | PR #2 5b3f4601 | as is (already on B) |
| **C+** | "Phase 1" below | PR #2 plus #1766, resolved so that `DSV4_CUDA_DEVICES` (more than one device) and `DSV4_CUDA_PEER_DEVICE` refuse each other at start-up, since both claim GPU 1 |

## C+: better than both (design)
1. **Keep all of #1766's single-GPU kernels** (part 1 above). They are orthogonal to PR #2 and help every
   configuration. They could also be their own small upstream PR, the shape the maintainer asked for on #1766.
2. **For two or more GPUs, spread experts across cards and compute them in parallel** (PR #2's peer compute) rather
   than splitting layers. Both give the same aggregate cache, but only this one makes the cards work at once.
3. **Full residency spread by expert:** when every routed expert fits in the combined VRAM, each card holds a fixed share
   of every layer's experts. Every layer's MoE then runs on all cards at once, with #1766's device routing. This is the
   EP that #1766 says it lacks.
4. **Generalise PR #2's single peer to N GPUs** (#1766 handles up to 16; PR #2 handles 2).
5. Later, optional: use #1766's layer placement only for dense tensors and KV, to free primary VRAM for long contexts,
   with experts still spread by expert.

| Phase | Scope | Effort | Measurable here? |
|---|---|---|---|
| 0 | Ablate #1766's kernel features one at a time (`DSV4_CUDA_RESIDENT_INDEXER=0`, `..._ROUTE=0`, and the others its docs name), to learn which pay | ~1 h GPU | yes (3090 x1, REAP) |
| **1** | Items 1 and 2: #1766's kernels plus PR #2's 2-GPU parallel tier (= C+) | 4-6 h beyond the plain merge | yes: 3090 x1 / x2, 5070 |
| 2 | Item 3: expert-spread residency and device routing | 1-2 days | code path yes (2 GPUs); full residency no (neither model fits in 48 GB) |
| 3 | Item 4: N-GPU peers | 2-3 days | no (we have 2 GPUs) |

First check in Phase 1: confirm that PR #2's peer really runs **at the same time** as the primary (an nsys timeline of
one decode layer; there are earlier traces in `~/colibri-bench/nsys-*.nsys-rep` on shadowfax). The design depends on it.

## Test matrix
The trees share one base, so **every arm must give byte-identical answers**. Any difference is a bug and gets
investigated before any speed is believed. Every CUDA arm: `COLI_CUDA_ATTN_BATCH=1 COLI_CUDA_MOE_BATCH=1`,
`--ram 40 --ctx 8192`, a fresh server per sample, interleaved.

**CUDA speed**
- **2x RTX 3090 (shadowfax, sm_86, CUDA 12.0, NO_TC):**
  - 1 GPU: B, M, P (`V4_GPU_RESIDENT_FIRST=1`), C+ (VRAM-first on)
  - 2 GPUs: M (`DSV4_CUDA_DEVICES=0,1`), P (peer GPU 1), C+ (#1766 kernels plus peer GPU 1)
  - each on REAP-150B (6 prompts x 300 tokens) and full V4 Flash (1,992- and 3,005-token prompts), 2 samples each,
    so 14 + 14 samples
- **RTX 5070 12 GB (computer4, sm_120, CUDA 12.8, generic build):** B, M, P, C+, full model, 6 x 300 tokens, 2 samples.
  First a pilot of M at defaults. If it floods handled OOMs (as #1766 did on its old base), every arm gets
  `DSV4_CUDA_VRAM_RESERVE_MB=2000`.

**Phase 0 ablation (3090, 1 GPU, REAP, 1 sample each):** M with each kernel feature turned off in turn.

**CPU**
- Correctness on counter: CPU-only builds of all four trees, `make check`, the tiny V4 oracle (21 PASS on dev), and
  counter's per-commit checks (build, GPU tier, Vulkan, `-Wall -Wextra`) for M and C+.
- Speed on computer4, after its 5070 runs: CPU-only builds, full model, 2 prompts x 128 tokens, 2 samples each for B, M,
  P and C+. The expected result is "no change". It shows that neither PR slows CPU-only decode, since both add code on
  the shared decode path.

**Vulkan.** `deepseek_v4.c` has 43 `#ifdef COLI_VULKAN` blocks, and both PRs make large changes to the file. So: pure
Vulkan on one 3090, REAP, 1 sample each of B, M, P and C+, checking that texts match and speed is unchanged.

**Metal.** DeepSeek V4 has no Metal backend: `deepseek_v4.c` and `Makefile.deepseek-v4` never mention it, and neither
PR touches `backend_metal.*`. The only risk is the macOS build, since #1766 adds a test binary that `make check` builds
everywhere. That is covered by upstream's CI workflow, which has "macOS (colibri + V4 platform gate)", "All engines
(macos)", Windows, ARM and Linux CPU jobs. #2405 passed all 116. **Run it on the fork:** draft PRs inside
bobsummerwill/colibri (fork branch into the fork's `dev`) for M and C+. Actions are enabled on the fork, and
`ci.yml`/`check.yml` trigger on pull_request to dev. This is fork-only, nothing upstream.

**Unit tests on GPU:** `dsv4-cuda-test` (all trees), #1766's `test_dsv4_multigpu_cuda 0,1` (M, C+), PR #2's peer
init/MoE tests (P, C+), and #1766's new CUDA tests (compressor, head, indexer, quant, route, WO).

## Timeline
GPU sample times are from final23/24 and c4m22.

| Step | Time | Note |
|---|---|---|
| Notes (this file) | 0.25 h | |
| M: merge #1766 onto B, resolve, build | 1-1.5 h | |
| C+ (Phase 1): merge into PR #2, resolve 29 hunks, mutual exclusion, concurrency check | 4-6 h | the risky part; after M |
| counter checks for M and C+ (make check ~50 min each, per-commit, CPU builds, oracle) | ~1.5 h | overlaps with the merges |
| Fork CI for M and C+ | 1-2 h | GitHub runners, background |
| Harness and pilots (shadowfax final25, computer4 c4m25) | 0.75 h | |
| **shadowfax window** (llama down): REAP 14 samples (~2.1 h), full 14 (~1.2 h), Vulkan 4 (~1 h), ablation (~1 h), CUDA tests (~0.3 h) | **~5.5-6 h** | |
| **computer4:** 5070 8 samples (~2.7 h), then CPU-only 8 (~2 h) | ~5 h | in parallel with shadowfax |
| Write-up, redrafted #1766 comment and #2405 text (shown to Bob, not posted) | 0.75 h | |

**Critical path: about 8.5 h of preparation, then 6 h on shadowfax, then 0.75 h of write-up, so about 15 h in total.**
That's the original 8-9 h plus about 6 h for C+ (Phase 1) and the ablation. Starting 10:15, the preparation ends around
19:00, shadowfax runs about 19:00-01:00, computer4 about 19:00-00:00, and results are written up by about 02:00. If
C+ fights back, B/M/P run on schedule and C+ follows as a separate window. Phases 2-3 come afterwards, as a design
note only.

## Decisions for Bob
- Go/no-go on this plan.
- shadowfax's llama down for about 6 h (overnight fits). hermes stays stopped until then.
- Whether the fork-internal draft PRs for the CI runs are OK. They're visible on the public fork, though not upstream.

## Revised timeline (10:10: C+ after the other steps)
**Stage 1: B, M, P only**
| Step | Time | Clock (from 10:15) |
|---|---|---|
| M: merge #1766 onto B, resolve, build | 1-1.5 h | to ~11:45 |
| counter checks for M (make check, per-commit, CPU build, tiny oracle); fork CI for M in background | ~1.25 h | to ~13:00 |
| Harness and pilots (shadowfax final25, computer4 c4m25) | 0.75 h | to ~13:45 |
| shadowfax: REAP 10 samples (~1.6 h), full 10 (~0.9 h), Vulkan 3 (~0.75 h), CUDA tests (~0.3 h) | ~3.5 h llama down | ~14:00-17:30 |
| computer4: 5070 pilot and 6 samples (~2.1 h), then CPU-only 6 (~1.5 h) | ~3.6 h | ~14:00-17:45 |
| Write-up | 0.5 h | **~18:15** |

If shadowfax must wait for the evening, its window moves (e.g. 19:00-22:30) and the write-up follows about 23:00.

**Stage 2: C+.** Its engineering starts once Stage 1 is running on the GPUs, without delaying Stage 1.
| Step | Time | Clock |
|---|---|---|
| C+ merge, 29 hunks, mutual exclusion, peer-concurrency check (nsys) | 4-6 h | ~14:00-19:30 |
| counter checks and fork CI for C+ | ~1.5 h | to ~21:00 |
| shadowfax: C+ 1 GPU and 2 GPUs on REAP and full (8 samples), Phase 0 ablation of #1766 (~1 h), Vulkan C+ (1), anchor samples of B and P in the same session (~0.75 h) | ~3.5 h llama down | ~21:00-00:30 |
| computer4: C+ on the 5070 (2) and CPU-only (2), plus a B anchor | ~1.5 h | ~21:00-22:30 |
| Write-up, redrafted #1766 / #2405 text (shown, not posted) | 0.75 h | **~01:15** |

## Progress log (2026-10-26)
- **10:15 Stage 1 launched**, far earlier than the plan's 14:00:
  - shadowfax final25 (B/M/P; REAP and full, 1 and 2 GPUs; Vulkan; CUDA tests; ~3.4 h, llama down)
  - computer4 c4m25 (pilot, 5070 B/M/P, CPU-only B/M/P; ~3-4 h)
  - Harness: [manifests/shadowfax-20261026-samebase](../manifests/shadowfax-20261026-samebase), [manifests/computer4-20261026-samebase](../manifests/computer4-20261026-samebase).
- **M** = fork `samebase/m1766`. Engine 63df714d (merge), then tests-only d4d8ab3f and the `dsv4.def` fix 819cd03d.
  - 13 hunks resolved. Main point: #2401's fuller guard moves into #1766's per-device cache fields.
  - Builds: CUDA sm_86 (shadowfax), sm_120 and CPU-only (computer4), no nvcc or gcc warnings.
  - Every deepseek_v4.c unit compiles clean in 4 configs (plain, GPU tier, +Vulkan, +prefetch).
  - Tiny V4 oracle: 21 PASS on B, P and M.
  - Python v4/dsv4 tests pass after the test follow-ups.
- **C+** = fork `samebase/cplus`. Engine d0ccbfe2 (M merged into PR #2: 24 hunks), then `dsv4.def` d5e6e2c1.
  - The engine open is rewritten: #1766's device list first, PR #2's peer settling for one device only, and the peer
    pool refused (with a warning) when there is a device list.
  - Attach, peek and the VRAM-held scan use the layer's own cache, then the peer.
  - Every unit compiles clean in 4 configs. Python tests pass, with PR #2's peer-init source test re-pinned to the
    three-way init.
  - nvcc: Bob installed Arch's `cuda` 13.4.3 on counter (~11:00).
    - M and C+ build the full CUDA engine for sm_86 (NO_TC, as on the 3090s).
    - Their backend also builds for sm_120 with the tensor-core path.
    - 0 warnings, 0 errors ([manifests/counter-20261026-samebase](../manifests/counter-20261026-samebase), `cb-*.log`).
- **Fork CI:** draft PRs bobsummerwill/colibri#1 (M) and #2 (C+); the fork's `dev` fast-forwarded to 77530b49.
  - **Slip:** PR #1's first body used `owner/repo#N` links, which put "mentioned this" backlinks on upstream #1766 and
    #2401. Edited within a minute, but the timeline entries stay (GitHub can't delete a PR). Memory rule added.
- **Found in #1766 itself:** `dsv4_cuda_wo_decode` is declared in `backend_cuda_dsv4.h` and resolved by the loader, but
  never listed in `dsv4.def`. An MSVC DLL would not export it, so Windows would silently run without WO decode. dev's
  `test_backend_loader_dsv4_parity.py` catches it once #1766 sits on current dev. Worth a line in the #1766 comment.
- **#1766's placement test after the merge** needs stubs for backend entries dev added later, and native-FP4 views
  (dev's mirror cache attaches FP4 only). It's a test-side drift, not an engine problem.
- **Upstream:** JustVugg opened batch PR #2412 with #2401-#2404 (108 checks passed, 8 pending at 10:40).
- **Ablation switches** (#1766, all default on): DSV4_CUDA_COMPRESSOR, _HEAD, _HEAD_BATCH, _INDEXER_PREP,
  _INDEXER_HEADS, _RESIDENT_INDEXER, _WO_DECODE, _RESIDENT_EXPERTS, _RESIDENT_ROUTE (full table only). Native E4M3
  conversion has no switch: its share is whatever "M, all off" keeps over B.
