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

## Restart on the new dev (2026-10-26 ~10:50)
- **10:41Z+: upstream merged batch #2412 (#2401-#2404 among 44 commits). dev = c1c06ac6.** Bob: "rebase everything
  and start everything again".
- Stage 1 stopped at ~10:45: final25 after B1 s1, M1 s1 and P1 (P1's client cut short); c4m25 during its pilot.
  llama restored and healthy. Partial logs archived under `manifests/*-samebase/stopped-*`.
  - One early sample, same base, one GPU, REAP: P1/M1 decode ≈ 0.84, so #1766's single-GPU kernels lead PR #2's
    one-GPU VRAM-first path. This is the expected direction. Not used, only noted.
- New trees on the fork:
  - **B = c1c06ac6** (dev)
  - **P = `samebase/pr2` 11c40038:** PR #2 rebased; range-diff shows all 11 patches identical; the #2401 commit is in dev
  - **M = `samebase/m1766` ecc92586:** dev merged into the previous M, keeping its resolutions
  - **C+ = `samebase/cplus` b659ac8d:** the same for C+
  - Cross-check: C+ minus P has the same file set as M minus dev, except the test each side adapted.
- counter: all four build the CUDA engine (sm_86, NO_TC) and the sm_120 backend with nvcc 13.4, 0 diagnostics. M and C+:
  every unit is clean in 4 configs; Python v4/dsv4/loader tests OK. Tiny oracle and make check on all four run in the
  background (`cb2/status`).
- Upstream draft #2405 still points at the pre-rebase branch (`docs5/pr2-two-gpu-decode` 5b3f4601). The rebased PR #2 is
  `samebase/pr2`. Updating #2405 is Bob's call.
- Next: the helper rebuilds f27-* (shadowfax) and s27* (computer4). shadowfax runs final27 (B/M/P) then final28 (C+,
  same-session anchors, ablation) as one chain. computer4 runs c4m27 (pilot, B/M/P/C on the 5070, then CPU-only).
- **~11:05 #2405 updated (Bob's OK):**
  - `docs5/pr2-two-gpu-decode` force-pushed with lease, 5b3f4601 → 11c40038 (= `samebase/pr2`). Upstream shows
    11 commits, mergeable, still a draft.
  - Banner rewritten: #2401 is merged, and the diff is the PR's own 11 commits. Body: [PR2_BODY_20261026.md](PR2_BODY_20261026.md).
  - Split branches rebased the same way, every patch unchanged: pr2-1 6a191e90, pr2-2 c45b6962, pr2-3 695a7be4,
    pr2-4 1c881700. pr2-4's tree equals P.

## PR #2's four splits in the suite (Bob, ~11:10)
- Splits on dev c1c06ac6:
  - S1 `split/pr2-1-guard` 6a191e90: the per-cache guard
  - S2 `split/pr2-2-settings` c45b6962: PIN_CAP_GB, stats log
  - S3 `split/pr2-3-vram-first` 695a7be4: VRAM-first
  - S4 = P (same tree)
- **shadowfax final29**, after final28, built between runs: REAP, one GPU.
  - Arms: S1 x2, S2 x2, S3 x2 (VRAM-first on), anchors B1 and P1 x1.
  - One text group: everything must equal B.
  - About 1.4 h more, so shadowfax ends ~19:40.
- **computer4 c4m28**, after c4m27: S1, S2, S3 and B anchor x1 on the 5070. About 1.3 h.
- **Expectations:** S1 and S2 are neutral against dev (that's what makes them easy to merge); S3 is VRAM-first's one-GPU
  gain on its own (P1 measures it with split 4's inactive code present).
- **How the splits fit with #1766:**
  - S1 is the shared foundation: #1766's per-device caches need its per-cache guard state, which is what the M merge did.
  - S2 is orthogonal.
  - S3 combines cleanly: C+ looks VRAM-first up in the layer's own cache, and C+ on one GPU (C1) is effectively #1766 + S3.
  - S4 is the overlap: two multi-GPU setups in the backend, both claiming GPU 1. C+ is the worked combination.
  - Natural landing order: S1, then #1766, then S2/S3, then S4 (if the numbers back it).

## Results so far (2026-10-26 14:05)
All trees on dev c1c06ac6. Every sample so far passes its gates: answers byte-identical within each text group, 0
handled out-of-memory lines, no OOM kills.

### shadowfax final27 (B / M / P), done 13:33, GATES: PASS
Logs: [manifests/shadowfax-20261026-samebase5/final27](../manifests/shadowfax-20261026-samebase5/final27) (run.log,
summary.txt, bench.tgz sha256).

REAP-150B, 6 prompts x 300 tokens, 2 samples per arm (decode = median of the per-sample mean; samples agree within 0.2%):

| Arm | Decode tok/s | vs B1 | Median TTFT |
|---|---|---|---|
| B1 dev, 1 GPU | 3.010 | 1.00x | 3.63 s |
| P1 PR #2 (VRAM-first), 1 GPU | 3.098 | 1.03x | 3.56 s |
| M1 #1766, 1 GPU | 3.784 | 1.26x | 3.47 s |
| P2 PR #2 (peer), 2 GPUs | 4.332 | 1.44x | 3.90 s |
| M2 #1766 (`DSV4_CUDA_DEVICES=0,1`), 2 GPUs | 4.818 | 1.60x | 3.50 s |

- What the second GPU adds: PR #2 1.40x (P2/P1), #1766 1.27x (M2/M1).
- Head to head: P1/M1 0.82x, P2/M2 0.90x.

Full V4 Flash, prompts of 1,992 and 3,005 tokens, default reserve (dev's bank-aware reserve keeps the bank: 0 OOM lines),
2 samples per arm:

| Arm | Decode, prompt 1 / 2 | Mean vs B1F | TTFT, prompt 1 / 2 |
|---|---|---|---|
| B1F dev, 1 GPU | 1.91 / 1.81 | 1.00x | 86 / 124 s |
| P1F PR #2, 1 GPU | 1.93 / 1.86 | 1.02x | 86 / 124 s |
| M1F #1766, 1 GPU | 2.24 / 2.11 | 1.17x | **98 / 141 s (1.14x slower)** |
| P2F PR #2, 2 GPUs | 1.96 / 2.30 | 1.15x | 86 / 124 s |
| M2F #1766, 2 GPUs | 2.29 / 2.28 | 1.23x | 98 / 140 s (1.13x slower) |

- **#1766 is slower to first token than dev on the full model, by 13-14%.** On the old base (88f25a60) it was faster
  (116 vs 137 s). Current dev's prefill dropped from 137 to 86 s and #1766 doesn't keep that gain. To look into
  (likely one of its prefill-side paths: compressor, indexer prepare, or the head batch; the ablation in final28 runs
  on REAP decode only, so it may not show this).
- **PR #2's peer helps only the second prompt on the full model:** 1.01x on prompt 1, 1.24x on prompt 2. To look into.
- Full model, head to head: P1F/M1F 0.87x, P2F/M2F 0.93x on decode; PR #2 reaches the first token 12-13% sooner.
- Vulkan only (REAP, one GPU, 1 sample each): B 4.25, M 4.40, P 4.38 tok/s. Neither PR touches Vulkan; the spread is
  noise-sized and texts are identical across the three.
- CUDA self-tests (dsv4-cuda-test, cuda-test) pass on B, P and M.

### shadowfax final28 (C+), started 13:37, running
Partial log: [manifests/shadowfax-20261026-samebase5/final28-partial](../manifests/shadowfax-20261026-samebase5/final28-partial).
First samples, REAP, one each, same client as final27:

| Arm | Decode tok/s | vs B1 (final27) | vs #1766, same GPU count |
|---|---|---|---|
| C1 C+, 1 GPU (VRAM-first on) | 3.911 | 1.30x | 1.03x |
| **C2 C+, 2 GPUs (peer)** | **6.054** | **2.01x** | **1.26x** |

- The two halves more than stack: C+'s second GPU adds 1.55x (C2/C1), against 1.40x for PR #2 and 1.27x for #1766.
  The plan's guess was ~5.3 tok/s.
- C2's answers equal M2's for all 6 prompts. Peer log: `peer-mirrors device=1 reserve=1024MB mode=compute`;
  residency `cache-fallback` (free 16764 MiB, reserve 2800 MiB), as for M.
- Median TTFT: C1 3.41 s, C2 3.79 s (the peer costs ~0.4 s to first token, as with P2).
- Still to come in final28: same-session anchors B1/P2/M2/M1, C2 and C1 second samples, the #1766 ablation, full model
  C1F/C2F/B1F/P2F, Vulkan C+, CUDA tests. ETA ~17:45, then final29 (splits) to ~19:30.

### computer4 c4m27 (RTX 5070, full model, default settings), running
Partial log: [manifests/computer4-20261026-samebase5](../manifests/computer4-20261026-samebase5).
Pilot: M at defaults logged 0 out-of-memory lines, so no arm sets a reserve.

| Arm | Decode tok/s, s1 / s2 | vs B |
|---|---|---|
| B dev | 1.54 / (s2 running) | 1.00x |
| P PR #2 | 1.54 / 1.56 | 1.00-1.02x |
| M #1766 | 1.81 / 1.86 | 1.18x |
| C C+ | 1.82 / 1.81 | 1.18x |

- All answers byte-identical, 0 out-of-memory lines. C+ equals #1766 on one card, as it should (no peer).
- CPU-only arms started 13:42; c4m28 (splits) follows. ETA 16:00-17:30.

### CI
Fork draft PRs 1 (M) and 2 (C+): 116 jobs queued since ~10:47 PDT, nothing in progress; upstream #2405: 2 pass, 114
queued. Standard hosted runners; githubstatus shows Actions operational. Re-trigger if still stuck when final28 ends.

### Where this points (not yet a decision)
- As separate PRs, #1766 beats PR #2 everywhere except full-model TTFT.
- PR #2's peer tier is complementary: on top of #1766 (C+) it gives the best result measured, 2.01x dev on two GPUs.
- So the pitch is not "PR #2 instead of #1766" but "#1766, then PR #2's peer tier on top": S1, #1766, S2/S3, S4/C+.
- Waiting for: C2 second sample and same-session anchors, C+ on the full model, the splits. #2405 stays draft;
  nothing is posted upstream without Bob's OK.

## Investigations added (Bob, ~14:20): the two open findings
Both run after what is already queued, so nothing in flight changes: shadowfax final30 after final29, computer4 c4m29
after c4m28. Each arm is one cold sample with the usual gates (text identity, 0 handled OOM).

### F1: #1766 is 13-14% slower to first token than dev on the full model
What we know:
- Full model, long prompts (1,992 / 3,005 tokens): B1F 86 / 124 s, M1F 98 / 141 s, M2F 98 / 140 s. P matches B.
- REAP, short prompts: M1 is *faster* to first token than B1 (3.47 vs 3.63 s). So it shows on long prefill only.
- On the old base (88f25a60) #1766 was faster than its dev (116 vs 137 s). dev's prefill got much faster since
  (137 → 86 s: the batched GPU prefill / bank work, e.g. f672fd73, 911b86db, 838416e6, 975f364f), and M doesn't keep
  all of that gain. 838416e6's kernel is still in M's backend (checked).
- The serve logs show the same MoE bank refills in B1F and M1F, so the extra time is outside the batched MoE.
- Code read (counter) so far: in M, #1766 changes the batched attention's setup (compressor/indexer mirrors uploaded
  in its packed decode form, `attn.indexer.weights_proj` for indexer prepare, mHC mirrors only with ATTN_BATCH) and adds
  a per-token GPU compressor projection to `coli_v4_compressor_step`. The batched projection still accepts the packed
  tensors (fmt 16 kept), so no smoking gun yet.

Tests:
- **final30 part A, full model, `pr1:both`, one GPU** (~4.5 min each): anchors B1F and M1F, then M with one #1766 switch
  off each: COMPRESSOR, INDEXER_PREP, INDEXER_HEADS, RESIDENT_INDEXER, HEAD, HEAD_BATCH, WO_DECODE, and all off (MX).
  An arm whose TTFT returns to ~86 / 124 s names the path. (final28's ablation is REAP decode only, so it can't show it.)
- **final30 part A', profiled pair**: B1F and M1F with `DSV4_ATTN_PROF=1 V4_MOE_BATCH_PROF=1` (both exist on dev), to
  get the per-phase prefill split. Timing from these two isn't used in the tables.
- C1F/C2F in final28 already show whether C+ inherits it (expected yes: C+ has #1766's paths).
- **computer4 c4m29**: B, M and M all-off on the 5070 with `pr1:both`, to see whether it reproduces on sm_120 with tensor
  cores.
- Then bisect in code on counter if the ablation points at one switch; if no switch restores it, it is in the merged
  non-switchable code (native E4M3 conversion, placement), and the next step is a prefill timeline (nsys) of B1F vs M1F.

### F2: PR #2's second GPU helps only the second full-model prompt
What we know:
- P2F vs P1F decode: prompt 1 1.01x, prompt 2 1.24x. On REAP (300 tokens per prompt) the peer helps every prompt.
- **The full-model client decodes very few tokens**: the model stops after 32 tokens (prompt 1) and 13 (prompt 2)
  (`v4_serve_turn ... completion=32/13`). So "decode tok/s" on the full model is measured over a few seconds, right
  after a cold prefill.
- P2F's serve stats: RAM expert-cache hit rate 4.1% on prompt 1, 12.1% on prompt 2; prompt 1's decode waits on SSD
  expert reads. The peer only speeds compute, so it can't help while decode is disk-bound.
  M2F/M1F shows the same pattern, smaller (1.03x / 1.08x).
- Working hypothesis: a warm-up effect plus a too-short measurement, not a peer defect.

Tests:
- **final30 part B, full model, a new client `fdec`**: three of bench_decode's open-ended prompts, 300 tokens each, so
  decode is measured over minutes after the cache warms. Arms B1F, P1F, P2F, M1F, M2F, C1F, C2F x1 (~10 min each).
  This also gives the full-model decode comparison on a proper length, which the write-up needs anyway.
- **final30 part B', order swap**: P1F and P2F with `pr1` in reverse order (prompt 2 first). If the peer then helps
  whichever prompt comes second, it is warm-up; if it still helps only the 3,005-token prompt, it is prompt-specific.
- No peer hit counter exists in PR #2's stats (only `vram_first_hits_total`). If B and B' don't settle it, add a
  peer-call counter to `V4_SERVE_STATS_LOG` on the fork (a PR #2 follow-up, measured, not pushed upstream without Bob).

### Timeline with the investigations
| Machine | Run | Clock |
|---|---|---|
| shadowfax | final28 (C+, anchors, ablation, full, Vulkan, tests) | to ~17:45 |
| shadowfax | final29 (splits) | to ~19:30 |
| shadowfax | **final30 (F1 A/A', F2 B/B'), ~2.5 h** | **to ~22:15** |
| computer4 | c4m27 CPU arms, c4m28 splits | to ~17:30 |
| computer4 | **c4m29 (F1 on the 5070), ~40 min** | **to ~18:15** |
| counter | code bisect / nsys plan as the ablation points | evening |
| — | write-up with both findings explained | **~22:45** |

**GPU power (14:40):** every shadowfax sample in final27-final30 runs with both 3090s at a 300 W limit (default 420 W).
Full-model prefill peaks at 267-286 W and decode at 210-277 W (5 s nvidia-smi samples), so prefill sits just under the cap.
Bob will change the power profile only after final30 is done; later runs are not comparable with these without anchors.

## Results update (2026-10-26 17:30)
### shadowfax final28 (C+), done 16:13, GATES: PASS
Logs: [manifests/shadowfax-20261026-samebase5/final28](../manifests/shadowfax-20261026-samebase5/final28).
Same-session anchors match final27 within 0.3% (B1 3.009, M1 3.790, M2 4.828, P2 4.336).

REAP, 2 samples for C+:

| Arm | Decode tok/s | vs B1 | vs #1766, same GPUs | vs PR #2, same GPUs |
|---|---|---|---|---|
| C1 C+, 1 GPU | 3.911 / 3.920 | 1.30x | 1.03x | 1.26x |
| C2 C+, 2 GPUs | 6.054 / 6.077 | **2.01x** | **1.26x** | **1.40x** |

Full model, 2 samples for C+ (decode prompt 1 / 2; TTFT prompt 1 / 2):

| Arm | Decode | Mean vs B1F | TTFT |
|---|---|---|---|
| B1F (anchor) | 1.91 / 1.81 | 1.00x | 86 / 124 s |
| P2F (anchor) | 1.95 / 2.31 | 1.14x | 86 / 124 s |
| C1F | 2.22-2.27 / 2.14-2.15 | 1.18x | 98-99 / 141 s |
| C2F | 2.31 / 2.76-2.78 | 1.37x | 98 / 141 s |

- C+ inherits finding F1 (13-14% slower first token on the full model, like M).
- On the full model C2F beats every other arm on decode (2.54 mean vs M2F 2.29, P2F 2.13).

#1766 switch ablation (M, one GPU, REAP decode, 1 sample each; M1 = 3.790):

| Off | Decode | vs M1 |
|---|---|---|
| all (MX) | 3.339 | 0.88x (still 1.11x over B1: native E4M3 and the non-switchable parts) |
| HEAD | 3.591 | 0.95x |
| RESIDENT_INDEXER | 3.600 | 0.95x |
| INDEXER_HEADS | 3.707 | 0.98x |
| INDEXER_PREP | 3.763 | 0.99x |
| WO_DECODE | 3.783 | 1.00x |
| COMPRESSOR | 3.820 | 1.01x |

- So #1766's 1.26x on one GPU: ~1.11x from code with no switch (native E4M3 etc.), ~5% each from the resident head and
  the resident indexer, ~2% indexer heads; compressor and WO decode add nothing measurable to decode here.
- Vulkan C+ (VKC): 4.41 tok/s, same as B/M/P under Vulkan.

**CUDA tests on C+:**
- `cuda-test` OK; #1766's `test_dsv4_multigpu_cuda 0,1` OK.
- **`dsv4-cuda-test`: PR #2's `test_dsv4_peer_init_cuda` had 4 failures. This is a real C+ merge bug, fixed.**
  - #1766's up-front device validation in `dsv4_cuda_init` returned on an invalid ordinal with the CUDA error still
    pending, and without init_fail's "device N did not initialise" line.
  - A retry with one device then succeeded, but the next checked call (`decode_state_set`) failed on the stale error.
  - #1766 alone has the same stale-error return (a note for the #1766 comment).
  - Fixed on fork `samebase/cplus` dec72809. nvcc sm_86 compiles clean, and the Python source and loader tests pass.
    **The GPU test is still to run** (after final30, or computer4 now that it's idle).
- **#1766's own CUDA unit tests (compressor, compressor_state, head, wo_decode, indexer_prepare) can't run on a 2-GPU
  box.**
  - They hard-code `int devices[]={0,5}`, so `assert(dsv4_cuda_init(devices,2))` aborts.
  - `test_dsv4_indexer_prepare_cuda` also doesn't build with shadowfax's g++: `ggml_quants.h:134` has designated
    initializers in C++.
  - These are #1766 test portability notes, not C+ defects. To run them here: patch to `{0,1}` locally after final30.

### computer4: c4m27, c4m28, c4m29 all done, GATES: PASS
Logs: [manifests/computer4-20261026-samebase5](../manifests/computer4-20261026-samebase5).
- **5070, full model, 6 prompts x 300 tokens:**
  - B 1.54, P 1.54-1.56, M 1.81-1.86, C+ 1.81-1.82 tok/s.
  - All texts identical, 0 OOM.
- **CPU only (full model, pr1):** B, M, P and C+ are all 0.91-0.94 tok/s, with identical texts. No PR changes CPU decode.
- **Splits on the 5070 (c4m28):** B 1.535, S1 1.565, S2 1.570, S3 1.561. S3/S2 0.995, so VRAM-first adds nothing
  measurable on the 5070.
- **c4m29 (F1 on the 5070, full model, pr1):** M/B TTFT 0.98x and 0.99x; decode 1.15x and 1.17x.
  - **#1766 is NOT slower to first token on the 5070.**
  - MX/B: TTFT 0.99x, decode 1.12x and 1.10x.
- **This narrows F1.** The 5070 runs without `COLI_CUDA_MOE_BATCH` (the bank doesn't fit in 12 GB); the 3090 arms run
  with it.
  - So the extra prefill time on the 3090s most likely comes from #1766 interacting with dev's batched-MoE / GPU-bank
    prefill, or from sm_86 NO_TC.
  - final30's per-switch arms (all with the bank) will show whether a switch is involved.
  - If none is, the next arm is M1F vs B1F with `COLI_CUDA_MOE_BATCH` unset.

### shadowfax final29 (splits, REAP, one GPU), running since 16:20
- So far: B1 3.011; S1 3.007 (1.00x); S2 3.017 / 3.018 (1.00x); S3 3.102 / 3.107 (1.03x); P1 3.105.
- As expected: S1 and S2 are neutral, and S3 carries all of PR #2's one-GPU gain.
- final30 follows at about 18:00 (it was ~19:30 in the plan).
- **17:35 C+ fix checked on the 5070.** `make dsv4-cuda-test` on b659ac8d reproduces the same 4 peer-init failures;
  on dec72809 it gives "probe, init rollback and retry: OK (1 device)", rc=0. Logs in
  [manifests/computer4-20261026-samebase5](../manifests/computer4-20261026-samebase5). The two-GPU parts (peer probe,
  peer MoE) still need shadowfax after final30.

## F1 result (2026-10-26 18:45, final30 arms A/A′, one sample each)

- No #1766 switch explains the slower first token. With each switch off, and with all of them off (FMX), #1766 stays at
  98-100 / 140-142 s, against dev's 86.8 / 124.3 s (AB1F 86.8/124.3, AM1F 98.6/141.3).
- Profiles (PB1F vs PM1F, sums over both prompts' prefill):

  | stage | dev | #1766 | change |
  |---|---|---|---|
  | attention block | 79.6 s | 65.2 s | -14.4 s |
  | MoE block | 123.3 s | 165.8 s | **+42.5 s** |
  | of which batched expert compute | 72.8 s | 109.4 s | **+36.6 s (1.50x)** |
  | of which refill | 48.3 s | 54.3 s | +6.0 s |
  | missing experts | 15,638 | 15,638 | same |

- So #1766 speeds up prefill attention and slows dev's batched-MoE expert compute by half on sm_86 NO_TC, through code
  that has no switch. That fits c4m29: the 5070 runs without `COLI_CUDA_MOE_BATCH` and shows no TTFT loss.
- Next: find which #1766 change touches the batched expert path (likely the native FP8 conversion or the expert matmul
  kernels), then confirm with M1F vs B1F with `COLI_CUDA_MOE_BATCH` unset.

## final29 / final30 complete (2026-10-26 20:15)

Both passed their gates. GPUs released to Bob for the power-profile change at ~20:10.

**final29, PR #2 splits, REAP, one GPU:** B1 3.011, S1 3.010 (1.000x), S2 3.018 (1.002x), S3 3.105 (1.031x), P1 3.105.
VRAM-first (S3) carries all of PR #2's one-GPU gain; S3/P1 = 1.000.

**F1 (final30 A arms):** every #1766 switch off, alone or together, leaves TTFT at 1.13-1.15x dev. See the profile table
above: batched expert compute in prefill is 1.50x slower. Profile sums: [final30/prefill-profile-PB1F-PM1F.txt](../manifests/shadowfax-20261026-samebase5/final30/prefill-profile-PB1F-PM1F.txt).

**F2 resolved: warm-up/order artifact of the short pr1 answers.**
- fdec (full model, 3 prompts x 300 tokens), what the second GPU adds: PR #2 1.235/1.266/1.232x on all three prompts,
  #1766 1.146/1.170/1.127x, C+ 1.285/1.340/1.278x.
- Swapping pr1's prompt order moves the gain: with long_prefill_2 first, PR #2's peer gives long_prefill 1.13x and
  long_prefill_2 0.93x (the opposite of final27). The peer helps whichever prompt runs second.

**fdec decode tok/s (full model, one sample):**

| | 1 GPU | 2 GPUs |
|---|---|---|
| dev | 2.646 | |
| PR #2 | 2.697 (1.02x) | 3.357 (1.27x) |
| #1766 | 3.221 (1.22x) | 3.697 (1.40x) |
| C+ | 3.295 (1.25x) | **4.288 (1.62x; 1.16x #1766 2 GPUs)** |

**C+ fix dec72809 on both 3090s:** `make dsv4-cuda-test` rc=0, "dsv4 peer init: probe, init rollback and retry: OK
(2 devices)"; b659ac8d had 4 FAILED in final28. Log: [final30/dsv4-cuda-test-cplus-dec72809.log](../manifests/shadowfax-20261026-samebase5/final30/dsv4-cuda-test-cplus-dec72809.log).

## F1 traced: #1766's new `f4()` (final31, final32; 2026-10-26 21:50-22:25)
One GPU, full model, pr1, `COLI_CUDA_MOE_BATCH=1`, profiled, one sample each. Each run has dev and #1766 as same-session
anchors, still at 300 W. Every answer equals the reference texts. Variant trees are local commits on shadowfax (tags
`f31/m1766q`, `f31/m1766-mf4`, `f31/m1766-me4`), not pushed.

| arm | `backend_cuda_dsv4_quant.cuh` | TTFT p1 / p2 | decode p1 / p2 | expert compute | attention block |
|---|---|---|---|---|---|
| dev (final32) | (dev's helpers) | 85.6 / 123.5 s | 1.90 / 1.81 | 72.4 s | 79.2 s |
| #1766 (final32) | #1766's | 98.4 / 140.6 s | 2.22 / 2.10 | 109.2 s | 65.5 s |
| MQ (final31) | all of dev's | 85.1 / 121.8 s | 2.07 / 1.96 | 72.8 s | 76.6 s |
| **MF** (final32) | **dev's `f4` only** | **78.1 / 111.8 s** | **2.20 / 2.06** | **69.5 s** | 67.7 s |
| ME (final32) | dev's `e4m3`/`e4m3_code`/`e4m3_round` only | 104.2 / 149.0 s | 2.12 / 1.99 | 112.6 s | 74.8 s |

- **The cause is #1766's bit-arithmetic `f4()`.** With dev's 8-entry table `f4` put back, batched expert compute returns to
  dev's level.
- MF is then 0.91x dev to the first token and keeps #1766's decode (0.99x / 0.98x of M).
- #1766's FP8 helpers are what speed up attention and decode. Restoring dev's FP8 helpers (ME) loses that and doesn't fix
  the MoE.
- A synthetic microbenchmark ([final32/f4bench-microbench.cu](../manifests/shadowfax-20261026-samebase5/final32/f4bench-microbench.cu))
  had the new `f4` faster in a simplified expert-major loop. The real kernels behave differently; the engine
  measurement is what counts.
- Suggested change for #1766: keep the native FP8 conversion and dev's table `f4`. Same for C+.

## C+ with the f4 fix, and posting (2026-10-26 22:20-23:00)
- **C+ e2933955** on fork `samebase/cplus`: dec72809 plus dev's table `f4` (#1766's FP8 helpers kept).
  - nvcc builds it for sm_86 and sm_120; the source and loader tests pass.
  - On shadowfax it is built as `cw22/f33-cplusf4`, with a logged clean build, 0 diagnostics beyond lto-wrapper.
- **final33** (shadowfax, full model, one sample each, at 300 W): AB1F, then old C+ (b659ac8d) against fixed C+ on two
  GPUs and on one (pr1), then DB1F/DC2F/DN2F (fdec). It runs until about 23:35.
  - First sample: XC2F (old C+, 2 GPUs) is again at 98.4 / 141.6 s to first token.
- **final34** (shadowfax, REAP): B1, then old against fixed C+, C2/N2 x2 and C1/N1 x1. It waits for final33 (unit
  cbench-f34) and runs until about 00:50.
- **Posted at 22:51 with Bob's OK:** the #1766 comment, and #2405's full description rewrite (#2405 still a draft).
  - Both say their C+ numbers are from before the `f4` fix and will be updated.
- `make check` on 11c40038 is running on counter. It goes into #2405 when it finishes.

## Fixes branch for #1766, and C+ test fixes (2026-10-26 ~23:05)
- **Fork `samebase/m1766-fixes` 28b35786** = `samebase/m1766` (ecc92586, the measured M, unchanged) plus four commits:
  1. aa1b4f52 `dsv4_cuda_init`: a refused device leaves no stale CUDA error. Hand-written for #1766: C+'s dec72809 builds
     on PR #2's init_fail, so it doesn't cherry-pick.
  2. 7411d188 `f4`: table decode again. This is C+'s e2933955 with the internal run names dropped from the comment.
  3. 2c675407 the dsv4 CUDA tests take two devices from `DSV4_TEST_DEVICES` (default 0,1), through #1766's own
     `coli_v4_gpu_devices_parse`.
     - It covers eight tests, not the seven the posted comment lists: `test_dsv4_quant_cuda.cu` also looped over {0,5}.
  4. 28b35786 `native_quant.h`: the GGUF helpers are C only.
     - **Correction to the posted comment's item 4:** the g++ build break is not #1766's doing. At #1766's head,
       native_quant.h includes no ggml header. dev later added `#include "ggml_view.h"` → ggml_quants.h, whose type
       table uses C99 designated array initializers.
     - #1766's test_dsv4_indexer_prepare_cuda.cpp is the only C++ includer, so it breaks only after the merge.
- **C+ `samebase/cplus` 66982895** = e2933955 + commits 3 and 4 (cherry-picked) + an f4 comment wording change. No
  engine change from e2933955, which final33/34 measure.
- Checks on counter: the C++ tests pass a syntax check, and the source and loader tests pass.
- **GPU check queued:** shadowfax unit `cbench-t35` (`~/colibri-bench/t35.sh`). It runs after cbench-f34 (~00:40) and
  builds both tips. It runs the 8 unit tests with `DSV4_TEST_DEVICES=0,1`, a bad pair (0,7) and a single device (both
  must be refused), and `dsv4-cuda-test` on C+.
- Then edit the posted #1766 comment:
  - item 3 lists eight files;
  - item 4 says it is dev's header;
  - add a pointer to `samebase/m1766-fixes`;
  - update the C+ numbers.
- **23:07 (Bob's OK): the #1766 comment's items 3 and 4 are corrected** (eight test files; item 4 is dev's header). The fixes-branch pointer and the C+ numbers come after cbench-t35 and final34.

## final33: C+ before/after the f4 fix, full model (22:37-23:12, GATES: PASS, one sample each, 300 W)
| | first token p1 / p2 | decode (pr1) mean | fdec decode, 2 GPUs |
|---|---|---|---|
| dev, 1 GPU (AB1F / DB1F) | 85.6 / 123.5 s | 1.857 | 2.649 (1 GPU) |
| C+ b659ac8d, 1 GPU | 98.5 / 140.9 s | 2.204 | |
| **C+ e2933955, 1 GPU** | **78.0 / 111.6 s** (0.91x dev) | 2.177 (0.99x old) | |
| C+ b659ac8d, 2 GPUs | 98.4 / 141.6 s | 2.499 | 4.292 (1.62x dev) |
| **C+ e2933955, 2 GPUs** | **77.9 / 112.3 s** (0.91x dev) | 2.508 (1.00x old) | **4.209 (1.59x dev, 0.98x old)** |

- The f4 fix takes C+'s first token from 1.15x dev to 0.91x dev, one GPU or two.
- Decode moves by 0-2% (fdec 0.98x), within what MF showed against M (0.98-0.99x).
- Answers are identical.
- For the C+ rows of both texts: V4 Flash 2 GPUs becomes 1.59x (was 1.62x). The 1-GPU fdec arm wasn't rerun, so 1.25x
  stays the pre-fix value. REAP comes from final34.

## final34: C+ before/after the f4 fix, REAP (run 1 23:14-00:05, run 2 00:05-00:57, both GATES: PASS, 300 W)
| arm | run 2 decode | run 1 decode | vs dev (run 2) |
|---|---|---|---|
| dev B1 | 3.012 | 3.008 | 1.00x |
| C+ b659ac8d, 1 GPU | 3.913 | 3.919 | 1.30x |
| **C+ e2933955, 1 GPU** | **3.828** | 3.829 | **1.27x** |
| C+ b659ac8d, 2 GPUs | 6.064 | 6.056 | 2.01x |
| **C+ e2933955, 2 GPUs** | **5.905** | 5.911 | **1.96x** |

- The f4 fix costs 2.2% (1 GPU) and 2.6% (2 GPUs) of REAP decode; TTFT 0.99-1.01x old. Answers identical.
- The waiter ran the arms once itself, then the unit ran them again; run 1's summary is final34/summary-run1.txt.
- t35 (00:58): all CUDA unit tests pass on both fixes branches; see m1766-unit-tests-02.
- Both texts updated 2026-10-27 08:10 with these numbers; notes links pinned to 45c5c5e.
