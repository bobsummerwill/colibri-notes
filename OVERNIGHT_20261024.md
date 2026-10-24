# Overnight plan and log, 2026-10-24 (night of 10-23 PDT)

> Bob's brief (2026-10-23 ~23:00 PDT): finish everything around **PR #1** (#2264). Then fix the two upstream
> problems on branches, **with no upstream issues and no PRs**. Carry on with **PR #2** (F18, rebase, retest). Do
> anything else useful, such as optimisations from the notes. GPUs and computers are free all night. This file is
> updated as the work goes on. Outward-facing steps (comments on #2264, issues, PRs) wait for Bob; drafts go in
> `review/`.

## Queue (in order)
| # | item | where | state |
|---|---|---|---|
| 1 | **final9**: pure Vulkan with the default host pool on the merged tips; Vulkan + CUDA diagnostic arms ([TEST_PLAN_20261024.md](TEST_PLAN_20261024.md)) | shadowfax | **done** 23:35 ([manifests/final9](manifests/final9/summary.txt)) |
| 2 | **final9b**: diagnostic builds that log the Vulkan dense hook's silent CPU fallbacks; DA dev union, DB PR #1 bank, DF bank with a 6000 MiB reserve | shadowfax | **done** 23:53 ([manifests/final9b](manifests/final9b/summary.txt)): **no refusals**, so the refusal theory is wrong; v2 of the patch (declines with their reason) runs after final10 |
| 3 | Corrected Vulkan + CUDA explanation for #2264 (draft in `review/`, not posted) | computer4 | **drafted**: [review/REPLY_2264_FOLLOWUP_20261024_DRAFT.md](review/REPLY_2264_FOLLOWUP_20261024_DRAFT.md) |
| 4 | PR #2 merged with the new PR #1 tip (47c8ec51): `docs5/pr2-two-gpu-decode` 14d67f05 | computer4 | **done**: `make check` 2,716 OK (292 skipped), pushed to the fork |
| 5 | **Upstream fix A**: the compressed-KV buffer (`dsv4_cuda_kv_comp_append`) fails to grow once the decode mirrors have filled VRAM, and attention falls back to the CPU (text changes). Root cause: the mirror growth guard overshoots the reserve by up to 64 mirrors between free-VRAM probes (F16, fixed inside PR #2's 6b7fc6c0 but not on dev). Branch `fix/dsv4-mirror-guard-overshoot` bd7103fd off dev 61646cbd (guard + docs + source test; pushed to the fork) | computer4 + shadowfax | **works** (final10): 421 -> 0 handled allocation failures per sample, texts identical, p2 TTFT 0.946x, answer-2 decode +5.2%; computer4 RTX 5070 (c4fa, done 03:10): 0 fallbacks vs 399-420, texts identical, p2 TTFT 0.918x, answer-2 decode +6.6% ([manifests/c4fa](manifests/c4fa/summary.txt)). Write-up: [review/UPSTREAM_FIXES_20261024.md](review/UPSTREAM_FIXES_20261024.md) |
| 6 | **Upstream fix B**: the Vulkan tier's host pool OOM-kills the server with the default size (dev 61646cbd: every default-pool sample in final9 and final10 M). final10: file RSS = the pool (`/dev/nvidiactl` mappings, 16.3 GiB), anon = the 37.5 GiB expert cache + ~3-4 GiB the plan does not count; M2 (`COLI_VK_UP_IMPORT=0`) survived with 0.5 GiB free (imports cost ~1.5 GiB, and halve the upload rate: 3.1 vs 6.6 GB/s). The pool's sizing leaves a fixed 4 GiB for everything the engine allocates later; deepseek_v4's `ram_later` counts 630 MiB of it. final11 M3/M4 (smaps every 15 s; pool off) locates the growth before the fix | computer4 + shadowfax | **fixed** (final15): branch `fix/vk-host-image-alloc` 72aa325a off dev (pushed to the fork): anon 45.4 -> 40.3 GiB, p1 TTFT 0.957x, the default pool runs both prompts (dev: OOM-killed). [review/UPSTREAM_FIXES_20261024.md](review/UPSTREAM_FIXES_20261024.md) |
| 7 | A possible **upstream fix C**, if final9b confirms it: the Vulkan dense hook's upload refusal is silent and permanent | | depends on 2 |
| 8 | **PR #2 F18**: after a peer fault, every later request fails (the error is sticky for the process's CUDA). Probe (final10, 00:02): host-wait does **not** help, device 0 is lost in all 4 modes. Fix c42ad524 on `docs5/pr2-two-gpu-decode`: a sticky error latches CUDA off process-wide (ctx() answers no device, `dsv4_cuda_lost()`), mirror lookups miss, the decode MoE takes host copies, honest messages; test hook `DSV4_CUDA_PEER_FAULT=n:sticky` + a sticky scenario in the peer test + a source test | computer4 + shadowfax | **fixed** (final13 S7c, 03:42-04:01): after the sticky fault all 6 answers complete on the host (2.94 tok/s), X3c (no fault) = D0's 6 texts; `docs5/pr2-two-gpu-decode` 6628f5b1 + 8eedc83b + 693204ed pushed to the fork |
| 9 | PR #2 retest on the new tip: final11 = final6's decode arms (D0 dev, G1 PR #2 switches off, X3 RESIDENT_FIRST + PEER_DEVICE; 6 x 300-token answers, x3) plus `dsv4-cuda-test` / `cuda-test`, and the F18 real-server fault (final7 F7) on the fixed build | shadowfax | **done** ([manifests/final11](manifests/final11/summary.txt)): D0 2.808, G1 2.807 (0.999x), X3 3.919 tok/s (**1.437x dev**), every text = the D0 reference, CLEAN |
| 10 | Optimisations from the notes (decode attention ~107 ms of ~258 ms per token, store fetch waits, uploads) | | if time |

## Log
- 22:49 final9 launched. 22:55 WQ (dev warm-up, default pool) was **OOM-killed after prompt 1**. The new sizing chose
  15.75 GiB ("630.2 MiB of RAM left for the KV and state the engine allocates later"). So 0f786bfa + 5c0469ae do
  not fix this case. QD s1 likewise.
- 23:3x Code reading for item 3. The hook uploads each matrix on first use, and with no device room it sets
  `refused = 1` ("CPU from now on") without logging anything. GPU 0 peaked at 23,476 MiB in the bank arm against
  21,784 in the union arm (final8). final9b tests this.
- 23:4x Item 5. On full V4 Flash with the default config, GPU 0 peaks at 24,116 of 24,576 MiB (final6 A). The
  failures start on prompt 2, after prompt 1's decode grew the mirrors (252 per sample). The buffer is small
  (2,048 rows × 512 × 4 B = 4 MiB per layer at ctx 8192), so the card is effectively full. The 600 MiB reserve is
  checked only when the mirror cache grows.
- 23:5x GPU queue: final9 (running) -> final9b (diagnostic builds, then 3 arms) -> final10 (fix A CUDA builds made
  during final9b; W A F F A A F on the full model; M / M2 memory probe; F18 host-wait probe last). Each starts by
  itself when the previous one ends (background chains on computer4).
- final9 so far: QD x3 and QE x3 (default pool) **all OOM-killed after prompt 1**. VA9 (dev, Vulkan + CUDA, merged
  tips) = final8's VA2 byte for byte (`6ec1e4b7` / `1bdc5d70`).
- **final9 results (23:35).** Pure Vulkan, default pool: all 7 samples (WQ, QD x3, QE x3) OOM-killed after prompt 1
  (RSS 58-58.7 GiB, MemAvailable min 0-0.6 GiB). The prompt-1 texts differ because the answers were cut off while the
  box thrashed (decode 1.2-4.1 tok/s). Vulkan + CUDA, REAP, 3100 MiB reserve, one sample each:

  | arm | p1 | p2 |
  |---|---|---|
  | VA9 dev union | `6ec1e4b7` VK+CUDA | `1bdc5d70` VK+CUDA union |
  | VB9 PR #1 bank | `6ec1e4b7` VK+CUDA | `1f019203` CUDA |
  | VD9 dev union, `COLI_VK_DENSE=0` | `32534c06` CUDA | `1f019203` CUDA |
  | VE9 PR #1 bank, `COLI_VK_DENSE=0` | `32534c06` CUDA | `1f019203` CUDA |
  | VX9 dev's VK build, `COLI_VULKAN=0` | `32534c06` CUDA | `1f019203` CUDA |

  **Every Vulkan + CUDA text difference comes from the Vulkan dense matmul hook.** With it off, dev's union and
  PR #1's bank give the same texts as the CUDA-only build, on both prompts. TTFT: bank / union 0.582 / 0.534.
- **final9b (23:53):** the diagnostic builds logged **0 refused uploads** in DA, DB and DF. DF (bank, 6000 MiB
  reserve, GPU 0 peak 20.2 GB) still gives `1f019203` on p2, so the cause is not VRAM room. The tally threshold
  (every 20000 calls) was never reached, so the hook makes fewer calls than that per run. v2 of the patch logs every
  declined prefill-sized call with its reason (`thread` / `omp` / other) and a tally every 500 calls; it runs after
  final10.
- **final10 first part (23:58-00:03).** M (dev, pure Vulkan, default pool 16 GiB) OOM-killed after prompt 1;
  memx: anon 41.7 + file 16.7 GiB, MemFree 0.6. M2 (same, `COLI_VK_UP_IMPORT=0`) completed both prompts: anon
  40.2 + file 17.0, MemFree 0.5 (a near miss, not a fix), and its uploads ran at 6.6 GB/s against 3.1 with imports
  (TTFT 96.1 vs 102.6 s): **the imports look like a slowdown on this box** (an optimisation lead).
  smaps at 55 GiB: a 30 GiB anon slab (the expert cache), `/dev/nvidiactl` 16.3 GiB (the pool), heap 1.6 GiB, and
  ~4 GiB in anon regions of 768-row-step sizes (11 x 123 MiB, 1024, 1014, 850 MiB).
- **F18 probe (00:02):** a real `__trap()` on the peer, before or after its down-projection, device 0's stream or
  the host waiting for the join: in all 4 modes the call is refused cleanly and **device 0's next call fails**
  ("PRIMARY LOST"). Host-wait does not help. So PR #2 latches CUDA off for the process (c42ad524).
- 00:18 fix A: W and A s1 done (dev: the reference texts `a397814f` / `8b3f437b`, TTFT 161 / 270 s).
- final11 queued (after final10): diag v2 (DA2 / DB2), M3 / M4, PR #2 + F18 CUDA tests, S7 (sticky fault in a
  real server), then the PR #2 decode retest (WR D0 G1 X3 x3).
- **final10 done (00:58).** Fix A: see the queue table and review/UPSTREAM_FIXES_20261024.md. F18 fix amended to
  6628f5b1 (a local `extern` declaration: the build warned about an implicit declaration of `dsv4_cuda_lost` in the
  decode MoE, which comes before the backend header; final11 runs the c42ad524 binary, which behaves the same).
- 01:00 final11 started on shadowfax; 01:01 c4fa (fix A on the RTX 5070, DeepGEMM) started on computer4.
- **final11 first part (01:00-01:17).** DA2 / DB2 (diag v2): the union and the bank make the **same 1,720
  prefill-sized hook calls on the device, 0 declined**, 0 refused; texts as before (`1bdc5d70` vs `1f019203` on p2).
  So the hook is used identically, and the p2 difference comes from the expert arithmetic (bank: CUDA; union: CPU;
  not bit-identical in general, F42 2026-10-16), landing on a near-tie under the hook's slightly different gate logits
  / projections. Draft follow-up for #2264 written.
- **M3** (default pool, smaps every 15 s): OOM-killed again. **M4 (pool off): completed, anon 45.6 GiB**, against the
  engine's own RAM plan `projected=39.61GiB` (cache 37.48). So in pure Vulkan the engine itself overshoots its `--ram 40`
  plan by ~6-8 GiB (CUDA runs of the same engine peak at 39.9 GiB RSS, in plan). M3's smaps: ~6 GiB of non-cache anon
  already at server start (a 1010 MiB region, many 63-127 MiB regions: per-thread malloc arenas?), more regions of
  0.3-1.8 GiB during prefill. Next: an LD_PRELOAD tracer of allocations >= 32 MiB with backtraces (after final11).
- **final11 done (02:57)** ([manifests/final11](manifests/final11/summary.txt)). PR #2 decode retest on the new tip
  (REAP, 6 x 300-token answers, x3): D0 dev 2.808 tok/s, G1 (PR #2, switches off) 2.807 (0.999x), X3
  (RESIDENT_FIRST + PEER_DEVICE) 3.919 (**1.437x dev**, 1.439x G1); all 54 answers identical to D0's references.
  (final6 measured 3.011 / 2.983 / 4.287 on the old tips; this run's box is ~7% slower overall, the ratios hold:
  final6 X3/D0 1.424x.) S7 (sticky peer fault at pass 3000 in a real server): the new messages print, nothing
  crashes, but every later request fails after 1 token ("block computation failed"). M3 OOM-killed (gate FAIL is
  that arm only, expected).
- 02:59 final12 started: S7b (stage-named decode errors, 8eedc83b), T1 (bigalloc trace of pure Vulkan, pool off),
  then the `COLI_VK_UP_IMPORT` A/B (W, I1/I0 x3). computer4 c4fa: 5 A samples all 420 kv comp + 1 weight
  fallbacks; B s5 running.
- **final12 done (03:40)** ([manifests/final12](manifests/final12/summary.txt)).
  - **S7b** names the stage: `block computation failed in attention, layer 20: attention computation failed` on the
    faulting request, then `in attention, layer 0` on each later one. Cause: the attention's grouped `wo_a` matvec
    (`coli_v4_gpu_matvec_grouped`, under `V4_GPU_RESIDENT_FIRST`) returned the backend's refusal as the attention's
    error; every other GPU call falls back. Fix **693204ed** on `docs5/pr2-two-gpu-decode` (the CPU groups run when the
    backend refuses; source test). `make check` 2,721 OK (292 skipped). final13 S7c tests it in a server; X3c checks
    the no-fault texts.
  - **Imports A/B** (pure Vulkan, pool 12 GiB, x3): `COLI_VK_UP_IMPORT=0` uploads at 6.2-6.6 GB/s against 3.5-4.0,
    p1 TTFT **0.931x** (96.4 vs 103.5 s), p2 0.988x, decode +0.8%, and **5 GiB less anon** (40.3 vs 45.3 GiB). Its p2
    text differs (`a683dba1` vs `1f019203`, all 3 samples): the tier's device/CPU split per expert depends on the
    measured upload rate (I1 s1 logged "cold experts with 23 rows or more" once), so the arithmetic differs; p1 is
    the same.
  - **T1** (bigalloc trace, pool off): the >= 32 MiB allocations live at the peak are 10.2 GiB, of which 4.0 GiB are
    `v4_dho_place`'s PROT_NONE drops (no RSS), 1.0 GiB the session, 1.0 GiB the expert store. So the 6-8 GiB past the
    plan is in smaller allocations. With final12's 5 GiB imports delta, the lead is the uploader's host images
    (`up_img_alloc`, backend_vulkan.c): with imports they are `posix_memalign` + `memset`, otherwise `calloc`, and
    glibc's dynamic mmap threshold could serve the 5-15 MiB images from per-thread arenas that keep the pages (M3's
    smaps had many 63-127 MiB anon regions). final14 tests it: H1 pool 12 + imports + a fixed
    `glibc.malloc.mmap_threshold=4194304`; H3 the default pool without imports; H2 the default pool with imports and
    the fixed threshold.
- **c4fa done (03:10)**: fix A on the RTX 5070 (DeepGEMM, full model, x5 pairs): 0 handled allocation failures
  (dev 399-420 per sample), texts identical, p1 TTFT 1.004x, **p2 TTFT 0.918x**, answer-2 decode +6.6%. (B s5's p2
  overlapped my `make check` on computer4 and is an outlier.)
- **final13 done (04:01)** ([manifests/final13](manifests/final13/summary.txt)). **F18 fixed.** S7c (PR #2 all on,
  sticky fault at the 3000th peer pass, on 693204ed): the latch messages print once, and **all 6 answers complete**
  on the host paths (2.85-3.01 tok/s; texts differ from D0's, as expected: decode_0 is cut over mid-answer, and the
  host expert sums are not CUDA's bits). X3c (no fault): all 6 texts = D0's references, 4.33 tok/s. Pushed
  `docs5/pr2-two-gpu-decode` (6628f5b1, 8eedc83b, 693204ed) to the fork; it backs no upstream PR.
- 04:03 final14 started (fix B allocator theory: H1, H3, H2).
- **final14 done (04:20)** ([manifests/final14](manifests/final14/summary.txt)): the allocator theory holds.
  H1 (pool 12 GiB, imports on, `GLIBC_TUNABLES=glibc.malloc.mmap_threshold=4194304`): anon **40.1 GiB** (I1 45.3).
  H2 (the **default pool**, imports on, the fixed threshold) and H3 (the default pool, imports off): **both
  completed, no OOM** (anon 40.2, file 17.0, MemAvailable min ~1 GiB), the first default-pool pure-Vulkan runs on
  dev that were not killed. But the fixed threshold costs speed (an mmap + munmap per image: uploads 2.8 GB/s,
  decode 3.7 vs 4.4 tok/s), so it is not the fix. The difference between the two image allocators (imports:
  `posix_memalign` + `memset`; no imports: `calloc`) is the lead: candidate **8c1f0381** on
  `fix/vk-host-image-alloc` (off dev) aligns the image by hand inside a `calloc`'d block. final15 measures it (J2
  fixb pool 12 vs C1 dev x3, J1 fixb with the default pool x2).
- **final15 done (05:05)** ([manifests/final15](manifests/final15/summary.txt)): **fix B works.** J2 (fix B, pool
  12 GiB, imports on) vs C1 (dev, same), x3 interleaved: anon 40.3 vs 45.4 GiB, MemAvailable min 4.8 vs 0.5-1.0,
  p1 TTFT 0.957x, p2 0.985x, decode 1.007x. J1 (fix B, **default pool**): both prompts 2 of 2 (dev OOM-killed
  every time). Pure-Vulkan p2 texts vary run to run on dev itself (C1 gave 3 different ones), so they are not a
  criterion. Committed as 72aa325a (comment + source test on top of the measured 8c1f0381), `make check` 2,702 OK,
  pushed to the fork. No upstream issue or PR (Bob).
