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
| 3 | Corrected Vulkan + CUDA explanation for #2264 (draft in `review/`, not posted) | computer4 | after 1-2 |
| 4 | PR #2 merged with the new PR #1 tip (47c8ec51): `docs5/pr2-two-gpu-decode` 14d67f05 | computer4 | **done**: `make check` 2,716 OK (292 skipped), pushed to the fork |
| 5 | **Upstream fix A**: the compressed-KV buffer (`dsv4_cuda_kv_comp_append`) fails to grow once the decode mirrors have filled VRAM, and attention falls back to the CPU (text changes). Root cause: the mirror growth guard overshoots the reserve by up to 64 mirrors between free-VRAM probes (F16, fixed inside PR #2's 6b7fc6c0 but not on dev). Branch `fix/dsv4-mirror-guard-overshoot` bd7103fd off dev 61646cbd (guard + docs + source test; pushed to the fork) | computer4 + shadowfax | **works** (final10): 421 -> 0 handled allocation failures per sample, texts identical, p2 TTFT 0.946x, answer-2 decode +5.2%; computer4 RTX 5070 run c4fa (W + 5 pairs) running since 01:01. Write-up: [review/UPSTREAM_FIXES_20261024.md](review/UPSTREAM_FIXES_20261024.md) |
| 6 | **Upstream fix B**: the Vulkan tier's host pool OOM-kills the server with the default size (dev 61646cbd: every default-pool sample in final9 and final10 M). final10: file RSS = the pool (`/dev/nvidiactl` mappings, 16.3 GiB), anon = the 37.5 GiB expert cache + ~3-4 GiB the plan does not count; M2 (`COLI_VK_UP_IMPORT=0`) survived with 0.5 GiB free (imports cost ~1.5 GiB, and halve the upload rate: 3.1 vs 6.6 GB/s). The pool's sizing leaves a fixed 4 GiB for everything the engine allocates later; deepseek_v4's `ram_later` counts 630 MiB of it. final11 M3/M4 (smaps every 15 s; pool off) locates the growth before the fix | computer4 + shadowfax | measuring |
| 7 | A possible **upstream fix C**, if final9b confirms it: the Vulkan dense hook's upload refusal is silent and permanent | | depends on 2 |
| 8 | **PR #2 F18**: after a peer fault, every later request fails (the error is sticky for the process's CUDA). Probe (final10, 00:02): host-wait does **not** help, device 0 is lost in all 4 modes. Fix c42ad524 on `docs5/pr2-two-gpu-decode`: a sticky error latches CUDA off process-wide (ctx() answers no device, `dsv4_cuda_lost()`), mirror lookups miss, the decode MoE takes host copies, honest messages; test hook `DSV4_CUDA_PEER_FAULT=n:sticky` + a sticky scenario in the peer test + a source test | computer4 + shadowfax | committed (not pushed yet); final11 tests it (dsv4-cuda-test, S7 in a real server) |
| 9 | PR #2 retest on the new tip: final11 = final6's decode arms (D0 dev, G1 PR #2 switches off, X3 RESIDENT_FIRST + PEER_DEVICE; 6 x 300-token answers, x3) plus `dsv4-cuda-test` / `cuda-test`, and the F18 real-server fault (final7 F7) on the fixed build | shadowfax | after 8 |
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
