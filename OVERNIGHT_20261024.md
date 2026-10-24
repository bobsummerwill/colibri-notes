# Overnight plan and log, 2026-10-24 (night of 10-23 PDT)

> Bob's brief (2026-10-23 ~23:00 PDT): finish everything around **PR #1** (#2264). Then fix the two upstream
> problems on branches, **with no upstream issues and no PRs**. Carry on with **PR #2** (F18, rebase, retest). Do
> anything else useful, such as optimisations from the notes. GPUs and computers are free all night. This file is
> updated as the work goes on. Outward-facing steps (comments on #2264, issues, PRs) wait for Bob; drafts go in
> `review/`.

## Queue (in order)
| # | item | where | state |
|---|---|---|---|
| 1 | **final9**: pure Vulkan with the default host pool on the merged tips; Vulkan + CUDA diagnostic arms ([TEST_PLAN_20261024.md](TEST_PLAN_20261024.md)) | shadowfax | running (started 22:52) |
| 2 | **final9b**: diagnostic builds that log the Vulkan dense hook's silent CPU fallbacks; DA dev union, DB PR #1 bank, DF bank with a 6000 MiB reserve | shadowfax | chained after final9 |
| 3 | Corrected Vulkan + CUDA explanation for #2264 (draft in `review/`, not posted) | computer4 | after 1-2 |
| 4 | PR #2 merged with the new PR #1 tip (47c8ec51): `docs5/pr2-two-gpu-decode` 14d67f05 | computer4 | **done**: `make check` 2,716 OK (292 skipped), pushed to the fork |
| 5 | **Upstream fix A**: the compressed-KV buffer (`dsv4_cuda_kv_comp_append`) fails to grow once the decode mirrors have filled VRAM, and attention falls back to the CPU (text changes). Root cause: the mirror growth guard overshoots the reserve by up to 64 mirrors between free-VRAM probes (F16, fixed inside PR #2's 6b7fc6c0 but not on dev). Branch `fix/dsv4-mirror-guard-overshoot` bd7103fd off dev 61646cbd (guard + docs + source test; pushed to the fork) | computer4 + shadowfax | committed; **final10** measures it (W A F F A A F, full model, default config) |
| 6 | **Upstream fix B**: the Vulkan tier's host pool OOM-kills the server with the default size (still on dev 61646cbd: final9 WQ/QD/QE, every sample, 15.75 GiB pool, only 630 MiB counted for later state). The kill reports ~44 GB anon + ~17 GB file RSS. **final10** M/M2 records memory by class every 5 s (and M2 turns off the uploads' host-page imports) before any code change | computer4 + shadowfax | measuring |
| 7 | A possible **upstream fix C**, if final9b confirms it: the Vulkan dense hook's upload refusal is silent and permanent | | depends on 2 |
| 8 | **PR #2 F18**: after a peer fault, every later request fails (the error is sticky for the process's CUDA). The host-wait probe variant (f18hw 1b93f1ef) was built but never run: it is now built on shadowfax and runs at the end of final10. If device 0 survives with the host waiting for the peer's join, PR #2 waits on the host; if not, the backend latches CUDA off process-wide and the message says so | computer4 + shadowfax | probe queued |
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
