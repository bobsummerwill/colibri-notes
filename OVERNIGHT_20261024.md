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
| 4 | PR #2 merged with the new PR #1 tip (47c8ec51): `docs5/pr2-two-gpu-decode` 14d67f05, `make check` | computer4 | merged; check running |
| 5 | **Upstream fix A**: the compressed-KV buffer (`dsv4_cuda_kv_comp_append`) fails to grow once the decode mirrors have filled VRAM, and attention falls back to the CPU (text changes). Branch off dev 61646cbd | computer4 + shadowfax | investigating |
| 6 | **Upstream fix B**: the Vulkan tier's host pool OOM-kills the server with the default size (still on dev 61646cbd: final9 WQ/QD, 15.75 GiB pool, only 630 MiB counted for later state). Branch off dev | computer4 + shadowfax | after 5 |
| 7 | A possible **upstream fix C**, if final9b confirms it: the Vulkan dense hook's upload refusal is silent and permanent | | depends on 2 |
| 8 | **PR #2 F18**: after a peer fault, every later request fails (the error is sticky for the process's CUDA). Fix the false message; latch the CUDA backend off so later requests run on the host | computer4 + shadowfax | after 5-6 |
| 9 | PR #2 retest on the new tip (two-GPU decode A/B, peer tests) | shadowfax | after 8 |
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
