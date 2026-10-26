# Draft: comment for JustVugg/colibri#1766 (NOT POSTED; needs Bob's OK)

Written 2026-10-26 ~04:00 from final23/final24 (shadowfax) and c4m22 (computer4). Data pinned at colibri-notes a1f96fc.
To post: `gh pr comment 1766 -R JustVugg/colibri --body-file <the part between the lines>`.

Optional second step, same decision: add the "#2405 Related" paragraph at the bottom to our own #2405 body.

---

Here is the dev-vs-branch table from the open point, measured on two machines.

**Trees.** This branch at ca55947c, against the dev it last merged, 88f25a60 ("dev" below), with no other changes.
- Both are built from their own tree with the same make line.
- Both run with this branch's documented serve line (`COLI_CUDA_ATTN_BATCH=1 COLI_CUDA_MOE_BATCH=1`) and `--ram 40 --ctx 8192`.
- Each sample gets a fresh server, and samples are interleaved.
- The branch runs at its defaults (`DSV4_CUDA_RESIDENT_EXPERTS=auto`) unless a row says otherwise.

### 2x RTX 3090 24 GB (NVLink), Ryzen 9 9950X3D, 62 GB

`make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_ARCH=sm_86 NO_TC=1`, CUDA 12.0.

**REAP-150B, 6 prompts x 300 tokens, 2 samples per row** (decode = median of the per-sample mean):

| | GPUs | decode tok/s | vs dev | per prompt | TTFT vs dev |
|---|---|---|---|---|---|
| dev 88f25a60 | 1 | 2.875 | 1.00x | | |
| this branch | 1 | 3.567 | **1.24x** | 1.21-1.30x | 0.95-0.97x |
| this branch, `DSV4_CUDA_DEVICES=0,1` | 2 | 4.603 | **1.60x** | 1.55-1.66x | 0.95-0.97x |

- All 36 answers are byte-identical across the three rows.
- In both configurations the branch's residency fell back to the cache, which is what `auto` should do here:
  - one GPU: `resident-experts=cache-fallback (resident experts exceed device 0 budget (free=16749 MiB reserve=2800 MiB))`
  - two GPUs: the same, with `free=20313 MiB`.
- No handled allocation failures in any sample.

**Full DeepSeek V4 Flash, prompts of 1,992 and 3,005 tokens** (`DSV4_CUDA_VRAM_RESERVE_MB=4700` on both sides; see the
note below):

| | GPUs | samples | decode tok/s, prompt 1 / 2 | mean | vs dev | TTFT, prompt 1 / 2 |
|---|---|---|---|---|---|---|
| dev 88f25a60 | 1 | 2 | 1.866 / 1.768 | 1.817 | 1.00x | 137.6 / 200.2 s |
| this branch | 1 | 2 | 2.172 / 2.046 | 2.109 | **1.16x** | 116.1 / 167.0 s (0.84x / 0.83x) |
| this branch, `DSV4_CUDA_DEVICES=0,1` | 2 | 2 | 2.298 / 2.102 | 2.200 | **1.21x** | 116.3 / 167.3 s |

- Every answer equals the reference texts that dev gives on this model.
- No handled allocation failures in any sample.

Why the reserve: at the default 2800 MiB, dev and this branch both lose the 256-slot MoE bank on the second prompt.
- Each logs `expert FC2 bank allocation: out of memory` (or FC1), then `moe-batch=off`.
- The second prompt then takes 388-402 s to first token and gives a different answer, on both sides.
- That is dev's mirror reserve not counting the bank. Current dev has the same gap, and #2401 fixes it (its bank-aware
  default is 4681 MiB). It is not this branch's doing.

The default-reserve runs, one sample each:

| | GPUs | decode mean | TTFT, prompt 1 / 2 |
|---|---|---|---|
| dev | 1 | 1.947 | 137.1 / 402.2 s |
| this branch | 1 | 2.312 | 115.3 / 388.5 s |
| this branch | 2 | 2.514 | 117.3 / 388.6 s |

This branch's `test_dsv4_multigpu_cuda 0,1` passes on these two cards.

### 1x RTX 5070 12 GB, CUDA 12.8, `CUDA_ARCH=sm_120`

Full V4 Flash, 6 prompts x 300 tokens, 2 samples each, `DSV4_CUDA_VRAM_RESERVE_MB=2000` on both sides (the desktop holds
1.9 GB of the card):

| | decode tok/s, s1 / s2 | vs dev | median TTFT |
|---|---|---|---|
| dev 88f25a60 | 1.592 / 1.595 | 1.00x | 6.5 s |
| this branch | 1.885 / 1.884 | **1.18x** | 6.4-6.6 s |

- All 24 answers are byte-identical, with no handled failures.
- Residency fell back: `free=2577 MiB reserve=2000 MiB`.
- At the default reserve this card is too tight for a clean comparison. The branch logged 543k and 312k handled
  `[DSV4 CUDA] ... out of memory` lines in its two runs, and dev logged 0 and 167k. Texts changed on both sides, so I
  left those runs out of the table.

Logs, scripts and per-request numbers:
- 3090s, REAP and default-reserve full model: https://github.com/bobsummerwill/colibri-notes/tree/a1f96fc/manifests/shadowfax-20261026/final23
- 3090s, reserve 4700: https://github.com/bobsummerwill/colibri-notes/tree/a1f96fc/manifests/shadowfax-20261026/final24
- 5070: https://github.com/bobsummerwill/colibri-notes/blob/a1f96fc/manifests/computer4-20261026-m1766/SUMMARY.md

---

## Optional: #2405 "Related" paragraph (our own draft PR)

Add these sub-bullets under the #1766 bullet in #2405's Related section, after the two that are there:

>   - Measured in the same session on 2x RTX 3090, REAP-150B, 6 prompts x 300 tokens, 2 samples each, each against its
>     own dev base: #1766 on two GPUs **1.60x** (4.60 tok/s), this PR **1.43x** (4.34 tok/s). Head to head, this PR
>     decodes at 0.94x of #1766 and reaches the first token 6-12% sooner. The two bases differ: this PR's one-GPU base
>     runs at 3.02 tok/s and #1766's at 2.87.
>   - On the full V4 Flash with the bank kept, one sample of this PR and two of #1766: this PR 2.10 tok/s, #1766 2.20.
>     Time to first token: 86 / 124 s for this PR, 116 / 167 s for #1766.
>   - Logs: [final23/final24](https://github.com/bobsummerwill/colibri-notes/tree/a1f96fc/manifests/shadowfax-20261026)
