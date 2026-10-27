# Draft: comment for JustVugg/colibri#1766 (NOT POSTED; needs Bob's OK)

Redrafted 2026-10-26 ~22:00 from the same-base runs (shadowfax final27-final30, computer4 c4m27-c4m29). Replaces the
04:00 draft (base-mixed; see git history of this file). Data pinned at colibri-notes d00475b.
To post: `gh pr comment 1766 -R JustVugg/colibri --body-file <the part between the lines>`.

Open before posting:
- [ ] F1 cause: fill in the sentence marked TRACE once the trace (task 1) is done, or drop it.
- [x] #1766's own unit tests patched to devices {0,1}: filled in (manifests/shadowfax-20261026-samebase5/m1766-unit-tests-01).
- [ ] Bob: keep or drop the optional last section (this branch together with #2405's peer tier).

---

I re-measured this branch on current dev, with everything on one base this time. My earlier numbers compared the branch
against the older dev it was built on. Current dev has moved a long way since then, especially in prefill.

**Trees.**
- I merged dev c1c06ac6 into this branch's head ca55947c.
- I added one line to `dsv4.def`, explained under "Things I ran into" below.
- The result is [`ecc92586`](https://github.com/bobsummerwill/colibri/commits/ecc92586) on my fork. "dev" below is
  c1c06ac6 itself.
- Both sides use the same build line and the same serve line (`COLI_CUDA_ATTN_BATCH=1 COLI_CUDA_MOE_BATCH=1` on the
  3090s), with `--ram 40 --ctx 8192`.
- Each sample starts a fresh server. Neither side sets a VRAM reserve: dev's reserve now accounts for the MoE bank, and
  no run logged an out-of-memory line.
- Every answer is byte-identical between dev and the branch.

### 2x RTX 3090 24 GB (NVLink), Ryzen 9 9950X3D, 62 GB, CUDA 12.0, `CUDA_ARCH=sm_86 NO_TC=1`

**REAP-150B, 6 prompts x 300 tokens, 2 samples per row** (decode = median of the per-sample mean):

| | GPUs | decode tok/s | vs dev | median TTFT |
|---|---|---|---|---|
| dev | 1 | 3.010 | 1.00x | 3.63 s |
| this branch | 1 | 3.784 | **1.26x** | 3.47 s |
| this branch, `DSV4_CUDA_DEVICES=0,1` | 2 | 4.818 | **1.60x** | 3.50 s |

A second session reproduced every row to within 0.3%.

**Full DeepSeek V4 Flash, 3 open-ended prompts x 300 tokens, one sample per row:**

| | GPUs | decode tok/s | vs dev |
|---|---|---|---|
| dev | 1 | 2.646 | 1.00x |
| this branch | 1 | 3.221 | **1.22x** |
| this branch, 2 GPUs | 2 | 3.697 | **1.40x** |

**Where the one-GPU gain comes from.** On REAP, turning off one of the branch's switches at a time, one sample each:

| switched off | decode tok/s | vs all on (3.790) |
|---|---|---|
| `DSV4_CUDA_HEAD` | 3.591 | 0.95x |
| `DSV4_CUDA_RESIDENT_INDEXER` | 3.600 | 0.95x |
| `DSV4_CUDA_INDEXER_HEADS` | 3.707 | 0.98x |
| `DSV4_CUDA_INDEXER_PREP` | 3.763 | 0.99x |
| `DSV4_CUDA_WO_DECODE` | 3.783 | 1.00x |
| `DSV4_CUDA_COMPRESSOR` | 3.820 | 1.01x |
| all of the above | 3.339 | 0.88x |

With every switch off, the branch still decodes at 1.11x dev. That part comes from code that has no switch.

**First token is slower than dev on the full model with the MoE bank.**
- With long prompts (1,992 and 3,005 tokens), the branch takes 98 s and 141 s to the first token, against dev's 86 s and
  124 s, about 1.14x as long. Decode on those prompts is 1.17x dev.
- No switch changes this. With each switch off, and with all of them off, the times stay at 98-100 s and 140-142 s.
- `DSV4_ATTN_PROF=1 V4_MOE_BATCH_PROF=1`, summed over both prompts:

  | prefill stage | dev | this branch |
  |---|---|---|
  | attention block | 79.6 s | 65.2 s |
  | MoE block | 123.3 s | 165.8 s |
  | of which batched expert compute | 72.8 s | 109.4 s (1.50x) |
  | of which bank refill | 48.3 s | 54.3 s |
  | missing experts | 15,638 | 15,638 |

- So the branch speeds up prefill attention by 14 s, but dev's batched expert compute takes 1.5x as long.
- TRACE: <which change does it, once traced>
- On the 5070 below, which runs without `COLI_CUDA_MOE_BATCH`, the branch's first token is as fast as dev's (0.98x).
- On REAP's short prompts, the branch is faster to the first token than dev.

### 1x RTX 5070 12 GB, CUDA 12.8, `CUDA_ARCH=sm_120`, default settings

| full V4 Flash | decode tok/s | vs dev |
|---|---|---|
| dev, 6 prompts x 300 tokens | 1.54 | 1.00x |
| this branch | 1.81 / 1.86 (2 samples) | **1.18x** |

- The answers are identical, and no run logged an out-of-memory line.
- On the long prompts, first-token time is 0.98x and 0.99x dev's, and decode is 1.15x and 1.17x.
- With no GPU (CPU only), the branch and dev both decode at 0.91-0.94 tok/s with identical answers.

### Things I ran into

1. **`dsv4_cuda_wo_decode` is missing from `dsv4.def`.** It is declared in `backend_cuda_dsv4.h` and the loader resolves
   it, but the MSVC DLL won't export it. On Windows the engine would then run without WO decode and give no error.
   dev's `tests/test_backend_loader_dsv4_parity.py` fails on it once the branch is on current dev. The fix is one line
   (`819cd03d` on my fork).
2. **`dsv4_cuda_init` leaves the CUDA error set when it rejects a device.** The up-front check
   `if(!ok(cudaGetDeviceProperties(&p,dev[i]),"validate device"))return 0;` returns on an invalid ordinal while
   `cudaErrorInvalidDevice` is still pending.
   - A caller that then retries with a valid device list succeeds, but its next checked CUDA call fails on the old error.
   - The rejection also doesn't say which device failed.
   - Calling `cudaGetLastError()` before the return fixes it. Here is what I used:

     ```c
     if(!ok(cudaGetDeviceProperties(&p,dev[i]),"validate device")){
         if(n>1)fprintf(stderr,"[DSV4 CUDA] device %d did not initialise\n",dev[i]);
         cudaGetLastError();return 0;}
     ```
3. **The new CUDA unit tests need device 5.**
   - These tests hard-code `int devices[]={0,5}`, so on a machine with fewer than six GPUs `assert(dsv4_cuda_init(devices,2))`
     aborts: `test_dsv4_compressor_cuda`, `_compressor_state_cuda`, `_head_cuda`, `_indexer_prepare_cuda`,
     `_wo_decode_cuda`, `_resident_indexer`, `_resident_route`.
   - Reading the device list from the command line, as `test_dsv4_multigpu_cuda` does, would let them run on 1- and
     2-GPU machines.
   - `test_dsv4_wo_decode_cuda` also picks device 5 at line 46 (`device == 0 ? 5 : 0`).
   - With both changed to devices 0 and 1, `_head_cuda`, `_wo_decode_cuda`, `_compressor_cuda` and
     `_compressor_state_cuda` all pass on the two 3090s, and every result is exact.
   - `_resident_indexer` and `_resident_route` have no make target, so I didn't run them.
4. **`test_dsv4_indexer_prepare_cuda` doesn't build with g++ 13.3.** `ggml_quants.h:134` uses C designated initializers,
   which g++ rejects in C++ code.

`test_dsv4_multigpu_cuda 0,1` passes on the 3090s.

### Optional (Bob to decide): this branch with #2405's peer tier on top

I also merged #2405's peer tier onto this branch ([`dec72809`](https://github.com/bobsummerwill/colibri/commits/dec72809)
on my fork, including fix 2 above). The two parts add up:

| | REAP decode, 2 GPUs | full model decode, 2 GPUs |
|---|---|---|
| this branch | 4.818 (1.60x dev) | 3.697 (1.40x dev) |
| this branch + #2405's peer tier | 6.07 (**2.01x** dev) | 4.288 (**1.62x** dev) |

That suggests landing this branch first and #2405's peer tier on top.

Logs, scripts and per-request numbers:
- 3090s: [final27-final30](https://github.com/bobsummerwill/colibri-notes/tree/d00475b/manifests/shadowfax-20261026-samebase5)
  (same-base runs, switch ablation, first-token arms with profiles, 300-token full-model arms)
- 5070 and CPU: [c4m27-c4m29](https://github.com/bobsummerwill/colibri-notes/tree/d00475b/manifests/computer4-20261026-samebase5)

---

## #2405 "Related" section (our own draft PR): redraft, NOT APPLIED

Replaces the on-hold base-mixed paragraph. Apply to #2405's body only with Bob's OK. #2401 is merged, so it is not
mentioned.

> **Related**
> - #1766 (multi-GPU residency and faster CUDA execution) overlaps with this PR's second-GPU work. I measured both on
>   the same dev (c1c06ac6), 2x RTX 3090:
>   - On their own, #1766 is faster. REAP decode: #1766 1.26x dev on one GPU and 1.60x on two. This PR is 1.03x on one
>     GPU and 1.44x on two.
>   - The two are complementary. This PR's peer tier on top of #1766 gives **2.01x** dev on REAP and **1.62x** on the
>     full model, 300-token answers. That is the best result measured, 1.26x and 1.16x over #1766 alone.
>   - What a second GPU adds: this PR's peer gives 1.40x over its one-GPU run, and #1766's second device 1.27x.
>   - The merge of the two is on my fork as `dec72809`. It needs one fix, in #1766's device check, which leaves a CUDA
>     error set; it is in the #1766 comment.
>   - Logs: [same-base runs](https://github.com/bobsummerwill/colibri-notes/tree/d00475b/manifests/shadowfax-20261026-samebase5)
> - Of this PR's four commits, only VRAM-first changes one-GPU speed: all of the 1.03x. The guard and settings commits
>   measure at 1.000x and 1.002x; the peer commit adds the second GPU.
