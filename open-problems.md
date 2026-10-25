# Open problems / next steps

> **Naming key.** These notes use internal names for the upstream pull requests. **PR #0** is the six
> fixes opened as JustVugg/colibri [#2212](https://github.com/JustVugg/colibri/pull/2212) (0a), [#2213](https://github.com/JustVugg/colibri/pull/2213) (0b), [#2214](https://github.com/JustVugg/colibri/pull/2214) (0c), [#2215](https://github.com/JustVugg/colibri/pull/2215) (0d),
> [#2216](https://github.com/JustVugg/colibri/pull/2216) (0e) and [#2217](https://github.com/JustVugg/colibri/pull/2217) (0f). **PR #1** (batched GPU prefill for expert-pruned checkpoints) is
> [#2264](https://github.com/JustVugg/colibri/pull/2264), opened 2026-10-20, **merged 2026-10-24** in JustVugg's batch [#2360](https://github.com/JustVugg/colibri/pull/2360) (merge commit 60a73ec; dev 561966e).
> **PR #2** (two-GPU decode) is not opened yet. Documents dated before 2026-10-17 describe earlier
> versions of these changes, on the old base 7b55a9b3.

## Open now (2026-10-24 evening, after #2264 merged)
shadowfax and computer4 are unreachable; counter is the only build/test machine. Each item notes whether it needs a GPU.

> **Progress (2026-10-24 night, counter; nothing pushed):** items 1, 2, 4, 5 and 7 are done locally, see the README's
> night status. Left: **push** the three branches (counter needs `gh auth login`), item 1's CUDA/DeepGEMM builds
> (counter has no nvcc), item 3 (shadowfax), item 6's branch retirement (Bob's go-ahead), item 8. Then Bob decides on
> opening PR #2 and fixes A and B.

1. **PR #2: bring onto dev 561966e.** *(counter, no GPU)* **Done locally** (090d8160; triage: no V4 GPU code changed; `make check` and per-commit OK; CUDA builds not possible: no nvcc).
   - Merge dev 561966e into `docs5/pr2-two-gpu-decode`; push to the fork only.
   - **Triage first:** `git diff --stat 61646cb..561966e` and `git log` on `c/deepseek_v4*.c`, `c/backend_cuda_dsv4.cu`, `cu_tier.c`, the Vulkan backend and the Makefile's CUDA targets. Batch #2360 is mostly API/schema/OLMoE/Qwen3.6; check whether #2332 / #2359 (Qwen3.6 CUDA chain) touch shared CUDA or Makefile code. *(verify)*
   - If the V4/dsv4 GPU code is untouched, final11/13 (texts = dev's, X3 1.437x dev's decode, F18 fault → later requests complete on the host) still describe the merged tree. In that case PR #2 can be opened now, with the GPU results stated as measured on 14d67f05 + F18 fix.
   - `make -C c check`, CPU-only build, engine units under `-DCOLI_V4_GPU_TIER` (and with `-DCOLI_VULKAN`), per-commit `-Wall -Wextra`, dsv4 export parity test, Python source tests (incl. the F18 latch's).
   - CUDA engine build (generic sm_86, NO_TC) and DeepGEMM sm_120a compile: only if counter has nvcc (12.8+ for DeepGEMM).
2. **PR #2 description.** *(no machine)* **Drafted:** [review/PR2_DESCRIPTION_20261024_DRAFT.md](review/PR2_DESCRIPTION_20261024_DRAFT.md). Replace the final5 numbers with final6 / final11 / final13. Describe the F18 fix (a sticky peer fault now falls back to the host instead of failing every later request) and correct the earlier "decode continues on device 0" claim. Say that #2264 is merged and this PR now carries only its own commits. Draft first; Bob opens.
3. **PR #2 GPU re-check on the merged tip.** *(shadowfax, 2x GPU; blocked)* `make cuda-test`, `dsv4-cuda-test`, the F18 injection (`DSV4_CUDA_PEER_FAULT=n:sticky`) in a server, a short X3 sanity A/B. Only required before opening if item 1's triage finds GPU code changed on dev.
4. **Fix A vs PR #2.** *(counter, no GPU)* **Done:** PR #2 already contains fix A; trial merge c89353e4 `make check` OK; fix A first. Both touch the decode-mirror growth guard / VRAM accounting. Trial-merge `fix/dsv4-mirror-guard-overshoot` with PR #2 in a scratch branch (`git merge-tree` first). Run `make check` on the result. Decide the order and mention the other in each description.
5. **Fixes A and B onto dev 561966e.** *(counter, no GPU)* **Done locally** (f30c7b23 with a conflict resolved, 97b26c0a clean; `make check` OK); descriptions drafted: [review/UPSTREAM_FIX_PRS_20261024_DRAFT.md](review/UPSTREAM_FIX_PRS_20261024_DRAFT.md). Both are on 61646cbd; rebase or merge. Run `make check` and their source tests. Write PR descriptions from [review/UPSTREAM_FIXES_20261024.md](review/UPSTREAM_FIXES_20261024.md) (evidence on 61646cbd stands; no rerun needed to open). JustVugg did not reply to the offer on #2264; whether to open issues or PRs is Bob's decision. Fix B is independent of both PRs.
6. **Notes housekeeping.** *(no machine)* This update. Retire history branches on the fork (Bob's go-ahead).
7. **Superseded / low priority.** *(The kv comp issue text is folded into fix A's draft.)* The kv comp fallback issue draft (`review/UPSTREAM_ISSUES_20261023_DRAFT.md`) is superseded by fix A; fold its text into fix A's description. The `COLI_VK_UP_IMPORT` note and pure-Vulkan text variance are already reported on #2264; an issue only if JustVugg wants one. Fix B's Vulkan memory measurements on the new base need shadowfax *(blocked, optional)*.
8. **tasks-personal #2824** (driver alignment, needs Bob's sudo) unchanged; not colibri work, and both machines are offline anyway.

<details><summary>Open on 2026-10-24 morning (superseded)</summary>

## Open now (2026-10-24 morning, after the overnight run)
1. **#2264:** follow-up posted 2026-10-24 ([comment](https://github.com/JustVugg/colibri/pull/2264#issuecomment-6387132721): the Vulkan + CUDA correction, fixes A and B offered, imports / pure-Vulkan variance notes; [text](review/REPLY_2264_FOLLOWUP_20261024_v2.md)). Waiting on JustVugg.
2. **Fixes A and B** are on fork branches off dev with measurements ([review/UPSTREAM_FIXES_20261024.md](review/UPSTREAM_FIXES_20261024.md)).
   Whether to open issues / PRs is Bob's decision.
3. **PR #2:** F18 is fixed and the branch retested; open it upstream when #2264 lands (or as Bob decides).
4. **`COLI_VK_UP_IMPORT`:** imports are slower than staged uploads on this NVIDIA card; the default is device dependent,
   so at most a note upstream (final16, on top of fix B: staged 0.980x TTFT, +2.6% decode).
5. Pure-Vulkan greedy texts vary run to run on dev itself (the tier's device/CPU split follows measured upload rates).
   Known, not a bug of these branches; it means pure-Vulkan text identity can't be a gate.
6. tasks-personal #2824 (driver alignment, needs Bob's sudo) unchanged.
</details>

<details><summary>Open on 2026-10-23 evening (superseded)</summary>

## Open now (2026-10-23, after final7 + final8)
1. **#2264:** every PR #1-vs-dev question this hardware can ask is answered (final6-8: default config identical and
   0.98-0.99x TTFT; bank 0.66x like for like on the full model, 0.54-0.60x on REAP; Vulkan and Vulkan + CUDA identical
   to dev given the same pool / reserve). Reply with final6-8 and the dev 61646cbd merge posted 2026-10-23 ([comment](https://github.com/JustVugg/colibri/pull/2264#issuecomment-6380394089), the complete version; the first, shorter one was deleted) and the PR description updated; waiting on JustVugg.
2. **F18 (PR #2):** in a real server a peer fault leaves every later request failing until restart (final7 F7), and
   PR #2's "decode continues on device 0" is false. Fix the message, then try the host-wait probe variant / latch CUDA off.
3. **Upstream issue drafts** (`review/UPSTREAM_ISSUES_20261023_DRAFT.md`, not filed, no fixes attempted): kv comp
   fallbacks changing default-config texts (still stands; re-check on dev tip 61646cbd, maybe prototype the fix);
   Vulkan host pool OOM possibly fixed upstream by 0f786bfa + 5c0469ae (#2318), re-run QO on the dev tip; the Vulkan
   timing draft is withdrawn (final8).
4. **tasks-personal #2824** (driver alignment): same driver 595.99.02 on both, but shadowfax still has the
   graphics-drivers PPA and two precompiled `linux-modules-nvidia-595-open-*` packages (needs Bob's sudo; not colibri work).
5. Manifests in `c/experiment_manifest.py` format for final7/8 not made (only QF has 3 samples in final8); evidence
   tarballs are in `manifests/final7/`, `manifests/final8/`.

</details>

<details><summary>Open on 2026-10-23 morning (superseded)</summary>

1. **#2264:** final6 answered JustVugg's question (bit-identical, 0.98-0.99x TTFT, decode equal; verify 0 mismatches;
   DeepGEMM the same). To do: manifests (`manifests/final6/`), and the comment on #2264 (draft first; Bob posts or OKs).
2. **Bank like-for-like:** dev loses its bank on a second prompt (its 2800 MiB default reserve is below the bank),
   so BK/AK's prompt-2 ratio is not like-for-like. final7 arm 1 (AR: dev with 4681 MiB).
3. **Vulkan:** QB was a host OOM kill (pool sized from free RAM at start, upstream `vk_tier.c`); Vulkan texts likely
   depend on run-time timings (CPU/GPU split). final7 arms 2-5; upstream issue drafts if confirmed.
4. **HY** differs from D0: probably dev's hybrid path. final7 arm 6.
5. **F18 (PR #2):** a real peer fault is caught (no hang, no wrong output), but CUDA is then lost for the whole process;
   PR #2's "decode continues on device 0" is wrong. final7 arm 7 (fault inside a real server), then a PR #2 fix.
6. **Handled `kv comp` allocation failures** in dev and PR #1 alike at 24 GB on the full model (252 a sample);
   computer4 B s2 had 421. Pre-existing; final7 step 1b re-checks computer4.
7. **tasks-personal #2824** (align NVIDIA drivers between shadowfax and computer4): both already report 595.99.02
   (final6's host.txt on each); check and close it after final7.

</details>

## Open on 2026-10-22 (superseded by the list above)
1. **#2264 (PR #1):** run the pilot, then the overnight `final6` A/B on the full V4 Flash, default configuration
   (dev vs the merged branch, prompt and decode, plus `DSV4_MM_WARP_VERIFY=1`), and post the result as a manifest and a
   comment. If PR #1 is slower there or not bit-identical, make `DSV4_MM_WARP` default to 0 (opt-in).
2. **PR #2:** not opened. It now carries dev's CUDA expert tier (`cu_tier.c`); the merge is clean and the tiers do not
   meet at run time, but its numbers (final5) are from before that merge. Its fate follows PR #1's `DSV4_MM_WARP` decision.
3. **tasks-personal #2824** (align NVIDIA drivers between shadowfax and computer4): do not change shadowfax's driver or
   CUDA between the pilot and `final6`, or the two arms stop being comparable.

## To finish before the PRs are submitted (2026-10-17, superseded: PR #0 merged, PR #1 opened as #2264)
1. Bob: review `review/PR_DESCRIPTIONS_20261017.md` and the `rebased/*` branches on the fork; decide whether to force-push them over the old names (or open the PRs from the `rebased/*` names).
2. Rebase again if upstream dev moves before submission (it moved 298 commits in 6 days): re-run `make check`, the CUDA build and, for code conflicts, `run_final4.sh`.
3. Attach `manifests/final4/` (4 validated manifests, `evidence/`, `bench.tgz`) to PR #1; PR #2's numbers are a one-sample sanity check (a manifest would need >=3 samples per arm).

## Untested / known gaps
- **F18:** fixed 2026-10-24 (final13: a sticky fault injected at the 3000th peer pass, all later requests complete on the host). A real `__trap()` was tested 2026-10-22 / final10 (device 0 is lost with the peer); the end-to-end server test uses the `DSV4_CUDA_PEER_FAULT=n:sticky` injection.
- Windows build (the loader and `dsv4.def` parity test passes on Linux only; upstream CI builds it, incl. CUDA/MSVC, and it passes on #2264). DeepGEMM was tested on computer4 (RTX 5070, final6) on one GPU only. The full 256-expert checkpoint was tested in final6.
- Apple Silicon (zebra): not tested and not planned for these PRs; the V4 engine has no Metal path and the PRs are CUDA-only (TEST_PLAN_20261023.md).
- Upstream test `test_dsv4_attention_batch_cuda` uses a stub that exits 4. This is pre-existing and not caused by this work.

## Performance ideas not pursued
- Decode is now bound by attention (~107 ms/token) and the remaining expert fetch/attach. Next steps would be attention kernels and smarter expert placement across the two GPUs.
- The PCIe 4.0 x8 links (~12 GB/s) cap expert refill. Pinned staging buffers are already used where possible.
