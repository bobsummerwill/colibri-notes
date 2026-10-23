# Open problems / next steps

> **Naming key.** These notes use internal names for the upstream pull requests. **PR #0** is the six
> fixes opened as JustVugg/colibri [#2212](https://github.com/JustVugg/colibri/pull/2212) (0a), [#2213](https://github.com/JustVugg/colibri/pull/2213) (0b), [#2214](https://github.com/JustVugg/colibri/pull/2214) (0c), [#2215](https://github.com/JustVugg/colibri/pull/2215) (0d),
> [#2216](https://github.com/JustVugg/colibri/pull/2216) (0e) and [#2217](https://github.com/JustVugg/colibri/pull/2217) (0f). **PR #1** (batched GPU prefill for expert-pruned checkpoints) is
> [#2264](https://github.com/JustVugg/colibri/pull/2264), opened 2026-10-20; **PR #2** (two-GPU decode) is not opened yet. Documents dated before 2026-10-17 describe earlier
> versions of these changes, on the old base 7b55a9b3.

## Open now (2026-10-23) — plan: [TEST_PLAN_20261023.md](TEST_PLAN_20261023.md)
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
7. **tasks-personal #2824** (align NVIDIA drivers between shadowfax and computer4): still hold until final7 is done.

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
- **F18:** tested 2026-10-22 with a real `__trap()` on the peer (final6): caught, no hang, no wrong output; CUDA is lost for the process afterwards (see Open now #5).
- Windows build (the loader and `dsv4.def` parity test passes on Linux only; upstream CI builds it, incl. CUDA/MSVC, and it passes on #2264). DeepGEMM was tested on computer4 (RTX 5070, final6) on one GPU only. The full 256-expert checkpoint was tested in final6.
- Apple Silicon (zebra): not tested and not planned for these PRs; the V4 engine has no Metal path and the PRs are CUDA-only (TEST_PLAN_20261023.md).
- Upstream test `test_dsv4_attention_batch_cuda` uses a stub that exits 4. This is pre-existing and not caused by this work.

## Performance ideas not pursued
- Decode is now bound by attention (~107 ms/token) and the remaining expert fetch/attach. Next steps would be attention kernels and smarter expert placement across the two GPUs.
- The PCIe 4.0 x8 links (~12 GB/s) cap expert refill. Pinned staging buffers are already used where possible.
