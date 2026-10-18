# Decisions

- **No upstream PRs or pushes.** Bob's instruction (2026-10-14): build the PRs so they are ready, but never submit them. Work goes to the fork only. The shadowfax clone's upstream push URL is deliberately disabled.
- **Three stacked PRs, not two.**
  - The review found 7 bugs that already exist on upstream dev, so they become a standalone PR #0 that maintainers can merge without taking any new features.
  - PR #1 (prefill) and PR #2 (two-GPU decode) stack on top.
  - The restructure must reproduce the validated trees exactly: the PR #1 tip == a406901b's tree, and the PR #2 tip == fe28d80a's tree.
- **Bit-identical output is the correctness bar.**
  - Every kernel change keeps the reference summation order and has a VERIFY mode that runs the old and new kernels side by side and counts mismatches.
  - This matches upstream's experiment-manifest correctness gate and avoids arguments about "close enough".
- **New behaviour is opt-in.**
  - Opt-in switches: two-GPU decode (`V4_GPU_RESIDENT_FIRST=1`, `DSV4_CUDA_PEER_DEVICE=1`) and the double bank (`COLI_CUDA_MOE_DOUBLE=1`).
  - The expert-major and warp dense kernels default on only where they were measured to win (warp dense only for I <= 1024).
- **Benchmarks follow upstream CONTRIBUTING:**
  - at least 3 cold samples per arm;
  - one changed variable per manifest;
  - hashed evidence;
  - full output text compared.
- **Hermes stays on llama.cpp.**
  - colibri decode is ~3.9 tok/s at best, against ~72 tok/s for Qwen3.8-27B on llama.cpp.
  - The work is worth having as an upstream contribution and a learning exercise, not as Hermes's model server.
  - Every GPU window stops llama and restores it in an EXIT trap.
- **Model download kept** (84.7 GB, expensive to re-fetch).
- **Nothing installed system-wide; Hermes config, cron jobs and Exocortex left untouched; port 8000 never bound** (the handoff constraints).

- **PR #0 split into six focused PRs (Bob, 2026-10-16).** Upstream has no single-commit or squash rule (CONTRIBUTING: "Keep changes focused"; template: "the smallest change that solves it"; merges are merge commits, and about half of external PRs have several commits), but one PR per problem fits "focused". The 13 PR #0 commits are almost all independent (10 apply and revert alone on dev; the only chains are textual: the CUDA-error fix after the upload-ordering fix, the shutdown fix after that). Split: 0a upload ordering + CUDA error hygiene + shutdown zeroing (4), 0b prefill bank lifecycle (3), 0c decode error paths (2), 0d DLL export + parity test (1), 0e CUDA test target (2), 0f docs sizes (1); each on dev with unchanged patches. The combined linear branch stays as PR #1's base; PR #1/#2 descriptions say they depend on the six.
- **F42 fix added to PR #1** (commit "GPU-bank prefill keeps the expert store's routing books"): measured +6.3% decode after a long GPU-bank prefill, recovering 67-79% of the gap with no prefill cost or output change; the remaining ~2% (residency in the first ~5 tokens) is left as possible follow-up because closing it needs extra reads during prefill.
- **Final GPU benchmark scheduled for 2026-10-17 01:00 PDT** (Bob's choice of downtime window), as a one-off systemd timer on shadowfax.

## Cost of the work (to 2026-10-14)
- About 257M effective tokens. Effective tokens are weighted: cache reads 0.1x, cache writes 2x, output 5x. That is roughly 1.4B raw tokens.
- Subagents account for 87% of it: 298 agents in total, and the in-depth review alone used about 116M effective tokens.
- Script: `tools/analysis/usage_analysis.py`. Summary: `tools/analysis/usage_summary.json`.

## Rebase onto upstream dev 62eaa9dc (2026-10-17)
- **Rebased all eight branches onto 62eaa9dc** (Bob asked for the branches to be refreshed and prepped). PR #0 cherry-picked cleanly (identical patches); PR #1 had one conflict with #2109 (serve prewarm: hot_lookup counts nothing while warming — kept, with our hot_count_locked inside it); PR #2 one with upstream's GGUF format guard in the mirror attach (kept, with V4_MIRROR_RECYCLE_MIN). pr2 was restacked onto pr1/v7 (v7 = v6 + the F42 docs numbers).
- **Review fixes folded into their commits**: no VRAM reserve for a bank a GGUF can never create (`v4_gpu_mirror_reserve_size(.., bank_possible)`); upstream's new `keep_free_bytes` in four test initializers; the colibri session's open PR #2 items (note_vram source assertion, hit[e] check, pin-ranking docs clause); a GGUF note in the two-GPU docs; measurements labelled with their base.
- **Left out on purpose**: review item #6 (format check in the bank-upload paths) — that is upstream's own code (9876c4db, 66fca1e7), not ours; a possible separate follow-up.
- **Benchmarks re-done on the new base** (final4) and the docs/PR text use those numbers; the old 2.7-3.1x is history.
- **Fork**: rebased branches pushed as `rebased/<name>`; overwriting the old names (force-push) was not done — it needs Bob's explicit OK.

