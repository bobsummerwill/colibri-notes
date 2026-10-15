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

## Cost of the work (to 2026-10-14)
- About 257M effective tokens. Effective tokens are weighted: cache reads 0.1x, cache writes 2x, output 5x. That is roughly 1.4B raw tokens.
- Subagents account for 87% of it: 298 agents in total, and the in-depth review alone used about 116M effective tokens.
- Script: `tools/analysis/usage_analysis.py`. Summary: `tools/analysis/usage_summary.json`.
