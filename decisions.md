# Decisions

> **Naming key.** These notes use internal names for the upstream pull requests. **PR #0** is the six
> fixes opened as JustVugg/colibri [#2212](https://github.com/JustVugg/colibri/pull/2212) (0a), [#2213](https://github.com/JustVugg/colibri/pull/2213) (0b), [#2214](https://github.com/JustVugg/colibri/pull/2214) (0c), [#2215](https://github.com/JustVugg/colibri/pull/2215) (0d),
> [#2216](https://github.com/JustVugg/colibri/pull/2216) (0e) and [#2217](https://github.com/JustVugg/colibri/pull/2217) (0f). **PR #1** (batched GPU prefill for expert-pruned checkpoints) is
> [#2264](https://github.com/JustVugg/colibri/pull/2264), opened 2026-10-20; **PR #2** (two-GPU decode) is not opened yet. Documents dated before 2026-10-17 describe earlier
> versions of these changes, on the old base 7b55a9b3.

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

## final6 results and the follow-ups (2026-10-23)
- **`DSV4_MM_WARP` stays on by default**: the plan's rule (bit-identical and not slower in the default configuration on
  the full model) is met on both builds (generic CUDA on shadowfax, DeepGEMM on computer4).
- **The failed gate is not read as a PR #1 failure**, but each item gets its own run before anything is claimed
  upstream (final7, [TEST_PLAN_20261023.md](TEST_PLAN_20261023.md)): the bank ratio from the 1-sample AK arm is quoted
  for prompt 1 only (0.665x), where dev's bank was on; the like-for-like number is BK vs dev with
  `DSV4_CUDA_VRAM_RESERVE_MB=4681` (AR, final7), dev's own documented setting for the bank.
- **The unused-variable fix (b286167e) is a separate commit on #2264**: upstream code in the DeepGEMM kernels that the
  first DeepGEMM build here warned about (Bob asked, 2026-10-22). Generic builds do not compile those functions, so the
  generic-CUDA final6 numbers stand. Pushed after `make check` at both tips.
- **Vulkan runs get a memory cap** (`MemoryMax` on the server's unit) from final7 on: QB's host OOM was global and
  could have taken another process on shadowfax.
- **No reply on #2264 until the follow-ups are in**; a draft goes in `review/` and Bob posts it or OKs it.
- **zebra not used**: no Metal path in the V4 engine and the PRs are CUDA-only; upstream CI covers macOS and ARM.

## #2264 review: dev merge, full V4 Flash measurement (2026-10-22)
- **JustVugg's review** ([comment](https://github.com/JustVugg/colibri/pull/2264#issuecomment-6335156265)) kept PR #1 out
  of the day's batch (#2284) for three points: merge dev; measure the default configuration on the full V4 Flash because
  `DSV4_MM_WARP` (auto) is on for every FP8 checkpoint's `mm_batch`, not only under `COLI_CUDA_MOE_BATCH=1`; and say why
  `test_v4_moe_batch_speculative_source.py` is deleted.
- **Merged, not rebased** (as asked, and no force-push): upstream dev 7df8c27c into `docs5/pr1-reap-prefill` (ebe82123).
  One conflict, the one JustVugg named: dev's `g_v4_gpu_open_device = device;` and our `v4_gpu_mirror_reserve_size(...)`,
  both kept. Then ebe82123 into `docs5/pr2-two-gpu-decode` (073f18d2), two more conflicts, both from dev's new CUDA
  routed-expert tier (`cu_tier.c`, 9327e66e): the MoE cleanup in `moe_token_pipeline` (our lease-aware release and decode
  timing under dev's widened `#if COLI_VULKAN || COLI_CUDA`) and `deepseek_v4_internal.h` (our VRAM-first declarations,
  then dev's tier table). The two tiers never meet at run time: `v4_cu_open` leaves the experts to the FP4 checkpoint's own
  CUDA tier unless the model is a GGUF, and VRAM-first decode is on that FP4 tier only.
- **The deleted test:** its checks moved unchanged into `test_v4_moe_batch_lifecycle_source.py` (commit f672fd73 says so);
  one line added to the PR description.
- **Measure rather than turn `DSV4_MM_WARP` off by default** (Bob's choice). The kernel is compiled only into generic
  builds (`#ifndef COLI_DSV4_DEEPGEMM`) and auto takes only fp8 shapes with I a multiple of 256 and <= 1024 (e.g.
  `attn.wq_b`), so the default configuration exercises it through the batched attention (`COLI_CUDA_ATTN_BATCH=1`).
- **Full V4 Flash** (`deepseek-ai/DeepSeek-V4-Flash-0731`, 48 shards, 167 GB) downloaded to shadowfax
  `~/colibri-models/deepseek-v4-flash` with `hf` from its own venv (`~/colibri-bench/hf-venv`, made with uv; Hermes's venvs
  untouched). Never run here before.
- **One long GPU window instead of a short one** (Bob, 2026-10-22: Hermes can be down from the evening through the
  night). Besides the requested A/B, `final6` takes every open item this box can test: the 256-expert bank (F7/F8/F12,
  never run), speculative decoding with the bank on (F2), two requests at once, an ~8k-token prompt, PR #2's decode arms
  re-measured after the dev merge, and PR #2 paths never run end to end (`DSV4_CUDA_PEER_COMPUTE=0` = F45,
  `DSV4_HYBRID=1`, `DSV4_CUDA_BATCHED=0`). Left out: F18 fault injection (needs a debug build first), DeepGEMM (needs
  CUDA >= 12.8 and an sm_90+ card: computer4's RTX 5070 after tasks-personal #2822), Vulkan (no `glslc` on shadowfax),
  Windows (no machine; upstream CI builds it), mixed-GPU P2P (no such pair).
- **Harness `final6`** ([tools/bench/final6/](tools/bench/final6/)): final5's runner and helpers with a per-arm model,
  context and client; the base environment is final5's minus `COLI_CUDA_MOE_BATCH=1` (the canonical serve line without
  it), and arms add it back where they need it. Arm table and premises in `final6_tools.py`; texts are compared within
  groups (full model / ~8k prompt / REAP decode) against each group's first sample, as the full model has no older
  reference. `bench_f6.py` adds the ~8k prompt and the two-at-once requests. Only A/B/V failures can abort the run; any
  other arm that breaks a premise is skipped. Trees and builds under `~/colibri-bench/cw22` (persistent).
  **Pilot first** (Bob): `PILOT=1` runs one default and one bank sample on the full model, then `dsv4-cuda-test` at both
  merged tips and `cuda-test` at PR #2. Launched 16:51 (llama stop 17:30).
- **CUDA builds at the merged tips** (sm_86, NO_TC=1): dev, PR #1 and PR #2 build with 0 warnings
  (`~/colibri-bench/cw22/build-*.log`).

## REAP-150B with thinking on, through the Hermes bake-off (2026-10-21)
- Run on colibri bb08dc4b (= PR #1 + PR #2, VRAM-first + second GPU, `--ctx 32768`) via `bakeoff.py` endpoint mode
  with `enable_thinking` and `reasoning_effort=high` (colibri has no hard reasoning cap; "high" is DeepSeek's soft
  ~1,500-token prompt), token limits ×3. Results (openclaw-team af28e3d8, `20261021-think-comparison.md`):
  tool calls 11/12, agentic 5/7; core_v1 fails because colibri's deepseek_v4 engine rejects `response_format`
  grammars; coding invalid (19/23 hit the harness's 600 s request timeout; the 4 that finished passed);
  stopped before agentic_v2 at Bob's request.
- Against the Qwen3.8-27B incumbent with a 4K thinking cap (tool calls 11/12, coding 21/23, agentic 7/7, 6/6;
  ~34 min) and the full DeepSeek-V4-Flash on llama.cpp RPC with the same cap (12/12, 19/23, 6/7, 6/6; ~2 h 12 min),
  REAP is weaker and 15–20× slower than Qwen. **Verdict (Bob): not worth pursuing, no coding rerun.** The PRs stand
  on their own as engine contributions; REAP is not a model for this setup.
- Possible upstream follow-up (not done): `response_format` grammar support in the deepseek_v4 engine.

## PR #1 opened (2026-10-20)
- **Opened as [JustVugg/colibri#2264](https://github.com/JustVugg/colibri/pull/2264)** at Bob's request, from
  `docs5/pr1-reap-prefill` (833c6762) into `dev`, body = the PR #1 text of review/PR_DESCRIPTIONS_20261020.md on
  upstream's template. Upstream dev had moved 32 commits overnight (to bf2962f2, touching deepseek_v4.c and
  Makefile.deepseek-v4); PR #1 merges cleanly into it, and on that merge `make -C c check` (2612 OK) and the CPU and
  CUDA sm_86 engine builds (0 diagnostics) pass, as the PR body says. Not rebased (the measured base stays 736a0682).
  PR #2 not opened.

## final5 benchmark and docs refresh (2026-10-20)
- **Overnight GPU run `final5`** (shadowfax, 00:15–03:00, Hermes's model server down and restored by the EXIT trap): the
  final4 harness pointed at dev 736a0682 and the refresh tips, plus two decode arms (D0 = dev, R1 = VRAM-first only)
  and 3 samples for every decode arm. Fixes needed for the Tabby production server: the GPU-exclusivity check
  accepts every process in llama-server-active's cgroup, and the health check accepts `"healthy"`. Every gate
  passed; all 92 outputs identical to dev's.
- **Results** (manifests/final5, six manifests validated): PR #1 TTFT 1.70× / 1.86× (2.03× / 2.10× with the double
  bank), as on 62eaa9dc; PR #2 VRAM-first +3.0 %, second GPU +39.5 %, +43.7 % overall: the first PR #2 measurement with
  3 samples per arm and one switch per manifest.
- **Explained, not hidden:** PR #2 with every switch off decodes 0.9 % below dev, and decode after a bank prefill is
  1.6–3.2 % behind. With `COLI_CUDA_MOE_BATCH=1` the bank's share of the VRAM reserve (3100 MiB) leaves room for fewer
  decode mirrors (GPU0 peak 21,192 vs 22,008 MiB). Stated in the PR texts and docs, with the workarounds.
- **Per-commit check** on the new base: all 16 commits build and compile with `-Wall -Wextra`, 0 diagnostics
  (manifests/final5/percommit.txt).
- **Docs refresh as new branches** `docs5/pr1-reap-prefill` / `docs5/pr2-two-gpu-decode`: only the two docs commits
  changed (tables and notes to the final5 numbers, PR #2's docs commit message to match); the code equals the
  benchmarked trees; `make check` 2619 OK at the PR #2 tip. New names, so no force-push: these are the branches to
  open the PRs from. PR texts: review/PR_DESCRIPTIONS_20261020.md.

## Refresh onto upstream dev 736a0682 (2026-10-19)
- **All six PR #0 PRs (#2212–#2217) were merged upstream.** `git cherry` confirms all 13 PR #0 commits are on upstream dev, so PR #1 now sits directly on dev.
- **Fork dev fast-forwarded** from 7b55a9b3 to upstream dev 736a0682 (574 commits, no force-push).
- **PR #1 and PR #2 rebased onto 736a0682** (Bob asked): no conflicts. `git range-diff` shows all 8 PR #1 patches identical; 7 of 8 PR #2 patches identical, and the 8th ("peer mirror pool") differs only in one context line (upstream's new GGUF-to-Vulkan hand-off sits above our peer init, which is unchanged and still correct). Pushed as `refresh/pr1-reap-prefill` and `refresh/pr2-two-gpu-decode` (new names; the `rebased/*` branches are untouched, which avoided a force-push).
- **Tests at the new tips:** `make -C c check` 2612 OK (PR #1) and 2619 OK (PR #2), 284 skipped. On shadowfax, 2× RTX 3090 + NVLink, CUDA 12.0, with `NVCCFLAGS=... -DCOLI_DSV4_NO_TC` (needed: CUDA 12.0 lacks the 12.8 cuBLASLt block-scaling API): `dsv4-cuda-test` passes at both tips with 0 warnings, including the peer tests (bit-identical peer compute, fault latch). `cuda-test` passes at PR #2. Its 36 warnings are all in upstream files these PRs do not touch (OpenMP pragmas in quant.h/idot.h, unused parameters in backend_cuda.cu).
- **Not yet redone:** the performance measurements. The PR descriptions still cite 62eaa9dc numbers, and upstream has 574 more commits, including CUDA perf work (#2255).

## Rebase onto upstream dev 62eaa9dc (2026-10-17)
- **Rebased all eight branches onto 62eaa9dc** (Bob asked for the branches to be refreshed and prepped). PR #0 cherry-picked cleanly (identical patches); PR #1 had one conflict with #2109 (serve prewarm: hot_lookup counts nothing while warming — kept, with our hot_count_locked inside it); PR #2 one with upstream's GGUF format guard in the mirror attach (kept, with V4_MIRROR_RECYCLE_MIN). pr2 was restacked onto pr1/v7 (v7 = v6 + the F42 docs numbers).
- **Review fixes folded into their commits**: no VRAM reserve for a bank a GGUF can never create (`v4_gpu_mirror_reserve_size(.., bank_possible)`); upstream's new `keep_free_bytes` in four test initializers; the colibri session's open PR #2 items (note_vram source assertion, hit[e] check, pin-ranking docs clause); a GGUF note in the two-GPU docs; measurements labelled with their base.
- **Left out on purpose**: review item #6 (format check in the bank-upload paths) — that is upstream's own code (9876c4db, 66fca1e7), not ours; a possible separate follow-up.
- **Benchmarks re-done on the new base** (final4) and the docs/PR text use those numbers; the old 2.7-3.1x is history.
- **Fork**: rebased branches pushed as `rebased/<name>`; overwriting the old names (force-push) was not done — it needs Bob's explicit OK.

