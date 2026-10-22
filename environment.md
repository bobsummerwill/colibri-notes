# Environment

> **Naming key.** These notes use internal names for the upstream pull requests. **PR #0** is the six
> fixes opened as JustVugg/colibri [#2212](https://github.com/JustVugg/colibri/pull/2212) (0a), [#2213](https://github.com/JustVugg/colibri/pull/2213) (0b), [#2214](https://github.com/JustVugg/colibri/pull/2214) (0c), [#2215](https://github.com/JustVugg/colibri/pull/2215) (0d),
> [#2216](https://github.com/JustVugg/colibri/pull/2216) (0e) and [#2217](https://github.com/JustVugg/colibri/pull/2217) (0f). **PR #1** (batched GPU prefill for expert-pruned checkpoints) is
> [#2264](https://github.com/JustVugg/colibri/pull/2264), opened 2026-10-20; **PR #2** (two-GPU decode) is not opened yet. Documents dated before 2026-10-17 describe earlier
> versions of these changes, on the old base 7b55a9b3.

Two machines: **computer4** runs Claude Code, holds the git clones (`~/Projects/colibri`, `~/Projects/colibri-claude`), pushes to GitHub and does CPU builds / `make check`. **shadowfax** is the GPU testground only (CUDA builds, GPU tests, benchmarks), reached with `ssh maplemercury@shadowfax`. Paths below are on the machine whose section they are in.

## shadowfax (GPU box)
- AMD Ryzen 9 9950X3D (16C/32T), 62 GB DDR5, WD_BLACK SN850X 4 TB (ext4, O_DIRECT reads ~6.5 GB/s), Ubuntu 24.04.4, kernel 7.0.0-34, gcc 13.3, CUDA toolkit 12.0.140 (apt), driver 595.84.
- 2x RTX 3090 24 GB, **each on PCIe 4.0 x8** (host->GPU uploads measured ~12 GB/s), **joined by an NVLink bridge** (`nvidia-smi topo -m` shows NV4, P2P OK, ~56 GB/s). GeForce cards normally lack P2P; the bridge is why the peer pool works.
- Reach: `ssh maplemercury@shadowfax` from computer4 (Tailscale SSH; user `zombietiger` is refused). Claude Code allow rule `Bash(ssh maplemercury@shadowfax:*)` exists in computer4 `~/.claude/settings.json`.
- Services (do not break them):
  - `llama-server-active.service` (**user** unit): llama.cpp serving Qwen3.8-27B (Q4_K_M + MTP draft) on port 18080, both GPUs (~19.5 + 22.7 GB). It IS the Hermes agent's brain. `Restart=on-failure` only — a clean stop is not auto-restarted.
  - `hermes-gateway.service` (**system** unit) — survives llama restarts.
  - `tailnet_http_proxy.py` on 127.0.0.1:8080 -> 18080; Exocortex static servers on 8000/8001 — never bind those ports. colibri tests use 8090.
- Safe GPU-downtime pattern: a runner script run as its own transient user unit (`systemd-run --user --unit=... --collect`) that stops llama (`systemctl --user stop llama-server-active`), runs, and **restores it in an EXIT trap** (`systemctl --user start llama-server-active`, poll `curl 127.0.0.1:18080/health`). See `tools/bench/run_x.sh`, `run_pr1_bench.sh`.
- Persistent work dir (on shadowfax): `~/colibri-bench` (raw logs, evidence, scripts). Model: `~/colibri-models/deepseek-v4-flash-reap` (do not delete). `/tmp/colibri*` and `/tmp/cw/*` (clones/worktrees/builds) vanish on reboot.
- The shadowfax clone `/tmp/colibri` has its upstream push URL set to `DISABLED-no-upstream-pushes` on purpose.

## computer4 (where Claude Code runs; every `~/Projects/...` path is here)
- Has Bob's git identity (Bob Summerwill <bob@summerwill.net>) and `gh` logged in as bobsummerwill (HTTPS; the ssh key is not registered with GitHub). No nvcc — CUDA builds happen on shadowfax.
- Local clone of the code: `~/Projects/colibri` (remote `origin` = the fork bobsummerwill/colibri, HTTPS with `credential.helper=!gh auth git-credential`; remote `upstream` = JustVugg/colibri, push URL disabled; remote `shadowfax` = maplemercury@shadowfax:/tmp/colibri, fetch only). These notes: `~/Projects/colibri-claude`.

## Build / run
- CUDA engine (generic, no DeepGEMM on sm_86): `cd c && make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j8`
- CPU-only engine: `make -C c -f Makefile.deepseek-v4 deepseek-v4 -j8`; full checks: `make check` from repo root (~12-14 min; upstream dev passes 2366 tests, 253 skipped).
- Serve: `python3 c/coli serve --model ~/colibri-models/deepseek-v4-flash-reap --ram 40 --ctx 8192 --host 127.0.0.1 --port 8090` with env
  `COLI_CUDA=1 COLI_GPU=0 DSV4_CUDA_DEVICE=0 CUDA_DENSE=1 COLI_CUDA_PIPE=2 COLI_CUDA_ATTN_BATCH=1 COLI_CUDA_MOE_BATCH=1 V4_MOE_REFILL_GROUP=12 V4_LOADER_LANES=3 V4_PREFIX_CKPT_DISK=0`
  (+ `COLI_CUDA_MOE_DOUBLE=1` for the double bank; PR #2: `V4_GPU_RESIDENT_FIRST=1 DSV4_CUDA_PEER_DEVICE=1`). Served model id: `deepseek-v4-colibri`. The engine is greedy (temperature ignored). The gateway refuses logprobs for this engine.
- Diagnostics: `V4_MOE_BATCH_PROF=1` (per chunk-layer route/refill/compute), `DSV4_ATTN_PROF=1`, `DSV4_DECODE_PROF=1` (per token hc/attn/moe ms), `V4_SERVE_STATS_LOG=1` (PR #2: per-turn decode stats), verify modes `DSV4_MOE_EXPERT_MAJOR_VERIFY=1`, `DSV4_MM_WARP_VERIFY=1` (bit-mismatch counters). Nsight: `/usr/bin/nsys` is 2022.4 — run with `--session-new`, stop with `nsys stop --session=...`, convert with `/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter`.
- Upstream CONTRIBUTING: PRs target `dev`; performance PRs need an experiment manifest (>=3 samples per arm, exactly one changed variable, hashed evidence, passing correctness gate), validate with `python3 c/experiment_manifest.py`.
