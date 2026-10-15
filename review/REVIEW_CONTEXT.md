# Review context: colibri DeepSeek-V4 patches (branch `vram-first`, 5 commits on upstream bf24429)

Repo: JustVugg/colibri, a pure-C inference engine for huge MoE models that streams routed experts
from NVMe and caches hot experts in RAM / VRAM. These patches target the DeepSeek V4 engine
(`c/deepseek_v4.c`, CUDA backend `c/backend_cuda_dsv4.cu`, Windows DLL loader `c/backend_loader_dsv4.c`).

## Files in this folder (all paths relative to this `review/` directory)
- `full.diff`: unified diff, upstream `main` (bf24429) → `vram-first` (5 commits). **The thing under review.**
- `commits.txt`: per-commit file stats and subjects.
- `tree/`: the patched source tree (`tree/c/deepseek_v4.c`, 21k lines; `tree/c/backend_cuda_dsv4.cu`, 2.6k lines;
  plus every other source file under `c/`, docs `tree/docs/deepseek-v4.md`, `cuda.md`, `benchmarks.md`).
- `base/`: the upstream (pre-patch) versions of the 5 changed files, for comparison.

`deepseek_v4.c` is compiled as many separate **units**: the same file is compiled repeatedly with
`-DCOLI_V4_UNIT_<NAME>` and each unit only compiles its `#ifdef COLI_V4_UNIT_<NAME>` section; globals
shared across units need one definition and `extern` elsewhere. GPU code is under `COLI_V4_GPU_TIER`.
Builds: Linux CUDA (`make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 ...`, no DeepGEMM on sm_86),
DeepGEMM builds (`DEEPGEMM=1`, sm_120 only, define `COLI_DSV4_DEEPGEMM`), Windows DLL builds through
the loader, CPU-only builds, Vulkan builds (`COLI_VULKAN`), Metal (separate backend).

## What the patches do (author's claims; verify them)
1. **Prefill for expert-pruned checkpoints** (`coli_v4_gpu_moe_batch_union`, deepseek_v4.c ~12042).
   The batched GPU prefill MoE refused anything but 256 routed experts. Now accepts 1..256. For
   counts != 256 the routing runs on the CPU with `coli_v4_route_bf16` (forced_indices=NULL for
   non-hash layers), tokens in parallel via OpenMP, claimed **bit-identical** to the CPU path because
   each token's dot products keep their serial order. Gate f32 mirror only required when routing on
   GPU; device taken from the shared expert mirror otherwise. Loops over experts use n_routed_experts.
2. **Bank sized to the expert count** (`dsv4_cuda_expert_bank_slots`, .cu ~2046): generic builds
   allocate `n_routed_experts` slots, DeepGEMM builds keep 256. Windows loader: optional symbol,
   fallback 256.
3. **Expert-major fp4 MoE kernel** `mv_fp4_expert_major<TB>` (.cu ~138) used by the generic
   (non-DeepGEMM) batched prefill MoE `dsv4_cuda_route_moe_ids_batch` (.cu ~2359), default ON
   (`DSV4_MOE_EXPERT_MAJOR`, `_VERIFY`, `DSV4_MOE_EM_TB`). One warp per expert output row walks all
   routes that chose that expert, TB at a time. **Claimed bit-identical** to `mv_fp4_grouped<4>`:
   lane l keeps the 256-wide tree's partials j = l + 32q (q=0..7); partial j sums i = j, j+256, …
   in order like thread j of the original; the tree's 128/64/32 steps are in-lane adds
   (p[q] + p[q+4], then +2, then +1), and 16..1 are `__shfl_down_sync`.
4. **Warp-per-row dense batch matmul** `mm_batch_w<F,TB>` (.cu ~232) inside `run_mm_batch` (.cu ~260),
   fp8 formats (fmt 8 and 9) only, default "auto" = only when I <= 1024 (`DSV4_MM_WARP`, `_VERIFY`).
   Same bit-identity claim vs `mm_batch<F>`; e4m3 decoded via a 256-entry shared-memory table
   filled with exactly `e4m3()`'s values.
5. **VRAM reserve with `COLI_CUDA_MOE_DOUBLE`** (+3300 MB) and clearing the sticky cudaMalloc error
   after a failed bank allocation (`cudaGetLastError()` in `dsv4_cuda_expert_bank_create`).
6. **Diagnostics**: `V4_MOE_BATCH_PROF` (moebprof lines), `V4_SERVE_STATS_LOG` (v4_serve_turn and
   v4_dec_moe lines in `v4_serve_finish`, ~16569), decode MoE timers (globals `g_v4_dec_*`).
7. **VRAM-first decode** (`moe_token_pipeline`, ~5408; switch `v4_gpu_resident_first` ~5399,
   `V4_GPU_RESIDENT_FIRST=1`, default OFF). Before starting expert-store loads, peek the VRAM mirror
   cache(s); mirrored experts get GPU-only views (no store lookup / lease). Only misses go through the
   loader lanes (`fetch[]` list). `leased[]` marks views holding a store lease; only those are
   released. If the fused GPU path can't be used (an attach failed), GPU-only views are fetched from
   the store lazily (`v4_lookup_waited`) so CPU fallbacks get host data.
8. **Pinned-host cap** `DSV4_CUDA_PIN_CAP_GB` (.cu ~2095), default 20 (unchanged behavior).
9. **Peer mirror pool on a second GPU** (`DSV4_CUDA_PEER_DEVICE=N`, default unset/off):
   - `dsv4_cuda_init` now called with two devices; it already enabled peer access; patch records a
     `g_peer[a][b]` matrix and `dev_ok(a,b)`; new export `dsv4_cuda_peer_access`.
   - second `V4GpuExpertMirrorCache` `g_v4_peer_mirrors` on the peer GPU (`v4_peer_mirrors_create`,
     ~11521, reserve `DSV4_CUDA_PEER_RESERVE_MB` default 1024, capacity `DSV4_CUDA_PEER_MIRRORS` 4096).
   - growth-guard state (reserve, last free probe, unified flag) moved from function statics into
     per-cache fields; one global LRU clock `g_v4_mirror_clock` (atomic) for all caches.
   - `v4_gpu_expert_attach_cached_mode(..., grow_only)` returns -2 instead of recycling;
     `v4_gpu_expert_attach_tiered` (~11782): use a mirror in either cache; else grow primary, else
     grow peer, else recycle the cache with the older LRU entry. `coli_v4_gpu_expert_attach` and
     `coli_v4_gpu_expert_peek` search both caches.
   - `dsv4_cuda_moe` (fused decode MoE, .cu ~1907) now takes its device from the shared expert `sg`
     and accepts routed-expert tensors on peer devices (`dev_ok`). **Peer compute**
     (`DSV4_CUDA_PEER_COMPUTE`, default 1): experts on the peer run THERE: event fork on the primary
     stream → peer stream waits → `cudaMemcpyPeerAsync` of the input → same FC1 / act / fp8_sim / FC2
     kernels on peer-local weights, FC2 output rows written directly into the primary device's `c->p4`
     slots (P2P write) → event join → primary stream waits before `expert_reduce`. Claimed
     bit-identical to computing all experts on one GPU. `dsv4_cuda_expert_group` relaxed to `dev_ok`.
10. Docs in `tree/docs/deepseek-v4.md` (new sections + env table rows).

## Facts established by measurement (shadowfax: 2x RTX 3090 24 GB with an NVLink bridge, P2P OK;
Ryzen 9 9950X3D; 62 GB RAM; WD SN850X; Ubuntu 24.04; CUDA 12.0 toolkit, driver 595; generic build
`CUDA_ARCH=sm_86 NO_TC=1`; model DeepSeek V4 Flash REAP 150B with 132 routed experts, fp4 experts)
- Verify modes reported 0 bit mismatches on real prefill (600+ chunk-layers ×2 for the MoE kernel,
  10,000 dense calls for mm_batch_w). End-to-end greedy outputs identical in every run, including
  six 300-token decodes with the peer pool (remote-read and peer-compute variants) vs single GPU.
- Prefill 2k tokens 220.5 s → 69.1 s; decode 2.78 → 3.87 tok/s with VRAM-first + peer pool.
- Untested: DeepGEMM builds, Windows (loader not compiled), a real 256-expert checkpoint, Vulkan,
  `DSV4_HYBRID=1` combined with the new paths, MTP/dspark drafting, multi-request concurrency,
  long contexts beyond ~3k tokens, `DSV4_CUDA_BATCHED=0`.

## Optional live checks
You may SSH to the test box for compile-only checks or tiny CUDA experiments:
`ssh maplemercury@shadowfax '<command>'`. The patched tree is at `/tmp/colibri-vf` (git worktree,
branch `vram-first`). Rules: do NOT stop/restart services, do NOT modify `/tmp/colibri*`, do NOT run the
engine or the model. Put any experiment under `~/colibri-bench/review-tmp/<your-label>/`, compile with
`/usr/bin/nvcc -arch=sm_86`, run with `CUDA_VISIBLE_DEVICES=0` and keep GPU memory under 1 GB (a
production llama.cpp server is using most of both GPUs).
