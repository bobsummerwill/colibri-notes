# Final review: colibri DeepSeek‑V4 patches (`vram-first`, 5 commits on bf24429)

All paths are relative to `review/tree/` unless marked otherwise. `c/dsv4.def` is not included in `tree/`; its contents were checked on the test box at `/tmp/colibri-vf`. PR #1 is commits 1–2 (dfdc47d, ed001c6). PR #2 is the two‑GPU decode work (b347a73, 2a7ae83). Commit 3 (21a83c5, `V4_SERVE_STATS_LOG`) belongs with PR #2, because its VRAM‑first fields come from b347a73.

## 1. Verdict

**Neither PR is ready to go upstream as it stands. The core numerical claims, though, held up well.**

What held up:
- Expert‑major kernel and warp‑dense kernel bit‑identity: proved statically, and checked at runtime with 0 mismatches over tens of millions of outputs on native and JIT paths, including sm_100/sm_120 opcode checks. Mutation controls confirm the harness would catch a break.
- CPU routing for pruned models: bit‑identical in builds without Vulkan.
- "Unchanged when off" claims: hold.
- Peer compute: bit‑identical across all on_peer masks, checked with a logical‑device shim.

No finding is critical or high. Nothing produces wrong model output in a default or documented configuration.

**What blocks PR #1:**
- (a) `dsv4.def` does not export `dsv4_cuda_expert_bank_slots`, so on Windows the bank sizing never happens (F1).
- (b) A stale prefill hint sends speculative‑verify batches on pruned models through the GPU bank (F2).
- (c) The fixed +3300 MB double‑bank reserve is wrong in every direction, and the docs claim it is right (F7/F8).
- (d) Inaccurate benchmark and doc claims (F40, F42, F46–F48), plus CONTRIBUTING requirements: retarget to `dev`, a performance manifest, and a regression test for the new kernels (F49).

**What blocks PR #2:**
- The tiered mirror attach does not degrade gracefully. A peer pool that is small, misconfigured, VRAM‑starved, or has had one failed upload can push most decode MoE work onto the CPU, which is *worse than not enabling the pool* (F3–F5, F16).
- Two‑device init failure handling: leak plus stale error (F13–F15).
- No latching or detection of peer failures (F17, F18).
- VRAM‑first skews the persistent `.coli_usage` history (F6).
- Benchmark provenance gaps: F43–F45.
- Commit hygiene: a "wip" commit and an unset author identity (F50).

## 2. Confirmed findings, most severe first

Tally format: `n/3 confirmed; severities voted`. "Not real" means a verifier rejected it. Where my final severity differs from the title or the majority vote, I say why. I re‑checked F1, F2, F3, F4, F5, F7 and F11 against the code myself.

### Medium

**F1 (medium). The Windows DLL never exports `dsv4_cuda_expert_bank_slots` or `dsv4_cuda_peer_access`.** [PR1 + PR2] · 3/3, all medium
- Where: `c/backend_loader_dsv4.c:523-524` (optional GetProcAddress) and `:955-962` (fallbacks); `c/Makefile:1021-1043` links with `-Xlinker /DEF:dsv4.def`; `c/dsv4.def` (82 lines, last changed in bf24429) lists neither name. The existing `dsv4_cuda_device_unified` is also missing.
- What's wrong: MSVC exports only the names in the .def file, and nothing is marked `__declspec(dllexport)`. A freshly built DLL therefore always falls back to `experts<=256 ? 256 : 0` slots and `peer_access = 0`. `PR_DESCRIPTION.md:30` says "older DLLs → 256"; in fact current DLLs fall back too.
- Scenario:
  - Windows generic DLL with REAP‑132: the bank is 3264 MiB instead of 1683 MiB, doubled with `COLI_CUDA_MOE_DOUBLE`. On 8–16 GB cards it can fail to allocate, and prefill then runs on the CPU union, about 3× slower.
  - `DSV4_CUDA_PEER_DEVICE=1` on a P2P box: the log says "no peer access … pool off", while the DLL has already initialised the second GPU.
  - Output is never wrong.
- Fix: add all three names to `dsv4.def`. Add a text‑only parity test that checks each `dsv4_cuda_*` in `backend_cuda_dsv4.h` against `dsv4.def` and the loader. `tests/test_backend_loader_header_parity.py` currently covers only `coli_cuda_*`.

**F2 (medium). A stale prefill batch hint lets speculative‑verify batches on pruned checkpoints use the GPU bank.** [PR1, widens existing behaviour] · 3/3, all medium
- Where: `c/deepseek_v4.c:12069` (the only size gate). The hint is set once at `:15334` and never cleared. The bank is released at `:15435`. Verify calls `target_batch(batch=proposals+1)` at `:15608` and `:15720`, which reaches the union for any batch > 1 at `:6741`.
- What's wrong: before the patch, pruned models were refused by the geometry check, so verify used the CPU union. Now they reach the bank. Nothing limits `double_on` (`:12118`) by phase or by batch size.
- Scenario: REAP with `COLI_CUDA_MOE_BATCH=1`, drafting on, and a prompt of at least 256 fresh tokens. On each verify step:
  - the bank (about 1.6 GiB) is re-created mid-decode and kept for the rest of the decode;
  - all 43 layers reset `bank_valid` and re-upload their routed experts;
  - with DOUBLE, a full‑layer prefetch is spawned for each layer, about 70 GB of store reads and uploads per step, which also flushes the RAM cache.
- Fix: add a dedicated in‑prefill flag set around the generate() segment loop, or pass the `use_prefill_pool` intent down. Refuse the GPU union and never start the bank2 prefetch outside prefill. **Do not use `coli_v4_gpu_moe_batch_hint(0)`:** 0 means "unknown, allow", and the oracle and teacher‑forcing callers rely on that.

**F3 (medium). The tiered victim choice compares LRU clocks only, so a cache that cannot recycle captures every miss.** [PR2] · 3/3, all medium
- Where: `c/deepseek_v4.c:11792-11794`. The victim's `-1` comes from `!grow && count < 8` (`:11646`). Only `-2` falls through to the next option (`:11789`, `:11791`).
- Scenario: `DSV4_CUDA_PEER_DEVICE=1` with GPU1 holding less than about 1 GB free (shared card) and the default 1024 MiB reserve.
  - The peer uploads one mirror and freezes at count = 1.
  - Once every primary entry is newer than that one, every miss picks the peer and gets -1, so it runs on the CPU.
  - Harnesses built from the verbatim code measured 48–97% of layer-tokens on the CPU, against 0% with no peer pool.
  - The one-time warning names the wrong environment variable (see F37).
  - Correction: a frozen *primary* with a healthy peer is better than base, which sends 100% to the CPU, but it still misses the documented "recycle the older entry" behaviour.
- Fix:
  - Exclude any cache that cannot recycle (count < 8 and not growing, or no live tensors) from victim selection.
  - If the victim returns -1, retry the other cache.
  - Treat any result from the peer's grow_only call other than 0 or -2 as "peer unavailable".

**F4 (medium). A failed mirror upload leaves a dead entry with clock 0, and the tiered attach never falls back.** [PR2; per-cache dead entries predate the patch] · 3/3, all medium
- Where: `c/deepseek_v4.c:11702` (`found = cache->count++` happens before the upload), `:11732-11738` (returns -1 with tensors NULL and clock 0), and `c/backend_cuda_dsv4.cu:1402` (`upload()` never clears the CUDA error).
- What's wrong: a failed grow_only call returns -1, not -2, so the other cache is never tried. `v4_mirror_lru_clock` returns 0, which makes that cache the permanent victim. Its dead slot cannot be refilled in place, so each recycle needs a new cudaMalloc, and that fails again while the card is full.
- Scenario: the peer is still growing when another process takes GPU1 VRAM, or `DSV4_CUDA_PEER_RESERVE_MB` is below about 820 MiB (F16). Harness: 88–99% of layer-tokens on the CPU, tens of thousands of "weight allocation: out of memory" lines, and the primary is never refilled. Each failure also leaves a stale CUDA error behind.
- Fix:
  - On upload failure, roll back or free the slot (decrement count if it was appended; otherwise mark it free so the found scan cannot match it).
  - Call `cudaGetLastError()`.
  - Return -2 in grow_only mode.
  - Skip NULL-tensor entries in `v4_mirror_lru_clock` and in the found scan.

**F5 (medium). Mirror caches with capacity below 8 return -1 before the grow_only check.** [PR2] · 3/3 (2 medium, 1 low)
- Where: `c/deepseek_v4.c:11578` (`if (cache->capacity < 8) return -1;`). `v4_peer_mirrors_create` (`:11524`) uses `atoi` with no clamp, and `calloc(0)` returns non-NULL, so a pool is created and the banner is still printed.
- Scenario: `DSV4_CUDA_PEER_MIRRORS` set to 0–7 or to a non-numeric value such as "off". Once the primary is full, every miss returns -1. Harness: 12,787 of 12,900 layer-tokens went to the CPU, against 0 without a peer pool.
- Fix: in create, refuse a capacity below 8 with a "peer pool off" message and return NULL. Document the minimum of 8 for both `*_MIRRORS` variables. The F3 fix also covers this case.

**F6 (medium). VRAM‑first skips all of the store's routing accounting, which skews the persistent `.coli_usage` history and the diagnostics.** [PR2] · 3/3 (2 medium, 1 low)
- Where: `c/deepseek_v4.c:5545` (`continue` past the store). `rt_count` and `ehit`/`eheat` are updated only in `lookup_hot` (`:9638-9669`) and in `coli_v4_expert_store_note_routed`, and the latter is compiled only under `#ifdef COLI_VULKAN` (`:9843-9870`).
- Scenario: run several serve sessions with `V4_GPU_RESIDENT_FIRST=1`. The saved history then undercounts the hottest experts, since those are the mirrored ones. Later runs, even with the flag off or on CPU or Vulkan builds, seed autopin and prewarm from that skewed history. The dashboard HITS and EMAP views show the hottest experts as cold. DONE hit% quietly becomes "RAM hit rate among non‑VRAM experts".
- Fix: for each VRAM‑first hit, record `rt_count` and `ehit`/`eheat`, batched once per token-layer and moved out of `COLI_VULKAN`. Leave `stats.hits`/`misses` alone. Decide `policy->usage` separately (see F24). Document what hit% means in this mode.

**F7 (medium; votes were 1 medium, 2 low, and I raised it). The `COLI_CUDA_MOE_DOUBLE` reserve is a fixed `2800 + 3300` MiB that does not depend on the bank slot count.** [PR1] · 3/3
- Where: `c/deepseek_v4.c:11603-11607`. A bank slot is 12.75 MiB (13.5 MiB on DeepGEMM), per `.cu:2061-2066`.
- What's wrong:
  - A 256‑slot pair needs 6528 MiB (6912 MiB DeepGEMM), more than the 6100 MiB reserve. The free-VRAM probe runs only every 64 misses, so up to about 816 MiB more can be used past it. From turn 2 on the second bank fails and latches off (F12), while decode keeps paying for the reserve.
  - For the measured generic REAP‑132 pair (3366 MiB), about 1.4–1.9 GiB of decode mirrors are held back for nothing.
  - `docs/deepseek-v4.md:275-277` ("allocates after decode has grown its mirror cache") is true only for pruned models on generic builds.
  - Base's 2800 MiB was already smaller than one 256‑slot bank (pre-existing).
- Why I raised it: the same constant causes F8, and it makes a documented claim false. One fix closes both.
- Fix:
  - Compute `bank_slots(n) × per-slot bytes × number of banks`, plus attention headroom, plus a 64‑mirror probe margin.
  - Store the result in a global at engine open (the attach path has no config).
  - Drop the double-bank add-on once bank2 is disabled.
  - Cap the default so a minimum mirror budget always fits.

### Low

**F8 (low; votes 1 medium, 2 low). With `COLI_CUDA_MOE_DOUBLE=1` on cards of 12 GB or less and no explicit reserve, the decode mirror cache freezes at 1 entry and every token's MoE runs on the CPU.** [PR1] · `c/deepseek_v4.c:11646` together with the reserve at `:11606`.
- After the dense mirrors are uploaded (about 6.3 GB), roughly 5 GB is free, which is below 6100. The `<8` rule then returns -1 for every miss.
- The docs' 10–12 GB row sets `DSV4_CUDA_VRAM_RESERVE_MB=2500` and is not affected. The 6 GB row combined with DOUBLE is affected.
- Worst case: a 256‑expert model on 12 GB. Bank2 never fits, latches off, and the reserve still applies.
- Fix: the F7 fix, plus fixing docs `:243`/`:276`.

**F9 (low). A recycled-mirror refill does not honour `sync` when `down` takes the pageable path.** [PR2] · 3/3 low · `c/backend_cuda_dsv4.cu:1424-1431`, recycle at `c/deepseek_v4.c:11674-11686`.
- What's wrong: the pageable branch does a legacy-stream `cudaMemcpy` that is not ordered against the non‑blocking `c->stream`, where the gate/up `cudaMemcpyAsync` are queued.
- New cross-stream readers: `DSV4_CUDA_PEER_COMPUTE=0`, `DSV4_CUDA_BATCHED=0`, and the `expert_group` fallback.
- Live result: 0 of 300 stale reads on an idle stream, but 299 of 300 with a stream backlog. Today it is latent.
- The default peer-compute path is safe.
- With the default 20 GiB pin cap and V4 dimensions, no slot ends up mixed (the 3.5 MiB remainder is smaller than a 4 MiB range). A mixed slot needs other caps (for example 12, 15, 22, 25 or 28 GiB) or other expert sizes.
- Fix: use `cudaMemcpyAsync(..., c->stream)` in **both** branches, then `if(sync) cudaStreamSynchronize`. A stream sync alone does not order the legacy copy.

**F10 (low). After a failed prefill, engine destroy runs while the DOUBLE prefetch worker is still alive (use-after-free).** [Pre-existing; PR1 makes it reachable for REAP] · 3/3 low
- Where: `c/deepseek_v4.c:15382-15389` skips release. `coli_v4_gpu_engine_close` (`:11033-11064`) never joins the worker. The store's `destroy` frees the store and asserts `active_leases == 0`.
- Scenario: CLI with BATCH=1 and DOUBLE=1, and a read error in the middle of a layer. Teardown ends in SIGABRT or SIGSEGV instead of exit 1.
- Correction: cancel and embedding failures happen between segments, where no worker is alive. On those paths only the banks leak.
- Fix: call `coli_v4_gpu_moe_batch_release()` on every exit of generate, and at the top of `coli_v4_gpu_engine_close`, before `dsv4_cuda_shutdown`.

**F11 (low; title said medium, all three votes low). `release` is skipped on CANCELLED and on failures, so `bank_failed=1` carries into the next request and the bank stays allocated while idle.** [Pre-existing loop; PR1 widens it] · `c/deepseek_v4.c:15371`, `:15377`, `:15388`, `:15404` against the single `release` call at `:15435`. `bank_failed` is cleared only at `:12020`.
- The next request runs its whole prefill on the CPU union (one request only). The bank (or bank pair) stays held while the server is idle, which matters on shared GPUs.
- KV_SLOTS mux passes no `should_abort`, so only real failures hit this there.
- Fix: same as F10.

**F12 (low; votes low, not real, low). `v4_bank2_disabled` is never reset.** [Pre-existing, an intentional upstream latch] · `c/deepseek_v4.c:12011` and `:12146`. One transient failure of the second-bank allocation leaves the whole process single‑bank, about 15% slower prefill. Since the patch, the 3300 MB reserve also stays in place. Fix later: once F7 is fixed, reset the flag in `release` when the cause was an allocation failure, keep the structural (no‑aux) latch permanent, and drop the reserve add-on while latched. This does not block PR #1.

**F13 (low; title said medium, all three votes low). A failed two-device `dsv4_cuda_init` leaks the first attempt and leaves a stale CUDA error.** [PR2] · `c/backend_cuda_dsv4.cu:1353` has no cleanup. The new retry is at `c/deepseek_v4.c:10983-10991`.
- Measured: 36 MiB lost against 2 MiB in the control. The first `decode_state_set` after the retry returns 0 with "invalid device ordinal".
- The warning always blames the peer.
- Fix:
  - Add a per-device teardown helper, called inside `init` on failure. It should zero the fields, clear `g_peer`, and call `cudaGetLastError()`.
  - Name the device that failed.
  - **Do not** call `dsv4_cuda_shutdown()` from `engine_open` before the retry: on Windows the loader's shutdown unloads the DLL.

**F14 (low). The peer GPU stays fully initialised when the pool is refused (no P2P or wrong arch).** [PR2] · `c/deepseek_v4.c:11021-11027`. `init` succeeds with `can=0`, so GPU1 keeps at least about 290 MiB (measured) for the whole run. Fix: check `cudaDeviceCanAccessPeer` in both directions plus the arch before the two-device init, or shut down and re-init with one device when the pool is refused. Print the actual reason.

**F15 (low). Peers with no kernel image are rejected inside `init`, not by the new arch check, and the warning can blame the wrong card.** [PR2] · `c/deepseek_v4.c:10984` and `:11022`.
- `arch_ok(peer)` is effectively dead code: it never decides anything on DeepGEMM builds, and on generic builds only a Maxwell-targeted build could reach it.
- The failure shows up as "scale table upload: no kernel image" with no device id.
- A DeepGEMM build with a 3090 primary and a CC 12.0 peer prints "peer device 1 did not initialise".
- Fix: probe each device first. `cudaFuncGetAttributes` returns 209 or 98 on such devices (measured). Name the device and give a rebuild hint.

**F16 (low; votes 1 medium, 2 low). The peer reserve floor (256 MiB) is below the probe overshoot (64 × 12.75 MiB ≈ 816 MiB).** [PR2; same mechanism as the primary] · `c/deepseek_v4.c:11528` and `:11629-11631`.
- Replay of the exact guard: OOM in 744 of 1000 start points at reserve 256, 331 of 1000 at 512, 0 at 1024. At 1024 the free VRAM still drops to 208 MiB, against the documented "1024 kept free".
- On its own the OOM is graceful; together with F4 it becomes permanent CPU fallback.
- Fix: subtract each new mirror's bytes from `last_free_mb`, or re-probe whenever free VRAM is within 64 mirrors of the reserve. Do this in the shared function so the primary is fixed as well.

**F17 (low; votes low, not real, low). After a peer-side CUDA failure nothing latches: requests keep failing, with a misleading message.** [PR2] · `.cu:1937-1951` (early returns leave the last error set), the fallback `expert_group` (picks its device from `gate[0]`, `:1851`; checks at `:1889`), then `moe_fail` reports "check nvidia-smi for free VRAM".
- Correction: with a truly dead peer, the fallback still reads the peer's weights, so clearing the error alone does not help. Base has no latch for the primary GPU either.
- Fix: latch the peer pool off on a peer-branch failure, treat its mirrors as misses, log once which device failed, and clear the error before each early return.

**F18 (low). The peer-compute success path never checks the peer context for asynchronous faults.** [PR2] · `.cu:1961`, `:1967`. A sticky peer fault either lets the token through with a wrong output (if the cross-context wait is released) or hangs. Which one happens was not tested. Fix: `cudaEventQuery(pc->join)` after the final sync covers the release case. A host `cudaEventSynchronize(pc->join)` before the download would also cover the hang.

**F19 (low; votes low, not real, low). Error returns after peer work is queued skip the join.** [PR2] · `.cu:1951`, `:1955`, `:1959`, `:1961`.
- Correction: the fallback never reallocates `c->p4` (same sizes) and, on default builds, writes identical values. The only remaining risk is a later reuse of `p4` while the peer kernel is still running.
- Fix: on any exit after the peer FC2 is queued, `cudaSetDevice(pdev)`, then `cudaStreamSynchronize(pc->stream)`, then restore the device.

**F20 (low). Peer-compute selection does not check that the peer can write to the primary.** [PR2] · `.cu:1931` lacks `dev_ok(d, sg->device)`, although the peer FC2 writes into `c->p4`. The engine cannot reach this: the pool requires access in both directions. Fix: add the check.

**F21 (low). The peer-compute "bit-identical" claim has no same-SM-version precondition.** [PR2] · `.cu:1922-1928` and `docs/deepseek-v4.md:294`. The pool requires only `major >= 6`. The `expf` in `expert_act_grouped` compiles to `MUFU.EX2` on every target (SASS checked). Same-SM pairs are unaffected. Fix: require matching major and minor for peer compute (otherwise use the `PEER_COMPUTE=0` mode), or state the precondition.

**F22 (low). In `COLI_VULKAN` builds the OpenMP router bypasses the Vulkan gate matmul for batches of 16 or more.** [PR1] · `c/deepseek_v4.c:12215`. `v4_vk_matmul_impl` refuses calls inside a parallel region (`:13204-13211`). As a result:
- chunks of 16+ use serial CPU dot products;
- chunks under 16 and decode use the device GEMV;
- the CPU union uses the device GEMM.

So the bit-identity claims at `:12210-12213`, `docs:249` and in the PR are false in CUDA+VK builds, and 256-expert hash-layer weights now differ from base. Fix: when the hook is set, compute the logits once on the main thread and use `route_bf16_logits`, or keep the loop serial, or qualify the claim.

**F23 (low). The 256-expert GPU router ignores the bias when the bias mirror is missing.** [Pre-existing] · `c/deepseek_v4.c:12232`. The trigger is a 1 KiB cudaMalloc failing right after a 4 MiB one succeeds. Fix: route on the CPU when `raw_bias && !bias`, as decode already does (`:11438-11440`).

**F24 (low; votes low, low, medium). With VRAM‑first, the hot-pin policy keeps mirrored experts pinned in RAM.** [PR2] · `c/deepseek_v4.c:5545`.
- A seeded history freezes `usage`, so up to 16 experts per layer × 43 layers (about 9 GB) stay pinned for thousands of tokens, contradicting `docs:286-288`. It is not a regression against the flag being off.
- **Pitfalls for the fix:**
  - Driving `note_routed` would make the pinning permanent.
  - A Vulkan-style `ram_first` backfired in simulation on a single GPU: 1.76 against 1.15 SSD reads per token per layer.
- Fix: exclude mirrored experts from `hot_repin_locked`'s ranking (simulation: 1.04 → 0.82 reads at 60 mirrors per layer), and fix the docs and the comment at `:5383-5387`.

**F25 (low). The fallback after a failed attach reads GPU-only experts one at a time on the compute thread.** [PR2] · `c/deepseek_v4.c:5666-5681`. Triggers include prewarm rows16 packing before `store->gpu` is set (pre-existing) and zombie entries (F4). Fix: issue these reads through the loader lanes, or use the hybrid-style split.

**F26 (low). Loader jobs still in flight are never finished when an error occurs after preload.** [Pre-existing] · Cleanup at `c/deepseek_v4.c:5656-5661` sits inside `if (!result)`.
- Preload-start failures (`:5560`) or shared-expert failures (`:5584-5590`) leak lanes, and the worker writes into the dead stack frame. ASan reproduced a stack-use-after-return at `:4970`. Leaked lanes later cause a hang.
- Correction: a missing tensor in the checkpoint is rejected at load time. The real triggers are OOM and pthread failures.
- Fix: run the cleanup loop on every path.

**F27 (low). A failed cudaMalloc outside the bank path leaves the CUDA last error set.** [PR1 verify path, PR2 peer path] · `upload()` (`.cu:1402`), `buf()` (`:73`), `run_mm_batch` verify (`:269-270`).
- When verify cannot allocate, it silently switches itself off for good, and the call reports OOM even though its output is valid.
- A failure in the peer `pc` `buf()` makes the fallback fail and calls `moe_fail`.
- Fix: an `alloc_ok()` helper that clears the error, and a one-time "verify disabled" message.

**F28 (low; votes low, nit, low). `dsv4_cuda_shutdown` does not free `emaj`/`emv1`/`emv2`/`emv4`/`emcount` or `vbuf`/`vcount`, and does not reset `g_peer`.** [PR1/PR2] · `.cu:56` and `:1359-1369`. The retained memory is bounded (about 5 KB by default, about 25 MB in verify mode), not cumulative. **Pitfall:** shutdown already leaves the freed pointers non-NULL (pre-existing use-after-free on re-init). Freeing without zeroing would extend that. Fix: free the buffers, then `memset(&g[i], 0, sizeof g[i])`.

**F29 (low). `run_mm_batch` verify buffers are process-global.** [PR1] · `.cu:264`. Latent, because every caller is on `engine->gpu.device`. Fix: move them into `struct Dev`.

**F30 (low). The default expert-major TB=4 is slower than the per-route kernel at low route density.** [PR1] · `.cu:2421`.
- TB=4 uses 80 registers with a 12 B spill. At 2–12 tokens it is 9–54% slower; it breaks even at about 1.5 routes per expert and is up to 3× faster at full chunks. TB=2 was never slower than the grouped kernel.
- Cost: about 20–60 ms per prompt on tail chunks, more on F2's verify path.
- Fix: choose TB per call, for example TB=2 when `routes < ~2–3 × nex`. Bit-identity holds for every TB.

**F31 (low; votes low, nit, low). `v4_bank2_worker` still loops over 256 experts.** [PR1] · `c/deepseek_v4.c:11962`. Harmless, because `get_record` rejects out-of-range ids, but it contradicts `PR_DESCRIPTION.md:25`. Fix: pass `n_routed_experts` in the job.

**F32 (low). `coli_v4_gpu_expert_peek` scans the primary cache twice on every hit when a peer pool exists.** [PR2] · `c/deepseek_v4.c:11829`. Measured cost is about 0.05–0.13 ms per token. The described stale-pointer case cannot be reached on a single thread. Fix: `return v4_mirror_peek_cache(primary,v)==0 ? 0 : v4_mirror_peek_cache(g_v4_peer_mirrors,v);`

**F33 (low). Interactions between the new switches are not documented.** [PR2]
- Under `DSV4_HYBRID=1`, VRAM‑first is off (`:5540`), and `attach_async` (`:11808`) uses only the primary, so the peer pool stays empty while the banner still prints.
- `PEER_COMPUTE` has no effect with `BATCHED=0`.
- Batched decode and verify use neither feature.

**Pitfall:** if `attach_async` is made tiered, peer refills must be synchronous, because `drain` only drains the primary stream. Fix: document these, print a one-time warning, or skip creating the pool under hybrid.

**F34 (low). The decode and VRAM‑first counters are inaccurate.** [PR2] · `:5546`, `:5605`, `:5934`.
- Hits are counted before a CPU fallback.
- Lazy fetches are not timed.
- Prefill rows are counted, and batched decode is missed.
- `dec_t0` is taken after routing.
- "attaches" counts calls, not uploads.

Fix as the finding suggests, or document the counters' scope.

**F35 (low). The `v4_serve_turn` line: one "(cumulative)" suffix covers a mix of per-turn and cumulative fields, `store_fetches` is 0 when VRAM‑first is off, and the "decode stats" include prefill.** [PR2/commit 3] · `:16669`, `:5553`. Fix: name the totals `*_total`, count fetches unconditionally, and document the scope (docs `:309`, `:483`).

**F36 (low). The moebprof split leaves out the bank2 join wait.** [PR1] · `moeb_t0` at `:12183` is taken after `v4_bank2_join` (`:12117`). Fix: add a `join=` or `switch=` field.

**F37 (low). The "frozen below the VRAM reserve" warning is one shared static, names neither cache nor device, and names `DSV4_CUDA_VRAM_RESERVE_MB` even for the peer.** [PR2] · `:11647`. Fix: a per-cache flag that prints the device, `reserve_mb`, and the correct variable (`DSV4_CUDA_PEER_RESERVE_MB` for the peer).

**F38 (low). The startup banner and comments say "read in place over P2P", although peer compute is the default.** [PR2] · `:11529-11531`, `:11496`, `.cu:1909-1910`. The active mode is never logged. Fix: print the mode and reword the comments: in-place reads happen only with `PEER_COMPUTE=0`, `BATCHED=0`, or the `expert_group` fallback.

**F39 (low; votes low, nit, low). The multi-GPU row of the build table puts the decode text in the prefill column.** [PR2] · `docs/deepseek-v4.md:396`. The GPUs and decode columns read "—". `:355` still says "multi-GPU design exists on paper only".

**F40 (low). Bank-size comments and docs are stale or contradictory.** [PR1, partly pre-existing]
- docs `:211` and `:223` say 2.2 GB, while `:252` says 3.2 GB. The real figures are 3264 MiB (256 slots) and 1683 MiB (132 slots).
- `:215` leaves out the +3300 MB.
- Comments at `:11591`, `:11596`, `:11880`, `:11912` and `:15432` say "~2.2 GiB".
- The new comment at `:12074-12077` ("still fit the 256-slot bank") is wrong for generic builds.

**F41 (low). The "~8 MB per decode mirror" figure is about 37% too low (pre-existing).** · docs `:213`, `:468`, and the comment at `:11016`. The real size is 12.75 MiB. The VRAM table's mirror counts cannot be reached: the 24 GB row would need 21.2 GiB of mirrors on top of 6.3 GB of dense weights. The new peer-pool sizing rows inherit the error.

**F42 (low; title said medium, all three votes low). "Decode is unchanged (2.2–2.9 tok/s)" does not hold for answers after a long prompt.** [PR1] · docs `:274-275` and `PR_DESCRIPTION.md:70`.
- Like-for-like comparisons: the 2k-prompt answer drops from 2.42 to 2.18 tok/s (about 10%, about +1.1 s over 24 tokens, consistent across R2, R3 and S). The 3k-prompt answer drops from 2.85 to about 2.75 (about 3%). Short prompts are unchanged.
- The quoted range is built from the after-runs only.
- Correction: the finding's "10–20%" mixed in cold first-request runs and is overstated.
- Likely cause, not proven: the bank refill looks experts up less often than the CPU union does, so the RAM hot set warms less.
- Fix: give decode numbers per prompt.

**F43 (low; title said medium, all three votes low). The decode table was measured with `COLI_CUDA_MOE_DOUBLE=1` (a 6.1 GB GPU0 reserve), and the docs do not say so.** [PR2] · docs `:296-304`. The systemd journal confirms the flag on every row, including "one GPU (before)". GPU0 had about 3.2 GB fewer mirrors. With default settings, the baseline is likely 3–5% higher and the relative gain somewhat smaller. Fix: state the environment, or re-measure.

**F44 (low). "Answers identical in every run" for the decode table rests on 200-character heads for the baseline.** [PR2] · docs `:297`. V0, V1 and W stored only `text_head` (about 45 of 300 tokens). Full-text equality exists only among X1, X2 and X3, which all have VRAM‑first on. `cmp_x.py` treats missing text as "" and reports identical. The prefill-table answers are under 200 characters and were compared in full. Fix: re-run V0 saving the full text, or narrow the wording.

**F45 (low). The `DSV4_CUDA_PEER_COMPUTE=0` row (3.32) was measured before that variable existed.** [PR2] · docs `:303`. Run X2 used an intermediate binary (vf4, linking vf3's `.cu` object). The X1 row also comes from that binary. The final code's `=0` path was never run; by inspection it launches the same kernels in the same order. Fix: re-run on the final binary.

**F46 (low). The "CPU union (before)" cell for about 3,000 tokens (334.4 s) is not an upstream measurement.** [PR1] · docs `:272` and PR `:65`. It comes from the first patched build (256-slot bank), as the second request, after a bank OOM and a sticky "batched dense launch: out of memory". Upstream never ran that prompt. The per-token rate does match upstream (0.111 s/token). Fix: re-measure on bf24429, or add a footnote.

**F47 (low). The prefill-table cells are not like-for-like.** [PR1] · docs `:266` and PR `:60`.
- The timed request is at a different position in each column (1st or 2nd vs 3rd or 4th).
- Column 2 ran with `DSV4_ATTN_PROF=1`.
- Every cell is a single run.

Corrections: the "~72 s at first position" runs lacked the warp-dense kernel (worth about 2.5 s), so the position effect is about 0–4%. Same-position repeats agree within 0.5%. R3's reap9 build makes no functional difference. Fix: run one script for every column on the final build, report which request was timed, and use at least 3 runs.

**F48 (low). The PR's "Kernel time (Nsight Systems) ~65 s → ~22 s" compares two different metrics.** [PR1] · PR `:72-73`. The 65 s is moebprof host wall time (serve‑D: 64.9 s, which includes shared experts, the download and the sync). The 22 s is an nsys kernel sum. Measured the same way: 64.9 → 26.4 s (2.46×). Fix: quote one metric, or capture nsys with `DSV4_MOE_EXPERT_MAJOR=0`.

**F49 (low). The PR text does not meet CONTRIBUTING, and no regression tests were added.** [PR1/PR2]
- No experiment manifest (at least 3 samples per arm, exactly one changed variable). The "+ expert-major + warp dense" arm changes two variables.
- No record of `make check` or `cuda-test` runs.
- The branch is based on `main`, and `dev` (88f25a6) conflicts on the `dsv4_cuda_init` line and in `struct Dev`.
- No tests were added. The existing `test_dsv4_moe_batch_cuda` calls a DeepGEMM-only entry point that is a `return 0` stub on generic builds. `test_dsv4_dense_batch_cuda` uses I=4096, which never reaches `mm_batch_w` by default.
- The PR description does not cover commits 3–5.

**F50 (low). Commit hygiene.** [PR2] · b347a73 is "wip:" with no body (283+/53-) and bundles pin cap, timers, VRAM‑first and the peer pool. All five commits are authored `maplemercury <maplemercury@shadowfax>` because `user.email` is unset. Fix: split the commit, write bodies, set the identity, and run `git rebase --exec 'git commit --amend --reset-author --no-edit' bf24429`.

### Nit

- **F51 (nit; votes low, not real, nit).** `DSV4_MM_WARP` parsing (`.cu:265`): the literal "auto" or an empty value turns the kernel off; any non-zero value forces it; I%256 is still required; VERIFY overrides 0. Treat empty or "auto" as unset, and document this.
- **F52 (nit, 3/3).** `std::copy` is used without `#include <algorithm>` (`.cu:2432`).
- **F53 (nit, 3/3).** `run_fp4_expert_major` and its kernels are unreferenced in DeepGEMM builds (#177‑D; `run_mm_batch` already had the same). Guard them with `#ifndef COLI_DSV4_DEEPGEMM`.
- **F54 (nit, 3/3).** Expert-major mode still builds and sizes the unused MvDesc tables (`.cu:2409-2457`). There is nullptr+1 arithmetic at `:2439` when the mode is off. `DSV4_MOE_EM_TB` silently maps unsupported values to 4.
- **F55 (nit, 3/3).** The warp-kernel comment and docs say "slower on 2048/4096-row shapes". On 4096×2048 (shared w2) it measured 0.85×, i.e. faster. It is slower only for I=4096 (about 1.1×) and I=8192 (about 1.7×). Reword, or extend auto to I≤2048 after an in-engine check.
- **F56 (nit, 3/3).** Dead or redundant code:
  - the unused `clock` field (`:11480`);
  - a duplicate `g_v4_peer_mirrors` (`:11498`);
  - a redundant `count_bit_mismatch` declaration (`.cu:259`);
  - `gpu_timed` (`:5742`);
  - `v4_dec_now`, a duplicate of `v4_wait_clock`;
  - `v4_bank_double_on` given external linkage for no reason (a static forward declaration works).
- **F57 (nit; votes low, nit, nit).** `DSV4_CUDA_PIN_CAP_GB`: values above about 1.7e10, or "inf", convert to size_t with undefined behaviour, and in practice give 0, which disables pinning. Empty gives 0. The unit is GiB. Clamp it.
- **F58 (nit; votes nit, not real, nit).** Both GPUs run act/fp8_sim over all `count` rows. Measured cost is about 0.5 µs per layer at topk=6.
- **F59 (nit; votes nit, nit, low).** `COLI_V4_EXPERIMENTAL_PREFETCH` (`:5507-5517`) prefetches experts that will be served from VRAM. No build defines the flag.
- **F60 (nit, 3/3).** Block-scope externs (`:16657-16659`) instead of the internal header, and a misaligned ternary (`:11605-11607`). **Pitfall:** the header declarations must go *outside* the `COLI_V4_GPU_TIER` block, or CPU-only builds break.

## 3. Claims verified correct

| Claim | Evidence |
|---|---|
| `mv_fp4_expert_major<TB>` is bit-identical to `mv_fp4_grouped<4>` | Static: same per-term `(x·f4)·e8` as `fma.rn`; lane l, slot q holds partial j=l+32q and sums in i order; in-lane +4/+2/+1 and shfl 16..1 reproduce the 256-wide tree. Runtime: 0 mismatches on counts 1..128 and TB 2/3/4/6/7, about 39M outputs per TB, with ±0/denormal/Inf/NaN/overflow inputs. A mutation control finds 11–13M mismatches. |
| `mm_batch_w<F,8>` is bit-identical to `mm_batch<F>` (fmt 8 and 9) | 1224 configurations plus a 7.7M-output sweep, 0 mismatches, including NaN codes and extreme scales. Through the backend, hashes match for `MM_WARP` = 0, unset and 1. LUT values equal `e4m3()` for all 256 codes. |
| Bit-identity does not depend on GPU generation or JIT | PTX for compute_61/80/86/90 (nvcc 12.0 and 12.6) has 0 contractable mul→add pairs. SASS FFMA/FMUL structure is the same on sm_61–90. Driver JIT for sm_100/120 gives the same opcode counts. Runtime: 0 mismatches over 8 native and PTX‑JIT paths. |
| CPU routing for n≠256 is bit-identical to the CPU union and to decode (no Vulkan) | Harness with the real units: `routing_mismatch=0` for E∈{6,7,13,24,32}, 1–4 hash layers, OMP 1–32. The dot loop is not vectorized. |
| The OpenMP router is thread-safe | It uses a reduction flag rather than an early return, stack scratch, and writes disjoint output slices. Compiles cleanly with `-fopenmp -Wall -Wextra`. |
| The 256-expert batch-union control flow is unchanged (no Vulkan) | Same refusals; `bank_slots(256)=256`; the same `route_top6_batch` and loops. |
| Pruned bank: correct contents, no stale or out-of-bounds reads | Slot-map checks show `stale_routes=0` and `stale_valid=0` across DOUBLE, BANK_FULL, lease pressure and injected failures. compute-sanitizer memcheck, racecheck, initcheck and synccheck report 0 errors. |
| A 256-slot DeepGEMM bank used for a pruned model never touches unrouted slots | A CPU replay of the `dg_moe_batch_contiguous` layout over 20k chunks showed 0 violations and a maximum slot of 131. `bank_slots` returns 256 in the DeepGEMM-configured object. |
| Unchanged when off (`EXPERT_MAJOR=0`, `MM_WARP=0`, VRAM‑first unset, no peer) | Base kernels and launches run unchanged. Fetch order is the identity. A base-tree harness gives byte-identical output to the patch with the flag off and with it on. |
| VRAM‑first lease bookkeeping | Over 150 ASan runs plus a 45-case fault fuzz: each lease released exactly once, no unleased view released, no job in flight on main-loop error paths. The only exception is F26, which predates the patch. |
| VRAM‑first on is bit-identical to off on paths without failures | `cmp` SAME for full, thrashing, frozen (the lazy path), hybrid, BATCHED=0, and missing shared mirrors. |
| Two caches cannot reintroduce the #1538 in-place recycle hazard | About 30M refills with a version-tagged stub: 0 stale references. A negative control without the 8-entry rule does detect the hazard. |
| Peer compute is bit-identical, with correct fork/join ordering | Logical-device shim: 1330 calls across every on_peer mask, 0 mismatches. Injected 3 ms delays left output unchanged. Removing the join or fork wait produces mismatches. memcheck, synccheck and racecheck are clean. |
| Peer pool in the full decode pipeline equals a single device | 40 configurations, including recycling in both caches and both VRAM‑first settings: byte-identical. |
| Cross-device event and P2P usage is legal; p4 slots are disjoint | Events are recorded on their own devices and only waits cross devices. On_peer and !on_peer descriptors are disjoint. |
| The `cudaGetLastError()` added after a failed bank cudaMalloc is necessary | Live: the OOM error persists through later successful launches until it is read. |
| `cudaErrorPeerAccessAlreadyEnabled` is now cleared | Added at the end of `init`. |
| The pin cap default is unchanged | `20.0·2^30 == (size_t)20<<30`. |
| Windows loader fallbacks are safe with older DLLs | An old DLL's `count==256` check accepts the 256-slot fallback. `peer_access=0` turns the pool off. |
| Builds and bisectability | CPU and CUDA sm_86 builds and the portable gencode set compile. A 54-compile flag matrix gave identical results on base and patch. Every commit builds. No link breaks across unit splits. |
| No regressions in the project test suite | Python suite (2296/2305 tests), `make check`, ASan/UBSan, the single-device CUDA tests and `cuda-test` all match between base and patch. |
| DeepGEMM-configured compile | With stubbed DeepGEMM, nvcc 12.6 and sm_90a, the only new diagnostic is #177‑D (F53). |
| Mirror caches are touched only by the generation thread | Call-site audit: loaders, the bank2 worker and prefill never touch the caches. dspark attach has no callers. |
| The main measurement numbers match the raw logs | Prefill TTFTs; decode means 2.78/2.84/3.32/3.87; greedy runs are deterministic; about 12.9 GB/s HtoD; DOUBLE costs about 3–5% of decode; prefill is unchanged with the peer pool (69.46 vs 69.11 s). |
| `cudaErrorNoKernelImageForDevice` is not sticky; one-way P2P cannot reach the F20 gap through the pool | Live: later launches succeed. The pool requires `peer_access` in both directions. |

## 4. Rejected findings

- **Hybrid + peer pool: the expert_group fallback on the peer races async primary refills.** Rejected 3/3. With hybrid async on, `attach_async` uses only the primary, so the peer pool never fills. With async off, every refill is synchronous. A peer-backed view and pending refills never coexist. This is latent only; see F33's pitfall.
- **The verify-mode evidence does not cover the shipped selection.** Rejected 2/3 (one verifier rated it a nit). The old all-shapes selection is a superset of the shipped one, so attn.wq_b was verified in run J as well. Only every 500th call is logged, which explains why that shape never appeared in the samples. The kernel bodies are unchanged from dfdc47d to 2a7ae83.

## 5. Risks not covered by testing or review

- **The real DeepGEMM build** (nvcc ≥ 12.8 plus DeepGEMM on sm_120a) was never compiled or run. Only a stubbed compile with nvcc 12.6 and sm_90a, plus a CPU layout replay, was done.
- **The Windows/MSVC DLL build** was not compiled. F1 is derived from `.def` semantics, not a real link.
- **A real 256-expert checkpoint** was never run. The 256-slot reserve and second-bank behaviour (F7/F8/F12) are known from arithmetic only.
- **Vulkan builds** were compiled only (F22 is static analysis).
- **Two physical GPUs under sanitizers or fault injection.** Peer-compute correctness came from a logical-device shim on one GPU plus the author's NVLink runs. What the driver does with a cross-context wait after the peer faults (release or hang; F18) is unknown.
- **Mixed-SM P2P pairs**: whether MUFU.EX2 differs across generations (F21) is untested.
- **PCIe-only or WDDM P2P** behaviour, and the GeForce P2P assumption.
- **MTP/dspark/n-gram drafting** end to end. F2's cost is estimated from code, not measured.
- **Multi-request concurrency and KV_SLOTS mux** with the new paths.
- **Contexts beyond about 3k tokens**, and long-running servers (accumulated cache freeze, latch effects).
- **The final `DSV4_CUDA_PEER_COMPUTE=0` path** was never run end to end (F45).
- **The cause of the post-GPU-prefill decode slowdown** (F42) is still a hypothesis.

## 6. Prioritized fix plan

**Before PR #1 (commits 1–2):**
1. **F1:** add `dsv4_cuda_expert_bank_slots` and `dsv4_cuda_device_unified` to `dsv4.def`, and add a dsv4 export parity test.
2. **F2:** add an in‑prefill flag (not `hint(0)`). Refuse the GPU union and the bank2 prefetch outside prefill.
3. **F10, F11:** call `coli_v4_gpu_moe_batch_release()` on every generate() exit and at the top of `coli_v4_gpu_engine_close`. Then decide F12 (reset the latch on allocation-failure causes).
4. **F7, F8:** derive the double-bank reserve from the slot count, with headroom and probe margin, and cap it so a minimum mirror budget fits. Fix the docs and comments listed in F40.
5. **Docs and PR accuracy:**
   - F42: give decode per prompt.
   - F46: re-measure or footnote the 334.4 s cell.
   - F47: state request positions; same script per column.
   - F48: use one metric.
   - F22: say Vulkan is excluded from the bit-identity claim.
   - F51/F55: MM_WARP "auto" handling and the 2048 comment.
6. **F49:** rebase onto `dev` (resolve the `dsv4_cuda_init`/`struct Dev` conflict). Write a manifest with at least 3 samples per arm, and measure expert-major and warp-dense separately. Add a CUDA test that calls `dsv4_cuda_route_moe_ids_batch` directly on a small E bank, comparing `EXPERT_MAJOR=0` vs 1 and TB 2/3/4/6 bitwise, plus an `mm_batch_w` test with I≤1024. Make the MoE batch test skip on generic builds.
7. **Cheap hygiene:**
   - F52: `#include <algorithm>`.
   - F53: `#ifndef COLI_DSV4_DEEPGEMM` guard.
   - F28: free and zero `Dev` in shutdown (zeroing is required).
   - F27: clear recoverable cudaMalloc errors.
   - F29: per-device verify buffers.
   - F54: skip the MvDesc build in expert-major mode and validate TB.
   - F31: bound the bank2 worker loop by n.
   - F30: optionally use TB=2 at low density.

**Before PR #2 (two-GPU decode):**
1. **Tiered attach rework** (F3, F4, F5, F16, F37):
   - refuse peer capacity below 8;
   - exclude caches that cannot recycle from victim selection, and retry the other cache on -1;
   - on upload failure, roll back the slot, clear the error, and return -2 in grow_only mode;
   - skip NULL entries in the LRU and found scans;
   - keep `last_free_mb` conservative between probes;
   - per-cache warning with device and the correct variable.
2. **F1:** add `dsv4_cuda_peer_access` to `dsv4.def`.
3. **Device init** (F13, F14, F15): clean up inside `init` on failure (do not call shutdown from `engine_open`). Probe the peer first: device count, `CanAccessPeer` in both directions, arch, and kernel image. Name the device that failed, and do not keep the peer initialised when the pool is refused.
4. **Peer failure handling** (F17, F18, F19, F20, F21):
   - clear errors on early returns, and sync `pc->stream` on exits after FC2 is queued;
   - check the peer join after the final sync;
   - latch the pool off and treat its mirrors as misses after a peer failure;
   - add `dev_ok(d, sg)` to the on_peer condition;
   - require the same SM version for peer compute, or document it.
5. **F9:** use `cudaMemcpyAsync` on `c->stream` in both refill branches, then honour `sync`.
6. **VRAM‑first accounting** (F6, F24): record routing for VRAM hits outside `COLI_VULKAN`. Exclude mirrored experts from pin ranking; do not use Vulkan-style `ram_first`. Fix the docs.
7. **Interactions and fallbacks:**
   - F33: document or warn about hybrid; if `attach_async` becomes tiered, peer refills must sync.
   - F25: route lazy fetches through the loader lanes.
   - F26: finish in-flight loader jobs on every exit.
8. **Diagnostics and docs:** F34, F35, F38, F39, F41 (use 12.75 MiB and recompute the VRAM table), and F32.
9. **Benchmarks:**
   - F43: state the decode table's environment (`COLI_CUDA_MOE_BATCH=1 COLI_CUDA_MOE_DOUBLE=1`) or re-measure.
   - F44: re-run V0 saving full text, and fix `cmp_x.py`.
   - F45: re-run the `PEER_COMPUTE=0` and X1 rows on the final binary.
10. **F50:** split the wip commit, write commit bodies, fix the author identity.
11. **Nits:** F56, F57, F58, F59, F60 (header declarations must go outside the GPU_TIER guard).
