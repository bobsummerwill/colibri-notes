export const meta = {
  name: 'pr2-fixes',
  description: 'Implement, review and integrate the PR #2 (two-GPU decode) review fixes for colibri',
  phases: [
    { title: 'Implement', detail: '5 fix groups on pr2/split' },
    { title: 'Review', detail: 'independent adversarial review per group' },
    { title: 'Fixup', detail: 'address review issues, re-review' },
    { title: 'Integrate', detail: 'merge, full builds + tests' },
  ],
}

const S = '/home/zombietiger/.config/Claude/scratch-workspaces/c9774333-2509-4b09-a6db-1980582ceec7/32aceebc-9fe5-4957-8332-03b2bcab77b9/scratch-2026-10-13-2b21f7'
const CLONE = `${S}/colibri-local`
const REPORT = `${S}/review/REVIEW_REPORT.md`
const CONTEXT = `${S}/review/REVIEW_CONTEXT.md`
const BASE = '44a5e7e1'   // tip of pr2/split

const COMMON = `You are fixing review findings in colibri (https://github.com/JustVugg/colibri, a C/CUDA MoE inference engine), DeepSeek V4 engine, PR #2 = "two-GPU decode": VRAM-first decode lookups (V4_GPU_RESIDENT_FIRST), a peer mirror pool on a second GPU (DSV4_CUDA_PEER_DEVICE) whose experts the fused decode MoE either reads over P2P/NVLink or (default) computes on the peer GPU (DSV4_CUDA_PEER_COMPUTE), plus decode diagnostics (V4_SERVE_STATS_LOG) and DSV4_CUDA_PIN_CAP_GB.
Background: ${CONTEXT} (line numbers/paths there are from an OLDER tree — search by content). Full review with every finding (F1..F60), failure scenarios and suggested fixes, including pitfalls: ${REPORT}. Read your assigned findings there in full before coding.
Repository: local clone ${CLONE} (origin = maplemercury@shadowfax:/tmp/colibri; nothing ever goes to GitHub). PR #2 is origin/pr2/split at ${BASE}: 7 commits on top of origin/pr1/fixed (PR #1, final, do not change it). Key files: c/deepseek_v4.c (compiled as many units via -DCOLI_V4_UNIT_<NAME>; GPU code under COLI_V4_GPU_TIER; mirror caches V4GpuExpertMirrorCache, v4_gpu_expert_attach_cached_mode, v4_gpu_expert_attach_tiered, coli_v4_gpu_expert_peek, v4_peer_mirrors_create, coli_v4_gpu_engine_open/close, moe_token_pipeline, v4_serve_finish), c/backend_cuda_dsv4.cu (compact one-line CUDA style; dsv4_cuda_init, dsv4_cuda_shutdown, dsv4_cuda_moe, dsv4_cuda_expert_group, refill/upload, pin cap), c/backend_cuda_dsv4.h, c/backend_loader_dsv4.c + c/dsv4.def (a parity test c/tests/test_backend_loader_dsv4_parity.py requires every dsv4_cuda_* in the header to be exported in dsv4.def and resolved/wrapped by the loader — new backend functions must be added to all three; optional GetProcAddress with a safe fallback for older DLLs), docs/deepseek-v4.md.
Rules:
- Work ONLY in your own worktree: git -C ${CLONE} fetch -q origin && git -C ${CLONE} worktree add -b fix2/<key> ${S}/wt/fix2-<key> origin/pr2/split (reuse if it exists). Edit locally with Read/Edit/Write. Commit as the clone's identity (Bob Summerwill <bob@summerwill.net>), subject "deepseek-v4: <what>", body with what/why and the finding ids, trailer "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>". Push: git push -f origin fix2/<key>.
- IMPORTANT TIMING: a GPU benchmark is running on the test box. Before ANY ssh command other than git push, wait until it has finished: poll  ssh maplemercury@shadowfax 'grep -c "restore:" ~/colibri-bench/run_pr1_bench.log'  every 60 s until it prints 1 (it should finish around 11:00 local time). Do your local implementation work first.
- CUDA builds only on the test box (no nvcc locally), after the benchmark: ssh maplemercury@shadowfax 'cd /tmp/colibri && (git worktree add -q --detach /tmp/cw/b2-<key> fix2/<key> 2>/dev/null || git -C /tmp/cw/b2-<key> checkout -q --detach -f fix2/<key>) && cd /tmp/cw/b2-<key>/c && make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j6 2>&1 | grep -iE "error|warning" | grep -v lto'  → must be 0 warnings, 0 errors. Never put single quotes inside an ssh '...' command; push code instead of heredocs.
- Local checks: CPU-only build make -C <wt>/c -f Makefile.deepseek-v4 deepseek-v4 -j8 (0 warnings); gcc -fsyntax-only -DCOLI_V4_GPU_TIER of the GPU-tier units you touch (see Makefile.deepseek-v4 for flags/units); python3 -m unittest of the parity/loader/source tests from c/.
- GPU tests on the test box (after the benchmark): small only. The test box has 2x RTX 3090 with an NVLink bridge (P2P works), but a production llama.cpp server uses ~19.5 GB of GPU0 and ~22.7 GB of GPU1: keep each test under ~400 MB per GPU, never stop/restart services, never run the engine on the real model, never touch /tmp/colibri-vf, /tmp/cw/pr1, /tmp/cw/pr2, /tmp/cw/devbase, /tmp/cw/bench-*. Put scratch files under ~/colibri-bench/review-tmp/<key>/.
- Default behaviour with the PR #2 switches OFF must remain exactly PR #1's behaviour; kernel changes must keep results bit-identical (the PR's central claim).
- Stay in scope; match surrounding style and comment density; no unrelated refactors. If a suggested fix is wrong or you choose differently, explain. Update docs/deepseek-v4.md only for behaviour your group changes.`

const RESULT_SCHEMA = {
  type: 'object',
  properties: {
    branch: { type: 'string' }, head_sha: { type: 'string' },
    commits: { type: 'array', items: { type: 'object', properties: { sha: { type: 'string' }, subject: { type: 'string' } }, required: ['sha', 'subject'] } },
    findings: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, status: { type: 'string', enum: ['fixed', 'partial', 'deferred', 'not_applicable'] }, summary: { type: 'string' } }, required: ['id', 'status', 'summary'] } },
    files_changed: { type: 'array', items: { type: 'string' } },
    builds: { type: 'object', properties: { cuda_sm86: { type: 'string' }, cpu: { type: 'string' } }, required: ['cuda_sm86', 'cpu'] },
    tests: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, command: { type: 'string' }, result: { type: 'string' } }, required: ['name', 'result'] } },
    integration_notes: { type: 'string' },
  },
  required: ['branch', 'head_sha', 'commits', 'findings', 'files_changed', 'builds', 'tests', 'integration_notes'],
}

const PEER_HEALTH_API = `Shared contract between groups: the peer-failure group adds a backend function  int dsv4_cuda_peer_healthy(int device);  (header + dsv4.def + Windows loader optional symbol with fallback 1) that returns 0 once a peer-side CUDA failure has latched that device off, else 1. The peer-failure group ALSO adds the engine-side consumer: at the top of v4_gpu_expert_attach_tiered and coli_v4_gpu_expert_peek, a latched peer pool is treated as absent (its mirrors are misses). The tiered-attach group must not add this check itself, but should keep the peer-pool access in those two functions easy to guard (one place where the peer cache pointer is read).`

const GROUPS = [
  { key: 'tiered-attach', ids: 'F3, F4, F5, F16, F37, F32, F56 (only: dead V4GpuExpertMirrorCache.clock field, the duplicate g_v4_peer_mirrors tentative definition, v4_bank_double_on external linkage)', task: `Rework the mirror-cache attach path in c/deepseek_v4.c so a peer pool can never make decode worse than having no pool. F3: exclude any cache that cannot recycle (fewer than the minimum live entries and not growing, or no live tensors) from victim selection; if the chosen victim's attach returns -1, retry with the other cache; treat any peer grow_only result other than 0/-2 as "peer unavailable for this attach". F4: on a failed upload, roll back the slot (decrement count if it was appended, else mark it free so the found scan cannot match it), clear the CUDA error (the backend upload path should not leave a sticky error — if that needs a backend change, keep it minimal and in the .cu), return -2 in grow_only mode; skip NULL-tensor entries in v4_mirror_lru_clock and in the found scan. F5: v4_peer_mirrors_create refuses capacity < 8 (and non-numeric DSV4_CUDA_PEER_MIRRORS) with a one-line "peer pool off" message and returns NULL; document the minimum of 8 for both *_MIRRORS variables. F16: make the growth-guard probe conservative between probes (subtract each new mirror's bytes from the cached free figure, or re-probe whenever free VRAM is within the probe margin of the reserve) in the shared function so both primary and peer benefit — keep PR #1's bank-derived primary reserve and probe-margin semantics. F37: per-cache "frozen below the VRAM reserve" warning naming the device, reserve_mb and the correct variable (DSV4_CUDA_PEER_RESERVE_MB for the peer). F32: peek scans the primary once (return primary hit, else peer). F56 parts listed. Prove the fixes with a host-only harness (stub the backend upload/free/mem_free functions) that replays the review's scenarios (frozen peer at count 1, failed uploads, capacity 0-7, reserve below probe overshoot) and shows the CPU-fallback fraction no worse than with no peer pool, and that no entry referenced by the current token is recycled (the #1538 hazard). ${PEER_HEALTH_API}` },
  { key: 'device-init', ids: 'F13, F14, F15, F33 (code part: skip/explain the peer pool under DSV4_HYBRID=1)', task: `Two-device startup in coli_v4_gpu_engine_open (c/deepseek_v4.c) and dsv4_cuda_init (c/backend_cuda_dsv4.cu). F13: a failed two-device dsv4_cuda_init must clean up inside init (per-device teardown helper that frees streams/events/handles/buffers, zeroes the Dev, clears g_peer, and calls cudaGetLastError()), and the warning must name the device that actually failed. Do NOT call dsv4_cuda_shutdown() from engine_open before the single-device retry (on Windows the loader's shutdown unloads the DLL). F14 + F15: probe the peer BEFORE the two-device init — device count/index valid, peer != primary, cudaDeviceCanAccessPeer in both directions, compute capability/arch acceptable, and that the build has a kernel image for it (e.g. cudaFuncGetAttributes on a kernel; measured to return 209/98 on unsupported devices) — via a small new backend probe function (add to header, dsv4.def, Windows loader optional symbol with fallback "no peer") so a refused peer is never initialised (no ~290 MB held on GPU1), and print the actual reason and a rebuild hint where relevant. Keep the primary-only path byte-for-byte unchanged when DSV4_CUDA_PEER_DEVICE is unset. F33 code part: under DSV4_HYBRID=1 the peer pool never fills (attach_async is primary-only) and VRAM-first is off — do not create the pool then; print a one-time explanation. Add a CUDA test for the probe on the test box (both GPUs visible, tiny memory).` },
  { key: 'peer-failure', ids: 'F17, F18, F19, F20, F21, F9, F38, F57', task: `Backend robustness of the peer path (c/backend_cuda_dsv4.cu) plus its engine-side consumer. F19: on any exit after the peer FC2 is queued, cudaSetDevice(peer), cudaStreamSynchronize(pc->stream), restore the primary device; clear errors on every early return of dsv4_cuda_moe's peer branch. F18: after the final primary sync, check the peer join (cudaEventQuery or cudaEventSynchronize(pc->join)) so an asynchronous peer fault cannot let a token through with wrong output. F17: latch the peer off after a peer-branch failure (new int dsv4_cuda_peer_healthy(int device)), log once which device failed and why, and make the engine treat the latched pool's mirrors as misses (guard at the top of v4_gpu_expert_attach_tiered and coli_v4_gpu_expert_peek in c/deepseek_v4.c) so decode keeps working on the primary; the failing token itself must still produce correct output (fall back to computing those experts on the primary only if their weights are still readable, otherwise via the existing host-data fallback). F20: add dev_ok(d, sg->device) (peer can write the primary) to the on_peer condition. F21: require matching SM major.minor between primary and peer for peer COMPUTE (otherwise fall back to the in-place-read mode, which runs the primary's kernels) — or document the precondition; prefer enforcing. F9: in the recycled-mirror refill path use cudaMemcpyAsync on the same stream in BOTH the pinned and pageable branches, then honour sync with a stream sync (a stream sync alone does not order a legacy-stream cudaMemcpy). F38: startup banner prints the active peer mode (compute vs in-place reads) and comments no longer claim in-place reads are the default. F57: clamp DSV4_CUDA_PIN_CAP_GB (finite, non-negative, sane max; empty = default 20 GiB) to avoid UB in the size_t conversion. Add a CUDA test (both GPUs, tiny) that exercises dsv4_cuda_moe with on-peer experts (bit-identical vs all-on-primary) and an injected peer failure (e.g. invalid peer tensor) showing the latch and correct fallback output. ${PEER_HEALTH_API}` },
  { key: 'vramfirst-pipeline', ids: 'F6, F24, F25, F26, F34, F35, F59, F60 (block-scope externs only), F56 (only: gpu_timed, v4_dec_now duplicate of v4_wait_clock)', task: `moe_token_pipeline, the expert store, and v4_serve_finish in c/deepseek_v4.c. F6: for each VRAM-first hit, record the store's routing accounting (rt_count and ehit/eheat — currently only updated in lookup_hot and in coli_v4_expert_store_note_routed, which is compiled only under COLI_VULKAN), batched once per token-layer, outside COLI_VULKAN, without touching stats.hits/misses; document what DONE hit% means with VRAM-first on. F24: exclude VRAM-mirrored experts from hot_repin_locked's ranking so RAM pins go to experts that are not in VRAM (heed the pitfalls in the report: do not drive note_routed for this, do not use a Vulkan-style ram_first); fix the docs/comment that say their RAM slot "goes cold". F25: lazy host fetches for GPU-only views (when the fused path cannot be used) go through the loader lanes in parallel instead of one at a time on the compute thread. F26: finish in-flight loader jobs on every exit path (pre-existing bug; ASan reproduced stack-use-after-return). F34: make the decode/VRAM-first counters accurate (hits counted only when served from VRAM, lazy fetches timed, scope: decode tokens only, dec_t0 before routing, 'attaches' counts uploads) or rename/document their scope. F35: v4_serve_turn line — name cumulative totals *_total, count store fetches unconditionally, document scope. F59: COLI_V4_EXPERIMENTAL_PREFETCH must not prefetch experts that will be served from VRAM. F60: move the block-scope externs (g_v4_gpu_first_*, g_v4_dec_*) into the internal header OUTSIDE the COLI_V4_GPU_TIER block (CPU-only builds must still compile and link). F56 parts listed. With V4_GPU_RESIDENT_FIRST unset, behaviour (lookups, leases, outputs) must stay exactly as before. Use ASan/fault-injection host harnesses where useful.` },
  { key: 'docs', ids: 'F39, F41 (peer-pool sizing rows), F33 (docs part), F38 (docs part)', task: `docs/deepseek-v4.md only (plus code comments if a doc statement points at them). F39: fix the multi-GPU row of the build/coverage table (decode text sits in the prefill column; GPUs and decode columns read "—") and remove the stale "multi-GPU design exists on paper only" sentence. F41: the peer-pool sizing text uses the real decode mirror size (12.75 MiB on generic builds; PR #1 already fixed the primary VRAM table — keep consistent). F33: document the interactions between the switches: under DSV4_HYBRID=1 VRAM-first is off and the peer pool is not used; DSV4_CUDA_PEER_COMPUTE has no effect with DSV4_CUDA_BATCHED=0; batched decode/verify batches use neither feature. F38: say that peer compute is the default and in-place P2P reads happen only with DSV4_CUDA_PEER_COMPUTE=0, DSV4_CUDA_BATCHED=0 or the expert_group fallback. Also: the decode measurement table must state its environment (COLI_CUDA_MOE_BATCH=1, COLI_CUDA_MOE_DOUBLE=1 were set on every row) and mark it as "to be re-measured" (the author will replace the numbers with a fresh 3-sample measurement on the final build) — keep the table structure. Do not edit the PR #1 sections except where they contradict PR #2.` },
]

async function implement(g) {
  return agent(`${COMMON}

YOUR GROUP: ${g.key}
Findings: ${g.ids}
Branch fix2/${g.key}; local worktree ${S}/wt/fix2-${g.key}; test-box build worktree key ${g.key}.
Task:
${g.task}

When done: commits pushed, CUDA build (after the benchmark) and CPU build clean, tests run. Return the structured result with integration_notes (regions touched that other groups may also touch, new symbols, contracts).`,
    { label: `impl:${g.key}`, phase: 'Implement', schema: RESULT_SCHEMA })
}

const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['approve', 'changes_requested'] },
    issues: { type: 'array', items: { type: 'object', properties: { severity: { type: 'string', enum: ['blocker', 'major', 'minor', 'nit'] }, file: { type: 'string' }, line: { type: 'integer' }, description: { type: 'string' }, fix: { type: 'string' } }, required: ['severity', 'description', 'fix'] } },
    finding_assessment: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, really_fixed: { type: 'boolean' }, note: { type: 'string' } }, required: ['id', 'really_fixed', 'note'] } },
    summary: { type: 'string' },
  },
  required: ['verdict', 'issues', 'finding_assessment', 'summary'],
}

async function review(g, res, round) {
  return agent(`${COMMON}

You are an independent, adversarial REVIEWER (round ${round}) for fix group "${g.key}" (findings ${g.ids}). Do not modify the branch. Inspect git -C ${CLONE} fetch -q origin && git -C ${CLONE} diff origin/pr2/split..origin/fix2/${g.key} and the commit messages; read the findings in ${REPORT} and the surrounding code. For each finding decide whether it is really fixed (the failure scenario can no longer happen) and whether the fix introduces a new bug, a behaviour change with the PR #2 switches off, a bit-identity break, a concurrency/ordering problem (streams/events across devices), missing wiring (CPU-only, Vulkan, DeepGEMM, Windows loader/dsv4.def/parity test), a build warning, or a style mismatch. You may build/test on the test box under the same rules (build worktree key review2-${g.key}; respect the benchmark wait). Request changes only for real problems.
Implementer's report:
${JSON.stringify(res)}`,
    { label: `review${round}:${g.key}`, phase: 'Review', schema: REVIEW_SCHEMA })
}

async function fixup(g, res, rev) {
  return agent(`${COMMON}

YOUR GROUP: ${g.key} (findings ${g.ids}). Branch fix2/${g.key} exists (worktree ${S}/wt/fix2-${g.key}); a reviewer requested changes. Address every blocker/major issue and each correct minor issue (explain rejections) as additional commits; rebuild CUDA (key ${g.key}) and CPU, re-test, push, return the updated structured result.
Previous result:
${JSON.stringify(res)}
Review:
${JSON.stringify(rev)}`,
    { label: `fixup:${g.key}`, phase: 'Fixup', schema: RESULT_SCHEMA })
}

const done = await pipeline(GROUPS,
  async (g) => implement(g),
  async (res, g) => {
    if (!res) return { g, res: null, rev: null }
    let rev = await review(g, res, 1)
    let cur = res
    if (rev && rev.verdict === 'changes_requested') {
      const fx = await fixup(g, cur, rev)
      if (fx) cur = fx
      rev = await review(g, cur, 2)
    }
    return { g, res: cur, rev }
  })
const ok = done.filter(Boolean)
for (const d of ok) log(`${d.g.key}: ${d.res ? d.res.head_sha : 'FAILED'} review=${d.rev ? d.rev.verdict : 'none'}`)

phase('Integrate')
const INTEG_SCHEMA = {
  type: 'object',
  properties: {
    branch: { type: 'string' }, head_sha: { type: 'string' },
    merged: { type: 'array', items: { type: 'string' } },
    conflicts: { type: 'array', items: { type: 'string' } },
    builds: { type: 'object', properties: { cuda_sm86: { type: 'string' }, cpu: { type: 'string' } }, required: ['cuda_sm86', 'cpu'] },
    make_check: { type: 'string' }, cuda_tests: { type: 'string' },
    open_problems: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
  required: ['branch', 'head_sha', 'merged', 'conflicts', 'builds', 'make_check', 'cuda_tests', 'open_problems', 'summary'],
}
const integ = await agent(`${COMMON}

You are the INTEGRATOR. NOTE the benchmark has finished; the wait rule no longer applies. Three steps.
STEP A: create branch pr2/fixed: git -C ${CLONE} fetch -q origin && git -C ${CLONE} worktree add -b pr2/fixed ${S}/wt/pr2-fixed origin/pr2/split (if a stale worktree/branch of that name exists from an interrupted attempt, remove it first). Cherry-pick every fix branch's commits (linear history), resolving conflicts correctly (read both sides; several groups touch c/deepseek_v4.c mirror-cache/engine-open code, c/backend_cuda_dsv4.cu, the loader, dsv4.def and docs; honour the dsv4_cuda_peer_healthy contract).
STEP B: the device-init group's final review requested one minor change that was never made — apply it yourself as an extra commit: in dsv4_cuda_peer_probe (c/backend_cuda_dsv4.cu) add the "rebuild with CUDA_ARCH=..." / DeepGEMM "sm_120a code only" hint ONLY when the error is cudaErrorNoKernelImageForDevice or cudaErrorInvalidDeviceFunction; for any other error (e.g. out of memory on a nearly full peer, devices unavailable) print just role, device, name, sm and cudaGetErrorString. Extend the device-init group's peer-probe test if it has a fail-mode shim (e.g. a real non-image runtime error) and check the reason names the device and does not contain "rebuild".
STEP C: PR #1 gained one more commit after PR #2 was split: origin/pr1/fixed is now a406901b "deepseek-v4: order host->device uploads on the compute stream" (on top of 48767c01, which pr2/split is based on). It replaces every legacy host->device cudaMemcpy in c/backend_cuda_dsv4.cu with a new helper h2d() (cudaMemcpyAsync on c->stream + cudaStreamSynchronize) and makes dsv4_cuda_tensor_refill_fp4 use cudaMemcpyAsync on c->stream in BOTH the pinned and pageable branches (then sync if requested) — this was a real race (decode read half-written mirrors from pageable slabs), verified fixed on the GPU. Rebase the whole PR #2 series (pr2/split's 7 commits + the fix commits + your STEP B commit) onto origin/pr1/fixed: git rebase --onto origin/pr1/fixed 48767c01 pr2/fixed. Resolve conflicts so PR #1's h2d/refill code is kept exactly; the peer-failure group's F9 change to the same refill function becomes redundant — keep PR #1's version, and keep any genuinely additional PR #2 behaviour around it (e.g. peer-health checks) on top. Make sure no new legacy host->device cudaMemcpy is introduced by PR #2 code (any such copy whose data a kernel on c->stream later reads must use h2d or cudaMemcpyAsync on that stream). Then: CUDA build on the test box (key pr2-fixed, after the benchmark) with 0 warnings; local CPU build 0 warnings; full 'make check' locally from the worktree root (10-15 min) compared with the pristine dev baseline /tmp/cw/check-devbase.log on the test box (2366 tests OK, 253 skipped, exit 0) and with PR #1's result (2379 OK) — nothing that passes there may fail here; run all new dsv4 CUDA tests from both PRs on the test box (tiny memory, both GPUs allowed under the limits). Fix integration breakage with an extra commit. Push pr2/fixed. Return the structured result with open problems.
Fix branches with final reports and reviews:
${JSON.stringify(ok.map(d => ({ group: d.g.key, findings: d.g.ids, result: d.res, review: d.rev })))}`,
  { label: 'integrate', phase: 'Integrate', schema: INTEG_SCHEMA })

return { groups: ok.map(d => ({ key: d.g.key, result: d.res, review: d.rev })), integration: integ }
