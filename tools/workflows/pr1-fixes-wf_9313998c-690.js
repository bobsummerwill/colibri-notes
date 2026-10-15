export const meta = {
  name: 'pr1-fixes',
  description: 'Implement, review and integrate the PR #1 (REAP prefill) review fixes for colibri',
  phases: [
    { title: 'Implement', detail: '5 fix groups, each in its own worktree/branch' },
    { title: 'Review', detail: 'independent adversarial review per group' },
    { title: 'Fixup', detail: 'address review issues, re-review' },
    { title: 'Integrate', detail: 'merge all groups, full builds + tests' },
  ],
}

const S = '/home/zombietiger/.config/Claude/scratch-workspaces/c9774333-2509-4b09-a6db-1980582ceec7/32aceebc-9fe5-4957-8332-03b2bcab77b9/scratch-2026-10-13-2b21f7'
const CLONE = `${S}/colibri-local`
const REPORT = `${S}/review/REVIEW_REPORT.md`
const CONTEXT = `${S}/review/REVIEW_CONTEXT.md`
const BASE = '869aa82c'   // tip of pr1/reap-prefill (commits 1-2 rebased on upstream dev)

const COMMON = `You are fixing review findings in colibri (a C/CUDA MoE inference engine), DeepSeek V4 engine, PR #1 = "GPU batched prefill for expert-pruned (REAP) checkpoints + faster generic CUDA kernels".
Background: ${CONTEXT} (what the patches do; NOTE its line numbers and paths refer to an older tree — search by content). The full review with every finding (F1..F60), failure scenarios and suggested fixes: ${REPORT}. Read the findings assigned to you there in full before coding.
Repository: local git clone at ${CLONE} (remote "origin" = the test box repo maplemercury@shadowfax:/tmp/colibri). PR #1 is branch origin/pr1/reap-prefill at ${BASE} (2 commits on upstream dev 7b55a9b3). Key files: c/deepseek_v4.c (21k lines, compiled as many units via -DCOLI_V4_UNIT_<NAME>; GPU code under COLI_V4_GPU_TIER), c/backend_cuda_dsv4.cu (compact one-line CUDA style), c/backend_cuda_dsv4.h, c/backend_loader_dsv4.c (Windows DLL loader), c/dsv4.def, docs/deepseek-v4.md, c/tests/, tests/.
Workflow rules:
- Work ONLY in your own git worktree. Edit files locally with the normal Read/Edit/Write tools (do not edit code over ssh). Commit with the clone's configured identity (Bob Summerwill) — subject "deepseek-v4: <what>" (or "test(dsv4): ..."), a body explaining what and why with the finding ids, and the trailer line "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>".
- CUDA builds only exist on the test box (no nvcc locally). To build: git push -f origin <your-branch>; then
  ssh maplemercury@shadowfax 'cd /tmp/colibri && (git worktree add -q --detach /tmp/cw/b-<key> <your-branch> 2>/dev/null || git -C /tmp/cw/b-<key> checkout -q --detach -f <your-branch>) && cd /tmp/cw/b-<key>/c && make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j6 2>&1 | grep -iE "error|warning" | grep -v lto'
  The build must produce 0 warnings and 0 errors (lto-wrapper notes are fine). Never use ssh heredocs containing single quotes; push code instead.
- CPU-only build locally (must be 0 warnings): make -C <worktree>/c -f Makefile.deepseek-v4 deepseek-v4 -j8. Also run any relevant existing C unit tests / Python tests for the code you touch, locally.
- GPU tests on the test box: only small tests (< 1 GB GPU memory) with CUDA_VISIBLE_DEVICES=0. A production llama.cpp server is using both GPUs: NEVER stop/restart services, never run the engine on the real model, never touch /tmp/colibri-vf, /tmp/cw/pr1, /tmp/cw/pr2 or /tmp/cw/devbase.
- Kernel changes must keep results bit-identical (the PR's central claim); verify with a test when you touch a kernel or its launch choice.
- Stay in scope: fix your assigned findings, match the surrounding code style and comment density, no unrelated refactors. If a suggested fix is wrong or you choose differently, say why. If a finding belongs to PR #2 code that does not exist on this branch, mark it not_applicable.`

const RESULT_SCHEMA = {
  type: 'object',
  properties: {
    branch: { type: 'string' },
    head_sha: { type: 'string' },
    commits: { type: 'array', items: { type: 'object', properties: { sha: { type: 'string' }, subject: { type: 'string' } }, required: ['sha', 'subject'] } },
    findings: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, status: { type: 'string', enum: ['fixed', 'partial', 'deferred', 'not_applicable'] }, summary: { type: 'string' } }, required: ['id', 'status', 'summary'] } },
    files_changed: { type: 'array', items: { type: 'string' } },
    builds: { type: 'object', properties: { cuda_sm86: { type: 'string' }, cpu: { type: 'string' } }, required: ['cuda_sm86', 'cpu'] },
    tests: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, command: { type: 'string' }, result: { type: 'string' } }, required: ['name', 'result'] } },
    integration_notes: { type: 'string' },
  },
  required: ['branch', 'head_sha', 'commits', 'findings', 'files_changed', 'builds', 'tests', 'integration_notes'],
}

const GROUPS = [
  { key: 'exports-tests', ids: 'F1 (PR1 part only: add dsv4_cuda_expert_bank_slots and the already-missing dsv4_cuda_device_unified to c/dsv4.def; dsv4_cuda_peer_access does not exist on this branch), F49 (tests part only)', task: `1) F1: add the missing exports to c/dsv4.def and add a text-only parity test (model it on tests/test_backend_loader_header_parity.py, which covers only coli_cuda_*) that checks every dsv4_cuda_* function declared in c/backend_cuda_dsv4.h is exported by c/dsv4.def and resolved/wrapped by c/backend_loader_dsv4.c; wire it wherever the existing parity test is run (make check / python test discovery). 2) F49 tests: add a CUDA regression test that exercises the GENERIC (non-DeepGEMM) batched MoE entry dsv4_cuda_route_moe_ids_batch directly on a small synthetic expert bank (E small, H=4096, I=2048 as required by the entry point, a few chunk sizes including tiny/tail chunks) and checks that DSV4_MOE_EXPERT_MAJOR=0 and =1 and DSV4_MOE_EM_TB=2/3/4/6 produce bit-identical outputs (env vars are read once per process via statics, so run the test binary once per setting and compare hashes/dumps, e.g. a small driver script or a test that re-execs itself); and a test for the warp dense kernel through dsv4_cuda_matmul_batch with I<=1024 (e.g. 32768x1024 or smaller rows x 1024, fmt 8 and 9) comparing DSV4_MM_WARP=0 vs default vs 1 bitwise. Make the existing test_dsv4_moe_batch_cuda (which calls a DeepGEMM-only entry that is a return-0 stub on generic builds) skip cleanly with a message on generic builds instead of failing. Wire the new CUDA tests into the existing CUDA test target (find how 'make -C c cuda-test' is defined) and run them on the test box with CUDA_VISIBLE_DEVICES=0 (keep memory small: a handful of experts).` },
  { key: 'prefill-lifecycle', ids: 'F2, F10, F11, F12', task: `F2: add a dedicated "in prefill" flag (set around the generate() prefill segment loop, or pass the use_prefill_pool intent down) and make coli_v4_gpu_moe_batch_union refuse the GPU union, and never start the bank2 (COLI_CUDA_MOE_DOUBLE) prefetch, outside prefill — so speculative-verify batches during decode use the CPU path as before the patch. Do NOT implement it via coli_v4_gpu_moe_batch_hint(0): 0 means "unknown, allow" and the oracle/teacher-forcing callers rely on that; check every caller of the union/hint (oracle, TF, KV_SLOTS mux, verify/draft paths) and keep their behaviour. F10/F11: call coli_v4_gpu_moe_batch_release() on every exit path of generate() (success, cancel, every failure) and at the top of coli_v4_gpu_engine_close before the backend shutdown, so the DOUBLE prefetch worker is joined before teardown and bank_failed does not leak into the next request; make sure release is idempotent and safe when nothing was allocated. F12: reset the v4_bank2_disabled latch in release when the cause was an allocation failure (keep the structural no-aux-stream latch permanent) — coordinate with the reserve logic only by leaving a clearly named accessor if another group needs it (another group owns the VRAM reserve computation in the mirror growth guard; do not edit that block).` },
  { key: 'reserve-sizing-docs', ids: 'F7, F8, F40, F41, F60 (only the misaligned ternary in the reserve block)', task: `F7/F8: replace the fixed default mirror VRAM reserve (2800 MiB, +3300 MiB with COLI_CUDA_MOE_DOUBLE) in the decode mirror growth guard with one derived from the actual batched-MoE bank: dsv4_cuda_expert_bank_slots(n_routed_experts) x per-slot bytes (fc1+fc1 scales+fc2+fc2 scales; DeepGEMM adds its dg scale copies — compute from the same formulas as dsv4_cuda_expert_bank_create, ideally via a small new backend helper returning bytes per slot, or computed in C from H/I) x number of banks (2 with DOUBLE unless bank2 is disabled), plus attention/work headroom, plus a probe margin of 64 mirrors (the free-VRAM probe runs every 64 misses). Compute it at engine open (the attach path has no model config) into a global; drop the second-bank add-on once bank2 is latched off (read the existing v4_bank2_disabled state; another group may add a reset for it in release — do not change bank2 latch logic yourself). Cap the default so a minimum decode mirror budget (at least 8 live mirrors, ideally more) always fits on small cards, and keep DSV4_CUDA_VRAM_RESERVE_MB as the explicit override. Keep the non-batch default (600) behaviour. If a new backend function is added, add it to backend_cuda_dsv4.h, c/dsv4.def and the Windows loader (optional symbol with a sane fallback), mirroring how dsv4_cuda_expert_bank_slots is done. F40/F41: fix every stale bank-size and mirror-size statement in comments and docs/deepseek-v4.md (real sizes: 12.75 MiB per routed expert slot / decode mirror on generic builds, 3264 MiB for a 256-slot bank, 1683 MiB for 132 slots; recompute the per-card VRAM table's mirror counts from the real sizes and the dense ~6.3 GB; fix the reserve prose and the double-bank sentence). F60: fix the misaligned ternary in that reserve block.` },
  { key: 'cuda-hygiene', ids: 'F27, F28 (PR1 part: free emaj/emv1/emv2/emv4/emcount and the run_mm_batch verify buffers, then zero the Dev; g_peer does not exist here), F29, F30, F51, F52, F53, F54, F55, F57, F56 (only the redundant count_bit_mismatch forward declaration in the .cu)', task: `All in c/backend_cuda_dsv4.cu unless noted. F27: an alloc_ok()-style helper that clears the sticky CUDA error after a failed cudaMalloc in upload(), buf() and the verify-buffer allocations, and a one-time stderr message when a verify mode disables itself. F28: dsv4_cuda_shutdown frees the new expert-major buffers and verify buffers, and then memset(&g[i],0,sizeof g[i]) (zeroing is required — shutdown currently leaves freed pointers non-NULL). F29: move run_mm_batch's static verify buffers into struct Dev (per device). F30: choose the expert-major TB per call — TB=2 when route density is low (e.g. routes < ~2-3x the number of distinct experts in the chunk), else the configured default 4 — DSV4_MOE_EM_TB still forces a value; justify the threshold with a quick microbenchmark on the test box (GPU0, small) and confirm bit-identity across TB. F51: DSV4_MM_WARP parsing — treat unset, empty or "auto" as auto; "0" off; "1" force; document; VERIFY must not silently override an explicit 0 (decide and document). F52: #include <algorithm>. F53: guard run_fp4_expert_major and its kernels (and the mm_batch_w path if unreferenced) with #ifndef COLI_DSV4_DEEPGEMM so DeepGEMM builds have no #177-D unused warnings. F54: in expert-major mode do not build/size the unused MvDesc tables; remove the nullptr+1 arithmetic when the mode is off; validate DSV4_MOE_EM_TB (accept 2/3/4/6, warn once and fall back to 4 otherwise). F55: reword the warp-kernel comment and the docs line ("slower on 2048/4096-row shapes" is inaccurate: it is slower only for I=4096 and I=8192; 4096x2048 measured faster) — do not change the auto rule. F57: clamp DSV4_CUDA_PIN_CAP_GB (finite, 0..some sane max, empty = default 20) to avoid UB in the size_t conversion. F56: drop the redundant forward declaration of count_bit_mismatch.` },
  { key: 'routing-cpu', ids: 'F22, F23, F31, F36', task: `In coli_v4_gpu_moe_batch_union and helpers (c/deepseek_v4.c). F22: in COLI_VULKAN builds the new OpenMP-parallel CPU router bypasses the Vulkan gate-logit matmul hook for batches >= 16 (v4_vk_matmul_impl refuses inside a parallel region), so routing differs from the CPU union/decode there. Fix so the routing is consistent with the union in Vulkan builds too (e.g. when the Vulkan matmul hook is set, compute the gate logits once on the main thread for the whole chunk — or per token serially — and use coli_v4_route_bf16_logits; or keep the loop serial when the hook is active) and keep the non-Vulkan path unchanged and bit-identical; make the code comment and docs claim accurate. F23: when the 256-expert GPU router path has a raw router bias but the bias GPU mirror is missing, route on the CPU (as decode already does) instead of silently ignoring the bias. F31: v4_bank2_worker still loops over 256 experts — pass n_routed_experts (or the bank slot count) in the job and bound the loop. F36: the moebprof (V4_MOE_BATCH_PROF) line omits the bank2 join/switch wait taken before moeb_t0 — add a field for it.` },
]

async function implement(g) {
  return agent(`${COMMON}

YOUR GROUP: ${g.key}
Findings: ${g.ids}
Branch / worktree: create with  git -C ${CLONE} worktree add -b fix1/${g.key} ${S}/wt/fix1-${g.key} origin/pr1/reap-prefill  (if it exists already, reuse it). Test-box build worktree key: ${g.key}.
Task:
${g.task}

When done: all commits pushed (git push -f origin fix1/${g.key}), CUDA build on the test box and local CPU build clean, tests run. Return the structured result (findings statuses, commits, builds, tests, and integration_notes listing anything the integrator must know — e.g. functions/regions you touched that another group might also touch, new symbols needing exports).`,
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

You are an independent, adversarial REVIEWER (round ${round}) for fix group "${g.key}" (findings ${g.ids}). Do not modify the branch. Inspect: git -C ${CLONE} fetch -q origin && git -C ${CLONE} diff origin/pr1/reap-prefill..origin/fix1/${g.key} (and git log for the commit messages). Read the findings in ${REPORT} and the surrounding code. For each finding decide whether it is really fixed (the failure scenario can no longer happen) and whether the fix introduces any new bug, regression, behaviour change when the feature is off, bit-identity break, missing #ifdef/unit/export wiring (CPU-only, Vulkan, DeepGEMM, Windows loader/dsv4.def), build warning, or style mismatch. You may build/test on the test box under the same rules (use worktree key review-${g.key}). Request changes only for real problems; approve otherwise.
Implementer's report:
${JSON.stringify(res)}`,
    { label: `review${round}:${g.key}`, phase: 'Review', schema: REVIEW_SCHEMA })
}

async function fixup(g, res, rev) {
  return agent(`${COMMON}

YOUR GROUP: ${g.key} (findings ${g.ids}). Your branch fix1/${g.key} already exists (worktree ${S}/wt/fix1-${g.key}); an independent reviewer requested changes. Address every blocker/major issue and any minor issue that is correct (explain any you reject), as additional commits on the same branch (do not rewrite history). Rebuild CUDA on the test box (worktree key ${g.key}) and CPU locally, re-run the tests, push, and return the updated structured result.
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
    make_check: { type: 'string' },
    cuda_tests: { type: 'string' },
    open_problems: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
  required: ['branch', 'head_sha', 'merged', 'conflicts', 'builds', 'make_check', 'cuda_tests', 'open_problems', 'summary'],
}
const integ = await agent(`${COMMON}

You are the INTEGRATOR. Create branch pr1/fixed in a new worktree: git -C ${CLONE} fetch -q origin && git -C ${CLONE} worktree add -b pr1/fixed ${S}/wt/pr1-fixed origin/pr1/reap-prefill. Bring in every fix branch below (cherry-pick their commits in order, or merge — prefer a linear history of cherry-picks), resolving conflicts carefully and correctly (read both sides; several groups edited c/deepseek_v4.c, c/backend_cuda_dsv4.cu, c/dsv4.def, the Windows loader and docs/deepseek-v4.md). After integration: CUDA build on the test box (worktree key pr1-fixed) with 0 warnings; local CPU-only build with 0 warnings; run the full 'make check' locally from the worktree root (it takes 10+ minutes) and compare with the pristine upstream-dev baseline log on the test box at /tmp/cw/check-devbase.log (ssh maplemercury@shadowfax 'cat /tmp/cw/check-devbase.log'; wait for its final 'exit=' line if it is still running) — any test that passes on the baseline must pass on pr1/fixed; run the CUDA tests added by the exports-tests group on the test box (CUDA_VISIBLE_DEVICES=0, small memory). Fix any integration breakage with an extra commit. Push: git push -f origin pr1/fixed. Return the structured result, including open problems (e.g. a group that failed or whose review was not approved).
Fix branches and their final reports/reviews:
${JSON.stringify(ok.map(d => ({ group: d.g.key, findings: d.g.ids, result: d.res, review: d.rev })))}`,
  { label: 'integrate', phase: 'Integrate', schema: INTEG_SCHEMA })

return { groups: ok.map(d => ({ key: d.g.key, result: d.res, review: d.rev })), integration: integ }
