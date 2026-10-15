export const meta = {
  name: 'three-pr-restructure',
  description: 'Restructure the colibri work into PR0 (upstream bug fixes), PR1 (REAP prefill) and PR2 (two-GPU decode) with clean logical commits',
  phases: [
    { title: 'PR0', detail: 'standalone upstream bug fixes on dev' },
    { title: 'PR1', detail: 'REAP prefill as logical commits on PR0, tree == pr1/fixed' },
    { title: 'PR2', detail: 'two-GPU decode as logical commits on PR1, tree == pr2/fixed' },
    { title: 'Review', detail: 'independent review of each PR' },
    { title: 'Fixup', detail: 'address review issues' },
  ],
}

const S = '/home/zombietiger/.config/Claude/scratch-workspaces/c9774333-2509-4b09-a6db-1980582ceec7/32aceebc-9fe5-4957-8332-03b2bcab77b9/scratch-2026-10-13-2b21f7'
const CLONE = `${S}/colibri-local`
const REPORT = `${S}/review/REVIEW_REPORT.md`
const DEV = '7b55a9b3792527f52d1a2337fd9e312b9a7983e3'
const PR1F = 'a406901b'   // origin/pr1/fixed: final PR #1 code (dev + 18 commits)
const PR2F = 'fe28d80a'   // origin/pr2/fixed: final PR #2 code (pr1/fixed + 33 commits)

const COMMON = `Project: colibri (https://github.com/JustVugg/colibri), C/CUDA MoE inference engine; DeepSeek V4 engine (c/deepseek_v4.c compiled as many -DCOLI_V4_UNIT_* units, GPU code under COLI_V4_GPU_TIER; c/backend_cuda_dsv4.cu compact one-line CUDA style; c/backend_cuda_dsv4.h; Windows loader c/backend_loader_dsv4.c + c/dsv4.def with a parity test c/tests/test_backend_loader_dsv4_parity.py; docs/deepseek-v4.md).
Repository: local clone ${CLONE} (origin = maplemercury@shadowfax:/tmp/colibri; NOTHING goes to GitHub). Fetch first: git -C ${CLONE} fetch -q origin.
Reference branches (all on origin, all reviewed and validated — the CODE in them is final; only the history is being reorganised):
- upstream dev = ${DEV} (origin/dev may not exist in this clone; use the SHA).
- origin/pr1/fixed = ${PR1F}: dev + 18 commits (2 feature commits, 15 review-fix commits, 1 upload-ordering bug fix). git log --reverse ${DEV}..${PR1F}.
- origin/pr2/fixed = ${PR2F}: pr1/fixed + 33 commits (7 feature commits + fix/integration commits). git log --reverse ${PR1F}..${PR2F}.
Review findings referenced by id (F1..F60): ${REPORT}.
GOAL: three stacked branches, each a short series of coherent, bisectable, well-described commits:
- pr0/dsv4-fixes on dev: standalone fixes for bugs that exist in UPSTREAM dev code (independent of the new features).
- pr1/reap-prefill on pr0/dsv4-fixes: final tree IDENTICAL to ${PR1F} (git diff ${PR1F} <tip> must be empty).
- pr2/two-gpu-decode on pr1/reap-prefill: final tree IDENTICAL to ${PR2F}.
Rules:
- Work only in your own local worktrees under ${S}/wt/ (git -C ${CLONE} worktree add ...). Local branches may be rewritten freely before you push; push with git push -f origin <branch>.
- Commit identity = the clone's (Bob Summerwill <bob@summerwill.net>); every commit message: subject "deepseek-v4: <what>" (or "test(dsv4): ..."), a body explaining what, why (failure scenario for bug fixes), how it is switched on/off where relevant, how it was verified, the finding ids where applicable; trailer "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>". No "wip", no "review nits", no "integration" commits in the final series — fold such content into the commit it belongs to.
- Every commit must build: local CPU-only engine build (make -C <wt>/c -f Makefile.deepseek-v4 deepseek-v4 -j8, 0 warnings; run make -C <wt>/c -f Makefile.deepseek-v4 deepseek-v4-clean when switching commits), gcc -fsyntax-only -Wall -Wextra -DCOLI_V4_GPU_TIER on the units it touches (see Makefile.deepseek-v4), the dsv4 parity test, and — for commits touching c/backend_cuda_dsv4.cu — a CUDA compile on the test box: git push the branch, then ssh maplemercury@shadowfax 'cd /tmp/colibri && (git worktree add -q --detach /tmp/cw/r3-<key> <sha> 2>/dev/null || git -C /tmp/cw/r3-<key> checkout -q --detach -f <sha>) && cd /tmp/cw/r3-<key>/c && make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_HOME=/usr NVCC=/usr/bin/nvcc CUDA_ARCH=sm_86 NO_TC=1 -j8 2>&1 | grep -iE "error|warning" | grep -v lto'  (0 warnings/errors). Never put single quotes inside the ssh '...' string. Do not stop services, run the model, or touch /tmp/colibri-vf, /tmp/cw/pr*, /tmp/cw/bench-*, /tmp/cw/dbg*, /tmp/cw/devbase on the test box. Tiny CUDA tests (< 500 MB, CUDA_VISIBLE_DEVICES=0 or 0,1) are allowed.
- At each branch tip run the full 'make check' locally from the worktree root (~14 min); baseline: upstream dev passes 2366 tests (253 skipped), pr1/fixed 2379, pr2/fixed 2384 — nothing that passes on the reference may fail.`

const BR_SCHEMA = {
  type: 'object',
  properties: {
    branch: { type: 'string' }, tip: { type: 'string' },
    commits: { type: 'array', items: { type: 'object', properties: { sha: { type: 'string' }, subject: { type: 'string' }, checks: { type: 'string' } }, required: ['sha', 'subject', 'checks'] } },
    tree_check: { type: 'string' },
    make_check: { type: 'string' },
    deviations: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
  required: ['branch', 'tip', 'commits', 'tree_check', 'make_check', 'deviations', 'notes'],
}
const RV_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['approve', 'changes_requested'] },
    issues: { type: 'array', items: { type: 'object', properties: { severity: { type: 'string', enum: ['blocker', 'major', 'minor', 'nit'] }, where: { type: 'string' }, description: { type: 'string' }, fix: { type: 'string' } }, required: ['severity', 'description', 'fix'] } },
    summary: { type: 'string' },
  },
  required: ['verdict', 'issues', 'summary'],
}

const PR0_TASK = `Build branch pr0/dsv4-fixes from ${DEV}: one commit per upstream bug, each adapted to dev's code (dev lacks the new features, so port the FIX, not the feature around it), each independently correct and independently revertible, in this order:
 1. Unordered host->device uploads (the decode-mirror race): port ${PR1F}'s h2d() helper and the refill change (a406901b) to dev's upload()/dsv4_cuda_tensor_refill_fp4/expert table/hash router/KV RoPE/TP2 uploads (dev's upload() uses ok(cudaMalloc...) where PR #1 has alloc_ok). This was verified on the GPU to remove nondeterministic decode drift; say so.
 2. Stale CUDA error after a failed expert-bank cudaMalloc (the (void)cudaGetLastError() in dsv4_cuda_expert_bank_create's failure path, from 869aa82c) — failure scenario: the next unrelated launch check reports the OOM and a chunk drops to the CPU union.
 3. Prefill bank not released on cancel/failure and never at engine close (F10/F11, PR #1 commit 6fa0b85c and its source test a02595f7): release on every generate() exit and at the top of coli_v4_gpu_engine_close; adapt to dev (no in-prefill flag from F2; no pruned-model path).
 4. Router bias silently ignored when the bias GPU mirror is missing (F23): on dev the batched GPU router path has no CPU router for non-hash layers, so the dev fix is: refuse the GPU union (return -1 so the CPU union runs) when a raw router bias exists but its mirror does not. (PR #1 later replaces this with routing on the CPU — make sure the PR #1 series transforms it into ${PR1F}'s final code.)
 5. dsv4_cuda_shutdown leaves freed device pointers in g[] (F28 pre-existing part): free what dev allocates and memset each Dev after freeing (from 58945580, without PR #1-only buffers).
 6. c/dsv4.def does not export dsv4_cuda_device_unified (pre-existing part of F1, from d0c33310) + the dsv4 export parity test c/tests/test_backend_loader_dsv4_parity.py (adapted so it passes on dev: dev's header has no dsv4_cuda_expert_bank_slots).
 7. In-flight expert loader jobs are not finished when moe_token_pipeline exits on an error after preload (F26; stack-use-after-return reproduced with ASan): port the fix from ${PR2F}'s commit "finish in-flight expert loads on every MoE exit" to dev's moe_token_pipeline (no VRAM-first fetch[] list there).
 8. Docs: the pre-existing wrong figures in docs/deepseek-v4.md and comments: decode mirror size (~8 MB -> 12.75 MiB on generic builds) and the per-card VRAM table's mirror counts recomputed from the real sizes; the 256-slot bank size (3.2 GB, ~3264 MiB) where dev says ~2.2 GB. Only statements about upstream behaviour — nothing about PR #1/#2 features.
 If any item turns out not to be reproducible/applicable on dev, say so and drop it (deviation). Verify: each commit builds as specified; parity test; make check at the tip; for item 7 a small ASan harness or the PR #2 test if it can run against dev; for items 1-3 and 5 at least a CUDA compile and the tiny dsv4 CUDA tests that exist on dev. Push pr0/dsv4-fixes.`

const PR1_TASK = (pr0) => `Build branch pr1/reap-prefill on top of origin/pr0/dsv4-fixes (${pr0}) whose final tree is IDENTICAL to ${PR1F}. Fold the 18 pr1/fixed commits, minus what PR #0 already contains, into a short series of coherent commits, for example (adjust to real dependencies; every commit must build and pass the parity test):
 1. bank sized to the routed-expert count (dsv4_cuda_expert_bank_slots, dsv4_cuda_expert_bank_slot_bytes, dsv4.def + loader + parity, bank-bytes CUDA test);
 2. batched GPU prefill for expert-pruned checkpoints (1..256 experts; CPU routing in parallel and bit-identical, incl. the Vulkan gate-logit path and the missing-bias-mirror case now routed on the CPU — this commit transforms PR #0's refuse-on-missing-bias into the final CPU-routing form; gate mirror optional; n_routed loops; speculative-decode batches kept off the bank (F2); bank2 worker bounded by the expert count; second bank retried at the next prefill (F12); lifecycle source test);
 3. decode-mirror VRAM reserve derived from the bank size (F7/F8) + the related docs;
 4. expert-major fp4 MoE kernel (default on, VERIFY mode, TB by route density, DSV4_MOE_EM_TB validation) + its part of the bitwise CUDA test;
 5. warp-per-row dense batch matmul (auto for I<=1024, DSV4_MM_WARP parsing, VERIFY) + its part of the bitwise test;
 6. diagnostics (V4_MOE_BATCH_PROF incl. switch=) and CUDA hygiene that belongs to these features (per-device verify buffers, freeing the new buffers in shutdown, alloc_ok, verify-disabled messages, include, DeepGEMM guards);
 7. docs for the features (REAP/generic-kernel section, env rows, measurements).
Verify tree equality: git diff ${PR1F} pr1/reap-prefill is empty. Push pr1/reap-prefill.`

const PR2_TASK = (pr1) => `Build branch pr2/two-gpu-decode on top of origin/pr1/reap-prefill (${pr1}) whose final tree is IDENTICAL to ${PR2F}. Fold the 33 commits ${PR1F}..${PR2F}, minus what PR #0 already contains (the F26 loader-job fix), into a short coherent series, for example:
 1. DSV4_CUDA_PIN_CAP_GB (with clamping);
 2. per-turn decode stats V4_SERVE_STATS_LOG and decode MoE counters/timers (accurate, *_total naming);
 3. VRAM-first decode lookups V4_GPU_RESIDENT_FIRST (store routing accounting, RAM pins skip mirrored experts, lazy fetches through the loader lanes, prefetch only what the store reads, off without the dual loader);
 4. per-cache mirror growth-guard state and one LRU clock (refactor, no behaviour change) — if the final growth-guard behaviour changes (F16: stop at the reserve from 8 live mirrors on; failed upload rollback F4) belong here or in 5, put them where the description is honest about the behaviour change with the PR #2 switches off;
 5. peer mirror pool on a second GPU DSV4_CUDA_PEER_DEVICE: probe before init (dsv4_cuda_peer_probe), init rollback, tiered attach that never makes decode worse (F3/F4/F5/F16/F37/F32), hybrid skip, dsv4_cuda_peer_access, loader/def, peer-init CUDA test;
 6. peer compute DSV4_CUDA_PEER_COMPUTE (same-SM + write-back checks, mode banner, peer failure latch dsv4_cuda_peer_healthy / dsv4_cuda_peer_compute, lost-peer fallback, peer MoE CUDA test);
 7. docs (two-GPU section, switch interactions, env rows; the decode table marked as to be re-measured).
Verify tree equality: git diff ${PR2F} pr2/two-gpu-decode is empty. Push pr2/two-gpu-decode.`

async function build(label, phaseName, task) {
  return agent(`${COMMON}\n\nTASK (${label}):\n${task}\nReturn the structured result: per-commit checks, tree equality check, make check result, and any deviations from the plan with reasons.`,
    { label, phase: phaseName, schema: BR_SCHEMA })
}
async function review(label, res, focus) {
  return agent(`${COMMON}\n\nYou are an independent adversarial REVIEWER of the branch ${res ? res.branch : '?'} (${label}). Do not modify branches. Check: ${focus}\nActually run the per-commit build/syntax/parity checks yourself on a sample of commits (at least every commit that touches c/backend_cuda_dsv4.cu or c/deepseek_v4.c's GPU unit) and the tree-equality check. Request changes only for real problems.\nBuilder's report:\n${JSON.stringify(res)}`,
    { label: `review:${label}`, phase: 'Review', schema: RV_SCHEMA })
}
async function fixup(label, res, rev, task) {
  return agent(`${COMMON}\n\nTASK (${label}, fix-up): an independent reviewer requested changes to the branch you built. Address every blocker/major and each correct minor issue (explain rejections); you may rewrite the branch (force-push). Keep every invariant of the original task:\n${task}\nPrevious result:\n${JSON.stringify(res)}\nReview:\n${JSON.stringify(rev)}`,
    { label: `fixup:${label}`, phase: 'Fixup', schema: BR_SCHEMA })
}
async function stage(label, phaseName, task, focus) {
  let res = await build(label, phaseName, task)
  if (!res) return { res: null, rev: null }
  let rev = await review(label, res, focus)
  if (rev && rev.verdict === 'changes_requested') {
    const fx = await fixup(label, res, rev, task)
    if (fx) res = fx
    rev = await review(label, res, focus)
  }
  log(`${label}: ${res.branch} @ ${res.tip} review=${rev ? rev.verdict : 'none'}`)
  return { res, rev }
}

phase('PR0')
const p0 = await stage('pr0', 'PR0', PR0_TASK, `(1) each commit fixes a bug that really exists in upstream dev ${DEV} (read dev's code; reproduce or reason concretely), is correct and minimal on dev, independent of the others (could be reverted alone), and does not drag in PR #1/#2 feature code; (2) the adapted versions are faithful to the validated fixes in ${PR1F}/${PR2F}; (3) messages describe the failure scenario accurately; (4) per-commit builds, parity test, make check at the tip.`)
if (!p0.res) return { error: 'pr0 failed' }

phase('PR1')
const t1 = PR1_TASK(p0.res.tip)
const p1 = await stage('pr1', 'PR1', t1, `(1) git diff ${PR1F} <tip> is empty; (2) the series is coherent (each commit one logical change, messages accurate, nothing from PR #0 duplicated, PR #0's interim forms correctly transformed); (3) every commit builds and passes the parity test; (4) make check at the tip.`)
if (!p1.res) return { pr0: p0, error: 'pr1 failed' }

phase('PR2')
const t2 = PR2_TASK(p1.res.tip)
const p2 = await stage('pr2', 'PR2', t2, `(1) git diff ${PR2F} <tip> is empty; (2) coherent series with honest messages (especially behaviour changes when the PR #2 switches are off); (3) every commit builds (CPU, GPU-unit syntax, CUDA for .cu commits) and passes the parity test; (4) make check at the tip.`)

return { pr0: p0, pr1: p1, pr2: p2 }
