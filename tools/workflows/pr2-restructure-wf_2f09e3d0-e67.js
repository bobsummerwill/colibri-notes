export const meta = {
  name: 'pr2-restructure',
  description: 'Rebase the PR #2 two-GPU decode work onto pr1/fixed and split it into clean commits',
  phases: [
    { title: 'Restructure', detail: 'reference rebase + logical commit split' },
    { title: 'Review', detail: 'tree equality, per-commit builds, messages' },
    { title: 'Fixup', detail: 'address review issues' },
  ],
}

const S = '/home/zombietiger/.config/Claude/scratch-workspaces/c9774333-2509-4b09-a6db-1980582ceec7/32aceebc-9fe5-4957-8332-03b2bcab77b9/scratch-2026-10-13-2b21f7'
const CLONE = `${S}/colibri-local`
const REPORT = `${S}/review/REVIEW_REPORT.md`
const CONTEXT = `${S}/review/REVIEW_CONTEXT.md`

const COMMON = `Repository: local git clone ${CLONE} (remote origin = maplemercury@shadowfax:/tmp/colibri). Run git -C ${CLONE} fetch -q origin first.
Branches on origin:
- pr1/fixed (48767c01): upstream dev + PR #1 (REAP prefill) + 13 review-fix commits. FINAL; do not change it.
- pr1/reap-prefill (869aa82c): PR #1 before its review fixes.
- pr2/two-gpu-decode (0caeac70): pr1/reap-prefill + 3 PR #2 commits: 8c5dfa7b "V4_SERVE_STATS_LOG per-turn decode stats", 950f0c4b "wip: VRAM-first decode, decode timers, pin cap, peer mirror pool (remote reads)", 0caeac70 "decode peer compute on the second GPU, docs".
Background on what PR #2 does: ${CONTEXT} (items 6-9; line numbers there are from an older tree). Review findings: ${REPORT} (do NOT fix findings now — a later step does that; this task is only rebasing and restructuring).
Rules: work only in your own local worktree (git -C ${CLONE} worktree add ...). Commit with the clone's identity (Bob Summerwill <bob@summerwill.net>), message style "deepseek-v4: <what>" with an explanatory body, ending with the trailer "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>". Do NOT ssh to the test box or build there (a GPU benchmark is running on it); only git push to origin is allowed. Builds/tests locally only: CPU-only engine build make -C <wt>/c -f Makefile.deepseek-v4 deepseek-v4 -j8 must have 0 warnings, and quick syntax checks of the GPU units with gcc -fsyntax-only -DCOLI_V4_GPU_TIER on the relevant -DCOLI_V4_UNIT_* (look at Makefile.deepseek-v4 for the unit list/flags) to catch errors in GPU-tier C code; the .cu cannot be compiled locally (no nvcc) so review .cu conflict resolutions by careful reading.`

const RS_SCHEMA = {
  type: 'object',
  properties: {
    ref_branch: { type: 'string' }, ref_sha: { type: 'string' },
    split_branch: { type: 'string' }, split_sha: { type: 'string' },
    commits: { type: 'array', items: { type: 'object', properties: { sha: { type: 'string' }, subject: { type: 'string' }, cpu_build: { type: 'string' } }, required: ['sha', 'subject', 'cpu_build'] } },
    tree_equal: { type: 'boolean' },
    conflict_resolutions: { type: 'array', items: { type: 'string' } },
    tests: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
  required: ['ref_branch', 'ref_sha', 'split_branch', 'split_sha', 'commits', 'tree_equal', 'conflict_resolutions', 'tests', 'notes'],
}

phase('Restructure')
const rs = await agent(`${COMMON}

TASK, two steps.
STEP 1 (reference): create branch pr2/ref from origin/pr1/fixed and replay the 3 PR #2 commits onto it (git cherry-pick 8c5dfa7b 950f0c4b 0caeac70), resolving every conflict correctly against the PR #1 fixes. Known interactions to handle:
 (a) PR #1 replaced the fixed decode-mirror VRAM reserve with a bank-derived one (v4_gpu_mirror_reserve_mb(...), v4_bank_double_usable(), globals computed at engine open). PR #2 moved growth-guard state into per-cache fields (reserve_mb, last_free_mb, probes, unified) so a peer pool on another GPU can have its own reserve. Resolve so the PRIMARY cache keeps PR #1's bank-derived reserve semantics exactly (evaluated the way PR #1 does it, including the bank2-usable dependency — do not freeze it into a field at open if PR #1 computes it per call) while the PEER cache uses its own reserve (DSV4_CUDA_PEER_RESERVE_MB, default 1024) and per-cache probe state. Keep PR #1's probe-margin / minimum-live-mirror logic for both caches where it applies.
 (b) PR #1 added a dsv4 export parity test (c/tests/test_backend_loader_dsv4_parity.py): every dsv4_cuda_* in c/backend_cuda_dsv4.h must be in c/dsv4.def and resolved/wrapped by c/backend_loader_dsv4.c. PR #2 adds dsv4_cuda_peer_access — add it to dsv4.def in the commit that introduces it so the parity test passes at every commit.
 (c) PR #1 rewrote dsv4_cuda_shutdown to free buffers and memset each Dev; PR #2's g_peer matrix must be reset there too (and anything PR #2 adds to Dev freed).
 (d) docs/deepseek-v4.md env table / sections, c/backend_cuda_dsv4.cu Dev struct and init, and the mirror cache code in c/deepseek_v4.c.
 Verify pr2/ref: CPU build 0 warnings, GPU-unit syntax checks, and python3 -m unittest of the dsv4 parity test and other loader/header tests from c/. Push pr2/ref.
STEP 2 (split): create branch pr2/split from origin/pr1/fixed whose FINAL TREE IS IDENTICAL to pr2/ref (git diff pr2/ref pr2/split must be empty), as a series of focused, bisectable commits replacing the 3 messy ones (the 'wip' commit bundles unrelated things). Suggested order (adjust if dependencies require):
  1. pinned-host cap (DSV4_CUDA_PIN_CAP_GB) — backend only;
  2. per-turn decode diagnostics: V4_SERVE_STATS_LOG line and the decode MoE timers (v4_dec_moe) — may need to precede or follow VRAM-first depending on which fields reference VRAM-first counters (the review noted commit 3's fields come from the wip commit — make each commit self-consistent);
  3. VRAM-first decode (V4_GPU_RESIDENT_FIRST) in moe_token_pipeline;
  4. per-cache growth-guard state + shared LRU clock + grow_only/tiered attach + peer mirror pool on a second GPU with in-place P2P reads (DSV4_CUDA_PEER_DEVICE, dsv4_cuda_peer_access, dsv4.def, Windows loader, two-device init);
  5. peer compute in dsv4_cuda_moe (DSV4_CUDA_PEER_COMPUTE);
  6. docs.
 Each commit must build CPU-only with 0 warnings, pass the GPU-unit syntax checks, and keep the dsv4 parity test passing. Each commit message: what, why, how it is switched on/off, and that it is off by default where applicable. Push pr2/split. Report the per-commit build results and confirm tree equality.`,
  { label: 'restructure', phase: 'Restructure', schema: RS_SCHEMA })

phase('Review')
const RV_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['approve', 'changes_requested'] },
    issues: { type: 'array', items: { type: 'object', properties: { severity: { type: 'string', enum: ['blocker', 'major', 'minor', 'nit'] }, where: { type: 'string' }, description: { type: 'string' }, fix: { type: 'string' } }, required: ['severity', 'description', 'fix'] } },
    summary: { type: 'string' },
  },
  required: ['verdict', 'issues', 'summary'],
}
async function reviewRs(r, round) {
  return agent(`${COMMON}

You are an independent adversarial REVIEWER (round ${round}) of a rebase + commit restructure. Do not modify the branches. Check:
1. Conflict resolutions in pr2/ref are correct: compare git diff origin/pr1/reap-prefill origin/pr2/two-gpu-decode (the original PR #2 change) with git diff origin/pr1/fixed origin/pr2/ref (the replayed change). Every original PR #2 behaviour must be preserved, and every PR #1 fix (git log origin/pr1/reap-prefill..origin/pr1/fixed) must still be in effect — in particular the bank-derived primary reserve, probe margin/minimum-live-mirror logic, release/lifecycle fixes, shutdown memset, parity test, routing fixes. Read the .cu resolutions carefully (it cannot be compiled locally).
2. git diff origin/pr2/ref origin/pr2/split is empty.
3. Each commit of origin/pr1/fixed..origin/pr2/split builds CPU-only with 0 warnings and passes the GPU-unit syntax checks and the dsv4 parity test (actually run these per commit in your own worktree), and is self-consistent and focused; messages accurate.
Restructure report:
${JSON.stringify(r)}`,
    { label: `review${round}`, phase: 'Review', schema: RV_SCHEMA })
}
let rv = rs ? await reviewRs(rs, 1) : null
let cur = rs
if (rv && rv.verdict === 'changes_requested') {
  phase('Fixup')
  const fx = await agent(`${COMMON}

Fix the issues an independent reviewer found in your rebase/restructure (branches pr2/ref and pr2/split, your worktrees under ${S}/wt/). You MAY rewrite these two branches (force-push) since they are unpublished. Keep the invariant: pr2/split's final tree identical to pr2/ref, every commit builds CPU-only with 0 warnings and passes syntax checks + parity test. Return the updated report.
Previous report:
${JSON.stringify(rs)}
Review:
${JSON.stringify(rv)}`,
    { label: 'fixup', phase: 'Fixup', schema: RS_SCHEMA })
  if (fx) cur = fx
  rv = await reviewRs(cur, 2)
}
return { restructure: cur, review: rv }
