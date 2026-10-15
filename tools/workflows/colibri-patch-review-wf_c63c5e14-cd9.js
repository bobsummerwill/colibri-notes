export const meta = {
  name: 'colibri-patch-review',
  description: 'In-depth adversarial review of the colibri DeepSeek-V4 prefill + two-GPU decode patches',
  phases: [
    { title: 'Find', detail: '10 dimension reviewers over the diff and surrounding code' },
    { title: 'Dedup', detail: 'merge overlapping findings' },
    { title: 'Verify', detail: '3 independent lenses per finding, majority must confirm' },
    { title: 'Gaps', detail: 'completeness critic + extra reviewers until dry' },
    { title: 'Synthesize', detail: 'ranked report' },
  ],
}

const R = '/home/zombietiger/.config/Claude/scratch-workspaces/c9774333-2509-4b09-a6db-1980582ceec7/32aceebc-9fe5-4957-8332-03b2bcab77b9/scratch-2026-10-13-2b21f7/review'

const PREAMBLE = `You are one reviewer in an in-depth, adversarial code review of performance patches to colibri (a C/CUDA MoE inference engine).
Start by reading ${R}/REVIEW_CONTEXT.md (what changed, the author's claims, measured facts, what is untested, rules for optional live checks).
The change under review is ${R}/full.diff. The patched sources are under ${R}/tree/ and the upstream originals of the changed files under ${R}/base/.
Read the actual code around every hunk you assess (not just the diff) — callers, callees, macros, #ifdef context, the unit (#ifdef COLI_V4_UNIT_*) a hunk lives in.
Standards:
- Report only defects you can tie to specific code: cite file paths relative to ${R}/tree and line numbers, quote the relevant lines, and give a concrete failure scenario (config/env/inputs -> wrong output, crash, hang, leak, build break, perf regression, or misleading docs).
- Severity: critical = wrong model output / memory corruption / crash in a realistic default or documented config; high = same but needs a plausible non-default config, or a build break on a supported platform; medium = resource leak, perf regression, robustness gap, misleading docs on a key claim; low = minor; nit = style/naming.
- Be skeptical of the author's claims (bit-identity, ordering, "unchanged when off") and try to break them; but do NOT invent issues — if something is fine, record it under verified_ok with your evidence.
- List what you could not check under areas_not_covered.`

const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low', 'nit'] },
          category: { type: 'string' },
          file: { type: 'string' },
          line: { type: 'integer' },
          description: { type: 'string' },
          code_quote: { type: 'string' },
          failure_scenario: { type: 'string' },
          suggested_fix: { type: 'string' },
          confidence: { type: 'number' },
        },
        required: ['title', 'severity', 'category', 'file', 'line', 'description', 'failure_scenario', 'suggested_fix', 'confidence'],
      },
    },
    verified_ok: {
      type: 'array',
      items: { type: 'object', properties: { claim: { type: 'string' }, evidence: { type: 'string' } }, required: ['claim', 'evidence'] },
    },
    areas_not_covered: { type: 'array', items: { type: 'string' } },
  },
  required: ['findings', 'verified_ok', 'areas_not_covered'],
}

const DIMENSIONS = [
  { key: 'prefill-pruned', focus: `Patch areas 1-2: coli_v4_gpu_moe_batch_union (tree/c/deepseek_v4.c ~12042) and the bank sizing (dsv4_cuda_expert_bank_slots, bank create/bank2/double-bank, refill loops, V4_MOE_BANK_FULL loop, missing[] list, bank_valid indexing, hash-router layers). Check: behaviour for exactly 256 experts is unchanged (GPU router path still taken, DeepGEMM path still gets a 256-slot bank); CPU routing for !=256 is correct and really bit-identical to the CPU union / decode routing (coli_v4_route_bf16 with forced_indices NULL vs hashed); the OpenMP parallel-for (nested OpenMP? is the union called from inside an omp region? thread-safety of coli_v4_route_bf16, coli_v4_layer_data, route scratch on stack, reduction(|:route_failed)); gate mirror optionality and device selection from sg; every remaining assumption of 256 experts in this function and in the CUDA bank / refill / upload code (e.g. arrays sized 256, bank_valid[256], expert ids >= slots).` },
  { key: 'expert-major-kernel', focus: `Patch area 3: mv_fp4_expert_major<TB> and run_fp4_expert_major (tree/c/backend_cuda_dsv4.cu ~138-200) and their host side inside the generic dsv4_cuda_route_moe_ids_batch (~2359): route-list construction (cnt[256], slot_of, packed [experts][offsets][routes], nex==0, routes up to 128*6), FC1 (gate+up rows as one O=2I matrix, split) and FC2 calls, arguments order, strides (wstride/sstride vs bank layout in bank create/upload), xdiv/xstride, ystride, grid dims ((O+7)/8, nex), early return for o>=O vs shuffles (are all 32 lanes of a warp always active at __shfl_down_sync? partial warps?), I % 256 assumptions (loop c0 < I/32 step 8), TB template tails, verify mode (buffers, counter, sync), static em_mode init (thread safety, multiple devices), DSV4_MOE_EM_TB invalid values. Most important: rigorously check the BIT-IDENTITY claim vs mv_fp4_grouped<4> (exact per-term expression and evaluation order, FMA contraction differences between the two expressions as compiled by nvcc, partial-sum assignment of i to partials, the reduction tree order). Compare against the original kernel text in base/c/backend_cuda_dsv4.cu.` },
  { key: 'dense-warp-kernel', focus: `Patch area 4: mm_batch_w<F,TB> and the new run_mm_batch dispatch (tree/c/backend_cuda_dsv4.cu ~232-300) vs original mm_batch<F>. Rigorously check the bit-identity claim (per-term expression x*v with v precomputed, fmt 9 bf16 rounding placement, scale index formula, LUT values == e4m3() for all 256 codes incl. NaN/denormal/zero codes, shared-memory LUT init + __syncthreads before any early return, TB/token tails, I%256 requirement, I<=1024 auto rule). Find every caller of run_mm_batch (attention batch, shared experts, matmul_batch, attention_output_batch) and check shapes/T values/fmt where the warp path is taken. Verify-mode implementation: static vbuf/vcount allocated with cudaMalloc on whatever device is current — what happens with two devices or when a later call is on another device; stream used for memset/copy; cudaStreamSynchronize inside run_mm_batch changing semantics. Also the forward declaration of count_bit_mismatch.` },
  { key: 'cuda-resources-errors', focus: `All new/changed CUDA resource and error handling in tree/c/backend_cuda_dsv4.cu: new Dev fields (emaj, emv1/2/4, emcount) zero-initialised? freed?; buf() reallocation semantics (cudaFree on the CURRENT device; async work still reading the old buffer?); every path that calls cudaSetDevice(peer) and returns early without restoring the primary device; the sticky-error clearing in dsv4_cuda_expert_bank_create (does it clear errors it should not? other cudaMalloc failure sites that still leave sticky errors and break the new paths); cudaGetLastError checks after launches that now include peer-device launches; the pin cap (pin_cap_bytes static init, units, cap shared across devices, cudaHostRegisterPortable semantics with 2 devices); memory growth from per-call allocations; leaks on error paths in the new code.` },
  { key: 'peer-compute-concurrency', focus: `Patch area 9, the fused decode dsv4_cuda_moe (tree/c/backend_cuda_dsv4.cu ~1907) with peer compute. Check stream/event ordering end to end: c->fork recorded after the input upload; peer stream waits; cudaMemcpyPeerAsync direction/args; weights upload on the peer stream from host memory; host stack arrays pd[]/hd[] passed to cudaMemcpyAsync from pageable memory (safe?); peer FC2 writing into c->p4 (P2P write) while the primary stream runs its own kernels touching c->p4 / other slots; join event recorded and waited before expert_reduce; the final download+sync on the primary stream also covering the peer work; reuse of c->fork / pc->join events by OTHER code (e.g. aux-stream double-buffer prefetch, attention fork/join) concurrently or in between; event objects created on which device vs used on which device (cudaEventRecord on a stream of another device is an error!); expert_act_grouped/fp8_sim on the primary computing all count rows including peer slots with garbage p1/p2 (NaN/Inf/uninitialised memory, any effect on other rows?); count/pd[32] bounds; the DSV4_CUDA_BATCHED=0 path and dsv4_cuda_expert_group with mixed devices (now dev_ok) — kernels reading peer memory via UVA; bit-identity claim for the peer split. Also: what happens when peer access exists one way only, or dev_ok true but the peer Dev context missing.` },
  { key: 'mirror-cache-tiered', focus: `Patch area 9, the mirror caches in tree/c/deepseek_v4.c (~11476-11870): struct changes, per-cache growth state (reserve_mb default for primary vs peer, last_free_mb probes, unified), global atomic LRU clock, v4_gpu_expert_attach_cached_mode grow_only semantics (-2 vs -1 returns on every branch: count==0, <8 live entries rule, capacity full, reserve), v4_gpu_expert_attach_tiered (search both, grow primary, grow peer, recycle older LRU; UINT64_MAX for empty caches), coli_v4_gpu_expert_peek over two caches (note the original peek body still exists below the new early branch — trace exact control flow), coli_v4_gpu_expert_attach_async still primary-only (hybrid), dspark mirror cache sharing the code and clock. CRITICAL: the recycle-in-place hazard (#1538 in comments) — a view of the current token referencing a mirror entry that a later attach in the same token recycles (stale pointer => garbage output with no error). With two caches, grow_only, tiered victim choice, VRAM-first peeks happening before attaches, and the '<8 live entries' rule, prove or disprove that an entry referenced by the current token can never be recycled before compute. Also thread-safety (decode thread vs loader threads vs prefill vs MTP drafting) and the linear-scan cost with ~4000+ entries.` },
  { key: 'vram-first-pipeline', focus: `Patch area 7, moe_token_pipeline (tree/c/deepseek_v4.c ~5408-5960) vs base/c/deepseek_v4.c. Check every control path for leased[]/fetch[]/views bookkeeping: allocation failure; earlier errors (result set before the new block) — is views freed/released; loader preload over fetch[] indices (jobs[f] vs expert ids), slot = f % lanes, next = f + lanes; the cleanup loop for in-flight loader jobs; lazy host fetch for GPU-only views when !gpu_compute (views[i] = host copy: does it overwrite key? are gpu pointers restored? lease marked); hybrid path (V4_GPU_RESIDENT_FIRST disabled when hybrid on — verify), Vulkan path (vk_step uses views), the non-dual-loader #else build branch, COLI_V4_EXPERIMENTAL_PREFETCH, decode timers. Prove that with V4_GPU_RESIDENT_FIRST unset the behaviour (expert-store lookups, leases, attaches, outputs) is EXACTLY the old behaviour. Also check the GPU-only view stays valid until compute (mirror recycled by attaches of later experts in the same token or by another thread?), and that store statistics/hit accounting and the expert store's own pinning/hot-pin logic (autopin, repin) still behave when mirrored experts are no longer looked up (e.g. hot-pin policy starving, RAM slots for mirrored experts never refreshed).` },
  { key: 'engine-init-portability', focus: `Engine open/close and portability: coli_v4_gpu_engine_open/close (tree/c/deepseek_v4.c ~10972-11060): dsv4_cuda_init called with 2 devices then possibly again with 1 (ng reset, leaked streams/events/cuBLASLt handles, partially initialised g[]), DSV4_CUDA_DEVICE != 0 combined with DSV4_CUDA_PEER_DEVICE, peer == device, out-of-range peer index, arch check, peer pool freed on close and re-open. Compilation of every build flavour: CPU-only, Vulkan (COLI_VULKAN), DeepGEMM (COLI_DSV4_DEEPGEMM), Windows: is deepseek_v4.c built with MSVC or MinGW for Windows (look at build scripts tree/c/build_cuda.bat, Makefiles, coli.cmd) — __atomic_add_fetch, UINT64_MAX, #pragma omp with reduction, pthread usage; the Windows DLL loader tree/c/backend_loader_dsv4.c changes (typedefs, struct members, GetProcAddress of optional symbols, wrapper functions, cast warnings treated as errors?). Unit split: globals defined in one COLI_V4_UNIT and extern-declared in another (g_v4_gpu_first_hits, g_v4_dec_*, g_v4_peer_mirrors static in GPU unit but used where?), symbols referenced in units compiled into binaries that do not link the defining unit (test objects, tools, other targets in the Makefiles that compile deepseek_v4.c units).` },
  { key: 'diagnostics-docs-claims', focus: `Diagnostics and documentation: V4_MOE_BATCH_PROF (moebprof timing points: does 'refill' include waiting? compute includes download+sync), V4_SERVE_STATS_LOG lines in v4_serve_finish (~16569: units, cumulative vs per-turn, extern declarations inside a function, correctness of tok_s), decode timers placement (are attach/fetch/gpu timers measuring what docs say). Docs: tree/docs/deepseek-v4.md new sections and env table rows — every env var name, default and behaviour stated must match the code; measurements tables consistent with REVIEW_CONTEXT facts; the multi-GPU row; claims like 'bit-identical', 'unchanged prefill', 'PCIe x8'. Also ${R}/../PR_DESCRIPTION.md (draft upstream PR text for the first two commits): accuracy vs code. Commit hygiene: commits.txt subjects (a 'wip:' commit), dead code (V4GpuExpertMirrorCache.clock now unused, duplicate declarations), code style consistency with the surrounding codebase (comment density, naming, 2-space vs 4-space, compact one-line CUDA style in .cu).` },
  { key: 'regression-other-configs', focus: `Regressions for configurations NOT on the test box: a real 256-expert DeepSeek V4 Flash on the generic CUDA path (expert-major now default ON with ~3 tokens/expert per chunk — could it be slower? bank 256 slots), DeepGEMM builds (sm_120: which new code paths are compiled/used there?), small VRAM cards 6/8/12/16 GB (new reserve +3300 MB when COLI_CUDA_MOE_DOUBLE=1 — can the mirror cache now freeze below 8 entries or the bank never fit? the per-card table in docs), unified memory (GB10/Jetson: unified flag now per cache), CPU-only, Vulkan, old Windows DLLs with a new engine, multi-GPU WITHOUT P2P (DSV4_CUDA_PEER_DEVICE set but no peer access), DSV4_CUDA_DEVICE=1 single GPU, MTP/dspark drafting on, DSV4_HYBRID=1, long contexts (prefill segments > 4096, many chunks), concurrent requests / kv slots. Also performance regressions of the CPU routing path when 256-expert hash layers or OpenMP thread counts differ, and of mm_batch_w auto (I<=1024) on other GPU generations (sm_80/89/120).` },
]

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    real: { type: 'boolean' },
    severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low', 'nit', 'none'] },
    confidence: { type: 'number' },
    reasoning: { type: 'string' },
    refined_failure_scenario: { type: 'string' },
    fix_assessment: { type: 'string' },
    live_check: { type: 'string' },
  },
  required: ['real', 'severity', 'confidence', 'reasoning'],
}

const LENSES = [
  { key: 'trace', text: 'CODE-TRACE lens: independently re-derive from the actual source (not the finding text) whether the defect exists. Trace the exact control/data flow, #ifdef context and callers. Quote the lines that prove or disprove it.' },
  { key: 'refute', text: 'DEVIL\'S-ADVOCATE lens: your job is to REFUTE the finding. Look for guards, invariants, call-site constraints, or semantics (C, CUDA, OpenMP) that make it impossible or harmless. If after honest effort you cannot refute it, say real=true. If genuinely uncertain, lean real=false.' },
  { key: 'repro', text: 'REPRODUCER/IMPACT lens: construct the most concrete trigger (exact env vars, model shape, hardware, call sequence, values). Decide whether it can actually happen in a supported configuration and how bad it is. You MAY run a cheap live check per the rules in REVIEW_CONTEXT.md (compile-only or a tiny CUDA test under ~/colibri-bench/review-tmp/, never touching services or /tmp/colibri*); report what you ran in live_check.' },
]

function findingText(f) {
  return JSON.stringify({ title: f.title, severity: f.severity, category: f.category, file: f.file, line: f.line, description: f.description, code_quote: f.code_quote, failure_scenario: f.failure_scenario, suggested_fix: f.suggested_fix }, null, 2)
}

async function verifyFinding(f, idx) {
  const votes = await parallel(LENSES.map(l => () => agent(
    `${PREAMBLE}\n\nYou are VERIFYING one reported finding (do not review anything else).\n${l.text}\n\nFinding:\n${findingText(f)}\n\nReturn real=true only if the defect genuinely exists as described (or in a corrected form you describe in reasoning/refined_failure_scenario). Give your own severity ('none' if not real).`,
    { label: `verify:${l.key}:${(f.title || '').slice(0, 40)}`, phase: 'Verify', schema: VERDICT_SCHEMA })))
  const vs = votes.filter(Boolean)
  const yes = vs.filter(v => v.real)
  return { finding: f, votes: vs.map((v, i) => ({ lens: LENSES[i] ? LENSES[i].key : '?', ...v })), confirmed: yes.length >= 2 }
}

const DEDUP_SCHEMA = {
  type: 'object',
  properties: {
    unique: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' }, severity: { type: 'string' }, category: { type: 'string' }, file: { type: 'string' }, line: { type: 'integer' },
          description: { type: 'string' }, code_quote: { type: 'string' }, failure_scenario: { type: 'string' }, suggested_fix: { type: 'string' },
          merged_from: { type: 'array', items: { type: 'string' } },
        },
        required: ['title', 'severity', 'category', 'file', 'line', 'description', 'failure_scenario', 'suggested_fix', 'merged_from'],
      },
    },
  },
  required: ['unique'],
}

async function dedup(found, seenTitles, round) {
  if (!found.length) return []
  const r = await agent(
    `You merge code-review findings. Below are ${found.length} raw findings from independent reviewers of the same patch set (context: ${R}/REVIEW_CONTEXT.md). Merge findings that describe the SAME underlying defect (same root cause), keeping the most precise description, the highest justified severity, and the best fix; list the merged titles in merged_from. Keep genuinely distinct defects separate even if in the same function. ${seenTitles.length ? `Also DROP any finding that is the same defect as one of these already-processed findings from earlier rounds:\n- ${seenTitles.join('\n- ')}\n` : ''}Do not judge validity; only merge/drop duplicates. Raw findings JSON:\n${JSON.stringify(found)}`,
    { label: `dedup:round${round}`, phase: 'Dedup', schema: DEDUP_SCHEMA })
  return r ? r.unique : found
}

// ---------------- Round 1 ----------------
phase('Find')
log(`Round 1: ${DIMENSIONS.length} dimension reviewers`)
const r1 = await parallel(DIMENSIONS.map(d => () => agent(
  `${PREAMBLE}\n\nYOUR DIMENSION: ${d.key}\n${d.focus}\n\nBe exhaustive within your dimension; read all relevant code paths. Return every defect you can substantiate, plus verified_ok entries for claims you checked and found correct.`,
  { label: `find:${d.key}`, phase: 'Find', schema: FINDINGS_SCHEMA })))

const reports = r1.map((r, i) => ({ dim: DIMENSIONS[i].key, ...(r || { findings: [], verified_ok: [], areas_not_covered: ['REVIEWER FAILED'] }) }))
const raw1 = reports.flatMap(r => r.findings.map(f => ({ ...f, dim: r.dim })))
log(`Round 1 raw findings: ${raw1.length}`)

phase('Dedup')
const uniq1 = await dedup(raw1, [], 1)
log(`Round 1 unique findings: ${uniq1.length}`)

phase('Verify')
const verified = await parallel(uniq1.map((f, i) => () => verifyFinding(f, i)))
const allVerified = verified.filter(Boolean)
const seen = uniq1.map(f => `${f.title} (${f.file}:${f.line})`)

// ---------------- Gap rounds ----------------
const CRITIC_SCHEMA = {
  type: 'object',
  properties: {
    new_dimensions: {
      type: 'array',
      items: { type: 'object', properties: { key: { type: 'string' }, focus: { type: 'string' } }, required: ['key', 'focus'] },
    },
    rationale: { type: 'string' },
  },
  required: ['new_dimensions', 'rationale'],
}

let round = 2
let dry = 0
const extraReports = []
while (round <= 4 && dry < 1) {
  phase('Gaps')
  const coverage = reports.concat(extraReports).map(r => ({ dim: r.dim, findings: r.findings.map(f => f.title), verified_ok: r.verified_ok.map(v => v.claim), not_covered: r.areas_not_covered }))
  const critic = await agent(
    `${PREAMBLE}\n\nYou are the COMPLETENESS CRITIC. Below is what the reviewers covered so far (dimension, finding titles, claims verified OK, and what each said it could NOT cover), plus the verification outcome of findings. Identify the most important risk areas of this patch set that are still NOT adequately reviewed (unchecked claims, code paths nobody traced, build flavours, interactions between features, failure modes). Read the diff yourself to spot hunks nobody examined. Propose at most 5 new focused reviewer dimensions (each a precise focus paragraph like the originals), or an empty list if coverage is genuinely complete.\n\nCoverage so far:\n${JSON.stringify(coverage)}\n\nVerification results so far (title -> confirmed):\n${JSON.stringify(allVerified.map(v => ({ t: v.finding.title, confirmed: v.confirmed })))}`,
    { label: `critic:round${round}`, phase: 'Gaps', schema: CRITIC_SCHEMA })
  const dims = critic ? critic.new_dimensions.slice(0, 5) : []
  if (!dims.length) { log(`Critic round ${round}: coverage complete`); break }
  log(`Gap round ${round}: ${dims.map(d => d.key).join(', ')}`)
  const rr = await parallel(dims.map(d => () => agent(
    `${PREAMBLE}\n\nYOUR DIMENSION (gap round ${round}): ${d.key}\n${d.focus}\n\nOther reviewers already reported these findings — do not repeat them, find NEW defects:\n- ${seen.join('\n- ')}`,
    { label: `gap${round}:${d.key}`, phase: 'Gaps', schema: FINDINGS_SCHEMA })))
  const reps = rr.map((r, i) => ({ dim: `gap${round}:${dims[i].key}`, ...(r || { findings: [], verified_ok: [], areas_not_covered: ['REVIEWER FAILED'] }) }))
  extraReports.push(...reps)
  const raw = reps.flatMap(r => r.findings.map(f => ({ ...f, dim: r.dim })))
  const fresh = await dedup(raw, seen, round)
  log(`Gap round ${round}: ${raw.length} raw, ${fresh.length} new after dedup`)
  if (!fresh.length) { dry++; round++; continue }
  seen.push(...fresh.map(f => `${f.title} (${f.file}:${f.line})`))
  const vv = await parallel(fresh.map((f, i) => () => verifyFinding(f, i)))
  allVerified.push(...vv.filter(Boolean))
  round++
}

// ---------------- Synthesis ----------------
phase('Synthesize')
const confirmed = allVerified.filter(v => v.confirmed)
const rejected = allVerified.filter(v => !v.confirmed)
log(`Confirmed ${confirmed.length}, rejected ${rejected.length}`)
const okClaims = reports.concat(extraReports).flatMap(r => r.verified_ok.map(v => ({ dim: r.dim, ...v })))

const SYN_SCHEMA = {
  type: 'object',
  properties: { report_markdown: { type: 'string' } },
  required: ['report_markdown'],
}
const syn = await agent(
  `${PREAMBLE}\n\nYou are writing the FINAL REVIEW REPORT for the patch author (who will fix issues before opening upstream PRs). Inputs below: confirmed findings with their 3 verifier votes, rejected findings (for transparency), and claims reviewers verified as correct. Re-check the top findings against the code yourself where votes disagree on severity.\n\nWrite a markdown report with:\n1. Verdict summary (is it ready for upstream? what blocks it?) in a few sentences.\n2. Confirmed findings ranked by severity (critical→nit). For each: id (F1, F2…), severity, title, file:line (relative to tree/), what's wrong, concrete failure scenario, recommended fix, and the vote tally (e.g. 3/3 confirmed, severity range).\n3. A table of claims verified correct (bit-identity proofs, ordering, unchanged-when-off, etc.) with one-line evidence.\n4. Rejected findings: one line each with why they were rejected.\n5. Untested/unreviewable areas that remain risks.\n6. A prioritized fix plan (what to fix before PR #1 = commits 1-2, before PR #2 = two-GPU decode).\nBe precise and concise; no padding.\n\nCONFIRMED:\n${JSON.stringify(confirmed)}\n\nREJECTED:\n${JSON.stringify(rejected.map(v => ({ finding: { title: v.finding.title, file: v.finding.file, line: v.finding.line, severity: v.finding.severity }, votes: v.votes.map(x => ({ lens: x.lens, real: x.real, reasoning: x.reasoning })) })))}\n\nVERIFIED OK CLAIMS:\n${JSON.stringify(okClaims)}`,
  { label: 'synthesize', phase: 'Synthesize', schema: SYN_SCHEMA })

return {
  report: syn ? syn.report_markdown : null,
  counts: { raw_round1: raw1.length, unique_round1: uniq1.length, verified_total: allVerified.length, confirmed: confirmed.length, rejected: rejected.length },
  confirmed: confirmed.map(v => ({ title: v.finding.title, severity: v.finding.severity, file: v.finding.file, line: v.finding.line, votes: v.votes.map(x => `${x.lens}:${x.real ? 'real' : 'no'}:${x.severity}`) })),
}
