# Restructure plan: 6 recomposed PRs (2026-11-03)

Status: **planned, not executed.** This supersedes both earlier drafts (18 PRs,
then 7 PRs). The 8-PR linear stack (#4–#12) stays open and mergeable until the
window.

## Principle

A PR is defined by its **aggregate behaviour**, not by the commits that happened
to produce it. The author's history contains churn from how the branches were
built (a perf choice reverted in the next commit, a default flipped afterwards,
docs written at the end for features added earlier, fixes authored against the
merged tree). None of that belongs in the split.

So each PR below is assembled from the **final resolved tree** and carries a
small, purposeful commit list. Where two source commits touch the same code to
add-then-correct it, they are one commit here. Where a later "fix" only exists
because of the merge, its content moves into the feature that owns it.

## Commit accounting

| stream | commits | used |
|---|---:|---|
| #1766 (`3b953308..aabd99d4`) | 26 (20 non-merge) | 14 code/test; 6 dropped (experiment reports, docs-bench, two "samebase" harness commits) |
| your fixes (`aabd99d4..f4586dca`) | 7 | 7, redistributed into the owning PRs |
| #2405 (`c1c06ac6..11c40038`) | 11 | 10 code; 1 docs distributed |
| **total** | **31 + 1 merge** | **15 commits in 6 PRs** |

## The 6 PRs

Lanes A/B/C are independent; only #6 crosses lanes.

### PR 1 — `split/dsv4-build-fixes` → `dev` (2 commits)

| commit | content |
|---|---|
| `dsv4_cuda_init`: refused device leaves no stale error | `aa1b4f52` |
| `native_quant.h`: GGUF helpers are C only | `28b35786` |

### PR 2 — `split/dsv4-multigpu-resident` → PR 1 (3 commits, ~1,300 lines)

| commit | content |
|---|---|
| distribute layers and expert caches across devices | `bfaee5b2` + KV-owner guard (`9bcbc95b`, KV half) + `7ad5319a` (test-device header in rules) + placement docs |
| resident routed experts with budgeted `auto` | `460f21cb` + `919c58cd` + `96d2000e` (auto half) + `07dc142a`'s `experts_init` hard-failure logic, `DSV4_CUDA_RESIDENT_RESERVE_MB` split, NULL guard, mock-test updates + reserve docs |
| device-resident routing | `96d2000e` (routing half): `DSV4_CUDA_RESIDENT_ROUTE`, backend/loader/def entry point, route test, route docs |

### PR 3 — `split/dsv4-cuda-kernels` → PR 2 (4 commits, ~1,800 lines)

| commit | content |
|---|---|
| resident indexer, native quant and the `f4` table | `1c42795b` + `7411d188` (squashed: same feature, the table choice belongs in it) + `2c675407` |
| compressor and indexer preparation on CUDA | `7019d0af` |
| decode output projection on CUDA | `371ed4e3` |
| exact head, batched head and head placement | `81c6cb39` + `97a47941` + `35544b2d` (squashed default flip) + `07dc142a`'s head-placement half + `f4586dca` (`DSV4_CUDA_HEAD_DEVICE`) + `6b2efd31`/`0956307c` (test target + header deps) + head docs |

This is the largest PR. If it still reads too big, split the last commit out as a
"head" PR (7 PRs total); nothing else changes.

### PR 4 — `split/dsv4-mirror-vram-first` → `dev` (2 commits, ~850 lines)

| commit | content |
|---|---|
| mirror cache growth guard, reserve and pin cap | `99a37137` + `93ec3778` + `4293d4c9` + reserve/guard docs |
| VRAM-first decode and per-turn stats | `0d6454f6` + `897ff79e` (squashed: the stats print already carries VRAM-first counters) + `V4_GPU_RESIDENT_FIRST` docs |

### PR 5 — `split/dsv4-peer-fault` → PR 4 (3 commits, ~1,450 lines)

| commit | content |
|---|---|
| peer mirror pool on a second GPU | `6928922a` + peer env docs |
| peer compute on the second GPU | `83850cf9` |
| sticky CUDA fault latch and decode failure reporting | `14339ca2` + `0ce6e5dc` + `11c40038` + fault/peer-fault docs |

The docs in the fault commit that describe the peer pool, VRAM-first and the pin
cap move to the commits that own those features; the fault commit keeps only its
own docs.

### PR 6 — `split/dsv4-cplus-integration` → `split/integration-base` (1 commit, ~150–200 lines)

`split/integration-base` is a temporary branch: the merge of PR 3 and PR 5.
The single commit carries only what neither lane can own:

- mutual exclusion of `DSV4_CUDA_DEVICES` (>1) and `DSV4_CUDA_PEER_DEVICE`;
- per-device cache lookup plus peer dispatch in `v4_gpu_expert_vram_held`,
  `peek`, the tiered attach and the resident route;
- verify peer access after the best-effort init before creating the pool;
- strict `dsv4_cuda_init` default + `set_peer_best_effort` around the layer-split
  init;
- `upload_failed ||` in the growth-guard condition;
- the merged docs (layer placement + peer pool on one page).

There is **no followups PR**: KV guard → PR 2, test target → PR 3, docs →
features, `auto`/reserve → PR 2, head placement/override → PR 3.

## Graph

```text
dev ── PR1 build fixes ── PR2 multi-GPU + resident ── PR3 kernels
dev ── PR4 mirror + VRAM-first ── PR5 peer + fault
                  PR3 + PR5 ── PR6 integration
```

#4 does not depend on #1–#3. #6 depends on the two lane tips only.

## Mess ledger (what recomposition removes)

| historical artefact | resolution |
|---|---|
| experiment-report and docs-bench commits (`b7c763cb`, `a24d02dd`, `a019014e`, `40f6f7eb`) and `ca55947c` ("keep reports out") | never added; nothing to commit |
| `7411d188` ("f4: decode FP4 with a table again") reverting a perf choice made earlier in the same feature | folded into the indexer/quant commit |
| `35544b2d` ("enable validated CUDA batch heads by default") flipping a default one commit after the head | folded into the head commit |
| `0d6454f6` stats + `897ff79e` VRAM-first, where the stats print gains VRAM-first counters in the next commit | one commit in PR 4 |
| `73c05be2` (our repair: resident route hop + `moe_done` label lost in the VRAM-first merge) | PR 4's commit is written correctly; no repair commit |
| `14339ca2` docs catch-up for peer/VRAM-first/pin cap | docs distributed to PR 4/PR 5 features |
| `07dc142a`/`f4586dca` written against the merged tree | `auto`/reserve/mock → PR 2; head placement/override → PR 3 |
| `coli_v4_gpu_experts_init` introduced by the merge | introduced in PR 2, where residency lives |
| `9bcbc95b`'s best-effort peer half (superseded by the strict-init design) | only the KV guard survives (PR 2); the peer half becomes PR 6's `set_peer_best_effort` |
| two "samebase" harness commits (`819cd03d`, `d4d8ab3f`) | not carried |
| broken `experiments/dsv4-*.md` links in the docs | decide before merge: keep (as the author's tree does) or drop those sentences; noted, not changed |

## Safety rules for the recomposition

1. Each PR's tree must compile against its own base. That means the test mocks
   (`tests/test_v4_gpu_placement.c`) evolve with the feature: the peer stubs only
   appear in PR 6, the resident mocks only in PR 2.
2. No commit references a symbol introduced by a later PR. The `DSV4_CUDA_RESIDENT_RESERVE_MB`
   helper is added in PR 2 and consumed in PR 3's head commit.
3. Docs travel with the feature that first needs them; PR 6 documents only the
   combined behaviour.
4. `docs/experiments/` stays out of every PR.

## Execution method

Not a rebase of the existing branches: assemble each PR from the final tree.

1. Freeze #4–#12 on `origin`.
2. For each lane, start at its base, then apply the **aggregate diff of that
   feature** (diff final tree against base restricted to the feature's files/
   hunks), writing the commit list above. The current linear branches are the
   reference for each resolved hunk.
3. PR 2 and PR 3 need the followup hunks moved down; PR 4 and PR 5 need the docs
   hunks from `14339ca2` distributed.
4. Build `split/integration-base` = merge of PR 3 and PR 5, then write PR 6's
   single commit from the known resolutions.
5. Verify per tip: no conflict markers; `dsv4.def` exports all bound;
   `deepseek_v4.c` function set equals `8a30a985`; `goto moe_done` has a label;
   mocks only reference symbols available at that tip.
6. Open the 6 PRs, close the old set, update the notes with links.

Estimated effort: ~4–6 h (more than a pure rebase because the trees are
recomposed), no builds, no GPU time. The earlier pure-rebase estimate of ~2.5–3 h
does not buy the clean history.

## Rollback

Old branches stay on `origin` until all six are verified. The reference trees
(`64fd75e3`, `8a30a985`) and the current linear branches hold every resolved
hunk, so any recomposition can be diffed against them.

## Granularity

| option | PRs | note |
|---|---:|---|
| **recommended** | **6** | one per coherent feature; two parallel lanes; one integration |
| split the head | 7 | only if PR 3 reads too big |
| the earlier 7-PR grouping | 7 | keeps a followups PR; unnecessary once fixes are folded |
| 18 | — | rejected: serialises review and adds base juggling |
