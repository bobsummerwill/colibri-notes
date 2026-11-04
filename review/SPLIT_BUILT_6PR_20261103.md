# Recomposed split, built: 6 PRs (2026-11-03)

Status: **executed.** The linear 8-PR stack (#4–#12) was replaced by six
recomposed PRs. No build, test or profiling has been run.

`dev` = `6d907b56` (refreshed from `JustVugg/colibri:dev`).

## The six PRs

| # | PR | branch | base | commits |
|---:|---:|---|---|---:|
| 1 | [#4](https://github.com/bobsummerwill/colibri/pull/4) | `split/dsv4-build-fixes` | `dev` | 2 |
| 2 | [#5](https://github.com/bobsummerwill/colibri/pull/5) | `split/dsv4-multigpu-layers` | #4 | 1 |
| 3 | [#6](https://github.com/bobsummerwill/colibri/pull/6) | `split/dsv4-resident` | #5 | 2 |
| 4 | [#7](https://github.com/bobsummerwill/colibri/pull/7) | `split/dsv4-cuda-kernels` | #6 | 4 |
| 5 | [#8](https://github.com/bobsummerwill/colibri/pull/8) | `split/dsv4-mirror-vram-first` | `dev` | 2 |
| 6 | [#9](https://github.com/bobsummerwill/colibri/pull/9) | `split/dsv4-peer-pool` | #8 | 3 |
| – | [#11](https://github.com/bobsummerwill/colibri/pull/11) | `split/dsv4-cplus-integration` | `split/integration-base` (merge of #7+#9) | 1 |

Two independent lanes: 1→2→3→4 (1766) and 5→6 (2405). Only #11 crosses them.

Closed as folded: #10 (`split/dsv4-cuda-fault`), #12
(`split/dsv4-head-followups`). Their content is in #6/#7/#9/#11.

## What was recomposed (not replayed)

| historical artefact | where it went |
|---|---|
| `7411d188` (f4 table, a revert of the earlier perf choice) | folded into #7's indexer commit |
| `35544b2d` (batch-head default flipped one commit later) | folded into #7's head commit |
| stats commit + VRAM-first commit that rewrites the same print | one commit in #8 |
| `73c05be2` (our repair: route hop + `moe_done` label) | not needed: #8 is written correctly |
| `14339ca2`'s docs catch-up (peer, VRAM-first, pin cap, stats) | kept in #9 with a note; splitting it into #8 is possible if reviewers ask |
| `07dc142a`/`f4586dca` (written against the merged tree) | `auto`/reserve/mocks → #6; head placement/override + test target → #7 |
| `coli_v4_gpu_experts_init` introduced by the merge | introduced in #6 where residency lives |
| KV-owner guard (`9bcbc95b`) | #5 (it is the ownership rule of the placement) |
| experiment reports, docs-bench commits, two "samebase" commits | dropped |
| integration glue (mutual exclusion, per-device+peer dispatch, upload_failed guard, strict peer init / best-effort split) | #11 |

31 source commits + 1 merge became **15 commits in 7 PRs (6 feature + 1
integration)**.

## Integration base

`split/integration-base` = merge of `split/dsv4-cuda-kernels` and
`split/dsv4-peer-pool`, conflicts resolved toward the 2405 side. #11 then brings
the merged tree to the validated aggregate, so its diff is exactly the
reconciliation.

The final tree of #11 is **byte-identical** to the old validated aggregate
(`split/dsv4-head-followups` tip), so every resolution used earlier is preserved.

## Checks performed (no builds)

Per branch tip:

- no conflict markers in `c/` or `docs/`;
- all `dsv4.def` exports present in `c/backend_cuda_dsv4.cu` and bound in
  `c/backend_loader_dsv4.c` (84–100 exports depending on the PR);
- `goto moe_done` has its label wherever the resident pipeline exists;
- #11's tree equals the aggregate tree.

## Known limitations / pre-merge decisions

- Docs catch-up in #9: the peer, VRAM-first, pin-cap and stats rows/sections
  still travel with the fault commit. They can be moved into #8 if reviewers
  prefer; it is a docs-only rebase.
- `docs/experiments/dsv4-*.md` links in the docs point at files not carried in
  the split (same as the author's branch). Decide keep-or-drop before merge.
- `tests/test_v4_gpu_placement.c` mocks evolve per PR (peer stubs only in #11);
  a build of each tip is the only way to be certain every stub set is complete.
- Profiling: `nsys` on shadowfax is fixed via a user-level shim
  (`ops/NSYS_SHADOWFAX.md`), GPU tracing verified with a real kernel; CPU
  sampling remains blocked by `perf_event_paranoid=4` (root).

## Next: the test/profiling round

See `review/TEST_ROUND_ESTIMATE_20261103.md` for the per-PR cost.
