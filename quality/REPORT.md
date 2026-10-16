# Quality comparison: DeepSeek V4 Flash REAP-150B (colibri) vs Qwen3.8-27B (llama.cpp)

Run on 2026-10-16, shadowfax (answers) and computer4 (grading). Protocol v2. Results committed to the colibri-notes repo on 2026-10-16.

The question: are REAP-150B's answers enough better than Qwen3.8-27B's to be worth its speed cost on shadowfax, and for what kinds of tasks?

## Headline
| | REAP-150B (greedy, 1 run) | Qwen3.8-27B, Hermes sampling (3 seeds) | Qwen3.8-27B, greedy (1 run) |
|---|---|---|---|
| Math, 4 items: fully correct | 4/4 on content, 3/4 under strict extraction (M4 answered with letters it had defined) | 4/4 on every seed | 4/4 |
| Code, 3 items: all hidden tests pass | 1/3 (C1 17/20, C2 24/24, C3 45/46) | 3, 2, 3 (mean 2.67) | 2/3 (C1 19/20) |
| Code: sum of pass fractions (max 3) | 2.83 | 2.97 | 2.95 |
| Factual F1 / F2 (out of 10) | 8.0 / 8.0 | 8.17 / 9.83 (mean of 3) | 8.0 / 9.5 |
| Open-ended O1 / O2 / O3 (mean of 3 judges) | 4.0 / 3.77 / 6.83 | 2.5 / 7.5 / 6.17 (seed 1 only) | not judged |
| Rubric items vs Hermes: REAP win / loss / tie (1.0-point threshold) | 1 / 2 / 2 | | |
| Verdict by the pre-registered rule | Primary (vs Hermes): Qwen better (automatic items); rubric items not decisive. Secondary (vs greedy): Qwen better, borderline | | |
| Median decode speed | 4.12 tok/s | 86.3 tok/s | 90.4 tok/s |
| Time for all 12 answers | 57.6 min | 4.3 min per seed (3.97-4.55) | 4.1 min |
| Median time per answer | 317 s | 16.5 s | 13.8 s |
| Completion tokens, 12 answers | 13,446 | 21,783 per seed (mean) | 21,694 |
| Truncated / loop-stopped / missing | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Format violations | 2 (M4 letters; C1 sentence before the code block) | 0 | 0 |
| Cost to Hermes | llama.cpp stopped for 3475 s (58 min) | none | none |

## Conclusion
No. On this 12-prompt set, REAP-150B is not better than Qwen3.8-27B, and in some places it is worse. It is not worth its speed cost on shadowfax for general Hermes-style work.

Compared with Qwen at its production (Hermes) settings:
- **Math:** a tie. Both get 4/4 on content, but Qwen is at the ceiling, so these items could never show REAP better.
- **Code:** Qwen is ahead. Its three runs passed every test on 3, 2 and 3 of the 3 code items; REAP did so on 1. REAP rejects day-of-week 7 in C1, and it gives 4 for `-2**2` in C3.
- **Factual precision:** Qwen is ahead on F2 (9.83 vs 8.0). F1 is a tie at about 8.
- **Open-ended:** split.
  - REAP won the PostgreSQL disk-full plan (O1, 4.0 vs 2.5), but both answers were poor and both missed replication slots.
  - Qwen clearly won the lost-update explanation (O2, 7.5 vs 3.8). REAP made the listed critical error about REPEATABLE READ.
  - The incident email (O3) was a tie, inside the 1-point threshold.

The cost side:
- REAP decodes at about 4.1 tok/s against Qwen's 86, roughly 21x slower.
- It took 57.6 min for 12 answers against about 4.3 min per Qwen run. That is about 13x, even though REAP's answers used 38% fewer tokens.
- It needs both GPUs, so Hermes had no model for the whole 58 minutes.

REAP looked as good or better in only two places:
- **Concise math:** the same answers in a third to a half of Qwen's tokens.
- **One ops-planning answer:** it invented fewer commands and was the less dangerous plan.

Both signals are too weak and too small to justify the cost. If REAP is used at all, it fits as an occasional offline second opinion on a hard reasoning or operations question, at a time when Hermes can be paused. It should not replace Qwen for any category tested here.

Caveats:
- n is small: 12 prompts and one REAP run.
- REAP was greedy and Qwen-Hermes was sampled. Greedy Qwen shows the same pattern.
- Qwen is at the ceiling on math and code.
- Thinking was off on both models.
- REAP is limited to an 8192-token context here.
- The judges and the factual grader are Claude models, and the factual grader was not blind.
- The open-ended items were judged against Qwen's seed-1 sample only.

## 1. Protocol

**Models and servers.** Nothing was changed on either server.
- **Qwen3.8-27B:** Q4_K_M with an MTP draft model, on llama.cpp b2912 (`llama-server-active`, port 18080).
  - ExecStart: `--jinja --reasoning-format deepseek --reasoning off --temp 0.7 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.5 --repeat-penalty 1.0 --parallel 2 --ctx-size 262144`, which is 131072 tokens per slot.
- **DeepSeek V4 Flash REAP-150B:** 132 of 256 routed experts kept, fp4 experts.
  - It ran on colibri at d0e19de2 (`/tmp/cw/r6-pr2`, CUDA sm_86, NO_TC=1).
  - Command: `coli serve --ram 40 --ctx 8192 --port 8090`.
  - Decode used both GPUs; the serve log confirms `v4_gpu peer-mirrors device=1 ... mode=compute`.
  - The engine is greedy only.

**Prompts.** There are 12, in priority order. The index is the 0-based position in `prompts.json`.

| # | Prompt | Category | max_tokens |
|---|---|---|---|
| 0 | M1_strings | math | 7936 (ceiling) |
| 1 | C1_cron | code | 4096 |
| 2 | O1_pg_disk_plan | open | 3072 |
| 3 | M3_spinner | math | 7936 (ceiling) |
| 4 | F1_systemd | factual | 2048 |
| 5 | C3_expr | code | 4096 |
| 6 | O2_lost_update | open | 2048 |
| 7 | M2_crt | math | 8000 (ceiling) |
| 8 | O3_incident_email | open | 2048 |
| 9 | C2_shell_split | code | 4096 |
| 10 | M4_restart_order | math | 7808 (ceiling) |
| 11 | F2_tcp_time_wait | factual | 2048 |

- Both models got the same single user message: no system prompt and no tools.
- The prompt bytes are identical (sha256 recorded per answer), and so are the budgets.
- Every budget plus its prompt fits in 8192 tokens, so colibri never clamped a budget (`ctx_clamped=false` on all 12).

**Arms.**
- **reap:** greedy, 1 sample, `enable_thinking=false`.
- **qwen-hermes** (primary comparator): Hermes sampling sent explicitly, seeds 1, 2 and 3.
- **qwen-greedy:** temperature 0, top_k 1, no penalties.
- **qwen-think:** not run.

**Run order.**
1. The Qwen arms ran first with Hermes up, one request at a time, with a slot wait and 2 s pauses.
2. A gate checked the Qwen results.
3. REAP ran in one GPU window: llama.cpp was stopped at 00:31:24 and /health was ok again at 01:29:19, 3475 s in all.

**Thinking.** Thinking was off on both models, which is each server's production default.
- llama.cpp `--reasoning off` adds an empty `<think></think>`.
- colibri thinks only if the request asks for it, or if the server has COLI_THINK=1. The client sent `enable_thinking=false`.
- All 60 answers have `reasoning_chars=0`. Only the final answer text was graded.

**Grading.**
- **Status:** only status ok counts as an answer. All 60 answers were ok, so nothing was missing.
- **Math:** exact match on the last `ANSWER:` line. All four references were re-checked by brute force or exact computation.
- **Code:** hidden tests ran on computer4 only, sandboxed in fresh dirs under `quality/grading/` with unshare -rn, a timeout, rlimits and `python3 -I`.
  - The reference solutions pass 20/20, 24/24 and 46/46.
  - The AST check found no forbidden constructs in any answer.
- **Factual (F1, F2):** a Claude grader scored per-question points against the reference. It was not blind.
- **Open-ended (O1-O3):** three blind Claude judges compared REAP with qwen-hermes seed 1. Each scored the answers 1-10 on the prompt's rubric, including its critical errors, and the three scores were averaged.
- **Verdict rule (`grade.py compare`):**
  - Automatic items: "Qwen better" needs Qwen's mean number of fully correct items >= REAP + 1, and Qwen's worst seed >= REAP. "REAP better" is the mirror image.
  - Rubric items: a win or loss needs a margin of at least 1.0 point, and "better" needs wins minus losses >= 2.
- **Possible judge bias:** neither contestant is a Claude model, so self-preference is unlikely. Claude judges may still prefer some styles over others (headings, hedging, formatting). The same model family also did the factual scoring and proposed the M4 override.

## 2. Per-prompt results

Objective results; open-ended scores are the de-blinded mean of three judges.

| # | Prompt | REAP-150B | Qwen Hermes s1 / s2 / s3 | Qwen greedy | vs Hermes |
|---|---|---|---|---|---|
| 0 | M1_strings | correct (20320) | correct x3 | correct | tie |
| 1 | C1_cron | 17/20: rejects day-of-week 7, accepts `5/2`; text before the code block | 20/20 x3 | 19/20 (accepts `5/2`) | Qwen ahead |
| 2 | O1_pg_disk_plan | 4.0 | 2.5 (s1 only) | not judged | REAP win (+1.5) |
| 3 | M3_spinner | correct (11437/20736) | correct x3 | correct | tie |
| 4 | F1_systemd | 8.0; critical error: says SIGTERM death is restarted | 8.5 / 8.0 / 8.0 (mean 8.17) | 8.0; same error | tie (-0.17) |
| 5 | C3_expr | 45/46: `-2**2` gives 4 | 46 / 42 / 46 (s2 used floor `%`) | 46/46 | Qwen ahead |
| 6 | O2_lost_update | 3.77; critical error: says REPEATABLE READ still loses the update | 7.5 (s1 only) | not judged | Qwen win (-3.73) |
| 7 | M2_crt | correct (188) | correct x3 | correct | tie |
| 8 | O3_incident_email | 6.83; 181 words, "about 40 min" (should be 37) | 6.17 (s1 only); 199 words, "40 minutes" | not judged | tie (+0.67) |
| 9 | C2_shell_split | 24/24 | 24/24 x3 | 24/24 | tie |
| 10 | M4_restart_order | correct on content; `ANSWER: P, A, C, U, D, W` uses its own letters; strict 0, override to 1 | correct x3 | correct | tie (with override) |
| 11 | F2_tcp_time_wait | 8.0: gives one of the two TIME_WAIT reasons; says tcp_fin_timeout "does not exist" | 10 / 9.5 / 10 (mean 9.83) | 9.5 | Qwen win (-1.83) |

**Primary verdict (REAP vs qwen-hermes).**
- **Automatic items** (fully correct out of 7): REAP 5, or 4 without the override. Hermes seeds 7, 6 and 7, mean 6.67.
- The test passes: 6.67 >= 5 + 1, and the worst seed (6) >= 5. So Qwen is better on these items, with or without the override.
- `auto_ceiling` is true.
- **Rubric items:** REAP has 1 win (O1), 2 losses (F2, O2) and 2 ties (F1, O3). Wins minus losses is -1, so not decisive. Rubric mean: REAP 6.12, Qwen 6.83.
- **Verdict:** qwen-hermes better.

**Secondary verdict (REAP vs qwen-greedy).**
- Automatic items: 6 vs 5. The difference equals the noise yardstick (the Hermes seed range of 1), so this is borderline.
- Only F1 and F2 are paired on the rubric, and they are not decisive.
- **Verdict:** qwen-greedy better (borderline).

Math explanations were not judged.

## 3. Judge votes (de-blinded) and mapping check

| Prompt (index) | Judge 1 (A=qwen) | Judge 2 (A=reap) | Judge 3 (A=reap) | Majority | REAP mean | Qwen s1 mean | Outcome |
|---|---|---|---|---|---|---|---|
| O1 (2) | REAP 4.0 vs 2.5 | REAP 4.0 vs 3.0 | REAP 4 vs 2 | REAP 3-0 | 4.00 | 2.50 | REAP win |
| O2 (6) | Qwen 7.5 vs 3.8 | Qwen 7.5 vs 4.0 | Qwen 7.5 vs 3.5 | Qwen 3-0 | 3.77 | 7.50 | Qwen win |
| O3 (8) | REAP 6.5 vs 6.0 | REAP 7.0 vs 6.5 | REAP 7 vs 6 | REAP 3-0 (all called it close) | 6.83 | 6.17 | tie (< 1.0) |

What the judges said:
- **O1:** both answers are weak. Neither checks replication slots or `pg_stat_archiver`, and both are over 600 words.
  - Qwen's answer uses commands and functions that do not exist (`pg_lsdir`, `pg_control_check()`) and the wrong path `pg_tmp`. It invents "pg_xact bloat", advises raising `wal_keep_size`, and offers to delete pg_wal files older than 7 days.
  - REAP's commands mostly run, and it gates WAL deletion behind a human. It does kill idle-in-transaction sessions without asking.
- **O2:** REAP makes the REPEATABLE READ critical error. Qwen handles 40001 and retry correctly and covers write skew.
- **O3:** both say 40 minutes instead of 37. REAP keeps the "below 50" detail and has clearer sections. Qwen explains the cause slightly better but ends with a `[Your Name]` placeholder.

**Mapping check.** The rule given to the judges:
- Judge 1: A is qwen on even indices.
- Judge 2: A is reap on even indices.
- Judge 3: A is qwen on even indices, then swap.

O1, O2 and O3 are at indices 2, 6 and 8, all even. So the rule gives Judge 1 A=qwen and Judges 2 and 3 A=reap on all three prompts, which is what each judge recorded.

The recorded attributions also match the answer texts:
- **O1:** `pg_lsdir`, `pg_control_check`, `pg_tmp`, `pg_xact` and "older than 7 days" appear only in the qwen-hermes s1 answer. CHECKPOINT and "idle in transaction" appear only in REAP's.
- **O2:** "lost update still happens" under REPEATABLE READ is REAP's. Write skew appears only in Qwen's.
- **O3:** "about 40", 09:44 and "below 50" are REAP's. "[Your Name]" and "limited its ability" are Qwen's.

Notes on the scheme:
- "Even index" is ambiguous. If it meant the position among the open-ended prompts, O2 would have flipped. All three judges read it the same way, and the content check shows the attributions are right either way.
- Judges 2 and 3 saw the same order, so REAP was shown first in 6 of 9 comparisons. The unanimous winners mean position bias changed no outcome.
- Blinding was by instruction only: the judges built their pairs from files that name the arms.

## 4. Speed

| | REAP-150B | Qwen Hermes (mean per seed) | Qwen greedy |
|---|---|---|---|
| Completion tokens, 12 answers | 13,446 | 21,783 | 21,694 |
| Generation time, 12 answers | 57.6 min (3456 s) | 4.3 min (259 / 273 / 238 s) | 4.1 min (245 s) |
| Mean / median time per answer | 288 / 317 s | 21.4 / 16.5 s | 20.4 / 13.8 s |
| Median decode | 4.12 tok/s (3.83-4.51) | 86.3 tok/s | 90.4 tok/s |
| Effective tok/s (including prefill) | 3.89 | 84.8 | 88.6 |
| Time to first token | 7.7-28.3 s | 0.22-0.71 s | 0.22-0.49 s |
| Hermes availability | llama.cpp down 58 min | unaffected | unaffected |

Per prompt, as tokens / seconds, then the REAP-to-Hermes ratio of tokens and of time:

| Prompt | REAP | Qwen Hermes (mean of 3) | Qwen greedy | Tokens ratio | Time ratio |
|---|---|---|---|---|---|
| M1 | 1560 / 369 | 5024 / 55.8 | 3638 / 38.3 | 0.31 | 6.6x |
| C1 | 1122 / 304 | 1503 / 17.5 | 1754 / 20.0 | 0.75 | 17.4x |
| O1 | 1374 / 381 | 1087 / 16.4 | 938 / 13.6 | 1.26 | 23.2x |
| M3 | 1418 / 329 | 4492 / 49.1 | 6267 / 66.6 | 0.32 | 6.7x |
| F1 | 522 / 152 | 346 / 5.5 | 755 / 10.4 | 1.51 | 27.7x |
| C3 | 2179 / 540 | 1629 / 19.4 | 1248 / 14.0 | 1.34 | 27.8x |
| O2 | 923 / 257 | 966 / 13.6 | 917 / 12.7 | 0.96 | 18.9x |
| M2 | 1586 / 360 | 2755 / 30.7 | 2310 / 24.6 | 0.58 | 11.7x |
| O3 | 286 / 99 | 279 / 4.4 | 256 / 3.6 | 1.02 | 22.7x |
| C2 | 439 / 123 | 514 / 6.0 | 460 / 5.0 | 0.85 | 20.4x |
| M4 | 1495 / 383 | 2812 / 32.5 | 2740 / 30.3 | 0.53 | 11.8x |
| F2 | 542 / 160 | 377 / 5.9 | 411 / 5.8 | 1.44 | 27.2x |

REAP's short math answers are why the GPU window took 58 min instead of the estimated 115.

## 5. Truncations, missing answers and anomalies

- **No truncations.** All 60 answers ended with stop: no loop-guard stop and no context clamp.
  - The closest to a limit: Qwen Hermes seed 2 used 7263 of 7936 tokens on M1 (0.92 of the budget), and Qwen greedy used 0.79 of the M3 budget. REAP's highest was 0.53, on C3.
- **No missing answers.** Every comparison is fully paired.
- **REAP format slips** (Qwen had none):
  - On M4 it answered with letters, so strict extraction gave 0. The letters map exactly to the unique solution, which brute force over all 720 orders confirms. The grading agent recorded an override in `grades/overrides.json`; Bob should confirm it.
  - On C1 it put a sentence before the code block.
- **Possible engine artefact.** REAP's F1 answer contains the garbled phrase "likely meant as SIGTERM". One run cannot separate a model weakness from a colibri numerical issue.
- **Deviations from the grading plan:**
  - The open-ended items were scored holistically, not through the `grade.py packets` criteria.
  - Qwen's open-ended scores come from seed 1 only.
  - Greedy Qwen was not judged on O1-O3.
  - Math explanations were not judged.

## 6. Caveats

- **Small n, one run.** There are 12 prompts and one REAP sample each.
- **Greedy vs sampled.** REAP can only decode greedily. Greedy Qwen shows the same pattern as sampled Qwen, but REAP's sampled quality is unknown.
- **Ceiling effect.** Qwen gets every math answer right, so math and code could not show REAP better.
- **Thinking was off on both.** At about 4 tok/s, a REAP run with thinking on would be very slow.
- **Model tested.** This is REAP-150B (pruned, fp4) on colibri, not the full V4 Flash. Losses from pruning and any engine error cannot be separated from the base model.
- **Context.** REAP has 8192 tokens here, against 131072 per Qwen slot.
- **Grading was all by Claude models.** The factual grader was not blind. The open-ended judges were blind by instruction only, and two of the three saw REAP first. Bob spot-checking O1 and the M4 override would reduce this.
- **Untested:** long context, tool use, niche knowledge, other languages, and reasoning harder than Qwen's ceiling.

## 7. Files (under `quality/`)

- Answers: `outputs/qwen.jsonl` (8b244482...) and `outputs/reap.jsonl` (66ec718f...)
- REAP run evidence: `outputs/reap.serve.log`, `outputs/reap.serve.nvsmi.csv` and `outputs/reap-run/`
- Per-arm copies for grade.py: `results/`
- Qwen run log: `logs/qwen_arms.log`
- Grades: `grades/auto.json`, `grades/objective.json` and `grades/overrides.json`
- Code-test sandbox dirs: `grading/`
- Copies are on shadowfax under `~/colibri-bench/quality/`.
