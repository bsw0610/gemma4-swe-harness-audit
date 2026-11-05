# C3 results: context-window overflow in the reference agent (Colab A100, 2026-10-07)

> 2026-11-05: these runs used harness 0.2.7. Release 0.2.11 submits work in progress when a run overflows. See `harness_0_2_11_recheck.md`.
>
> 2026-11-05 correction: missing test dependencies blocked verification of 40 of the 120 tasks in this sandbox. See "Judgeable tasks" below. The resolved counts are unchanged; their denominator and two readings are not.

Design (fixed before the main run): 120 tasks (positions 6–125 of `colab/c3_tasks.txt`, a seed-0
random order of the 125 tasks with an existing gold code file; positions 1–5 were the pilot and are excluded), 1 run per arm,
the two arms of a task run concurrently on one vLLM server (A100 40 GB, max_model_len 32,768, TP 1).
- base: sample_submission prompt/tools/sampling without the LoRA adapter and the code_analyzer sub-agent (max_output_tokens 16,384).
- out8k: identical except max_output_tokens 8,192.
- Budgets: 100 tool calls, 30 min, 300 s per command, 500 turns; compaction per HARNESS_README 7.2 (interval 5, overlap 2,
  token threshold 14,336, retention 5).
- resolved = official swegemma verification in the Colab environment (harness env + pytest 8.3.4 from the competition wheels);
  the subprocess sandbox skips test dependencies (see "Judgeable tasks"), and we did not check whether the scoring image has them.
Run data: `work/c3/main/g4swe_c3_main/` (Drive export). Table: `uv run python analysis/c3_report.py work/c3/main/g4swe_c3_main base out8k`
(`work/c3/main/report_v3.txt`, rep 1 table identical to v2). Counts re-derived independently from c3_runs.jsonl (same numbers).

## Paired outcomes (n = 120 tasks; CI = task bootstrap, 10,000 draws, seed 0; McNemar exact)
| outcome | base | out8k | out8k − base [95% CI] | discordant (only out8k, only base) | McNemar p |
|---|---|---|---|---|---|
| context overflow (run ends with ContextWindowExceededError) | 26 | 0 | −21.7%p [−29.2, −14.2] | (0, 26) | < 0.001 |
| empty patch | 29 | 2 | −22.5%p [−30.0, −15.0] | (0, 27) | < 0.001 |
| gold file edited, call succeeded, before overflow (work in progress discarded) | 10 | 0 | −8.3%p [−13.3, −3.3] | (0, 10) | 0.002 |
| resolved | 29 | 34 | +4.2%p [−0.8, +9.2] | (7, 2) | 0.18 |
| 30-min timeout | 1 | 4 | +2.5%p [−0.8, +6.7] | (4, 1) | 0.38 |
| tool budget used up (100 calls) | 9 | 12 | +2.5%p [−2.5, +7.5] | (6, 3) | 0.51 |

The "gold file edited before overflow" row is not "a correct fix was discarded". It cannot test that either: 8 of those 10 tasks are unjudgeable, and on the other 2 the separate out8k runs resolved 0.
Metrics count successful tool calls only (before 2026-11-05 this row counted attempts, 12, with 1 resolved).

Continuous outcomes (median of per-task differences out8k − base, task bootstrap 95% CI): agent seconds +20.8 [+1.4, +50.4]
(medians 270 vs 302); tool calls +1.0 [0.0, +5.0] (medians 23.5 vs 29.5). First successful gold-file read: tool call 3 in both arms (113, 111 runs). Gold lines first displayed: median tool call 4 (109, 106 runs). This measures access, not understanding.

## Mechanism
- vLLM rejects a request when prompt + max_output_tokens > max_model_len. With 16,384 output tokens the prompt ceiling is 16,384;
  the compaction trigger (14,336) leaves 2,048 tokens of headroom, less than one large tool response (pilot: up to +5,492 tokens per step).
- Per-run maximum prompt length: base median 14,859, p95 16,047, max 16,334 (never above the ceiling: the next request fails instead;
  printed by `analysis/c3_fig_prompt.py`, which also draws Figure 1 of the paper);
  out8k median 15,027, p95 17,406, max 20,859. 19/120 out8k runs went past 16,384 (each would have failed under base); none came
  near the new ceiling of 24,576.
- The harness discards the working tree when the model call raises, so all 26 overflowing base runs submitted nothing.

## Judgeable tasks (correction, 2026-11-05)

`swegemma` skips `install_test_dependencies` for the subprocess sandbox (`harness/container_setup.py`, same in 0.2.7 and 0.2.11), and verification requires pytest exit code 0. The blockers below prevent that for a fix that leaves the test dependencies as they are. List: `results/c3_unjudgeable_tasks.csv`. We found this after the runs, from their test output, so the subset was not fixed in advance.

| Cause | Tasks | Evidence |
|---|---|---|
| `inline_snapshot` or `dirty_equals` missing | 31 fastapi | `No module named ...` at collection in every run that reached verification |
| The same, hidden by an earlier error | 1 (fastapi_14609) | A test file in its test patch imports `inline_snapshot`; the only verifying run stopped earlier on an import error of its own |
| `httpbin` fixture broken | 8 requests | `recursive dependency involving fixture 'httpbin'` in both arms |

All 112 runs on these 40 tasks (rep 1 and rep 2) are unresolved. A targeted static check (`kaggle/unjudgeable/unjudgeable.py`, on the snapshots and test patches, following local test modules such as `tests/utils.py`) flags exactly these 40 of the 120 tasks. The other 80 have no such blocker identified, which does not prove each one can pass; 42 of them were never resolved in any run. It also flags 2 public tasks outside C3 (fastapi_14962, fastapi_15023). None of the three names appears in the 124 competition wheels or the wheelhouse file lists. We did not check the scoring image. Paired outcomes on the other 80 tasks, with the same method: `analysis/c3_report.py results/c3_run_rows.csv base out8k --exclude <the 40 ids>`.

| outcome (n = 80) | base | out8k | out8k − base [95% CI] | discordant (only out8k, only base) | McNemar p |
|---|---|---|---|---|---|
| resolved | 29 (36.2%) | 34 (42.5%) | +6.2%p [−1.3, +13.8] | (7, 2) | 0.18 |
| context overflow | 15 | 0 | −18.8%p [−27.5, −10.0] | (0, 15) | < 0.001 |
| empty patch | 17 | 2 | −18.8%p [−27.5, −10.0] | (0, 15) | < 0.001 |
| gold file edited before overflow | 2 | 0 | −2.5%p [−6.2, 0.0] | (0, 2) | 0.50 |

Excluding fastapi_14851 as well gives 28 vs 33 of 79. All 9 resolved-discordant tasks are judgeable. The 120-task table above still holds for what the agent did, because overflow, patch and budget outcomes do not depend on verification. An error naming these packages appears in 6 of the 80 traces on the unjudgeable tasks, including 1 of their 11 base overflows, and in none of the 160 others, so direct exposure explains little. Other indirect effects are not ruled out: within fastapi, base overflowed on 11 of 32 unjudgeable tasks and 3 of 30 others (Fisher p = 0.03), an association that also fits harder tasks. Judgeable tasks are not a random sample: fastapi falls from 62 to 30, and fastapi and rich make up 92.5%.

## What removing the overflow buys
- On the 15 judgeable tasks where base overflowed, out8k resolved 2, timed out 2 and submitted a failing patch 11 times. The other 11 overflow tasks are unjudgeable (out8k: 10 failing patches, 1 tool budget). Over all 26: 21 failing patches.
- Of the 10 tasks where the base run had already edited a gold file, 8 are unjudgeable; the out8k runs resolved neither of the other 2. These data cannot say whether discarded edits were correct: they were never verified.
- On the 65 judgeable tasks without a base overflow, resolved counts are 29 (base) vs 32 (out8k); 7 of the 9 resolved-discordant tasks are tasks without a base
  overflow. With one run per arm this discordance may reflect run-to-run variation (temperature 0.2) that the design cannot separate from
  an effect.
- Reading: on tasks whose paired base run overflowed, the separate out8k run usually failed. Whether the discarded edits were correct is unknown.
  The resolved gain is not distinguishable from zero (n = 80, upper bound +13.8%p).

## Checks and caveats
- Output cap (the treatment also lowers it): no base step came near 16,384 (max 7,197). One out8k step reached 8,192: rich_3105,
  whose issue text is only "Fix #3104 ... See #3104"; the model looped in its thinking ("I don't know what issue #3104 is"), the
  capped step carried no tool call, and the run later timed out (base overflowed on the same task). The next-largest out8k step was
  5,402 tokens, so the lower cap did not truncate tool calls.
- Time budget: 30 min is half the official 60. The extra out8k timeouts (4 vs 1) and its longer runs are partly a product of the
  cap; with the official budget some of those runs would have continued.
- Run-to-run variation: 7 of the 9 resolved-discordant tasks had no base overflow. Calibrated by repetition 2 (below): it is of the
  same size as the resolved difference.
- fastapi_14851 is resolved by both arms with an empty patch: its 9 target tests pass on the unmodified snapshot in this environment
  (test_output). It is concordant, so the resolved-count difference is unchanged; excluding it gives 28 vs 33 (of 119 tasks, or of 79 judgeable ones).
- Overflow by repository: fastapi 14/62, rich 11/44, httpx 1/1, requests 0/13.
- Prior reports of the failure mode exist in the competition forum (#743456, #745977); this run measures its rate and the effect
  of the one-line fix on a random task sample.

## Repetition 2 (calibration, 2026-11-04)

First 40 tasks of the main run, both arms, same harness (swegemma 0.2.7), same sampling. 24 of the 40 are judgeable; every resolved change below falls in those 24, so the discordance rates among them are 4/24, 3/24, 3/24 and 6/24. Rows: `c3_runs.jsonl` rep 2 (80 rows, 0 harness errors,
every row has LLM calls). Table: `uv run python analysis/c3_report.py work/c3/main/g4swe_c3_main base out8k --rep 2`. Counts re-derived independently from the rows (same numbers).

| On the 40 tasks | rep 1 | rep 2 |
|---|---|---|
| resolved, base | 11 | 11 |
| resolved, out8k | 12 | 11 |
| overflow, base | 9 | 7 |
| overflow, out8k | 0 | 0 |
| arms disagree on resolved (out8k only, base only) | 3 (2, 1) | 6 (3, 3) |

| Same arm, rep 1 vs rep 2 | changed (rep 2 only, rep 1 only) | rate [95% CI, task bootstrap] |
|---|---|---|
| resolved, base | 4/40 (2, 2) | 0.100 [0.025, 0.200] |
| resolved, out8k | 3/40 (1, 2) | 0.075 [0.000, 0.175] |
| overflow, base | 12/40 (5, 7) | 0.300 [0.175, 0.450] |

- Between-arm disagreement on resolved is the same size as within-arm disagreement. The main run's (7, 2) cannot be told apart from noise.
  Of the 3 rep 1 discordant tasks in this subset: rich_3067 repeated (base only), rich_3278 flipped direction, fastapi_14419 failed in both arms.
- The overflow arm effect repeated (7/40 vs 0/40, McNemar p = 0.016). Only 2 tasks overflowed in both base runs; independence would give 9 x 7 / 40 = 1.6.
- Limits: 40 of 120 tasks. Within-arm change mixes sampling noise with any environment drift between runs, because the environment was rebuilt a month later:
  wheelhouse dataset v28 (rep 1's use of v28 is inferred from wheel hashes), PyPI dependencies pinned with `uv --exclude-newer 2026-10-06T12:00:00Z`.
- Cost: 6.85 h of run time. Traces for rep 2 are on Drive (`g4swe_c3_main/<arm>/rep2/`), not downloaded, so `gold_edit_then_overflow` is not computed for rep 2.
