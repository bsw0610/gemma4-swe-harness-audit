# Harness re-check: swegemma 0.2.7 (our runs) vs 0.2.11 (published 2026-10-12)

Checked 2026-11-05. Our C1 audit and C2/C3 runs used the 0.2.7 wheel. This file lists every paper claim the update touches.

Sources: `swegemma-0.2.11-py3-none-any.whl` (SHA-256 `9ad87658…aea9d3`, metric/gemma-4-developer-agent-wheelhouse),
`HARNESS_README.md` of 2026-10-13 (55,940 bytes; earlier copy 49,356 bytes). Both are kept under `work/harness_0_2_11/`, and `data/raw/` is unchanged.
adk_submission 0.2.13 was read only through the saved diff (`work/c3/rep2/harness_diff_v28_vs_latest.txt`), not installed.

## Claims
| Claim | 0.2.7 | 0.2.11 | How checked | Paper change |
|---|---|---|---|---|
| Short-name resolver does not exist | true | **no longer true**: `resolve_node_name`, 4 tiers, returns one best match | wheel executed | past tense; new Table 1 rows |
| `search_similar_code` matches exact ids only | true | **no longer true**: suffix and case-insensitive match, lexical fallback when nothing resolves | wheel executed | past tense; new Table 1 rows |
| Documented `CALLS`, `DEFINED_IN`, `IMPORTS` return nothing | true | **fixed**: filter is case-insensitive, README now lists `calls` | code + README | past tense |
| All edges are `calls`, no async functions, 50.8% isolated nodes, nearly one-dimensional embeddings | true | **stands**: the hash over all 127 graphs on Kaggle equals our earlier value; the 127 archives have the same byte sizes | Kaggle notebook (graphs), file listing (embeddings) | none |
| In- and out-edges merged and sorted, so direction is hidden | true | **stands** | code | none |
| Gate test (graph expansion lowers BM25 hit@50) | true | **stands**: same data, and the test does not use the resolver | reasoning from the above | none |
| Skill path problem, kebab-case names | true | **stands**: `sandbox/` is unchanged, google_adk is still 1.36.1 | `diff -rq`, wheel listing | narrowed to these constraints |
| Agent ignored the skill (0/8) | measured | not re-measured; tool descriptions, skill-argument handling and general tool-error recovery (`ToolErrorPlugin`) changed | code | threat note |
| Overflow ends the run | true | **stands**: run still ends at the first overflow | executed with a mock model | none |
| Overflow discards the working tree | true | **no longer true**: the source edit is submitted, protected test files stripped | executed with a mock model | version-pinned; new paragraph |
| Reference `max_output_tokens` is 16,384 | true | **stands**: sample unchanged by size, README default and tip still 16,384 | file listing, README | cite in recommendation |
| Compaction interval 5 (README 7.2) | as documented then | **README now says 15**. We used 5; the pilot used 15 | README diff | threat note |
| Only Gemma 4 31B allowed | true | **no longer true**: 12B, E4B and E2B quantized builds also allowed | README diff | intro changed |

## Table 1 measured by calling the wheels' own functions
`analysis/reach_0211.py`. Same 115 tasks, gold, identifier extraction and merge as `analysis/reach.py`. Seeds are listed first, which favours the tools.
| Strategy | hit@10 | hit@25 | hit@50 | hit@100 | hit@50 [95% CI] | minus BM25@50 [95% CI] | only strategy, only BM25 |
|---|---|---|---|---|---|---|---|
| 0.2.7 neighbours (control) | 2 | 2 | 2 | 2 | 1.7% [0.0, 4.3] | −70.4 pp [−79.1, −61.7] | 1, 82 |
| 0.2.7 similarity (control) | 2 | 2 | 2 | 2 | 1.7% [0.0, 4.3] | −70.4 pp [−79.1, −61.7] | 1, 82 |
| 0.2.11 neighbours | 30 | 34 | 35 | 35 | 30.4% [22.6, 39.1] | −41.7 pp [−52.2, −31.3] | 5, 53 |
| 0.2.11 similarity, lexical fallback included | 42 | 51 | 55 | 62 | 47.8% [39.1, 57.4] | −24.3 pp [−34.8, −13.9] | 9, 37 |
| 0.2.11 `search_similar_code(whole issue text)`, lexical fallback only | 32 | 43 | 64 | 84 | 55.7% [46.9, 64.3] | −16.5 pp [−25.2, −7.8] | 5, 24 |
| BM25 over AST spans | 51 | 66 | 83 | 98 | 72.2% [63.5, 80.0] | | |

CI: task bootstrap, 10,000 draws, seed 0. Earlier counterfactual (up to 5 matches per identifier, length ≥ 4): 33/35/36 for 1-hop, 33/40/44 for embeddings at k = 10/50/100.
The shipped similarity tool beats that counterfactual because of its lexical fallback (298 identifier queries used it).

Checks:
- The 0.2.7 control reproduces the paper's 2 / 2 / 2.
- An independent re-implementation with no swegemma import gives the same whole-issue counts (32 / 64 / 84, 0 per-task mismatches at k = 50) and the same number of tasks with a gold seed (29).
- 4 issue texts contain `;`, and the tool treats the text before it as a repo name. We kept that behaviour.

Limits: these are static policies at the function level. The tool layer also cuts output at 5,000 characters, which a full `k` list would hit.

## Overflow behaviour, executed
`analysis/test_overflow_salvage.py` calls the real `run_agent_sandbox` with the real subprocess sandbox. Only the LLM is mocked: it edits `pkg/mod.py`, writes `tests/test_x.py`, then raises `litellm.ContextWindowExceededError`.
| swegemma | Case | Patch | Model calls |
|---|---|---|---|
| 0.2.7 | overflow after edit | empty | 2 |
| 0.2.7 | overflow at turn 1 | empty | 1 |
| 0.2.11 | overflow after edit | `pkg/mod.py` only | 2 |
| 0.2.11 | overflow at turn 1 | empty | 1 |

Rerun twice with identical output (`work/harness_0_2_11/overflow_salvage_*.json`). Environment: macOS, litellm 1.104.0, adk_submission 0.2.12 under both wheels.
The test shows harness control flow, not what a real vLLM server raises. In the main run all 26 overflow rows carry the same `Sandbox execution error: litellm.ContextWindowExceededError` string with an empty patch, which matches the mock's 0.2.7 path.

`analysis/c3_overflow_edits.py`: of those 26 runs, 16 had a successful edit-tool call and 10 on a gold file. Shell edits and later reverts are not tracked, so this does not measure non-empty recoverable patches.

## Versions of the C3 runs
`swegemma` 0.2.7, `adk-submission` 0.2.12, `adk-eval-core` 0.1.0 and `google-adk` 1.36.1 (file names of the wheels used). The Colab environment was installed by hand from the competition wheelhouse before `colab/c3_run.py` ran; that script does not install anything. The wheelhouse's file listing on Kaggle shows `vllm-0.19.1`, but no run-time package manifest was kept, so the paper does not state a vLLM version.

## Rerun
Paths are relative to the project folder.

`reach_0211.py` needs a throwaway env with cachetools 7.2.8, networkx 3.7, numpy 2.5.4, pandas 3.1.1 (not project dependencies), plus the unpacked wheels and `work/c1/report/`:
```bash
uv venv --python 3.12 work/harness_0_2_11/rt/venv_reach
uv pip install --python work/harness_0_2_11/rt/venv_reach/bin/python networkx numpy cachetools pandas
work/harness_0_2_11/rt/venv_reach/bin/python analysis/reach_0211.py --wheel-src work/harness_0_2_11/src --out work/c1/reach_0211
work/harness_0_2_11/rt/venv_reach/bin/python analysis/reach_0211.py --wheel-src work/pkgsrc --out work/c1/reach_0207
```
`test_overflow_salvage.py` needs one venv per swegemma version, built from the local wheels (swegemma, adk_submission 0.2.12, adk_eval_core 0.1.0, google_adk 1.36.1) with `uv pip install --exclude-newer 2026-10-06T12:00:00Z`. It runs from the repository only, not in a Kaggle notebook:
```bash
work/harness_0_2_11/rt/venv_0.2.7/bin/python -I analysis/test_overflow_salvage.py work/harness_0_2_11/overflow_salvage_0.2.7.json
work/harness_0_2_11/rt/venv_0.2.11/bin/python -I analysis/test_overflow_salvage.py work/harness_0_2_11/overflow_salvage_0.2.11.json
```
`c3_overflow_edits.py` runs in the project env: `uv run python analysis/c3_overflow_edits.py`.

## What is not known
- **Resolved rate under 0.2.11.** Not re-run. Recoverable patches and their outcomes are unmeasured. The separate out8k runs resolved 0 of the 10 tasks with a gold-file edit, which says little about what the recovered edits would have done.
- **Overflow rate under 0.2.11.** The token arithmetic is unchanged, but the reshaped workspace tree (pruned of docs and tests, up to 200 entries instead of 150) and any change in compaction interval could shift the rate. The README's interval 15 was used only in our pilot (4 of 10 runs overflowed).
- **Whether the embedding files are byte-identical.** Sizes match; hashes were not compared. The graphs are confirmed identical by hash.
