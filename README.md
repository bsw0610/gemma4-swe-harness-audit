# What Actually Limits a Small-Model SWE Agent?

Code for an audit of the [Gemma 4 Developer Agent](https://www.kaggle.com/competitions/gemma-4-developer-agent) harness (`swegemma`), written for the competition's paper track.

Reproduction notebook: https://www.kaggle.com/code/bsw0610/what-limits-a-small-model-swe-agent-reproduction

## Findings

**C1. Tools against documentation.** In `swegemma` 0.2.7 the documented short-name resolver did not exist. Release 0.2.11 added it, which raises reach from the issue text, but BM25 still leads at k = 50.

| Strategy (n = 115 tasks with an in-graph gold symbol) | hit@10 | hit@50 | hit@100 |
|---|---|---|---|
| 0.2.7 tools, issue identifiers as seeds | 2 | 2 | 2 |
| 0.2.11 resolver, 1-hop neighbours | 30 | 35 | 35 |
| 0.2.11 resolver, similarity search with lexical fallback | 42 | 55 | 62 |
| 0.2.11 `search_similar_code` on the whole issue text (lexical fallback only) | 32 | 64 | 84 |
| BM25 over AST function and class spans | 51 | 83 | 98 |

**C2. A localisation skill.** `skill/localize-code/` uses only the standard library and reaches an in-graph gold symbol in its default top 15 for 51.3% of tasks offline (95% CI [42.6, 60.0], n = 115). The reference agent never called it in 8 runs, and its first successful read of a gold file came within a few tool calls.

**C3. Context overflow.** On 120 tasks with `swegemma` 0.2.7, the default `max_output_tokens` of 16,384 ended 26 runs in a context-window error. Halving it removed all of them, and resolved tasks rose from 29 to 34 of the 80 tasks not blocked by missing test dependencies (+6.2 pp, 95% CI [−1.3, +13.8]). The subprocess sandbox skips test dependencies, which blocks verification of the other 40 tasks there. Whether the scoring image has those packages was not checked. Release 0.2.11 submits work in progress when a run overflows; we did not re-run the experiment on it.

Details and limits are in `reports/`, starting with `reports/harness_0_2_11_recheck.md`.

## Reusable resources

These work outside this paper, and the three scripts use only the Python standard library.

| Resource | What it does | Needs |
|---|---|---|
| `skill/localize-code/` | Ranks a Python repository's functions and classes against issue text (BM25 over `ast` spans) and prints the top symbols and files in under 4,000 characters. Copy the folder into a submission's `skills/`. | Any Python repository |
| `analysis/gold_core.py` | Maps the changed lines of a Python unified diff to the innermost enclosing `def` or `class` of the base file, or to module scope, so localisation can be scored by symbol. Change blocks holding only blank lines, outside strings, are skipped. | The patch and the base file |
| `analysis/trace_metrics.py` | Reads one harness trace and reports when the agent first mentioned, read, displayed the lines of and edited a gold file, and whether it called the skill. | A trace and `tasks.jsonl`; tested on traces from `swegemma` 0.2.7 |
| `results/c3_run_rows.csv` | Outcomes for 320 runs of the reference agent without its LoRA adapter and sub-agent (120 tasks in two arms, 40 of them repeated), to compare your own agent against. Token counts and trace metrics cover the first repetition. Leave out the 40 tasks in `results/c3_unjudgeable_tasks.csv` when you compare resolved rates. | Nothing |

```bash
# Rank a repository's functions and classes against an issue
python3 skill/localize-code/scripts/localize.py --root /path/to/repo "issue text"
```

```bash
# Measure one run
python3 analysis/trace_metrics.py trace_<id>.json <id> data/raw/tasks.jsonl
```

```bash
# Run the gold rule's regression cases
python3 analysis/test_gold_core.py
```

```python
import sys; sys.path.insert(0, "analysis")
from gold_core import parse_patch, attribute

hunks = parse_patch(patch)["pkg/mod.py"]
list(attribute(hunks, base_source))  # [(0, 'A.f', 'def', 3, 'replace')]: hunk, symbol, kind, line, block type
```

Evidence and limits:
- **Skill.** Offline, an in-graph gold symbol is in its top 15 for 51.3% of 115 tasks. The reference agent did not call it in 8 runs, so check your own traces to see whether your agent does.
- **Gold rule.** 14 regression cases, and a mutation test on all 129 snapshots (`kaggle/gold_mutation/`) attributes 1,821 of 1,821 synthetic patches correctly.
- **Trace metrics.** They measure access, not understanding, and do not parse shell output beyond checking that it is non-empty.

## Layout

| Path | Contents |
|---|---|
| `notebook.ipynb`, `src/` | The reproduction notebook and its generator |
| `kaggle/` | Scripts that need the snapshots, and notebook metadata |
| `analysis/` | Reports, reach, trace metrics, tests |
| `colab/` | C3 experiment runner (GPU) |
| `skill/localize-code/` | The skill, ready to drop into a submission's `skills/` folder |
| `results/c3_run_rows.csv` | One row per C3 run: outcomes and token counts, with no issue text, code or patches |
| `results/c3_unjudgeable_tasks.csv` | The 40 C3 tasks whose verification is blocked by missing test dependencies in the subprocess sandbox, with the cause |
| `reports/` | Results for each contribution |

## Reproduce

The notebook recomputes the paper's tables and Figure 1 on a Kaggle CPU from the competition files. It takes under 40 minutes. Attach the competition and [the wheelhouse dataset](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse), and run it.

To work locally, join the competition and download its data into `data/raw/`. The snapshots stages run only on Kaggle.

```bash
uv sync
uv run python src/make_notebook.py
uv run python analysis/c3_report.py results/c3_run_rows.csv base out8k
```

`analysis/reach_0211.py` and `analysis/test_overflow_salvage.py` need throwaway environments, because they use the harness wheels. `reports/harness_0_2_11_recheck.md` gives the commands.

The GPU runs are not bit-reproducible, and `reports/c3_results.md` measures how much they vary. They ran on one A100 in Colab, not on the scoring hardware, so official scores can differ.

## Data

The competition rules restrict redistribution of competition data (section 4.b). This repository bundles no raw competition files, issue text, source snapshots or patches. It does include aggregate results and per-run metrics of our own runs (`results/c3_run_rows.csv`, `results/c3_unjudgeable_tasks.csv`). Scripts rebuild the task-level tables from the competition files.

## License

Apache 2.0, see `LICENSE`. Independent work; not affiliated with or endorsed by Google or Kaggle.
