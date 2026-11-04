"""Generate notebook.ipynb: code cells from src/code_a.py, the repository's scripts embedded verbatim, Markdown below.

The analysis scripts import each other and resolve paths from __file__, so they are written to a scratch folder and run as
files rather than pasted into cells. Run: uv run python src/make_notebook.py
"""
import hashlib
import json
import re
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "notebook.ipynb"

code = {}
parts = re.split(r"^# ===== CELL: (\w+) =====\n", (HERE / "code_a.py").read_text(), flags=re.M)
for name, src in zip(parts[1::2], parts[2::2]):
    code[name] = src.strip("\n")

# target inside the scratch folder -> repository file
EMBED = {f"analysis/{n}.py": f"analysis/{n}.py" for n in ("gold_core", "c1_report", "gold_heuristic", "embed_geometry", "reach", "reach_0211", "skill_eval",
                                                          "trace_metrics", "c3_report", "c3_fig_prompt", "c3_overflow_edits")}
EMBED |= {f"kaggle/{d}/{n}.py": f"kaggle/{d}/{n}.py" for d, n in (("gold_ast", "gold_ast"), ("bm25", "bm25"), ("gold_mutation", "gold_mutation"), ("skill_eval", "skill_eval"))}
EMBED["results/c3_run_rows.csv"] = "results/c3_run_rows.csv"
SOURCES = {t: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for t, s in EMBED.items()}

MD = {}
MD["title"] = """# What Actually Limits a Small-Model SWE Agent? Reproduction Notebook

Companion to the paper-track writeup on the Gemma 4 Developer Agent harness. It recomputes the paper's tables and Figure 1 from the competition files and from one published table of run metrics.

Writeup: https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/writeups/what-actually-limits-a-small-model-swe-agent-a-ge. Code and reports: https://github.com/bsw0610/gemma4-swe-harness-audit.

**TL;DR**
- **C1.** In `swegemma` 0.2.7 the documented short-name resolver did not exist. Release 0.2.11 added it, and reach from the issue text rose, but BM25 still leads every shipped graph policy at k = 50. The tools search the whole graph and BM25 ranks spans outside tests and docs, so this compares the tools as shipped, not ranking methods on one corpus.
- **C2.** The offline skill puts an in-graph gold symbol in its default top-15 list for about half of the tasks. Two undocumented harness rules break skills until they are handled.
- **C3.** With `max_output_tokens` 16,384, about one run in five ends in a context-window error. Halving it removes these errors, but the gain in resolved tasks is small and cannot be told apart from run-to-run variation.
"""
MD["bounds"] = """## Definitions and boundaries

| Item | Choice | Why |
|---|---|---|
| Harness versions | C1 audit, C2 and C3 runs on `swegemma` 0.2.7; Table 1 also on 0.2.11 | 0.2.7 was current when we began; 0.2.11 was published on 2026-10-12 |
| Gold symbols | Innermost `def` or `class` enclosing each changed base-commit line, by `ast` | A text-matching heuristic agrees on too few hunks (shown below) |
| Reach sample | Tasks with at least one gold symbol in the graph | The graph tools cannot reach a symbol that is not in the graph |
| Candidate list | Resolved seeds first, then round-robin over tool results, at most k entries | Crediting seeds favours the tools |
| Trace events | A file read or edit counts only if the tool call succeeded | Failed calls are not access |
| C3 tasks | Positions 6-125 of a seed-0 random order of the 125 tasks with an existing gold code file | The first 5 were the pilot and informed the design |
| C3 arms | `base`: reference agent without LoRA adapter and sub-agent, `max_output_tokens` 16,384. `out8k`: the same with 8,192 | One value changed |
| C3 budgets | 100 tool calls, 300 s per command, 500 turns, 30 minutes, compaction interval 5 | Interval as in the README at the time; the current README lists 15 |
| Statistics | Unit = task; 95% CI = task bootstrap, 10,000 draws, seed 0; McNemar exact for paired binary outcomes | Tasks are the sampling unit |

**What this notebook does not recompute.** The C3 GPU runs (one A100 in Colab, 320 runs): their metrics are read from `results/c3_run_rows.csv`, embedded below, which holds no issue text, code or patches. Also the pilot's counts (0 of 8 runs called the skill; traces are not published, see `analysis/c2_pilot_summary.py`), the skill's per-task run time, and the mock-model test of the overflow path, which needs the harness and ADK wheels (`analysis/test_overflow_salvage.py` in the repository).
"""
MD["sources_note"] = """## Scripts

The next cells write the repository's scripts and the C3 table to a scratch folder outside `/kaggle/working`. Their inputs are collapsed; the digests after them identify each file.
"""
MD["c1_facts"] = """## C1. The provided tools and data

The tool behaviour itself was read from the wheels' source (`reports/tool_mechanics.md`). Here we recount the data facts over all 127 graphs. The file listing shows the same data under 0.2.11; the all-graphs hash below confirms it byte for byte.
"""
MD["c1_gold"] = """### Gold fix locations and the BM25 baseline

These cells extract gold symbols from every snapshot and rank spans with BM25, then compare the output digests with those of the runs behind the paper. They take several minutes.
"""
MD["c1_mut"] = """### Mutation test of the gold rule

Synthetic patches of known location on all 129 snapshots: the attribution rule must recover each location.
"""
MD["c1_report"] = """### Gold coverage, BM25 reach and the gate test

The `§3` block gives gold coverage, including the async definitions missing from the graphs and the agreement with the text heuristic. The `§4` block gives BM25 reach. The gate block tests graph expansion on top of a library-only BM25 ranking (85 of 115 at k = 50, printed as 73.9%) at equal budget.
"""
MD["c1_embed"] = """### Embedding geometry

Effective rank and first-component share, as medians over the 127 archives (first numeric column).
"""
MD["c1_reach"] = """### Table 1: reach from the issue text

The 0.2.11 rows call the wheel's own `resolve_node_name`, `get_neighbor` and `get_similar_nodes`.
"""
MD["c1_read"] = """**Reading.** The resolver matters: a policy that reached almost nothing now reaches about a third of the tasks through neighbours, and about half through similarity search. A large part of that comes from text matching, because the whole-issue query uses no embeddings at all. Each shipped policy still trails BM25, and each paired confidence interval printed above excludes zero.

These are static policies ("start from identifiers as written in the issue"). An agent that searches first could do better, so this does not measure what an agent would reach. fastapi and rich also make up 89% of the sample (see the leave-one-repo-out lines above).
"""
MD["c1_gate"] = """### Gate test at k = 50

Does expanding BM25's top m units along `calls` edges help at equal budget? The confidence intervals below are recomputed from the per-task table. This tests one family of expansion policies, not whether structure can help localisation in general.
"""
MD["c2"] = """## C2. The offline skill evaluation

The skill script is embedded in the kernel below, unchanged from `skill/localize-code/scripts/localize.py`. It is a separate BM25 implementation from the C1 baseline. Hit rates are for gold symbols in the graph (first column) and for any gold symbol (second column). Run times are left out of this output because they vary between runs.
"""
MD["c3"] = """## C3. Context overflow

vLLM rejects a request when prompt tokens plus `max_output_tokens` exceed 32,768. With 16,384 reserved, the prompt ceiling is 16,384, and the compaction threshold of 14,336 leaves 2,048 tokens of headroom. One large tool response can jump over it.

The first block below is Table 3 of the paper; the second block is the repetition on 40 tasks.
"""
MD["c3_read"] = """**Reading.** The fix removes the failure it targets, but the effect on resolved tasks is small. Its confidence interval includes zero, and the arms disagree with each other about as often as an arm disagrees with itself on a second run. On most tasks whose paired base run overflowed, the separate out8k run ended in a failing patch.

That is a task-level observation. It does not show whether the discarded base edits were correct, and the data cannot separate it from task difficulty.
"""
MD["c3_fig"] = """### Figure 1

Maximum prompt tokens per run. An overflowing base run never records the rejected request, so its dot is the largest prompt that succeeded.
"""
MD["c3_edits"] = """### What 0.2.11 changes

On 0.2.7 an overflowing run submits nothing. In a mock-model test of the real `run_agent_sandbox`, 0.2.11 instead submits the source edits, without test files. We did not re-run the experiment on 0.2.11, so its overflow rate and resolved count are unmeasured. The line below counts successful edit calls in the overflowing runs. Shell edits and later reverts are not tracked, so it does not measure recoverable patches, and whether any would pass is unknown.
"""
MD["credits"] = """## Sources, licenses and limits

- **Competition data.** [Google - The Gemma 4 Developer Agent](https://www.kaggle.com/competitions/gemma-4-developer-agent): tasks, graphs, embeddings and snapshots, read in place under the competition rules. Section 4.b restricts redistribution of competition data, so the only output file is Figure 1. Aggregate results appear in the cell outputs; issue text, code and patches do not.
- **Harness wheel.** `swegemma` 0.2.11 from the [wheelhouse dataset](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse), checked by digest.
- **This notebook's code.** Apache 2.0.
- **Environment limits.** C3 ran on one A100 (tensor parallel 1), not the scoring setup, with a 30-minute budget. "Resolved" means the official `swegemma` verification in that environment, so official scores can differ on individual tasks.
- **Scope.** Four public repositories, with fastapi and rich dominating; the hidden test set uses private repositories. Results describe associations in this setup. They do not establish why particular runs fail.

Independent work; not affiliated with or endorsed by Google or Kaggle.

### Appendix: changes to expected values

| Check | Before | After | Reason |
|---|---|---|---|
| Gold file edited before overflow, base (no check existed; now in `c3_tables`) | 12, out8k resolved 1 on those tasks | 10, out8k resolved 0 | Count successful tool calls only, after a read-only audit |
"""

ORDER = [("md", "title"), ("md", "bounds"), ("code", "path"), ("md", "sources_note"), ("embed", ""), ("code", "verify_sources"),
         ("md", "c1_facts"), ("code", "graph_facts"), ("md", "c1_gold"), ("code", "gold_stage"), ("md", "c1_mut"), ("code", "mutation_stage"),
         ("md", "c1_report"), ("code", "c1_report"), ("md", "c1_embed"), ("code", "embed_stage"), ("md", "c1_reach"), ("code", "reach_stage"), ("code", "table1"), ("md", "c1_read"),
         ("md", "c1_gate"), ("code", "gate"), ("md", "c2"), ("code", "c2_stage"), ("md", "c3"), ("code", "c3_tables"), ("md", "c3_read"),
         ("md", "c3_fig"), ("code", "c3_figure"), ("md", "c3_edits"), ("code", "c3_edits"), ("code", "closing"), ("md", "credits")]


def hidden(cell):
    cell.metadata["_kg_hide-input"] = True
    cell.metadata.setdefault("jupyter", {})["source_hidden"] = True
    return cell


nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
for kind, name in ORDER:
    if kind == "md":
        nb.cells.append(nbf.v4.new_markdown_cell(MD[name]))
    elif kind == "code":
        src = code[name]
        if name == "verify_sources":
            src = f"SOURCES = {json.dumps(SOURCES, indent=1)}\n" + src
        nb.cells.append(nbf.v4.new_code_cell(src))
    else:
        for target, source in EMBED.items():
            text = (ROOT / source).read_text()
            nb.cells.append(hidden(nbf.v4.new_code_cell(f"%%writefile {{W}}/{target}\n{text}")))
for i, c in enumerate(nb.cells):
    c["id"] = f"c{i:03d}"
nbf.write(nb, OUT)
print("wrote", OUT, len(nb.cells), "cells")
