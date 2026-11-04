"""Overflowing base runs of the C3 main run (rep 1): what 0.2.11's working-tree recovery would have had to submit.

The patch itself was lost in 0.2.7, so this only counts successful edit-tool calls in the trace; shell edits (sed) and later reverts are not tracked,
so it does not measure non-empty recoverable patches.
Run: uv run python analysis/c3_overflow_edits.py [results/c3_run_rows.csv]
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from trace_metrics import EDIT_TOOLS, flatten, metrics, succeeded  # noqa: E402

if len(sys.argv) > 1:  # published per-run table instead of traces
    rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["rep"] == "1" and r["arm"] == "base" and r["overflow"] == "True"]
    print(f"overflow runs {len(rows)}; empty patch {sum(r['empty_patch'] == 'True' for r in rows)}; "
          f"successful edit-tool call {sum(r['edit_succeeded'] == 'True' for r in rows)}; "
          f"successfully edited a gold file {sum(r['lost_edit'] == 'True' for r in rows)}")
    sys.exit(0)

RUN = ROOT / "work/c3/main/g4swe_c3_main"
rows = [json.loads(line) for line in open(RUN / "c3_runs.jsonl")]
patches = {t["instance_id"]: t["patch"] for t in map(json.loads, open(ROOT / "data/raw/tasks.jsonl"))}
over = [r for r in rows if str(r["rep"]) == "1" and r["arm"] == "base" and "ContextWindowExceededError" in str(r["error"])]
prefix = sum(str(r["error"]).startswith("Sandbox execution error:") for r in over)
empty = sum(int(r["patch_chars"]) == 0 for r in over)
any_edit = gold_edit = 0
for r in over:
    atif = json.load(open(RUN / f"base/rep1/traces/trace_{r['task']}.json"))
    any_edit += any(c["tool"] in EDIT_TOOLS and succeeded(c["result"]) for c in flatten(atif))
    gold_edit += metrics(atif, patches[r["task"]])["first_edit_step"] is not None
print(f"overflow runs {len(over)}; 'Sandbox execution error:' prefix {prefix}; empty patch {empty}; "
      f"successful edit-tool call {any_edit}; successfully edited a gold file {gold_edit}")
