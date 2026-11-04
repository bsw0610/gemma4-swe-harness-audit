"""Per-run C3 metrics without any trace text: issue text, code and patches stay out, so the table can be published.

Run: uv run python analysis/c3_export_rows.py   -> results/c3_run_rows.csv
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from c3_report import load  # noqa: E402
from trace_metrics import EDIT_TOOLS, flatten, succeeded  # noqa: E402

RUN = ROOT / "work/c3/main/g4swe_c3_main"
COLS = ["task", "arm", "rep", "resolved", "overflow", "timeout", "tool_budget", "other_error", "empty_patch", "seconds",
        "tool_calls", "has_trace", "lost_edit", "first_open_step", "first_see_lines_step", "edit_arg_errors", "edit_succeeded", "max_prompt",
        "max_step_output", "second_step_output"]
rows = load(RUN)
for r in rows:
    if not r["has_trace"]:
        continue
    atif = json.loads((RUN / r["arm"] / f"rep{r['rep']}" / "traces" / f"trace_{r['task']}.json").read_text())
    r["edit_succeeded"] = any(c["tool"] in EDIT_TOOLS and succeeded(c["result"]) for c in flatten(atif))
    steps = [s["metrics"] for s in atif["steps"] if s.get("source") == "agent" and s.get("metrics")]
    r["max_prompt"] = max(m["prompt_tokens"] for m in steps)
    out = sorted((m.get("completion_tokens") or 0 for m in steps), reverse=True) + [0, 0]
    r["max_step_output"], r["second_step_output"] = out[0], out[1]
path = ROOT / "results/c3_run_rows.csv"
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
print(f"wrote {path.name}: {len(rows)} rows, {sum(r['has_trace'] for r in rows)} with trace metrics")
