"""Skill uptake and gold-file access in the retained pilot and pre-flight traces (work/c3/pilot, work/c3/preflight).

Traces hold issue text and code, so they are not published; this prints the counts quoted in reports/c2_results.md.
Run: uv run python analysis/c2_pilot_summary.py
"""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from trace_metrics import metrics  # noqa: E402

patches = {t["instance_id"]: t["patch"] for t in map(json.loads, open(ROOT / "data/raw/tasks.jsonl"))}
rows = []
for phase in ("preflight", "pilot"):
    for f in sorted(glob.glob(str(ROOT / f"work/c3/{phase}/trace_*.json"))):
        task, arm = Path(f).stem.removeprefix("trace_").split("_", 2)[:2], Path(f).stem.split("_", 3)[3]
        m = metrics(json.load(open(f)), patches["_".join(task)])
        rows.append(dict(phase=phase, task="_".join(task), arm=arm, open=m["first_open_step"], skill=m["first_skill_any_step"]))
        print(rows[-1])
skill = [r for r in rows if r["arm"].startswith("skill")]
pilot = [r for r in rows if r["phase"] == "pilot"]
print(f"skill-arm traces {len(skill)}, with any skill call {sum(r['skill'] is not None for r in skill)}")
print(f"pilot runs {len(pilot)}, first successful gold-file read within 4 tool calls {sum(r['open'] is not None and r['open'] <= 4 for r in pilot)}")
