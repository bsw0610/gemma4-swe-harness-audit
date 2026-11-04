"""C3 task order: tasks with >= 1 gold code file that exists in the base snapshot, in a fixed random order (seed 0).
Runs follow this order, so any prefix is a simple random sample of the eligible tasks.
Run: uv run python analysis/c3_tasks.py   (writes colab/c3_tasks.txt: instance ids only)
"""
import json
import random
from pathlib import Path

from trace_metrics import gold_of

ROOT = Path(__file__).resolve().parents[1]
tasks = [json.loads(l) for l in open(ROOT / "data/raw/tasks.jsonl")]
eligible = sorted(t["instance_id"] for t in tasks if gold_of(t["patch"]))
order = eligible[:]
random.Random(0).shuffle(order)
(ROOT / "colab/c3_tasks.txt").write_text("\n".join(order) + "\n")
by_repo = {}
for i in order[:40]:
    by_repo[i.split("_")[0]] = by_repo.get(i.split("_")[0], 0) + 1
print(f"tasks {len(tasks)}, eligible {len(eligible)}; first 10: {order[:10]}; repos in first 40: {by_repo}")
