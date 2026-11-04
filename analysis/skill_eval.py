"""Score the localize-code skill's offline rankings (Kaggle kernel kaggle/skill_eval) against AST gold v4.

Same definitions and bootstrap as analysis/c1_report.py: n = tasks with >=1 in-graph gold symbol (115),
hit@k = any gold symbol among the first k ranked symbols; symbol id = module path (src/ stripped) + qualname.
Also reports hit on ALL gold symbols (including async/nested ones the graph lacks) and file hit@k.
Run: uv run python analysis/skill_eval.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RANKS = ROOT / "work/c2/skill_eval/skill_ranks.csv"
TIMING = ROOT / "work/c2/skill_eval/skill_timing.csv"
g = pd.read_csv(ROOT / "work/c1/report/gold_mapped.csv")
sym = g[g.kind != "module"].drop_duplicates(["instance_id", "node"])
gold_in = sym[sym.in_graph == True].groupby("instance_id").node.agg(set)
gold_all = sym.groupby("instance_id").node.agg(set)
gold_files = g[g.kind != "module"].groupby("instance_id").file.agg(set)
tasks = sorted(gold_in.index)
n = len(tasks)
IDX = np.random.default_rng(0).integers(0, n, size=(10_000, n))


def module_of(path):
    p = path[4:] if path.startswith("src/") else path
    m = p[:-3].replace("/", ".")
    return m[: -len(".__init__")] if m.endswith(".__init__") else m


r = pd.read_csv(RANKS, keep_default_na=False)
rs = r[r.granularity == "symbol"].sort_values(["instance_id", "rank"])
rs = rs.assign(node=[f"{module_of(p)}.{q}" for p, q in zip(rs.path, rs.qualname)])
rank_sym = rs.groupby("instance_id").node.agg(list)
rank_file = r[r.granularity == "file"].sort_values(["instance_id", "rank"]).groupby("instance_id").path.agg(list)


def fmt(x):
    x = np.asarray(x, float)
    bs = x[IDX].mean(axis=1)
    return f"{int(x.sum())}/{len(x)} = {x.mean() * 100:.1f}% [{np.percentile(bs, 2.5) * 100:.1f}, {np.percentile(bs, 97.5) * 100:.1f}]"


print(f"n = {n} tasks (in-graph gold)")
for k in (5, 10, 15, 25, 50, 100):
    hin = [bool(set(rank_sym.get(t, [])[:k]) & gold_in[t]) for t in tasks]
    hall = [bool(set(rank_sym.get(t, [])[:k]) & gold_all[t]) for t in tasks]
    print(f"symbol hit@{k:3d}: in-graph gold {fmt(hin)} | any gold {fmt(hall)}")
for k in (1, 3, 5, 10):
    print(f"file hit@{k:2d}: {fmt([bool(set(rank_file.get(t, [])[:k]) & gold_files.get(t, set())) for t in tasks])}")
tm = pd.read_csv(TIMING)
print(f"runtime per task: median {tm.seconds.median():.2f}s, max {tm.seconds.max():.2f}s (Kaggle CPU); output chars at --top 100: max {tm.output_chars.max()}")
