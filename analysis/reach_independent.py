"""Independent re-implementation of the documented-resolver (DOC-5) 1-hop reach, written without reach.py's
extraction code, to measure how much the seed-extraction heuristic moves the result (audit 2 N05).

Differences from analysis/reach.py (by design):
- candidate identifiers: backtick spans re-tokenised into dotted/plain names, dotted names, CamelCase, snake_case,
  and `name(`; stripped of leading/trailing dots; no stoplist, no minimum length for exact matches;
- resolver tiers as documented (exact -> '.name' suffix -> case-insensitive equal/suffix -> case-insensitive substring),
  tiers 2-4 only for candidates of >= 4 characters, at most 5 matches, shortest ids first;
- neighbours: in+out, self-loops dropped, alphabetical, first 50 per seed, merged round-robin, seeds excluded.
Gold: work/c1/report/gold_mapped.csv (written by analysis/c1_report.py). Output: work/c1/reach_independent/per_task.csv.
Run: uv run python analysis/reach_independent.py
"""
import collections
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/c1/reach_independent"
OUT.mkdir(parents=True, exist_ok=True)
TASKS = {json.loads(l)["instance_id"]: json.loads(l) for l in open(ROOT / "data/raw/tasks.jsonl")}
g = pd.read_csv(ROOT / "work/c1/report/gold_mapped.csv")
sym = g[(g.kind != "module") & (g.in_graph == True)].drop_duplicates(["instance_id", "node"])
gold = sym.groupby("instance_id").node.agg(set)
tasks = sorted(gold.index)


def candidates(text):
    out = []
    for m in re.finditer(r"`([^`\n]{1,80})`", text):
        out += re.findall(r"[A-Za-z_][\w\.]*\w", m.group(1))
    out += re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b", text)
    out += re.findall(r"\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+\b", text)
    out += re.findall(r"\b[a-z0-9]+(?:_[a-z0-9]+)+\b", text)
    out += re.findall(r"\b([A-Za-z_]\w*)\(", text)
    return list(dict.fromkeys(c.strip(".") for c in out if c.strip(".")))


rows = []
for t in tasks:
    task = TASKS[t]
    G = json.load(open(ROOT / f"data/raw/graphs/{task['repo'].split('/')[1].lower()}_{task['base_commit']}.json"))
    ids = [n["id"] for n in G["nodes"]]
    idset, low = set(ids), [i.lower() for i in ids]
    adj = collections.defaultdict(set)
    for e in G["edges"]:
        if e["source"] != e["target"]:
            adj[e["source"]].add(e["target"])
            adj[e["target"]].add(e["source"])
    seeds = []
    for c in candidates(task["problem_statement"]):
        if c in idset:
            match = [c]
        elif len(c) < 4:
            continue
        else:
            cl = c.lower()
            match = [i for i in ids if i.endswith("." + c)] \
                or [i for i, l in zip(ids, low) if l == cl or l.endswith("." + cl)] \
                or [i for i, l in zip(ids, low) if cl in l]
            match = sorted(match, key=lambda i: (len(i), i))[:5]
        for x in match:
            if x not in seeds:
                seeds.append(x)
    lists = [sorted(adj[x])[:50] for x in seeds]
    expansion, seen = [], set(seeds)
    for i in range(max((len(l) for l in lists), default=0)):
        for l in lists:
            if i < len(l) and l[i] not in seen:
                seen.add(l[i])
                expansion.append(l[i])
    row = dict(task=t, n_seeds=len(seeds))
    for k in (10, 25, 50, 100):
        row[f"exp@{k}"] = bool(set(expansion[:k]) & gold[t])
        row[f"e2e@{k}"] = bool(set((seeds + expansion)[:k]) & gold[t])
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT / "per_task.csv", index=False)
print(f"n = {len(df)} tasks; tasks with >=1 seed: {(df.n_seeds > 0).sum()}")
for k in (10, 25, 50, 100):
    print(f"@{k}: expansion-only {int(df[f'exp@{k}'].sum())}, end-to-end {int(df[f'e2e@{k}'].sum())}")
