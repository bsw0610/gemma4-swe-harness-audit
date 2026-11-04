"""Regenerate every number in reports/c1_results.md §3-§4 from raw data + Kaggle kernel outputs.

Inputs (read-only):
  data/raw/tasks.jsonl, data/raw/graphs/*.json
  work/c1/gold_ast_v4/gold_ast.csv      (kaggle/gold_ast, kernel version 4)
  work/c1/bm25_v3/bm25_ranks.csv        (kaggle/bm25, kernel version 3)
  work/c1/gold_coverage/gold.csv        (deprecated text-matching heuristic; only for the agreement check)
Outputs: work/c1/report/*.csv and stdout.

Statistics: unit = task; 95% CI = percentile interval of a task-resampling bootstrap (10,000 draws, seed 0,
tasks sorted by id). The CI is conditional on this task set (4 repositories, fastapi + rich = 89%);
it does not cover repository-to-repository variation. Leave-one-repo-out rows are reported for that reason.
Run: uv run python analysis/c1_report.py
"""
import collections
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, WORK = ROOT / "data/raw", ROOT / "work/c1"
OUT = WORK / "report"
OUT.mkdir(parents=True, exist_ok=True)
GOLD_CSV = WORK / "gold_ast_v4/gold_ast.csv"
BM25_CSV = WORK / "bm25_v3/bm25_ranks.csv"
KERNEL_VERSIONS = {"bsw0610/g4swe-gold-ast": 4, "bsw0610/g4swe-bm25": 3}
PKG = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich", "psf/requests": "requests", "encode/httpx": "httpx"}
KS = (10, 25, 50, 100)
B, SEED = 10_000, 0

TASKS = {json.loads(l)["instance_id"]: json.loads(l) for l in open(RAW / "tasks.jsonl")}


def module_of(path):
    p = path[4:] if path.startswith("src/") else path
    m = p[:-3].replace("/", ".")
    return m[: -len(".__init__")] if m.endswith(".__init__") else m


_graph_cache = {}


def graph(tid):
    t = TASKS[tid]
    f = RAW / "graphs" / f"{t['repo'].split('/')[1].lower()}_{t['base_commit']}.json"
    if f not in _graph_cache:
        g = json.load(open(f))
        adj = collections.defaultdict(set)
        for e in g["edges"]:
            if e["source"] != e["target"]:
                adj[e["source"]].add(e["target"])
                adj[e["target"]].add(e["source"])
        _graph_cache.clear()  # keep one graph in memory
        _graph_cache[f] = ({n["id"]: (n.get("text") or "") for n in g["nodes"]}, adj)
    return _graph_cache[f]


def boot(x, idx):
    x = np.asarray(x, float)
    bs = x[idx].mean(axis=1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def fmt(x, idx):
    m, lo, hi = boot(x, idx)
    return f"{int(round(m * len(x)))}/{len(x)} = {m * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]"


def fmt_diff(d, idx):
    m, lo, hi = boot(d, idx)
    return f"{m * 100:+.1f} pp [{lo * 100:+.1f}, {hi * 100:+.1f}]"


# ---------------------------------------------------------------- §3 gold and coverage
g = pd.read_csv(GOLD_CSV)
g["node"] = [f"{module_of(f)}.{q}" if q != "<module>" else None for f, q in zip(g.file, g.qualname)]
g["in_graph"] = [bool(n in graph(t)[0]) if n else None for n, t in zip(g.node, g.instance_id)]
g.to_csv(OUT / "gold_mapped.csv", index=False)
sym = g[g.kind != "module"].drop_duplicates(["instance_id", "node"])
print("## §3 Gold (AST, kernel v4)")
print(f"tasks 129 -> with symbol gold {sym.instance_id.nunique()} -> with in-graph gold {sym[sym.in_graph == True].instance_id.nunique()}")
print(f"unique (task, symbol): {len(sym)}; in graph {int(sym.in_graph.sum())} ({sym.in_graph.mean() * 100:.1f}%)")
print(sym.groupby("kind").in_graph.agg(n="size", in_graph="sum").to_string())
miss_tasks = sym.groupby("instance_id").in_graph.apply(lambda s: (~s.astype(bool)).any())
print(f"tasks with >=1 gold symbol missing from graph: {int(miss_tasks.sum())}/{len(miss_tasks)}")
print("block types:", g.block_type.value_counts().to_dict())

# heuristic vs AST agreement (A05): unit = (task, file, hunk) with >=1 in-graph symbol under EITHER method
h = pd.read_csv(WORK / "gold_coverage/gold.csv")
h = h[(h.level == "symbol") & (h.in_graph == True)]
H = h.groupby(["instance_id", "file", "hunk_index"]).symbol.agg(set)
# hunk-level sets come from the undeduplicated rows (audit 2 N03): dedup only within (task, file, hunk, node)
gh = g[(g.kind != "module") & (g.in_graph == True)].drop_duplicates(["instance_id", "file", "hunk_index", "node"])
A = gh.groupby(["instance_id", "file", "hunk_index"]).node.agg(set)
keys = H.index.union(A.index)
exact = sum(H.get(k, set()) == A.get(k, set()) for k in keys)
overlap = sum(bool(H.get(k, set()) & A.get(k, set())) for k in keys)
pd.DataFrame([dict(instance_id=k[0], file=k[1], hunk_index=k[2], heuristic=sorted(H.get(k, set())), ast=sorted(A.get(k, set())))
              for k in keys]).to_csv(OUT / "hunk_agreement.csv", index=False)
print(f"heuristic vs AST, hunks with an in-graph symbol under either method: {len(keys)}; "
      f"exact {exact} ({exact / len(keys) * 100:.1f}%), any overlap {overlap} ({overlap / len(keys) * 100:.1f}%)")

# ---------------------------------------------------------------- §4 reach: BM25
gold = sym[sym.in_graph == True].groupby("instance_id").node.agg(set)
tasks = sorted(gold.index)
n = len(tasks)
rng = np.random.default_rng(SEED)
IDX = rng.integers(0, n, size=(B, n))
repo_of = {t: TASKS[t]["repo"] for t in tasks}
gold_files = g[g.kind != "module"].groupby("instance_id").file.agg(set)

b = pd.read_csv(BM25_CSV)
bs = b[b.granularity == "symbol"].sort_values(["instance_id", "rank"])
bl = b[b.granularity == "symbol_lib"].sort_values(["instance_id", "rank"])
bf = b[b.granularity == "file"].sort_values(["instance_id", "rank"])
rank_sym = bs.groupby("instance_id").unit.agg(list)
rank_file = bf.groupby("instance_id").file.agg(list)
rank_lib = bl.groupby("instance_id").unit.agg(list)
hits = {}
for k in KS:
    hits[("bm25_snapshot", k)] = [bool(set(rank_sym.get(t, [])[:k]) & gold[t]) for t in tasks]
    hits[("bm25_file", k)] = [bool(set(rank_file.get(t, [])[:k]) & gold_files.get(t, set())) for t in tasks]

# graph-text BM25 on library-package nodes only (A04): corpus = graph nodes whose id starts with '<pkg>.'
TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
STOP = set("the a an and or of to in for is are be this that with on as it by not if from at we can should when use using".split())


def toks(s):
    out = []
    for w in TOK.findall(s):
        out += [p.lower() for p in [w] + [p for p in re.split(r"_|(?<=[a-z])(?=[A-Z])", w) if p] if len(p) > 1]
    return [x for x in out if x not in STOP]


def bm25_order(query, ids, texts, k1=1.2, bb=0.75):
    tf = [collections.Counter(toks(x)) for x in texts]
    N = len(tf)
    avg = sum(sum(c.values()) for c in tf) / N
    df = collections.Counter(w for c in tf for w in c)
    q = set(toks(query))
    sc = []
    for c in tf:
        dl = sum(c.values())
        sc.append(sum(math.log(1 + (N - df[w] + .5) / (df[w] + .5)) * c[w] * (k1 + 1) / (c[w] + k1 * (1 - bb + bb * dl / avg))
                      for w in q if w in c))
    return [ids[i] for i in sorted(range(N), key=lambda i: (-sc[i], ids[i]))]


graph_rank = {}
for t in tasks:
    texts, _ = graph(t)
    pkg = PKG[repo_of[t]] + "."
    ids = sorted(i for i in texts if i.startswith(pkg))
    graph_rank[t] = bm25_order(TASKS[t]["problem_statement"], ids, [texts[i] for i in ids])
for k in KS:
    hits[("bm25_graph_library", k)] = [bool(set(graph_rank[t][:k]) & gold[t]) for t in tasks]

# ---------------------------------------------------------------- gate (A07): equal unique-id budget, package filter
def expand(t, R, m, k, order):
    _, adj = graph(t)
    pkg = PKG[repo_of[t]] + "."
    cand = list(dict.fromkeys(R[:m]))
    seen = set(cand)
    nbrs = [sorted(v for v in adj.get(s, ()) if v.startswith(pkg)) for s in R[:m]]
    if order == "sequential":
        stream = [v for lst in nbrs for v in lst]
    else:  # round-robin over seeds
        stream = [lst[i] for i in range(max((len(x) for x in nbrs), default=0)) for lst in nbrs if i < len(lst)]
    for v in stream:
        if v not in seen:
            cand.append(v)
            seen.add(v)
    for u in R[m:]:
        if len(cand) >= k:
            break
        if u not in seen:
            cand.append(u)
            seen.add(u)
    return cand[:k]


gate_rows = []
for t in tasks:
    R = list(dict.fromkeys(rank_lib.get(t, [])))
    assert all(u.startswith(PKG[repo_of[t]] + ".") for u in R)
    for k in (25, 50, 100):
        assert len(R) >= k, (t, k, len(R))  # equal budget needs k library units (audit 2 N04)
        base = bool(set(R[:k]) & gold[t])
        for m in (5, 10, 25):
            if m >= k:
                continue
            for order in ("sequential", "round_robin"):
                cand = expand(t, R, m, k, order)
                assert len(cand) == k == len(set(cand)), (t, k, m, order, len(cand))
                gate_rows.append(dict(task=t, k=k, m=m, order=order, base=base, hit=bool(set(cand) & gold[t])))
gate = pd.DataFrame(gate_rows)
gate.to_csv(OUT / "gate.csv", index=False)

# ---------------------------------------------------------------- print §4
print("\n## §4 Reach (n = %d tasks with in-graph gold; repo counts %s)" % (n, dict(collections.Counter(repo_of.values()))))
for name in ("bm25_snapshot", "bm25_graph_library", "bm25_file"):
    print(name, " | ".join(f"@{k} {fmt(hits[(name, k)], IDX)}" for k in KS))
print("\nGate: BM25 (library units ranked by the kernel after filtering, unique ids, every list exactly k) top-m seeds + 1-hop calls neighbours (library only), filled to k with BM25")
for (k, m, order), d in gate.groupby(["k", "m", "order"]):
    d = d.set_index("task").reindex(tasks)
    diff = d.hit.astype(float).values - d.base.astype(float).values
    print(f"k={k:3d} m={m:2d} {order:11s} base {d.base.mean() * 100:5.1f}%  expanded {d.hit.mean() * 100:5.1f}%  diff {fmt_diff(diff, IDX)}  "
          f"(+{int((diff > 0).sum())} / -{int((diff < 0).sum())} tasks)")

print("\nLeave-one-repo-out, BM25 snapshot @50 and gate k=50 m=5 sequential diff:")
for r in sorted(set(repo_of.values())):
    keep = [i for i, t in enumerate(tasks) if repo_of[t] != r]
    x = np.array(hits[("bm25_snapshot", 50)], float)[keep]
    d = gate[(gate.k == 50) & (gate.m == 5) & (gate.order == "sequential")].set_index("task").reindex(tasks).iloc[keep]
    print(f"  without {r:16s} n={len(keep):3d}  BM25@50 {x.mean() * 100:.1f}%  gate diff {(d.hit.astype(float) - d.base.astype(float)).mean() * 100:+.1f} pp")
print("Per repo BM25 snapshot @50:", {r: f"{sum(h for h, t in zip(hits[('bm25_snapshot', 50)], tasks) if repo_of[t] == r)}/{sum(repo_of[t] == r for t in tasks)}"
                                      for r in sorted(set(repo_of.values()))})

verb = [any(re.search(r"\b" + re.escape(x.split(".")[-1]) + r"\b", TASKS[t]["problem_statement"]) for x in gold[t]) for t in tasks]
h50 = hits[("bm25_snapshot", 50)]
for v in (True, False):
    sel = [hh for hh, vv in zip(h50, verb) if vv == v]
    print(f"BM25@50, gold symbol's last name {'appears' if v else 'absent'} in problem_statement: {sum(sel)}/{len(sel)} = {np.mean(sel) * 100:.1f}%")
pd.DataFrame({"task": tasks, **{f"{a}@{k}": hits[(a, k)] for a, k in hits}}).to_csv(OUT / "hits.csv", index=False)

# ---------------------------------------------------------------- provenance manifest (A12)
import hashlib


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


manifest = {"kernel_versions": KERNEL_VERSIONS,
            "inputs": {str(p.relative_to(ROOT)): sha(p) for p in [RAW / "tasks.jsonl", GOLD_CSV, BM25_CSV,
                                                                 WORK / "gold_coverage/gold.csv",
                                                                 ROOT / "kaggle/gold_ast/gold_ast.py", ROOT / "kaggle/bm25/bm25.py"]},
            "graphs_sha256_of_sorted_hashes": hashlib.sha256("".join(sha(p) for p in sorted((RAW / "graphs").glob("*.json"))).encode()).hexdigest()}
(ROOT / "reports/c1_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("manifest -> reports/c1_manifest.json")

# ---------------------------------------------------------------- PR-description objection: does the text carry code?
fence = [("```" in TASKS[t]["problem_statement"]) for t in tasks]
tick = [("`" in TASKS[t]["problem_statement"]) for t in tasks]
for name, flag in (("code fence (```)", fence), ("any backtick", tick)):
    for v in (True, False):
        sel = [hh for hh, ff in zip(h50, flag) if ff == v]
        print(f"BM25@50, problem_statement {'has' if v else 'has no'} {name}: {sum(sel)}/{len(sel)} = {np.mean(sel) * 100:.1f}%")
