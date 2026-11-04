"""Table 1 reach, measured by calling the shipped swegemma graph functions instead of a counterfactual resolver.

Same 115 tasks, same gold, same identifier extraction and round-robin merge as analysis/reach.py. Differences:
the seeds and candidate lists come from the wheel's own resolve_node_name / get_neighbor / get_similar_nodes.
Seeds are credited end-to-end (seeds first, then expansion) as in Table 1, which favours the graph tools:
the tools do not report which node a short name resolved to.

Needs a throwaway env with networkx, numpy, cachetools, pandas (not project dependencies):
  python analysis/reach_0211.py --wheel-src work/harness_0_2_11/src --out work/c1/reach_0211   # shipped 0.2.11
  python analysis/reach_0211.py --wheel-src work/pkgsrc --out work/c1/reach_0207               # control: must give 2 / 2 / 2
"""
import argparse, collections, json, logging, sys, types
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from reach import extract_candidates, rr  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--wheel-src", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

# swegemma/__init__.py pulls in the whole harness; only the graph subpackage is needed here.
pkg = types.ModuleType("swegemma")
pkg.__path__ = [str(Path(a.wheel_src) / "swegemma")]
sys.modules["swegemma"] = pkg
import swegemma.graph as sg  # noqa: E402

logging.disable(logging.WARNING)
RAW = ROOT / "data/raw"
OUT = ROOT / a.out
OUT.mkdir(parents=True, exist_ok=True)
GD, ED = str(RAW / "graphs"), str(RAW / "embeddings")
KS = (10, 25, 50, 100)
CAP, SIM_K = 50, 200  # reach.py: 50 neighbours per call, 200 embedding results per seed
HAS_RESOLVER = hasattr(sg, "resolve_node_name")

tasks = {t["instance_id"]: t for t in map(json.loads, open(RAW / "tasks.jsonl"))}
gold = pd.read_csv(ROOT / "work/c1/report/gold_mapped.csv")
ing = gold[(gold.kind != "module") & (gold.in_graph == True)].drop_duplicates(["instance_id", "node"])  # noqa: E712
ids = sorted(ing.instance_id.unique())
rc = collections.Counter(tasks[i]["repo"].split("/")[1] for i in ids)
assert len(ids) == 115 and rc == {"fastapi": 59, "rich": 43, "requests": 12, "httpx": 1}, rc


def bare(s):
    return s.split(";", 1)[1] if ";" in s else s


rows, seed_rows = [], []
for iid in ids:
    t = tasks[iid]
    repo, commit = t["repo"], t["base_commit"]
    kw = dict(use_local_storage=True, repo_name=repo, graph_dir=GD, embeddings_dir=ED, base_commit=commit)
    g = sg.get_graph(repo, use_local_storage=True, graph_dir=GD, embeddings_dir=ED, base_commit=commit)
    assert g is not None and g.number_of_nodes() > 0, iid
    goldset = set(ing[ing.instance_id == iid].node)
    assert goldset <= set(g.nodes()), iid

    seeds, nb_lists, sim_lists = [], [], []
    n_resolved = n_lexical = 0
    cands = [c for c, _, _ in extract_candidates(t["problem_statement"])]
    for c in cands:
        if HAS_RESOLVER:
            r = sg.resolve_node_name(c, g)
        else:
            r = c if g.has_node(c) else None
        if r is not None:
            n_resolved += 1
            if r not in seeds:
                seeds.append(r)
            nb = sg.get_neighbor(node=f"{repo};{c}", max_neighbors=CAP, **kw)
            nb_lists.append([bare(x) for x in nb])
        sim = sg.get_similar_nodes(node=c, k=SIM_K, repo_name=repo, use_local_storage=True, graph_dir=GD,
                                   embeddings_dir=ED, base_commit=commit)
        if sim:
            n_lexical += int(r is None)
            sim_lists.append([bare(x["node_name"]) for x in sim])
    issue = sg.get_similar_nodes(node=t["problem_statement"], k=max(KS), repo_name=repo, use_local_storage=True,
                                 graph_dir=GD, embeddings_dir=ED, base_commit=commit)
    strat = {
        "neighbors_e2e": seeds + rr(nb_lists, exclude=seeds),
        "similar_e2e": seeds + rr(sim_lists, exclude=seeds),
        "issue_text_query": [bare(x["node_name"]) for x in issue],
    }
    seed_rows.append(dict(task=iid, repo=repo, n_cands=len(cands), n_resolved=n_resolved, n_seeds=len(seeds),
                          n_lexical=n_lexical, seed_is_gold=bool(set(seeds) & goldset), n_issue=len(issue)))
    for nm, lst in strat.items():
        row = dict(task=iid, strategy=nm)
        for k in KS:
            row[f"hit{k}"] = int(bool(set(lst[:k]) & goldset))
        rows.append(row)
    print(iid, len(cands), n_resolved, len(seeds), n_lexical, len(issue), file=sys.stderr, flush=True)

PT = pd.DataFrame(rows)
PT.to_csv(OUT / "per_task.csv", index=False)
SR = pd.DataFrame(seed_rows)
SR.to_csv(OUT / "seeds_per_task.csv", index=False)

rng = np.random.default_rng(0)
B = rng.integers(0, len(ids), size=(10000, len(ids)))
hits = pd.read_csv(ROOT / "work/c1/report/hits.csv").set_index("task").loc[ids]


def ci(x):
    d = np.asarray(x, float)[B].mean(axis=1) * 100
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


lines = [f"wheel source: {a.wheel_src}; resolver present: {HAS_RESOLVER}; n = {len(ids)}", ""]
lines += ["| strategy | " + " | ".join(f"hit@{k}" for k in KS) + " | hit@50 95% CI | hit@50 minus BM25@50 [95% CI] | only strategy, only BM25 |",
          "|---|" + "---|" * (len(KS) + 3)]
bm = hits["bm25_snapshot@50"].astype(int).to_numpy()
for nm, df in PT.groupby("strategy", sort=False):
    df = df.set_index("task").loc[ids]
    x = df["hit50"].to_numpy()
    lo, hi = ci(x)
    dl, dh = ci(x - bm)
    lines.append(f"| {nm} | " + " | ".join(str(int(df[f'hit{k}'].sum())) for k in KS)
                 + f" | {x.mean()*100:.1f}% [{lo:.1f}, {hi:.1f}] | {(x-bm).mean()*100:+.1f} pp [{dl:+.1f}, {dh:+.1f}] | {int(((x==1)&(bm==0)).sum())}, {int(((x==0)&(bm==1)).sum())} |")
lines += ["", f"BM25 (work/c1/report/hits.csv): @10 {int(hits['bm25_snapshot@10'].sum())}, @25 {int(hits['bm25_snapshot@25'].sum())}, "
          f"@50 {int(bm.sum())}, @100 {int(hits['bm25_snapshot@100'].sum())}", "",
          f"seeds: median candidates {SR.n_cands.median():.0f}, median resolved {SR.n_resolved.median():.0f}, median distinct seeds {SR.n_seeds.median():.0f}, "
          f"tasks with a gold seed {int(SR.seed_is_gold.sum())}, candidates that reached the lexical fallback (total) {int(SR.n_lexical.sum())}, "
          f"tasks where the whole-issue query returned nothing {int((SR.n_issue == 0).sum())}"]
(OUT / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
