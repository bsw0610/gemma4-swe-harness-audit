"""C1 reach analysis v2 (post Codex audit A02/A06/A10): issue text -> gold symbols via graph/embedding tools.
Run from project root:  uv run python analysis/reach.py
Only stdlib/numpy/pandas/scipy. Reads data/raw (read-only); writes work/c1/reach/.
"""
import json, re, sys, collections, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/c1/reach_v3"
OUT.mkdir(parents=True, exist_ok=True)
RAW = ROOT / "data/raw"
KS = [10, 25, 50, 100]
CAP = 50
PKG = {"fastapi", "rich", "requests", "httpx"}

# ---------------------------------------------------------------- seeds
STOP = set("""if else elif for while try except finally with as import from return yield raise pass break continue lambda
class def not and or in is del global nonlocal assert await async None True False self cls print len str int float bool list
dict set tuple type range open super isinstance getattr setattr hasattr object bytes sorted enumerate zip map filter any all min max sum
repr id e.g i.e etc Exception ValueError TypeError KeyError RuntimeError AttributeError NotImplementedError OSError
IndexError StopIteration ImportError""".split())
TOK = re.compile(r"[A-Za-z_]\w*")
DOT = re.compile(r"(?<![\w.])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
BT = re.compile(r"`([^`\n]+)`")
BTOK = re.compile(r"^[\w.]+(\(\))?$")


def is_camel(t):
    return len(t) > 2 and t[0].isupper() and any(c.islower() for c in t) and any(c.isupper() for c in t[1:])


def is_snake(t):
    return "_" in t.strip("_") and any(c.isalpha() for c in t) and not t.isupper()


def clean(c):
    c = c.strip().rstrip(".,;:")
    if c.endswith("()"):
        c = c[:-2]
    return c.strip(".")


def extract_candidates(text):
    """ordered list of (candidate, kind) by first appearance; kinds joined."""
    text0 = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).replace("`", " "), text)  # keep code text, drop fences
    found = {}  # cand -> (pos, set(kinds))

    def add(c, pos, kind):
        c = clean(c)
        if len(c) < 2 or c in STOP:
            return
        if c not in found:
            found[c] = [pos, {kind}]
        else:
            found[c][0] = min(found[c][0], pos)
            found[c][1].add(kind)

    for m in BT.finditer(text0):
        s = m.group(1).strip()
        if BTOK.match(s):
            add(s, m.start(), "backtick")
    for m in DOT.finditer(text0):
        add(m.group(0), m.start(), "dotted")
    for m in TOK.finditer(text0):
        t = m.group(0)
        if is_camel(t):
            add(t, m.start(), "camel")
        if is_snake(t):
            add(t, m.start(), "snake")
    for m in CALL.finditer(text0):
        add(m.group(1), m.start(), "call")
    return sorted(((c, v[0], sorted(v[1])) for c, v in found.items()), key=lambda x: (x[1], x[0]))


# ---------------------------------------------------------------- graph
class G:
    pass


def load_graph(repo, commit):
    short = repo.split("/")[1].lower()
    name = f"{short}_{commit}"
    d = json.load(open(RAW / f"graphs/{name}.json"))
    ids = sorted({n["id"] for n in d["nodes"]})
    idx = {s: i for i, s in enumerate(ids)}  # index == alphabetical rank
    N = len(ids)
    outs = [set() for _ in range(N)]
    ins = [set() for _ in range(N)]
    for e in d["edges"]:
        u, v = idx[e["source"]], idx[e["target"]]
        outs[u].add(v)
        ins[v].add(u)
    g = G()
    g.name, g.ids, g.idx, g.N = name, ids, idx, N
    g.adj = [sorted(outs[i] | ins[i]) for i in range(N)]  # tool: out+in merged, dedup, alphabetical (self loops kept as in tool)
    g.und = [[j for j in a if j != i] for i, a in enumerate(g.adj)]
    g.lower = [s.lower() for s in ids]
    pkg = repo.split("/")[1].lower()
    cls = []
    for s in ids:
        comps = s.split(".")
        if comps[0] == pkg:
            cls.append("lib")
        elif comps[0] in ("tests", "test") or any(c.startswith("test_") or c == "conftest" for c in comps):
            cls.append("test")
        else:
            cls.append("other")
    g.cls = cls
    # WCC
    comp = [-1] * N
    c = 0
    for s in range(N):
        if comp[s] >= 0:
            continue
        comp[s] = c
        st = [s]
        while st:
            x = st.pop()
            for y in g.und[x]:
                if comp[y] < 0:
                    comp[y] = c
                    st.append(y)
        c += 1
    g.comp = comp
    emb = np.load(RAW / f"embeddings/{name}.npz")
    M = np.zeros((N, 256), dtype=np.float32)
    miss = 0
    for s, i in idx.items():
        if s in emb.files:
            M[i] = emb[s]
        else:
            miss += 1
    g.M, g.emb_missing = M, miss
    g.M32 = M.astype(np.float32)
    g.zero_vec = int((np.linalg.norm(M, axis=1) == 0).sum())
    return g


def bfs_dist(g, sources, maxd=50):
    dist = {s: 0 for s in sources}
    fr = list(dist)
    d = 0
    while fr and d < maxd:
        d += 1
        nf = []
        for x in fr:
            for y in g.und[x]:
                if y not in dist:
                    dist[y] = d
                    nf.append(y)
        fr = nf
    return dist


# ---------------------------------------------------------------- resolvers
def resolve_impl(g, cands):
    """-> list of (cand, [node idx]) ; exact id only."""
    out = []
    for c, pos, kinds in cands:
        if c in g.idx:
            out.append((c, [g.idx[c]], "exact"))
    return out


def resolve_doc(g, cands):
    """tiers: exact -> suffix on '.name' boundary -> case-insens (equal or suffix) -> substring (ci).
    Tiers 2-4 need len>=4. returns list of (cand, [idx sorted shortest id first], tier)."""
    out = []
    for c, pos, kinds in cands:
        if c in g.idx:
            out.append((c, [g.idx[c]], "exact"))
            continue
        if len(c) < 4:
            continue
        suf = "." + c
        m = [i for i, s in enumerate(g.ids) if s.endswith(suf)]
        tier = "suffix"
        if not m:
            cl = c.lower()
            sufl = "." + cl
            m = [i for i, s in enumerate(g.lower) if s == cl or s.endswith(sufl)]
            tier = "caseins"
        if not m:
            cl = c.lower()
            m = [i for i, s in enumerate(g.lower) if cl in s]
            tier = "substring"
        if m:
            m = sorted(m, key=lambda i: (len(g.ids[i]), g.ids[i]))
            out.append((c, m, tier))
    return out


def seeds_from(res, maxper=None):
    seeds, seen = [], set()
    for c, m, tier in res:
        for i in (m if maxper is None else m[:maxper]):
            if i not in seen:
                seen.add(i)
                seeds.append(i)
    return seeds


# ---------------------------------------------------------------- candidate generation
def rr(lists, exclude=()):
    """round-robin merge, dedup, drop excluded."""
    seen = set(exclude)
    out = []
    mx = max((len(l) for l in lists), default=0)
    for i in range(mx):
        for l in lists:
            if i < len(l):
                x = l[i]
                if x not in seen:
                    seen.add(x)
                    out.append(x)
    return out


def hop1(g, seeds, cap):
    return rr([g.adj[s][:cap] if cap else g.adj[s] for s in seeds], exclude=seeds)


def hop2(g, seeds, cap):
    r1 = hop1(g, seeds, cap)
    seen = set(seeds) | set(r1)
    sec = rr([g.adj[p][:cap] if cap else g.adj[p] for p in r1], exclude=seen)
    return r1 + sec, len(seeds) + len(r1)


def emb_lists(g, seeds, K=200):
    """Exactly as retrieval_utils.py:819-833: float32 normalise, float32 dot, sort by (-float32 sim, name) on UNROUNDED value."""
    E = g.M32
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    NE = E / norms
    ar = np.arange(g.N)  # g.ids is sorted -> index order == name order
    out = []
    for s in seeds:
        t = E[s]
        tn = np.linalg.norm(t)
        if tn == 0:
            tn = 1e-12
        nt = t / tn
        sims = np.dot(NE, nt.T).flatten()
        assert sims.dtype == np.float32
        sims[s] = -np.inf
        order = np.lexsort((ar, -sims))
        out.append([int(x) for x in order if x != s][:K])
    return out


# ---------------------------------------------------------------- stats
def boot_idx(n, B=10000, seed=0):
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


_BI = {}


def bi(n):
    if n not in _BI:
        _BI[n] = boot_idx(n)
    return _BI[n]


def ci(x):
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    if n < 2:
        return (x.mean(), np.nan, np.nan)
    m = x[bi(n)].mean(axis=1)
    return (x.mean(), *np.percentile(m, [2.5, 97.5]))


def cid(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    m = d[bi(n)].mean(axis=1)
    return (d.mean(), *np.percentile(m, [2.5, 97.5]))


def fmt(t, pct=True):
    m, lo, hi = t
    if np.isnan(lo):
        return f"{m*100:.1f}% (n=1, no CI)" if pct else f"{m:.2f}"
    return f"{m*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]" if pct else f"{m:.2f} [{lo:.2f}, {hi:.2f}]"


def fmtd(t):
    return f"{t[0]*100:+.1f}pp [{t[1]*100:+.1f}, {t[2]*100:+.1f}]"


# ================================================================= main
VIEWS = ("exp", "e2e")


def table(headers, rows):
    s = "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def main():
    tasks = {}
    for l in open(RAW / "tasks.jsonl"):
        t = json.loads(l)
        tasks[t["instance_id"]] = t
    gold = pd.read_csv(ROOT / "work/c1/report/gold_mapped.csv")
    ing = gold[(gold.kind != "module") & (gold.in_graph == True)].drop_duplicates(["instance_id", "node"])  # noqa: E712
    eval_ids = sorted(ing.instance_id.unique())
    n = len(eval_ids)
    rc = {tasks[i]["repo"].split("/")[1]: 0 for i in eval_ids}
    for i in eval_ids:
        rc[tasks[i]["repo"].split("/")[1]] += 1
    assert n == 115 and rc == {"fastapi": 59, "rich": 43, "requests": 12, "httpx": 1}, (n, rc)  # gold v4

    rows, seed_rows, trunc = [], [], collections.Counter()
    gcache = {}
    for iid in eval_ids:
        t = tasks[iid]
        repo = t["repo"]
        key = (repo, t["base_commit"])
        if key not in gcache:
            gcache.clear()
            gcache[key] = load_graph(*key)
        g = gcache[key]
        gold_idx = sorted(g.idx[x] for x in ing[ing.instance_id == iid].node)
        goldset = set(gold_idx)
        cands = extract_candidates(t["problem_statement"])
        r_impl, r_doc = resolve_impl(g, cands), resolve_doc(g, cands)
        S = {"impl": seeds_from(r_impl), "doc5": seeds_from(r_doc, maxper=5)}
        seed_rows.append(dict(instance_id=iid, repo=repo, n_cands=len(cands), n_impl_seeds=len(S["impl"]), n_doc5_seeds=len(S["doc5"]),
                              impl_trivial=bool(set(S["impl"]) & goldset), doc5_trivial=bool(set(S["doc5"]) & goldset), n_gold=len(gold_idx)))
        strat = {}
        for lab, sd in S.items():
            for s in sd:
                trunc[(lab, "1hop_calls")] += 1
                trunc[(lab, "1hop_truncated")] += int(len(g.adj[s]) > CAP)
            if not sd:
                for nm in ("nb1", "nb1u", "nb2", "nb2u", "emb"):
                    strat[f"{nm}_{lab}"] = (sd, [])
                continue
            strat[f"nb1_{lab}"] = (sd, hop1(g, sd, CAP))
            strat[f"nb1u_{lab}"] = (sd, hop1(g, sd, None))
            strat[f"nb2_{lab}"] = (sd, hop2(g, sd, CAP)[0])
            strat[f"nb2u_{lab}"] = (sd, hop2(g, sd, None)[0])
            L = emb_lists(g, sd)
            strat[f"emb_{lab}"] = (sd, rr(L, exclude=sd))
        for nm, (sd, cl) in strat.items():
            triv = bool(set(sd) & goldset)
            for k in KS:
                for view in VIEWS:
                    top = cl[:k] if view == "exp" else (sd + cl)[:k]
                    rows.append(dict(instance_id=iid, repo=repo, strategy=nm, k=k, view=view, n_seeds=len(sd), n_cands=len(top),
                                     trivial=triv, hit=int(bool(set(top) & goldset))))
        print(iid, len(S["impl"]), len(S["doc5"]), file=sys.stderr)
    PT = pd.DataFrame(rows)
    PT.to_csv(OUT / "per_task.csv", index=False)
    SR = pd.DataFrame(seed_rows)
    SR.to_csv(OUT / "seeds_per_task.csv", index=False)
    report(PT, SR, trunc, n, rc)


def report(PT, SR, trunc, n, rc):
    md = []
    A = md.append

    def H(s, k, v):
        return PT[(PT.strategy == s) & (PT.k == k) & (PT.view == v)].sort_values("instance_id").hit.values

    def cell(x):
        m, lo, hi = ci(x)
        return f"{int(x.sum())}/{n} = {m*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]"

    A("# C1 reach v2 (AST gold v3; float32 embedding order; end-to-end view)\n")
    A(f"Unit = task, n={n} ({rc}). Gold = `work/c1/report/gold_mapped.csv`, kind != module, in_graph, dedup (instance_id, node). "
      "Bootstrap as analysis/c1_report.py: tasks sorted by id, rng=default_rng(0), idx=integers(0,n,(10000,n)), percentile 2.5/97.5; "
      "task-resampling only, repository clustering not reflected (fastapi+rich = 102/115).\n")
    A("## Policies (stated)\n")
    A("""- **IMPL**: candidate must equal a node id exactly (what the tools do).
- **DOC-5** (README counterfactual, NOT specified by the docs): tier 1 exact id; tiers 2-4 (suffix on `.name` boundary, case-insensitive equal/suffix, case-insensitive substring) require **min length 4**; the first non-empty tier wins; **at most 5 matches per candidate, shortest ids first** (then alphabetical). Seeds = union in candidate first-appearance order.
- **Neighbours**: out+in merged, dedup, alphabetical, `max_neighbors=50` cap (default of the tool, not a fixed limit); `u` = no cap. Per-seed lists merged round-robin, deduped. 2-hop = 1-hop list followed by round-robin expansion of all 1-hop nodes.
- **Embedding (tool-faithful, A02)**: float32 normalisation and float32 dot as `retrieval_utils.py:819-833`, sort by (-sim, name) on the UNROUNDED float32 value, no rank-time rounding; per-seed top-200, round-robin merged.
- **Expansion-only (exp)**: candidates exclude seeds. **End-to-end (e2e)**: candidate list = resolved seeds first (resolver order) then the expansion, whole list cut to the same k.
""")
    A("## 1. Seed gating\n")
    rows = []
    for lab, c in (("IMPL (exact id)", "n_impl_seeds"), ("DOC-5", "n_doc5_seeds")):
        x = (SR[c] > 0).astype(int).values
        rows.append((lab, cell(x)))
    for lab, c in (("IMPL seed is already gold (trivial)", "impl_trivial"), ("DOC-5 seed is already gold (trivial)", "doc5_trivial")):
        x = SR[c].astype(int).values
        rows.append((lab, cell(x)))
    A(table(["measure", "tasks"], rows))
    A("Seed counts per task, median [min, max]: " + "; ".join(f"{c} {int(SR[c].median())} [{SR[c].min()}, {SR[c].max()}]" for c in ["n_impl_seeds", "n_doc5_seeds"]) + "\n")
    names = {"nb1_impl": "IMPL 1-hop (cap50)", "nb2_impl": "IMPL 2-hop (cap50)", "emb_impl": "IMPL embedding (float32 order)",
             "nb1_doc5": "DOC-5 1-hop (cap50)", "nb2_doc5": "DOC-5 2-hop (cap50)", "emb_doc5": "DOC-5 embedding (float32 order)"}
    for v, title in (("exp", "2a. Hit@k, expansion-only (seeds excluded)"), ("e2e", "2b. Hit@k, END-TO-END (seeds first, same budget k)")):
        A(f"## {title}\n")
        A(table(["strategy (policy)"] + [f"@{k}" for k in KS], [[f"{names[s]}"] + [cell(H(s, k, v)) for k in KS] for s in names]))
    A("DOC-5 policy: min length 4 (tiers 2-4), <=5 matches per candidate, shortest ids first. IMPL = exact id only.\n")
    A("## 3. Paired difference e2e minus exp\n")
    rows = []
    for s in names:
        rows.append([names[s]] + [fmtd(cid(H(s, k, "e2e"), H(s, k, "exp"))) for k in KS])
    A(table(["strategy"] + [f"@{k}" for k in KS], rows))
    A("## 4. Effect of removing the alphabetical max_neighbors=50 cap (task-level hit changes in the evaluated settings only)\n")
    A("Counts of tasks whose hit status changes when the cap is removed: gained (uncapped hits, capped misses) / lost (capped hits, uncapped misses). Not a claim about other settings.\n")
    rows = []
    for a, b, lab in (("nb1_impl", "nb1u_impl", "IMPL 1-hop"), ("nb2_impl", "nb2u_impl", "IMPL 2-hop"), ("nb1_doc5", "nb1u_doc5", "DOC-5 1-hop"), ("nb2_doc5", "nb2u_doc5", "DOC-5 2-hop")):
        for v in VIEWS:
            r = [lab, v]
            for k in KS:
                x, y = H(a, k, v), H(b, k, v)
                r.append(f"+{int(((y==1)&(x==0)).sum())} / -{int(((x==1)&(y==0)).sum())} (hits {int(x.sum())}->{int(y.sum())})")
            rows.append(r)
    A(table(["setting", "view"] + [f"@{k}" for k in KS], rows))
    A("Share of 1-hop seed calls whose neighbour list was actually truncated by the cap (merged degree incl. self-loop > 50):\n")
    rows = []
    for lab in ("impl", "doc5"):
        c, tr = trunc[(lab, "1hop_calls")], trunc[(lab, "1hop_truncated")]
        rows.append((lab, c, tr, f"{tr/c*100:.1f}%" if c else "n/a"))
    A(table(["seeds", "1-hop seed calls", "truncated", "share"], rows))
    A("## 5. Per-repo hit@50 (e2e / exp), hits/n\n")
    rows = []
    for s in names:
        r = [names[s]]
        for repo in ("fastapi/fastapi", "Textualize/rich", "psf/requests", "encode/httpx"):
            r.append("/".join(f"{int(PT[(PT.strategy==s)&(PT.k==50)&(PT.view==v)&(PT.repo==repo)].hit.sum())}" for v in ("e2e", "exp")) + f" of {int(((PT.strategy==s)&(PT.k==50)&(PT.view=='exp')&(PT.repo==repo)).sum())}")
        rows.append(r)
    A(table(["strategy", "fastapi", "rich", "requests", "httpx"], rows))
    (OUT / "summary.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
