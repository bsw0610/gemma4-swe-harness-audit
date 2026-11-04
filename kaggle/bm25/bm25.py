"""Lexical baseline (v3: .git/HEAD root, decorators in spans, ids merged, library ranking top-200): BM25 from problem_statement to code units on base_commit snapshots.

Units: (1) every def/class span parsed with ast (async included, nested included, innermost spans
are separate units), (2) whole files. Corpus = non-test, non-docs .py files of the snapshot.
Output: ranked top-100 units per task for both granularities (no gold used here).
"""
import ast, csv, glob, json, math, re, tarfile, collections
from pathlib import Path

ROOT = Path(sorted(glob.glob("/kaggle/input/**/gemma-4-developer-agent/tasks.jsonl", recursive=True))[0]).parent
SNAP, OUT = ROOT / "snapshots", Path("/kaggle/working")
TASKS = [json.loads(l) for l in open(ROOT / "tasks.jsonl")]
TEST_RE = re.compile(r"(^|/)tests?/|(^|/)test_[^/]*$|_test\.py$")
PKG = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich", "psf/requests": "requests", "encode/httpx": "httpx"}
TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
STOP = set("the a an and or of to in for is are be this that with on as it by not if from at we can should when use using".split())


def toks(s):
    out = []
    for w in TOK.findall(s):
        parts = [w] + [p for p in re.split(r"_|(?<=[a-z])(?=[A-Z])", w) if p]
        out += [p.lower() for p in parts if len(p) > 1]
    return [t for t in out if t not in STOP]


def units(path, src):
    mod = (path[4:] if path.startswith("src/") else path)[:-3].replace("/", ".")
    mod = mod[:-9] if mod.endswith(".__init__") else mod
    lines = src.splitlines()
    res = [(f"FILE:{path}", path, src)]
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return res

    def walk(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = f"{prefix}.{ch.name}" if prefix else ch.name
                start = min([ch.lineno] + [d.lineno for d in ch.decorator_list])
                body = "\n".join(lines[start - 1: ch.end_lineno])
                res.append((f"{mod}.{q}", path, body))
                walk(ch, q)
            else:
                walk(ch, prefix)
    walk(tree, "")
    return res


def bm25_rank(query, docs, k1=1.2, b=0.75, top=100):
    tf = [collections.Counter(toks(d)) for d in docs]
    N = len(docs); avgdl = sum(sum(c.values()) for c in tf) / max(N, 1)
    df = collections.Counter(t for c in tf for t in c)
    q = collections.Counter(toks(query))
    scores = []
    for i, c in enumerate(tf):
        dl = sum(c.values()); s = 0.0
        for t, qf in q.items():
            if t in c:
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * c[t] * (k1 + 1) / (c[t] + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    order = sorted(range(N), key=lambda i: (-scores[i], i))[:top]
    return [(i, scores[i]) for i in order]


rows = []
prefixes = collections.Counter()
for t in TASKS:
    with tarfile.open(SNAP / f"{t['instance_id']}.tgz") as tf:
        mem = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".py")]
        # repository root = directory holding .git/HEAD (v2: no guessing from a shared first folder, audit A11)
        heads = sorted((m.name for m in tf.getmembers() if m.name == ".git/HEAD" or m.name.endswith("/.git/HEAD")), key=len)
        if not heads:
            raise RuntimeError(f"no .git/HEAD in {t['instance_id']}")
        prefix = heads[0][: -len(".git/HEAD")]
        prefixes[prefix] += 1
        corpus = []
        for m in mem:
            rel = m.name[len(prefix):]
            if rel.startswith(".git/") or TEST_RE.search(rel) or rel.startswith(("docs/", "docs_src/")):
                continue
            src = tf.extractfile(m).read().decode("utf-8", errors="replace")
            corpus += units(rel, src)
    # merge units sharing an id (overloads / redefinitions) so each id occupies one rank (audit A07)
    merged = {}
    for uid, path, text in corpus:
        if uid in merged:
            merged[uid] = (uid, merged[uid][1], merged[uid][2] + "\n" + text)
        else:
            merged[uid] = (uid, path, text)
    corpus = list(merged.values())
    query = t["problem_statement"]
    pkg = PKG[t["repo"]] + "."
    for gran in ("file", "symbol", "symbol_lib"):
        if gran == "symbol_lib":  # library-package units only, filtered BEFORE truncation (audit 2 N04)
            idx = [i for i, u in enumerate(corpus) if not u[0].startswith("FILE:") and u[0].startswith(pkg)]
        else:
            idx = [i for i, u in enumerate(corpus) if (u[0].startswith("FILE:")) == (gran == "file")]
        ranked = bm25_rank(query, [corpus[i][2] for i in idx], top=200 if gran == "symbol_lib" else 100)
        for r, (j, s) in enumerate(ranked, 1):
            u = corpus[idx[j]]
            rows.append(dict(instance_id=t["instance_id"], granularity=gran, rank=r, unit=u[0], file=u[1], score=round(s, 4)))
    print(t["instance_id"], len(corpus), flush=True)

with open(OUT / "bm25_ranks.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("rows", len(rows), "archive prefixes", dict(prefixes))
