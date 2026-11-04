"""embed_geometry v2 (audit A10 fixes). Run: uv run python analysis/embed_geometry.py [n_files]
Fixes vs v1: AST node kind (decorated classes), exact mean pairwise cosine,
explicit aggregation units, display-tie vs true float32 tie at rank 10/11.
Output: work/c1/embed_geometry_v2/{per_file.csv,summary.md}
"""
import json, glob, os, re, sys, ast, textwrap
import numpy as np, pandas as pd
ROOT = str(__import__('pathlib').Path(__file__).resolve().parents[1])
OUT = ROOT + '/work/c1/embed_geometry_v2'
K = 10

def unit(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); return X / np.maximum(n, 1e-12)

def exact_mean_pair_cos(U):
    n = len(U); s = U.sum(0)
    return float((s @ s - n) / (n * (n - 1)))   # mean over i!=j (rows unit-norm)

def node_kind(text):
    try:
        t = ast.parse(textwrap.dedent(text))
    except (SyntaxError, ValueError):
        return 'unknown'
    if not t.body: return 'unknown'
    s = t.body[0]
    if isinstance(s, ast.ClassDef): return 'class'
    if isinstance(s, ast.AsyncFunctionDef): return 'asyncfunc'
    if isinstance(s, ast.FunctionDef): return 'func'
    return 'unknown'

def is_docs(i): return any(re.fullmatch(r'tutorial\d+.*', p) for p in i.split('.'))
def is_test(i): return any(p.startswith('test') for p in i.split('.'))

def run_file(gf):
    base = os.path.basename(gf)[:-5]; repo = base.split('_')[0]
    g = json.load(open(gf)); z = np.load(f'{ROOT}/data/raw/embeddings/{base}.npz')
    nodes = sorted(g['nodes'], key=lambda nd: nd['id'])   # name order => stable sort reproduces tool's (-sim, name) tiebreak
    ids = [nd['id'] for nd in nodes]; n = len(ids); idx = {x: k for k, x in enumerate(ids)}
    X32 = np.stack([z[i] for i in ids]); dtype = str(X32.dtype)
    X32 = X32.astype(np.float32); X = X32.astype(np.float64)
    r = {'file': base, 'repo': repo, 'n': n, 'emb_dtype': dtype}
    nr = np.linalg.norm(X, axis=1)
    r.update(zero_frac=(nr == 0).mean(), nan_any=int(np.isnan(X).any()),
             dup_vec_frac=1 - len(np.unique(X32, axis=0)) / n)
    U = unit(X); mu = X.mean(0); Xc = X - mu; Uc = unit(Xc)
    r['raw_mean_pair_cos'] = exact_mean_pair_cos(U)
    r['cen_mean_pair_cos'] = exact_mean_pair_cos(Uc)
    r['cos_to_mean'] = float((U @ unit(mu[None])[0]).mean())
    ev = np.clip(np.linalg.eigvalsh(np.cov(Xc.T))[::-1], 0, None); p = ev / ev.sum()
    r['eff_rank'] = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    r['pc1_share'] = float(p[0])
    # node kinds
    kinds = np.array([node_kind(nd['text']) for nd in nodes])
    r['n_unknown_kind'] = int((kinds == 'unknown').sum()); r['n_class'] = int((kinds == 'class').sum())
    r['n_func'] = int((kinds == 'func').sum()); r['n_asyncfunc'] = int((kinds == 'asyncfunc').sum())
    r['n_class_startswith_old'] = int(sum(nd['text'].lstrip().startswith('class ') for nd in nodes))
    cls = {i for i, k in zip(ids, kinds) if k == 'class'}
    par = lambda i: i.rsplit('.', 1)[0] if '.' in i else ''
    cpar = []; mod = []
    for i in ids:
        p_ = par(i); cpar.append(p_ if p_ in cls else ''); m = p_
        while m in cls: m = par(m)
        mod.append(m)
    mod = np.array(mod); cpar = np.array(cpar)
    A = np.zeros((n, n), bool)
    for e in g['edges']:
        a, b = idx.get(e['source']), idx.get(e['target'])
        if a is not None and b is not None and a != b: A[a, b] = A[b, a] = True
    docs = np.array([is_docs(i) for i in ids]); tst = np.array([is_test(i) for i in ids])
    r.update(frac_docs=docs.mean(), frac_test=tst.mean(), frac_with_class_parent=(cpar != '').mean())
    # queries: seed 0, up to 2000 (index space = name-sorted order; seed fixed)
    qo = np.random.default_rng(0).choice(n, min(2000, n), replace=False)   # sampled in ORIGINAL graph-node order (as v1)
    q = np.array([idx[g['nodes'][k]['id']] for k in qo]); nq = len(q)
    r['n_query'] = nq
    modc = pd.Series(mod).map(pd.Series(mod).value_counts()).values
    clsc = pd.Series(cpar).map(pd.Series(cpar).value_counts()).values
    # tool-faithful ties: float32 normalized, float32 dot, full sort
    nn32 = np.linalg.norm(X32, axis=1, keepdims=True); nn32[nn32 == 0] = 1e-12
    V32 = X32 / nn32
    S = (V32[q] @ V32.T).astype(np.float32); S[np.arange(nq), q] = -np.inf
    srt = -np.sort(-S, axis=1)                      # descending float32
    a, b = srt[:, K - 1], srt[:, K]
    ok = n - 1 > K; r['tie_eligible_q'] = nq if ok else 0
    if ok:
        r['n_disp_tie'] = int((np.round(a.astype(np.float64), 4) == np.round(b.astype(np.float64), 4)).sum())
        r['n_true_tie'] = int((a == b).sum())
    else: r['n_disp_tie'] = r['n_true_tie'] = 0
    for tag, V in (('raw', U), ('cen', Uc)):
        Sv = S if tag == 'raw' else (V[q] @ V.T)
        if tag == 'cen': Sv[np.arange(nq), q] = -np.inf
        top = np.argsort(-Sv, axis=1, kind='stable')[:, :min(K, n - 1)]   # name order tiebreak
        sims = np.take_along_axis(Sv, top, 1); k_ = top.shape[1]
        r[tag + '_nn_slots'] = nq * k_
        r[tag + '_nn_adj_num'] = int(A[q[:, None], top].sum())
        r[tag + '_nn_mod_num'] = int((mod[top] == mod[q][:, None]).sum())
        hc = cpar[q] != ''
        r[tag + '_cls_q'] = int(hc.sum()); r[tag + '_cls_slots'] = int(hc.sum()) * k_
        r[tag + '_nn_cls_num'] = int((cpar[top][hc] == cpar[q][hc][:, None]).sum())
        r[tag + '_nn_docs_num'] = int(docs[top].sum()); r[tag + '_nn_test_num'] = int(tst[top].sum())
        dq = docs[q]; tq = tst[q]
        r[tag + '_docsq_slots'] = int(dq.sum()) * k_; r[tag + '_nn_docs_given_docsq_num'] = int(docs[top][dq].sum())
        r[tag + '_testq_slots'] = int(tq.sum()) * k_; r[tag + '_nn_test_given_testq_num'] = int(tst[top][tq].sum())
        r[tag + '_dupvec_num'] = int((np.round(sims, 6) >= 1.0).sum()) if tag == 'raw' else 0
    deg = A.sum(1)[q]
    r['base_adj'] = float((deg / (n - 1)).mean()); r['base_mod'] = float(((modc[q] - 1) / (n - 1)).mean())
    hc = cpar[q] != ''
    r['base_cls'] = float((((clsc[q] - 1) / (n - 1))[hc]).mean()) if hc.any() else np.nan
    return r

def med(s): return float(np.median(s))

def summarize(d):
    L = []; w = L.append
    G = len(d)
    w('# embed_geometry v2 summary\n')
    w(f'Denominators: {G} graphs (files); queries = seed 0, up to 2000 per graph; K={K}. '
      f'Total queries = {int(d.n_query.sum()):,}; total nodes = {int(d.n.sum()):,}.\n')
    w('## 1. Geometry (unit: per-graph statistic, then MEDIAN over graphs; "repo median" = median of per-graph values within repo)\n')
    cols = [('eff_rank', 'effective rank (exp entropy of PC variance share)'), ('pc1_share', 'PC1 variance share'),
            ('raw_mean_pair_cos', 'mean pairwise cosine, exact, i!=j, raw L2-normalized'),
            ('cen_mean_pair_cos', 'mean pairwise cosine, exact, mean-centered'),
            ('cos_to_mean', 'cosine to mean direction (mean over nodes)'), ('dup_vec_frac', 'duplicate-vector share (1 - unique/n)')]
    w(f'| statistic | median over {G} graphs | min | max | ' + ' | '.join(sorted(d.repo.unique())) + ' |')
    w('|' + '---|' * (4 + d.repo.nunique()))
    for c, nm in cols:
        rm = d.groupby('repo')[c].median()
        w(f'| {nm} | {med(d[c]):.4f} | {d[c].min():.4f} | {d[c].max():.4f} | ' + ' | '.join(f'{rm[r]:.4f}' for r in sorted(d.repo.unique())) + ' |')
    w(f'\nPooled duplicate-vector share (sum dup nodes / sum nodes) not computed; the per-graph median above is the unit. Mean per-graph (not median) exact pairwise cosine = {d.raw_mean_pair_cos.mean():.4f}; node-count-weighted = {np.average(d.raw_mean_pair_cos, weights=d.n):.4f}.\n')
    w('## 2. Node kind (AST on node text)\n')
    w(f'Nodes: {int(d.n.sum()):,}; class {int(d.n_class.sum()):,}; func {int(d.n_func.sum()):,}; asyncfunc {int(d.n_asyncfunc.sum()):,}; **unknown (SyntaxError/empty) {int(d.n_unknown_kind.sum()):,}**. '
      f'Old startswith("class ") would count {int(d.n_class_startswith_old.sum()):,} classes (missed {int(d.n_class.sum()-d.n_class_startswith_old.sum()):,}).\n')
    w('## 3. Rank 10/11 ties (sort key = float32 cosine; display = round 4)\n')
    E = d.tie_eligible_q.sum()
    d = d.assign(disp_share=d.n_disp_tie / d.tie_eligible_q.replace(0, np.nan), true_share=d.n_true_tie / d.tie_eligible_q.replace(0, np.nan))
    w('| measure | pooled over queries (num/denom) | per-graph median share |')
    w('|---|---|---|')
    w(f'| (a) display tie: rank-10 == rank-11 at 4 decimals | {int(d.n_disp_tie.sum()):,}/{int(E):,} = {d.n_disp_tie.sum()/E:.4%} | {med(d.disp_share.dropna()):.4%} |')
    w(f'| (b) true tie: float32 rank-10 == rank-11 exactly | {int(d.n_true_tie.sum()):,}/{int(E):,} = {d.n_true_tie.sum()/E:.4%} | {med(d.true_share.dropna()):.4%} |')
    w(f'\nDenominator = queries in graphs with n-1 > {K} ({int((d.tie_eligible_q>0).sum())} graphs).\n')
    w('## 4. Neighbor structure vs random baseline (top-10 by cosine, name tiebreak)\n')
    w('Each share = hits / neighbor slots. Overall = pooled over slots; "per-graph median" = median of per-graph shares. Baseline = per-graph mean over queries of (candidates-in-group)/(n-1).\n')
    def share(num, den): return d[num].sum() / d[den].sum(), med(d[num] / d[den].replace(0, np.nan))
    for tag in ('raw', 'cen'):
        w(f'### {tag} vectors\n')
        w('| measure | denominator (slots) | pooled | per-graph median | baseline per-graph median |')
        w('|---|---|---|---|---|')
        rows = [('neighbor adjacent (edge)', 'nn_adj_num', 'nn_slots', 'base_adj'), ('same module', 'nn_mod_num', 'nn_slots', 'base_mod'),
                ('same class parent (queries with class parent)', 'nn_cls_num', 'cls_slots', 'base_cls'),
                ('neighbor is docs_src', 'nn_docs_num', 'nn_slots', None), ('neighbor is test', 'nn_test_num', 'nn_slots', None),
                ('docs neighbor | docs query', 'nn_docs_given_docsq_num', 'docsq_slots', None), ('test neighbor | test query', 'nn_test_given_testq_num', 'testq_slots', None)]
        for nm, nu, de, bs in rows:
            a, b = share(f'{tag}_{nu}', f'{tag}_{de}')
            w(f'| {nm} | {int(d[f"{tag}_{de}"].sum()):,} | {a:.4f} | {b:.4f} | ' + (f'{med(d[bs].dropna()):.4f}' if bs else '-') + ' |')
        if tag == 'raw':
            a, b = share('raw_dupvec_num', 'raw_nn_slots'); w(f'| neighbor is exact duplicate vector (cos>=1 at 6dp) | {int(d.raw_nn_slots.sum()):,} | {a:.4f} | {b:.4f} | - |')
        w('')
    w('## 5. Same-class share per repo (raw vectors; slots = queries with class parent x 10)\n')
    w('| repo | graphs | slots | pooled | per-graph median | baseline median |')
    w('|---|---|---|---|---|---|')
    for r, s in d.groupby('repo'):
        w(f'| {r} | {len(s)} | {int(s.raw_cls_slots.sum()):,} | {s.raw_nn_cls_num.sum()/s.raw_cls_slots.sum():.4f} | {med(s.raw_nn_cls_num/s.raw_cls_slots.replace(0,np.nan)):.4f} | {med(s.base_cls.dropna()):.4f} |')
    w(f'| ALL | {G} | {int(d.raw_cls_slots.sum()):,} | {d.raw_nn_cls_num.sum()/d.raw_cls_slots.sum():.4f} | {med(d.raw_nn_cls_num/d.raw_cls_slots.replace(0,np.nan)):.4f} | {med(d.base_cls.dropna()):.4f} |')
    w('\n## 6. Confound shares (per-graph median of node shares)\n')
    w(f'docs_src node share {med(d.frac_docs):.4f}; test node share {med(d.frac_test):.4f}; nodes with class parent {med(d.frac_with_class_parent):.4f}.')
    return '\n'.join(L)

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fs = sorted(glob.glob(ROOT + '/data/raw/graphs/*.json'))
    if len(sys.argv) > 1: fs = fs[:int(sys.argv[1])]
    rows = []
    for f in fs:
        if not os.path.exists(f'{ROOT}/data/raw/embeddings/{os.path.basename(f)[:-5]}.npz'): continue
        rows.append(run_file(f)); print(rows[-1]['file'], rows[-1]['n'], flush=True)
    d = pd.DataFrame(rows)
    suf = '' if len(sys.argv) == 1 else '_dev'
    d.to_csv(f'{OUT}/per_file{suf}.csv', index=False)
    open(f'{OUT}/summary{suf}.md', 'w').write(summarize(d))
