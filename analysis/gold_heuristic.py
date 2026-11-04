"""DEPRECATED text-matching gold heuristic (kept only to reproduce the heuristic-vs-AST agreement in reports/c1_results.md §3).
Writes work/c1/gold_coverage/. Run: uv run python analysis/gold_heuristic.py"""
import json,glob,re,os,collections,random,sys
import numpy as np, pandas as pd
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
OUT=ROOT+'/work/c1/gold_coverage'
os.makedirs(OUT, exist_ok=True)
tasks=[json.loads(l) for l in open(ROOT+'/data/raw/tasks.jsonl')]
TRIV={'',')',']','}','pass','"""',"'''"}
def trivial(s):
    return s in TRIV or s.startswith('#') or s in ('return','...','else:','try:','finally:') 
def is_test(p):
    b=os.path.basename(p)
    return bool(re.search(r'(^|/)tests?/',p)) or b.startswith('test_') or b.endswith('_test.py')
def is_docs(p): return p.startswith('docs/') or p.startswith('docs_src/')
def module_of(path):
    p=path[:-3]
    if p.startswith('src/'): p=p[4:]   # src-layout (requests, httpx)
    m=p.replace('/','.')
    if m.endswith('.__init__'): m=m[:-9]
    return m
def parse_patch(patch):
    files=[];cur=None
    for line in patch.split('\n'):
        if line.startswith('--- '):
            old=line[4:]; cur=None; continue
        if line.startswith('+++ '):
            new=line[4:]; path=new[2:] if new.startswith('b/') else new
            if new=='/dev/null': path=old[2:]
            cur={'path':path,'new_file':old=='/dev/null','hunks':[]}; files.append(cur); continue
        if cur is None: continue
        m=re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)',line)
        if m:
            cur['hunks'].append({'old_start':int(m.group(1)),'header':m.group(5),'pre':[],'minus':[],'plus':[]}); continue
        if not cur['hunks']: continue
        h=cur['hunks'][-1]
        if line.startswith('-'): h['pre'].append(('-',line[1:].strip())); h['minus'].append(line[1:].strip())
        elif line.startswith('+'): h['plus'].append(line[1:].strip())
        elif line.startswith(' ') : h['pre'].append((' ',line[1:].strip()))
    return files
# ---------- load graphs
graphs={}
def gpath(t):
    short=t['repo'].split('/')[1].lower()
    return f"{ROOT}/data/raw/graphs/{short}_{t['base_commit']}.json"
def load(t):
    p=gpath(t)
    if p not in graphs:
        if not os.path.exists(p): graphs[p]=None
        else:
            d=json.load(open(p)); nodes={}
            for n in d['nodes']: nodes.setdefault(n['id'],n)
            bymod=collections.defaultdict(list)
            for nid,n in nodes.items(): bymod[nid]  # placeholder
            graphs[p]={'d':d,'nodes':nodes}
    return graphs[p]
def mod_nodes(g,mod):
    if 'idx' not in g:
        # index by every dotted prefix
        idx=collections.defaultdict(list)
        for nid in g['nodes']:
            parts=nid.split('.')
            for i in range(1,len(parts)): idx['.'.join(parts[:i])].append(nid)
        g['idx']=idx
    return [n for n in g['idx'].get(mod,[]) if g['nodes'][n]['text']]
def kind_of(s):
    if re.match(r'async def ',s): return 'async_def'
    if s.startswith('def '): return 'def'
    if s.startswith('class '): return 'class'
    return 'unknown'
defre=re.compile(r'^(async def|def|class)\s+([A-Za-z_]\w*)')
def split_lines(txt): return [l.strip() for l in txt.split('\n') if l.strip()]
rows=[];summ=[]
nmiss_graph=0
for t in tasks:
    iid=t['instance_id'];repo=t['repo'].split('/')[1]
    files=parse_patch(t['patch'])
    g=load(t)
    code=[f for f in files if f['path'].endswith('.py') and not is_test(f['path']) and not is_docs(f['path'])]
    docs=[f for f in files if is_docs(f['path'])]
    tests=[f for f in files if is_test(f['path']) and not is_docs(f['path'])]
    nsym_in=0;nsym_out=0;newonly=0;nofilenode=0;nhunks=0
    for f in code:
        mod=module_of(f['path']); mn=mod_nodes(g,mod) if g else []
        has_node=len(mn)>0
        if g and not has_node: nofilenode+=1
        rows.append(dict(instance_id=iid,repo=repo,file=f['path'],module=mod,symbol='',level='file',in_graph=has_node,kind='file',match_method='',hunk_index=-1,n_tied=0,new_file=f['new_file']))
        if not mn or True:
            tl={n:split_lines(g['nodes'][n]['text']) for n in mn} if g else {}
            tj={n:'\n'.join(v) for n,v in tl.items()}
            name_set={n.split('.')[-1] for n in mn}
        for hi,h in enumerate(f['hunks']):
            nhunks+=1
            if f['new_file'] or h['old_start']==0 and not h['pre']:
                newonly+=1; rows.append(dict(instance_id=iid,repo=repo,file=f['path'],module=mod,symbol='',level='symbol',in_graph=False,kind='new_only',match_method='none',hunk_index=hi,n_tied=0,new_file=f['new_file'])); continue
            n_rows_before=len(rows)
            found={}  # node -> (method,tied)
            minus=[s for s in dict.fromkeys(h['minus']) if not trivial(s)]
            for s in minus:
                c=[n for n in mn if s in tl[n]]
                if c:
                    mlen=min(len(g['nodes'][n]['text']) for n in c)
                    best=[n for n in c if len(g['nodes'][n]['text'])==mlen]
                    for n in best: found.setdefault(n,('removed_line',len(best)))
            if not found:
                ctxl=[s for k,s in h['pre'] if k==' ']
                if not h['minus']:
                    pairs=[]
                    # consecutive context lines (in the diff pre-image order), nontrivial both
                    for i in range(len(h['pre'])-1):
                        a,b=h['pre'][i],h['pre'][i+1]
                        if a[0]==' ' and b[0]==' ' and not trivial(a[1]) and not trivial(b[1]): pairs.append((a[1],b[1]))
                    for a,b in pairs:
                        c=[n for n in mn if any(tl[n][i]==a and tl[n][i+1]==b for i in range(len(tl[n])-1))]
                        if c:
                            mlen=min(len(g['nodes'][n]['text']) for n in c)
                            best=[n for n in c if len(g['nodes'][n]['text'])==mlen]
                            for n in best: found.setdefault(n,('context',len(best)))
            # missing defs from '-' lines
            for s in h['minus']:
                m=defre.match(s)
                if m:
                    nm=m.group(2)
                    if nm not in name_set:
                        rows.append(dict(instance_id=iid,repo=repo,file=f['path'],module=mod,symbol=nm,level='symbol',in_graph=False,kind=kind_of(s),match_method='removed_line',hunk_index=hi,n_tied=0,new_file=False))
                        nsym_out+=1
            # header
            hm=defre.match(h['header'].strip()) if h['header'] else None
            if not found and hm:
                nm=hm.group(2); cands=[n for n in mn if n.split('.')[-1]==nm]
                if cands:
                    best=min(cands,key=lambda n:len(g['nodes'][n]['text'])) 
                    found[best]=('header',len(cands))
                else:
                    rows.append(dict(instance_id=iid,repo=repo,file=f['path'],module=mod,symbol=nm,level='symbol',in_graph=False,kind=kind_of(h['header'].strip()),match_method='header',hunk_index=hi,n_tied=0,new_file=False)); nsym_out+=1
            for n,(meth,tied) in found.items():
                rows.append(dict(instance_id=iid,repo=repo,file=f['path'],module=mod,symbol=n,level='symbol',in_graph=True,kind=kind_of(next((l.strip() for l in g['nodes'][n]['text'].split('\n') if l.strip() and not l.strip().startswith('@')),'')),match_method=meth,hunk_index=hi,n_tied=tied,new_file=False))
            if len(rows)==n_rows_before:
                rows.append(dict(instance_id=iid,repo=repo,file=f['path'],module=mod,symbol='',level='symbol',in_graph=False,kind=('module_level' if not h['header'].strip() else 'unmatched'),match_method='none',hunk_index=hi,n_tied=0,new_file=False))
    gdf_task=[r for r in rows if r['instance_id']==iid and r['level']=='symbol' and r['symbol']!='']
    nin=len({r['symbol'] for r in gdf_task if r['in_graph']}); nout=len({(r['file'],r['symbol']) for r in gdf_task if not r['in_graph']})
    summ.append(dict(instance_id=iid,repo=repo,graph_exists=g is not None,n_code_files=len(code),n_docs_files=len(docs),n_test_files=len(tests),
        n_hunks_code=nhunks,n_gold_in_graph=nin,n_gold_missing=nout,n_newonly_hunks=newonly,n_code_files_no_node=nofilenode,
        docs_only=bool(files) and len(docs)==len(files),test_only=bool(files) and len(tests)==len(files),no_code_file=len(code)==0))
gold=pd.DataFrame(rows);ts=pd.DataFrame(summ)
gold.to_csv(OUT+'/gold.csv',index=False);ts.to_csv(OUT+'/task_summary.csv',index=False)

# ================= B. census
cen=[];typ=collections.Counter();asyncnodes=0;nnodes_total=0
for p in sorted(glob.glob(ROOT+'/data/raw/graphs/*.json')):
    d=json.load(open(p));ns=d['nodes'];es=d['edges'];ids=[n['id'] for n in ns]
    idset=set(ids);ec=collections.Counter(e['type'] for e in es);typ.update(ec)
    deg=collections.Counter()
    for e in es: deg[e['source']]+=1;deg[e['target']]+=1
    selfl=sum(e['source']==e['target'] for e in es)
    dangling=sum((e['source'] not in idset) or (e['target'] not in idset) for e in es)
    iso=sum(1 for i in ids if deg[i]==0)
    cont=sum(1 for e in es if e['target'].startswith(e['source']+'.'))
    modlevel=sum(1 for i in ids if i.count('.')==0)
    asy=sum(1 for n in ns if re.search(r'(^|\n)\s*async def ',n['text'] or ''))
    asy_first=sum(1 for n in ns if re.match(r'\s*(@.*\n\s*)*async def ',n['text'] or ''))
    cen.append(dict(file=os.path.basename(p),n_nodes=len(ns),n_edges=len(es),types=dict(ec),self_loops=selfl,dangling_edges=dangling,isolated=iso,
        id_ne_name=sum(n['id']!=n['name'] for n in ns),dup_ids=len(ids)-len(idset),parallel_edges=sum(1 for e in es if e.get('key',0)>0),
        containment_edges=cont,single_segment_ids=modlevel,nodes_text_async_anywhere=asy,nodes_text_starts_async=asy_first,
        empty_text=sum(1 for n in ns if not n['text'])))
cdf=pd.DataFrame(cen);cdf['types']=cdf['types'].astype(str);cdf.to_csv(OUT+'/graph_census.csv',index=False)
# containment direction check: class->method calls edge exists?
# ================= C. coverage with bootstrap
rng=np.random.default_rng(0)
def boot(vals,B=10000):
    v=np.asarray(vals,float);n=len(v)
    if n==0: return (np.nan,np.nan,np.nan,0)
    idx=rng.integers(0,n,(B,n));m=v[idx].mean(1)
    return (v.mean(),*np.percentile(m,[2.5,97.5]),n)
def boot_ratio(num,den,B=10000):
    num=np.asarray(num,float);den=np.asarray(den,float);n=len(num)
    if den.sum()==0: return (np.nan,np.nan,np.nan,0)
    idx=rng.integers(0,n,(B,n));r=num[idx].sum(1)/np.maximum(den[idx].sum(1),1e-9)
    return (num.sum()/den.sum(),*np.percentile(r,[2.5,97.5]),int(den.sum()))
sym=gold[(gold.level=='symbol')&(gold.symbol!='')]
ts['has_code']=ts.n_code_files>0
lines=[]
def fmt(name,r): return f"| {name} | {r[3]} | {r[0]:.3f} | [{r[1]:.3f}, {r[2]:.3f}] |"
res=[]
def block(sub,label):
    c=sub[sub.has_code]
    res.append(fmt(f"{label}: tasks with >=1 code file (n tasks)",boot([1]*len(c))))
    res.append(fmt(f"{label}: share tasks with >=1 in-graph gold symbol",boot((c.n_gold_in_graph>0).astype(int))))
    res.append(fmt(f"{label}: share tasks with >=1 gold symbol (in or missing)",boot(((c.n_gold_in_graph+c.n_gold_missing)>0).astype(int))))
    res.append(fmt(f"{label}: share tasks with all code files having a node",boot((c.n_code_files_no_node==0).astype(int))))
    res.append(fmt(f"{label}: share tasks with >=1 code file with NO graph node",boot((c.n_code_files_no_node>0).astype(int))))
    res.append(fmt(f"{label}: share of gold symbols missing (pooled)",boot_ratio(c.n_gold_missing,c.n_gold_in_graph+c.n_gold_missing)))
block(ts,'ALL')
for r in ['fastapi','rich','requests','httpx']:
    if (ts[ts.repo==r].has_code).any(): block(ts[ts.repo==r],r)
# missing by kind
mk=sym[~sym.in_graph].drop_duplicates(['instance_id','file','symbol','kind']).groupby('kind').size()
tot_in=sym[sym.in_graph].drop_duplicates(['instance_id','symbol']).shape[0]
tot_out=sym[~sym.in_graph].drop_duplicates(['instance_id','file','symbol']).shape[0]
# async check
asy_pre=[];asy_hdr_missing=[]
for t in tasks:
    g=load(t)
    for f in parse_patch(t['patch']):
        if not f['path'].endswith('.py') or is_test(f['path']) or is_docs(f['path']): continue
        mod=module_of(f['path']);names={n.split('.')[-1] for n in mod_nodes(g,mod)} if g else set()
        for h in f['hunks']:
            for k,s in h['pre']:
                m=re.match(r'async def (\w+)',s)
                if m: asy_pre.append((t['instance_id'],f['path'],m.group(1),k,m.group(1) in names))
            m=re.match(r'async def (\w+)',h['header'].strip())
            if m: asy_hdr_missing.append((t['instance_id'],f['path'],m.group(1),m.group(1) in names))
apdf=pd.DataFrame(asy_pre,columns=['iid','file','name','pre_kind','node_exists'])
ahdf=pd.DataFrame(asy_hdr_missing,columns=['iid','file','name','node_exists'])
apdf.to_csv(OUT+'/async_pre_image.csv',index=False)
# repo-level gold-edge: do any async-def names from patch pre-images exist as nodes
mc=ts[ts.has_code]
# ================= D. eyeball
rs=random.Random(0);elig=[t for t in tasks if (ts.set_index('instance_id').loc[t['instance_id'],'n_gold_in_graph']>0)]
pick=rs.sample(elig,5);E=[]
for t in pick:
    E.append(f"## {t['instance_id']}\n")
    sg=sym[sym.instance_id==t['instance_id']]
    for f in parse_patch(t['patch']):
        if not f['path'].endswith('.py') or is_test(f['path']) or is_docs(f['path']): continue
        E.append(f"### {f['path']} (module `{module_of(f['path'])}`)\n")
        for hi,h in enumerate(f['hunks'][:6]):
            gs=sg[(sg.file==f['path'])&(sg.hunk_index==hi)]
            E.append(f"- hunk {hi}: header `@@ {h['header']}`; first removed: `{(h['minus'] or ['(none: addition-only)'])[0][:100]}`; first added: `{(h['plus'] or ['-'])[0][:100]}`")
            E.append("  - gold: "+("; ".join(f"`{r.symbol}` [{r.match_method}, in_graph={r.in_graph}, tied={r.n_tied}]" for r in gs.itertuples()) or "(none: module-level / new-only)"))
    E.append('')
open(OUT+'/eyeball.md','w').write("# Eyeball check: 5 random tasks (seed 0) with >=1 in-graph gold symbol\n\n"+"\n".join(E))
# ================= summary
S=[]
S.append(f"# Gold coverage summary\nGraph files: {len(cdf)} of 127 expected; tasks {len(tasks)}; graph_exists for {int(ts.graph_exists.sum())} tasks.\n")
S.append("## Coverage (task-level bootstrap, 10000 resamples, seed 0)\n| metric | n | estimate | 95% CI |\n|---|---|---|---|\n"+"\n".join(res))
S.append(f"\n## Gold symbols\nIn-graph unique (task,symbol): {tot_in}; missing unique (task,file,symbol): {tot_out}; missing by kind: {mk.to_dict()}")
S.append("\nmatch_method among in-graph symbol rows: "+str(sym[sym.in_graph].match_method.value_counts().to_dict())+"; tied-candidate rows (n_tied>1): "+str(int((sym[sym.in_graph].n_tied>1).sum())))
S.append("\nHunks without symbol gold: "+str(gold[gold.kind.isin(['module_level','new_only','unmatched'])].kind.value_counts().to_dict()))
S.append(f"\nTask flags: {ts[['docs_only','test_only','no_code_file']].sum().to_dict()}")
S.append(f"Code files (non-test, non-docs) total: {int(ts.n_code_files.sum())}; with NO node in graph: {int(ts.n_code_files_no_node.sum())} ({ts.n_code_files_no_node.sum()/ts.n_code_files.sum():.3f})")
nn=gold[(gold.level=='file')&(~gold.in_graph)]
S.append("No-node code files by repo: "+str(nn.groupby('repo').size().to_dict())+"\nBy file (top): "+str(nn.file.value_counts().head(12).to_dict()))
S.append("\n## Census\n"+f"graphs {len(cdf)}; nodes {cdf.n_nodes.sum()}; edges {cdf.n_edges.sum()}; edge type counts {dict(typ)}; self loops {cdf.self_loops.sum()}; dangling {cdf.dangling_edges.sum()}; isolated nodes {cdf.isolated.sum()} ({cdf.isolated.sum()/cdf.n_nodes.sum():.3f}); id!=name {cdf.id_ne_name.sum()}; dup ids {cdf.dup_ids.sum()}; parallel edges(key>0) {cdf.parallel_edges.sum()}; class->member containment edges {cdf.containment_edges.sum()}; single-segment (module-level) ids {cdf.single_segment_ids.sum()}; empty text {cdf.empty_text.sum()}; nodes with 'async def' line anywhere in text {cdf.nodes_text_async_anywhere.sum()}; nodes whose text starts (after decorators) with async def {cdf.nodes_text_starts_async.sum()}")
S.append("README documents CALLS / DEFINED_IN / IMPORTS; real types: "+str(dict(typ)))
S.append("\n## async def check\n"+f"async def pre-image lines in code-file hunks (patch, '-' or ' '): {len(apdf)} (unique task/file/name: {apdf.drop_duplicates(['iid','file','name']).shape[0]}); of which a node with that bare name exists in the module: {int(apdf.drop_duplicates(['iid','file','name']).node_exists.sum())}. Header-context async defs: {len(ahdf)}, with node: {int(ahdf.node_exists.sum())}.")
open(OUT+'/summary.md','w').write("\n".join(S))
print("\n".join(S))
