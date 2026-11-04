

# ===== Mutation test of gold attribution (model-independent check) =====
# Synthetic patches with KNOWN gold are generated from real base-snapshot library files, diffed with difflib
# (as a unified diff), and fed through parse_patch + attribute. Expected gold is computed with a separate ast
# parent walk (not gold_core.innermost).
import csv, difflib, glob, json, random, tarfile
from pathlib import Path

ROOT = Path(sorted(glob.glob("/kaggle/input/**/gemma-4-developer-agent/tasks.jsonl", recursive=True))[0]).parent
SNAP, OUT = ROOT / "snapshots", Path("/kaggle/working")
TASKS = [json.loads(l) for l in open(ROOT / "tasks.jsonl")]
PKG = {"fastapi/fastapi": "fastapi/", "Textualize/rich": "rich/", "psf/requests": "src/requests/", "encode/httpx": "src/httpx/"}
rng = random.Random(0)


def defs_with_parents(src):
    """[(node, qualname, parent_qualname or None)] via an explicit parent map."""
    tree = ast.parse(src)
    parent = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            parent[c] = p
    out = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            chain, p = [n.name], parent.get(n)
            while p is not None:
                if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    chain.append(p.name)
                p = parent.get(p)
            out.append((n, ".".join(reversed(chain))))
    return out


def owner_of_line(src, line):
    """Expected owner of an existing line: the deepest def/class whose [first decorator, end] contains it."""
    best = None
    for n, q in defs_with_parents(src):
        s = min([n.lineno] + [d.lineno for d in n.decorator_list])
        if s <= line <= n.end_lineno and (best is None or q.count(".") > best.count(".")):
            best = q
    return best or "<module>"


def patch_of(path, old, new):
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{path}", f"b/{path}", n=3))


def run(path, old, new):
    files = parse_patch(patch_of(path, old, new))
    return sorted({q for _, q, *_ in attribute(files.get(path, []), old)}) if files else []


def mutations(path, src):
    L = src.splitlines(True)
    D = defs_with_parents(src)
    funcs = [(n, q) for n, q in D if not isinstance(n, ast.ClassDef) and len(n.body) >= 2]
    classes = [(n, q) for n, q in D if isinstance(n, ast.ClassDef)]
    strings = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.STRING:
                strings.update(range(tok.start[0], tok.end[0] + 1))
    except Exception:
        return
    if funcs:
        n, q = rng.choice(funcs)
        body = [ln for ln in range(n.body[0].lineno, n.end_lineno + 1) if L[ln - 1].strip() and ln not in strings
                and not L[ln - 1].lstrip().startswith(("@", "def ", "async def ", "class "))]
        if body:
            ln = rng.choice(body)
            exp = owner_of_line(src, ln)
            new = L[:]; new[ln - 1] = L[ln - 1].rstrip("\n") + "  # mutated\n"
            yield "replace_line", [exp], "".join(new)
            new = L[:ln - 1] + L[ln:]
            yield "delete_line", [exp], "".join(new)
            ind = " " * (len(L[ln - 1]) - len(L[ln - 1].lstrip()))
            new = L[:ln] + [ind + "_mutation_marker = 1\n"] + L[ln:]
            yield "insert_after_line", [owner_of_line(src, ln)], "".join(new)
        # append a statement at the end of the function body (body indentation)
        last = n.end_lineno
        bind = " " * n.body[-1].col_offset
        if last not in strings:
            new = L[:last] + [bind + "_mutation_tail = 1\n"] + L[last:]
            yield "append_body", [q], "".join(new)
        # decorator-only insertion before an existing def
        dec_start = min([n.lineno] + [d.lineno for d in n.decorator_list])
        new = L[:dec_start - 1] + [" " * n.col_offset + "@_mutation_decorator\n"] + L[dec_start - 1:]
        yield "add_decorator", [q], "".join(new)
    if classes:
        c, q = rng.choice(classes)
        mind = " " * (c.col_offset + 4)
        last = c.end_lineno
        for deco in (False, True):
            add = ([mind + "@property\n"] if deco else []) + [mind + "def _mutation_new(self):\n", mind + "    return 1\n"]
            # after the class's last line, then also after one trailing blank line if present
            new = L[:last] + ["\n"] + add + L[last:]
            yield f"new_method{'_decorated' if deco else ''}", [q], "".join(new)
            if last < len(L) and not L[last].strip():
                new = L[:last + 1] + add + ["\n"] + L[last + 1:]
                yield f"new_method_after_blank{'_decorated' if deco else ''}", [q], "".join(new)
    top = [n for n, q in D if "." not in q]
    if top:
        n = rng.choice(top)
        s = min([n.lineno] + [d.lineno for d in n.decorator_list])
        for deco in (False, True):
            add = (["@_mutation_decorator\n"] if deco else []) + ["def _mutation_top():\n", "    return 1\n", "\n", "\n"]
            new = L[:s - 1] + add + L[s - 1:]
            yield f"new_toplevel{'_decorated' if deco else ''}", ["<module>"], "".join(new)
    # blank-line-only insertion outside strings: expect no gold
    cand = [ln for ln in range(1, len(L)) if ln not in strings and (ln + 1) not in strings]
    if cand:
        ln = rng.choice(cand)
        yield "blank_insert", [], "".join(L[:ln] + ["\n"] + L[ln:])


rows = []
for t in TASKS:
    prefix = PKG[t["repo"]]
    with tarfile.open(SNAP / f"{t['instance_id']}.tgz") as tf:
        mem = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".py")]
        lib = [m for m in mem if m.name.lstrip("./").startswith(prefix) and "/tests/" not in m.name]
        for m in rng.sample(lib, min(2, len(lib))):
            path = m.name.lstrip("./")
            src = tf.extractfile(m).read().decode("utf-8", errors="replace")
            try:
                ast.parse(src)
            except SyntaxError:
                continue
            for kind, exp, new in mutations(path, src):
                try:
                    got = run(path, src, new)
                    err = ""
                except Exception as e:  # noqa: BLE001
                    got, err = None, f"{type(e).__name__}: {e}"[:200]
                rows.append(dict(instance_id=t["instance_id"], file=path, kind=kind, expected="|".join(exp),
                                 got="" if got is None else "|".join(got), ok=(got == sorted(exp)), error=err))

with open(OUT / "mutation_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
from collections import Counter
tot, ok = Counter(r["kind"] for r in rows), Counter(r["kind"] for r in rows if r["ok"])
for k in sorted(tot):
    print(f"{k:32s} {ok[k]:4d}/{tot[k]:4d}")
print("ALL", sum(ok.values()), "/", sum(tot.values()))
