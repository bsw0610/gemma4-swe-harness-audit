"""Rank the functions/classes of a Python repository by relevance to an issue text (BM25 over AST spans).

Pure standard library (Python >= 3.8). Read-only: never writes inside the repository.
Usage:
    python3 localize.py "issue text ..."                 # query from arguments
    python3 localize.py < issue.txt                      # query from stdin
    python3 localize.py --root /workspace --top 15 --files 5 --max-chars 4000 "..."
    run_skill_script(skill_name="localize-code", file_path="scripts/localize.py", args={"query": "issue text"})
Repository root: --root if it is a directory, else the first git checkout among /workspace (docker sandbox), $PWD and
$PYTHONPATH (subprocess sandbox: the wrapper runs from a temporary directory, and "/workspace" is rewritten only in
shell commands, not inside scripts), else the first of those that exists, else the current directory.
Output (plain text, < 4,000 characters): top files, then top symbols as `path:line  qualified.name`.
"""
import ast
import collections
import math
import os
import re
import sys

SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env", "build", "dist",
             ".tox", ".nox", ".mypy_cache", ".pytest_cache", "site-packages", "docs", "doc", "docs_src",
             "examples", "tests", "test", "testing", "benchmarks"}
TEST_FILE = re.compile(r"(^test_.*|.*_test)\.py$")
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CAMEL = re.compile(r"_|(?<=[a-z0-9])(?=[A-Z])")
STOP = set("""the a an and or of to in for is are be this that with on as it by not if from at we can should when use
using was were has have had but so do does did into than then there their them they you your our its also only
self none true false return def class import""".split())
MAX_FILE_BYTES = 1_000_000


def tokens(text):
    out = []
    for word in TOKEN.findall(text):
        parts = [p for p in CAMEL.split(word) if p]
        for t in [word] + (parts if len(parts) > 1 else []):
            t = t.lower()
            if len(t) > 1 and t not in STOP:
                out.append(t)
    return out


def python_files(root):
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith("."))
        for f in sorted(files):
            if f.endswith(".py") and not TEST_FILE.match(f):
                p = os.path.join(d, f)
                try:
                    if os.path.getsize(p) <= MAX_FILE_BYTES:
                        yield p
                except OSError:
                    pass


def units(root):
    """Yield (relative path, line, qualified name, text) for every def/class (decorators included) and each file."""
    for path in python_files(root):
        rel = os.path.relpath(path, root)
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                src = fh.read()
            tree = ast.parse(src)
        except (SyntaxError, ValueError, OSError):
            continue
        lines = src.splitlines()
        yield rel, 1, None, src
        stack = [(tree, "")]
        while stack:
            node, prefix = stack.pop()
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    q = f"{prefix}.{ch.name}" if prefix else ch.name
                    start = min([ch.lineno] + [d.lineno for d in ch.decorator_list])
                    end = getattr(ch, "end_lineno", None) or start
                    yield rel, ch.lineno, q, "\n".join(lines[start - 1:end])
                    stack.append((ch, q))
                else:
                    stack.append((ch, prefix))


def bm25(query_terms, docs, k1=1.2, b=0.75):
    tfs = [collections.Counter(tokens(d)) for d in docs]
    n = len(tfs) or 1
    avg = sum(sum(t.values()) for t in tfs) / n or 1.0
    df = collections.Counter(w for t in tfs for w in t)
    scores = []
    for t in tfs:
        dl = sum(t.values())
        s = 0.0
        for w in query_terms:
            if w in t:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * t[w] * (k1 + 1) / (t[w] + k1 * (1 - b + b * dl / avg))
        scores.append(s)
    return scores


def find_root(given):
    if given and os.path.isdir(given):
        return given
    cands = ["/workspace", os.environ.get("PWD", "")] + os.environ.get("PYTHONPATH", "").split(os.pathsep)
    for c in cands:
        if c and os.path.isdir(os.path.join(c, ".git")):
            return c
    return next((c for c in cands if c and os.path.isdir(c)), os.getcwd())


def main(argv):
    root, top, top_files, max_chars, args = None, 15, 5, 4000, []
    it = iter(argv)
    for a in it:
        if a == "--root":
            root = next(it)
        elif a == "--top":
            top = int(next(it))
        elif a == "--files":
            top_files = int(next(it))
        elif a == "--max-chars":
            max_chars = int(next(it))
        elif a == "--query":
            args.append(next(it))
        elif a == "--":  # run_skill_script puts '--' before positional_args
            continue
        else:
            args.append(a)
    query = " ".join(args).strip() or sys.stdin.read()
    q = set(tokens(query))
    if not q:
        print("localize: empty query")
        return 1
    root = find_root(root)
    files, syms = [], []
    for rel, line, qual, text in units(root):
        (files if qual is None else syms).append((rel, line, qual, text))
    if not syms:
        print(f"localize: no parsable Python files under {root}")
        return 1
    fs = bm25(q, [t for *_, t in files])
    ss = bm25(q, [t for *_, t in syms])
    file_rank = sorted(range(len(files)), key=lambda i: (-fs[i], files[i][0]))[:top_files]
    seen, sym_rank = set(), []
    for i in sorted(range(len(syms)), key=lambda i: (-ss[i], syms[i][0], syms[i][1])):
        key = (syms[i][0], syms[i][2])
        if key not in seen and ss[i] > 0:
            seen.add(key)
            sym_rank.append(i)
        if len(sym_rank) >= top:
            break
    out = [f"Likely files (BM25 over {len(files)} non-test files):"]
    out += [f"  {files[i][0]}" for i in file_rank if fs[i] > 0]
    out.append(f"Likely functions/classes (of {len(syms)}):")
    out += [f"  {syms[i][0]}:{syms[i][1]}  {syms[i][2]}" for i in sym_rank]
    print("\n".join(out)[:max_chars])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
