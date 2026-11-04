"""Gold-location core shared by the Kaggle kernel (kaggle/gold_ast) and local regression tests.

attribute(patch_hunks, src) maps every changed old-file line of a code file to the innermost
enclosing def/class of the base snapshot (ast).

Rules (v2, after Codex audit A01/A11):
- A hunk is split into change blocks: maximal runs of '-'/'+' lines between context lines.
- Replacement block (has '-' lines): attributed to the scopes of its removed old lines only.
  Its '+' lines are NOT treated as insertions (A01 fix).
- Pure insertion block (only '+'), inserted between old lines a and a+1 (v3, audit 2 N01/N02):
  * decorator-only block (only '@' lines and continuation, no def/class) placed right before an existing scope T
    whose header column equals the decorator indent -> owner = T (the existing definition gains a decorator);
  * otherwise owner = innermost scope S that lexically encloses the gap and whose body column the new code sits in:
    S.start <= a' <= S.end, where a' = last non-blank old line <= a (trailing blank lines after a scope still belong to it),
    and indent(first non-blank '+' line) > S.col. A new method -> its class; a new top-level def/class -> '<module>'.
- '\\ No newline at end of file' lines are ignored; '+++ /dev/null' (deleted files) are skipped.
- Whitespace-only blocks (every removed old line and every added line is blank) are formatting, not gold: skipped,
  unless they lie inside a multi-line string literal (tokenize), where blank lines are part of the value (N06).
"""
import ast
import io
import re
import tokenize

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
TEST_RE = re.compile(r"(^|/)tests?/|(^|/)test_[^/]*$|_test\.py$")


def code_file(path):
    return path.endswith(".py") and not TEST_RE.search(path) and not path.startswith(("docs/", "docs_src/"))


def parse_patch(patch):
    """{path: [hunk]}; hunk = {'old_start', 'header', 'blocks': [{'removed': [old line nos], 'added': [text],
    'after': old line no. preceding the block}]}"""
    files, cur, hunk, old, block = {}, None, None, 0, None
    for line in patch.splitlines():
        if line.startswith("--- "):
            continue
        if line.startswith("+++ "):
            p = line[4:].strip()
            cur = None if p == "/dev/null" else (p[2:] if p.startswith("b/") else p)
            if cur is not None:
                files.setdefault(cur, [])
            hunk = None
            continue
        if cur is None:
            continue
        m = HUNK_RE.match(line)
        if m:
            old = int(m.group(1)) if (m.group(2) is None or m.group(2) != "0") else int(m.group(1)) + 1
            hunk = {"old_start": int(m.group(1)), "header": line, "blocks": []}
            files[cur].append(hunk)
            block = None
            continue
        if hunk is None or line.startswith("\\"):
            continue
        if line.startswith("-") or line.startswith("+"):
            if block is None:
                block = {"removed": [], "added": [], "after": old - 1}
                hunk["blocks"].append(block)
            if line.startswith("-"):
                block["removed"].append(old)
                old += 1
            else:
                block["added"].append(line[1:])
        else:
            block = None
            old += 1
    return files


def scopes(src):
    """[(start, end, qualname, kind, col)] for every def/class; start includes decorators."""
    tree = ast.parse(src)
    out = []

    def walk(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = f"{prefix}.{ch.name}" if prefix else ch.name
                start = min([ch.lineno] + [d.lineno for d in ch.decorator_list])
                kind = {ast.FunctionDef: "def", ast.AsyncFunctionDef: "async_def", ast.ClassDef: "class"}[type(ch)]
                out.append((start, ch.end_lineno, q, kind, ch.col_offset))
                walk(ch, q)
            else:
                walk(ch, prefix)
    walk(tree, "")
    return out


def innermost(sc, line):
    best = None
    for s in sc:
        if s[0] <= line <= s[1] and (best is None or (s[1] - s[0]) < (best[1] - best[0])):
            best = s
    return best


def _indent(text):
    return len(text) - len(text.lstrip(" \t"))


DEF_RE = re.compile(r"\s*(async\s+def|def|class)\b")


def insertion_owner(sc, a, added, lines):
    nonblank = [t for t in added if t.strip()]
    if not nonblank:
        return None
    ind = _indent(nonblank[0])
    top = [t for t in nonblank if _indent(t) == ind]
    decorator_only = top[0].lstrip().startswith("@") and not any(DEF_RE.match(t) for t in top)
    if decorator_only:
        nxt = [x for x in sc if x[0] == a + 1 and x[4] == ind]
        if nxt:
            return min(nxt, key=lambda x: x[1] - x[0])
    a2 = a
    while a2 > 0 and a2 <= len(lines) and not lines[a2 - 1].strip():
        a2 -= 1
    cands = [x for x in sc if x[0] <= a2 <= x[1] and ind > x[4]]
    return min(cands, key=lambda x: x[1] - x[0]) if cands else None


def string_continuation_lines(src):
    """Old line numbers that lie inside a multi-line string literal (after its first line)."""
    out = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
                out.update(range(tok.start[0] + 1, tok.end[0] + 1))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return out


def attribute(hunks, src):
    """Yield (hunk_index, qualname | '<module>', kind, line, block_type)."""
    sc = scopes(src)
    lines = src.splitlines()
    in_string = string_continuation_lines(src)
    for hi, h in enumerate(hunks):
        seen = set()
        for b in h["blocks"]:
            removed_text = [lines[ln - 1] if 0 < ln <= len(lines) else "x" for ln in b["removed"]]
            if all(not t.strip() for t in removed_text + b["added"]):
                inside = any(ln in in_string for ln in b["removed"]) or (not b["removed"] and (b["after"] + 1) in in_string)
                if not inside:
                    continue
                s_ = innermost(sc, b["removed"][0] if b["removed"] else b["after"] + 1)
                key = s_[2] if s_ else "<module>"
                if key not in seen:
                    seen.add(key)
                    yield hi, key, (s_[3] if s_ else "module"), (b["removed"] or [b["after"]])[0], "string_ws"
                continue
            if b["removed"]:
                for ln in b["removed"]:
                    s = innermost(sc, ln)
                    key = s[2] if s else "<module>"
                    if key not in seen:
                        seen.add(key)
                        yield hi, key, (s[3] if s else "module"), ln, "replace"
            else:
                s = insertion_owner(sc, b["after"], b["added"], lines)
                key = s[2] if s else "<module>"
                if key not in seen:
                    seen.add(key)
                    yield hi, key, (s[3] if s else "module"), b["after"], "insert"
