"""Targeted static check: which tasks have tests blocked by missing test dependencies in the subprocess sandbox, read from the snapshots and test patches.

swegemma's subprocess sandbox skips install_test_dependencies, so test-only packages are absent. Verification runs the test
files named in test_patch and needs exit code 0. A task is flagged when a test file after the patch imports inline_snapshot or
dirty_equals at top level, directly or through a local test module such as tests/utils.py, or has a test or fixture that takes
the httpbin fixture (a rule fitted to this data; it does not check that the fixture is collected). Output: /kaggle/working/unjudgeable_static.csv (task, reason, file, found_in).
"""
import ast, csv, glob, json, re, tarfile
from pathlib import Path

ROOT = Path(sorted(glob.glob("/kaggle/input/**/gemma-4-developer-agent/tasks.jsonl", recursive=True))[0]).parent
SNAP, OUT = ROOT / "snapshots", Path("/kaggle/working")
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
MISSING = {"inline_snapshot", "dirty_equals"}
HTTPBIN = re.compile(r"^\s*(?:async\s+)?def \w+\([^)]*\bhttpbin\b", re.M)


def split_patch(patch):
    """{path after the patch: [(old start, [diff lines])]}; deleted files are left out. Hunk lengths come from the headers."""
    files, cur, left_old, left_new = {}, None, 0, 0
    for line in patch.splitlines():
        if left_old or left_new:
            kind = line[:1]
            left_old -= kind in (" ", "-", "")
            left_new -= kind in (" ", "+", "")
            if cur:
                files[cur][-1][1].append(line)
        elif line.startswith("+++ "):
            p = line[4:].strip()
            cur = None if p == "/dev/null" else p[2:]
            if cur:
                files.setdefault(cur, [])
        elif (m := HUNK.match(line)):
            left_old, left_new = int(m.group(2) or 1), int(m.group(4) or 1)
            if cur:
                files[cur].append((int(m.group(1)), []))
    return files


def apply_hunks(path, text, hunks):
    """Apply hunks to the base text; stop if a context or removed line differs from the base file."""
    old, out, pos = text.splitlines(), [], 0
    for start, lines in hunks:
        # A hunk with no old lines ("@@ -n,0 ...") inserts after line n, others start at line n.
        first = start if not any(ln[:1] in (" ", "-", "") for ln in lines) else start - 1
        out += old[pos:first]
        pos = first
        for ln in lines:
            if ln.startswith("\\"):
                continue
            if ln[:1] == "+":
                out.append(ln[1:])
                continue
            assert old[pos] == ln[1:], (path, pos, ln)
            if ln[:1] != "-":
                out.append(ln[1:])
            pos += 1
    return "\n".join(out + old[pos:])


def top_level_imports(path, text):
    """(third-party roots imported at top level, local test modules imported)."""
    roots, local = set(), []
    pkg = path.split("/")[:-1]
    for node in ast.parse(text).body:
        if isinstance(node, ast.Import):
            for a in node.names:
                roots.add(a.name.split(".")[0])
                if a.name.startswith("tests."):
                    local.append(a.name.replace(".", "/"))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                roots.add((node.module or "").split(".")[0])
            if node.level:
                base = pkg[:len(pkg) - node.level + 1] + (node.module.split(".") if node.module else [])
            elif node.module == "tests" or (node.module or "").startswith("tests."):
                base = node.module.split(".")
            else:
                continue
            local.append("/".join(base))
            local += ["/".join(base + [a.name]) for a in node.names]
    return roots, local


tasks = [json.loads(l) for l in open(ROOT / "tasks.jsonl")]
rows = []
for t in tasks:
    tid = t["instance_id"]
    files = split_patch(t["test_patch"])
    with tarfile.open(SNAP / f"{tid}.tgz") as tf:
        base = {}
        for m in tf:
            if m.isfile() and m.name.endswith(".py") and "/tests/" in "/" + m.name:
                rel = m.name[m.name.index("tests/") if m.name.startswith("tests/") else m.name.index("/tests/") + 1:]
                base[rel] = tf.extractfile(m).read().decode("utf-8", "replace")
    post = {path: apply_hunks(path, base.get(path, ""), hunks) for path, hunks in files.items()}
    for path in sorted(files):
        if not path.startswith("tests/"):
            continue
        seen, todo = set(), [path]
        while todo:
            cur = todo.pop()
            if cur in seen or (cur not in post and cur not in base):
                continue
            seen.add(cur)
            text = post.get(cur, base.get(cur, ""))
            roots, local = top_level_imports(cur, text)
            for name in sorted(roots & MISSING):
                rows.append((tid, name, path, cur))
            if cur == path and HTTPBIN.search(text):
                rows.append((tid, "httpbin", path, cur))
            todo += [m + ".py" for m in local] + [m + "/__init__.py" for m in local]
with open(OUT / "unjudgeable_static.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "reason", "file", "found_in"])
    w.writerows(rows)
print("tasks", len(tasks), "flagged", len({r[0] for r in rows}), "rows", len(rows))
