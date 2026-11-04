

# ===== Kaggle main (gold_ast, core v3 rules) =====
import csv, glob, json, tarfile
from pathlib import Path

ROOT = Path(sorted(glob.glob("/kaggle/input/**/gemma-4-developer-agent/tasks.jsonl", recursive=True))[0]).parent
SNAP, OUT = ROOT / "snapshots", Path("/kaggle/working")
TASKS = [json.loads(l) for l in open(ROOT / "tasks.jsonl")]
print("root", ROOT, "tasks", len(TASKS), "snapshots", len(list(SNAP.glob("*.tgz"))))


def read_member(names, tf, path):
    cands = sorted([n for n in names if n == path or n.endswith("/" + path)], key=len)
    return tf.extractfile(cands[0]).read().decode("utf-8", errors="replace") if cands else None


rows, filerows = [], []
for t in TASKS:
    tid = t["instance_id"]
    files = parse_patch(t["patch"])
    with tarfile.open(SNAP / f"{tid}.tgz") as tf:
        names = tf.getnames()
        for path, hunks in files.items():
            if not code_file(path):
                continue
            src = read_member(names, tf, path)
            if src is None:
                filerows.append(dict(instance_id=tid, file=path, exists=False, parses=None, n_async=None, n_def=None))
                continue
            try:
                sc = scopes(src)
                ok = True
            except SyntaxError:
                sc, ok = [], False
            filerows.append(dict(instance_id=tid, file=path, exists=True, parses=ok,
                                 n_async=sum(s[3] == "async_def" for s in sc), n_def=len(sc)))
            if not ok:
                continue
            for hi, q, kind, line, btype in attribute(hunks, src):
                rows.append(dict(instance_id=tid, repo=t["repo"], file=path, hunk_index=hi, qualname=q,
                                 kind=kind, line=line, block_type=btype))

for name, data in [("gold_ast.csv", rows), ("gold_files.csv", filerows)]:
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)
print("gold rows", len(rows), "files", len(filerows),
      "missing files", sum(not r["exists"] for r in filerows),
      "parse failures", sum(r["parses"] is False for r in filerows))
