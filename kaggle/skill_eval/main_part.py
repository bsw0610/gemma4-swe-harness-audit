

# ===== Offline evaluation of the localize-code skill script (exact same code as skill/localize-code/scripts/localize.py) =====
import csv, glob, io, json, shutil, tarfile, time, contextlib
from pathlib import Path

ROOT_IN = Path(sorted(glob.glob("/kaggle/input/**/gemma-4-developer-agent/tasks.jsonl", recursive=True))[0]).parent
OUT = Path("/kaggle/working")
TASKS = [json.loads(l) for l in open(ROOT_IN / "tasks.jsonl")]
rows, timing = [], []
for t in TASKS:
    repo = Path("/tmp/repo")
    shutil.rmtree(repo, ignore_errors=True)
    repo.mkdir()
    with tarfile.open(ROOT_IN / "snapshots" / f"{t['instance_id']}.tgz") as tf:
        tf.extractall(repo, filter="data")
    heads = sorted(repo.rglob(".git/HEAD"), key=lambda p: len(p.parts))
    root = heads[0].parent.parent if heads else repo
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        main(["--root", str(root), "--top", "100", "--files", "20", "--max-chars", "1000000", t["problem_statement"]])
    dt = time.time() - t0
    text = buf.getvalue()
    timing.append(dict(instance_id=t["instance_id"], seconds=round(dt, 2), output_chars=len(text)))
    section, rank_f, rank_s = None, 0, 0
    for line in text.splitlines():
        if line.startswith("Likely files"):
            section = "file"; continue
        if line.startswith("Likely functions"):
            section = "symbol"; continue
        s = line.strip()
        if section == "file" and s:
            rank_f += 1
            rows.append(dict(instance_id=t["instance_id"], granularity="file", rank=rank_f, path=s, line="", qualname=""))
        elif section == "symbol" and s:
            rank_s += 1
            loc, qual = s.split("  ", 1)
            path, ln = loc.rsplit(":", 1)
            rows.append(dict(instance_id=t["instance_id"], granularity="symbol", rank=rank_s, path=path, line=ln, qualname=qual.strip()))
    print(t["instance_id"], round(dt, 2), flush=True)
for name, data in (("skill_ranks.csv", rows), ("skill_timing.csv", timing)):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
print("rows", len(rows), "max seconds", max(x["seconds"] for x in timing), "max chars", max(x["output_chars"] for x in timing))
