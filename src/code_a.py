# ===== CELL: path =====
# Locate the inputs. On Kaggle: the competition files and the wheelhouse dataset. Locally: data/raw (read-only) plus cached
# kernel outputs under work/, because the snapshots are only on Kaggle.
import hashlib, json, os, re, shutil, subprocess, sys, tempfile, zipfile
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
ON_KAGGLE = KAGGLE_INPUT.exists()
if ON_KAGGLE:
    INPUT = sorted(KAGGLE_INPUT.glob("**/gemma-4-developer-agent/tasks.jsonl"))[0].parent
    WHEEL = sorted(KAGGLE_INPUT.glob("**/swegemma-0.2.11-py3-none-any.whl"))[0]
    PROJECT = None
else:
    PROJECT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "raw").is_dir())
    INPUT = PROJECT / "data/raw"
    WHEEL = PROJECT / "work/harness_0_2_11/wheels/swegemma-0.2.11-py3-none-any.whl"

# Intermediate files stay outside /kaggle/working; aggregate results remain in cell outputs.
W = str(Path(tempfile.gettempdir()) / "g4swe")
WORK = Path(W)
shutil.rmtree(WORK, ignore_errors=True)
for d in ("analysis", "kaggle/gold_ast", "kaggle/bm25", "kaggle/gold_mutation", "kaggle/skill_eval", "kaggle/unjudgeable", "results", "reports", "data", "work/c1", "work/c2"):
    (WORK / d).mkdir(parents=True)
(WORK / "data/raw").symlink_to(INPUT, target_is_directory=True)
FINAL = Path("/kaggle/working") if ON_KAGGLE else WORK / "final"
FINAL.mkdir(exist_ok=True)
print("environment:", "Kaggle" if ON_KAGGLE else "local dry run with cached kernel outputs")
print("graphs:", len(list(INPUT.glob("graphs/*.json"))), "embeddings:", len(list(INPUT.glob("embeddings/*.npz"))))


def run(cmd, tail=None, drop=None, ret=False):
    """Run a script from WORK; print its stdout (last `tail` lines), without lines matching `drop` (timings)."""
    p = subprocess.run([sys.executable, *cmd], cwd=WORK, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise RuntimeError(f"{cmd} exited with {p.returncode}")
    lines = [ln for ln in p.stdout.splitlines() if not (drop and re.search(drop, ln))]
    print("\n".join(lines if tail is None else lines[-tail:] if tail else []))
    return p.stdout if ret else None


def kernel(script, rel_dir, files, tail=1, drop=None):
    """Run a snapshot script on Kaggle and move its outputs out of /kaggle/working; locally copy the cached outputs."""
    (WORK / rel_dir).mkdir(parents=True, exist_ok=True)
    if ON_KAGGLE:
        run([f"kaggle/{script}"], tail=tail, drop=drop)
    for f in files:
        src = Path("/kaggle/working") / f if ON_KAGGLE else PROJECT / rel_dir / f
        (shutil.move if ON_KAGGLE else shutil.copy)(str(src), str(WORK / rel_dir / f))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


CHECKS = []


def check(name, got, want):
    """Stop the notebook if a key count differs from its expected value."""
    assert got == want, f"{name}: got {got!r}, expected {want!r}"
    CHECKS.append(name)
    print(f"ok  {name} = {got}")


# ===== CELL: verify_sources =====
# The scripts above were written byte for byte from the repository; these digests identify them.
for rel, want in SOURCES.items():
    got = sha(WORK / rel)
    assert got == want, rel
    print(got[:16], rel)
check("swegemma wheel sha256", sha(WHEEL), "9ad876581f3acde8ec3c369858f0834e2b9f41a3cffce3ca42f09cb090aea9d3")
zipfile.ZipFile(WHEEL).extractall(WORK / "wheel_0_2_11")
check("tasks.jsonl sha256", sha(INPUT / "tasks.jsonl"), "e4b3fd60f69dbc2b9213e54eeb9636db78aefe92c1d06269d73d9f5f8f3c8ad6")

# ===== CELL: graph_facts =====
import collections

import numpy as np
import pandas as pd

KIND = re.compile(r"^\s*(async\s+def|def|class)\b", re.M)
nodes = edges = isolated = 0
edge_types, kinds = collections.Counter(), collections.Counter()
for p in sorted(INPUT.glob("graphs/*.json")):
    g = json.load(open(p))
    ids = {n["id"] for n in g["nodes"]}
    touched = {e["source"] for e in g["edges"]} | {e["target"] for e in g["edges"]}
    nodes += len(ids)
    edges += len(g["edges"])
    isolated += len(ids - touched)
    edge_types.update(e["type"] for e in g["edges"])
    for n in g["nodes"]:
        m = KIND.search(n.get("text") or "")
        kinds[re.sub(r"\s+", " ", m.group(1)) if m else "other"] += 1
check("graph nodes", nodes, 367985)
check("graph edges", edges, 452588)
check("edge types", dict(edge_types), {"calls": 452588})
check("async function nodes", kinds["async def"], 0)
print(f"nodes without any edge: {isolated:,} = {isolated / nodes * 100:.1f}%")
print("node kinds:", dict(kinds))

# ===== CELL: gold_stage =====
kernel("gold_ast/gold_ast.py", "work/c1/gold_ast_v4", ["gold_ast.csv", "gold_files.csv"])
kernel("bm25/bm25.py", "work/c1/bm25_v3", ["bm25_ranks.csv"])
check("gold_ast.csv sha256", sha(WORK / "work/c1/gold_ast_v4/gold_ast.csv"), "75a03c85496548e86c8cb0acf08dcbb8b5e4b611cd6bbe84ec9e47a5a8993b23")
check("bm25_ranks.csv sha256", sha(WORK / "work/c1/bm25_v3/bm25_ranks.csv"), "00575f4d5f52fe9c04b4322a6f4c886f611d9e603b1797c5474ccdde4590cec2")
run(["analysis/gold_heuristic.py"], tail=0)
check("heuristic gold.csv sha256", sha(WORK / "work/c1/gold_coverage/gold.csv"), "2ed7d354465e5a1c4391007a42d13be3ed187fc9d058037e0b943f43d02743b6")

# ===== CELL: mutation_stage =====
kernel("gold_mutation/gold_mutation.py", "work/c1/gold_mutation", ["mutation_results.csv"], tail=1)
mut = pd.read_csv(WORK / "work/c1/gold_mutation/mutation_results.csv", keep_default_na=False)
check("mutation patches attributed correctly", (int((mut.ok.astype(str) == "True").sum()), len(mut)), (1821, 1821))

# ===== CELL: c1_report =====
run(["analysis/c1_report.py"])
manifest = json.load(open(WORK / "reports/c1_manifest.json"))
check("hash over all 127 graphs", manifest["graphs_sha256_of_sorted_hashes"], "06d0752a022bbddeb7f1500bbff3f51867d1f18a0fbf7c2d9861ee14533584a7")

# ===== CELL: embed_stage =====
run(["analysis/embed_geometry.py"], tail=0)
summary = (WORK / "work/c1/embed_geometry_v2/summary.md").read_text()
medians = {}
for label in ("effective rank", "PC1 variance share"):
    line = next(ln for ln in summary.splitlines() if ln.startswith(f"| {label}"))
    print(line)
    medians[label] = line.split("|")[2].strip()
check("median effective rank, PC1 share", (medians["effective rank"], medians["PC1 variance share"]), ("4.0559", "0.6308"))

# ===== CELL: reach_stage =====
run(["analysis/reach.py"], tail=0)
run(["analysis/reach_0211.py", "--wheel-src", "wheel_0_2_11", "--out", "work/c1/reach_0211"])

# ===== CELL: table1 =====
ids = pd.read_csv(WORK / "work/c1/reach_0211/seeds_per_task.csv").task.tolist()
old = pd.read_csv(WORK / "work/c1/reach_v3/per_task.csv")
old = old[(old.view == "e2e") & old.strategy.isin(["nb1_impl", "emb_impl"])]
new = pd.read_csv(WORK / "work/c1/reach_0211/per_task.csv")
hits = pd.read_csv(WORK / "work/c1/report/hits.csv").set_index("task").loc[ids]
KS1 = (10, 50, 100)
rows = {}
for label, strat in (("0.2.7 tools, issue identifiers as seeds (neighbours)", "nb1_impl"), ("0.2.7 tools, issue identifiers as seeds (similarity)", "emb_impl")):
    d = old[old.strategy == strat]
    rows[label] = [int(d[d.k == k].hit.sum()) for k in KS1]
for label, strat in (("0.2.11 resolver, 1-hop neighbours", "neighbors_e2e"), ("0.2.11 resolver, similarity search with lexical fallback", "similar_e2e"),
                     ("0.2.11 search_similar_code on the whole issue text (lexical fallback only)", "issue_text_query")):
    d = new[new.strategy == strat]
    rows[label] = [int(d[f"hit{k}"].sum()) for k in KS1]
rows["BM25 over AST function/class spans"] = [int(hits[f"bm25_snapshot@{k}"].sum()) for k in KS1]
table1 = pd.DataFrame(rows, index=[f"hit@{k}" for k in KS1]).T
display(table1)
check("tasks with an in-graph gold symbol", len(ids), 115)
check("0.2.7 neighbours", rows["0.2.7 tools, issue identifiers as seeds (neighbours)"], [2, 2, 2])
check("0.2.11 neighbours", rows["0.2.11 resolver, 1-hop neighbours"], [30, 35, 35])
check("0.2.11 similarity", rows["0.2.11 resolver, similarity search with lexical fallback"], [42, 55, 62])
check("0.2.11 whole issue", rows["0.2.11 search_similar_code on the whole issue text (lexical fallback only)"], [32, 64, 84])
check("BM25", rows["BM25 over AST function/class spans"], [51, 83, 98])

# ===== CELL: gate =====
gate = pd.read_csv(WORK / "work/c1/report/gate.csv")
g50 = gate[gate.k == 50]
B = np.random.default_rng(0).integers(0, len(ids), size=(10_000, len(ids)))
out = []
for (m, order), d in g50.groupby(["m", "order"]):
    d = d.set_index("task").loc[ids]
    x = d.hit.astype(int).to_numpy() - d.base.astype(int).to_numpy()
    bs = x[B].mean(axis=1) * 100
    out.append(dict(m=m, order=order, diff_pp=round(x.mean() * 100, 1), ci_low=round(np.percentile(bs, 2.5), 1), ci_high=round(np.percentile(bs, 97.5), 1), gained=int((x > 0).sum()), lost=int((x < 0).sum())))
gate_table = pd.DataFrame(out)
display(gate_table)
check("gate settings at k = 50 whose CI excludes zero", int((gate_table.ci_high < 0).sum()), 6)

# ===== CELL: c2_stage =====
kernel("skill_eval/skill_eval.py", "work/c2/skill_eval", ["skill_ranks.csv", "skill_timing.csv"], tail=0)
skill_out = run(["analysis/skill_eval.py"], drop=r"^runtime per task", ret=True)
for k, want in (("5", 46), ("10", 53), ("15", 59), ("50", 89), ("100", 100)):
    check(f"skill in-graph gold hit@{k}", int(re.search(rf"symbol hit@ *{k}: in-graph gold (\d+)/115", skill_out).group(1)), want)
check("skill file hit@10", int(re.search(r"file hit@10: (\d+)/115", skill_out).group(1)), 95)

# ===== CELL: c3_tables =====
run(["analysis/c3_report.py", "results/c3_run_rows.csv", "base", "out8k"])
c3 = pd.read_csv(WORK / "results/c3_run_rows.csv")
main = c3[c3.rep == 1]
wide = {a: main[main.arm == a].set_index("task") for a in ("base", "out8k")}
check("C3 rows in total, repetition 1", (len(c3), len(main)), (320, 240))
check("overflow runs base, out8k", (int(wide["base"].overflow.sum()), int(wide["out8k"].overflow.sum())), (26, 0))
check("resolved base, out8k", (int(wide["base"].resolved.sum()), int(wide["out8k"].resolved.sum())), (29, 34))
for col, want in (("empty_patch", (29, 2)), ("timeout", (1, 4)), ("tool_budget", (9, 12))):
    check(f"{col} base, out8k", (int(wide["base"][col].sum()), int(wide["out8k"][col].sum())), want)
check("gold file edited before overflow, base", int((wide["base"].lost_edit == True).sum()), 10)  # noqa: E712
check("runs with a first successful gold read, base, out8k", (int(wide["base"].first_open_step.notna().sum()), int(wide["out8k"].first_open_step.notna().sum())), (113, 111))
check("median first gold read, base, out8k", (float(wide["base"].first_open_step.median()), float(wide["out8k"].first_open_step.median())), (3.0, 3.0))
check("runs with gold lines displayed, base, out8k", (int(wide["base"].first_see_lines_step.notna().sum()), int(wide["out8k"].first_see_lines_step.notna().sum())), (109, 106))
over = wide["base"][wide["base"].overflow].index
o8 = wide["out8k"].loc[over]
print("out8k on the 26 tasks where base overflowed: resolved", int(o8.resolved.sum()), "| timeout", int(o8.timeout.sum()), "| tool budget", int(o8.tool_budget.sum()),
      "| failing patch", int((~o8.resolved & ~o8.timeout & ~o8.tool_budget & ~o8.empty_patch).sum()))
lost = wide["base"][wide["base"].lost_edit == True].index  # noqa: E712
print("base runs with a successful gold-file edit before overflowing:", len(lost), "| out8k resolved on those tasks:", int(wide["out8k"].loc[lost].resolved.sum()))
print("largest single-step output: base", int(wide["base"].max_step_output.max()), "| out8k", int(wide["out8k"].max_step_output.max()),
      "| next largest in out8k", int(sorted(list(wide["out8k"].max_step_output) + list(wide["out8k"].second_step_output))[-2]))
repo = main[main.arm == "base"].assign(repo=lambda d: d.task.str.split("_").str[0]).groupby("repo").overflow.agg(["sum", "count"])
print(repo.to_string())
rep2 = c3[(c3.rep == 2)].groupby("arm").overflow.sum().to_dict()
twice = set(c3[(c3.rep == 1) & (c3.arm == "base") & c3.overflow].task) & set(c3[(c3.rep == 2) & (c3.arm == "base") & c3.overflow].task)
print("repetition 2 overflows:", rep2, "| tasks overflowing in both base runs:", len(twice))

# ===== CELL: c3_judgeable =====
un = pd.read_csv(WORK / "results/c3_unjudgeable_tasks.csv")
check("unjudgeable tasks by cause", un.reason.value_counts().to_dict(), {"test_module": 31, "httpbin_fixture": 8, "test_module_masked": 1})
unj = c3[c3.task.isin(un.task)]
check("runs on unjudgeable tasks, resolved among them", (len(unj), int(unj.resolved.sum())), (112, 0))
kernel("unjudgeable/unjudgeable.py", "work/c3/unjudgeable", ["unjudgeable_static.csv"])
static = pd.read_csv(WORK / "work/c3/unjudgeable/unjudgeable_static.csv")
flagged = set(static.task) & set(c3.task)
check("static check on C3 tasks: flagged, missed from the list, flagged but not listed", (len(flagged), len(set(un.task) - flagged), len(flagged - set(un.task))), (40, 0, 0))
judgeable = wide["base"].index.difference(un.task)
check("judgeable tasks", len(judgeable), 80)
check("resolved on judgeable tasks, base, out8k", (int(wide["base"].loc[judgeable].resolved.sum()), int(wide["out8k"].loc[judgeable].resolved.sum())), (29, 34))
print(f"resolved rate on judgeable tasks: base {29 / 80:.1%}, out8k {34 / 80:.1%}")
over_j = over.intersection(judgeable)
o8j = wide["out8k"].loc[over_j]
check("base overflows on judgeable tasks", len(over_j), 15)
check("out8k on those: resolved, timeout, failing patch", (int(o8j.resolved.sum()), int(o8j.timeout.sum()), int((~o8j.resolved & ~o8j.timeout & ~o8j.tool_budget & ~o8j.empty_patch).sum())), (2, 2, 11))
rest = judgeable.difference(over)
check("judgeable tasks without a base overflow, out8k resolved", (len(rest), int(wide["out8k"].loc[rest].resolved.sum())), (65, 32))
check("gold edit before overflow: unjudgeable, judgeable, out8k resolved on judgeable", (len(lost.intersection(un.task)), len(lost.intersection(judgeable)), int(wide["out8k"].loc[lost.intersection(judgeable)].resolved.sum())), (8, 2, 0))
run(["analysis/c3_report.py", "results/c3_run_rows.csv", "base", "out8k", "--exclude", ",".join(un.task)], drop="^run dir")
run(["analysis/c3_report.py", "results/c3_run_rows.csv", "base", "out8k", "--exclude", ",".join([*un.task, "fastapi_14851"])], drop="^run dir|^repeatab|^  |^seconds|^tool_calls|^first_|^edit_arg|^$|^timeout|^tool_budget|^other_error|^empty_patch|^gold_edit|^overflow")

# ===== CELL: c3_figure =====
from IPython.display import Image

fig = FINAL / "fig1_max_prompt.png"
run(["analysis/c3_fig_prompt.py", "results/c3_run_rows.csv", str(fig)], tail=3, drop="^wrote")
display(Image(filename=str(fig), width=800))

# ===== CELL: c3_edits =====
run(["analysis/c3_overflow_edits.py", "results/c3_run_rows.csv"])

# ===== CELL: closing =====
# Keep intermediate files outside /kaggle/working: only the figure remains as an output file.
if ON_KAGGLE:
    for p in Path("/kaggle/working").iterdir():
        if p.name != fig.name and p.suffix != ".ipynb":
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    check("files left in /kaggle/working", sorted(p.name for p in Path("/kaggle/working").iterdir() if p.suffix != ".ipynb"), [fig.name])
print(len(CHECKS), "checks passed")
