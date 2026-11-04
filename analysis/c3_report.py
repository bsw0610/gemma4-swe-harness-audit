"""C3 report: paired comparison of two agent arms from colab/c3_run.py output.

Input: a run directory laid out like the Colab OUT folder: c3_runs.jsonl and <arm>/rep<k>/traces/trace_<id>.json
(a flat folder of trace_<id>_<arm>.json files, as in work/c3/pilot/, is also accepted). Gold comes from data/raw/tasks.jsonl.
Outcome definitions (per run):
- overflow: error contains "ContextWindowExceeded" (vLLM rejected prompt + max_output_tokens > max_model_len);
- timeout: error contains "session timeout"; tool_budget: tool_calls >= 100 (official budget); other_error: any other error;
- resolved: official verification (Colab environment); empty_patch: patch_chars == 0;
- lost_edit (printed as gold_edit_then_overflow): overflow and the trace shows an edit of a gold file before the failure.
  This is "work in progress was discarded", not "a correct fix was discarded";
- edit_arg_errors: tool responses with "mandatory input parameters are not present" (harness tool-call parsing failures).
Paired analysis over tasks with both arms present: difference in proportions (B - A) with a task bootstrap 95% CI
(10,000 draws, seed 0, as in C1/C2) and an exact McNemar test on discordant pairs; for seconds and tool calls, the median
of per-task differences (B - A) with a task bootstrap 95% CI.
Run: uv run python analysis/c3_report.py <run_dir> [armA armB] [--exclude id,id] [--rep 1]
With a second repetition present, also prints within-arm rep1-vs-rep2 discordance (run-to-run variation).
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_metrics import flatten, metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BOOT, SEED = 10_000, 0


def load_csv(path):
    """results/c3_run_rows.csv (written by c3_export_rows.py): same rows as load(), no traces needed."""
    def val(k, v):
        if v == "":
            return None
        if v in ("True", "False"):
            return v == "True"
        return v if k in ("task", "arm") else (float(v) if k == "seconds" else int(v))
    rows = [{k: val(k, v) for k, v in r.items()} for r in csv.DictReader(open(path))]
    return [{k: v for k, v in r.items() if v is not None or k in ("seconds", "tool_calls", "first_open_step", "first_see_lines_step")} for r in rows]


def load(run_dir):
    if run_dir.suffix == ".csv":
        return load_csv(run_dir)
    runs = [json.loads(l) for l in open(run_dir / "c3_runs.jsonl") if l.strip()]
    patches = {json.loads(l)["instance_id"]: json.loads(l)["patch"] for l in open(ROOT / "data/raw/tasks.jsonl")}
    out = []
    for r in runs:
        cands = [run_dir / r["arm"] / f"rep{r['rep']}" / "traces" / f"trace_{r['task']}.json",
                 run_dir / f"trace_{r['task']}_{r['arm']}.json"]
        tp = next((p for p in cands if p.exists()), None)
        err = r.get("error") or r.get("harness_error") or ""
        row = dict(task=r["task"], arm=r["arm"], rep=r["rep"], resolved=bool(r.get("resolved")),
                   overflow="ContextWindowExceeded" in err, timeout="session timeout" in err,
                   tool_budget=(r.get("tool_calls") or 0) >= 100, empty_patch=(r.get("patch_chars") or 0) == 0,
                   seconds=r.get("agent_seconds"), tool_calls=r.get("tool_calls"), has_trace=tp is not None)
        row["other_error"] = bool(err) and not (row["overflow"] or row["timeout"])
        if tp:
            atif = json.loads(tp.read_text())
            m = metrics(atif, patches[r["task"]])
            row["lost_edit"] = row["overflow"] and m["first_edit_step"] is not None
            row["first_open_step"] = m["first_open_step"]
            row["first_see_lines_step"] = m["first_see_lines_step"]
            row["edit_arg_errors"] = sum("mandatory input parameters are not present" in (c["result"] or "")
                                         for c in flatten(atif))
        out.append(row)
    return out


def boot_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = np.random.default_rng(SEED).integers(0, len(a), size=(BOOT, len(a)))
    d = b[idx].mean(1) - a[idx].mean(1)
    return b.mean() - a.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def mcnemar_exact(a, b):
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n01, n10, min(1.0, 2 * p)


def discord(x, y):
    """(n, number of discordant pairs, rate, task-bootstrap 95% CI of the rate)."""
    d = np.array([bool(a) != bool(b) for a, b in zip(x, y)], float)
    idx = np.random.default_rng(SEED).integers(0, len(d), size=(BOOT, len(d)))
    bm = d[idx].mean(1)
    return len(d), int(d.sum()), d.mean(), np.percentile(bm, 2.5), np.percentile(bm, 97.5)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("arms", nargs="*")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--rep", type=int, default=1, help="repetition used for the between-arm comparison")
    a_ = ap.parse_args(argv)
    exclude = set(filter(None, a_.exclude.split(",")))
    all_rows = [r for r in load(a_.run_dir) if r["task"] not in exclude]
    A, B = a_.arms[:2] if len(a_.arms) >= 2 else sorted({r["arm"] for r in all_rows})
    by = {(r["task"], r["rep"], r["arm"]): r for r in all_rows}
    keys = sorted({(t, k) for t, k, _ in by if k == a_.rep and (t, k, A) in by and (t, k, B) in by})
    print(f"run dir {a_.run_dir}; arms A={A} B={B}; rep {a_.rep}; paired tasks = {len(keys)}; "
          f"traces present {sum(r['has_trace'] for r in all_rows)}/{len(all_rows)}; excluded {sorted(exclude)}")
    if not keys:
        return 1
    print(f"\n{'outcome':16} {A:>12} {B:>12}   diff B-A [95% CI]        discordant(A0B1,A1B0)  McNemar p")
    for o in ("resolved", "overflow", "timeout", "tool_budget", "other_error", "empty_patch", "lost_edit"):
        if any(o not in by[(t, k, arm)] for t, k in keys for arm in (A, B)):
            continue
        a = [bool(by[(t, k, A)][o]) for t, k in keys]
        b = [bool(by[(t, k, B)][o]) for t, k in keys]
        d, lo, hi = boot_diff(a, b)
        n01, n10, p = mcnemar_exact(a, b)
        label = "gold_edit_then_overflow" if o == "lost_edit" else o
        print(f"{label:16} {sum(a):>5}/{len(a):<6} {sum(b):>5}/{len(b):<6}   {d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
              f"      ({n01}, {n10})              {p:.3f}")
    for o in ("seconds", "tool_calls", "first_open_step", "first_see_lines_step", "edit_arg_errors"):
        va = [by[(t, k, A)].get(o) for t, k in keys]
        vb = [by[(t, k, B)].get(o) for t, k in keys]
        pa = [x for x in va if x is not None]
        pb = [x for x in vb if x is not None]
        if pa and pb:
            line = f"{o:16} median {np.median(pa):>8.1f} {np.median(pb):>8.1f}   (n {len(pa)}, {len(pb)})"
            d = np.array([y - x for x, y in zip(va, vb) if x is not None and y is not None], float)
            if o in ("seconds", "tool_calls") and len(d):
                idx = np.random.default_rng(SEED).integers(0, len(d), size=(BOOT, len(d)))
                bm = np.median(d[idx], axis=1)
                line += (f"   paired median diff B-A {np.median(d):+.1f} "
                         f"[{np.percentile(bm, 2.5):+.1f}, {np.percentile(bm, 97.5):+.1f}]")
            print(line)

    # Repeatability: same arm, rep 1 vs rep 2, compared with between-arm discordance (rep 1) on the same tasks.
    rep_tasks = sorted({t for t, k, _ in by if k == 2 and all((t, r, arm) in by for r in (1, 2) for arm in (A, B))})
    if rep_tasks:
        print(f"\nrepeatability on {len(rep_tasks)} tasks with both reps of both arms "
              f"(discordant runs / tasks, rate [95% CI])")
        for o in ("resolved", "overflow"):
            cells = [(f"{A} rep1 vs rep2", [by[(t, 1, A)][o] for t in rep_tasks], [by[(t, 2, A)][o] for t in rep_tasks]),
                     (f"{B} rep1 vs rep2", [by[(t, 1, B)][o] for t in rep_tasks], [by[(t, 2, B)][o] for t in rep_tasks]),
                     (f"{A} vs {B} (rep1)", [by[(t, 1, A)][o] for t in rep_tasks], [by[(t, 1, B)][o] for t in rep_tasks]),
                     (f"{A} vs {B} (rep2)", [by[(t, 2, A)][o] for t in rep_tasks], [by[(t, 2, B)][o] for t in rep_tasks])]
            for name, x, y in cells:
                n, k, rate, lo, hi = discord(x, y)
                print(f"  {o:9} {name:24} {k:>3}/{n:<4} {rate:.3f} [{lo:.3f}, {hi:.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
