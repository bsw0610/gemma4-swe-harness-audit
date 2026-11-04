"""C3 Figure 1: maximum prompt tokens per run, by arm.

Per run, the maximum `metrics.prompt_tokens` over the agent steps of the ATIF trace. An overflowing base run never records the
rejected request, so its maximum is the largest prompt that succeeded. Prints the summary quoted in reports/c3_results.md and writes the PNG.

    uv run python analysis/c3_fig_prompt.py work/c3/main/g4swe_c3_main paper/fig1_max_prompt.png
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ARMS = ["base", "out8k"]
LINES = [(14_336, "compaction\nthreshold", "right"), (16_384, "base prompt\nceiling", "left"),
         (24_576, "out8k prompt\nceiling", "left")]
COLOR = {"base": "#2a78d6", "out8k": "#eb6834"}  # dataviz reference palette, slots 1-2 (light)
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load(run_dir):
    if run_dir.suffix == ".csv":  # results/c3_run_rows.csv, no traces needed
        return [dict(task=r["task"], arm=r["arm"], max_prompt=int(r["max_prompt"]), overflow=r["overflow"] == "True")
                for r in csv.DictReader(open(run_dir)) if r["rep"] == "1"]
    rows = []
    for line in open(run_dir / "c3_runs.jsonl"):
        r = json.loads(line)
        if r["rep"] != 1:  # the figure shows the main run only; repetition 2 is a calibration on 40 tasks
            continue
        trace = json.loads((run_dir / r["arm"] / f"rep{r['rep']}" / "traces" / f"trace_{r['task']}.json").read_text())
        prompts = [s["metrics"]["prompt_tokens"] for s in trace["steps"] if s.get("source") == "agent" and s.get("metrics")]
        rows.append(dict(task=r["task"], arm=r["arm"], max_prompt=max(prompts),
                         overflow="ContextWindowExceeded" in (r.get("error") or "")))
    return rows


def main(run_dir, out_png):
    rows = load(Path(run_dir))
    for arm in ARMS:
        v = np.array([r["max_prompt"] for r in rows if r["arm"] == arm])
        print(f"{arm}: n {len(v)}  median {np.median(v):,.0f}  p95 {np.percentile(v, 95):,.0f}  max {v.max():,}  "
              f"> 16,384: {(v > 16_384).sum()}  > 24,576: {(v > 24_576).sum()}")

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 3.3), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for x, label, side in LINES:
        ax.axvline(x, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(x + (150 if side == "left" else -150), 1.62, label, color=MUTED, fontsize=8, va="top",
                ha="left" if side == "left" else "right")
    for i, arm in enumerate(ARMS):
        sub = [r for r in rows if r["arm"] == arm]
        x = np.array([r["max_prompt"] for r in sub])
        y = i + rng.uniform(-0.22, 0.22, len(sub))
        ovf = np.array([r["overflow"] for r in sub])
        ax.scatter(x[~ovf], y[~ovf], s=22, color=COLOR[arm], edgecolor=SURFACE, linewidth=0.6, zorder=3,
                   label="base, run finished" if arm == "base" else "out8k (no run overflowed)")
        if ovf.any():
            ax.scatter(x[ovf], y[ovf], s=22, facecolor=SURFACE, edgecolor=COLOR[arm], linewidth=1.2, zorder=3,
                       label=f"base, overflow: next request rejected (n = {ovf.sum()})")
    ax.set_yticks(range(len(ARMS)), [f"{a}\n(max_output {m})" for a, m in zip(ARMS, ["16,384", "8,192"])], color=INK, fontsize=9)
    ax.set_ylim(-0.5, 1.7)
    ax.set_xlim(4_000, 26_000)
    ax.set_xticks(range(4_000, 26_001, 4_000), [f"{v // 1000}k" for v in range(4_000, 26_001, 4_000)])
    ax.set_xlabel("Maximum prompt tokens in a run (one dot per run, n = 120 per arm)", color=INK, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(loc="lower left", fontsize=7.5, frameon=False, labelcolor=INK, ncol=3, bbox_to_anchor=(0.0, 1.0),
              handletextpad=0.3, columnspacing=1.5)
    fig.suptitle("Halving max_output_tokens lets prompts grow past the old ceiling, never near the new one",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_png, facecolor=SURFACE)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
