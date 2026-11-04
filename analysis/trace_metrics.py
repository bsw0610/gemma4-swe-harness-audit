"""Intermediate outcomes of one agent run, measured from the harness trace (C3).

Input: results/traces/trace_<id>.json written by swegemma (adk_eval_core SessionTrace.save, ATIF v1.7) and the task's
gold `patch` from tasks.jsonl. Standard library only; gold is recomputed from the patch with gold_core.

Definitions (step = 1-based ordinal of the tool call among all tool calls in the trace, in emission order):
- gold files: code files (gold_core.code_file) changed by the gold patch that exist in the base snapshot
  (a file whose only hunk is "@@ -0,0" is new and cannot be opened; it is excluded).
- gold lines: per gold file, the base-snapshot lines the patch removes, plus both neighbours of every pure insertion.
- mention: a gold file path appears in a tool response (grep/find output, graph tools, skill output, ...).
- open: a successful read_file on a gold file that returned content, or a successful read-like run_command (READ_CMD) whose
  command contains a gold path and that printed something. Shell output is not parsed further, so this measures access, not understanding.
- see lines: a read_file response on a gold file whose displayed range (start_line .. start_line + shown lines - 1)
  contains a gold line. Ranges refer to the file as it was read; reads after the agent edited that file are skipped.
- edit: a successful edit_file / write_file on a gold file. It may be undone later, so it does not imply a non-empty final patch.
- skill: run_skill_script with skill_name == "localize-code"; skill_hit = its first response mentions a gold file.
- skill_any: any skill tool (run_skill_script, load_skill, list_skills, load_skill_resource); ADK's own skill instruction
  tells the model to load_skill first (google/adk/tools/skill_toolset.py:61-88), so this counts engagement without a run.
- *_work_step: the same event counted in tool calls other than the skill tools (run_skill_script, load_skill, list_skills,
  load_skill_resource). The skill arm is told to call the skill first, so its raw step is >= 2 by construction; compare
  arms on *_work_step or on seconds (*_s), and use the raw step only within an arm.
Path match: tool paths are normalised by stripping "/workspace/" and "./"; run_command uses substring match
(so "cd rich && cat console.py" is not counted as an open; this undercounts opens via run_command only).
Run: uv run python analysis/trace_metrics.py <trace.json> <instance_id> [tasks.jsonl]
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gold_core import HUNK_RE, code_file, parse_patch  # noqa: E402

SKILL = "localize-code"
EDIT_TOOLS = ("edit_file", "write_file")
READ_CMD = re.compile(r"\b(cat|sed|head|tail|nl|awk|less|more|grep|rg)\b")
SKILL_TOOLS = ("run_skill_script", "load_skill", "list_skills", "load_skill_resource")


def flatten(atif):
    """ATIF v1.7 dict -> [{'step', 'tool', 'args', 'elapsed_s', 'result'}] in call order.

    The converter puts consecutive calls in one step's tool_calls[] but attaches only the first response to that step;
    later responses become separate system steps. Responses are therefore paired with calls FIFO per tool name.
    """
    calls, pending = [], {}
    for st in atif.get("steps", []):
        for tc in st.get("tool_calls") or []:
            ex = tc.get("extra") or {}
            c = {"step": len(calls) + 1, "tool": tc.get("function_name", ""), "args": tc.get("arguments") or {},
                 "elapsed_s": float(ex.get("elapsed_s", (st.get("extra") or {}).get("elapsed_s", 0.0)) or 0.0),
                 "result": None}
            calls.append(c)
            pending.setdefault(c["tool"], []).append(c)
        obs = st.get("observation")
        if obs:
            name = (obs.get("extra") or {}).get("tool_name", "")
            q = pending.get(name)
            if q:
                q.pop(0)["result"] = str(obs.get("content", ""))
    return calls


def norm(path):
    p = str(path or "").strip()
    for pre in ("/workspace/", "./"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p


def gold_of(patch):
    """{gold file: set of base-snapshot gold lines} for existing code files changed by the patch."""
    files = parse_patch(patch)
    out = {}
    for path, hunks in files.items():
        if not code_file(path) or not hunks:
            continue
        if all(HUNK_RE.match(h["header"]).group(1) == "0" for h in hunks):
            continue  # new file
        lines = set()
        for h in hunks:
            for b in h["blocks"]:
                lines.update(b["removed"])
                if not b["removed"]:
                    lines.update((b["after"], b["after"] + 1))
        out[path] = lines
    return out


def shown_range(result):
    try:
        r = json.loads(result)
    except (TypeError, ValueError):
        return None
    if not isinstance(r, dict) or r.get("status") != "ok" or "start_line" not in r:
        return None
    n = len(str(r.get("content", "")).splitlines())
    return (int(r["start_line"]), int(r["start_line"]) + n - 1) if n else None


def succeeded(result):
    """True when a tool response is JSON with status ok; failed and malformed calls are not counted as events."""
    try:
        d = json.loads(result or "")
    except ValueError:
        return False
    return isinstance(d, dict) and d.get("status") == "ok"


def has_output(result, key):
    """True when a successful tool response has non-blank text under `key` (content of read_file, stdout of run_command)."""
    try:
        return bool(str(json.loads(result or "").get(key) or "").strip())
    except (ValueError, AttributeError):
        return False


def metrics(atif, patch):
    gold = gold_of(patch)
    calls = flatten(atif)
    first = {k: None for k in ("mention", "open", "see_lines", "edit", "skill", "skill_any")}
    edited, by_tool, skill_hit, n_skill = set(), {}, None, 0

    def mark(key, c):
        if first[key] is None:
            first[key] = (c["step"], c["step"] - n_skill, round(c["elapsed_s"], 1))

    for c in calls:
        t, a, res = c["tool"], c["args"], c["result"] or ""
        n_skill += t in SKILL_TOOLS
        by_tool[t] = by_tool.get(t, 0) + 1
        path = norm(a.get("filepath"))
        ok = succeeded(res)
        if gold and any(g in res for g in gold):
            mark("mention", c)
        if t == "read_file" and path in gold and ok and has_output(res, "content"):
            mark("open", c)
            rng = shown_range(res)
            if rng and path not in edited and any(rng[0] <= ln <= rng[1] for ln in gold[path]):
                mark("see_lines", c)
        if t == "run_command" and ok and has_output(res, "stdout") and READ_CMD.search(str(a.get("command", ""))) and any(g in str(a.get("command", "")) for g in gold):
            mark("open", c)
        if t in EDIT_TOOLS and ok:
            if path in gold:
                mark("edit", c)
            edited.add(path)
        if t in SKILL_TOOLS:
            mark("skill_any", c)
        if t == "run_skill_script" and a.get("skill_name") == SKILL:
            if first["skill"] is None:
                skill_hit = bool(gold) and any(g in res for g in gold)
            mark("skill", c)
    row = {"n_gold_files": len(gold), "n_tool_calls": len(calls), "tool_counts": json.dumps(by_tool, sort_keys=True),
           "skill_hit": skill_hit, "edited_files": json.dumps(sorted(edited)),
           "total_tokens": (atif.get("final_metrics") or {}).get("total_tokens")}
    for k, v in first.items():
        row[f"first_{k}_step"], row[f"first_{k}_work_step"], row[f"first_{k}_s"] = v if v else (None, None, None)
    return row


if __name__ == "__main__":
    trace, iid = Path(sys.argv[1]), sys.argv[2]
    tasks = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).resolve().parents[1] / "data/raw/tasks.jsonl"
    patch = next(json.loads(l)["patch"] for l in open(tasks) if json.loads(l)["instance_id"] == iid)
    print(json.dumps(metrics(json.loads(trace.read_text()), patch), indent=2))
