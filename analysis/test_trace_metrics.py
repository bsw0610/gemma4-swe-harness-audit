"""Round-trip tests for trace_metrics against the harness's own trace writer (adk_eval_core 0.1.0 SessionTrace, ATIF v1.7).
The package source is read from work/pkgsrc (unpacked wheel); yaml is stubbed because only the tracing code is used.
Run: uv run python analysis/test_trace_metrics.py
"""
import json
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
pkg = types.ModuleType("adk_eval_core")
pkg.__path__ = [str(ROOT / "work/pkgsrc/adk_eval_core")]
sys.modules["adk_eval_core"] = pkg
from adk_eval_core.tracing.trace import SessionTrace, TraceEntry  # noqa: E402

from trace_metrics import flatten, gold_of, metrics  # noqa: E402

PATCH = (
    "--- a/pkg/core.py\n+++ b/pkg/core.py\n@@ -10,3 +10,3 @@ def f():\n     a = 1\n-    b = 2\n+    b = 3\n     c = 4\n"
    "--- a/pkg/new.py\n+++ b/pkg/new.py\n@@ -0,0 +1,2 @@\n+x = 1\n+y = 2\n"
    "--- a/tests/test_core.py\n+++ b/tests/test_core.py\n@@ -1,1 +1,2 @@\n x = 1\n+y = 2\n"
)


def resp(**kw):
    return json.dumps({"status": "ok", **kw})


def build():
    t = SessionTrace()
    t.start()
    seq = [  # (type, tool, args, result)
        ("thinking", "", None, ""),
        ("tool_call", "run_skill_script", {"skill_name": "localize-code", "file_path": "scripts/localize.py",
                                           "args": {"query": "b is wrong"}}, ""),
        ("tool_response", "run_skill_script", None, resp(stdout="Likely files:\n  pkg/core.py\n")),
        # two calls in one model turn -> one ATIF step with two tool_calls, second response as a system step
        ("tool_call", "read_file", {"filepath": "/workspace/pkg/util.py"}, ""),
        ("tool_call", "read_file", {"filepath": "pkg/core.py", "start_line": 1, "end_line": 5}, ""),
        ("tool_response", "read_file", None, resp(filepath="pkg/util.py", content="u\n", start_line=1)),
        ("tool_response", "read_file", None, resp(filepath="pkg/core.py", content="1\n2\n3\n4\n5", start_line=1)),
        ("text", "", None, ""),
        ("tool_call", "read_file", {"filepath": "./pkg/core.py", "start_line": 8}, ""),
        ("tool_response", "read_file", None, resp(filepath="pkg/core.py", content="8\n9\n10\n11", start_line=8)),
        ("tool_call", "edit_file", {"filepath": "pkg/core.py", "old_string": "b = 2", "new_string": "b = 3"}, ""),
        ("tool_response", "edit_file", None, resp()),
        ("tool_call", "submit_patch", {}, ""),
        ("tool_response", "submit_patch", None, resp(patch_size=10)),
    ]
    for i, (typ, tool, args, res) in enumerate(seq):
        t._add(TraceEntry(timestamp=1000.0 + i, elapsed=float(i), event_type=typ, author="agent",
                          content="..." if typ in ("thinking", "text") else "", tool_name=tool, tool_args=args,
                          tool_result=res))
    return t, seq


def main():
    t, seq = build()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "trace_x.json"
        t.save(p)
        atif = json.loads(p.read_text())
    assert atif["schema_version"].startswith("ATIF"), atif["schema_version"]
    multi = [s for s in atif["steps"] if len(s.get("tool_calls") or []) > 1]
    assert multi, "fixture must exercise a step with several tool calls"

    calls = flatten(atif)
    want = [(tool, args) for typ, tool, args, _ in seq if typ == "tool_call"]
    assert [(c["tool"], c["args"]) for c in calls] == want, calls
    want_res = [res for typ, _, _, res in seq if typ == "tool_response"]
    assert [c["result"] for c in calls] == want_res, [c["result"] for c in calls]

    # second oracle: the harness's own loader gives the same call/response order
    back = SessionTrace.from_dict(atif).entries
    assert [(e.tool_name, e.tool_args) for e in back if e.event_type == "tool_call"] == want
    assert [e.tool_result for e in back if e.event_type == "tool_response"] == want_res

    assert gold_of(PATCH) == {"pkg/core.py": {11}}, gold_of(PATCH)
    ins = "--- a/m.py\n+++ b/m.py\n@@ -3,2 +3,3 @@\n a\n+b\n c\n"
    assert gold_of(ins) == {"m.py": {3, 4}}, gold_of(ins)

    m = metrics(atif, PATCH)
    expect = {"n_gold_files": 1, "n_tool_calls": 6, "skill_hit": True,
              "first_skill_step": 1, "first_mention_step": 1, "first_open_step": 3,
              "first_see_lines_step": 4, "first_edit_step": 5,
              "first_open_work_step": 2, "first_edit_work_step": 4, "first_skill_work_step": 0, "first_skill_any_step": 1, "edited_files": '["pkg/core.py"]'}
    bad = {k: (m[k], v) for k, v in expect.items() if m[k] != v}
    assert not bad, bad
    print("trace_metrics: all checks passed;", json.dumps(m))


if __name__ == "__main__":
    main()

# Failed calls and non-read commands are not events.
from trace_metrics import succeeded  # noqa: E402
assert succeeded(json.dumps({"status": "ok"})) and not succeeded(json.dumps({"status": "error"})) and not succeeded("mandatory input parameters are not present")
_step = lambda name, args, res: {"tool_calls": [{"function_name": name, "arguments": args}], "observation": {"content": res, "extra": {"tool_name": name}}}
_bad = {"steps": [_step("read_file", {"filepath": "pkg/core.py"}, json.dumps({"status": "error"})),
                  _step("run_command", {"command": "echo pkg/core.py"}, resp(stdout="pkg/core.py")),
                  _step("run_command", {"command": "grep x pkg/core.py | head"}, resp(stdout="")),
                  _step("edit_file", {"filepath": "pkg/core.py"}, json.dumps({"status": "error"}))]}
_m = metrics(_bad, PATCH)
assert _m["first_open_step"] is None and _m["first_edit_step"] is None, _m
