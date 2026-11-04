"""Execute the REAL swegemma run_agent_sandbox with a mock LLM that raises litellm.ContextWindowExceededError.

Cases: "overflow_after_edit" (turn 1 edits pkg/mod.py + writes tests/test_x.py, turn 2 raises) and
"overflow_at_turn1" (control: raises before any edit). Real: run_agent_sandbox, SubprocessManager, compile_submission,
ADK Runner, snapshot extraction, fallback git diff. Only the LLM is mocked. Usage: python test_overflow_salvage.py [out.json]
"""
import asyncio, importlib.metadata as md, json, logging, os, re, subprocess, sys, tarfile, tempfile
from pathlib import Path

os.environ.update({"LITELLM_LOCAL_MODEL_COST_MAP": "True", "OTEL_SDK_DISABLED": "true",
                   "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})
import litellm
from google.adk.models import BaseLlm, LlmResponse
from google.genai import types as gt
from adk_submission import ModelRegistry
from swegemma.config import EvalConfig, EvaluationBudget
from swegemma.harness.agent_runner import run_agent_sandbox
from swegemma.models import Task
from swegemma.sandbox import SubprocessManager

STATE = {"mode": None, "calls": 0, "tool_results": []}  # module-level: ADK deep-copies the model object


class MockLlm(BaseLlm):
    @classmethod
    def supported_models(cls):
        return ["mock"]

    async def generate_content_async(self, llm_request, stream=False):
        STATE["calls"] += 1
        for c in llm_request.contents:  # record tool results the model saw, to prove both edits really executed
            for pt in c.parts or []:
                if pt.function_response:
                    STATE["tool_results"].append([pt.function_response.name, str(pt.function_response.response)[:120]])
        if STATE["mode"] == "overflow_at_turn1" or STATE["calls"] >= 2:
            raise litellm.ContextWindowExceededError(
                message="context length exceeded (mock)", model="mock", llm_provider="hosted_vllm")
        fc = lambda n, a: gt.Part(function_call=gt.FunctionCall(name=n, args=a))
        yield LlmResponse(content=gt.Content(role="model", parts=[
            fc("edit_file", {"filepath": "pkg/mod.py", "old_string": "return 1", "new_string": "return 2"}),
            fc("write_file", {"filepath": "tests/test_x.py", "content": "def test_x():\n    assert True\n"})]))


def make_snapshot(root: Path) -> Path:
    repo = root / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "mod.py").write_text("def f():\n    return 1\n")
    for c in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "init"]):
        subprocess.run(["git", "-C", str(repo), *c], check=True)
    snap = root / "snapshots" / "toy__1.tgz"
    snap.parent.mkdir()
    with tarfile.open(snap, "w:gz") as t:
        t.add(repo, arcname=".")
    return snap


async def run_case(mode: str) -> dict:
    STATE.update(mode=mode, calls=0, tool_results=[])
    with tempfile.TemporaryDirectory(prefix="overflow_") as td:
        root = Path(td)
        snap = make_snapshot(root)
        sub = root / "sub"
        (sub / "prompts").mkdir(parents=True)
        (sub / "prompts" / "system.md").write_text("You are a coding agent. Fix the issue.\n")
        (sub / "agent.yaml").write_text("name: toy_agent\nmodel: mock\ninstruction: !include prompts/system.md\n"
                                        "tools:\n  - edit_file\n  - write_file\n  - submit_patch\n")
        models = ModelRegistry()
        models.register("mock", MockLlm(model="mock"))
        cfg = EvalConfig(tasks_path=root / "tasks.jsonl", snapshots_dir=snap.parent, results_dir=root / "res",
                         submission_dir=sub, models=models, sandbox="subprocess", timeout_seconds=60,
                         budget=EvaluationBudget(time_minutes=5, tool_calls=20, turns=20),
                         graph_dir=root / "g", embeddings_dir=root / "e", display_mode="quiet")
        test_patch = ("diff --git a/tests/test_x.py b/tests/test_x.py\nnew file mode 100644\n--- /dev/null\n"
                      "+++ b/tests/test_x.py\n@@ -0,0 +1 @@\n+def test_x(): pass\n")
        task = Task(instance_id="toy__1", repo="org/toy", base_commit="0" * 40, problem_statement="f should return 2",
                    test_patch=test_patch)
        patch, err, _trace = await run_agent_sandbox(SubprocessManager(timeout_seconds=60), cfg, task, snap)
    return {"case": mode, "swegemma": md.version("swegemma"), "litellm": md.version("litellm"),
            "mock_llm_calls": STATE["calls"], "tool_results_seen": STATE["tool_results"], "patch_empty": not patch.strip(), "agent_error": err,
            "patch_files": re.findall(r"^diff --git a/(\S+) b/", patch, re.M), "agent_patch": patch}


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    out = [asyncio.run(run_case(m)) for m in ("overflow_after_edit", "overflow_at_turn1")]
    text = json.dumps(out, indent=2)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text)
    print(text)
