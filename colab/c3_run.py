"""C3: two agent arms paired on the same task and repetition (Colab A100, one GPU).

Arms (--arms, exactly two):
- base : sample_submission prompt, tools and sampling (max_output_tokens 16384), without the LoRA adapter and the
         code_analyzer sub-agent (same agent as colab/feasibility_run.py).
- out8k: base with max_output_tokens 8192, nothing else changed (main C3 treatment, decided 2026-10-06 after the pilot).
         vLLM needs prompt + max_output_tokens <= max_model_len, so the input ceiling rises from 16,384 to 24,576 tokens.
- skill: base + `skills: [skills/localize-code]` in agent.yaml + the one prompt sentence SKILL_LINE below (pilot arm).
Compaction and caching follow HARNESS_README 7.2 (interval 5, overlap 2, threshold 14,336, retention 5) in every arm.
Order: tasks from colab/c3_tasks.txt (fixed random order; --start skips the first N, e.g. the 5 pilot tasks), for each
task reps 1..R; the two arms of one (task, rep) run
concurrently against the same vLLM server, so both see the same load. Resumable: finished (task, rep, arm) rows in
OUT/c3_runs.jsonl are skipped. Put OUT on Google Drive so a disconnect loses at most the pairs in flight.

Run inside the Python 3.12 venv (pre-flight on an already downloaded task: --task-ids rich_3882 --reps 1):
  /content/py312/bin/python /content/c3_run.py --start 5 --n-tasks 120 --reps 1 --out /content/drive/MyDrive/g4swe_c3_main
Writes OUT/c3_runs.jsonl, OUT/<arm>/rep<k>/traces/trace_<id>.json, OUT/<arm>/rep<k>/patches/<id>.patch and
OUT/<arm>/rep<k>/test_output/<id>.txt (official verification output).
"""
import argparse
import asyncio
import glob
import json
import os
import pathlib
import shutil
import time

os.environ.update({
    "LITELLM_LOCAL_MODEL_COST_MAP": "True", "TRANSFORMERS_NO_TF": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1", "VLLM_ENGINE_READY_TIMEOUT_S": "1800", "VLLM_NO_USAGE_STATS": "1",
    "OTEL_SDK_DISABLED": "true", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
})

import venv  # noqa: E402

# Colab's python3.12 has no ensurepip, and venv then calls sys.exit(1). swegemma's subprocess sandbox catches only Exception
# around venv.create(with_pip=True) (swegemma/sandbox/subprocess.py:331-355), so SystemExit kills the run instead of taking
# its own fallback (venv without pip, as in the official Kaggle starter notebook). Turn it into an exception.
_setup_pip = venv.EnvBuilder._setup_pip


def _setup_pip_or_raise(self, context):
    try:
        _setup_pip(self, context)
    except SystemExit as e:
        raise RuntimeError("ensurepip unavailable") from e


venv.EnvBuilder._setup_pip = _setup_pip_or_raise

from adk_submission import VllmConfig, VllmServer  # noqa: E402
from google.adk.tools import skill_toolset as _st  # noqa: E402

# The trace records only agent.instruction, so log once what the skill arm's model actually receives: the tool
# declarations in the first LLM request after SkillToolset adds its instruction (google/adk/tools/skill_toolset.py:1062).
_skill_plr, _seen = _st.SkillToolset.process_llm_request, []


async def _skill_plr_logged(self, *, tool_context, llm_request):
    await _skill_plr(self, tool_context=tool_context, llm_request=llm_request)
    if not _seen:
        _seen.append(1)
        si = str(getattr(llm_request.config, "system_instruction", "") or "")
        print(f"[skill-probe] tools={sorted(getattr(llm_request, 'tools_dict', {}) or {})} "
              f"system_instruction_chars={len(si)} lists_localize_code={'localize-code' in si}", flush=True)


_st.SkillToolset.process_llm_request = _skill_plr_logged
from google.adk.agents.context_cache_config import ContextCacheConfig  # noqa: E402
from google.adk.apps._configs import EventsCompactionConfig  # noqa: E402
from swegemma.config import EvalConfig, build_submission_limits  # noqa: E402
from swegemma.evaluate import Evaluator  # noqa: E402
from swegemma.models import load_tasks  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--n-tasks", type=int, default=0)
p.add_argument("--task-ids", default="", help="comma-separated ids instead of the first --n-tasks of c3_tasks.txt (pre-flight)")
p.add_argument("--start", type=int, default=0, help="skip the first N tasks of c3_tasks.txt")
p.add_argument("--arms", default="base,out8k")
p.add_argument("--reps", type=int, default=2)
p.add_argument("--out", default="/content/drive/MyDrive/g4swe_c3")
p.add_argument("--max-tool-calls", type=int, default=100)
p.add_argument("--max-time-minutes", type=float, default=30.0)
p.add_argument("--max-model-len", type=int, default=32768)
p.add_argument("--sequential", action="store_true", help="run the two arms one after the other (arm order alternates)")
a = p.parse_args()

DATA, HERE = pathlib.Path("/content/data"), pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "gemma-4-31b-it-qat-w4a16-ct"
MODEL_DIR = pathlib.Path(sorted(glob.glob("/content/model/**/config.json", recursive=True))[0]).parent
BUDGET = dict(timeout_seconds=300, max_tool_calls=a.max_tool_calls, max_time_minutes=a.max_time_minutes, max_turns=500)
TASK_IDS = a.task_ids.split(",") if a.task_ids else (HERE / "c3_tasks.txt").read_text().split()[a.start: a.start + a.n_tasks]
assert TASK_IDS, "give --n-tasks N or --task-ids"
SKILL_LINE = ("- Before reading any file, run the `localize-code` skill with the issue text as the query "
              "(`run_skill_script(skill_name=\"localize-code\", file_path=\"scripts/localize.py\", "
              "args={\"query\": \"<issue text>\"})`) and start from the files and functions it ranks highest.\n")
ANCHOR = "### 1. Identify Target Files Immediately\n"


def make_agent(arm):
    d = OUT / "agents" / arm
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(DATA / "sample_submission", d, ignore=shutil.ignore_patterns("adapters", "sub_agents"))
    skills = "skills:\n  - skills/localize-code\n" if arm == "skill" else ""
    (d / "agent.yaml").write_text(
        "name: swe_baseline_agent\n"
        f"model: {MODEL_NAME}\n"
        "instruction: !include prompts/system.md\n"
        "tools:\n  - run_command\n  - read_file\n  - edit_file\n  - write_file\n  - get_status\n  - submit_patch\n"
        "  - get_code_neighbors\n  - search_similar_code\n  - get_code_subgraph\n"
        + skills +
        "generate_content_config: !include configs/sampling.yaml\n")
    max_out = 8192 if arm == "out8k" else 16384
    (d / "configs" / "sampling.yaml").write_text(
        f"temperature: 0.2\ntop_p: 0.95\nmax_output_tokens: {max_out}\nthinking_config:\n  thinking_budget: 4096\n"
        "  include_thoughts: true\n")
    (d / "eval_config.yaml").write_text("evaluation:\n" + "".join(f"  {k}: {v}\n" for k, v in BUDGET.items()))
    if arm == "skill":
        shutil.copytree(HERE / "skill" / "localize-code", d / "skills" / "localize-code")
        sp = d / "prompts" / "system.md"
        text = sp.read_text()
        assert text.count(ANCHOR) == 1, "prompt anchor not found exactly once"
        sp.write_text(text.replace(ANCHOR, ANCHOR + SKILL_LINE))
    return d


ARMS = tuple(a.arms.split(","))
assert len(ARMS) == 2 and set(ARMS) <= {"base", "out8k", "skill"}, ARMS
AGENTS = {arm: make_agent(arm) for arm in ARMS}
(OUT / "c3_config.json").write_text(json.dumps(dict(args=vars(a), budget=BUDGET, tasks=TASK_IDS, skill_line=SKILL_LINE,
                                                     model_dir=str(MODEL_DIR)), indent=2))

cfg = VllmConfig(model=str(MODEL_DIR), port=8000, host="127.0.0.1", tool_call_parser="gemma4", reasoning_parser="gemma4",
                 max_model_len=a.max_model_len, dtype="bfloat16", gpu_memory_utilization=0.92,
                 enable_auto_tool_choice=True, enable_lora=False, tensor_parallel_size=1, startup_timeout=1800,
                 nvidia_lib_dir="/usr/lib64-nvidia")
server = VllmServer(cfg, adapter_manifest=None)
t0 = time.time()
server.start()
print(f"vLLM up in {time.time() - t0:.0f}s", flush=True)
models = server.create_model_registry(aliases=[MODEL_NAME], model_prefix="openai/", api_key="EMPTY")
limits, gen_constraints = build_submission_limits()


EVALUATORS = {}


def evaluator(arm, rep):
    if (arm, rep) in EVALUATORS:
        return EVALUATORS[(arm, rep)]
    rd = OUT / arm / f"rep{rep}"
    rd.mkdir(parents=True, exist_ok=True)
    EVALUATORS[(arm, rep)] = Evaluator(EvalConfig(
        tasks_path=DATA / "tasks.jsonl", snapshots_dir=DATA / "snapshots", results_dir=rd, submission_dir=AGENTS[arm],
        models=models, sandbox="subprocess", timeout_seconds=BUDGET["timeout_seconds"],
        max_time_minutes=BUDGET["max_time_minutes"], max_tool_calls=BUDGET["max_tool_calls"], max_turns=BUDGET["max_turns"],
        limits=limits, generation_constraints=gen_constraints, adapter_manifest=None,
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(compaction_interval=5, overlap_size=2, token_threshold=14336,
                                                        event_retention_size=5),
        graph_dir=DATA / "graphs", embeddings_dir=DATA / "embeddings", wheels_dir=DATA / "wheels", verbose=False))
    return EVALUATORS[(arm, rep)]


async def one(task, arm, rep, i):
    t0 = time.time()
    try:
        r = await evaluator(arm, rep).evaluate_task(task=task, task_index=i, total_tasks=len(TASK_IDS))
    except Exception as e:  # noqa: BLE001  (record and continue; the row marks the failure)
        return dict(task=task.instance_id, arm=arm, rep=rep, resolved=None, harness_error=f"{type(e).__name__}: {e}"[:500],
                    wall_seconds=round(time.time() - t0, 1), started=t0)
    patch = r.agent_patch or ""
    pd = OUT / arm / f"rep{rep}" / "patches"
    pd.mkdir(parents=True, exist_ok=True)
    (pd / f"{task.instance_id}.patch").write_text(patch)
    td = OUT / arm / f"rep{rep}" / "test_output"
    td.mkdir(parents=True, exist_ok=True)
    (td / f"{task.instance_id}.txt").write_text(r.test_output or "")
    return dict(task=task.instance_id, arm=arm, rep=rep, resolved=r.resolved, test_exit_code=r.test_exit_code,
                tool_calls=r.tool_calls, llm_calls=getattr(r, "total_llm_calls", None), error=r.error,
                patch_chars=len(patch), agent_seconds=round(getattr(r, "duration_seconds", 0) or 0, 1),
                wall_seconds=round(time.time() - t0, 1), started=t0)


done_path = OUT / "c3_runs.jsonl"
done = set()
if done_path.exists():
    for line in done_path.read_text().splitlines():
        row = json.loads(line)
        if row.get("resolved") is not None:
            done.add((row["task"], row["rep"], row["arm"]))
tasks = {t.instance_id: t for t in load_tasks(DATA / "tasks.jsonl") if t.instance_id in TASK_IDS}
missing = [tid for tid in TASK_IDS if tid not in tasks or not (DATA / "snapshots" / f"{tid}.tgz").exists()]
assert not missing, f"tasks without snapshot: {missing}"


async def main():  # one event loop for the whole run, so the cached evaluators stay on the loop they were used on
    for i, tid in enumerate(TASK_IDS, 1):
        for rep in range(1, a.reps + 1):
            arms = [arm for arm in (ARMS if (i + rep) % 2 == 0 else ARMS[::-1])
                    if (tid, rep, arm) not in done]
            if not arms:
                continue
            if a.sequential:
                rows = [await one(tasks[tid], arm, rep, i) for arm in arms]
            else:
                rows = await asyncio.gather(*(one(tasks[tid], arm, rep, i) for arm in arms))
            with open(done_path, "a") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
                    print(row, flush=True)
            if any(r.get("harness_error") for r in rows):  # fail fast: a broken arm would burn GPU time unpaired
                raise SystemExit(f"harness error in {tid} rep {rep}; fix before continuing")


try:
    asyncio.run(main())
finally:
    server.stop()
