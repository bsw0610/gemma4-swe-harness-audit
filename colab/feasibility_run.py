"""Colab A100 feasibility run (step 2): does gemma-4-31b-it-qat-w4a16-ct serve under vLLM on one A100-40GB,
and what does one agent task cost? Mirrors the official starter notebook (ryanholbrook/getting-started-gemma-4-developer-agent)
but with tensor_parallel_size=1, no LoRA adapters and realistic per-task budgets.

Run inside the Python 3.12 venv:  /content/py312/bin/python /content/feasibility_run.py
Writes /content/work/feasibility.json and /content/work/results/ (patches, traces, logs).
"""
import asyncio
import glob
import json
import os
import pathlib
import re
import shutil
import time

os.environ.update({
    "LITELLM_LOCAL_MODEL_COST_MAP": "True", "TRANSFORMERS_NO_TF": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1", "VLLM_ENGINE_READY_TIMEOUT_S": "1800", "VLLM_NO_USAGE_STATS": "1",
    "OTEL_SDK_DISABLED": "true", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
})

from adk_submission import VllmConfig, VllmServer  # noqa: E402
from google.adk.agents.context_cache_config import ContextCacheConfig  # noqa: E402
from google.adk.apps._configs import EventsCompactionConfig  # noqa: E402
from swegemma.config import EvalConfig, build_submission_limits  # noqa: E402
from swegemma.evaluate import Evaluator  # noqa: E402
from swegemma.models import load_tasks  # noqa: E402

DATA = pathlib.Path("/content/data")
WORK = pathlib.Path("/content/work")
TASK_IDS = ["rich_3882", "rich_4076"]
MODEL_NAME = "gemma-4-31b-it-qat-w4a16-ct"
MODEL_DIR = pathlib.Path(sorted(glob.glob("/content/model/**/config.json", recursive=True))[0]).parent
BUDGET = dict(timeout_seconds=300, max_tool_calls=100, max_time_minutes=30.0, max_turns=500)
out = {"model_dir": str(MODEL_DIR), "tasks": TASK_IDS, "budget": BUDGET}

# ---- minimal agent: the sample prompt and sampling config, no LoRA adapter, no sub-agent
AGENT = WORK / "agent"
shutil.rmtree(AGENT, ignore_errors=True)
shutil.copytree(DATA / "sample_submission", AGENT, ignore=shutil.ignore_patterns("adapters", "sub_agents"))
(AGENT / "agent.yaml").write_text(
    "name: swe_baseline_agent\n"
    f"model: {MODEL_NAME}\n"
    "instruction: !include prompts/system.md\n"
    "tools:\n  - run_command\n  - read_file\n  - edit_file\n  - write_file\n  - get_status\n  - submit_patch\n"
    "  - get_code_neighbors\n  - search_similar_code\n  - get_code_subgraph\n"
    "generate_content_config: !include configs/sampling.yaml\n")
(AGENT / "configs" / "sampling.yaml").write_text(
    "temperature: 0.2\ntop_p: 0.95\nmax_output_tokens: 16384\nthinking_config:\n  thinking_budget: 4096\n  include_thoughts: true\n")
(AGENT / "eval_config.yaml").write_text("evaluation:\n" + "".join(f"  {k}: {v}\n" for k, v in BUDGET.items()))

# ---- start vLLM: try the scoring context length first, then shorter ones
server = None
for max_len in (32768, 24576, 16384):
    cfg = VllmConfig(model=str(MODEL_DIR), port=8000, host="127.0.0.1", tool_call_parser="gemma4", reasoning_parser="gemma4",
                     max_model_len=max_len, dtype="bfloat16", gpu_memory_utilization=0.92, enable_auto_tool_choice=True,
                     enable_lora=False, tensor_parallel_size=1, startup_timeout=1800, nvidia_lib_dir="/usr/lib64-nvidia")
    srv = VllmServer(cfg, adapter_manifest=None)
    t0 = time.time()
    try:
        srv.start()
        server = srv
        out["max_model_len"] = max_len
        out["startup_seconds"] = round(time.time() - t0, 1)
        break
    except Exception as e:  # noqa: BLE001
        out.setdefault("failed_lengths", []).append({"max_model_len": max_len, "error": f"{type(e).__name__}: {e}"[:300]})
        try:
            srv.stop()
        except Exception:  # noqa: BLE001
            pass
log_text = pathlib.Path(srv.log_path).read_text(errors="replace") if getattr(srv, "log_path", None) else ""
out["vllm_log_key_lines"] = [l.strip()[:200] for l in log_text.splitlines()
                             if re.search(r"KV cache|Maximum concurrency|GPU blocks|model weights took|memory|Error|error", l)][-25:]
print(json.dumps({k: out[k] for k in out if k != "vllm_log_key_lines"}, indent=2))
print("\n".join(out["vllm_log_key_lines"]))
if server is None:
    (WORK / "feasibility.json").write_text(json.dumps(out, indent=2))
    raise SystemExit("vLLM did not start at any context length")

# ---- run the tasks with the official harness (subprocess sandbox, as in the starter notebook)
models = server.create_model_registry(aliases=[MODEL_NAME], model_prefix="openai/", api_key="EMPTY")
limits, gen_constraints = build_submission_limits()
results_dir = WORK / "results"
shutil.rmtree(results_dir, ignore_errors=True)
eval_config = EvalConfig(
    tasks_path=DATA / "tasks.jsonl", snapshots_dir=DATA / "snapshots", results_dir=results_dir, submission_dir=AGENT,
    models=models, sandbox="subprocess", timeout_seconds=BUDGET["timeout_seconds"], max_time_minutes=BUDGET["max_time_minutes"],
    max_tool_calls=BUDGET["max_tool_calls"], max_turns=BUDGET["max_turns"], limits=limits, generation_constraints=gen_constraints,
    adapter_manifest=None, context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
    events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
    graph_dir=DATA / "graphs", embeddings_dir=DATA / "embeddings", wheels_dir=DATA / "wheels", verbose=False)
evaluator = Evaluator(eval_config)
tasks = [t for t in load_tasks(DATA / "tasks.jsonl") if t.instance_id in TASK_IDS]
out["runs"] = []
for i, task in enumerate(tasks, 1):
    t0 = time.time()
    r = asyncio.run(evaluator.evaluate_task(task=task, task_index=i, total_tasks=len(tasks)))
    out["runs"].append(dict(task=task.instance_id, resolved=r.resolved, exit_code=r.test_exit_code,
                            patch_chars=len(r.agent_patch or ""), tool_calls=r.tool_calls,
                            agent_seconds=round(getattr(r, "duration_seconds", 0) or 0, 1), wall_seconds=round(time.time() - t0, 1)))
    print(out["runs"][-1], flush=True)
out["result_files"] = sorted(str(p.relative_to(results_dir)) for p in results_dir.rglob("*") if p.is_file())
(WORK / "feasibility.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out["runs"], indent=2))
print("files:", out["result_files"])
server.stop()
