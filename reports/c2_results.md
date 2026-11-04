# C2 results: localize-code skill (offline, 2026-10-06)

Skill: `skill/localize-code/` (`SKILL.md`, `scripts/localize.py`; standard library only, Python ≥ 3.8, read-only).
Evaluation: Kaggle kernel `kaggle/skill_eval` (version 2) runs the exact script on all 129 base snapshots with the issue
text as the query; `analysis/skill_eval.py` scores it against AST gold v4 (`work/c2/skill_eval/score_stdout.txt`).
n = 115 tasks with in-graph gold; 95% CI = task bootstrap, 10,000 draws, seed 0 (as in C1).

| k | symbol hit@k (in-graph gold) | symbol hit@k (any gold, incl. async/nested) |
|---|---|---|
| 5 | 46/115 = 40.0% [31.3, 48.7] | 48/115 = 41.7% |
| 10 | 53/115 = 46.1% [36.5, 55.7] | 56/115 = 48.7% |
| 15 (default output) | 59/115 = 51.3% [42.6, 60.0] | 62/115 = 53.9% |
| 50 | 89/115 = 77.4% [69.6, 85.2] | 90/115 = 78.3% |
| 100 | 100/115 = 87.0% [80.9, 93.0] | 100/115 = 87.0% |

- File hit@1/3/5/10: 28.7% / 54.8% / 65.2% / 82.6%.
- Runtime per task on Kaggle CPU: median 0.57 s, max 1.50 s. Default output (top 15 symbols, 5 files) stays under 4,000 characters.
- Compared with the provided 0.2.7 graph tools (2/115 end-to-end at any k) and the 0.2.11 tools (30–48% end-to-end at k = 50, `harness_0_2_11_recheck.md`), the skill's default top-15 list already contains an in-graph gold symbol for 51% of tasks. These lists differ in corpus and length, so this is not a controlled comparison.
- The script is a separate BM25 implementation from the C1 snapshot baseline, with different filtering, tokenisation and ranking details; it was not tuned on these tasks.

## Invocation contract (verified from code: google_adk 1.36.1, adk_submission 0.2.12, adk_eval_core 0.1.0)
- Declared per agent in `agent.yaml`: `skills: [skills/localize-code]`. Skill directories may contain only .py/.md/.txt/.json/.yaml/.yml/.safetensors files,
  at most 50 MiB, and no symlinks (`swegemma/config.py:20-57`).
- Declaring a skill gives the agent four tools: `list_skills`, `load_skill`, `load_skill_resource` and
  `run_skill_script(skill_name, file_path, args | short_options | positional_args)` (`google/adk/tools/skill_toolset.py:92-800`).
- `run_skill_script` builds a self-extracting wrapper (`skill_toolset.py:571-712`) that is written to `/workspace/.adk_exec_<hex>.py` and run with `python3`
  in the task sandbox (`adk_eval_core/sandbox/base.py:164-223`).
  - The wrapper writes the skill's files into a temporary directory, changes into it, sets `sys.argv` and runs the script with `runpy`.
  - A dict argument becomes `--key value`; positional arguments follow a `--`.
- Each call costs one tool call and wall-clock time; the timeout is min(300 s, remaining session time). The sandbox has no network access.
- The script writes nothing and handles both `args={"query": ...}` and `positional_args`.
- Repository root (fixed 2026-10-06): the subprocess sandbox used on Kaggle/Colab (`swegemma/sandbox/subprocess.py:420-500`)
  rewrites "/workspace" only in shell command strings, and the wrapper runs the script from a temporary directory, so the first
  version (hard-coded `/workspace`) found no files there. The script now takes the first git checkout among `/workspace`,
  `$PWD` and `$PYTHONPATH` (the sandbox sets both to the task workspace). `analysis/test_skill_sandbox.py` reproduces the wrapper
  and the sandbox's command rewriting and environment: the old script fails, the fixed one ranks the right file in all three
  argument forms and leaves the workspace unchanged. Ranking code is unchanged, so the offline numbers above (run with `--root`) stand.
- Skill name (fixed 2026-10-06): google_adk 1.36.1 `load_skill_from_dir` requires the frontmatter name to equal the directory name and,
  unless the experimental SNAKE_CASE_SKILL_NAME feature is enabled (off by default; `features/_feature_registry.py:176-177`), to be
  lowercase kebab-case (`skills/models.py:80-103`). The first name `localize_code` would have failed submission compilation;
  the skill is now `localize-code` (`skills: [skills/localize-code]`, `run_skill_script(skill_name="localize-code", ...)`).

## Agent runs (`analysis/c2_pilot_summary.py`, traces not published)
Skill uptake: 0 of 8 skill-arm runs on 0.2.7 (5 pilot, 3 pre-flight). 7 traces are retained, all with no skill call; the first pre-flight trace was overwritten and its 0 is from our run log. In all 10 pilot runs (both arms) the first successful gold-file read came within 4 tool calls. Not re-measured on 0.2.11.
