"""Run the localize-code skill the way the harness's subprocess sandbox (Kaggle/Colab) runs it, and check it finds the repo.

Reproduces, from the source of the installed versions:
- the run_skill_script wrapper for .py scripts (google_adk 1.36.1, google/adk/tools/skill_toolset.py:617-712): skill files are
  written to a TemporaryDirectory, the wrapper chdirs there, sets sys.argv and runs the script with runpy;
- AdkSandboxCodeExecutor (adk_eval_core/sandbox/base.py:201-210): wrapper saved as <workspace>/.adk_exec_<hex>.py, run with
  `python3 /workspace/.adk_exec_<hex>.py`, removed afterwards;
- SubprocessSandboxManager.exec (swegemma/sandbox/subprocess.py:420-500): "/workspace" is rewritten in the COMMAND only,
  /bin/bash -c with cwd = workspace and a sanitized env whose PYTHONPATH is the workspace.
Also checks the skill directory against the loader rules of the same versions (a skill that fails them makes the
submission fail to compile):
- google_adk 1.36.1 load_skill_from_dir: frontmatter `name` must equal the directory name (google/adk/skills/_utils.py:150-157);
  name <= 64 chars and lowercase kebab-case, because snake_case is the experimental SNAKE_CASE_SKILL_NAME feature, off by
  default (skills/models.py:80-103, features/_feature_registry.py:176-177); description non-empty, <= 1024 chars;
- swegemma 0.2.7 submission limits: only .py/.md/.txt/.json/.yaml/.yml/.safetensors files, no symlinks, <= 50 MiB
  (swegemma/config.py:20-57).
Run: uv run python analysis/test_skill_sandbox.py
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skill/localize-code"


def wrapper(script_args=None, positional_args=None):
    files = {"SKILL.md": (SKILL / "SKILL.md").read_text(),
             "scripts/localize.py": (SKILL / "scripts/localize.py").read_text()}
    argv = ["scripts/localize.py"]
    for k, v in (script_args or {}).items():
        argv += [f"--{k}", str(v)]
    if positional_args:
        argv += ["--"] + [str(v) for v in positional_args]
    return "\n".join([
        "import os", "import tempfile", "import sys", "import json as _json", "import subprocess", "import runpy",
        f"_files = {files!r}",
        "def _materialize_and_run():",
        "  _orig_cwd = os.getcwd()",
        "  with tempfile.TemporaryDirectory() as td:",
        "    for rel_path, content in _files.items():",
        "      full_path = os.path.join(td, rel_path)",
        "      os.makedirs(os.path.dirname(full_path), exist_ok=True)",
        "      mode = 'wb' if isinstance(content, bytes) else 'w'",
        "      with open(full_path, mode) as f:",
        "        f.write(content)",
        "    os.chdir(td)",
        "    try:",
        f"      sys.argv = {argv!r}",
        "      try:",
        "        runpy.run_path('scripts/localize.py', run_name='__main__')",
        "      except SystemExit as e:",
        "        if e.code is not None and e.code != 0:",
        "          raise e",
        "    finally:",
        "      os.chdir(_orig_cwd)",
        "_materialize_and_run()",
    ])


def run_like_subprocess_sandbox(ws, tmp, code):
    name = ".adk_exec_0123abcd.py"
    (ws / name).write_text(code)
    cmd = re.sub(r"(?<![\w.-])/workspace(?![\w.-])", str(ws), f"python3 /workspace/{name}")
    env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "TERM", "USER", "LOGNAME") if k in os.environ}
    env.update(HOME=str(tmp), TMPDIR=str(tmp), TEST_TMPDIR=str(tmp), PYTHONNOUSERSITE="1", PYTHONPATH=str(ws))
    try:
        return subprocess.run(["/bin/bash", "-c", cmd], cwd=ws, env=env, capture_output=True, text=True, timeout=60)
    finally:
        (ws / name).unlink()


KEBAB = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
ALLOWED_EXT = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".safetensors"}


def load_contract_errors(skill_dir):
    text = (skill_dir / "SKILL.md").read_text()
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return ["SKILL.md has no frontmatter"]
    fm = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    name, desc = fm.get("name", "").strip(), fm.get("description", "").strip()
    errs = []
    if name != skill_dir.name:
        errs.append(f"name {name!r} != directory {skill_dir.name!r}")
    if len(name) > 64 or not KEBAB.match(name):
        errs.append(f"name {name!r} is not lowercase kebab-case (<= 64 chars)")
    if not desc or len(desc) > 1024:
        errs.append("description empty or > 1024 chars")
    files = [p for p in skill_dir.rglob("*") if p.is_file() or p.is_symlink()]
    errs += [f"symlink {p}" for p in files if p.is_symlink()]
    errs += [f"extension {p.name}" for p in files if p.suffix not in ALLOWED_EXT]
    if sum(p.stat().st_size for p in files) > 50 * 1024 * 1024:
        errs.append("skill directory > 50 MiB")
    return errs


def main():
    errs = load_contract_errors(SKILL)
    assert not errs, errs
    with tempfile.TemporaryDirectory() as d:  # negative control: the first (snake_case) name is rejected
        old = Path(d) / "localize_code"
        old.mkdir()
        (old / "SKILL.md").write_text((SKILL / "SKILL.md").read_text().replace("name: localize-code", "name: localize_code"))
        assert any("kebab" in e for e in load_contract_errors(old))
    print("skill load contract: ok (old snake_case name rejected)")

    assert not Path("/workspace").exists(), "this test models a host without /workspace"
    with tempfile.TemporaryDirectory() as d:
        ws, tmp = Path(d) / "sandbox/workspace", Path(d) / "sandbox/tmp"
        (ws / "pkg").mkdir(parents=True)
        tmp.mkdir()
        (ws / "pkg/__init__.py").write_text("")
        (ws / "pkg/render.py").write_text("def render_markup(text):\n    '''Render console markup.'''\n    return text\n")
        (ws / "pkg/other.py").write_text("def unrelated():\n    return 1\n")
        subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
        subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
        before = subprocess.run(["git", "status", "--porcelain", "--ignored"], cwd=ws, capture_output=True, text=True).stdout

        query = "render_markup breaks console markup"
        for label, code in [("args dict", wrapper(script_args={"query": query})),
                            ("positional", wrapper(positional_args=[query])),
                            ("stale --root /workspace", wrapper(script_args={"root": "/workspace", "query": query}))]:
            r = run_like_subprocess_sandbox(ws, tmp, code)
            assert r.returncode == 0, (label, r.returncode, r.stdout, r.stderr)
            assert "pkg/render.py:1  render_markup" in r.stdout, (label, r.stdout)
            assert "localize.py" not in r.stdout, (label, "ranked the skill's own temp dir", r.stdout)
            print(f"{label}: ok -> {r.stdout.splitlines()[1].strip()}")

        after = subprocess.run(["git", "status", "--porcelain", "--ignored"], cwd=ws, capture_output=True, text=True).stdout
        assert after == before, ("workspace changed", before, after)
    print("skill sandbox: all checks passed (workspace unchanged)")


if __name__ == "__main__":
    sys.exit(main())
