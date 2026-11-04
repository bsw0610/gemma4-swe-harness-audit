"""Colab step 2 downloads (approved 2026-10-06): competition subset, wheelhouse, Gemma 4 31B QAT weights.
Run inside the Python 3.12 venv with KAGGLE_API_TOKEN set from Colab Secrets.
C3: `download.py N` also fetches the first N tasks of c3_tasks.txt (next to this file); already present files are skipped."""
import json, pathlib, subprocess, sys, zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi(); api.authenticate()
C, D = "gemma-4-developer-agent", pathlib.Path("/content/data")
TASK_IDS = ["rich_3882", "rich_4076"]
if len(sys.argv) > 1:
    TASK_IDS += (pathlib.Path(__file__).resolve().parent / "c3_tasks.txt").read_text().split()[: int(sys.argv[1])]

names, tok = [], None
while True:
    r = api.competition_list_files(C, page_token=tok, page_size=200)
    names += [f.name for f in r.files]
    tok = getattr(r, "next_page_token", None)
    if not tok:
        break
print("competition files listed:", len(names))


def get(n):
    dest = D / pathlib.Path(n).parent
    dest.mkdir(parents=True, exist_ok=True)
    if (D / n).exists() and (D / n).stat().st_size > 0:
        return
    api.competition_download_file(C, n, path=str(dest), quiet=True)
    z = dest / (pathlib.Path(n).name + ".zip")
    if z.exists():
        zipfile.ZipFile(z).extractall(dest)
        z.unlink()


get("tasks.jsonl")
T = {json.loads(l)["instance_id"]: json.loads(l) for l in open(D / "tasks.jsonl")}
want = [n for n in names if n.startswith(("wheels/", "sample_submission/", "sandbox/", "docker/"))]
for tid in TASK_IDS:
    stem = f"{T[tid]['repo'].split('/')[1].lower()}_{T[tid]['base_commit']}"
    want += [f"snapshots/{tid}.tgz", f"graphs/{stem}.json", f"embeddings/{stem}.npz"]
for n in want:
    get(n)
print("competition subset:", len(want), "files,", round(sum(p.stat().st_size for p in D.rglob("*") if p.is_file()) / 1e9, 2), "GB")

api.dataset_download_files("metric/gemma-4-developer-agent-wheelhouse", path="/content/wheelhouse", unzip=True, quiet=True)
print("wheelhouse:", len(list(pathlib.Path("/content/wheelhouse").glob("*.whl"))), "wheels")

if not any(pathlib.Path("/content/model").rglob("config.json")):
    subprocess.run(["/content/py312/bin/kaggle", "models", "instances", "versions", "download",
                    "google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2", "-p", "/content/model", "--untar"], check=True)
print("model files:", sorted(str(p) for p in pathlib.Path("/content/model").rglob("*") if p.is_file()))
