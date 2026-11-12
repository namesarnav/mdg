"""
Generate one Colab notebook per trained model: 3 architectures x 11 datasets.

    python3 colab/make_notebooks.py

Writes colab/notebooks/<architecture>/<dataset>__<architecture>.ipynb and an
index, colab/notebooks/README.md. Edit THIS file and re-run rather than editing
a notebook by hand, or the 33 drift apart.

Each notebook attacks every data point of its dataset with every recipe via
colab/attack_all.py, so the full set produces 18 x 11 x 3 = 594 result files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = ROOT / "notebooks"
GITHUB = "namesarnav/mdg"

# Same grid as mdg/hpc/matrix.sh. The model for (dataset, arch) is
# namesarnav/<dataset>-<arch> on the Hub.
DATASETS = {
    "counterbench":         "YES,NO",
    "ac-reason":            "YES,NO",
    "bbh-causal-judgement": "YES,NO",
    "causalbench_code":     "YES,NO",
    "causalbench_math":     "YES,NO",
    "causalbench_text":     "YES,NO",
    "corr2cause":           "0,1",
    "e-care":               "0,1",
    "fincausal-task1":      "0,1",
    "natquest":             "YES,NO",
    "Quriosity":            "YES,NO",
}
ARCHS = ["bert-base-uncased", "roberta-base", "t5-base"]

# Rows per part file. A part is the unit copied to Drive when it completes
# (in-progress parts are also copied every 2 minutes). corr2cause averages
# 400 words per row, so its parts are kept small.
CHUNK = {"corr2cause": 50}
DEFAULT_CHUNK = 200


def dataset_stats(dataset: str):
    """(rows after dedup, mean words) from the local splits, if present."""
    data = REPO / "mdg" / "finetune" / "data"
    seen, words = set(), 0
    for split in ("train", "validation", "test"):
        p = data / f"namesarnav_{dataset}__{split}.jsonl"
        if not p.exists():
            continue
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            key = hashlib.sha1(json.dumps([r["text"], str(r["label"])]).encode()).digest()
            if key not in seen:
                seen.add(key)
                words += len(r["text"].split())
    return (len(seen), words / len(seen)) if seen else (None, None)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n")}


def notebook(dataset: str, arch: str, rows, words) -> dict:
    model = f"namesarnav/{dataset}-{arch}"
    labels = DATASETS[dataset]
    chunk = CHUNK.get(dataset, DEFAULT_CHUNK)
    size = f"{rows:,} rows, ~{words:.0f} words each" if rows else "row count unknown"
    out = f"results/namesarnav_{dataset}__all/{arch}/"

    cells = [
        md(f"""
# Adversarial attacks — `{dataset}` × `{arch}`

Attacks **[{model}](https://huggingface.co/{model})** on **every data point** of
`{dataset}` ({size}) with all 18 TextAttack recipes.

Output: one file per recipe on Drive, in `MyDrive/mdg/{out}`
- `<recipe>.jsonl`: one line per example, with the original and perturbed text and both predictions
- `<recipe>_summary.json`: attack success rate, plus clean and attacked micro/macro F1

**Use a GPU runtime** (*Runtime → Change runtime type → T4 or better*). Then *Runtime → Run all*.

**If the session dies:** reconnect and *Run all* again. Every attacked example
is on Drive within 2 minutes, and the run resumes at the exact example where it
stopped. Local disk only ever holds the part in progress, so it stays far below
Colab's limit.
"""),
        md("## 0. Settings"),
        code(f"""
DATASET = '{dataset}'
MODEL   = '{model}'
LABELS  = '{labels}'
CHUNK   = {chunk}          # rows per part file on Drive

# Running this SAME notebook in several Colab sessions at once? Give each a
# different slice: '0/2' and '1/2', or '0/3', '1/3', '2/3'. Leave '0/1' otherwise.
WORKER  = '0/1'

QUERY_BUDGET        = None   # None = unlimited search, as in the original runs
EXAMPLE_TIMEOUT_MIN = 30     # an example still running after this is recorded as Timeout

DRIVE = '/content/drive/MyDrive/mdg'
LOCAL = '/content/mdg-local'
"""),
        md("## 1. GPU"),
        code("""
import subprocess
try:
    out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    gpu, vram = out.strip().split(', ')
    vram = int(vram)
except Exception:
    gpu, vram = None, 0
# Queries per forward pass. TextAttack's default of 32 leaves most of a GPU idle.
BATCH = 512 if vram >= 38000 else 256 if vram >= 20000 else 128 if vram >= 14000 else 64 if vram else None
print(f'GPU: {gpu or "NONE -- switch the runtime type to a GPU"}  ({vram} MB)  query batch: {BATCH}')
"""),
        md("## 2. Mount Drive"),
        code("""
from google.colab import drive
drive.mount('/content/drive')
import os
os.makedirs(DRIVE, exist_ok=True)
os.makedirs(LOCAL, exist_ok=True)
"""),
        md("## 3. Code and dependencies"),
        code(f"""
import os
if not os.path.isdir('/content/mdg/.git'):
    # GIT_LFS_SKIP_SMUDGE: clone code only. The data comes from Drive, or just
    # this dataset's files are pulled in the next cell.
    !GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 https://github.com/{GITHUB}.git /content/mdg
else:
    !cd /content/mdg && GIT_LFS_SKIP_SMUDGE=1 git pull -q --ff-only
%cd /content/mdg
!git log --oneline -1
!pip install -q "textattack==0.3.11" "transformers>=4.46,<5" accelerate sentencepiece "datasets>=3.0" "huggingface_hub<1.0" tensorflow_hub scikit-learn
!pip cache purge -q 2>/dev/null; true
!python3 colab/nltk_check.py
"""),
        md("""
## 4. Data — every data point

`train + validation + test`, exact duplicates dropped. Built once from Git LFS
and kept in `MyDrive/mdg/data/`, so later sessions (and the other two models on
this dataset) copy it instead of pulling it again.
"""),
        code("""
import os, shutil
fname = f'namesarnav_{DATASET}__all.jsonl'
on_drive = f'{DRIVE}/data/{fname}'
local = f'/content/mdg/mdg/finetune/data/{fname}'
if os.path.exists(on_drive):
    shutil.copyfile(on_drive, local)
    print('copied from Drive:', on_drive)
else:
    !apt-get -qq install -y git-lfs > /dev/null
    !cd /content/mdg && git lfs install --local > /dev/null && git lfs pull --include="mdg/finetune/data/namesarnav_{DATASET}__*.jsonl"
    !cd /content/mdg && python3 colab/attack_all.py prepare --dataset {DATASET}
    os.makedirs(f'{DRIVE}/data', exist_ok=True)
    shutil.copyfile(local, on_drive)
    print('saved to Drive:', on_drive)
!wc -l {local}
"""),
        md("""
## 5. Check every recipe

Builds each recipe and attacks 2 examples. This also downloads every recipe's
assets (embeddings, sentence encoders, language models), a few GB in all, to
local disk. A recipe that fails here would fail on every row, so this stops
the notebook instead of letting it burn the session.
"""),
        code("""
ARGS = (f'--dataset {DATASET} --model {MODEL} --labels {LABELS} --chunk {CHUNK} '
        f'--worker {WORKER} --drive {DRIVE} --local {LOCAL}')
if BATCH:               ARGS += f' --batch {BATCH}'
if QUERY_BUDGET:        ARGS += f' --query-budget {QUERY_BUDGET}'
if EXAMPLE_TIMEOUT_MIN: ARGS += f' --example-timeout-min {EXAMPLE_TIMEOUT_MIN}'
r = !cd /content/mdg && python3 colab/attack_all.py check {ARGS} 2>&1
print('\\n'.join(r))
if any('[FAIL]' in l for l in r):
    raise SystemExit('A recipe failed its check -- see above. Not starting the run.')
"""),
        md("""
## 6. Run

Recipes go cheapest first. Progress prints every 2 minutes, and every finished
recipe becomes one `.jsonl` + `_summary.json` on Drive. **Re-run this cell
(or Run all) to continue after a disconnect.**
"""),
        code("""
!cd /content/mdg && python3 -u colab/attack_all.py run {ARGS} 2>&1 | tee -a {LOCAL}/run.log | grep --line-buffered -v -E "^(WARNING|I0000|W0000)"
"""),
        md("## 7. Progress (safe to run any time, from any session)"),
        code("""
!cd /content/mdg && python3 colab/attack_all.py status {ARGS}
!df -h /content | tail -1
"""),
    ]
    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4",
                      "name": f"{dataset}__{arch}.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def main() -> None:
    stats = {d: dataset_stats(d) for d in DATASETS}
    index = [
        "# Attack notebooks — one per trained model",
        "",
        "Generated by `colab/make_notebooks.py`; edit that, not the notebooks.",
        "Each attacks every data point of its dataset with all 18 recipes and",
        "writes one `.jsonl` + `_summary.json` per recipe to",
        "`MyDrive/mdg/results/namesarnav_<dataset>__all/<model>/`.",
        "",
        "Sorted cheapest first: rows × words per row. corr2cause outweighs the",
        "other ten datasets combined, so start it early and give it extra",
        "sessions (`WORKER = '0/3'`, `'1/3'`, `'2/3'`).",
        "",
        "| Dataset | Rows | Words/row | " + " | ".join(ARCHS) + " |",
        "|---|---:|---:|" + "---|" * len(ARCHS),
    ]
    order = sorted(DATASETS, key=lambda d: (stats[d][0] or 0) * (stats[d][1] or 0))
    for d in order:
        rows, words = stats[d]
        links = []
        for a in ARCHS:
            rel = f"colab/notebooks/{a}/{d}__{a}.ipynb"
            path = OUT / a / f"{d}__{a}.ipynb"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(notebook(d, a, rows, words), indent=1) + "\n")
            links.append(f"[open](https://colab.research.google.com/github/{GITHUB}/blob/main/{rel})")
        index.append(f"| {d} | {rows or '?':,} | {words or 0:.0f} | " + " | ".join(links) + " |"
                     if rows else f"| {d} | ? | ? | " + " | ".join(links) + " |")
    (OUT / "README.md").write_text("\n".join(index) + "\n")
    print(f"wrote {len(DATASETS) * len(ARCHS)} notebooks + README to {OUT}")


if __name__ == "__main__":
    main()
