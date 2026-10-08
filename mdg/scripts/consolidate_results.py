"""
Merge every result this project has produced into one CSV.

Results accumulate in several places over a project's life — per-recipe CSVs,
summary JSONs beside the perturbed data, an appended attack CSV, checkpoint
eval files — and across several machines. This walks all of them, dedupes, and
writes one row per (model, dataset, recipe).

Sources, in order of preference when the same cell appears twice:
  1. mdg/adv_attack/results/<split>/<model>/<recipe>_summary.json   (canonical)
  2. mdg/adv_attack/attack_results.csv                              (appended)
  3. mdg/adv_attack/legacy/**/*.csv and legacy summary JSONs        (older runs)
Training metrics come from mdg/finetune/train_results.csv when present, and
from each checkpoint's eval_results.json otherwise.

Older runs predate the F1 columns, so those cells are blank rather than zero —
the `source` column says where each row came from.

Usage:
  python -m mdg.scripts.consolidate_results
  python -m mdg.scripts.consolidate_results --extra /path/to/other_machine/results
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

ROOT        = Path("mdg")
RESULTS_DIR = ROOT / "adv_attack" / "results"
ATTACK_CSV  = ROOT / "adv_attack" / "attack_results.csv"
LEGACY_DIR  = ROOT / "adv_attack" / "legacy"
CKPT_DIR    = ROOT / "finetune" / "checkpoints"
TRAIN_CSV   = ROOT / "finetune" / "train_results.csv"
OUT_DIR     = ROOT / "results"

# (model, dataset, recipe) -> row
Key = Tuple[str, str, str]

FIELDS = [
    "model", "dataset", "recipe",
    "train_micro_f1", "train_macro_f1", "train_accuracy",
    "num_examples",
    "clean_micro_f1", "clean_macro_f1",
    "attacked_micro_f1", "attacked_macro_f1",
    "micro_f1_drop", "macro_f1_drop",
    "n_successful", "n_failed", "n_skipped",
    "attack_success_rate", "avg_queries",
    "perturbed_file", "source", "timestamp",
]


def norm_model(name: str) -> str:
    """bert-base-uncased from meta-llama/bert-base-uncased, etc."""
    return (name or "").strip().split("/")[-1]


def norm_dataset(name: str) -> str:
    """counterbench from namesarnav_counterbench__test."""
    d = (name or "").strip()
    d = re.sub(r"^namesarnav[_/]", "", d)
    d = re.sub(r"__(train|test|validation)$", "", d)
    return d


def _f(v) -> str:
    try:
        return str(round(float(v), 4))
    except (TypeError, ValueError):
        return ""


def _drop(clean, attacked) -> str:
    try:
        return str(round(float(clean) - float(attacked), 4))
    except (TypeError, ValueError):
        return ""


def blank_row(model: str, dataset: str, recipe: str) -> Dict[str, str]:
    return {f: "" for f in FIELDS} | {"model": model, "dataset": dataset, "recipe": recipe}


SOURCE_RANK = {"legacy_csv": 0, "legacy_summary": 1, "attack_csv": 2, "summary_json": 3}


def quality(row: Dict[str, str]) -> tuple:
    """Break duplicates: F1 beats no F1, then a row pointing at perturbed data
    on disk, then more examples, then the more canonical source."""
    has_f1 = 1 if row.get("attacked_micro_f1") else 0
    has_pt = 1 if row.get("perturbed_file") else 0
    try:
        n = int(float(row.get("num_examples") or 0))
    except ValueError:
        n = 0
    return (has_f1, has_pt, n, SOURCE_RANK.get(row.get("source", ""), 0))


def add(rows: Dict[Key, Dict[str, str]], row: Dict[str, str]) -> None:
    try:
        if int(float(row.get("num_examples") or 0)) <= 0:
            return                      # aborted run, nothing attacked
    except ValueError:
        return
    key = (row["model"], row["dataset"], row["recipe"])
    prev = rows.get(key)
    if prev is None:
        rows[key] = row
        return
    if quality(row) > quality(prev):
        # Keep anything the better row happens to be missing (e.g. timestamp,
        # which only the CSV sources carry).
        for f, v in prev.items():
            if v and not row.get(f):
                row[f] = v
        rows[key] = row
    else:
        for f, v in row.items():
            if v and not prev.get(f):
                prev[f] = v


def from_summary_json(path: Path, source: str) -> Optional[Dict[str, str]]:
    try:
        d = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(d, dict) or "recipe" not in d:
        return None
    # .../results/<split>/<model>/<recipe>_summary.json
    model   = norm_model(path.parent.name)
    dataset = norm_dataset(path.parent.parent.name)
    row = blank_row(model, dataset, d["recipe"])
    jsonl = path.parent / f"{d['recipe']}.jsonl"
    row.update({
        "num_examples":        str(d.get("num_examples", "")),
        "clean_micro_f1":      _f(d.get("clean_micro_f1")),
        "clean_macro_f1":      _f(d.get("clean_macro_f1")),
        "attacked_micro_f1":   _f(d.get("attacked_micro_f1")),
        "attacked_macro_f1":   _f(d.get("attacked_macro_f1")),
        "micro_f1_drop":       _drop(d.get("clean_micro_f1"), d.get("attacked_micro_f1")),
        "macro_f1_drop":       _drop(d.get("clean_macro_f1"), d.get("attacked_macro_f1")),
        "n_successful":        str(d.get("n_successful", "")),
        "n_failed":            str(d.get("n_failed", "")),
        "n_skipped":           str(d.get("n_skipped", "")),
        "attack_success_rate": _f(d.get("attack_success_rate")),
        "avg_queries":         _f(d.get("avg_queries")),
        "perturbed_file":      str(jsonl) if jsonl.exists() and jsonl.stat().st_size else "",
        "source":              source,
    })
    return row


def from_csv(path: Path, source: str) -> Iterable[Dict[str, str]]:
    try:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if not r.get("recipe"):
                    continue
                row = blank_row(norm_model(r.get("model", "")),
                                norm_dataset(r.get("dataset", "")),
                                r["recipe"])
                row.update({
                    "num_examples":        r.get("num_examples", ""),
                    "clean_micro_f1":      _f(r.get("clean_micro_f1")),
                    "clean_macro_f1":      _f(r.get("clean_macro_f1")),
                    "attacked_micro_f1":   _f(r.get("attacked_micro_f1")),
                    "attacked_macro_f1":   _f(r.get("attacked_macro_f1")),
                    "micro_f1_drop":       _drop(r.get("clean_micro_f1"), r.get("attacked_micro_f1")),
                    "macro_f1_drop":       _drop(r.get("clean_macro_f1"), r.get("attacked_macro_f1")),
                    "n_successful":        r.get("n_successful", ""),
                    "n_failed":            r.get("n_failed", ""),
                    "n_skipped":           r.get("n_skipped", ""),
                    "attack_success_rate": _f(r.get("attack_success_rate")),
                    "avg_queries":         _f(r.get("avg_queries")),
                    "timestamp":           r.get("timestamp", ""),
                    "source":              source,
                })
                yield row
    except OSError:
        return


def collect_train() -> Dict[Tuple[str, str], Dict[str, str]]:
    """(model, dataset) -> train metrics, from the CSV and from checkpoints."""
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    for ck in CKPT_DIR.glob("*/*/eval_results.json"):
        try:
            d = json.loads(ck.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        out[(norm_model(ck.parent.name), norm_dataset(ck.parent.parent.name))] = {
            "train_macro_f1": _f(d.get("eval_macro_f1")),
            "train_micro_f1": _f(d.get("eval_micro_f1")),
            "train_accuracy": _f(d.get("eval_accuracy")),
        }
    if TRAIN_CSV.exists():                      # newer, wins over checkpoints
        with open(TRAIN_CSV, newline="") as f:
            for r in csv.DictReader(f):
                out[(norm_model(r.get("model", "")), norm_dataset(r.get("dataset", "")))] = {
                    "train_macro_f1": _f(r.get("macro_f1")),
                    "train_micro_f1": _f(r.get("micro_f1")),
                    "train_accuracy": _f(r.get("accuracy")),
                }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[],
                    help="Extra results directories to fold in (e.g. copied from other machines)")
    args = ap.parse_args()

    rows: Dict[Key, Dict[str, str]] = {}

    # Lowest priority first — better sources overwrite via quality().
    for csv_path in sorted(LEGACY_DIR.glob("**/*.csv")):
        for row in from_csv(csv_path, "legacy_csv"):
            add(rows, row)
    for js in sorted(LEGACY_DIR.glob("**/*_summary.json")):
        row = from_summary_json(js, "legacy_summary")
        if row:
            add(rows, row)
    if ATTACK_CSV.exists():
        for row in from_csv(ATTACK_CSV, "attack_csv"):
            add(rows, row)
    for d in [RESULTS_DIR, *map(Path, args.extra)]:
        for js in sorted(d.glob("**/*_summary.json")):
            row = from_summary_json(js, "summary_json")
            if row:
                add(rows, row)

    train = collect_train()
    for (model, dataset, _), row in rows.items():
        row.update(train.get((model, dataset), {}))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "consolidated.csv"
    ordered = sorted(rows.values(), key=lambda r: (r["dataset"], r["model"], r["recipe"]))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(ordered)

    print(f"Wrote {len(ordered)} rows → {out_path}")
    by_source: Dict[str, int] = {}
    for r in ordered:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    for s, n in sorted(by_source.items()):
        print(f"  {n:4d} from {s}")
    with_f1  = sum(1 for r in ordered if r["attacked_micro_f1"])
    with_pt  = sum(1 for r in ordered if r["perturbed_file"])
    print(f"  {with_f1} rows have attack F1, {with_pt} have perturbed jsonl on disk")
    print(f"  models   : {sorted({r['model']   for r in ordered})}")
    print(f"  datasets : {sorted({r['dataset'] for r in ordered})}")
    print(f"  recipes  : {len({r['recipe'] for r in ordered})} distinct")


if __name__ == "__main__":
    main()
