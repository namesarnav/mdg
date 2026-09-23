"""
Merge all per-run CSVs into one consolidated results file.

Reads:
  mdg/finetune/train_results.csv   — F1/accuracy per model × dataset
  mdg/adv_attack/attack_results.csv — ASR per model × dataset × recipe

Writes:
  mdg/results/train_results.csv     (copy, cleaned)
  mdg/results/attack_results.csv    (copy, cleaned)
  mdg/results/consolidated.csv      — joined view: for each attack row, adds
                                       the corresponding training F1/accuracy

Usage:
  poetry run python -m mdg.scripts.consolidate_results
"""
import csv
import sys
from pathlib import Path

TRAIN_CSV   = Path("mdg/finetune/train_results.csv")
ATTACK_CSV  = Path("mdg/adv_attack/attack_results.csv")
OUT_DIR     = Path("mdg/results")


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[WARN] Missing: {path}", file=sys.stderr)
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"[SKIP] No rows for {path}", file=sys.stderr)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows → {path}")


def main() -> None:
    train_rows  = read_csv(TRAIN_CSV)
    attack_rows = read_csv(ATTACK_CSV)

    # Copy cleaned versions
    write_csv(OUT_DIR / "train_results.csv",  train_rows)
    write_csv(OUT_DIR / "attack_results.csv", attack_rows)

    # Build lookup: (model, dataset) → train metrics
    train_lookup: dict[tuple, dict] = {}
    for row in train_rows:
        key = (row.get("model", "").strip(), row.get("dataset", "").strip())
        train_lookup[key] = row

    # Join
    joined: list[dict] = []
    for row in attack_rows:
        key = (row.get("model", "").strip(), row.get("dataset", "").strip())
        train = train_lookup.get(key, {})
        joined_row = {
            "model":               row.get("model", ""),
            "dataset":             row.get("dataset", ""),
            "recipe":              row.get("recipe", ""),
            # Training metrics
            "train_macro_f1":      train.get("macro_f1", ""),
            "train_accuracy":      train.get("accuracy", ""),
            "num_train":           train.get("num_train", ""),
            "num_eval":            train.get("num_eval", ""),
            # Attack metrics
            "num_examples":        row.get("num_examples", ""),
            "n_successful":        row.get("n_successful", ""),
            "n_failed":            row.get("n_failed", ""),
            "n_skipped":           row.get("n_skipped", ""),
            "attack_success_rate": row.get("attack_success_rate", ""),
            "avg_queries":         row.get("avg_queries", ""),
            # Timestamps
            "train_timestamp":     train.get("timestamp", ""),
            "attack_timestamp":    row.get("timestamp", ""),
        }
        joined.append(joined_row)

    write_csv(OUT_DIR / "consolidated.csv", joined)

    # Print summary
    models   = sorted({r["model"]   for r in joined if r["model"]})
    datasets = sorted({r["dataset"] for r in joined if r["dataset"]})
    recipes  = sorted({r["recipe"]  for r in joined if r["recipe"]})
    print(f"\nConsolidated: {len(joined)} rows")
    print(f"  Models   ({len(models)}): {', '.join(models)}")
    print(f"  Datasets ({len(datasets)}): {', '.join(datasets)}")
    print(f"  Recipes  ({len(recipes)}): {len(recipes)} unique")

    if not joined:
        print("[INFO] No attack rows yet — run attack_all.sh first.")
        return

    # Quick stats
    def safe_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    asr_vals = [safe_float(r["attack_success_rate"]) for r in joined]
    asr_vals = [v for v in asr_vals if v is not None]
    if asr_vals:
        print(f"  Avg ASR  : {sum(asr_vals)/len(asr_vals):.1%}")
        print(f"  Max ASR  : {max(asr_vals):.1%}")

    f1_vals = [safe_float(r["train_macro_f1"]) for r in joined]
    f1_vals = [v for v in f1_vals if v is not None]
    if f1_vals:
        print(f"  Avg train F1: {sum(f1_vals)/len(f1_vals):.4f}")


if __name__ == "__main__":
    main()
