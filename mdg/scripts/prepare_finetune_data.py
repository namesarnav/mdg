"""
Export HuggingFace causal dataset splits to JSONL files with normalized
text + label fields, ready for fine-tuning.

Uses the existing CausalDatasetLoaderHF to apply FIELD_MAPS so the output
is always {"text": "...", "label": "YES/NO/..."} regardless of the source
dataset's column names.

Usage:
    poetry run python -m mdg.scripts.prepare_finetune_data \
        --datasets namesarnav/counterbench namesarnav/ac-reason namesarnav/bbh-causal-judgement \
        --output-dir mdg/finetune/data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mdg.dataset_loaders.causal import _BaseCausalLoaderHF


def _label_dist(records: list) -> dict:
    d: dict = {}
    for r in records:
        d[r["label"]] = d.get(r["label"], 0) + 1
    return d


def export_dataset(dataset_name: str, output_dir: Path, train_ratio: float = 0.75, seed: int = 42) -> None:
    import random
    hf_name = dataset_name.replace("/", "_").replace(":", "_")
    print(f"\n{'='*55}")
    print(f"  {dataset_name}")

    hf_loc, hf_cfg = (dataset_name.rsplit(":", 1) if ":" in dataset_name else (dataset_name, None))

    loader = _BaseCausalLoaderHF(config={
        "hf_location": hf_loc,
        **({"hf_config": hf_cfg} if hf_cfg else {}),
    })
    splits = loader.run()

    # Collect non-empty splits
    present = {k: v for k, v in splits.items() if v}

    has_train = bool(present.get("train"))
    has_test  = bool(present.get("test"))

    if has_train and has_test:
        # Dataset already has proper splits — use them as-is
        to_save = {"train": present["train"], "test": present["test"]}
        if present.get("validation"):
            to_save["validation"] = present["validation"]
    else:
        # Single-split dataset — merge everything and split 75/25
        all_records = []
        for recs in present.values():
            all_records.extend(recs)

        # Stratified split by label
        from collections import defaultdict
        by_label: dict = defaultdict(list)
        for r in all_records:
            by_label[r["label"]].append(r)

        train_records, test_records = [], []
        rng = random.Random(seed)
        for label, recs in by_label.items():
            rng.shuffle(recs)
            cut = max(1, int(len(recs) * train_ratio))
            train_records.extend(recs[:cut])
            test_records.extend(recs[cut:])

        rng.shuffle(train_records)
        rng.shuffle(test_records)
        to_save = {"train": train_records, "test": test_records}
        print(f"  [INFO] No train/test splits found — created 75/25 stratified split")

    for split_name, records in to_save.items():
        out_path = output_dir / f"{hf_name}__{split_name}.jsonl"
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps({
                    "text":  r["text"],
                    "label": r["label"],
                }, ensure_ascii=False) + "\n")
        print(f"  [{split_name}]  {len(records)} records  labels={_label_dist(records)}  → {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="HuggingFace dataset names")
    parser.add_argument("--output-dir", default="mdg/finetune/data")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        export_dataset(ds, out)

    print("\nDone.")


if __name__ == "__main__":
    main()
