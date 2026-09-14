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


def export_dataset(dataset_name: str, output_dir: Path) -> None:
    hf_name = dataset_name.replace("/", "_").replace(":", "_")
    print(f"\n{'='*55}")
    print(f"  {dataset_name}")

    hf_loc, hf_cfg = (dataset_name.rsplit(":", 1) if ":" in dataset_name else (dataset_name, None))

    loader = _BaseCausalLoaderHF(config={
        "hf_location": hf_loc,
        **({"hf_config": hf_cfg} if hf_cfg else {}),
    })
    splits = loader.run()

    for split_name, records in splits.items():
        if not records:
            continue
        out_path = output_dir / f"{hf_name}__{split_name}.jsonl"
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps({
                    "text":  r["text"],
                    "label": r["label"],
                }, ensure_ascii=False) + "\n")
        label_dist: dict = {}
        for r in records:
            label_dist[r["label"]] = label_dist.get(r["label"], 0) + 1
        print(f"  [{split_name}]  {len(records)} records  labels={label_dist}  → {out_path.name}")


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
