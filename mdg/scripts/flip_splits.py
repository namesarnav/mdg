"""
Merge train+test, re-split 20% train / 80% test, push to HuggingFace Hub.

Usage:
    poetry run python -m mdg.scripts.flip_splits
"""
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split

DATASETS = [
    "namesarnav/counterbench",
    "namesarnav/ac-reason",
    "namesarnav/bbh-causal-judgement",
]

TRAIN_RATIO = 0.20
SEED = 42


def flip(dataset_name: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {dataset_name}")
    dd = load_dataset(dataset_name)

    # Combine all splits
    all_splits = list(dd.values())
    from datasets import concatenate_datasets
    full = concatenate_datasets(all_splits)
    print(f"  Total rows: {len(full)}")

    # Stratify by label if possible
    label_col = None
    for col in ("answer", "label", "target", "gold", "Ground Truth"):
        if col in full.column_names:
            label_col = col
            break

    indices = list(range(len(full)))
    if label_col:
        labels = full[label_col]
        train_idx, test_idx = train_test_split(
            indices, train_size=TRAIN_RATIO, stratify=labels, random_state=SEED
        )
    else:
        train_idx, test_idx = train_test_split(
            indices, train_size=TRAIN_RATIO, random_state=SEED
        )

    train_ds = full.select(train_idx)
    test_ds  = full.select(test_idx)

    print(f"  New train: {len(train_ds)}  test: {len(test_ds)}")

    new_dd = DatasetDict({"train": train_ds, "test": test_ds})
    new_dd.push_to_hub(dataset_name)
    print(f"  Pushed → {dataset_name}")


if __name__ == "__main__":
    for ds in DATASETS:
        flip(ds)
    print("\nDone.")
