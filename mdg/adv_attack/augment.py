"""
Apply every TextAttack augmentation recipe to a causal dataset JSONL file.

For each dataset, every recipe is applied to the text field(s), and the
perturbed version is saved as a separate JSONL file in the same folder.
All recipes are run on a dataset before moving to the next one.

Usage:
    poetry run python -m mdg.adv_attack.textattack \
        --datasets namesarnav/counterbench namesarnav/ac-reason \
        --input-dir mdg/synthetic/data \
        --output-dir mdg/adv_attack/data

    # Or run on already-downloaded JSONL files directly:
    poetry run python -m mdg.adv_attack.textattack \
        --files mdg/synthetic/data/counterbench_task1.jsonl \
                mdg/synthetic/data/ac_reason_task1.jsonl \
        --output-dir mdg/adv_attack/data

Recipes applied (in order):
    01_easy_data          EasyDataAugmenter      (WordNet + deletion + swap + insertion)
    02_wordnet            WordNetAugmenter       (synonym substitution via WordNet)
    03_embedding          EmbeddingAugmenter     (embedding-based synonym substitution)
    04_char_swap          CharSwapAugmenter      (character-level noise)
    05_deletion           DeletionAugmenter      (random word deletion)
    06_swap               SwapAugmenter          (random word position swap)
    07_synonym_insert     SynonymInsertionAugmenter (random synonym insertion)
    08_checklist          CheckListAugmenter     (name/location/number replacements)
    09_clare              CLAREAugmenter         (MLM-based replace/insert/merge)
    10_back_translation   BackTranslationAugmenter (5-step back-translation chain)
    11_back_transcription BackTranscriptionAugmenter (TTS → STT roundtrip)

Notes:
    - BackTranslationAugmenter and BackTranscriptionAugmenter require network
      access and additional models; they are skipped gracefully if unavailable.
    - CLAREAugmenter downloads distilroberta-base on first run (~330MB).
    - The label field is never modified — only text fields are perturbed.
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# Per-dataset text field mapping
# key   : dataset name (or stem of the JSONL filename)
# value : list of field names that contain text to perturb

TEXT_FIELDS: Dict[str, List[str]] = {
    "counterbench":          ["given_info", "question"],
    "ac-reason":             ["story", "question"],
    "ac_reason":             ["story", "question"],
    "bbh-causal-judgement":  ["input"],
    "bbh_causal_judgement":  ["input"],
    "bbh-causal-understanding": ["input"],
    "bbh_causal_understanding": ["input"],
    "corr2cause":            ["input"],
    "e-care":                ["premise", "choice1", "choice2"],
    "e_care":                ["premise", "choice1", "choice2"],
    "fincausal-task1":       ["text"],
    "fincausal_task1":       ["text"],
    "causalbench":           ["Scenario and Question"],   
}

DEFAULT_TEXT_FIELDS = ["text"]   


def _detect_text_fields(stem: str) -> List[str]:
    """Infer text fields from the JSONL filename stem."""
    stem_lower = stem.lower()
    for key, fields in TEXT_FIELDS.items():
        if key in stem_lower:
            return fields
    return DEFAULT_TEXT_FIELDS

def _recipes() -> List[Tuple[str, Callable]]:
    from textattack.augmentation import (
        EasyDataAugmenter,
        WordNetAugmenter,
        EmbeddingAugmenter,
        CharSwapAugmenter,
        DeletionAugmenter,
        SwapAugmenter,
        SynonymInsertionAugmenter,
        CheckListAugmenter,
        CLAREAugmenter,
        BackTranslationAugmenter,
        BackTranscriptionAugmenter,
    )

    return [
        ("01_easy_data",          lambda: EasyDataAugmenter(pct_words_to_swap=0.1, transformations_per_example=1)),
        ("02_wordnet",            lambda: WordNetAugmenter(pct_words_to_swap=0.2, transformations_per_example=1)),
        ("03_embedding",          lambda: EmbeddingAugmenter(pct_words_to_swap=0.2, transformations_per_example=1)),
        ("04_char_swap",          lambda: CharSwapAugmenter(pct_words_to_swap=0.2, transformations_per_example=1)),
        ("05_deletion",           lambda: DeletionAugmenter(pct_words_to_swap=0.1, transformations_per_example=1)),
        ("06_swap",               lambda: SwapAugmenter(pct_words_to_swap=0.2, transformations_per_example=1)),
        ("07_synonym_insert",     lambda: SynonymInsertionAugmenter(pct_words_to_swap=0.2, transformations_per_example=1)),
        ("08_checklist",          lambda: CheckListAugmenter(pct_words_to_swap=0.2, transformations_per_example=1)),
        ("09_clare",              lambda: CLAREAugmenter(pct_words_to_swap=0.2, transformations_per_example=1)),
        ("10_back_translation",   lambda: BackTranslationAugmenter(transformations_per_example=1)),
        ("11_back_transcription", lambda: BackTranscriptionAugmenter(transformations_per_example=1)),
    ]


# -- Core augmentation logic -- 

def _augment_text(augmenter, text: str) -> str:
    """Run augmenter on a single text string; return original on failure."""
    try:
        results = augmenter.augment(text)
        if results:
            return results[0]
    except Exception:
        pass
    return text


def _augment_record(augmenter, record: dict, text_fields: List[str]) -> dict:
    """Return a copy of the record with text fields perturbed."""
    out = dict(record)
    for field in text_fields:
        if field in out and isinstance(out[field], str):
            out[field] = _augment_text(augmenter, out[field])
    return out


def apply_recipe(
    records: List[dict],
    recipe_name: str,
    factory: Callable,
    text_fields: List[str],
    output_path: Path,
) -> bool:
    """
    Instantiate the augmenter, apply to all records, save JSONL.
    Returns True on success, False if the recipe was skipped.
    """
    print(f"    [{recipe_name}] initialising...", end=" ", flush=True)
    try:
        augmenter = factory()
    except Exception as e:
        print(f"SKIP (init failed: {e})")
        return False

    print(f"augmenting {len(records)} records...", end=" ", flush=True)
    try:
        perturbed = [_augment_record(augmenter, r, text_fields) for r in records]
    except Exception as e:
        print(f"SKIP (augment failed: {e})")
        traceback.print_exc()
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in perturbed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"saved → {output_path.name}")
    return True


# Per-file pipeline

def process_file(input_path: Path, output_dir: Path) -> None:
    stem = input_path.stem            # e.g. "counterbench_task1"
    text_fields = _detect_text_fields(stem)

    print(f"\n{'='*60}")
    print(f"  File  : {input_path.name}")
    print(f"  Fields: {text_fields}")
    print(f"{'='*60}")

    with open(input_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"  Loaded {len(records)} records\n")

    file_out_dir = output_dir / stem
    file_out_dir.mkdir(parents=True, exist_ok=True)

    recipes = _recipes()
    succeeded, skipped = 0, 0
    for recipe_name, factory in recipes:
        out_path = file_out_dir / f"{stem}__{recipe_name}.jsonl"
        ok = apply_recipe(records, recipe_name, factory, text_fields, out_path)
        if ok:
            succeeded += 1
        else:
            skipped += 1

    print(f"\n  Done: {succeeded} saved, {skipped} skipped  →  {file_out_dir}")


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply all TextAttack augmentation recipes to causal dataset JSONL files."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--files", nargs="+", metavar="PATH",
        help="One or more JSONL file paths to augment directly.",
    )
    group.add_argument(
        "--input-dir", metavar="DIR",
        help="Directory containing JSONL files (all *.jsonl files are processed).",
    )
    parser.add_argument(
        "--output-dir", default="mdg/adv_attack/data",
        help="Root output directory (default: mdg/adv_attack/data). "
             "Each input file gets its own sub-folder.",
    )
    parser.add_argument(
        "--glob", default="*.jsonl",
        help="Glob pattern when using --input-dir (default: *.jsonl)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.files:
        paths = [Path(p) for p in args.files]
    else:
        paths = sorted(Path(args.input_dir).glob(args.glob))
        if not paths:
            print(f"No files matching {args.glob} in {args.input_dir}")
            return

    print(f"Processing {len(paths)} file(s) with 11 TextAttack recipes each.")
    for path in paths:
        process_file(path, output_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
