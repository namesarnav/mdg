"""
Adversarial attack on a fine-tuned causal classification model using every
TextAttack attack recipe.

For each recipe, the attack is run against the provided dataset and results
are saved to the output directory before the next recipe starts.

Usage:
    poetry run python -m mdg.adv_attack.attack \
        --model  mdg/finetune/checkpoints/bert-base-uncased \
        --dataset mdg/synthetic/data/counterbench_task1.jsonl \
        --output-dir mdg/adv_attack/results \
        --num-examples 200

    # Attack with a specific subset of recipes:
    poetry run python -m mdg.adv_attack.attack \
        --model  mdg/finetune/checkpoints/roberta-base \
        --dataset mdg/synthetic/data/ac_reason_task1.jsonl \
        --recipes TextFoolerJin2019 BERTAttackLi2020 \
        --output-dir mdg/adv_attack/results

Output per recipe (inside --output-dir/<dataset_stem>/<model_name>/):
    <recipe>.jsonl        — each attacked example with original/perturbed text,
                            original/perturbed label, and attack result
    <recipe>_summary.json — aggregate stats (attack success rate, avg queries, etc.)
    all_summaries.json    — combined summary across all recipes run

Notes:
    - Requires a fine-tuned HuggingFace model with a classification head.
    - Some recipes are slow (GeneticAlgorithm, PSO) — use --num-examples to limit.
    - Recipes that need specific model types (Seq2Sick, HotFlip) are skipped
      gracefully for standard classifiers.
    - Language-specific recipes (French, Spanish, Chinese) are excluded by default;
      pass --include-multilingual to enable them.
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


# All English attack recipes, ordered roughly fastest → slowest

RECIPES: List[Tuple[str, str]] = [
    # (short_name,               textattack_class)
    ("Pruthi2019",               "Pruthi2019"),               # character-level typos
    ("DeepWordBugGao2018",       "DeepWordBugGao2018"),        # character-level black-box
    ("BadCharacters2021",        "BadCharacters2021"),         # invisible unicode chars
    ("TextBuggerLi2018",         "TextBuggerLi2018"),          # char+word hybrid
    ("PWWSRen2019",              "PWWSRen2019"),               # word saliency + WordNet
    ("TextFoolerJin2019",        "TextFoolerJin2019"),         # embedding similarity
    ("A2TYoo2021",               "A2TYoo2021"),                # token-aware TextFooler
    ("BAEGarg2019",              "BAEGarg2019"),               # BERT masked LM
    ("BERTAttackLi2020",         "BERTAttackLi2020"),          # BERT sub-word attack
    ("CLARE2020",                "CLARE2020"),                 # context-aware MLM
    ("CheckList2020",            "CheckList2020"),             # behavioural testing
    ("IGAWang2019",              "IGAWang2019"),               # improved genetic
    ("PSOZang2020",              "PSOZang2020"),               # particle swarm
    ("MorpheusTan2020",          "MorpheusTan2020"),           # inflectional morphology
    ("Kuleshov2017",             "Kuleshov2017"),              # language model scoring
    ("GeneticAlgorithmAlzantot2018", "GeneticAlgorithmAlzantot2018"),  # genetic (slow)
    ("FasterGeneticAlgorithmJia2019","FasterGeneticAlgorithmJia2019"), # faster genetic
    ("InputReductionFeng2018",   "InputReductionFeng2018"),    # input reduction (targeted)
    # Seq2SickCheng2018BlackBox and HotFlipEbrahimi2017 require seq2seq / gradient
    # access — skipped here; add manually if your model supports them.
]

MULTILINGUAL_RECIPES: List[Tuple[str, str]] = [
    ("FrenchRecipe",   "FrenchRecipe"),
    ("SpanishRecipe",  "SpanishRecipe"),
    ("ChineseRecipe",  "ChineseRecipe"),
]


# Dataset loading


TEXT_FIELDS: Dict[str, List[str]] = {
    "counterbench":         ["given_info", "question"],
    "ac-reason":            ["story", "question"],
    "ac_reason":            ["story", "question"],
    "bbh-causal-judgement": ["input"],
    "bbh_causal_judgement": ["input"],
    "corr2cause":           ["input"],
    "e-care":               ["premise", "choice1", "choice2"],
    "e_care":               ["premise", "choice1", "choice2"],
    "fincausal-task1":      ["text"],
    "fincausal_task1":      ["text"],
    "causalbench":          ["Scenario and Question"],
}


def _detect_text_field(stem: str) -> str:
    """Return the primary text field for a dataset stem (first one)."""
    stem_lower = stem.lower()
    for key, fields in TEXT_FIELDS.items():
        if key in stem_lower:
            return fields[0]
    return "text"


def _build_text(record: dict, text_fields: List[str]) -> str:
    """Concatenate all text fields into a single string for the model."""
    return " ".join(str(record[f]) for f in text_fields if f in record)


def load_textattack_dataset(
    jsonl_path: Path,
    label_field: str,
    text_fields: List[str],
    label2id: Dict[str, int],
    max_examples: Optional[int],
) -> List[Tuple[str, int]]:
    """Load JSONL → list of (text_str, label_id) for TextAttack."""
    with open(jsonl_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    if max_examples:
        records = records[:max_examples]

    data = []
    for r in records:
        text = _build_text(r, text_fields)
        raw_label = str(r.get(label_field, "")).strip().upper()
        label_id = label2id.get(raw_label)
        if label_id is None:
            continue
        data.append((text, label_id))
    return data



# Model wrapper

def build_model_wrapper(model_path: str):
    """Wrap a fine-tuned HuggingFace sequence classifier for TextAttack."""
    from textattack.models.wrappers import HuggingFaceModelWrapper
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"  Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    return HuggingFaceModelWrapper(model, tokenizer)


# Single recipe attack

def _import_recipe(class_name: str):
    import textattack.attack_recipes as ar
    cls = getattr(ar, class_name, None)
    if cls is None:
        raise ImportError(f"Recipe {class_name!r} not found in textattack.attack_recipes")
    return cls


def run_recipe(
    recipe_name: str,
    recipe_class_name: str,
    model_wrapper,
    dataset,           # textattack.datasets.Dataset
    num_examples: int,
    output_dir: Path,
    query_budget: Optional[int],
) -> Optional[Dict]:
    """
    Run one attack recipe. Saves per-example JSONL and a summary JSON.
    Returns the summary dict, or None if skipped.
    """
    import textattack
    from textattack import Attacker, AttackArgs

    print(f"\n  [{recipe_name}]", end=" ", flush=True)

    try:
        recipe_cls = _import_recipe(recipe_class_name)
        attack = recipe_cls.build(model_wrapper)
    except Exception as e:
        print(f"SKIP (build failed: {e})")
        return None

    attack_args = AttackArgs(
        num_examples=num_examples,
        query_budget=query_budget,
        disable_stdout=True,
        silent=True,
    )

    try:
        attacker = Attacker(attack, dataset, attack_args)
        results = attacker.attack_dataset()
    except Exception as e:
        print(f"SKIP (attack failed: {e})")
        traceback.print_exc()
        return None

    # Stream results — write each example to disk immediately so nothing is
    # lost if the process is interrupted mid-attack.
    jsonl_path = output_dir / f"{recipe_name}.jsonl"
    n_successful = 0
    n_failed = 0
    n_skipped = 0
    total_queries = 0
    n_total = 0

    print(f"streaming → {jsonl_path.name} ", end="", flush=True)

    with open(jsonl_path, "w") as fout:
        for result in results:
            rtype = type(result).__name__
            entry = {
                "result_type":   rtype,
                "original_text": result.original_result.attacked_text.text,
                "original_label": result.original_result.output,
                "perturbed_text": (
                    result.perturbed_result.attacked_text.text
                    if hasattr(result, "perturbed_result") else None
                ),
                "perturbed_label": (
                    result.perturbed_result.output
                    if hasattr(result, "perturbed_result") else None
                ),
                "num_queries": result.num_queries if hasattr(result, "num_queries") else None,
            }

            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fout.flush()   # force OS write so disk has it immediately

            n_total += 1
            if "Successful" in rtype:
                n_successful += 1
                print("S", end="", flush=True)
            elif "Failed" in rtype:
                n_failed += 1
                print(".", end="", flush=True)
            else:
                n_skipped += 1
                print("_", end="", flush=True)

            if hasattr(result, "num_queries") and result.num_queries:
                total_queries += result.num_queries

    print()  # newline after progress dots

    attacked = n_successful + n_failed
    asr = n_successful / attacked if attacked > 0 else 0.0
    avg_queries = total_queries / n_total if n_total else 0.0

    summary = {
        "recipe":              recipe_name,
        "num_examples":        n_total,
        "n_successful":        n_successful,
        "n_failed":            n_failed,
        "n_skipped":           n_skipped,
        "attack_success_rate": round(asr, 4),
        "avg_queries":         round(avg_queries, 2),
    }

    summary_path = output_dir / f"{recipe_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"ASR={asr:.1%}  "
        f"(success={n_successful} failed={n_failed} skipped={n_skipped})  "
        f"avg_queries={avg_queries:.0f}  → {jsonl_path.name}"
    )
    return summary


# Main pipeline

def run(
    model_path: str,
    dataset_path: Path,
    output_dir: Path,
    label_field: str,
    label_space: List[str],
    text_field_override: Optional[List[str]],
    num_examples: int,
    query_budget: Optional[int],
    recipes_filter: Optional[List[str]],
    include_multilingual: bool,
) -> None:
    import textattack

    stem = dataset_path.stem
    model_name = Path(model_path).name

    text_fields_primary = text_field_override or [_detect_text_field(stem)]
    # For TEXT_FIELDS that have multiple fields, join them all
    stem_lower = stem.lower()
    all_text_fields = text_field_override
    if all_text_fields is None:
        for key, fields in TEXT_FIELDS.items():
            if key in stem_lower:
                all_text_fields = fields
                break
        if all_text_fields is None:
            all_text_fields = ["text"]

    label_space_upper = [l.strip().upper() for l in label_space]
    label2id = {l: i for i, l in enumerate(label_space_upper)}
    id2label = {i: l for l, i in label2id.items()}

    print(f"\n{'='*60}")
    print(f"  Model  : {model_path}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Labels : {label_space_upper}")
    print(f"  Fields : {all_text_fields}")
    print(f"{'='*60}")

    #  Load data 
    raw_data = load_textattack_dataset(
        dataset_path, label_field, all_text_fields, label2id, num_examples
    )
    print(f"  Loaded {len(raw_data)} examples for attack\n")

    ta_dataset = textattack.datasets.Dataset(raw_data, label_names=label_space_upper)

    #  Model wrapper 
    model_wrapper = build_model_wrapper(model_path)

    #  Output dir 
    run_dir = output_dir / stem / model_name
    run_dir.mkdir(parents=True, exist_ok=True)

    #  Recipes to run 
    all_recipes = list(RECIPES)
    if include_multilingual:
        all_recipes += MULTILINGUAL_RECIPES
    if recipes_filter:
        all_recipes = [(n, c) for n, c in all_recipes if n in recipes_filter or c in recipes_filter]

    print(f"  Running {len(all_recipes)} recipe(s)...\n")

    all_summaries = []
    for recipe_name, recipe_class in all_recipes:
        summary = run_recipe(
            recipe_name=recipe_name,
            recipe_class_name=recipe_class,
            model_wrapper=model_wrapper,
            dataset=ta_dataset,
            num_examples=len(raw_data),
            output_dir=run_dir,
            query_budget=query_budget,
        )
        if summary:
            all_summaries.append(summary)

    #  Combined summary 
    combined_path = run_dir / "all_summaries.json"
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Completed {len(all_summaries)}/{len(all_recipes)} recipes")
    print(f"  Results → {run_dir}")
    if all_summaries:
        avg_asr = sum(s["attack_success_rate"] for s in all_summaries) / len(all_summaries)
        best = max(all_summaries, key=lambda s: s["attack_success_rate"])
        print(f"  Avg ASR : {avg_asr:.1%}")
        print(f"  Best    : {best['recipe']}  (ASR={best['attack_success_rate']:.1%})")
    print(f"{'='*60}")


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attack a fine-tuned classifier with every TextAttack recipe."
    )
    parser.add_argument("--model", required=True,
                        help="Path to a fine-tuned HuggingFace classifier checkpoint, "
                             "e.g. mdg/finetune/checkpoints/bert-base-uncased")
    parser.add_argument("--dataset", required=True,
                        help="Path to the JSONL dataset to attack")
    parser.add_argument("--output-dir", default="mdg/adv_attack/results",
                        help="Root output directory (default: mdg/adv_attack/results)")
    parser.add_argument("--label-field", default=None,
                        help="Label column name in the JSONL (auto-detected if omitted)")
    parser.add_argument("--label-space", nargs="+", default=["YES", "NO"],
                        help="Valid label strings (default: YES NO)")
    parser.add_argument("--text-fields", nargs="+", default=None,
                        help="Text field name(s) to attack (auto-detected from filename if omitted)")
    parser.add_argument("--num-examples", type=int, default=200,
                        help="Max examples to attack per recipe (default: 200)")
    parser.add_argument("--query-budget", type=int, default=None,
                        help="Max model queries per example (default: unlimited)")
    parser.add_argument("--recipes", nargs="+", default=None,
                        help="Run only these recipes (by name). Default: all.")
    parser.add_argument("--include-multilingual", action="store_true",
                        help="Also run French/Spanish/Chinese recipes")
    args = parser.parse_args()

    # Auto-detect label field from dataset filename if not provided
    label_field = args.label_field
    if label_field is None:
        stem = Path(args.dataset).stem.lower()
        if "counterbench" in stem or "ac-reason" in stem or "ac_reason" in stem:
            label_field = "answer"
        elif "bbh" in stem:
            label_field = "target"
        elif "fincausal" in stem:
            label_field = "gold"
        elif "causalbench" in stem:
            label_field = "Ground Truth"
        else:
            label_field = "label"
        print(f"  Auto-detected label field: {label_field!r}")

    run(
        model_path=args.model,
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        label_field=label_field,
        label_space=args.label_space,
        text_field_override=args.text_fields,
        num_examples=args.num_examples,
        query_budget=args.query_budget,
        recipes_filter=args.recipes,
        include_multilingual=args.include_multilingual,
    )


if __name__ == "__main__":
    main()
