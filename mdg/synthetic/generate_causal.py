#!/usr/bin/env python3
"""
Synthetic causal dataset generator with feedback loop.

Output format matches the source HuggingFace dataset exactly (core content fields only).

Task 1: 2 seeds per label → model generates 1 new example per label per call.
Task 2: 2 seeds per label → model generates 5 new examples per label per call.

Usage:
    # Task 1: 36 datapoints for counterbench
    poetry run python -m mdg.synthetic.generate_causal \\
        --dataset namesarnav/counterbench --n 36 --task 1 \\
        --output mdg/synthetic/data/counterbench_task1.jsonl

    # Task 2: 60 datapoints for ac-reason
    poetry run python -m mdg.synthetic.generate_causal \\
        --dataset namesarnav/ac-reason --n 60 --task 2 \\
        --output mdg/synthetic/data/ac_reason_task2.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from datasets import load_dataset as hf_load_dataset

from mdg.dataset_loaders.causal import (
    FIELD_MAPS,
    _DEFAULT_MAP,
    _norm_label,
    _pretty_label,
    _route_splits,
)
from mdg.methods.openrouter_utils import call_openrouter_json

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DEFAULT_MODEL = "openai/gpt-5.6-sol"

# ──────────────────────────────────────────────────────────────────────────────
# Per-dataset output schemas
# Fields:      the core content fields to show in seeds and expect in output
# label_field: which field holds the answer
# text_fields: which fields form the scenario text (used for dedup check)
# fmt_example: shown in the OUTPUT FORMAT block of the prompt
# ──────────────────────────────────────────────────────────────────────────────

DATASET_SCHEMAS: Dict[str, Dict] = {
    "namesarnav/counterbench": {
        "fields":      ["given_info", "question", "answer"],
        "label_field": "answer",
        "text_fields": ["given_info", "question"],
        "fmt_example": '{"given_info": "<background context>", "question": "<causal yes/no question>", "answer": "yes/no"}',
    },
    "namesarnav/ac-reason": {
        "fields":      ["story", "question", "answer"],
        "label_field": "answer",
        "text_fields": ["story", "question"],
        "fmt_example": '{"story": "<background story>", "question": "<causal yes/no question>", "answer": "yes/no"}',
    },
    "namesarnav/bbh-causal-judgement": {
        "fields":      ["input", "target"],
        "label_field": "target",
        "text_fields": ["input"],
        "fmt_example": '{"input": "<scenario and causal question>", "target": "Yes/No"}',
    },
    "namesarnav/bbh-causal-understanding": {
        "fields":      ["input", "target"],
        "label_field": "target",
        "text_fields": ["input"],
        "fmt_example": '{"input": "<scenario and causal question>", "target": "Yes/No/Ambiguous"}',
    },
    "namesarnav/corr2cause": {
        "fields":      ["input", "label"],
        "label_field": "label",
        "text_fields": ["input"],
        "fmt_example": '{"input": "<correlation description and causal question>", "label": "0/1"}',
    },
    "namesarnav/e-care": {
        "fields":      ["premise", "question", "choice1", "choice2", "label"],
        "label_field": "label",
        "text_fields": ["premise", "choice1", "choice2"],
        "fmt_example": '{"premise": "<situation>", "question": "cause/effect", "choice1": "<option A>", "choice2": "<option B>", "label": "0/1"}',
    },
    "namesarnav/fincausal-task1": {
        "fields":      ["text", "gold"],
        "label_field": "gold",
        "text_fields": ["text"],
        "fmt_example": '{"text": "<financial sentence>", "gold": "0/1"}',
    },
    "namesarnav/causalbench:code": {
        "fields":      ["Code", "Question Type", "Question", "Ground Truth"],
        "label_field": "Ground Truth",
        "text_fields": ["Code", "Question"],
        "fmt_example": '{"Code": "<code snippet>", "Question Type": "<type>", "Question": "<causal question>", "Ground Truth": "Yes/No"}',
    },
    "namesarnav/causalbench:math": {
        "fields":      ["Mathematical Scenario", "Question Type", "Question", "Ground Truth"],
        "label_field": "Ground Truth",
        "text_fields": ["Mathematical Scenario", "Question"],
        "fmt_example": '{"Mathematical Scenario": "<math description>", "Question Type": "<type>", "Question": "<causal question>", "Ground Truth": "Yes/No"}',
    },
    "namesarnav/causalbench:text": {
        "fields":      ["Question Type", "Scenario and Question", "Ground Truth"],
        "label_field": "Ground Truth",
        "text_fields": ["Scenario and Question"],
        "fmt_example": '{"Question Type": "<type>", "Scenario and Question": "<scenario and causal question>", "Ground Truth": "Yes/No"}',
    },
}

_SYSTEM_TMPL = """\
You are a dataset generator for causal reasoning benchmarks. Multiple demonstration \
examples are provided below for each target label. Treat these as few shot demonstrations.

Instructions:
1. Identify all distinct target labels in the demonstrations.
2. {generation_instruction}
3. Preserve the same field names, field structure, label format, and writing style.
4. Do not paraphrase, copy, or make small lexical changes to the demonstrations.
5. Create genuinely new situations while preserving the causal reasoning characteristics.
{extra}
6. Do not add fields, explanations, or formatting not present in the demonstrations.
7. Use only information in each generated example to determine its correct label.
8. Return ONLY the generated examples as a JSON array. No explanation.

SOURCE DATASET: {dataset_name}
LABEL SPACE: {label_space}"""

SYSTEM_PROMPT_TASK1 = _SYSTEM_TMPL.replace(
    "{generation_instruction}",
    "Generate exactly ONE new example for each distinct target label.",
).replace("{extra}", "")

SYSTEM_PROMPT_TASK2 = _SYSTEM_TMPL.replace(
    "{generation_instruction}",
    "Generate exactly FIVE new examples for each distinct target label.",
).replace(
    "{extra}",
    "6. Ensure diversity among the five examples per label — different situations, not variations.\n",
)

_USER_TMPL = """\
Few shot examples:
{seed_list}

OUTPUT FORMAT (JSON array only, no extra text):
[
  {fmt_example}
]

Now generate the new examples:"""

_FEEDBACK_TMPL = """\
FEEDBACK FROM PREVIOUS BATCH:
{feedback_text}

Keep these issues in mind for the new examples.

Few shot examples:
{seed_list}

OUTPUT FORMAT (JSON array only, no extra text):
[
  {fmt_example}
]

Now generate the new examples:"""



# Dataset loading — raw records in original field format


def load_raw_records(dataset_name: str, schema: Dict) -> Tuple[List[dict], List[str]]:
    """
    Load records from HuggingFace keeping only core content fields.
    dataset_name may include a config suffix: "namesarnav/causalbench:text"
    Each record has the original field names plus two private keys:
      _norm_label  — normalised label for stratification/evaluation
      _raw_label   — original label string as it appears in the dataset
    """
    hf_name, hf_config = (dataset_name.rsplit(":", 1) if ":" in dataset_name else (dataset_name, None))
    fmap = FIELD_MAPS.get(dataset_name) or FIELD_MAPS.get(hf_name, _DEFAULT_MAP)
    dd = hf_load_dataset(hf_name, hf_config) if hf_config else hf_load_dataset(hf_name)
    routing = _route_splits(list(dd.keys()))

    label_field = schema["label_field"]
    fields = schema["fields"]

    # Sample seeds from train only — test must stay held out for evaluation.
    # If the dataset has no train split (single-split datasets routed entirely
    # to test), fall back to validation then test so generation still works.
    seed_slot = routing.get("train") or routing.get("validation") or routing.get("test")
    all_records: List[dict] = []
    if seed_slot and seed_slot in dd:
        for rec in dd[seed_slot]:
            core = {f: rec[f] for f in fields if f in rec}
            raw_label = str(rec.get(label_field, ""))
            core["_raw_label"] = raw_label
            core["_norm_label"] = _norm_label(raw_label)
            all_records.append(core)
    print(f"  Seeding from split: '{seed_slot}' ({len(all_records)} records)")

    explicit = fmap.get("labels")
    if explicit:
        label_space = [_norm_label(l) for l in explicit]
    else:
        label_space = sorted({r["_norm_label"] for r in all_records if r["_norm_label"]})

    return all_records, label_space


# Stratified sampling

def stratified_sample(
    records: List[dict],
    label_space: List[str],
    min_seeds: int = 2,
) -> Dict[str, List[dict]]:
    """k = max(min_seeds, min(Lt) // 10) seeds per class."""
    by_label: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        if r["_norm_label"] in label_space:
            by_label[r["_norm_label"]].append(r)

    counts = [len(by_label.get(l, [])) for l in label_space]
    min_count = min(counts) if counts else 0
    k = max(min_seeds, min_count // 10)

    print(f"  Class counts: {dict(zip(label_space, counts))}")
    print(f"  min(Lt)={min_count}  →  k = max({min_seeds}, {min_count}//10) = {k} seeds per class")

    sampled: Dict[str, List[dict]] = {}
    for label in label_space:
        pool = by_label.get(label, [])
        sampled[label] = random.sample(pool, min(k, len(pool)))
    return sampled




def format_seed_list(sampled: Dict[str, List[dict]], schema: Dict) -> str:
    """Format seeds in the original field format (no _norm_label/_raw_label)."""
    fields = schema["fields"]
    lines = []
    for recs in sampled.values():
        for r in recs:
            entry = {f: r[f] for f in fields if f in r}
            lines.append(json.dumps(entry, ensure_ascii=False))
    return "\n".join(lines)


# Batch evaluation

def _record_text(rec: dict, text_fields: List[str]) -> str:
    return " ".join(str(rec.get(f, "")) for f in text_fields)


def evaluate_batch(
    batch: List[Any],
    label_space: List[str],
    schema: Dict,
    seen_texts: List[str],
    similarity_threshold: float = 0.75,
) -> Tuple[List[dict], List[str]]:
    """
    Validate each generated example:
    1. All required fields present
    2. Label (normalised) is in label_space
    3. Not a near-duplicate of seen_texts

    Mutates seen_texts in place with accepted texts.
    Returns (valid_records, feedback_lines).
    """
    fields = schema["fields"]
    label_field = schema["label_field"]
    text_fields = schema["text_fields"]

    valid: List[dict] = []
    field_issues: List[str] = []
    label_issues: List[str] = []
    rep_issues: List[str] = []

    for i, raw in enumerate(batch):
        if not isinstance(raw, dict):
            field_issues.append(f"Example {i+1}: not a dict, skipped")
            continue

        missing = [f for f in fields if f not in raw]
        if missing:
            field_issues.append(f"Example {i+1}: missing fields {missing}, skipped")
            continue

        norm = _norm_label(raw.get(label_field, ""))
        if norm not in label_space:
            label_issues.append(
                f"Example {i+1}: label '{raw.get(label_field)}' not in {label_space} — skipped"
            )
            continue

        text = _record_text(raw, text_fields)
        is_dup = any(
            SequenceMatcher(None, text, seen).ratio() >= similarity_threshold
            for seen in seen_texts
        )
        if is_dup:
            rep_issues.append(f"Example {i+1} (label={norm}) is a near-duplicate — skipped")
            continue
        

        # Keep only the declared fields
        clean = {f: raw[f] for f in fields if f in raw}
        clean["_norm_label"] = norm
        clean["_raw_label"] = raw.get(label_field, "")
        valid.append(clean)
        seen_texts.append(text)

    feedback: List[str] = []
    if field_issues:
        feedback.append("Missing field issues:\n" + "\n".join(f"  - {x}" for x in field_issues))
    if label_issues:
        feedback.append("Label issues:\n" + "\n".join(f"  - {x}" for x in label_issues))
    if rep_issues:
        feedback.append("Near-duplicates to avoid:\n" + "\n".join(f"  - {x}" for x in rep_issues))
    if not feedback:
        feedback.append(
            "All examples in the previous batch were valid. "
            "Continue generating diverse, novel situations."
        )
    counts = {
        "total": len(batch),
        "valid": len(valid),
        "skipped_label": len(label_issues) + len(field_issues),
        "skipped_similarity": len(rep_issues),
    }
    return valid, feedback, counts


# Single generation call

def _call(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    seed_list: str,
    fmt_example: str,
    feedback: Optional[List[str]],
    print_prompt: bool,
) -> List[Any]:
    has_issues = feedback and any(
        kw in f.lower() for f in feedback for kw in ("issue", "missing", "near-dup", "label")
    )
    if has_issues:
        user_content = _FEEDBACK_TMPL.format(
            feedback_text="\n".join(feedback),
            seed_list=seed_list,
            fmt_example=fmt_example,
        )
    else:
        user_content = _USER_TMPL.format(seed_list=seed_list, fmt_example=fmt_example)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    if print_prompt:
        print("\n" + "=" * 70)
        print("FULL PROMPT (batch 1):")
        print("--- SYSTEM ---")
        print(system_prompt)
        print("--- USER ---")
        print(user_content)
        print("=" * 70 + "\n")

    result = call_openrouter_json(
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=0.9,
        max_tokens=4096,
        retries=3,
    )
    if not isinstance(result, list):
        print(f"  [WARN] Model returned non-list: {type(result)}, content={str(result)[:200]}")
        return []
    return result


# Main pipeline

def run_pipeline(
    *,
    dataset_name: str,
    n: int,
    task: int,
    output_path: str,
    model: str,
    min_seeds: int = 2,
    seeds_per_call: int = 5,
    max_batches: int = 4000,
) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment")

    schema = DATASET_SCHEMAS.get(dataset_name)
    if schema is None:
        raise ValueError(
            f"No schema defined for '{dataset_name}'. "
            f"Add an entry to DATASET_SCHEMAS in generate_causal.py."
        )

    print(f"\n{'='*60}")
    print(f"  generate_causal  |  task={task}  |  target={n}  |  model={model}")
    print(f"  dataset: {dataset_name}")
    print(f"{'='*60}")

    print("\n[1] Loading dataset...")
    records, label_space = load_raw_records(dataset_name, schema)
    print(f"  Loaded {len(records)} records, label_space={label_space}")

    print("\n[2] Stratified sampling of seed pool...")
    sampled = stratified_sample(records, label_space, min_seeds=min_seeds)
    # sampled is the pool; each call draws seeds_per_call examples per label from it
    seeds_per_call = min(seeds_per_call, min(len(v) for v in sampled.values()) if sampled else seeds_per_call)
    print(f"  Pool size per label: {dict((l, len(v)) for l, v in sampled.items())}")
    print(f"  Seeds shown per label per call: {seeds_per_call}")

    outputs_per_label = 1 if task == 1 else 5
    outputs_per_call = len(label_space) * outputs_per_label

    tmpl = SYSTEM_PROMPT_TASK1 if task == 1 else SYSTEM_PROMPT_TASK2
    system_prompt = tmpl.format(dataset_name=dataset_name, label_space=str(label_space))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    per_label = n // len(label_space)
    by_label: Dict[str, List[dict]] = defaultdict(list)
    fields = schema["fields"]

    # Block duplicates against the seed split (train) so generated examples
    # don't copy source training examples verbatim.
    seen_texts: List[str] = [_record_text(r, schema["text_fields"]) for r in records]
    feedback: Optional[List[str]] = None
    batch_num = 0
    total_returned = 0
    total_skipped_similarity = 0
    total_skipped_label = 0
    total_saved = 0

    def _all_full() -> bool:
        return all(len(by_label[l]) >= per_label for l in label_space)

    print(f"\n[3] Generating {per_label} examples × {len(label_space)} labels = {per_label * len(label_space)} total")
    print(f"    (~{outputs_per_call} per API call)\n")
    print(f"    Streaming saves to: {output_path}\n")

    with open(output_path, "w") as fout:
        while not _all_full() and batch_num < max_batches:
            batch_num += 1
            label_counts_now = {l: len(by_label[l]) for l in label_space}
            print(f"  [Batch {batch_num}]  per-label: {label_counts_now}  saved: {total_saved}")

            # Draw fresh seeds_per_call examples per label from the pool each call
            call_sample = {
                label: random.sample(pool, seeds_per_call)
                for label, pool in sampled.items()
                if label in label_space
            }
            seed_list = format_seed_list(call_sample, schema)

            raw_batch = _call(
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                seed_list=seed_list,
                fmt_example=schema["fmt_example"],
                feedback=feedback,
                print_prompt=(batch_num == 1),
            )
            print(f"    model returned {len(raw_batch)} examples")

            valid, feedback, batch_counts = evaluate_batch(raw_batch, label_space, schema, seen_texts)
            total_returned += batch_counts["total"]
            total_skipped_similarity += batch_counts["skipped_similarity"]
            total_skipped_label += batch_counts["skipped_label"]

            print(f"    valid: {batch_counts['valid']}  |  skipped_similarity: {batch_counts['skipped_similarity']}  |  skipped_label: {batch_counts['skipped_label']}")
            for fb in feedback:
                print(f"    [EVAL] {fb}")

            accepted = 0
            for rec in valid:
                label = rec["_norm_label"]
                if len(by_label[label]) < per_label:
                    by_label[label].append(rec)
                    # Strip private keys and write immediately
                    clean = {f: rec[f] for f in fields if f in rec}
                    fout.write(json.dumps(clean, ensure_ascii=False) + "\n")
                    fout.flush()
                    accepted += 1
                    total_saved += 1
            print(f"    accepted toward quota: {accepted}")
            time.sleep(0.5)

    if batch_num >= max_batches and not _all_full():
        short = {l: per_label - len(by_label[l]) for l in label_space if len(by_label[l]) < per_label}
        print(f"\n[WARN] Hit max_batches={max_batches}. Still short: {short}")

    # ── Generation summary ──────────────────────────────────────────────────
    similarity_rate = (total_skipped_similarity / total_returned * 100) if total_returned else 0
    print(f"\n[Generation Summary]")
    print(f"  API calls:          {batch_num}")
    print(f"  Total returned:     {total_returned}")
    print(f"  Skipped (similar):  {total_skipped_similarity}  ({similarity_rate:.1f}%)")
    print(f"  Skipped (label):    {total_skipped_label}")
    print(f"  Task:               {task} ({'1 per label/call' if task == 1 else '5 per label/call'})")

    label_field = schema["label_field"]
    dist: Dict[str, int] = defaultdict(int)
    for recs in by_label.values():
        for rec in recs:
            dist[rec.get(label_field, "?")] += 1

    print(f"\n[4] Final dataset: {total_saved} records")
    print(f"    Label distribution: {dict(dist)}")
    print(f"    Saved → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic causal reasoning datapoints using OpenRouter."
    )
    parser.add_argument("--dataset", required=True,
                        help="HuggingFace dataset name, e.g. namesarnav/counterbench")
    parser.add_argument("--n", type=int, required=True,
                        help="Total balanced datapoints to generate")
    parser.add_argument("--task", type=int, choices=[1, 2], default=1,
                        help="1 = one example per label per call; 2 = five per label per call")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file path")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"OpenRouter model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--seeds-per-label", type=int, default=2,
                        help="Minimum seed examples per label in pool (default: 2)")
    parser.add_argument("--seeds-per-call", type=int, default=5,
                        help="Seed examples per label shown per API call (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Max API calls before giving up. "
                             "Defaults to n * 3 (task 1) or n (task 2), "
                             "giving ~3x headroom over theoretical minimum.")
    args = parser.parse_args()

    random.seed(args.seed)

    outputs_per_label = 1 if args.task == 1 else 5
    if args.max_batches is None:
        # Theoretical minimum calls = (n/2) / outputs_per_label for binary.
        # Multiply by 3 to absorb similarity rejections.
        args.max_batches = max(200, (args.n // outputs_per_label) * 3)
    print(f"  max_batches: {args.max_batches}")

    run_pipeline(
        dataset_name=args.dataset,
        n=args.n,
        task=args.task,
        output_path=args.output,
        model=args.model,
        min_seeds=args.seeds_per_label,
        seeds_per_call=args.seeds_per_call,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
