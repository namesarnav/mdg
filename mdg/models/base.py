"""
Base template for fine-tuning transformer models on causal classification.

Supports:
  - Encoder-only models (BERT, RoBERTa, etc.): sequence classification head
  - Encoder-decoder models (T5, BART, etc.): generative classification via
    constrained decoding (label tokens only)

Subclass FinetuneConfig to configure a specific model, then call run().

Run: 
poetry run python -m mdg.models.qwen \
  --train mdg/synthetic/data/counterbench_task1.jsonl \
  --eval  mdg/synthetic/data/counterbench_task2.jsonl \
  --labels YES,NO --model Qwen/Qwen3-0.6B \
  --output mdg/finetune/checkpoints

"""
from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)


# Config

@dataclass
class FinetuneConfig:
    # Model
    model_name: str = "bert-base-uncased"
    model_type: str = "encoder"          # "encoder" | "encoder-decoder"

    # Data  (paths to JSONL files produced by generate_causal.py or HF loaders)
    train_path: str = ""
    eval_path:  str = ""
    text_field:  str = "text"
    label_field: str = "label"
    label_space: List[str] = field(default_factory=lambda: ["YES", "NO"])

    # Training
    output_dir:       str   = "mdg/finetune/checkpoints"
    num_epochs:       int   = 5
    batch_size:       int   = 16
    eval_batch_size:  int   = 32
    learning_rate:    float = 2e-5
    weight_decay:     float = 0.01
    max_length:       int   = 256
    seed:             int   = 42
    fp16:             bool  = torch.cuda.is_available()
    early_stopping_patience: int = 3

    # Encoder-decoder specific
    max_target_length: int = 8

    # Hub
    push_to_hub:   bool            = False
    hub_model_id:  Optional[str]   = None
    hub_token:     Optional[str]   = None

    # Results
    dataset_name:  str             = ""
    results_csv:   Optional[str]   = None


# Data loading

def load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(
    records: List[dict],
    text_field: str,
    label_field: str,
    label2id: Dict[str, int],
) -> Dataset:
    texts  = [str(r[text_field])  for r in records]
    labels = [label2id.get(str(r[label_field]).strip().upper(), -1) for r in records]
    return Dataset.from_dict({"text": texts, "label": labels})



# Tokenisation


def tokenize_encoder(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def _tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    return dataset.map(_tok, batched=True, remove_columns=["text"])


def tokenize_seq2seq(
    dataset: Dataset,
    tokenizer,
    max_length: int,
    max_target_length: int,
    id2label: Dict[int, str],
) -> Dataset:
    def _tok(batch):
        model_inputs = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        label_strings = [id2label.get(l, "") for l in batch["label"]]
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(
                label_strings,
                truncation=True,
                max_length=max_target_length,
                padding=False,
            )
        model_inputs["labels"] = targets["input_ids"]
        return model_inputs
    return dataset.map(_tok, batched=True, remove_columns=["text"])


# Metrics

def make_compute_metrics_encoder(label_space: List[str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        valid = labels != -1
        acc = accuracy_score(labels[valid], preds[valid])
        f1  = f1_score(labels[valid], preds[valid], average="macro", zero_division=0)
        return {"accuracy": acc, "macro_f1": f1}
    return compute_metrics


def make_compute_metrics_seq2seq(tokenizer, label2id: Dict[str, int]):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(
            [[t for t in l if t != -100] for l in labels],
            skip_special_tokens=True,
        )
        pred_ids  = [label2id.get(p.strip().upper(), -1) for p in decoded_preds]
        label_ids = [label2id.get(l.strip().upper(), -1) for l in decoded_labels]
        valid = [i for i, (p, l) in enumerate(zip(pred_ids, label_ids)) if l != -1]
        if not valid:
            return {"accuracy": 0.0, "macro_f1": 0.0}
        p_v = [pred_ids[i]  for i in valid]
        l_v = [label_ids[i] for i in valid]
        acc = accuracy_score(l_v, p_v)
        f1  = f1_score(l_v, p_v, average="macro", zero_division=0)
        return {"accuracy": acc, "macro_f1": f1}
    return compute_metrics


# Main entrypoint

def run(cfg: FinetuneConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    label_space = [l.strip().upper() for l in cfg.label_space]
    label2id    = {l: i for i, l in enumerate(label_space)}
    id2label    = {i: l for l, i in label2id.items()}
    num_labels  = len(label_space)

    hub_token = cfg.hub_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    hub_kwargs = dict(
        push_to_hub=cfg.push_to_hub,
        hub_model_id=cfg.hub_model_id if cfg.push_to_hub else None,
        hub_token=hub_token if cfg.push_to_hub else None,
        hub_strategy="end",
    ) if cfg.push_to_hub and cfg.hub_model_id else {}

    print(f"\n{'='*60}")
    print(f"  model : {cfg.model_name}  ({cfg.model_type})")
    print(f"  labels: {label_space}")
    print(f"{'='*60}")

    # ── Load data                     ──
    train_records = load_jsonl(cfg.train_path)
    eval_records  = load_jsonl(cfg.eval_path)
    print(f"  train: {len(train_records)}  eval: {len(eval_records)}")

    train_ds = build_dataset(train_records, cfg.text_field, cfg.label_field, label2id)
    eval_ds  = build_dataset(eval_records,  cfg.text_field, cfg.label_field, label2id)

    # ── Tokenizer                     ──
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    output_dir = os.path.join(cfg.output_dir, Path(cfg.model_name).name)

    if cfg.model_type == "encoder":
        # ── Encoder-only (BERT, RoBERTa, …)            ──
        train_tok = tokenize_encoder(train_ds, tokenizer, cfg.max_length)
        eval_tok  = tokenize_encoder(eval_ds,  tokenizer, cfg.max_length)

        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            fp16=cfg.fp16,
            seed=cfg.seed,
            report_to="none",
            **hub_kwargs,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=eval_tok,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=make_compute_metrics_encoder(label_space),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
        )

    elif cfg.model_type == "encoder-decoder":
        # ── Encoder-decoder (T5, BART, …)             ─
        train_tok = tokenize_seq2seq(train_ds, tokenizer, cfg.max_length, cfg.max_target_length, id2label)
        eval_tok  = tokenize_seq2seq(eval_ds,  tokenizer, cfg.max_length, cfg.max_target_length, id2label)

        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

        # Force generation to only produce valid label tokens
        label_token_ids = [
            tokenizer.encode(l, add_special_tokens=False) for l in label_space
        ]

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            predict_with_generate=True,
            generation_max_length=cfg.max_target_length,
            fp16=cfg.fp16,
            seed=cfg.seed,
            report_to="none",
            **hub_kwargs,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=eval_tok,
            compute_metrics=make_compute_metrics_seq2seq(tokenizer, label2id),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
        )

    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type!r}. Use 'encoder' or 'encoder-decoder'.")

    # ── Train
    print(f"\n[Training]")
    trainer.train()

    # ── Evaluate
    print(f"\n[Evaluation]")
    results = trainer.evaluate()
    print(json.dumps(results, indent=2))

    # ── Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n  Saved → {output_dir}")

    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results → {results_path}")

    # ── Save CSV row
    if cfg.results_csv:
        dataset_name = cfg.dataset_name or Path(cfg.train_path).stem.replace("__train", "")
        row = {
            "model":      cfg.model_name,
            "dataset":    dataset_name,
            "num_train":  len(train_records),
            "num_eval":   len(eval_records),
            "macro_f1":   round(results.get("eval_macro_f1", 0), 4),
            "accuracy":   round(results.get("eval_accuracy", 0), 4),
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
        }
        csv_path = Path(cfg.results_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"  CSV row → {cfg.results_csv}")

    # ── Push to Hub
    if cfg.push_to_hub and cfg.hub_model_id:
        print(f"\n  Pushing to Hub: {cfg.hub_model_id} ...")
        trainer.push_to_hub(commit_message="Fine-tuned on causal classification")
        print(f"  Done → https://huggingface.co/{cfg.hub_model_id}")
