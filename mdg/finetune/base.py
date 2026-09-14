"""
Shared fine-tuning logic for encoder-only classifiers (BERT, RoBERTa, …).

Usage via subclass entry-points (bert.py, roberta.py, …):
    poetry run python -m mdg.finetune.bert \
        --train  mdg/finetune/data/namesarnav_counterbench__train.jsonl \
        --eval   mdg/finetune/data/namesarnav_counterbench__test.jsonl \
        --labels YES,NO \
        --output mdg/finetune/checkpoints/counterbench \
        [--push-to-hub] [--hub-model-id namesarnav/my-model]
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


@dataclass
class FinetuneConfig:
    model_name: str
    train_path: str
    eval_path: str
    label_space: List[str]
    output_dir: str
    # training hyperparams
    num_epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 512
    seed: int = 42
    early_stopping_patience: int = 3
    # hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None


def _load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _make_hf_dataset(records: List[dict], label2id: dict, tokenizer, max_length: int) -> Dataset:
    texts = [r["text"] for r in records]
    labels = [label2id[str(r["label"]).strip().upper()] for r in records]

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )
    encodings["labels"] = labels
    return Dataset.from_dict(encodings)


def _compute_metrics(eval_pred, id2label):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    acc = (preds == labels).mean()
    return {"macro_f1": macro_f1, "accuracy": acc}


def run(cfg: FinetuneConfig) -> None:
    label_space = [l.strip().upper() for l in cfg.label_space]
    label2id = {l: i for i, l in enumerate(label_space)}
    id2label = {i: l for l, i in label2id.items()}

    print(f"\n{'='*60}")
    print(f"  Model : {cfg.model_name}")
    print(f"  Labels: {label_space}")
    print(f"  Train : {cfg.train_path}")
    print(f"  Eval  : {cfg.eval_path}")
    print(f"  Out   : {cfg.output_dir}")
    if cfg.push_to_hub:
        print(f"  Hub   : {cfg.hub_model_id}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(label_space),
        id2label=id2label,
        label2id=label2id,
    )

    train_records = _load_jsonl(cfg.train_path)
    eval_records = _load_jsonl(cfg.eval_path)
    train_ds = _make_hf_dataset(train_records, label2id, tokenizer, cfg.max_length)
    eval_ds = _make_hf_dataset(eval_records, label2id, tokenizer, cfg.max_length)

    output_dir = Path(cfg.output_dir) / cfg.model_name.replace("/", "__")
    output_dir.mkdir(parents=True, exist_ok=True)

    hub_token = cfg.hub_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=cfg.seed,
        logging_steps=50,
        report_to="none",
        # hub
        push_to_hub=cfg.push_to_hub,
        hub_model_id=cfg.hub_model_id if cfg.push_to_hub else None,
        hub_token=hub_token if cfg.push_to_hub else None,
        hub_strategy="end",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=lambda ep: _compute_metrics(ep, id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    trainer.train()

    best_metrics = trainer.evaluate()
    print(f"\nBest eval  macro_f1={best_metrics.get('eval_macro_f1', 0):.4f}  "
          f"accuracy={best_metrics.get('eval_accuracy', 0):.4f}")

    # Save the best checkpoint locally
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Push to hub
    if cfg.push_to_hub and cfg.hub_model_id:
        print(f"\nPushing to Hub: {cfg.hub_model_id} …")
        trainer.push_to_hub(commit_message=f"Fine-tuned on causal classification")
        print(f"Done → https://huggingface.co/{cfg.hub_model_id}")

    print(f"\nSaved to {output_dir}")


def make_parser(default_model: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--train",   required=True, help="Path to train JSONL")
    p.add_argument("--eval",    required=True, help="Path to eval JSONL")
    p.add_argument("--labels",  required=True, help="Comma-separated label space, e.g. YES,NO")
    p.add_argument("--output",  required=True, help="Base output dir (model subdir appended)")
    p.add_argument("--model",   default=default_model)
    p.add_argument("--epochs",  type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr",      type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--push-to-hub", action="store_true",
                   help="Push trained model to HuggingFace Hub")
    p.add_argument("--hub-model-id", default=None,
                   help="Hub repo id, e.g. namesarnav/counterbench-bert. "
                        "Auto-derived from output dir + model name if omitted.")
    return p


def args_to_config(args, default_model: str) -> FinetuneConfig:
    label_space = [l.strip().upper() for l in args.labels.split(",")]

    hub_model_id = args.hub_model_id
    if args.push_to_hub and not hub_model_id:
        # Auto-derive: namesarnav/<dataset_stem>-<short_model_name>
        dataset_stem = Path(args.train).stem.replace("__train", "").replace("namesarnav_", "")
        short_model = args.model.split("/")[-1]
        hub_model_id = f"namesarnav/{dataset_stem}-{short_model}"

    return FinetuneConfig(
        model_name=args.model,
        train_path=args.train,
        eval_path=args.eval,
        label_space=label_space,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=hub_model_id,
    )
