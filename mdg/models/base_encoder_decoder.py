"""
Base template for fine-tuning encoder-decoder models on causal classification.

Classification approach: generative — the model is trained to output a label
string (e.g. "Yes", "No") given the input text. At eval time the output token
is mapped back to a label id.

Optionally uses LoRA (PEFT) for large encoder-decoder models (FLAN-UL2, mT5-xl).

Supported families:
    T5 / FLAN-T5         google-t5/t5-*, google/flan-t5-*
    BART                 facebook/bart-*
    mT5                  google/mt5-*
    FLAN-UL2             google/flan-ul2
    UnifiedQA            allenai/unifiedqa-t5-*
    LED                  allenai/led-*

Dependencies:
    pip install transformers peft accelerate scikit-learn sentencepiece
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EncDecFinetuneConfig:
    # Model
    model_name: str = "google/flan-t5-base"

    # LoRA (recommended only for xl / xxl / flan-ul2 to save memory)
    use_lora:             bool      = False
    lora_r:               int       = 16
    lora_alpha:           int       = 32
    lora_dropout:         float     = 0.05
    lora_target_modules:  List[str] = field(
        default_factory=lambda: ["q", "v"]   # T5/FLAN attention projection names
    )

    # Data
    train_path:  str       = ""
    eval_path:   str       = ""
    text_field:  str       = "text"
    label_field: str       = "label"
    label_space: List[str] = field(default_factory=lambda: ["YES", "NO"])

    # Prompt prefix — helps instruction-tuned models (FLAN-T5, UnifiedQA)
    # Set to "" to disable. Example: "classify causal relation: "
    input_prefix: str = ""

    # Training
    output_dir:              str   = "mdg/finetune/checkpoints"
    num_epochs:              int   = 5
    batch_size:              int   = 16
    eval_batch_size:         int   = 32
    gradient_accumulation:   int   = 1
    learning_rate:           float = 3e-4
    weight_decay:            float = 0.01
    warmup_ratio:            float = 0.1
    max_input_length:        int   = 512
    max_target_length:       int   = 8      # label token(s) only
    seed:                    int   = 42
    fp16:                    bool  = False
    bf16:                    bool  = torch.cuda.is_available()
    early_stopping_patience: int   = 3


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(
    records: List[dict],
    text_field: str,
    label_field: str,
    label_space: List[str],
    input_prefix: str = "",
) -> Dataset:
    # Map label → its title-cased display string as the generation target
    label2target = {l: l.replace("_", " ").title() for l in label_space}
    texts   = [input_prefix + str(r[text_field]) for r in records]
    targets = [label2target.get(str(r[label_field]).strip().upper(), "") for r in records]
    return Dataset.from_dict({"text": texts, "target": targets})


# ──────────────────────────────────────────────────────────────────────────────
# Tokenisation
# ──────────────────────────────────────────────────────────────────────────────

def tokenize(dataset: Dataset, tokenizer, max_input: int, max_target: int) -> Dataset:
    def _tok(batch):
        model_inputs = tokenizer(
            batch["text"],
            max_length=max_input,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=max_target,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return dataset.map(_tok, batched=True, remove_columns=["text", "target"])


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def make_compute_metrics(tokenizer, label_space: List[str]):
    target2label = {l.replace("_", " ").title(): l for l in label_space}

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(
            [[t for t in l if t != -100] for l in labels],
            skip_special_tokens=True,
        )
        pred_ids  = [target2label.get(p.strip(), p.strip().upper()) for p in decoded_preds]
        label_ids = [target2label.get(l.strip(), l.strip().upper()) for l in decoded_labels]

        valid = [i for i, l in enumerate(label_ids) if l in label_space]
        if not valid:
            return {"accuracy": 0.0, "macro_f1": 0.0}
        p_v = [pred_ids[i]  for i in valid]
        l_v = [label_ids[i] for i in valid]
        acc = accuracy_score(l_v, p_v)
        f1  = f1_score(l_v, p_v, labels=label_space, average="macro", zero_division=0)
        return {"accuracy": acc, "macro_f1": f1}

    return compute_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def run(cfg: EncDecFinetuneConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    label_space = [l.strip().upper() for l in cfg.label_space]

    print(f"\n{'='*60}")
    print(f"  model  : {cfg.model_name}")
    print(f"  lora   : {cfg.use_lora}")
    print(f"  labels : {label_space}")
    print(f"  prefix : {cfg.input_prefix!r}")
    print(f"{'='*60}")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ── Data ───────────────────────────────────────────────────────────────────
    train_records = load_jsonl(cfg.train_path)
    eval_records  = load_jsonl(cfg.eval_path)
    print(f"  train: {len(train_records)}  eval: {len(eval_records)}")

    train_ds = build_dataset(train_records, cfg.text_field, cfg.label_field, label_space, cfg.input_prefix)
    eval_ds  = build_dataset(eval_records,  cfg.text_field, cfg.label_field, label_space, cfg.input_prefix)

    train_tok = tokenize(train_ds, tokenizer, cfg.max_input_length, cfg.max_target_length)
    eval_tok  = tokenize(eval_ds,  tokenizer, cfg.max_input_length, cfg.max_target_length)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
    )

    if cfg.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ── Training args ──────────────────────────────────────────────────────────
    output_dir = os.path.join(cfg.output_dir, Path(cfg.model_name).name)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=cfg.max_target_length,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        seed=cfg.seed,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        compute_metrics=make_compute_metrics(tokenizer, label_space),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    print(f"\n[Training]")
    trainer.train()

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print(f"\n[Evaluation]")
    results = trainer.evaluate()
    print(json.dumps(results, indent=2))

    # ── Save ───────────────────────────────────────────────────────────────────
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n  Saved → {output_dir}")

    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
