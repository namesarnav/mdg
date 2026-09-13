"""
Base template for fine-tuning decoder-only LLMs on causal classification.

Approach: AutoModelForSequenceClassification on top of the causal LM backbone,
with optional LoRA (PEFT) and 4-bit quantization (bitsandbytes) for large models.

Supports any HuggingFace causal LM: Mistral, LLaMA, Gemma, Qwen, DeepSeek, etc.

Dependencies:
    pip install transformers peft bitsandbytes accelerate scikit-learn
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
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DecoderFinetuneConfig:
    # Model
    model_name: str = "mistralai/Mistral-7B-v0.1"

    # LoRA
    use_lora:       bool      = True
    lora_r:         int       = 16
    lora_alpha:     int       = 32
    lora_dropout:   float     = 0.05
    # Which linear layers to apply LoRA to — override per architecture if needed
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Quantization (set load_in_4bit=True for 7B+ models on consumer GPUs)
    load_in_4bit:   bool = True
    load_in_8bit:   bool = False    # mutually exclusive with load_in_4bit

    # Data
    train_path:  str       = ""
    eval_path:   str       = ""
    text_field:  str       = "text"
    label_field: str       = "label"
    label_space: List[str] = field(default_factory=lambda: ["YES", "NO"])

    # Training
    output_dir:              str   = "mdg/finetune/checkpoints"
    num_epochs:              int   = 3
    batch_size:              int   = 8
    eval_batch_size:         int   = 16
    gradient_accumulation:   int   = 2
    learning_rate:           float = 2e-4
    weight_decay:            float = 0.01
    warmup_ratio:            float = 0.1
    max_length:              int   = 512
    seed:                    int   = 42
    fp16:                    bool  = False   # use bf16 instead for modern GPUs
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
    label2id: Dict[str, int],
) -> Dataset:
    texts  = [str(r[text_field])  for r in records]
    labels = [label2id.get(str(r[label_field]).strip().upper(), -1) for r in records]
    return Dataset.from_dict({"text": texts, "label": labels})


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    valid = labels != -1
    acc = accuracy_score(labels[valid], preds[valid])
    f1  = f1_score(labels[valid], preds[valid], average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": f1}


# ──────────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def run(cfg: DecoderFinetuneConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    label_space = [l.strip().upper() for l in cfg.label_space]
    label2id    = {l: i for i, l in enumerate(label_space)}
    id2label    = {i: l for l, i in label2id.items()}
    num_labels  = len(label_space)

    print(f"\n{'='*60}")
    print(f"  model : {cfg.model_name}")
    print(f"  lora  : {cfg.use_lora}  4bit={cfg.load_in_4bit}  8bit={cfg.load_in_8bit}")
    print(f"  labels: {label_space}")
    print(f"{'='*60}")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    # Decoder-only models often lack a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # right-pad for classification

    # ── Data ───────────────────────────────────────────────────────────────────
    train_records = load_jsonl(cfg.train_path)
    eval_records  = load_jsonl(cfg.eval_path)
    print(f"  train: {len(train_records)}  eval: {len(eval_records)}")

    train_ds = build_dataset(train_records, cfg.text_field, cfg.label_field, label2id)
    eval_ds  = build_dataset(eval_records,  cfg.text_field, cfg.label_field, label2id)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_length,
            padding=False,
        )

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    eval_tok  = eval_ds.map(tokenize,  batched=True, remove_columns=["text"])

    # ── Quantization config ────────────────────────────────────────────────────
    bnb_config = None
    if cfg.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif cfg.load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        quantization_config=bnb_config,
        device_map="auto" if (cfg.load_in_4bit or cfg.load_in_8bit) else None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
    )
    # Required when pad_token == eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # ── LoRA ───────────────────────────────────────────────────────────────────
    if cfg.use_lora:
        if cfg.load_in_4bit or cfg.load_in_8bit:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
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

    training_args = TrainingArguments(
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
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        seed=cfg.seed,
        report_to="none",
        # Needed when using gradient checkpointing with PEFT
        gradient_checkpointing=cfg.use_lora,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
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
