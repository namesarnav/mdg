"""
Fine-tune Google Gemma family models for causal classification.

Models:
    google/gemma-7b
    google/gemma-2b
    google/gemma-7b-it
    google/gemma-2b-it
    google/gemma-2-9b
    google/gemma-2-2b

Note: requires HuggingFace access token for gated Gemma repos.
    huggingface-cli login

Usage:
    poetry run python -m mdg.finetune.gemma \
        --train mdg/synthetic/data/counterbench_task1.jsonl \
        --eval  mdg/synthetic/data/counterbench_eval.jsonl \
        --labels YES,NO \
        --model google/gemma-2b
"""
import argparse
from mdg.models.base_decoder import DecoderFinetuneConfig, run

# Gemma uses standard attention projections; no gate/up/down in MLP by default
GEMMA_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",  required=True)
    parser.add_argument("--eval",   required=True)
    parser.add_argument("--labels", default="YES,NO")
    parser.add_argument("--output", default="mdg/finetune/checkpoints")
    parser.add_argument("--model",  default="google/gemma-2b")
    parser.add_argument("--epochs", type=int,   default=3)
    parser.add_argument("--batch",  type=int,   default=4)
    parser.add_argument("--lr",     type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int,   default=16)
    parser.add_argument("--no-lora",    action="store_true")
    parser.add_argument("--no-4bit",    action="store_true")
    parser.add_argument("--seed",   type=int,   default=42)
    args = parser.parse_args()

    cfg = DecoderFinetuneConfig(
        model_name           = args.model,
        use_lora             = not args.no_lora,
        lora_r               = args.lora_r,
        lora_target_modules  = GEMMA_LORA_TARGETS,
        load_in_4bit         = not args.no_4bit,
        train_path           = args.train,
        eval_path            = args.eval,
        label_space          = args.labels.split(","),
        output_dir           = args.output,
        num_epochs           = args.epochs,
        batch_size           = args.batch,
        learning_rate        = args.lr,
        seed                 = args.seed,
    )
    run(cfg)


if __name__ == "__main__":
    main()
