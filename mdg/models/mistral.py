"""
Fine-tune Mistral family models for causal classification.

Models:
    mistralai/Mistral-7B-v0.1
    mistralai/Mistral-7B-Instruct-v0.2
    mistralai/Mistral-7B-Instruct-v0.3
    mistralai/Mixtral-8x7B-v0.1   (MoE — needs more VRAM)

Usage:
    poetry run python -m mdg.finetune.mistral \
        --train mdg/synthetic/data/counterbench_task1.jsonl \
        --eval  mdg/synthetic/data/counterbench_eval.jsonl \
        --labels YES,NO \
        --model mistralai/Mistral-7B-v0.1
"""
import argparse
from mdg.models.base_decoder import DecoderFinetuneConfig, run

MISTRAL_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",  required=True)
    parser.add_argument("--eval",   required=True)
    parser.add_argument("--labels", default="YES,NO")
    parser.add_argument("--output", default="mdg/finetune/checkpoints")
    parser.add_argument("--model",  default="mistralai/Mistral-7B-v0.1")
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
        lora_target_modules  = MISTRAL_LORA_TARGETS,
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
