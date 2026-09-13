"""
Fine-tune Qwen family models for causal classification.

Models:
    Qwen/Qwen3-0.6B
    Qwen/Qwen3-1.7B
    Qwen/Qwen3-4B
    Qwen/Qwen3-8B
    Qwen/Qwen3-14B
    Qwen/Qwen2.5-0.5B
    Qwen/Qwen2.5-1.5B
    Qwen/Qwen2.5-3B
    Qwen/Qwen2.5-7B
    Qwen/Qwen2.5-14B

Usage:
    poetry run python -m mdg.finetune.qwen \
        --train mdg/synthetic/data/counterbench_task1.jsonl \
        --eval  mdg/synthetic/data/counterbench_eval.jsonl \
        --labels YES,NO \
        --model Qwen/Qwen3-0.6B

    # Larger model with less memory:
    poetry run python -m mdg.finetune.qwen \
        --model Qwen/Qwen3-8B --batch 2 --grad-accum 8
"""
import argparse
from mdg.models.base_decoder import DecoderFinetuneConfig, run

QWEN_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",      required=True)
    parser.add_argument("--eval",       required=True)
    parser.add_argument("--labels",     default="YES,NO")
    parser.add_argument("--output",     default="mdg/finetune/checkpoints")
    parser.add_argument("--model",      default="Qwen/Qwen3-0.6B")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch",      type=int,   default=8)
    parser.add_argument("--grad-accum", type=int,   default=2)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--lora-r",     type=int,   default=16)
    parser.add_argument("--no-lora",        action="store_true")
    parser.add_argument("--no-4bit",        action="store_true")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    # Small models (0.6B, 1.7B) don't need 4-bit quantization
    small_models = {"Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"}
    load_4bit = not args.no_4bit and args.model not in small_models

    cfg = DecoderFinetuneConfig(
        model_name              = args.model,
        use_lora                = not args.no_lora,
        lora_r                  = args.lora_r,
        lora_target_modules     = QWEN_LORA_TARGETS,
        load_in_4bit            = load_4bit,
        train_path              = args.train,
        eval_path               = args.eval,
        label_space             = args.labels.split(","),
        output_dir              = args.output,
        num_epochs              = args.epochs,
        batch_size              = args.batch,
        gradient_accumulation   = args.grad_accum,
        learning_rate           = args.lr,
        seed                    = args.seed,
    )
    run(cfg)


if __name__ == "__main__":
    main()
