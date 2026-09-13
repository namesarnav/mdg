"""
Fine-tune FLAN-T5 family for causal classification.

Models (recommended first):
    google/flan-t5-base    (250M)  — good baseline, fits on any GPU
    google/flan-t5-large   (780M)  — stronger, ~4GB VRAM
    google/flan-t5-xl      (3B)    — use --use-lora for consumer GPUs
    google/flan-t5-xxl     (11B)   — requires LoRA + multi-GPU

FLAN-T5 is instruction-tuned, so use a natural-language prefix:
    --prefix "Is the causal relationship in the following text correct? "

Usage:
    poetry run python -m mdg.finetune.flan_t5 \
        --train mdg/synthetic/data/counterbench_task1.jsonl \
        --eval  mdg/synthetic/data/counterbench_eval.jsonl \
        --labels YES,NO \
        --model google/flan-t5-base
"""
import argparse
from mdg.models.base_encoder_decoder import EncDecFinetuneConfig, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",     required=True)
    parser.add_argument("--eval",      required=True)
    parser.add_argument("--labels",    default="YES,NO")
    parser.add_argument("--output",    default="mdg/finetune/checkpoints")
    parser.add_argument("--model",     default="google/flan-t5-base")
    parser.add_argument("--prefix",    default="classify causal relation: ",
                        help="Instruction prefix prepended to each input")
    parser.add_argument("--epochs",    type=int,   default=5)
    parser.add_argument("--batch",     type=int,   default=16)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--use-lora",      action="store_true")
    parser.add_argument("--lora-r",    type=int,   default=16)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    cfg = EncDecFinetuneConfig(
        model_name    = args.model,
        use_lora      = args.use_lora,
        lora_r        = args.lora_r,
        lora_target_modules = ["q", "v"],   # T5 attention layer names
        input_prefix  = args.prefix,
        train_path    = args.train,
        eval_path     = args.eval,
        label_space   = args.labels.split(","),
        output_dir    = args.output,
        num_epochs    = args.epochs,
        batch_size    = args.batch,
        learning_rate = args.lr,
        seed          = args.seed,
    )
    run(cfg)


if __name__ == "__main__":
    main()
