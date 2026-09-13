"""
Fine-tune RoBERTa (roberta-base) for causal classification.

Usage:
    poetry run python -m mdg.finetune.roberta \
        --train mdg/synthetic/data/counterbench_task1.jsonl \
        --eval  mdg/data/counterbench_eval.jsonl \
        --labels YES,NO \
        --output mdg/finetune/checkpoints
"""
import argparse
from mdg.models.base import FinetuneConfig, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",   required=True)
    parser.add_argument("--eval",    required=True)
    parser.add_argument("--labels",  default="YES,NO")
    parser.add_argument("--output",  default="mdg/finetune/checkpoints")
    parser.add_argument("--epochs",  type=int,   default=5)
    parser.add_argument("--batch",   type=int,   default=16)
    parser.add_argument("--lr",      type=float, default=2e-5)
    parser.add_argument("--seed",    type=int,   default=42)
    args = parser.parse_args()

    cfg = FinetuneConfig(
        model_name  = "roberta-base",
        model_type  = "encoder",
        train_path  = args.train,
        eval_path   = args.eval,
        label_space = args.labels.split(","),
        output_dir  = args.output,
        num_epochs  = args.epochs,
        batch_size  = args.batch,
        learning_rate = args.lr,
        seed        = args.seed,
    )
    run(cfg)


if __name__ == "__main__":
    main()
