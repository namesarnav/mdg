"""
Fine-tune T5 (t5-base, encoder-decoder) for causal classification via
generative sequence-to-sequence: input = text, output = label token.

Usage:
    poetry run python -m mdg.finetune.t5 \
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
    parser.add_argument("--model",   default="t5-base",
                        help="Any T5/BART variant, e.g. t5-small, t5-large, facebook/bart-base")
    parser.add_argument("--epochs",  type=int,   default=5)
    parser.add_argument("--batch",   type=int,   default=16)
    parser.add_argument("--lr",      type=float, default=3e-4)
    parser.add_argument("--seed",    type=int,   default=42)
    args = parser.parse_args()

    cfg = FinetuneConfig(
        model_name  = args.model,
        model_type  = "encoder-decoder",
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
