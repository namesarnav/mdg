"""
Fine-tune BART family for causal classification.

Models:
    facebook/bart-base    (140M)
    facebook/bart-large   (400M)
    facebook/bart-large-mnli  (400M, pre-trained on NLI — good starting point
                               for binary causal classification)

Usage:
    poetry run python -m mdg.finetune.bart \
        --train mdg/synthetic/data/counterbench_task1.jsonl \
        --eval  mdg/synthetic/data/counterbench_eval.jsonl \
        --labels YES,NO \
        --model facebook/bart-large
"""
import argparse
from mdg.models.base_encoder_decoder import EncDecFinetuneConfig, run

BART_LORA_TARGETS = ["q_proj", "v_proj", "k_proj", "out_proj"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",     required=True)
    parser.add_argument("--eval",      required=True)
    parser.add_argument("--labels",    default="YES,NO")
    parser.add_argument("--output",    default="mdg/finetune/checkpoints")
    parser.add_argument("--model",     default="facebook/bart-base")
    parser.add_argument("--epochs",    type=int,   default=5)
    parser.add_argument("--batch",     type=int,   default=16)
    parser.add_argument("--lr",        type=float, default=2e-5)
    parser.add_argument("--use-lora",      action="store_true")
    parser.add_argument("--lora-r",    type=int,   default=16)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    cfg = EncDecFinetuneConfig(
        model_name    = args.model,
        use_lora      = args.use_lora,
        lora_r        = args.lora_r,
        lora_target_modules = BART_LORA_TARGETS,
        input_prefix  = "",   # BART doesn't need an instruction prefix
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
