"""Fine-tune roberta-base for causal classification."""
import argparse
from pathlib import Path
from mdg.models.base import FinetuneConfig, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",         required=True)
    parser.add_argument("--eval",          required=True)
    parser.add_argument("--labels",        default="YES,NO")
    parser.add_argument("--output",        default="mdg/finetune/checkpoints")
    parser.add_argument("--epochs",        type=int,   default=5)
    parser.add_argument("--batch",         type=int,   default=16)
    parser.add_argument("--lr",            type=float, default=2e-5)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--push-to-hub",   action="store_true")
    parser.add_argument("--hub-model-id",  default=None)
    parser.add_argument("--results-csv",   default=None)
    parser.add_argument("--dataset-name",  default=None)
    args = parser.parse_args()

    hub_model_id = args.hub_model_id
    if args.push_to_hub and not hub_model_id:
        dataset_stem = Path(args.train).stem.replace("__train", "").replace("namesarnav_", "")
        hub_model_id = f"namesarnav/{dataset_stem}-roberta-base"

    cfg = FinetuneConfig(
        model_name    = "roberta-base",
        model_type    = "encoder",
        train_path    = args.train,
        eval_path     = args.eval,
        label_space   = args.labels.split(","),
        output_dir    = args.output,
        num_epochs    = args.epochs,
        batch_size    = args.batch,
        learning_rate = args.lr,
        seed          = args.seed,
        push_to_hub   = args.push_to_hub,
        hub_model_id  = hub_model_id,
        dataset_name  = args.dataset_name or "",
        results_csv   = args.results_csv,
    )
    run(cfg)


if __name__ == "__main__":
    main()
