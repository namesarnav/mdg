"""Fine-tune bert-base-uncased for causal classification."""
from mdg.finetune.base import make_parser, args_to_config, run

MODEL = "bert-base-uncased"

if __name__ == "__main__":
    parser = make_parser(MODEL)
    args = parser.parse_args()
    cfg = args_to_config(args, MODEL)
    run(cfg)
