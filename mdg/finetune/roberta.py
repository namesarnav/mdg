"""Fine-tune roberta-base for causal classification."""
from mdg.finetune.base import make_parser, args_to_config, run

MODEL = "roberta-base"

if __name__ == "__main__":
    parser = make_parser(MODEL)
    args = parser.parse_args()
    cfg = args_to_config(args, MODEL)
    run(cfg)
