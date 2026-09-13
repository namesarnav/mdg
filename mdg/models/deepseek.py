"""
Fine-tune DeepSeek family models for causal classification.

Models:
    deepseek-ai/deepseek-llm-7b-base
    deepseek-ai/deepseek-llm-7b-chat
    deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    deepseek-ai/DeepSeek-R1-Distill-Llama-8B

Note: DeepSeek-R1 distill variants are based on Qwen/LLaMA architectures,
so the LoRA targets below work for all of them.

Usage:
    poetry run python -m mdg.finetune.deepseek \
        --train mdg/synthetic/data/counterbench_task1.jsonl \
        --eval  mdg/synthetic/data/counterbench_eval.jsonl \
        --labels YES,NO \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
"""
import argparse
from mdg.models.base_decoder import DecoderFinetuneConfig, run

DEEPSEEK_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",      required=True)
    parser.add_argument("--eval",       required=True)
    parser.add_argument("--labels",     default="YES,NO")
    parser.add_argument("--output",     default="mdg/finetune/checkpoints")
    parser.add_argument("--model",      default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch",      type=int,   default=4)
    parser.add_argument("--grad-accum", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--lora-r",     type=int,   default=16)
    parser.add_argument("--no-lora",        action="store_true")
    parser.add_argument("--no-4bit",        action="store_true")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    small_models = {
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    }
    load_4bit = not args.no_4bit and args.model not in small_models

    cfg = DecoderFinetuneConfig(
        model_name              = args.model,
        use_lora                = not args.no_lora,
        lora_r                  = args.lora_r,
        lora_target_modules     = DEEPSEEK_LORA_TARGETS,
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
