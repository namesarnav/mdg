poetry run python -m mdg.adv_attack.textattack \
  --files \
    mdg/synthetic/data/counterbench_task1.jsonl \
    mdg/synthetic/data/counterbench_task2.jsonl \
    mdg/synthetic/data/ac_reason_task1.jsonl \
    mdg/synthetic/data/ac_reason_task2.jsonl \
    mdg/synthetic/data/bbh_causal_judgement_task1.jsonl \
    mdg/synthetic/data/bbh_causal_judgement_task2.jsonl \
  --output-dir mdg/adv_attack/data

# ---

# Adversarial attack (requires a fine-tuned model):

# You need to fine-tune a model first (e.g. BERT), then run attack.py against it. Once you have a checkpoint:

# counterbench
poetry run python -m mdg.adv_attack.attack \
  --model  mdg/finetune/checkpoints/bert-base-uncased \
  --dataset mdg/synthetic/data/counterbench_task1.jsonl \
  --label-space YES NO \
  --num-examples 200 \
  --output-dir mdg/adv_attack/results

# ac-reason
poetry run python -m mdg.adv_attack.attack \
  --model  mdg/finetune/checkpoints/bert-base-uncased \
  --dataset mdg/synthetic/data/ac_reason_task1.jsonl \
  --label-space YES NO \
  --num-examples 200 \
  --output-dir mdg/adv_attack/results

# bbh-causal-judgement
poetry run python -m mdg.adv_attack.attack \
  --model  mdg/finetune/checkpoints/bert-base-uncased \
  --dataset mdg/synthetic/data/bbh_causal_judgement_task1.jsonl \
  --label-space YES NO \
  --num-examples 200 \
  --output-dir mdg/adv_attack/results