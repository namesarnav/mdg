#!/usr/bin/env bash
# Internal template — not meant to be run directly.
# Each dataset script sources this after setting DS, LABELS, NUM_ATTACK.
set -euo pipefail

STEM="${DS//\//_}"
STEM="${STEM//:/_}"
TRAIN_FILE="mdg/finetune/data/${STEM}__train.jsonl"
TEST_FILE="mdg/finetune/data/${STEM}__test.jsonl"
LABELS_SPACE="${LABELS//,/ }"

echo "========================================"
echo "  DATASET : $DS"
echo "  STEM    : $STEM"
echo "  LABELS  : $LABELS"
echo "========================================"

# Step 1 — export
echo ""
echo "[1/3] Exporting dataset..."
poetry run python -m mdg.scripts.prepare_finetune_data \
  --datasets "$DS" \
  --output-dir mdg/finetune/data

# Step 2 & 3 — fine-tune + attack per model
for MODEL in bert roberta; do
  case $MODEL in
    bert)    MODEL_ID="bert-base-uncased" ;;
    roberta) MODEL_ID="roberta-base" ;;
  esac

  CKPT_DIR="mdg/finetune/checkpoints/${STEM}/${MODEL_ID}"

  # Auto-derive Hub repo id: namesarnav/<dataset_stem>-<model_id>
  DATASET_STEM="${STEM#namesarnav_}"
  HUB_MODEL_ID="namesarnav/${DATASET_STEM}-${MODEL_ID}"

  echo ""
  echo "[2/3] Fine-tuning $MODEL_ID → will push to $HUB_MODEL_ID ..."
  poetry run python -m "mdg.finetune.${MODEL}" \
    --train  "$TRAIN_FILE" \
    --eval   "$TEST_FILE" \
    --labels "$LABELS" \
    --output "mdg/finetune/checkpoints/${STEM}" \
    --push-to-hub \
    --hub-model-id "$HUB_MODEL_ID"

  echo ""
  echo "[3/3] Attacking with $MODEL_ID (all recipes)..."
  poetry run python -m mdg.adv_attack.attack \
    --model      "$CKPT_DIR" \
    --dataset    "$TEST_FILE" \
    --label-space $LABELS_SPACE \
    --num-examples "${NUM_ATTACK:-200}" \
    --output-dir "mdg/adv_attack/results"

  echo "---- Done: $MODEL_ID × $DS ----"
done

echo ""
echo "========================================"
echo "  FINISHED: $DS"
echo "========================================"
