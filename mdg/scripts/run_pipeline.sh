#!/usr/bin/env bash
# Full pipeline: prepare data → fine-tune BERT & RoBERTa → TextAttack all recipes
# Usage: bash mdg/scripts/run_pipeline.sh
set -euo pipefail

NUM_ATTACK_EXAMPLES=200

# ── Dataset definitions ─────────────────────────────────────────────────────
# Format: "hf_name|label_space"
DATASETS=(
  "namesarnav/counterbench|YES,NO"
  "namesarnav/ac-reason|YES,NO"
  "namesarnav/bbh-causal-judgement|YES,NO"
  "namesarnav/causalbench:code|YES,NO"
  "namesarnav/causalbench:math|YES,NO"
  "namesarnav/causalbench:text|YES,NO"
  "namesarnav/corr2cause|0,1"
  "namesarnav/e-care|0,1"
  "namesarnav/fincausal-task1|0,1"
  "namesarnav/natquest|YES,NO"
  "namesarnav/Quriosity|YES,NO"
)

# ── Step 1: Export all datasets to JSONL ────────────────────────────────────
echo "========================================"
echo "STEP 1: Exporting datasets"
echo "========================================"
poetry run python -m mdg.scripts.prepare_finetune_data \
  --datasets \
    namesarnav/counterbench \
    namesarnav/ac-reason \
    namesarnav/bbh-causal-judgement \
    "namesarnav/causalbench:code" \
    "namesarnav/causalbench:math" \
    "namesarnav/causalbench:text" \
    namesarnav/corr2cause \
    namesarnav/e-care \
    namesarnav/fincausal-task1 \
    namesarnav/natquest \
    namesarnav/Quriosity \
  --output-dir mdg/finetune/data

# ── Steps 2 & 3: Fine-tune + Attack per dataset ─────────────────────────────
for entry in "${DATASETS[@]}"; do
  DS="${entry%%|*}"
  LABELS="${entry##*|}"
  LABELS_SPACE="${LABELS//,/ }"

  # Derive safe filename stem (replace / : with _)
  STEM="${DS//\//_}"
  STEM="${STEM//:/_}"

  TRAIN_FILE="mdg/finetune/data/${STEM}__train.jsonl"
  TEST_FILE="mdg/finetune/data/${STEM}__test.jsonl"

  # Some single-split datasets route everything to test — use test as train if no train
  if [ ! -f "$TRAIN_FILE" ]; then
    echo "[WARN] No train split for $DS — using test split as train"
    TRAIN_FILE="$TEST_FILE"
  fi

  if [ ! -f "$TEST_FILE" ]; then
    echo "[WARN] No test split for $DS — skipping"
    continue
  fi

  echo ""
  echo "========================================"
  echo "DATASET: $DS"
  echo "  train: $TRAIN_FILE"
  echo "  test:  $TEST_FILE"
  echo "  labels: $LABELS"
  echo "========================================"

  for MODEL in bert roberta; do
    case $MODEL in
      bert)    MODEL_ID="bert-base-uncased" ;;
      roberta) MODEL_ID="roberta-base" ;;
    esac

    CKPT_DIR="mdg/finetune/checkpoints/${STEM}/${MODEL_ID}"

    echo ""
    echo "---- Fine-tuning $MODEL_ID on $DS ----"
    poetry run python -m "mdg.finetune.${MODEL}" \
      --train  "$TRAIN_FILE" \
      --eval   "$TEST_FILE" \
      --labels "$LABELS" \
      --output "mdg/finetune/checkpoints/${STEM}"

    echo ""
    echo "---- Attacking $MODEL_ID on $DS ----"
    poetry run python -m mdg.adv_attack.attack \
      --model      "$CKPT_DIR" \
      --dataset    "$TEST_FILE" \
      --label-space $LABELS_SPACE \
      --num-examples $NUM_ATTACK_EXAMPLES \
      --output-dir "mdg/adv_attack/results"

    echo "---- Done: $MODEL_ID × $DS ----"
  done
done

echo ""
echo "========================================"
echo "ALL DONE"
echo "========================================"
