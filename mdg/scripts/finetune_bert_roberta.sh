#!/usr/bin/env bash
# Fine-tune BERT and RoBERTa on all 11 causal datasets.
# Each run pushes the model to HuggingFace Hub.
# Failures are logged and skipped — the script always continues to the next job.
#
# Usage:
#   bash mdg/scripts/finetune_bert_roberta.sh
#
# Prerequisites:
#   huggingface-cli login   (or export HF_TOKEN=hf_...)
#   poetry run python -m mdg.scripts.prepare_finetune_data ... (export datasets first)

set -uo pipefail   # no -e so failures don't stop the script

DATA_DIR="mdg/finetune/data"
CKPT_DIR="mdg/finetune/checkpoints"
LOG_DIR="mdg/finetune/logs"
mkdir -p "$LOG_DIR"

FAILED=()
PASSED=()

# ─── Helper ────────────────────────────────────────────────────────────────────
run_finetune() {
  local MODULE="$1"   # mdg.finetune.bert or mdg.finetune.roberta
  local MODEL_ID="$2" # bert-base-uncased or roberta-base
  local LABELS="$3"
  local STEM="$4"     # safe filesystem stem, e.g. namesarnav_counterbench
  local TRAIN_FILE="${DATA_DIR}/${STEM}__train.jsonl"
  local TEST_FILE="${DATA_DIR}/${STEM}__test.jsonl"

  # Strip leading namesarnav_ for hub id
  local DATASET_STEM="${STEM#namesarnav_}"
  local HUB_ID="namesarnav/${DATASET_STEM}-${MODEL_ID}"
  local LOG_FILE="${LOG_DIR}/${DATASET_STEM}-${MODEL_ID}.log"

  echo ""
  echo "────────────────────────────────────────────────────────"
  echo "  MODEL  : $MODEL_ID"
  echo "  DATASET: $STEM"
  echo "  HUB    : $HUB_ID"
  echo "  LOG    : $LOG_FILE"
  echo "────────────────────────────────────────────────────────"

  # Check data files exist
  if [ ! -f "$TRAIN_FILE" ]; then
    echo "  [SKIP] Train file not found: $TRAIN_FILE"
    FAILED+=("${DATASET_STEM}/${MODEL_ID} — missing train file")
    return
  fi
  if [ ! -f "$TEST_FILE" ]; then
    echo "  [SKIP] Test file not found: $TEST_FILE"
    FAILED+=("${DATASET_STEM}/${MODEL_ID} — missing test file")
    return
  fi

  poetry run python -m "$MODULE" \
    --train        "$TRAIN_FILE" \
    --eval         "$TEST_FILE" \
    --labels       "$LABELS" \
    --output       "${CKPT_DIR}/${STEM}" \
    --push-to-hub \
    --hub-model-id "$HUB_ID" \
    2>&1 | tee "$LOG_FILE"

  local EXIT_CODE="${PIPESTATUS[0]}"
  if [ "$EXIT_CODE" -ne 0 ]; then
    echo "  [ERROR] Exit code $EXIT_CODE — see $LOG_FILE"
    FAILED+=("${DATASET_STEM}/${MODEL_ID} — exit $EXIT_CODE")
  else
    echo "  [OK] → https://huggingface.co/$HUB_ID"
    PASSED+=("${DATASET_STEM}/${MODEL_ID}")
  fi
}

# ─── Step 1: Export all datasets ───────────────────────────────────────────────
echo "========================================================"
echo "STEP 1 — Exporting datasets to JSONL"
echo "========================================================"

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
  --output-dir "$DATA_DIR"

if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "[WARN] Data export had errors — some datasets may be missing. Continuing anyway."
fi

# ─── Step 2: BERT ──────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "STEP 2 — Fine-tuning BERT"
echo "========================================================"

run_finetune "mdg.finetune.bert" "bert-base-uncased" "YES,NO" "namesarnav_counterbench"
run_finetune "mdg.finetune.bert" "bert-base-uncased" "YES,NO" "namesarnav_ac-reason"
run_finetune "mdg.finetune.bert" "bert-base-uncased" "YES,NO" "namesarnav_bbh-causal-judgement"
run_finetune "mdg.finetune.bert" "bert-base-uncased" "YES,NO" "namesarnav_causalbench_code"
run_finetune "mdg.finetune.bert" "bert-base-uncased" "YES,NO" "namesarnav_causalbench_math"
run_finetune "mdg.finetune.bert" "bert-base-uncased" "YES,NO" "namesarnav_causalbench_text"
run_finetune "mdg.finetune.bert" "bert-base-uncased" "0,1"    "namesarnav_corr2cause"
run_finetune "mdg.finetune.bert" "bert-base-uncased" "0,1"    "namesarnav_e-care"
run_finetune "mdg.finetune.bert" "bert-base-uncased" "0,1"    "namesarnav_fincausal-task1"
run_finetune "mdg.finetune.bert" "bert-base-uncased" "YES,NO" "namesarnav_natquest"
run_finetune "mdg.finetune.bert" "bert-base-uncased" "YES,NO" "namesarnav_Quriosity"

# ─── Step 3: RoBERTa ───────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "STEP 3 — Fine-tuning RoBERTa"
echo "========================================================"

run_finetune "mdg.finetune.roberta" "roberta-base" "YES,NO" "namesarnav_counterbench"
run_finetune "mdg.finetune.roberta" "roberta-base" "YES,NO" "namesarnav_ac-reason"
run_finetune "mdg.finetune.roberta" "roberta-base" "YES,NO" "namesarnav_bbh-causal-judgement"
run_finetune "mdg.finetune.roberta" "roberta-base" "YES,NO" "namesarnav_causalbench_code"
run_finetune "mdg.finetune.roberta" "roberta-base" "YES,NO" "namesarnav_causalbench_math"
run_finetune "mdg.finetune.roberta" "roberta-base" "YES,NO" "namesarnav_causalbench_text"
run_finetune "mdg.finetune.roberta" "roberta-base" "0,1"    "namesarnav_corr2cause"
run_finetune "mdg.finetune.roberta" "roberta-base" "0,1"    "namesarnav_e-care"
run_finetune "mdg.finetune.roberta" "roberta-base" "0,1"    "namesarnav_fincausal-task1"
run_finetune "mdg.finetune.roberta" "roberta-base" "YES,NO" "namesarnav_natquest"
run_finetune "mdg.finetune.roberta" "roberta-base" "YES,NO" "namesarnav_Quriosity"

# ─── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  SUMMARY"
echo "========================================================"
echo "  Passed : ${#PASSED[@]}"
for job in "${PASSED[@]}"; do echo "    ✓  $job"; done

echo "  Failed : ${#FAILED[@]}"
for job in "${FAILED[@]}"; do echo "    ✗  $job"; done

echo ""
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "Some jobs failed. Logs are in $LOG_DIR"
  exit 1
else
  echo "All jobs completed successfully."
  exit 0
fi
