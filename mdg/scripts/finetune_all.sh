#!/usr/bin/env bash
# Fine-tune BERT, RoBERTa, and T5 on all datasets.
# - Skips any model/dataset pair whose checkpoint already exists
# - Uses the LARGER split for training, smaller for eval
# - Pushes each model to HuggingFace Hub
# - Appends F1 + accuracy to mdg/finetune/train_results.csv
#
# Usage: bash mdg/scripts/finetune_all.sh

set -uo pipefail

# ─── Poetry env check ─────────────────────────────────────────────────────────
echo "Checking poetry Python environment..."
POETRY_PYTHON_BIN=$(poetry env info -e 2>/dev/null || true)
if [ -z "$POETRY_PYTHON_BIN" ] || [ ! -f "$POETRY_PYTHON_BIN" ]; then
  echo "[FIX] Recreating poetry env with system python3..."
  poetry env remove --all 2>/dev/null || true
  poetry env use "$(which python3)"
  poetry install --no-interaction
  echo "[FIX] Done."
else
  echo "Poetry env OK: $POETRY_PYTHON_BIN"
fi

DATA_DIR="mdg/finetune/data"
CKPT_DIR="mdg/finetune/checkpoints"
LOG_DIR="mdg/finetune/logs"
RESULTS_CSV="mdg/finetune/train_results.csv"
mkdir -p "$LOG_DIR"

FAILED=()
PASSED=()
SKIPPED=()

# ─── Helper ───────────────────────────────────────────────────────────────────
run_finetune() {
  local MODULE="$1"    # e.g. mdg.models.bert
  local MODEL_ID="$2"  # e.g. bert-base-uncased
  local LABELS="$3"
  local STEM="$4"      # e.g. namesarnav_counterbench

  local TRAIN_FILE="${DATA_DIR}/${STEM}__train.jsonl"
  local TEST_FILE="${DATA_DIR}/${STEM}__test.jsonl"
  local DATASET_STEM="${STEM#namesarnav_}"
  local HUB_ID="namesarnav/${DATASET_STEM}-${MODEL_ID}"
  local CKPT="${CKPT_DIR}/${STEM}/${MODEL_ID}"
  local LOG_FILE="${LOG_DIR}/${DATASET_STEM}-${MODEL_ID}.log"

  echo ""
  echo "────────────────────────────────────────────────────────"
  echo "  MODEL  : $MODEL_ID"
  echo "  DATASET: $STEM"
  echo "  HUB    : $HUB_ID"
  echo "────────────────────────────────────────────────────────"

  # Skip if checkpoint already exists
  if [ -d "$CKPT" ] && [ -f "${CKPT}/config.json" ]; then
    echo "  [SKIP] Checkpoint already exists: $CKPT"
    SKIPPED+=("${DATASET_STEM}/${MODEL_ID}")
    return
  fi

  # Check data files
  if [ ! -f "$TRAIN_FILE" ]; then
    echo "  [SKIP] Missing train file: $TRAIN_FILE"
    FAILED+=("${DATASET_STEM}/${MODEL_ID} — missing train file")
    return
  fi
  if [ ! -f "$TEST_FILE" ]; then
    echo "  [SKIP] Missing test file: $TEST_FILE"
    FAILED+=("${DATASET_STEM}/${MODEL_ID} — missing test file")
    return
  fi

  # Use the LARGER split for training, smaller for eval
  TRAIN_COUNT=$(wc -l < "$TRAIN_FILE")
  TEST_COUNT=$(wc -l < "$TEST_FILE")
  if [ "$TRAIN_COUNT" -ge "$TEST_COUNT" ]; then
    FINETUNE_ON="$TRAIN_FILE"
    EVAL_ON="$TEST_FILE"
  else
    echo "  [INFO] Test split is larger — training on test, evaluating on train"
    FINETUNE_ON="$TEST_FILE"
    EVAL_ON="$TRAIN_FILE"
  fi
  echo "  Train: $TRAIN_COUNT lines → using $(basename $FINETUNE_ON) for training"
  echo "  Eval : $TEST_COUNT  lines → using $(basename $EVAL_ON) for eval"

  poetry run python -m "$MODULE" \
    --train        "$FINETUNE_ON" \
    --eval         "$EVAL_ON" \
    --labels       "$LABELS" \
    --output       "${CKPT_DIR}/${STEM}" \
    --push-to-hub \
    --hub-model-id "$HUB_ID" \
    --results-csv  "$RESULTS_CSV" \
    --dataset-name "$DATASET_STEM" \
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

# ─── Step 1: Export datasets ──────────────────────────────────────────────────
echo "========================================================"
echo "STEP 1 — Exporting datasets to JSONL"
echo "========================================================"

poetry run python -m mdg.scripts.prepare_finetune_data \
  --datasets \
    "namesarnav/causalbench:code" \
    "namesarnav/causalbench:math" \
    "namesarnav/causalbench:text" \
    namesarnav/corr2cause \
    namesarnav/e-care \
    namesarnav/fincausal-task1 \
    namesarnav/natquest \
    namesarnav/Quriosity \
  --local-files \
    "namesarnav_counterbench:mdg/synthetic/data/counterbench_task2_v2.jsonl" \
    "namesarnav_ac-reason:mdg/synthetic/data/ac_reason_task2.jsonl" \
    "namesarnav_bbh-causal-judgement:mdg/synthetic/data/bbh_causal_judgement_task2.jsonl" \
  --output-dir "$DATA_DIR" || echo "[WARN] Export had errors — continuing"

# ─── Step 2: BERT ─────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "STEP 2 — Fine-tuning BERT (skips already-done)"
echo "========================================================"

run_finetune "mdg.models.bert" "bert-base-uncased" "YES,NO" "namesarnav_counterbench"
run_finetune "mdg.models.bert" "bert-base-uncased" "YES,NO" "namesarnav_ac-reason"
run_finetune "mdg.models.bert" "bert-base-uncased" "YES,NO" "namesarnav_bbh-causal-judgement"
run_finetune "mdg.models.bert" "bert-base-uncased" "YES,NO" "namesarnav_causalbench_code"
run_finetune "mdg.models.bert" "bert-base-uncased" "YES,NO" "namesarnav_causalbench_math"
run_finetune "mdg.models.bert" "bert-base-uncased" "YES,NO" "namesarnav_causalbench_text"
run_finetune "mdg.models.bert" "bert-base-uncased" "0,1"    "namesarnav_corr2cause"
run_finetune "mdg.models.bert" "bert-base-uncased" "0,1"    "namesarnav_e-care"
run_finetune "mdg.models.bert" "bert-base-uncased" "0,1"    "namesarnav_fincausal-task1"
run_finetune "mdg.models.bert" "bert-base-uncased" "YES,NO" "namesarnav_natquest"
run_finetune "mdg.models.bert" "bert-base-uncased" "YES,NO" "namesarnav_Quriosity"

# ─── Step 3: RoBERTa ──────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "STEP 3 — Fine-tuning RoBERTa"
echo "========================================================"

run_finetune "mdg.models.roberta" "roberta-base" "YES,NO" "namesarnav_counterbench"
run_finetune "mdg.models.roberta" "roberta-base" "YES,NO" "namesarnav_ac-reason"
run_finetune "mdg.models.roberta" "roberta-base" "YES,NO" "namesarnav_bbh-causal-judgement"
run_finetune "mdg.models.roberta" "roberta-base" "YES,NO" "namesarnav_causalbench_code"
run_finetune "mdg.models.roberta" "roberta-base" "YES,NO" "namesarnav_causalbench_math"
run_finetune "mdg.models.roberta" "roberta-base" "YES,NO" "namesarnav_causalbench_text"
run_finetune "mdg.models.roberta" "roberta-base" "0,1"    "namesarnav_corr2cause"
run_finetune "mdg.models.roberta" "roberta-base" "0,1"    "namesarnav_e-care"
run_finetune "mdg.models.roberta" "roberta-base" "0,1"    "namesarnav_fincausal-task1"
run_finetune "mdg.models.roberta" "roberta-base" "YES,NO" "namesarnav_natquest"
run_finetune "mdg.models.roberta" "roberta-base" "YES,NO" "namesarnav_Quriosity"

# ─── Step 4: T5 ───────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "STEP 4 — Fine-tuning T5"
echo "========================================================"

run_finetune "mdg.models.t5" "t5-base" "YES,NO" "namesarnav_counterbench"
run_finetune "mdg.models.t5" "t5-base" "YES,NO" "namesarnav_ac-reason"
run_finetune "mdg.models.t5" "t5-base" "YES,NO" "namesarnav_bbh-causal-judgement"
run_finetune "mdg.models.t5" "t5-base" "YES,NO" "namesarnav_causalbench_code"
run_finetune "mdg.models.t5" "t5-base" "YES,NO" "namesarnav_causalbench_math"
run_finetune "mdg.models.t5" "t5-base" "YES,NO" "namesarnav_causalbench_text"
run_finetune "mdg.models.t5" "t5-base" "0,1"    "namesarnav_corr2cause"
run_finetune "mdg.models.t5" "t5-base" "0,1"    "namesarnav_e-care"
run_finetune "mdg.models.t5" "t5-base" "0,1"    "namesarnav_fincausal-task1"
run_finetune "mdg.models.t5" "t5-base" "YES,NO" "namesarnav_natquest"
run_finetune "mdg.models.t5" "t5-base" "YES,NO" "namesarnav_Quriosity"

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  SUMMARY"
echo "========================================================"
echo "  Passed  : ${#PASSED[@]}"
for job in "${PASSED[@]}";   do echo "    OK   $job"; done
echo "  Skipped : ${#SKIPPED[@]}"
for job in "${SKIPPED[@]}";  do echo "    SKIP $job"; done
echo "  Failed  : ${#FAILED[@]}"
for job in "${FAILED[@]}";   do echo "    FAIL $job"; done
echo ""
echo "  Train results CSV → $RESULTS_CSV"

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "Some jobs failed. Logs in $LOG_DIR"
  exit 1
else
  echo "All jobs done."
  exit 0
fi
