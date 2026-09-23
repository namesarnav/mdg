#!/usr/bin/env bash
# Run TextAttack adversarial attacks on all model × dataset × recipe combinations.
#
# For each model and dataset, uses the SMALLER split (the one NOT used for training).
# Results (perturbed JSONL + per-recipe CSV) go to mdg/adv_attack/results/.
# All per-recipe rows stream into mdg/adv_attack/attack_results.csv as they complete.
#
# Usage: bash mdg/scripts/attack_all.sh
# Prerequisite: finetune_all.sh must have run first (checkpoints must exist).

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
ATTACK_OUT="mdg/adv_attack/results"
RESULTS_CSV="mdg/adv_attack/attack_results.csv"
NUM_EXAMPLES=200   # examples per recipe per run; raise if you have time

FAILED=()
PASSED=()
SKIPPED=()

# ─── Helper ───────────────────────────────────────────────────────────────────
# run_attack MODEL_ID STEM LABELS
# MODEL_ID: bert-base-uncased | roberta-base | t5-base
# STEM    : namesarnav_counterbench (etc.)
# LABELS  : "YES,NO" | "0,1"
run_attack() {
  local MODEL_ID="$1"
  local STEM="$2"
  local LABELS="$3"

  local TRAIN_FILE="${DATA_DIR}/${STEM}__train.jsonl"
  local TEST_FILE="${DATA_DIR}/${STEM}__test.jsonl"
  local DATASET_STEM="${STEM#namesarnav_}"

  # Checkpoint path uses the same layout as finetune_all.sh
  local CKPT="${CKPT_DIR}/${STEM}/${MODEL_ID}"

  # Done-marker: all_summaries.json written at end of run
  local DONE_MARKER="${ATTACK_OUT}/${STEM}/${MODEL_ID}/all_summaries.json"

  echo ""
  echo "────────────────────────────────────────────────────────"
  echo "  ATTACK  : $MODEL_ID  ×  $DATASET_STEM"
  echo "────────────────────────────────────────────────────────"

  if [ ! -d "$CKPT" ] || [ ! -f "${CKPT}/config.json" ]; then
    echo "  [SKIP] No checkpoint: $CKPT"
    SKIPPED+=("${DATASET_STEM}/${MODEL_ID} — no checkpoint")
    return
  fi

  if [ -f "$DONE_MARKER" ]; then
    echo "  [SKIP] Already attacked (all_summaries.json exists)"
    SKIPPED+=("${DATASET_STEM}/${MODEL_ID}")
    return
  fi

  if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$TEST_FILE" ]; then
    echo "  [SKIP] Missing data files"
    FAILED+=("${DATASET_STEM}/${MODEL_ID} — missing data")
    return
  fi

  # Use the SMALLER split for attacks (the one left out during training)
  TRAIN_COUNT=$(wc -l < "$TRAIN_FILE")
  TEST_COUNT=$(wc -l < "$TEST_FILE")
  if [ "$TRAIN_COUNT" -ge "$TEST_COUNT" ]; then
    # Training used train, so attack on test (smaller)
    ATTACK_FILE="$TEST_FILE"
  else
    # Training used test (it was larger), so attack on train (smaller)
    ATTACK_FILE="$TRAIN_FILE"
  fi
  echo "  Attack file: $(basename $ATTACK_FILE)  (${TRAIN_COUNT} train / ${TEST_COUNT} test)"

  IFS=',' read -ra LABEL_ARR <<< "$LABELS"
  LABEL_SPACE="${LABEL_ARR[*]}"   # space-separated for --label-space

  poetry run python -m mdg.adv_attack.attack \
    --model      "$CKPT" \
    --dataset    "$ATTACK_FILE" \
    --output-dir "$ATTACK_OUT" \
    --label-space $LABEL_SPACE \
    --num-examples "$NUM_EXAMPLES" \
    --results-csv  "$RESULTS_CSV" \
    --model-name   "$MODEL_ID" \
    --dataset-name "$DATASET_STEM"

  local EXIT_CODE=$?
  if [ "$EXIT_CODE" -ne 0 ]; then
    echo "  [ERROR] Exit code $EXIT_CODE"
    FAILED+=("${DATASET_STEM}/${MODEL_ID} — exit $EXIT_CODE")
  else
    echo "  [OK] Results → ${ATTACK_OUT}/${STEM}/${MODEL_ID}/"
    PASSED+=("${DATASET_STEM}/${MODEL_ID}")
  fi
}

# ─── BERT attacks ─────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "BERT attacks"
echo "========================================================"

run_attack "bert-base-uncased" "namesarnav_counterbench"        "YES,NO"
run_attack "bert-base-uncased" "namesarnav_ac-reason"           "YES,NO"
run_attack "bert-base-uncased" "namesarnav_bbh-causal-judgement" "YES,NO"
run_attack "bert-base-uncased" "namesarnav_causalbench_code"    "YES,NO"
run_attack "bert-base-uncased" "namesarnav_causalbench_math"    "YES,NO"
run_attack "bert-base-uncased" "namesarnav_causalbench_text"    "YES,NO"
run_attack "bert-base-uncased" "namesarnav_corr2cause"          "0,1"
run_attack "bert-base-uncased" "namesarnav_e-care"              "0,1"
run_attack "bert-base-uncased" "namesarnav_fincausal-task1"     "0,1"
run_attack "bert-base-uncased" "namesarnav_natquest"            "YES,NO"
run_attack "bert-base-uncased" "namesarnav_Quriosity"           "YES,NO"

# ─── RoBERTa attacks ──────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "RoBERTa attacks"
echo "========================================================"

run_attack "roberta-base" "namesarnav_counterbench"        "YES,NO"
run_attack "roberta-base" "namesarnav_ac-reason"           "YES,NO"
run_attack "roberta-base" "namesarnav_bbh-causal-judgement" "YES,NO"
run_attack "roberta-base" "namesarnav_causalbench_code"    "YES,NO"
run_attack "roberta-base" "namesarnav_causalbench_math"    "YES,NO"
run_attack "roberta-base" "namesarnav_causalbench_text"    "YES,NO"
run_attack "roberta-base" "namesarnav_corr2cause"          "0,1"
run_attack "roberta-base" "namesarnav_e-care"              "0,1"
run_attack "roberta-base" "namesarnav_fincausal-task1"     "0,1"
run_attack "roberta-base" "namesarnav_natquest"            "YES,NO"
run_attack "roberta-base" "namesarnav_Quriosity"           "YES,NO"

# ─── T5 attacks ───────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "T5 attacks"
echo "========================================================"

run_attack "t5-base" "namesarnav_counterbench"        "YES,NO"
run_attack "t5-base" "namesarnav_ac-reason"           "YES,NO"
run_attack "t5-base" "namesarnav_bbh-causal-judgement" "YES,NO"
run_attack "t5-base" "namesarnav_causalbench_code"    "YES,NO"
run_attack "t5-base" "namesarnav_causalbench_math"    "YES,NO"
run_attack "t5-base" "namesarnav_causalbench_text"    "YES,NO"
run_attack "t5-base" "namesarnav_corr2cause"          "0,1"
run_attack "t5-base" "namesarnav_e-care"              "0,1"
run_attack "t5-base" "namesarnav_fincausal-task1"     "0,1"
run_attack "t5-base" "namesarnav_natquest"            "YES,NO"
run_attack "t5-base" "namesarnav_Quriosity"           "YES,NO"

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  ATTACK SUMMARY"
echo "========================================================"
echo "  Passed  : ${#PASSED[@]}"
for job in "${PASSED[@]}";   do echo "    OK   $job"; done
echo "  Skipped : ${#SKIPPED[@]}"
for job in "${SKIPPED[@]}";  do echo "    SKIP $job"; done
echo "  Failed  : ${#FAILED[@]}"
for job in "${FAILED[@]}";   do echo "    FAIL $job"; done
echo ""
echo "  Attack results CSV → $RESULTS_CSV"
echo "  Run consolidate_results.py to merge all CSVs."

if [ "${#FAILED[@]}" -gt 0 ]; then
  exit 1
else
  exit 0
fi
