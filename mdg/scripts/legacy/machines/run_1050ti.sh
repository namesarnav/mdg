#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  Windows/Linux machine — GTX 1050 Ti (4GB VRAM)            ║
# ║  Role: Phase 2 only — FAST attack recipes                   ║
# ║  Models pulled directly from HuggingFace Hub               ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Prerequisites:
#   1. Fine-tuning must be done first (run_macbook.sh or run_colab.sh)
#   2. Models must be pushed to HuggingFace Hub (finetune_all.sh does this)
#   3. Clone the repo and install dependencies:
#        git clone https://github.com/YOUR_USERNAME/mdg.git
#        cd mdg
#        pip install poetry && poetry install
#
# Usage (from repo root):
#   bash machines/run_1050ti.sh

set -uo pipefail

cd "$(dirname "$0")/.."   # repo root

echo "=== 1050 Ti attack pipeline starting ==="
echo "    Fast recipes only — models pulled from HuggingFace Hub"
echo "    $(date)"

DATA_DIR="mdg/finetune/data"
ATTACK_OUT="mdg/adv_attack/results"
RESULTS_CSV="mdg/adv_attack/attack_results_1050ti.csv"
NUM_EXAMPLES=200

# Fast recipes only (character-level / low query count)
RECIPES="Pruthi2019 DeepWordBugGao2018 BadCharacters2021 TextBuggerLi2018 PWWSRen2019 CheckList2020 MorpheusTan2020"

# Models on HuggingFace Hub — pulled automatically by from_pretrained()
# Format: "hub_model_id:model_short_name"
MODELS=(
  "bert-base-uncased"
  "roberta-base"
  "t5-base"
)

DATASETS=(
  "namesarnav_counterbench:YES,NO"
  "namesarnav_ac-reason:YES,NO"
  "namesarnav_bbh-causal-judgement:YES,NO"
  "namesarnav_causalbench_code:YES,NO"
  "namesarnav_causalbench_math:YES,NO"
  "namesarnav_causalbench_text:YES,NO"
  "namesarnav_corr2cause:0,1"
  "namesarnav_e-care:0,1"
  "namesarnav_fincausal-task1:0,1"
  "namesarnav_natquest:YES,NO"
  "namesarnav_Quriosity:YES,NO"
)

FAILED=()
SKIPPED=()

for MODEL_ID in "${MODELS[@]}"; do
  for ENTRY in "${DATASETS[@]}"; do
    STEM="${ENTRY%%:*}"
    LABELS="${ENTRY##*:}"
    DATASET_STEM="${STEM#namesarnav_}"

    # HuggingFace Hub model ID
    HUB_MODEL="namesarnav/${DATASET_STEM}-${MODEL_ID}"

    echo ""
    echo "── $MODEL_ID × $DATASET_STEM (fast recipes) ──"
    echo "   Model: $HUB_MODEL"

    TRAIN_FILE="${DATA_DIR}/${STEM}__train.jsonl"
    TEST_FILE="${DATA_DIR}/${STEM}__test.jsonl"

    if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$TEST_FILE" ]; then
      echo "  [SKIP] Missing data files — run prepare_finetune_data.py first"
      SKIPPED+=("${DATASET_STEM}/${MODEL_ID}")
      continue
    fi

    # Use smaller split for attacks
    TRAIN_COUNT=$(wc -l < "$TRAIN_FILE")
    TEST_COUNT=$(wc -l < "$TEST_FILE")
    ATTACK_FILE="$TEST_FILE"
    [ "$TEST_COUNT" -gt "$TRAIN_COUNT" ] && ATTACK_FILE="$TRAIN_FILE"

    # Skip if already done (all_summaries.json exists for these recipes)
    DONE_MARKER="${ATTACK_OUT}/${STEM}/${MODEL_ID}/fast_done.marker"
    if [ -f "$DONE_MARKER" ]; then
      echo "  [SKIP] Fast recipes already done"
      SKIPPED+=("${DATASET_STEM}/${MODEL_ID}")
      continue
    fi

    IFS=',' read -ra LABEL_ARR <<< "$LABELS"

    poetry run python -m mdg.adv_attack.attack \
      --model        "$HUB_MODEL" \
      --dataset      "$ATTACK_FILE" \
      --output-dir   "$ATTACK_OUT" \
      --label-space  ${LABEL_ARR[*]} \
      --num-examples "$NUM_EXAMPLES" \
      --recipes      $RECIPES \
      --results-csv  "$RESULTS_CSV" \
      --model-name   "$MODEL_ID" \
      --dataset-name "$DATASET_STEM" \
      && touch "$DONE_MARKER" \
      || FAILED+=("${DATASET_STEM}/${MODEL_ID}")
  done
done

echo ""
echo "=== 1050 Ti done: $(date) ==="
echo "  Results CSV → $RESULTS_CSV"
echo "  Skipped : ${#SKIPPED[@]}"
echo "  Failed  : ${#FAILED[@]}"
[ "${#FAILED[@]}" -gt 0 ] && echo "  Failed jobs: ${FAILED[*]}"
echo ""
echo "  Copy $RESULTS_CSV back to your MacBook and re-run:"
echo "    poetry run python -m mdg.scripts.consolidate_results"
