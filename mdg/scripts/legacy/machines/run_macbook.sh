#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  MacBook M4 Pro (24GB unified memory)                       ║
# ║  Role: Phase 1 (fine-tuning ALL models) +                   ║
# ║        Phase 2 (medium + slow attack recipes)               ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Usage (from repo root /Volumes/Github/mdg):
#   caffeinate -i bash machines/run_macbook.sh
#
# The caffeinate wrapper prevents macOS from sleeping mid-run.
# Keep the MacBook plugged in.

set -uo pipefail

export PYTORCH_ENABLE_MPS_FALLBACK=1   # use MPS where supported, CPU elsewhere

cd "$(dirname "$0")/.."   # repo root

echo "=== MacBook pipeline starting ==="
echo "    PYTORCH_ENABLE_MPS_FALLBACK=1"
echo "    $(date)"

# ─── Phase 1: Fine-tune all models ────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  PHASE 1 — Fine-tuning (BERT + RoBERTa + T5) ║"
echo "╚══════════════════════════════════════════════╝"
bash mdg/scripts/finetune_all.sh || echo "[WARN] finetune_all had errors — continuing"

# ─── Phase 2: Medium + slow attack recipes ────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  PHASE 2 — Attacks (medium + slow recipes)   ║"
echo "╚══════════════════════════════════════════════╝"

DATA_DIR="mdg/finetune/data"
CKPT_DIR="mdg/finetune/checkpoints"
ATTACK_OUT="mdg/adv_attack/results"
RESULTS_CSV="mdg/adv_attack/attack_results_macbook.csv"
NUM_EXAMPLES=-1

# Medium + slow recipes (leave fast ones to the 1050 Ti)
RECIPES="TextFoolerJin2019 A2TYoo2021 BAEGarg2019 BERTAttackLi2020 CLARE2020 IGAWang2019 PSOZang2020 Kuleshov2017 GeneticAlgorithmAlzantot2018 FasterGeneticAlgorithmJia2019 InputReductionFeng2018"

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

MODELS=("bert-base-uncased" "roberta-base" "t5-base")

FAILED=()

for MODEL_ID in "${MODELS[@]}"; do
  for ENTRY in "${DATASETS[@]}"; do
    STEM="${ENTRY%%:*}"
    LABELS="${ENTRY##*:}"
    DATASET_STEM="${STEM#namesarnav_}"
    CKPT="${CKPT_DIR}/${STEM}/${MODEL_ID}"

    echo ""
    echo "── $MODEL_ID × $DATASET_STEM (medium+slow recipes) ──"

    if [ ! -d "$CKPT" ] || [ ! -f "${CKPT}/config.json" ]; then
      echo "  [SKIP] No checkpoint yet: $CKPT"
      continue
    fi

    TRAIN_FILE="${DATA_DIR}/${STEM}__train.jsonl"
    TEST_FILE="${DATA_DIR}/${STEM}__test.jsonl"
    [ ! -f "$TRAIN_FILE" ] && { echo "  [SKIP] Missing $TRAIN_FILE"; continue; }
    [ ! -f "$TEST_FILE" ]  && { echo "  [SKIP] Missing $TEST_FILE";  continue; }

    TRAIN_COUNT=$(wc -l < "$TRAIN_FILE")
    TEST_COUNT=$(wc -l < "$TEST_FILE")
    ATTACK_FILE="$TEST_FILE"
    [ "$TEST_COUNT" -gt "$TRAIN_COUNT" ] && ATTACK_FILE="$TRAIN_FILE"

    IFS=',' read -ra LABEL_ARR <<< "$LABELS"

    poetry run python -m mdg.adv_attack.attack \
      --model        "$CKPT" \
      --dataset      "$ATTACK_FILE" \
      --output-dir   "$ATTACK_OUT" \
      --label-space  ${LABEL_ARR[*]} \
      --num-examples "$NUM_EXAMPLES" \
      --recipes      $RECIPES \
      --results-csv  "$RESULTS_CSV" \
      --model-name   "$MODEL_ID" \
      --dataset-name "$DATASET_STEM" \
      || FAILED+=("${DATASET_STEM}/${MODEL_ID}")
  done
done

# ─── Phase 3: Consolidate ─────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  PHASE 3 — Consolidating results            ║"
echo "╚══════════════════════════════════════════════╝"
poetry run python -m mdg.scripts.consolidate_results || echo "[WARN] consolidation failed"

echo ""
echo "=== MacBook pipeline done: $(date) ==="
[ "${#FAILED[@]}" -gt 0 ] && echo "Failed: ${FAILED[*]}"
