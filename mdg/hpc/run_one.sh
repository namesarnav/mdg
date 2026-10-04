#!/usr/bin/env bash
# Run ONE (model × dataset) task of a stage. The single source of truth for
# what a task does — train.sbatch, attack.sbatch and run_parallel.sh all call
# this, so the logic never diverges between the Slurm and GNU parallel paths.
#
#   bash mdg/hpc/run_one.sh train  0
#   bash mdg/hpc/run_one.sh attack 13
#
# Task ids run 0..N_JOBS-1 (see matrix.sh). Honours CUDA_VISIBLE_DEVICES, so a
# caller can pin each concurrent task to its own GPU.
set -uo pipefail

STAGE="${1:?usage: run_one.sh <train|attack> <task_id>}"
TASK_ID="${2:?usage: run_one.sh <train|attack> <task_id>}"

HPC_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$HPC_DIR/env.sh"
source "$HPC_DIR/matrix.sh"

resolve_task "$TASK_ID"

DATA_DIR="mdg/finetune/data"
TRAIN_FILE="${DATA_DIR}/${STEM}__train.jsonl"
TEST_FILE="${DATA_DIR}/${STEM}__test.jsonl"
CKPT_ROOT="mdg/finetune/checkpoints/${STEM}"
CKPT="${CKPT_ROOT}/${MODEL_NAME}"

echo "========================================================"
echo "  STAGE   : $STAGE   task $TASK_ID / $(( N_JOBS - 1 ))"
echo "  MODEL   : $HF_MODEL   DATASET: $STEM   labels=$LABELS"
echo "  GPU     : ${CUDA_VISIBLE_DEVICES:-all}   NODE: $(hostname)"
echo "========================================================"

if [ ! -f "$PROJECT/$TRAIN_FILE" ] || [ ! -f "$PROJECT/$TEST_FILE" ]; then
  echo "[FAIL] Missing splits for $STEM — run: bash mdg/hpc/01_prepare_data.sh"
  exit 1
fi

TRAIN_COUNT=$(wc -l < "$PROJECT/$TRAIN_FILE")
TEST_COUNT=$(wc -l < "$PROJECT/$TEST_FILE")

case "$STAGE" in

  train)
    if [ -f "$PROJECT/$CKPT/config.json" ]; then
      echo "[SKIP] Checkpoint exists: $CKPT"
      exit 0
    fi
    # Fit on the larger split, evaluate on the smaller one.
    if [ "$TRAIN_COUNT" -ge "$TEST_COUNT" ]; then
      FIT_ON="$TRAIN_FILE"; EVAL_ON="$TEST_FILE"
    else
      FIT_ON="$TEST_FILE";  EVAL_ON="$TRAIN_FILE"
    fi
    echo "  fit=$(basename $FIT_ON)  eval=$(basename $EVAL_ON)  ($TRAIN_COUNT/$TEST_COUNT)"

    in_container "python -m $MODULE \
      --train  '$FIT_ON' \
      --eval   '$EVAL_ON' \
      --labels '$LABELS' \
      --model  '$HF_MODEL' \
      --output '$CKPT_ROOT' \
      $EXTRA_ARGS \
      --push-to-hub \
      --hub-model-id '$HUB_ID' \
      --results-csv  'mdg/finetune/train_results.csv' \
      --dataset-name '$DATASET_STEM'"
    STATUS=$?
    [ "$STATUS" -eq 0 ] && echo "[OK] → https://huggingface.co/$HUB_ID"
    ;;

  attack)
    # Full split and unlimited queries by default. -1 = every example.
    NUM_EXAMPLES="${NUM_EXAMPLES:--1}"
    # Empty = no --query-budget flag = TextAttack searches without a cap.
    QUERY_BUDGET="${QUERY_BUDGET:-}"
    ATTACK_OUT="mdg/adv_attack/results"
    DONE_MARKER="$PROJECT/${ATTACK_OUT}/${STEM}/${MODEL_NAME}/all_summaries.json"

    if [ -f "$DONE_MARKER" ]; then
      echo "[SKIP] Already attacked: $DONE_MARKER"
      exit 0
    fi
    # Prefer the local checkpoint; fall back to the Hub copy from training.
    if [ -f "$PROJECT/$CKPT/config.json" ]; then
      MODEL_REF="$CKPT"
    else
      echo "[INFO] No local checkpoint — using Hub model $HUB_ID"
      MODEL_REF="$HUB_ID"
    fi
    # Attack the split NOT used for training.
    if [ "$TRAIN_COUNT" -ge "$TEST_COUNT" ]; then
      ATTACK_FILE="$TEST_FILE"
    else
      ATTACK_FILE="$TRAIN_FILE"
    fi
    BUDGET_ARG=""
    [ -n "$QUERY_BUDGET" ] && BUDGET_ARG="--query-budget $QUERY_BUDGET"
    echo "  attack split=$(basename $ATTACK_FILE)  examples=${NUM_EXAMPLES/-1/ALL}  budget=${QUERY_BUDGET:-unlimited}"

    in_container "python -m mdg.adv_attack.attack \
      --model        '$MODEL_REF' \
      --dataset      '$ATTACK_FILE' \
      --output-dir   '$ATTACK_OUT' \
      --label-space  ${LABELS//,/ } \
      --num-examples $NUM_EXAMPLES \
      $BUDGET_ARG \
      --results-csv  'mdg/adv_attack/attack_results.csv' \
      --model-name   '$MODEL_NAME' \
      --dataset-name '$DATASET_STEM'"
    STATUS=$?
    ;;

  *)
    echo "[FAIL] Unknown stage: $STAGE (expected train or attack)"
    exit 2
    ;;
esac

echo "[$([ ${STATUS:-1} -eq 0 ] && echo OK || echo FAIL)] $STAGE $DATASET_STEM × $MODEL_NAME (exit ${STATUS:-1})"
exit "${STATUS:-1}"
