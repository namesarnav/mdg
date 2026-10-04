#!/bin/bash
#SBATCH --job-name=mdg-attack
#SBATCH --array=0-21
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/mdg-logs/attack-%A_%a.out
#SBATCH --error=/scratch/%u/mdg-logs/attack-%A_%a.out
#
# Run EVERY TextAttack recipe against one (model, dataset) pair per array task.
# 2 models × 11 datasets = 22 tasks, 18 recipes inside each.
#
#   sbatch mdg/hpc/attack.sbatch                  # 200 examples per recipe
#   NUM_EXAMPLES=500 sbatch mdg/hpc/attack.sbatch # more examples, slower
#
# Per recipe this writes:
#   mdg/adv_attack/results/<stem>/<model>/<recipe>.jsonl         — every attacked
#       example: ground truth, original text/pred, perturbed text/pred, queries
#   mdg/adv_attack/results/<stem>/<model>/<recipe>_summary.json  — ASR + F1
# and appends one row per recipe to mdg/adv_attack/attack_results.csv.

set -uo pipefail
module purge

HPC_DIR="${SLURM_SUBMIT_DIR:-$PWD}/mdg/hpc"
source "$HPC_DIR/env.sh"
source "$HPC_DIR/matrix.sh"
mkdir -p "/scratch/$USER/mdg-logs"

resolve_task "$SLURM_ARRAY_TASK_ID"

# Examples attacked per recipe. Every recipe queries the model hundreds of times
# per example, so the full split is only realistic for the small datasets.
NUM_EXAMPLES="${NUM_EXAMPLES:-200}"
# Hard cap on model queries per example — stops PSO/genetic recipes from
# running for hours on a single long input. Raise or unset for fuller search.
QUERY_BUDGET="${QUERY_BUDGET:-2000}"

DATA_DIR="mdg/finetune/data"
CKPT="mdg/finetune/checkpoints/${STEM}/${MODEL_NAME}"
ATTACK_OUT="mdg/adv_attack/results"
RESULTS_CSV="mdg/adv_attack/attack_results.csv"
DONE_MARKER="$PROJECT/${ATTACK_OUT}/${STEM}/${MODEL_NAME}/all_summaries.json"

echo "========================================================"
echo "  TASK    : $SLURM_ARRAY_TASK_ID / $(( N_JOBS - 1 ))"
echo "  ATTACK  : $MODEL_NAME  ×  $DATASET_STEM"
echo "  EXAMPLES: $NUM_EXAMPLES   query budget: $QUERY_BUDGET"
echo "  NODE    : $(hostname)"
echo "========================================================"

# Prefer the local checkpoint; fall back to the Hub copy pushed by train.sbatch.
if [ -f "$PROJECT/$CKPT/config.json" ]; then
  MODEL_REF="$CKPT"
else
  echo "[INFO] No local checkpoint — using Hub model $HUB_ID"
  MODEL_REF="$HUB_ID"
fi

if [ -f "$DONE_MARKER" ]; then
  echo "[SKIP] Already attacked — $DONE_MARKER exists"
  exit 0
fi

# Attack the split NOT used for training (the smaller one).
TRAIN_FILE="${DATA_DIR}/${STEM}__train.jsonl"
TEST_FILE="${DATA_DIR}/${STEM}__test.jsonl"
TRAIN_COUNT=$(wc -l < "$PROJECT/$TRAIN_FILE")
TEST_COUNT=$(wc -l < "$PROJECT/$TEST_FILE")
if [ "$TRAIN_COUNT" -ge "$TEST_COUNT" ]; then
  ATTACK_FILE="$TEST_FILE"
else
  ATTACK_FILE="$TRAIN_FILE"
fi
echo "  attack split: $(basename $ATTACK_FILE)"

LABEL_SPACE="${LABELS//,/ }"

in_container "python -m mdg.adv_attack.attack \
  --model        '$MODEL_REF' \
  --dataset      '$ATTACK_FILE' \
  --output-dir   '$ATTACK_OUT' \
  --label-space  $LABEL_SPACE \
  --num-examples $NUM_EXAMPLES \
  --query-budget $QUERY_BUDGET \
  --results-csv  '$RESULTS_CSV' \
  --model-name   '$MODEL_NAME' \
  --dataset-name '$DATASET_STEM'"

STATUS=$?
echo "[$([ $STATUS -eq 0 ] && echo OK || echo FAIL)] $DATASET_STEM × $MODEL_NAME (exit $STATUS)"
exit "$STATUS"
