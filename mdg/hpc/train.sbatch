#!/bin/bash
#SBATCH --job-name=mdg-train
#SBATCH --array=0-21
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/mdg-logs/train-%A_%a.out
#SBATCH --error=/scratch/%u/mdg-logs/train-%A_%a.out
#
# Fine-tune every model on every dataset — one array task per (model, dataset)
# pair — then push each trained model to the HuggingFace Hub.
#
#   sbatch mdg/hpc/train.sbatch
#
# The array range must be 0-(N_JOBS-1); print N_JOBS with:
#   source mdg/hpc/matrix.sh && echo $N_JOBS
# 2 models × 11 datasets = 22 tasks → 0-21.

set -uo pipefail
module purge

# Submit from the repo root, so SLURM_SUBMIT_DIR is the project directory.
HPC_DIR="${SLURM_SUBMIT_DIR:-$PWD}/mdg/hpc"
source "$HPC_DIR/env.sh"
source "$HPC_DIR/matrix.sh"
mkdir -p "/scratch/$USER/mdg-logs"

resolve_task "$SLURM_ARRAY_TASK_ID"

DATA_DIR="mdg/finetune/data"
CKPT_ROOT="mdg/finetune/checkpoints/${STEM}"
CKPT="${CKPT_ROOT}/${MODEL_NAME}"
RESULTS_CSV="mdg/finetune/train_results.csv"

TRAIN_FILE="${DATA_DIR}/${STEM}__train.jsonl"
TEST_FILE="${DATA_DIR}/${STEM}__test.jsonl"

echo "========================================================"
echo "  TASK    : $SLURM_ARRAY_TASK_ID / $(( N_JOBS - 1 ))"
echo "  MODEL   : $HF_MODEL  (ckpt dir: $MODEL_NAME)"
echo "  DATASET : $STEM   labels=$LABELS"
echo "  HUB     : $HUB_ID"
echo "  NODE    : $(hostname)"
echo "========================================================"

if [ -f "$PROJECT/$CKPT/config.json" ]; then
  echo "[SKIP] Checkpoint already exists: $CKPT"
  exit 0
fi

# The splits are committed to the repo. If they are missing, regenerate them
# once on the login node with: bash mdg/hpc/01_prepare_data.sh
if [ ! -f "$PROJECT/$TRAIN_FILE" ] || [ ! -f "$PROJECT/$TEST_FILE" ]; then
  echo "[FAIL] Missing split files for $STEM:"
  echo "         $TRAIN_FILE"
  echo "         $TEST_FILE"
  echo "       Run: bash mdg/hpc/01_prepare_data.sh"
  exit 1
fi

# Train on the larger split, evaluate on the smaller one.
TRAIN_COUNT=$(wc -l < "$PROJECT/$TRAIN_FILE")
TEST_COUNT=$(wc -l < "$PROJECT/$TEST_FILE")
if [ "$TRAIN_COUNT" -ge "$TEST_COUNT" ]; then
  FIT_ON="$TRAIN_FILE";  EVAL_ON="$TEST_FILE"
else
  FIT_ON="$TEST_FILE";   EVAL_ON="$TRAIN_FILE"
fi
echo "  fit=$(basename $FIT_ON) ($TRAIN_COUNT/$TEST_COUNT lines)  eval=$(basename $EVAL_ON)"

in_container "python -m $MODULE \
  --train  '$FIT_ON' \
  --eval   '$EVAL_ON' \
  --labels '$LABELS' \
  --model  '$HF_MODEL' \
  --output '$CKPT_ROOT' \
  $EXTRA_ARGS \
  --push-to-hub \
  --hub-model-id '$HUB_ID' \
  --results-csv  '$RESULTS_CSV' \
  --dataset-name '$DATASET_STEM'"

STATUS=$?
if [ "$STATUS" -eq 0 ]; then
  echo "[OK] $DATASET_STEM × $MODEL_NAME → https://huggingface.co/$HUB_ID"
else
  echo "[FAIL] exit $STATUS"
fi
exit "$STATUS"
