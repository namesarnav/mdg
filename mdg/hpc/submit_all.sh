#!/usr/bin/env bash
# Submit the whole pipeline with dependencies, from the repo root:
#
#   cd /scratch/$USER/mdg && bash mdg/hpc/submit_all.sh
#
#   train (22 array tasks, GPU)
#     └─ afterok ─> attack (22 array tasks, GPU)
#                     └─ afterany ─> consolidate (CPU)
#
# attack depends on afterok so it never runs against missing checkpoints;
# consolidate uses afterany so you still get a CSV of whatever finished.
set -euo pipefail

HPC_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$HPC_DIR/matrix.sh"
LAST=$(( N_JOBS - 1 ))

echo "Grid: $N_MODELS models × $N_DATASETS datasets = $N_JOBS tasks per stage"

TRAIN_ID=$(sbatch --parsable --array=0-$LAST "$HPC_DIR/train.sbatch")
echo "  train       → job $TRAIN_ID"

ATTACK_ID=$(sbatch --parsable --array=0-$LAST \
  --dependency=afterok:"$TRAIN_ID" "$HPC_DIR/attack.sbatch")
echo "  attack      → job $ATTACK_ID  (after train)"

CONS_ID=$(sbatch --parsable \
  --dependency=afterany:"$ATTACK_ID" "$HPC_DIR/consolidate.sbatch")
echo "  consolidate → job $CONS_ID  (after attack)"

echo
echo "Watch:   squeue -u $USER"
echo "Logs:    tail -f /scratch/$USER/mdg-logs/train-${TRAIN_ID}_0.out"
echo "Cancel:  scancel $TRAIN_ID $ATTACK_ID $CONS_ID"
