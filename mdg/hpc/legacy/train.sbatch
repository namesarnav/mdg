#!/bin/bash
# NOTE: Torch has no default GPU partition — submit_all.sh passes --partition
# from $PARTITION (env.sh). Submitting by hand? add: -p l40s
#SBATCH --job-name=mdg-train
#SBATCH --array=0-21
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
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
mkdir -p "/scratch/$USER/mdg-logs"

# All the real work lives in run_one.sh, shared with run_parallel.sh.
bash "$HPC_DIR/run_one.sh" train "$SLURM_ARRAY_TASK_ID"
