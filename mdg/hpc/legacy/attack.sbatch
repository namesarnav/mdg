#!/bin/bash
# NOTE: Torch has no default GPU partition — submit_all.sh passes --partition
# from $PARTITION (env.sh). Submitting by hand? add: -p l40s
#SBATCH --job-name=mdg-attack
#SBATCH --array=0-21
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/%u/mdg-logs/attack-%A_%a.out
#SBATCH --error=/scratch/%u/mdg-logs/attack-%A_%a.out
#
# Run EVERY TextAttack recipe against one (model, dataset) pair per array task.
# 2 models × 11 datasets = 22 tasks, 18 recipes inside each.
#
#   sbatch mdg/hpc/attack.sbatch                  # EVERY example, no query cap
#   NUM_EXAMPLES=500 sbatch mdg/hpc/attack.sbatch # cap examples if time is short
#   QUERY_BUDGET=2000 sbatch mdg/hpc/attack.sbatch # cap queries per example
#
# Per recipe this writes:
#   mdg/adv_attack/results/<stem>/<model>/<recipe>.jsonl         — every attacked
#       example: ground truth, original text/pred, perturbed text/pred, queries
#   mdg/adv_attack/results/<stem>/<model>/<recipe>_summary.json  — ASR + F1
# and appends one row per recipe to mdg/adv_attack/attack_results.csv.

set -uo pipefail
module purge

# Submit from the repo root, so SLURM_SUBMIT_DIR is the project directory.
HPC_DIR="${SLURM_SUBMIT_DIR:-$PWD}/mdg/hpc"
mkdir -p "/scratch/$USER/mdg-logs"

# All the real work lives in run_one.sh, shared with run_parallel.sh.
bash "$HPC_DIR/run_one.sh" attack "$SLURM_ARRAY_TASK_ID"
