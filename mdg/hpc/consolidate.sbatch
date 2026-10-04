#!/bin/bash
#SBATCH --job-name=mdg-consolidate
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/mdg-logs/consolidate-%j.out
#SBATCH --error=/scratch/%u/mdg-logs/consolidate-%j.out
#
# Merge every per-run CSV into mdg/results/consolidated.csv — one row per
# (model × dataset × recipe) with train F1, clean F1, attacked F1 and ASR.
# CPU only, no GPU needed.
#
#   sbatch mdg/hpc/consolidate.sbatch

set -uo pipefail
module purge

HPC_DIR="${SLURM_SUBMIT_DIR:-$PWD}/mdg/hpc"
source "$HPC_DIR/env.sh"

in_container "python -m mdg.scripts.consolidate_results"
STATUS=$?

echo
echo "Outputs:"
echo "  $PROJECT/mdg/results/consolidated.csv"
echo "  $PROJECT/mdg/results/train_results.csv"
echo "  $PROJECT/mdg/results/attack_results.csv"
exit "$STATUS"
