#!/usr/bin/env bash
# Run the whole grid at once with GNU parallel, on ONE node.
#
#   bash mdg/hpc/run_parallel.sh train           # all 22 training runs
#   bash mdg/hpc/run_parallel.sh attack          # all 22 attack runs
#   bash mdg/hpc/run_parallel.sh all             # train, then attack, then consolidate
#   JOBS=4 bash mdg/hpc/run_parallel.sh attack   # force 4 concurrent tasks
#
# Each concurrent slot is pinned to its own GPU via CUDA_VISIBLE_DEVICES, so
# JOBS defaults to the number of GPUs visible on this node. Running more tasks
# than GPUs makes them share VRAM and usually ends in OOM — raise JOBS past the
# GPU count only for attack recipes that are CPU-bound in their search.
#
# NOTE: if you have Slurm, `submit_all.sh` is usually better — a job array
# spreads the same 22 tasks across MANY nodes, while this packs them onto one.
# Use this when you hold an interactive multi-GPU node, or on a box without Slurm.
#
# Progress is written to a --joblog, so a re-run resumes:
#   parallel --joblog <file> --resume-failed ...
set -uo pipefail

STAGE="${1:-all}"
HPC_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$HPC_DIR/env.sh"
source "$HPC_DIR/matrix.sh"

LOG_DIR="/scratch/$USER/mdg-logs"
mkdir -p "$LOG_DIR"

# ── GNU parallel ──────────────────────────────────────────────────────────────
if ! command -v parallel >/dev/null 2>&1; then
  module load parallel 2>/dev/null || true
fi
if ! command -v parallel >/dev/null 2>&1; then
  echo "[ERROR] GNU parallel not found. Try 'module avail parallel', or use the"
  echo "        xargs fallback:"
  echo "          seq 0 $(( N_JOBS - 1 )) | xargs -P 4 -I{} bash $HPC_DIR/run_one.sh $STAGE {}"
  exit 1
fi

# ── How many at once ──────────────────────────────────────────────────────────
detect_gpus() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l
  else
    echo 1
  fi
}
NGPU="$(detect_gpus)"
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
JOBS="${JOBS:-$NGPU}"

run_stage() {
  local stage="$1"
  local joblog="$LOG_DIR/parallel-${stage}.joblog"
  echo ""
  echo "========================================================"
  echo "  STAGE   : $stage"
  echo "  TASKS   : $N_JOBS  ($N_MODELS models × $N_DATASETS datasets)"
  echo "  PARALLEL: $JOBS at a time across $NGPU GPU(s)"
  echo "  JOBLOG  : $joblog"
  echo "========================================================"

  # {%} is parallel's 1-based slot number → pin slot N to GPU N-1.
  # --halt never: one failing pair must not kill the rest of the grid.
  seq 0 $(( N_JOBS - 1 )) | parallel \
    --jobs "$JOBS" \
    --joblog "$joblog" \
    --halt never \
    --line-buffer \
    --tagstring "[${stage} task {}]" \
    "CUDA_VISIBLE_DEVICES=\$(( ({%} - 1) % $NGPU )) bash '$HPC_DIR/run_one.sh' $stage {} \
       > '$LOG_DIR/${stage}-task{}.log' 2>&1"

  echo "  Finished $stage. Failures (exit != 0):"
  # joblog columns: Seq Host Starttime Runtime Send Receive Exitval Signal Command
  awk 'NR>1 && $7 != 0 {print "    seq " $1 " → exit " $7}' "$joblog" || true
  echo "  Re-run just the failures with:"
  echo "    parallel --joblog $joblog --resume-failed --jobs $JOBS ..."
}

case "$STAGE" in
  train)  run_stage train ;;
  attack) run_stage attack ;;
  all)
    run_stage train
    run_stage attack
    echo ""
    echo "[consolidate]"
    cd "$PROJECT" && in_container "python -m mdg.scripts.consolidate_results"
    ;;
  *)
    echo "usage: $0 <train|attack|all>"
    exit 2
    ;;
esac

echo ""
echo "Logs: $LOG_DIR/${STAGE}-task*.log"
