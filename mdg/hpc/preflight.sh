#!/usr/bin/env bash
# Check every (model x dataset) task can actually load its weights BEFORE
# committing days of CPU to the attack grid.
#
#   bash mdg/hpc/preflight.sh
#
# For each task: is there a local checkpoint, or a Hub repo that really holds
# weights? A Hub repo created by an interrupted push looks fine in the UI but
# contains only .gitattributes, and attack.py would then die hours into the run
# with a 404 buried in one of hundreds of logs.
#
# Prints a READY_TASKS line you can paste straight into ONLY_TASKS.
set -uo pipefail

HPC_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$HPC_DIR/env.sh"
source "$HPC_DIR/matrix.sh"
cd "$PROJECT" || exit 1

LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT
for ((i=0; i<N_JOBS; i++)); do
  resolve_task "$i"
  echo "$i|$DATASET_STEM|$MODEL_NAME|mdg/finetune/checkpoints/$STEM/$MODEL_NAME|$HUB_ID"
done > "$LIST"

in_container "python mdg/hpc/preflight_check.py '$LIST'"
