#!/usr/bin/env bash
# Every dataset, smallest test split first, so something finishes early and
# problems surface on 330 rows instead of 13,500.
#
#   caffeinate -i bash mac-scripts/all.sh
#
# One dataset failing does not stop the rest. Re-running skips finished work.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ORDER=(counterbench ac-reason bbh-causal-judgement corr2cause causalbench-code
       e-care causalbench-math fincausal-task1 causalbench-text natquest quriosity)

FAILED=()
for name in "${ORDER[@]}"; do
  bash "$ROOT/mac-scripts/$name.sh" || FAILED+=("$name")
done

echo "══════════════════════════════════════════════════════════════"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "  all 11 datasets completed"
else
  echo "  ${#FAILED[@]} failed: ${FAILED[*]}"
fi
python "$ROOT/mdg/hpc/status.py"
