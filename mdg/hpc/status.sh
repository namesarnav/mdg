#!/usr/bin/env bash
# What is finished and what is left, across the whole grid.
#
#   bash mdg/hpc/status.sh
#
# Counts per (model, dataset) how many recipe summaries exist. Use it between
# sessions to see progress and to decide what to run next.
set -uo pipefail

HPC_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$HPC_DIR/env.sh"
source "$HPC_DIR/matrix.sh"
cd "$PROJECT" || exit 1

RESULTS="mdg/adv_attack/results"
CKPTS="mdg/finetune/checkpoints"

# Recipe names, read from the attack module so the two never disagree.
# Count only the default (English) list — stop at MULTILINGUAL_RECIPES, which
# is opt-in via --include-multilingual.
N_RECIPES=$(awk '/^MULTILINGUAL_RECIPES/{exit} /^\s*\("[A-Za-z0-9]+",/{n++} END{print n+0}' mdg/adv_attack/attack.py)
[ "${N_RECIPES:-0}" -eq 0 ] && N_RECIPES=18

printf "\n%-28s %-16s %-8s %s\n" "DATASET" "MODEL" "TRAINED" "RECIPES DONE"
printf -- "---------------------------------------------------------------------\n"

total_done=0
total_cells=0
for ((i=0; i<N_JOBS; i++)); do
  resolve_task "$i"
  trained="no"
  [ -f "$CKPTS/$STEM/$MODEL_NAME/config.json" ] && trained="yes"
  done_n=$(ls -1 "$RESULTS/$STEM/$MODEL_NAME/"*_summary.json 2>/dev/null | wc -l | tr -d ' ')
  total_done=$(( total_done + done_n ))
  total_cells=$(( total_cells + N_RECIPES ))
  bar_n=$(( done_n * 20 / N_RECIPES ))
  bar=$(printf '%*s' "$bar_n" '' | tr ' ' '#')
  printf "%-28s %-16s %-8s %2d/%2d %s\n" \
    "$DATASET_STEM" "$MODEL_NAME" "$trained" "$done_n" "$N_RECIPES" "$bar"
done

printf -- "---------------------------------------------------------------------\n"
pct=0
[ "$total_cells" -gt 0 ] && pct=$(( total_done * 100 / total_cells ))
echo "  $total_done / $total_cells recipe-runs complete  (${pct}%)"
echo "  results: $PROJECT/$RESULTS"
echo ""
