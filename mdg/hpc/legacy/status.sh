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

# ROW_SHARDS=k counts progress in units of 1/k of a dataset, matching how
# run_parallel.sh cells splits the work when several machines share a cell.
ROW_SHARDS="${ROW_SHARDS:-1}"
N_UNITS=$(( N_RECIPES * ROW_SHARDS ))

printf "\n%-28s %-16s %-8s %s\n" "DATASET" "MODEL" "TRAINED" "UNITS DONE"
printf -- "---------------------------------------------------------------------\n"

total_done=0
total_cells=0
for ((i=0; i<N_JOBS; i++)); do
  resolve_task "$i"
  trained="no"
  [ -f "$CKPTS/$STEM/$MODEL_NAME/config.json" ] && trained="yes"
  # A merged cell counts as all of its shards, so the total stays meaningful
  # after merge_shards.py has folded the per-shard files away.
  done_n=$(ls -1 "$RESULTS/$ATTACK_STEM/$MODEL_NAME/"*_summary.json 2>/dev/null \
           | awk -v k="$ROW_SHARDS" '/\.sh[0-9]+of[0-9]+_summary\.json$/{n++; next} {n+=k} END{print n+0}')
  [ "$done_n" -gt "$N_UNITS" ] && done_n="$N_UNITS"
  total_done=$(( total_done + done_n ))
  total_cells=$(( total_cells + N_UNITS ))
  bar_n=$(( done_n * 20 / N_UNITS ))
  bar=$(printf '%*s' "$bar_n" '' | tr ' ' '#')
  printf "%-28s %-16s %-8s %2d/%2d %s\n" \
    "$DATASET_STEM" "$MODEL_NAME" "$trained" "$done_n" "$N_UNITS" "$bar"
done

printf -- "---------------------------------------------------------------------\n"
pct=0
[ "$total_cells" -gt 0 ] && pct=$(( total_done * 100 / total_cells ))
echo "  $total_done / $total_cells attack units complete  (${pct}%)   [ROW_SHARDS=$ROW_SHARDS]"
echo "  results: $PROJECT/$RESULTS"
echo ""
