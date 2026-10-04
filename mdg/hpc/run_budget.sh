#!/usr/bin/env bash
# Work through the grid for a fixed stretch of time, then stop cleanly.
# Built for capped sessions (Kaggle ~12h, Colab, a rented GPU by the hour).
#
#   bash mdg/hpc/run_budget.sh 11          # work for 11 hours, then stop
#   bash mdg/hpc/run_budget.sh 11 train    # training only
#   bash mdg/hpc/run_budget.sh 5 attack    # attacks only
#
# Order is cheapest-first — smallest dataset, fastest recipe — so a short
# session still finishes whole cells instead of stalling on natquest.
# Everything already done is skipped, so just run it again next session.
set -uo pipefail

HOURS="${1:-11}"
ONLY="${2:-both}"
HPC_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$HPC_DIR/env.sh"
source "$HPC_DIR/matrix.sh"
cd "$PROJECT" || exit 1

# awk, not bc: bc is not installed everywhere. Accepts fractional hours.
SECONDS_BUDGET=$(awk -v h="$HOURS" 'BEGIN{printf "%d", h*3600}')
DEADLINE=$(( $(date +%s) + SECONDS_BUDGET ))
left() { echo $(( DEADLINE - $(date +%s) )); }
human() { printf '%dh%02dm' $(( $1 / 3600 )) $(( ($1 % 3600) / 60 )); }

echo "========================================================"
echo "  BUDGET  : ${HOURS}h   (stops cleanly, resume by re-running)"
echo "  STAGES  : $ONLY"
echo "========================================================"

# ── Task order: smallest attacked split first ─────────────────────────────────
ORDER=()
for ((i=0; i<N_JOBS; i++)); do
  resolve_task "$i"
  tr_n=$(wc -l < "mdg/finetune/data/${STEM}__train.jsonl" 2>/dev/null || echo 0)
  te_n=$(wc -l < "mdg/finetune/data/${STEM}__test.jsonl"  2>/dev/null || echo 0)
  small=$(( tr_n < te_n ? tr_n : te_n ))
  ORDER+=("$small $i")
done
TASKS=$(printf '%s\n' "${ORDER[@]}" | sort -n | awk '{print $2}')

# ── Stage 1: training ─────────────────────────────────────────────────────────
if [ "$ONLY" = "both" ] || [ "$ONLY" = "train" ]; then
  for i in $TASKS; do
    [ "$(left)" -le 0 ] && { echo "[BUDGET] time up during training"; break; }
    resolve_task "$i"
    [ -f "mdg/finetune/checkpoints/$STEM/$MODEL_NAME/config.json" ] && continue
    echo ""
    echo "[$(human $(left)) left] train task $i — $DATASET_STEM × $MODEL_NAME"
    bash "$HPC_DIR/run_one.sh" train "$i"
  done
fi

# ── Stage 2: attacks, one recipe at a time ────────────────────────────────────
if [ "$ONLY" = "both" ] || [ "$ONLY" = "attack" ]; then
  # Recipe names in the module's own order (roughly fastest → slowest).
  # while-read, not mapfile: mapfile needs bash 4+ (macOS ships 3.2).
  RECIPE_LIST=()
  while IFS= read -r r; do
    [ -n "$r" ] && RECIPE_LIST+=("$r")
  done < <(awk '/^MULTILINGUAL_RECIPES/{exit}
                /^[[:space:]]*\("/{ if (match($0, /"[^"]+"/)) print substr($0, RSTART+1, RLENGTH-2) }' \
                mdg/adv_attack/attack.py)
  echo ""
  echo "  ${#RECIPE_LIST[@]} recipes per pair"

  for i in $TASKS; do
    resolve_task "$i"
    [ -f "mdg/finetune/checkpoints/$STEM/$MODEL_NAME/config.json" ] || continue
    for recipe in "${RECIPE_LIST[@]}"; do
      [ "$(left)" -le 0 ] && { echo ""; echo "[BUDGET] time up — stopping cleanly"; exit 0; }
      marker="mdg/adv_attack/results/${STEM}/${MODEL_NAME}/${recipe}_summary.json"
      [ -f "$marker" ] && continue
      echo ""
      echo "[$(human $(left)) left] attack $DATASET_STEM × $MODEL_NAME — $recipe"
      RECIPES="$recipe" bash "$HPC_DIR/run_one.sh" attack "$i"
    done
  done
fi

echo ""
echo "[DONE] budget window finished — run mdg/hpc/status.sh to see progress"
