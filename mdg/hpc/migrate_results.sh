#!/usr/bin/env bash
# Move older attack results into the layout the current scripts expect, so they
# count as done and are not recomputed.
#
#   bash mdg/hpc/migrate_results.sh          # dry run, shows what would move
#   bash mdg/hpc/migrate_results.sh --apply  # actually move
#
# Old (nested):  results/<ds>/<model>/<recipe>/namesarnav_<ds>__test/<model>/<recipe>.jsonl
# New (flat):    results/namesarnav_<ds>__test/<model>/<recipe>.jsonl
#
# A recipe is migrated only when BOTH its summary and its perturbed .jsonl
# exist and the jsonl is non-empty — a summary on its own would mark the recipe
# finished while leaving no perturbed data, so those are left to re-run.
set -uo pipefail

APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

HPC_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$HPC_DIR/env.sh"
RESULTS="$PROJECT/mdg/adv_attack/results"
cd "$RESULTS" || { echo "no results dir at $RESULTS"; exit 1; }

moved=0; skipped=0; incomplete=0

# Any path containing an inner "namesarnav_*__test/<model>/" segment is old-style.
while IFS= read -r src; do
  # src = ./<ds>/<model>/<recipe>/namesarnav_X__test/<model>/<recipe>.jsonl
  inner="${src#./}"
  rel="${inner#*/*/*/}"            # namesarnav_X__test/<model>/<recipe>.jsonl
  case "$rel" in namesarnav_*__test/*) ;; *) continue ;; esac

  sum_src="${src%.jsonl}_summary.json"
  dst="$RESULTS/$rel"
  dst_sum="${dst%.jsonl}_summary.json"

  if [ ! -s "$src" ]; then
    echo "  [empty]   $rel"; incomplete=$(( incomplete + 1 )); continue
  fi
  if [ ! -f "$sum_src" ]; then
    echo "  [no sum]  $rel"; incomplete=$(( incomplete + 1 )); continue
  fi
  if [ -f "$dst" ]; then
    echo "  [exists]  $rel"; skipped=$(( skipped + 1 )); continue
  fi

  echo "  [move]    $rel  ($(wc -l < "$src" | tr -d ' ') rows)"
  if [ "$APPLY" = "1" ]; then
    mkdir -p "$(dirname "$dst")"
    cp -p "$src" "$dst"
    cp -p "$sum_src" "$dst_sum"
  fi
  moved=$(( moved + 1 ))
done < <(find . -mindepth 4 -name "*.jsonl")

echo ""
echo "  to migrate : $moved"
echo "  already there: $skipped"
echo "  incomplete (will re-run): $incomplete"
[ "$APPLY" = "0" ] && echo "" && echo "  dry run — re-run with --apply to move them"
