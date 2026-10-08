#!/usr/bin/env bash
# Build <stem>__all.jsonl for every dataset: train + test + validation
# concatenated, so attacks cover every data point rather than one split.
#
#   bash mdg/hpc/02_build_all_splits.sh
#
# Rebuilds any file whose parts are newer. These are large (corr2cause alone is
# ~208k rows) and are derived data, so they are not committed.
set -euo pipefail

HPC_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$HPC_DIR/env.sh"
source "$HPC_DIR/matrix.sh"
DATA="$PROJECT/mdg/finetune/data"

total=0
for entry in "${DATASETS[@]}"; do
  IFS='|' read -r STEM _ <<< "$entry"
  out="$DATA/${STEM}__all.jsonl"
  parts=()
  for split in train validation test; do
    [ -f "$DATA/${STEM}__${split}.jsonl" ] && parts+=("$DATA/${STEM}__${split}.jsonl")
  done
  if [ "${#parts[@]}" -eq 0 ]; then
    echo "  [skip] $STEM — no split files"
    continue
  fi
  # Rebuild only when a part is newer than the output.
  newest=0
  for p in "${parts[@]}"; do
    [ "$p" -nt "$out" ] && newest=1
  done
  if [ -f "$out" ] && [ "$newest" -eq 0 ]; then
    n=$(wc -l < "$out" | tr -d ' ')
    echo "  [ok]   ${STEM}__all.jsonl — $n rows (current)"
    total=$(( total + n ))
    continue
  fi
  cat "${parts[@]}" > "$out"
  n=$(wc -l < "$out" | tr -d ' ')
  echo "  [built] ${STEM}__all.jsonl — $n rows from ${#parts[@]} split(s)"
  total=$(( total + n ))
done

echo ""
echo "  total data points: $total"
echo "  example-attacks  : $(( total * 18 * 2 )) (18 recipes x 2 models)"
