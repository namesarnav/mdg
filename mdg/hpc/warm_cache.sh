#!/usr/bin/env bash
# Download every recipe's model and constraint files ONCE, serially, before the
# fleet starts.
#
#   bash mdg/hpc/warm_cache.sh
#
# Recipes pull large shared assets on first use — word embeddings, a universal
# sentence encoder, a 1GB "learning-to-write" language model, BERT for BAE.
# Launching 60 workers cold means 60 processes racing to download and unzip the
# same files into one cache directory: wasted bandwidth at best, a half-written
# cache at worst.
#
# Writes to a throwaway output directory, never to mdg/adv_attack/results, so a
# warm-up can never be mistaken for a finished cell.
set -uo pipefail

HPC_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$HPC_DIR/env.sh"
source "$HPC_DIR/matrix.sh"
cd "$PROJECT" || exit 1

# Smallest dataset with a trained model — the point is the download, not the attack.
resolve_task "${WARM_TASK:-0}"
DATA="mdg/finetune/data/${ATTACK_STEM}.jsonl"
[ -f "$DATA" ] || DATA="mdg/finetune/data/${STEM}__test.jsonl"
if [ ! -f "$DATA" ]; then
  echo "[FAIL] no dataset to warm with ($DATA). Run 02_build_all_splits.sh first."
  exit 1
fi
if [ -f "$PROJECT/mdg/finetune/checkpoints/$STEM/$MODEL_NAME/config.json" ]; then
  MODEL_REF="mdg/finetune/checkpoints/$STEM/$MODEL_NAME"
else
  MODEL_REF="$HUB_ID"
fi

TMP_OUT="$(mktemp -d "${TMPDIR:-/tmp}/mdg-warm-XXXXXX")"
trap 'rm -rf "$TMP_OUT"' EXIT

RECIPE_LIST=()
while IFS= read -r r; do
  [ -n "$r" ] && RECIPE_LIST+=("$r")
done < <(awk '/^MULTILINGUAL_RECIPES/{exit}
              /^[[:space:]]*\("/{ if (match($0, /"[^"]+"/)) print substr($0, RSTART+1, RLENGTH-2) }' \
              "$PROJECT/mdg/adv_attack/attack.py")

echo "========================================================"
echo "  Warming caches for ${#RECIPE_LIST[@]} recipes, one at a time"
echo "  model   : $MODEL_REF"
echo "  dataset : $DATA (2 examples per recipe)"
echo "  HF cache: $HF_HOME"
echo "  TA cache: $TA_CACHE_DIR"
echo "  scratch : $TMP_OUT (discarded — results/ is untouched)"
echo "========================================================"

FAILED=()
for recipe in "${RECIPE_LIST[@]}"; do
  printf '  %-34s ' "$recipe"
  if in_container "python -m mdg.adv_attack.attack \
        --model        '$MODEL_REF' \
        --dataset      '$DATA' \
        --label-field  label \
        --text-fields  text \
        --output-dir   '$TMP_OUT' \
        --label-space  ${LABELS//,/ } \
        --num-examples 2 \
        --recipes      '$recipe'" > "$TMP_OUT/$recipe.log" 2>&1; then
    echo "ok"
  else
    echo "FAILED (see below)"
    FAILED+=("$recipe")
    tail -5 "$TMP_OUT/$recipe.log" | sed 's/^/      /'
  fi
done

echo ""
echo "  cache sizes:"
du -sh "$HF_HOME" "$TA_CACHE_DIR" 2>/dev/null | sed 's/^/    /'
if [ ${#FAILED[@]} -gt 0 ]; then
  echo ""
  echo "  ${#FAILED[@]} recipe(s) failed to warm: ${FAILED[*]}"
  echo "  These will fail the same way in the real run — fix before launching."
  exit 1
fi
cat <<MSG

  All recipes warmed. The fleet can now start without 60 workers racing to
  download the same files.

  To skip the download on the other machines, copy the cache instead:
    rsync -a $MDG_CACHE_ROOT/ vm2:$MDG_CACHE_ROOT/

MSG
