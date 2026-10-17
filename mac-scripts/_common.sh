#!/usr/bin/env bash
# Shared body for every mac-scripts/<dataset>.sh. Not run directly.
#
# One dataset end to end on this Mac: all 3 models x all 18 recipes, on the
# COMPLETE test split, no example cap and no query budget.
#
# Training is not a step here. All 33 models are already on the Hub, and
# run_one.sh falls back to the Hub copy when there is no local checkpoint.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PROJECT="$ROOT"
source "$ROOT/mdg/hpc/matrix.sh"

# ── Settings every dataset script inherits ────────────────────────────────────
export ATTACK_SPLIT=holdout          # test split only
unset NUM_EXAMPLES QUERY_BUDGET      # complete split, unlimited search depth

# ROW_SHARDS is baked into the output FILENAMES (<recipe>.sh<i>of<n>.jsonl).
# The VM fleet runs 6. A cell attacked here with a different count cannot be
# merged with the same cell attacked there, so leave this alone unless the
# dataset is yours alone.
export ROW_SHARDS="${ROW_SHARDS:-6}"

# Some TextAttack ops have no MPS kernel; fall back to CPU rather than crash.
export PYTORCH_ENABLE_MPS_FALLBACK=1

LOG_DIR="$ROOT/mac-scripts/logs"
mkdir -p "$LOG_DIR"
export MDG_LOG_DIR="$LOG_DIR"

# ── Download every recipe's assets once, serially ─────────────────────────────
# Concurrent workers racing for the same sentence encoder corrupt the cache.
warm_cache_once() {
  local marker="$ROOT/mac-scripts/.cache-warmed"
  if [ -f "$marker" ]; then
    echo "  cache  : already warmed ($(cat "$marker"))"
    return 0
  fi
  echo "  cache  : warming (one time, several minutes)"
  if bash "$ROOT/mdg/hpc/warm_cache.sh" >> "$LOG_DIR/warm_cache.log" 2>&1; then
    date > "$marker"
    echo "  cache  : done"
  else
    echo "  cache  : FAILED — see $LOG_DIR/warm_cache.log"
    return 1
  fi
}

# ── The pipeline ──────────────────────────────────────────────────────────────
run_dataset() {
  local d_idx="$1"
  local stem="${DATASETS[$d_idx]%%|*}"
  local test_file="$ROOT/mdg/finetune/data/${stem}__test.jsonl"

  # Task ids are derived, never hardcoded: resolve_task maps
  # idx -> (idx / N_DATASETS, idx % N_DATASETS), so this dataset's three
  # models are d_idx, d_idx + N_DATASETS, d_idx + 2*N_DATASETS.
  local tasks="" m
  for (( m = 0; m < N_MODELS; m++ )); do
    tasks="$tasks $(( m * N_DATASETS + d_idx ))"
  done
  tasks="${tasks# }"

  local rows=0
  [ -f "$test_file" ] && rows=$(wc -l < "$test_file" | tr -d ' ')

  echo "══════════════════════════════════════════════════════════════"
  echo "  ${stem#namesarnav_}"
  echo "══════════════════════════════════════════════════════════════"
  echo "  tasks  : $tasks  ($N_MODELS models)"
  echo "  rows   : $rows (complete test split, no cap)"
  echo "  cells  : $(( N_MODELS * 18 ))  x $ROW_SHARDS row shards"
  echo "  cores  : $(sysctl -n hw.logicalcpu) logical, $(( $(sysctl -n hw.memsize) / 1073741824 ))GB RAM"
  echo "  log    : $LOG_DIR/cells-task*.log"
  echo

  if [ "$rows" -eq 0 ]; then
    echo "[FAIL] Missing or empty: $test_file"
    echo "       Run: git lfs pull"
    return 1
  fi
  # A 3-line LFS pointer parses as JSON garbage 20 minutes into a run.
  if [ "$(head -c 9 "$test_file")" = "version h" ]; then
    echo "[FAIL] $test_file is an LFS pointer, not data. Run: git lfs pull"
    return 1
  fi

  warm_cache_once || return 1

  echo "  → attacking"
  ONLY_TASKS="$tasks" bash "$ROOT/mdg/hpc/run_parallel.sh" cells
  local status=$?

  echo
  echo "  → merging row shards"
  python "$ROOT/mdg/scripts/merge_shards.py" 2>&1 | tail -5

  echo
  echo "  → finished recipes for ${stem#namesarnav_}"
  local d
  for d in "$ROOT/mdg/adv_attack/results/${stem}__test"/*/; do
    [ -d "$d" ] || continue
    local done_n part_n
    done_n=$(find "$d" -maxdepth 1 -name '*.jsonl' ! -name '*.sh*of*.jsonl' | wc -l | tr -d ' ')
    part_n=$(find "$d" -maxdepth 1 -name '*.sh*of*.jsonl' | wc -l | tr -d ' ')
    printf "      %-22s %2s/18 merged, %3s shard files\n" "$(basename "$d")" "$done_n" "$part_n"
  done
  echo
  return $status
}
