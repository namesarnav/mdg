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
# Check the sizing before committing a big machine:
#   DRY_RUN=1 bash mdg/hpc/run_parallel.sh cells
#
# Progress is written to a --joblog, so a re-run resumes:
#   parallel --joblog <file> --resume-failed ...
set -uo pipefail

STAGE="${1:-all}"
HPC_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$HPC_DIR/env.sh"
source "$HPC_DIR/matrix.sh"

# /scratch exists on clusters; fall back to the repo on a plain machine.
LOG_DIR="${MDG_LOG_DIR:-/scratch/$USER/mdg-logs}"
if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
  LOG_DIR="$PROJECT/mdg/logs"
  mkdir -p "$LOG_DIR"
fi

# ── GNU parallel ──────────────────────────────────────────────────────────────
if ! command -v parallel >/dev/null 2>&1; then
  module load parallel 2>/dev/null || true
fi
# GNU parallel is preferred (joblog, resume); xargs -P is the fallback and is
# present on every POSIX system.
HAVE_PARALLEL=1
command -v parallel >/dev/null 2>&1 || HAVE_PARALLEL=0
[ "$HAVE_PARALLEL" = "0" ] && echo "[INFO] GNU parallel not found — using xargs -P (no joblog/resume-failed)"

# ── How many at once ──────────────────────────────────────────────────────────
detect_gpus() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l
  else
    echo 0
  fi
}
detect_cores() {
  # Respect a Slurm allocation if we are inside one, else take the whole box.
  if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then echo "$SLURM_CPUS_PER_TASK"
  elif [ -n "${SLURM_CPUS_ON_NODE:-}" ]; then echo "$SLURM_CPUS_ON_NODE"
  elif command -v nproc >/dev/null 2>&1; then nproc
  elif [ "$(uname)" = "Darwin" ]; then sysctl -n hw.ncpu
  else echo 4; fi
}

detect_mem_gb() {
  if [ -r /proc/meminfo ]; then
    awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo
  elif [ "$(uname)" = "Darwin" ]; then
    echo $(( $(sysctl -n hw.memsize) / 1073741824 ))
  else
    echo 8
  fi
}
NGPU="$(detect_gpus)"
NCORE="$(detect_cores)"

# CPU-only box (no GPU): run many single-threaded workers rather than one
# multi-threaded process. BERT-size inference does not scale well across many
# threads, but N independent attacks across N cores scale almost linearly.
# 1 thread per worker maximises throughput per GB: BERT-size inference barely
# scales across threads, so more concurrent attacks beats faster single ones.
THREADS_PER_JOB="${THREADS_PER_JOB:-1}"
# Each worker is its own python + torch + model copy. Measured ~2-3GB RSS;
# 3GB is the safe planning figure. This is what caps a 256-core/512GB box.
MEM_PER_WORKER_GB="${MEM_PER_WORKER_GB:-3}"
if [ "${NGPU:-0}" -lt 1 ]; then
  CPU_MODE=1
  MEM_GB="$(detect_mem_gb)"
  BY_CORES=$(( NCORE / THREADS_PER_JOB ))
  BY_MEM=$(( MEM_GB / MEM_PER_WORKER_GB ))
  AUTO_JOBS=$(( BY_CORES < BY_MEM ? BY_CORES : BY_MEM ))
  [ "$AUTO_JOBS" -lt 1 ] && AUTO_JOBS=1
  JOBS="${JOBS:-$AUTO_JOBS}"
  LIMITED_BY=$([ "$BY_CORES" -le "$BY_MEM" ] && echo cores || echo memory)
  export OMP_NUM_THREADS="$THREADS_PER_JOB"
  export MKL_NUM_THREADS="$THREADS_PER_JOB"
  export TOKENIZERS_PARALLELISM=false
  NGPU=1   # keeps the slot arithmetic below harmless on CPU
else
  CPU_MODE=0
  JOBS="${JOBS:-$NGPU}"
fi

run_stage() {
  local stage="$1"
  local joblog="$LOG_DIR/parallel-${stage}.joblog"
  echo ""
  echo "========================================================"
  echo "  STAGE   : $stage"
  echo "  TASKS   : $N_JOBS  ($N_MODELS models × $N_DATASETS datasets)"
  if [ "$CPU_MODE" = "1" ]; then
    echo "  HARDWARE: CPU-only — ${NCORE} cores, ${MEM_GB}GB RAM"
    echo "  SIZING  : cores allow $BY_CORES, memory allows $BY_MEM (~${MEM_PER_WORKER_GB}GB/worker)"
    echo "  PARALLEL: $JOBS workers (limited by $LIMITED_BY), $THREADS_PER_JOB thread(s) each"
  else
    echo "  PARALLEL: $JOBS at a time across $NGPU GPU(s)"
  fi
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

# Fan out over (task, recipe) CELLS rather than tasks: 396 units instead of 22,
# which is what keeps a many-core machine busy. Finished cells are skipped.
run_cells() {
  local joblog="$LOG_DIR/parallel-cells.joblog"
  local cells="$LOG_DIR/cells.txt"
  : > "$cells"

  RECIPE_LIST=()
  while IFS= read -r r; do
    [ -n "$r" ] && RECIPE_LIST+=("$r")
  done < <(awk '/^MULTILINGUAL_RECIPES/{exit}
                /^[[:space:]]*\("/{ if (match($0, /"[^"]+"/)) print substr($0, RSTART+1, RLENGTH-2) }' \
                "$PROJECT/mdg/adv_attack/attack.py")

  local i recipe pending=0 hub_only=""
  for ((i=0; i<N_JOBS; i++)); do
    resolve_task "$i"
    [ -n "${ONLY_TASKS:-}" ] && ! grep -qw "$i" <<< "$ONLY_TASKS" && continue
    # A pair trained on ANOTHER machine has no local checkpoint; run_one.sh
    # falls back to the Hub copy, so include it anyway. REQUIRE_LOCAL=1 limits
    # the run to locally-trained pairs.
    if [ ! -f "$PROJECT/mdg/finetune/checkpoints/$STEM/$MODEL_NAME/config.json" ]; then
      [ -n "${REQUIRE_LOCAL:-}" ] && continue
      hub_only="$hub_only $DATASET_STEM/$MODEL_NAME"
    fi
    for recipe in "${RECIPE_LIST[@]}"; do
      [ -f "$PROJECT/mdg/adv_attack/results/${ATTACK_STEM}/${MODEL_NAME}/${recipe}_summary.json" ] && continue
      echo "$i $recipe" >> "$cells"
      pending=$(( pending + 1 ))
    done
  done

  # SHARD="i/n" keeps several machines on disjoint slices of the same grid.
  # Round-robin over the cell list, so each shard gets a similar mix of big and
  # small datasets rather than one machine inheriting all of natquest.
  if [ -n "${SHARD:-}" ]; then
    local sidx="${SHARD%%/*}" scnt="${SHARD##*/}"
    if ! [ "$sidx" -ge 0 ] 2>/dev/null || ! [ "$scnt" -gt 0 ] 2>/dev/null || [ "$sidx" -ge "$scnt" ]; then
      echo "[ERROR] SHARD must be i/n with 0 <= i < n (got '$SHARD')"; return 2
    fi
    awk -v i="$sidx" -v n="$scnt" '(NR-1) % n == i' "$cells" > "$cells.shard"
    mv "$cells.shard" "$cells"
    pending=$(wc -l < "$cells" | tr -d ' ')
    echo ""
    echo "  SHARD   : $sidx of $scnt → $pending cells on this machine"
  fi

  if [ -n "$hub_only" ]; then
    echo ""
    echo "  NOTE    : no local checkpoint for:$hub_only"
    echo "            these will load from the HuggingFace Hub — make sure the"
    echo "            machine that trains them has finished pushing."
  fi

  echo ""
  echo "========================================================"
  echo "  STAGE   : attack (cell-level)"
  echo "  PENDING : $pending cells of $(( N_JOBS * ${#RECIPE_LIST[@]} ))"
  if [ "$CPU_MODE" = "1" ]; then
    echo "  HARDWARE: CPU-only — ${NCORE} cores, ${MEM_GB}GB RAM"
    echo "  SIZING  : cores allow $BY_CORES, memory allows $BY_MEM (~${MEM_PER_WORKER_GB}GB/worker)"
    echo "  PARALLEL: $JOBS workers (limited by $LIMITED_BY), $THREADS_PER_JOB thread(s) each"
  else
    echo "  PARALLEL: $JOBS workers across $NGPU GPU(s)"
  fi
  echo "========================================================"
  [ "$pending" -eq 0 ] && { echo "  nothing left to do"; return 0; }
  if [ -n "${DRY_RUN:-}" ]; then
    echo "  DRY_RUN set — not launching. First 5 cells:"
    head -5 "$cells" | sed 's/^/    /'
    return 0
  fi

  if [ "$HAVE_PARALLEL" = "1" ]; then
    parallel --jobs "$JOBS" --joblog "$joblog" --halt never --line-buffer --colsep ' ' \
      "RECIPES={2} bash '$HPC_DIR/run_one.sh' attack {1} \
         > '$LOG_DIR/cell-{1}-{2}.log' 2>&1" :::: "$cells"
    echo "  failures:"
    awk 'NR>1 && $7 != 0 {print "    seq " $1 " exit " $7}' "$joblog" || true
  else
    # One line per cell: "<task_id> <recipe>"
    xargs -P "$JOBS" -L1 bash -c \
      'RECIPES="$1" bash "'"$HPC_DIR"'/run_one.sh" attack "$0" > "'"$LOG_DIR"'/cell-$0-$1.log" 2>&1' \
      < "$cells"
  fi
}

case "$STAGE" in
  train)  run_stage train ;;
  attack) run_stage attack ;;
  cells)  run_cells ;;
  all)
    run_stage train
    run_stage attack
    echo ""
    echo "[consolidate]"
    cd "$PROJECT" && in_container "python -m mdg.scripts.consolidate_results"
    ;;
  *)
    echo "usage: $0 <train|attack|cells|all>"
    echo "  cells = fan out over (task, recipe) pairs — best for many-core CPU boxes"
    exit 2
    ;;
esac

echo ""
echo "Logs: $LOG_DIR/${STAGE}-task*.log"
