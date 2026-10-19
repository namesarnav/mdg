#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  MDG on Colab — work for N hours, then stop cleanly
#
#      !MDG_DRIVE=/content/drive/MyDrive/mdg bash colab/run.sh 11
#
#  Attacks the COMPLETE test split of every dataset. No example cap, no query
#  budget. The only limit is wall-clock: it stops between units when the
#  budget runs out, and every finished unit is skipped on the next run, so a
#  Colab timeout costs at most one unit of work.
#
#  SHARD=i/n splits the grid across several Colab accounts or sessions.
#
#  mdg/hpc/legacy/run_budget.sh does something similar but requires a local
#  checkpoint and omits the __test suffix from its done-markers, so it is not
#  usable here -- this iterates Hub models and uses run_one.sh's own markers.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

HOURS="${1:-11}"
PROJECT="${PROJECT:-/content/mdg}"
MDG_DRIVE="${MDG_DRIVE:-/content/drive/MyDrive/mdg}"
SHARD="${SHARD:-0/1}"
ROW_SHARDS="${ROW_SHARDS:-64}"      # must match the VM fleet for results to merge
JOBS="${JOBS:-}"

cd "$PROJECT" || { echo "[FAIL] no $PROJECT — run colab/setup.sh first"; exit 1; }

export ATTACK_SPLIT=holdout
unset NUM_EXAMPLES QUERY_BUDGET     # complete split, unlimited search depth
export ROW_SHARDS
export MDG_LOG_DIR="$MDG_DRIVE/logs"
# Local disk, never Drive: filelock and symlinks do not work on the FUSE mount.
export MDG_CACHE_ROOT="${MDG_CACHE_ROOT:-/content/mdg-cache}"
export HF_HOME="$MDG_CACHE_ROOT/hf_cache"
export MDG_KEEP_CHECKPOINTS=1
mkdir -p "$MDG_LOG_DIR"

# ── GPU sizing ───────────────────────────────────────────────────────────────
# With a GPU the workers share it, so the limit is host RAM and vCPU rather
# than VRAM: one bert-base in fp16 is ~250MB against an L4's 24GB.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
VRAM_MB="${VRAM_MB:-0}"

if [ -z "$JOBS" ]; then
  n=$(nproc)
  if [ "$VRAM_MB" -ge 20000 ]; then
    JOBS=$(( n > 8 ? 8 : n ))        # L4 / A100 class
  elif [ "$VRAM_MB" -ge 1 ]; then
    JOBS=$(( n > 4 ? 4 : n ))        # T4 class
  else
    JOBS=$(( n > 4 ? 4 : n ))        # CPU only
  fi
  [ "$JOBS" -lt 1 ] && JOBS=1
fi

# Goal-function query batch size. TextAttack's default of 32 wastes a big GPU;
# PSO-class recipes issue thousands of forward passes per example and they
# batch perfectly. Left unset on CPU, where a large batch only costs memory.
if [ -z "${MDG_MODEL_BATCH:-}" ]; then
  if   [ "$VRAM_MB" -ge 20000 ]; then MDG_MODEL_BATCH=256
  elif [ "$VRAM_MB" -ge 14000 ]; then MDG_MODEL_BATCH=128
  elif [ "$VRAM_MB" -ge 1 ];     then MDG_MODEL_BATCH=64
  fi
fi
[ -n "${MDG_MODEL_BATCH:-}" ] && export MDG_MODEL_BATCH

# ── refuse to run on a cold cache ────────────────────────────────────────────
# A failed warm makes every recipe fail identically, and the unit loop counts
# them anyway -- which is how a 12-hour session produces nothing while looking
# busy. Check before burning the session, not after.
if [ ! -d "$MDG_CACHE_ROOT/textattack_cache" ] || \
   [ -z "$(ls -A "$MDG_CACHE_ROOT/textattack_cache" 2>/dev/null)" ]; then
  if [ -d "$MDG_DRIVE/cache" ]; then
    echo "  restoring cache from Drive..."
    rsync -a "$MDG_DRIVE/cache/" "$MDG_CACHE_ROOT/" 2>/dev/null || true
  fi
fi
if [ ! -d "$MDG_CACHE_ROOT/textattack_cache" ] || \
   [ -z "$(ls -A "$MDG_CACHE_ROOT/textattack_cache" 2>/dev/null)" ]; then
  echo "[FAIL] attack asset cache is empty at $MDG_CACHE_ROOT/textattack_cache"
  echo "       Every recipe would fail instantly. Run colab/setup.sh first."
  exit 1
fi

DEADLINE=$(( $(date +%s) + $(awk -v h="$HOURS" 'BEGIN{printf "%d", h*3600}') ))
left() { echo $(( DEADLINE - $(date +%s) )); }
human() { printf '%dh%02dm' $(( $1 / 3600 )) $(( ($1 % 3600) / 60 )); }

# ── mirror results to Drive ──────────────────────────────────────────────────
# rsync of CLOSED files, which Drive's FUSE mount handles correctly, instead of
# writing the live .jsonl straight into Drive where it stays invisible until the
# unit ends.
mirror_to_drive() {
  mkdir -p "$MDG_DRIVE/results"
  rsync -a mdg/adv_attack/results/ "$MDG_DRIVE/results/" 2>/dev/null || \
    echo "  [WARN] mirror to Drive failed" >&2
}
( while true; do sleep 180; mirror_to_drive; done ) &
MIRROR_PID=$!
trap 'mirror_to_drive; kill "$MIRROR_PID" 2>/dev/null' EXIT INT TERM

echo "════════════════════════════════════════════════════════"
echo "  budget     : ${HOURS}h  (stops between units; re-run to resume)"
echo "  shard      : $SHARD      row shards: $ROW_SHARDS"
echo "  gpu        : ${GPU_NAME:-none}${VRAM_MB:+ (${VRAM_MB}MB)}"
echo "  workers    : $JOBS"
echo "  query batch: ${MDG_MODEL_BATCH:-TextAttack default (32)}"
echo "  examples   : ALL (complete test split, no cap)"
echo "  results    : $MDG_DRIVE/results"
echo "════════════════════════════════════════════════════════"

# ── Enumerate pending units, cheapest first ──────────────────────────────────
# Done in Python: 33 x 18 x ROW_SHARDS is up to 38k units, and spawning
# run_one.sh just to have it skip would cost half an hour of a 12h session.
PENDING="$(mktemp)"
python3 - "$ROW_SHARDS" "$SHARD" > "$PENDING" <<'PYEOF'
import json, sys, pathlib
row_shards = int(sys.argv[1])
si, sn = (int(x) for x in sys.argv[2].split("/"))

DATASETS = ["counterbench","ac-reason","bbh-causal-judgement","causalbench_code",
            "causalbench_math","causalbench_text","corr2cause","e-care",
            "fincausal-task1","natquest","Quriosity"]
MODELS = ["bert-base-uncased","roberta-base","t5-base"]
sys.path.insert(0, ".")
import mdg.adv_attack.attack as A
RECIPES = [n for n, _ in A.RECIPES]

# Queries per WORD. The four marked (*) are measured from completed runs in
# this repo; the rest are estimates of the same shape. attack.py's list order
# is NOT cost order -- CheckList (the cheapest recipe there is) sits at index
# 10 and PSO at 12 -- so ordering by list position would hand a short Colab
# session the most expensive work first.
QPW = {"CheckList2020": 0.35,            # *
       "Pruthi2019": 1.0,
       "BadCharacters2021": 0.8,
       "DeepWordBugGao2018": 1.4,        # *
       "TextBuggerLi2018": 2.5,
       "MorpheusTan2020": 2.5,
       "InputReductionFeng2018": 3.0,
       "A2TYoo2021": 5.0,
       "PWWSRen2019": 5.4,               # *
       "Kuleshov2017": 8.0,
       "TextFoolerJin2019": 10.0,
       "BAEGarg2019": 12.0,
       "BERTAttackLi2020": 16.0,
       "CLARE2020": 25.0,
       "IGAWang2019": 30.0,
       "FasterGeneticAlgorithmJia2019": 40.0,
       "GeneticAlgorithmAlzantot2018": 60.0,
       "PSOZang2020": 133.0}             # *
COST = {n: QPW.get(n, 10.0) for n in RECIPES}

DATA = pathlib.Path("mdg/finetune/data")
RES  = pathlib.Path("mdg/adv_attack/results")
rows = {d: sum(1 for _ in open(DATA / f"namesarnav_{d}__test.jsonl")) for d in DATASETS}
words = {d: (sum(len(json.loads(l)["text"].split())
             for l in open(DATA / f"namesarnav_{d}__test.jsonl")) / rows[d])
         for d in DATASETS}

units = []
for mi, m in enumerate(MODELS):
    for di, d in enumerate(DATASETS):
        task = mi * len(DATASETS) + di
        out = RES / f"namesarnav_{d}__test" / m
        for r in RECIPES:
            for s in range(row_shards):
                sfx = f".sh{s}of{row_shards}" if row_shards > 1 else ""
                if (out / f"{r}{sfx}_summary.json").exists():
                    continue
                # cheapest first: total queries for this unit
                units.append((rows[d] * words[d] * COST[r] / row_shards, task, r, s))

units.sort()
mine = [u for i, u in enumerate(units) if i % sn == si]
for _, task, r, s in mine:
    print(task, r, f"{s}/{row_shards}")
print(f"# {len(mine)} pending of {len(units)} total", file=sys.stderr)
PYEOF

TOTAL=$(grep -vc '^#' "$PENDING" 2>/dev/null || echo 0)
echo "  pending    : $TOTAL unit(s) for this shard"
[ "$TOTAL" -eq 0 ] && { echo; echo "[DONE] nothing left for shard $SHARD"; rm -f "$PENDING"; exit 0; }

# ── Work the list, JOBS at a time, until the budget runs out ─────────────────
done_n=0
while IFS=' ' read -r task recipe rowshard; do
  [ -z "${task:-}" ] && continue
  if [ "$(left)" -le 120 ]; then
    echo; echo "[BUDGET] time up after $done_n unit(s) — stopping cleanly"
    break
  fi
  echo "[$(human $(left)) left] task $task  $recipe  rows $rowshard"
  # tee, not >>: with everything redirected to a file the cell printed one
  # line per unit and looked dead for hours.
  RECIPES="$recipe" ROW_SHARD="$rowshard" \
    timeout "$(left)" bash mdg/hpc/run_one.sh attack "$task" 2>&1 \
    | tee -a "$MDG_LOG_DIR/colab-run.log" | grep -E "attack split|it/s|s/it|\[OK\]|\[FAIL\]|SKIP|Traceback|Error" || true
  done_n=$(( done_n + 1 ))
  mirror_to_drive
done < <(grep -v '^#' "$PENDING" | head -n 100000)
rm -f "$PENDING"

mirror_to_drive
kill "$MIRROR_PID" 2>/dev/null || true

echo
echo "  completed this session : $done_n unit(s)"
echo "  results on Drive       : $MDG_DRIVE/results"
echo "  re-run this cell to continue"
