#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  MDG on Colab — one-time setup for a session
#
#      !bash colab/setup.sh
#
#  Expects MDG_DRIVE to point at a mounted Drive folder (the notebook does the
#  mounting; bash cannot). Results are symlinked there so a session timeout
#  costs nothing.
#
#  No pyenv, no Poetry: Colab already has a usable Python, and compiling
#  CPython would burn ten minutes of a session you only get twelve hours of.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MDG_DRIVE="${MDG_DRIVE:-/content/drive/MyDrive/mdg}"
PROJECT="${PROJECT:-/content/mdg}"
REPO="${REPO:-https://github.com/namesarnav/mdg.git}"

say() { printf '\n\033[1m── %s\033[0m\n' "$*"; }

if [ ! -d "$(dirname "$MDG_DRIVE")" ]; then
  echo "[FAIL] $(dirname "$MDG_DRIVE") does not exist — mount Drive first:"
  echo "         from google.colab import drive; drive.mount('/content/drive')"
  exit 1
fi
if [ ! -f "$HOME/.hf_token" ]; then
  echo "[FAIL] ~/.hf_token missing. In a notebook cell:"
  echo "         import getpass,pathlib; p=pathlib.Path.home()/'.hf_token'"
  echo "         p.write_text(getpass.getpass()); p.chmod(0o600)"
  exit 1
fi
export HF_TOKEN="$(cat "$HOME/.hf_token")"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

say "hardware"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  no GPU — attacks will be far slower"
echo "  $(nproc) vCPU, $(free -g | awk 'NR==2{print $2}')GB RAM"

say "system packages"
apt-get install -y -qq git-lfs parallel tmux rsync >/dev/null 2>&1 || true

say "python dependencies"
pip install -q "textattack==0.3.11" "transformers>=4.46,<5" accelerate \
  sentencepiece peft "datasets>=3.0" scikit-learn 2>&1 | tail -2 || true

say "repository"
if [ -d "$PROJECT/.git" ]; then
  git -C "$PROJECT" pull --ff-only
else
  git clone "$REPO" "$PROJECT"
fi
cd "$PROJECT"

say "datasets (Git LFS)"
git lfs install
git lfs pull
missing=0
for f in mdg/finetune/data/*__test.jsonl; do
  # A 3-line pointer file parses as JSON garbage 20 minutes into an attack.
  [ "$(head -c 9 "$f")" = "version h" ] && { echo "  [POINTER] $f"; missing=1; }
done
[ "$missing" = "1" ] && { echo "[FAIL] git lfs pull incomplete"; exit 1; }
echo "  $(ls mdg/finetune/data/*__test.jsonl | wc -l) test splits, all real data"

# ── results live on Drive, not in the container ──────────────────────────────
say "persistence"
mkdir -p "$MDG_DRIVE/results" "$MDG_DRIVE/logs" "$MDG_DRIVE/hf_cache"
# Results are written to LOCAL disk and rsynced to Drive by run.sh.
#
# They used to be a symlink straight into $MDG_DRIVE. That looks tidier and is
# wrong: attack.py holds the per-example .jsonl open for the whole unit
# (attack.py:267), and Drive's FUSE mount does not publish a file's contents
# until it is closed. Hours of streamed results were invisible in Drive, which
# is indistinguishable from nothing happening.
if [ -L mdg/adv_attack/results ]; then
  rm -f mdg/adv_attack/results            # undo the old symlink layout
fi
mkdir -p mdg/adv_attack/results
# Anything already on Drive comes back so done-markers are seen and skipped.
if [ -d "$MDG_DRIVE/results" ]; then
  rsync -a "$MDG_DRIVE/results/" mdg/adv_attack/results/ 2>/dev/null || true
  echo "  restored $(find mdg/adv_attack/results -name '*_summary.json' | wc -l) finished unit(s) from Drive"
fi
echo "  results  -> local, mirrored to $MDG_DRIVE/results every 3 min"
echo "  logs     -> $MDG_DRIVE/logs"
# Model cache on Drive too, so a new session does not re-download 20GB.
export HF_HOME="$MDG_DRIVE/hf_cache"
echo "  hf_cache -> $MDG_DRIVE/hf_cache"

say "models (33 from the Hub, cached to Drive)"
HF_HOME="$MDG_DRIVE/hf_cache" python3 - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
DATASETS = ["counterbench","ac-reason","bbh-causal-judgement","causalbench_code",
            "causalbench_math","causalbench_text","corr2cause","e-care",
            "fincausal-task1","natquest","Quriosity"]
ok = bad = 0
for m in ("bert-base-uncased","roberta-base","t5-base"):
    for d in DATASETS:
        rid = f"namesarnav/{d}-{m}"
        try:
            snapshot_download(rid, token=os.environ.get("HF_TOKEN")); ok += 1
        except Exception as e:
            bad += 1; print(f"  [MISS] {rid}: {type(e).__name__}")
print(f"  cached {ok}/33" + (f", {bad} missing" if bad else ""))
PYEOF

say "attack assets (serial — concurrent workers corrupt the shared cache)"
MDG_CACHE_ROOT="$MDG_DRIVE/textattack_cache" bash mdg/hpc/warm_cache.sh 2>&1 | tail -4

say "verifying"
python3 - <<'PYEOF'
import sys; sys.path.insert(0, ".")
import textattack.attack_recipes as R, mdg.adv_attack.attack as A
missing = [c for _, c in A.RECIPES if not hasattr(R, c)]
if missing: raise SystemExit(f"[FAIL] recipes not found: {missing}")
print(f"  {len(A.RECIPES)} attack recipes available")
import torch; print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
PYEOF

cat <<MSG

Setup complete. Now run:

    !MDG_DRIVE=$MDG_DRIVE bash colab/run.sh 11

(11 = hours to work before stopping cleanly. Re-run to resume.)
MSG
