#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  MDG attack fleet — VM 1 of 3
#
#  Bare Ubuntu box to running attacks, in one command:
#
#      bash vm1.sh
#
#  Needs ~/.hf_token to exist first (chmod 600). Create it with:
#      printf '%s' 'hf_xxx' > ~/.hf_token && chmod 600 ~/.hf_token
#
#  Safe to re-run. Finished recipes are skipped via their done-markers, so a
#  crashed or evicted VM resumes by running this again.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── what this VM runs ────────────────────────────────────────────────────────
SHARD_INDEX=0                  # 0-based; VM1 takes slice 0 of 3
SHARD_TOTAL=3
NUM_EXAMPLES="${NUM_EXAMPLES:-500}" # test examples per dataset
QUERY_BUDGET="${QUERY_BUDGET:-200}" # max model queries per example
ROW_SHARDS="${ROW_SHARDS:-8}"       # row slices per cell; MUST match the other VMs
JOBS="${JOBS:-$(nproc)}"            # one worker per core
REPO="${REPO:-https://github.com/namesarnav/mdg.git}"
PROJECT="$HOME/mdg"
ARCHIVE="$HOME/mdg-archive"         # nothing here is ever deleted
PY_VERSION="${PY_VERSION:-3.12.7}"
PY_FALLBACK="3.11.9"                # poetry.lock was solved here; used if 3.12 fails

say() { printf '\n\033[1m── %s\033[0m\n' "$*"; }

# ── 0. preconditions ─────────────────────────────────────────────────────────
if [ ! -f "$HOME/.hf_token" ]; then
  echo "[FAIL] ~/.hf_token not found. Create it first:"
  echo "         printf '%s' 'hf_xxx' > ~/.hf_token && chmod 600 ~/.hf_token"
  exit 1
fi
chmod 600 "$HOME/.hf_token"
export HF_TOKEN="$(cat "$HOME/.hf_token")"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# ── 1. system packages ───────────────────────────────────────────────────────
say "system packages"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq \
  build-essential curl git git-lfs ca-certificates tmux parallel rsync \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# ── 2. pyenv ─────────────────────────────────────────────────────────────────
say "pyenv"
export PYENV_ROOT="$HOME/.pyenv"
if [ ! -d "$PYENV_ROOT" ]; then
  curl -fsSL https://pyenv.run | bash
fi
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
for rc in "$HOME/.bashrc" "$HOME/.profile"; do
  grep -q 'PYENV_ROOT' "$rc" 2>/dev/null || cat >> "$rc" <<'RC'
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
RC
done

install_py() {
  pyenv versions --bare 2>/dev/null | grep -qx "$1" || pyenv install -s "$1"
}
install_py "$PY_VERSION"

# ── 3. poetry ────────────────────────────────────────────────────────────────
say "poetry"
command -v poetry >/dev/null 2>&1 || curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
grep -q '.local/bin' "$HOME/.bashrc" 2>/dev/null || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"

# ── 4. code ──────────────────────────────────────────────────────────────────
say "repository"
if [ -d "$PROJECT/.git" ]; then
  git -C "$PROJECT" pull --ff-only
else
  git clone "$REPO" "$PROJECT"
fi
cd "$PROJECT"

# ── 5. datasets (Git LFS) ────────────────────────────────────────────────────
say "datasets"
git lfs install
git lfs pull
missing=0
for f in mdg/finetune/data/*__test.jsonl; do
  # A 3-line pointer file parses as JSON garbage 20 minutes into an attack.
  if [ "$(head -c 9 "$f")" = "version h" ]; then
    echo "  [POINTER] $f"; missing=1
  fi
done
[ "$missing" = "1" ] && { echo "[FAIL] git lfs pull incomplete"; exit 1; }
echo "  $(ls mdg/finetune/data/*__test.jsonl | wc -l) test splits, all real data"

# ── 6. dependencies ──────────────────────────────────────────────────────────
say "dependencies"
pyenv local "$PY_VERSION"
poetry config virtualenvs.in-project true --local
if poetry env use "$PYENV_ROOT/versions/$PY_VERSION/bin/python" && poetry install --no-interaction; then
  echo "  installed on $PY_VERSION"
else
  echo "  [WARN] $PY_VERSION failed — falling back to $PY_FALLBACK (what poetry.lock was solved with)"
  install_py "$PY_FALLBACK"
  pyenv local "$PY_FALLBACK"
  rm -rf .venv
  poetry env use "$PYENV_ROOT/versions/$PY_FALLBACK/bin/python"
  poetry install --no-interaction
fi
PY=".venv/bin/python"
$PY - <<'PYEOF'
import sys
sys.path.insert(0, ".")
import textattack.attack_recipes as R
import mdg.adv_attack.attack as A
missing = [c for _, c in A.RECIPES if not hasattr(R, c)]
if missing:
    raise SystemExit(f"[FAIL] recipes not found: {missing}")
print(f"  {len(A.RECIPES)} attack recipes available")
PYEOF

# ── 7. models ────────────────────────────────────────────────────────────────
# Pulled now, serially, so 64 workers do not all hit the Hub at once.
say "models (33 from the Hub)"
$PY - <<'PYEOF'
import os, sys
from huggingface_hub import snapshot_download
sys.path.insert(0, ".")
DATASETS = ["counterbench","ac-reason","bbh-causal-judgement","causalbench_code",
            "causalbench_math","causalbench_text","corr2cause","e-care",
            "fincausal-task1","natquest","Quriosity"]
MODELS = ["bert-base-uncased","roberta-base","t5-base"]
tok = os.environ.get("HF_TOKEN")
ok = bad = 0
for m in MODELS:
    for d in DATASETS:
        rid = f"namesarnav/{d}-{m}"
        try:
            snapshot_download(rid, token=tok)
            ok += 1
        except Exception as e:
            bad += 1
            print(f"  [MISS] {rid}: {type(e).__name__}")
print(f"  cached {ok}/33 models" + (f", {bad} missing" if bad else ""))
PYEOF

# ── 8. attack assets ─────────────────────────────────────────────────────────
# Serial on purpose: concurrent workers racing for the same sentence encoder
# corrupt the shared cache.
say "attack assets (one time, several minutes)"
mkdir -p "$ARCHIVE"
bash mdg/hpc/warm_cache.sh 2>&1 | tee -a "$ARCHIVE/warm_cache.log" | tail -5

# ── 9. launch ────────────────────────────────────────────────────────────────
say "launching"
export ATTACK_SPLIT=holdout
export NUM_EXAMPLES QUERY_BUDGET ROW_SHARDS
export MDG_LOG_DIR="$PROJECT/mdg/logs"
export MDG_KEEP_CHECKPOINTS=1          # never delete anything
mkdir -p "$MDG_LOG_DIR" "$ARCHIVE"

tmux kill-session -t mdg-attack  2>/dev/null || true
tmux kill-session -t mdg-backup  2>/dev/null || true
tmux kill-session -t mdg-status  2>/dev/null || true

# attacks: run_parallel fans out to $JOBS concurrent workers inside this session
tmux new-session -d -s mdg-attack -c "$PROJECT" \
  "ATTACK_SPLIT=holdout NUM_EXAMPLES=$NUM_EXAMPLES QUERY_BUDGET=$QUERY_BUDGET \
   ROW_SHARDS=$ROW_SHARDS MDG_LOG_DIR='$MDG_LOG_DIR' MDG_KEEP_CHECKPOINTS=1 \
   JOBS=$JOBS SHARD=$SHARD_INDEX/$SHARD_TOTAL \
   bash mdg/hpc/run_parallel.sh cells 2>&1 | tee -a '$ARCHIVE/attack.log'"

# backup: every result, summary, CSV and log copied out every 10 minutes,
# with a timestamped tarball every hour. Nothing is ever removed.
cat > "$PROJECT/.archive-loop.sh" <<ARCHEOF
#!/usr/bin/env bash
# Copies every result, summary, CSV and log out of the working tree.
# Errors are logged, never swallowed -- a silent backup failure is how you
# reach the end of a five-day run with nothing to show for it.
PROJECT="$PROJECT"
ARCHIVE="$ARCHIVE"
VM="vm1"
ERR="\$ARCHIVE/archive-errors.log"
cd "\$PROJECT" || exit 1
while true; do
  for d in mdg/adv_attack/results mdg/logs; do
    [ -d "\$d" ] || continue
    mkdir -p "\$ARCHIVE/\$d"
    rsync -a "\$d/" "\$ARCHIVE/\$d/" 2>>"\$ERR" \
      || echo "[\$(date -Is)] rsync failed: \$d" >> "\$ERR"
  done
  for f in mdg/adv_attack/attack_results.csv mdg/results/consolidated.csv; do
    [ -f "\$f" ] && { cp -f "\$f" "\$ARCHIVE/" 2>>"\$ERR" \
      || echo "[\$(date -Is)] cp failed: \$f" >> "\$ERR"; }
  done
  now=\$(date +%s)
  last=0; [ -f "\$ARCHIVE/.last-tar" ] && last=\$(cat "\$ARCHIVE/.last-tar")
  if [ \$(( now - last )) -ge 3600 ]; then
    tar czf "\$ARCHIVE/\$VM-\$(date +%Y%m%d-%H%M).tar.gz" \
      mdg/adv_attack/results mdg/logs 2>>"\$ERR" \
      || echo "[\$(date -Is)] tar failed" >> "\$ERR"
    echo "\$now" > "\$ARCHIVE/.last-tar"
  fi
  sleep 600
done
ARCHEOF
chmod +x "$PROJECT/.archive-loop.sh"
tmux new-session -d -s mdg-backup -c "$PROJECT" "bash '$PROJECT/.archive-loop.sh'"

# status: a live table, refreshed every 5 minutes
tmux new-session -d -s mdg-status -c "$PROJECT" \
  "while true; do clear; '$PROJECT/.venv/bin/python' mdg/hpc/status.py; sleep 300; done"

sleep 5
cat <<EOF

════════════════════════════════════════════════════════════════
  VM1 running — shard $SHARD_INDEX/$SHARD_TOTAL
════════════════════════════════════════════════════════════════
  examples/dataset : $NUM_EXAMPLES
  query budget     : $QUERY_BUDGET
  row shards       : $ROW_SHARDS   (must be identical on all VMs)
  workers          : $JOBS
  results          : $PROJECT/mdg/adv_attack/results/
  archive          : $ARCHIVE/   (synced every 10 min, tarball hourly)

  tmux attach -t mdg-attack     the run
  tmux attach -t mdg-status     progress table
  tmux attach -t mdg-backup     archiver
  (detach with Ctrl-B then D)

  tmux ls
EOF
tmux ls
