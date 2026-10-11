#!/usr/bin/env bash
# One command on a bare VM: system deps, pyenv, Poetry, the repo, the data.
#
#   curl -fsSL https://raw.githubusercontent.com/namesarnav/mdg/main/mdg/hpc/bootstrap_vm.sh | bash
#
# Idempotent — rerun it after a failure and it picks up where it stopped.
# setup_poetry.sh does the same thing for a repo you have already cloned.
set -euo pipefail

REPO="${REPO:-https://github.com/namesarnav/mdg.git}"
PROJECT="${PROJECT:-$HOME/mdg}"
PY_VERSION="${PY_VERSION:-3.11.9}"

echo "========================================================"
echo "  REPO    : $REPO"
echo "  PROJECT : $PROJECT"
echo "  PYTHON  : $PY_VERSION (pyenv) → $PROJECT/.venv"
echo "========================================================"

# ── 1. System packages ────────────────────────────────────────────────────────
# pyenv compiles CPython from source, so the headers must exist first. Without
# them the build quietly drops modules: no ssl means pip cannot reach PyPI, no
# sqlite3 means `datasets` fails at import, and neither failure names the cause.
# git-lfs is not optional either — the splits are LFS objects, and without it
# you get 3-line pointer files that fail with a JSON decode error mid-run.
echo "[1/6] System packages"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq --no-install-recommends \
    build-essential curl git git-lfs ca-certificates parallel \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev
elif command -v dnf >/dev/null 2>&1; then
  sudo dnf install -y -q gcc gcc-c++ make patch git git-lfs curl parallel \
    zlib-devel bzip2 bzip2-devel readline-devel sqlite sqlite-devel \
    openssl-devel tk-devel libffi-devel xz-devel libuuid-devel
else
  echo "  [warn] unknown package manager — install build deps and git-lfs by hand"
fi

# ── 2. pyenv ──────────────────────────────────────────────────────────────────
echo "[2/6] pyenv"
export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
[ -d "$PYENV_ROOT" ] || git clone --depth 1 https://github.com/pyenv/pyenv.git "$PYENV_ROOT"
export PATH="$PYENV_ROOT/bin:$HOME/.local/bin:$PATH"
eval "$(pyenv init -)"

if ! grep -q 'PYENV_ROOT' "$HOME/.bashrc" 2>/dev/null; then
  cat >> "$HOME/.bashrc" <<'RC'

# pyenv (added by mdg/hpc/bootstrap_vm.sh)
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$HOME/.local/bin:$PATH"
eval "$(pyenv init -)"
RC
fi

echo "[3/6] CPython $PY_VERSION (compiles from source, several minutes)"
pyenv install -s "$PY_VERSION"
PY_BIN="$PYENV_ROOT/versions/$PY_VERSION/bin/python"

# ── 4. Poetry ─────────────────────────────────────────────────────────────────
echo "[4/6] Poetry"
command -v poetry >/dev/null 2>&1 || curl -fsSL https://install.python-poetry.org | "$PY_BIN" -
export PATH="$HOME/.local/bin:$PATH"

# ── 5. Repo + data ────────────────────────────────────────────────────────────
echo "[5/6] Repo and data"
if [ -d "$PROJECT/.git" ]; then
  git -C "$PROJECT" pull --ff-only
else
  git clone "$REPO" "$PROJECT"
fi
cd "$PROJECT"
git lfs install
git lfs pull
# A pointer file is 3 lines of text; real data is not. Catch it now, not at
# hour nine of an attack run.
for f in mdg/finetune/data/*__test.jsonl; do
  [ -e "$f" ] || continue
  if head -c 60 "$f" | grep -q "git-lfs.github.com/spec"; then
    echo "[FAIL] $f is still an LFS pointer — 'git lfs pull' did not finish."
    exit 1
  fi
done

# ── 6. Project ────────────────────────────────────────────────────────────────
echo "[6/6] Installing the project into ./.venv"
poetry config virtualenvs.in-project true --local
poetry env use "$PY_BIN"
poetry install --no-interaction

"$PROJECT/.venv/bin/python" - <<'PYCHECK'
from importlib.metadata import version
import torch, transformers, sklearn
print(f"  torch {torch.__version__} | transformers {transformers.__version__} "
      f"| textattack {version('textattack')}")
print(f"  threads: {torch.get_num_threads()}")
import mdg.adv_attack.attack as A
import textattack.attack_recipes as R
missing = [c for _, c in A.RECIPES if not hasattr(R, c)]
assert not missing, f"recipes missing from this textattack: {missing}"
print(f"  {len(A.RECIPES)} attack recipes available")
PYCHECK

cat <<MSG

Ready. ./.venv is picked up automatically by mdg/hpc/env.sh.

  cd $PROJECT
  bash mdg/hpc/warm_cache.sh

  export ROW_SHARDS=6 ATTACK_SPLIT=holdout
  unset NUM_EXAMPLES QUERY_BUDGET
  JOBS=64 SHARD=<i>/3 nohup bash mdg/hpc/run_parallel.sh cells > ~/attack.log 2>&1 &

  Progress:  .venv/bin/python mdg/hpc/status.py

MSG
