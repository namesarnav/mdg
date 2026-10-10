#!/usr/bin/env bash
# Build the environment with pyenv + Poetry instead of conda.
#
#   git clone https://github.com/namesarnav/mdg.git ~/mdg && cd ~/mdg
#   bash mdg/hpc/setup_poetry.sh
#
# The venv lands in ./.venv, which mdg/hpc/env.sh picks up automatically, so
# run_parallel.sh / run_one.sh work afterwards with no extra configuration.
#
# setup_vm.sh does the same job with Miniforge. Use one or the other, not both.
set -euo pipefail

PY_VERSION="${PY_VERSION:-3.11.9}"
HPC_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(cd "$HPC_DIR/../.." && pwd)"
cd "$PROJECT"

echo "========================================================"
echo "  PROJECT : $PROJECT"
echo "  PYTHON  : $PY_VERSION (pyenv)"
echo "  VENV    : $PROJECT/.venv (poetry, in-project)"
echo "========================================================"

# ── Build deps ────────────────────────────────────────────────────────────────
# pyenv compiles CPython from source, so the headers have to be there first.
# Without them the build silently drops modules (no ssl -> pip cannot reach
# PyPI; no sqlite3 -> datasets fails at import).
if command -v apt-get >/dev/null 2>&1; then
  echo "[1/5] Build dependencies (sudo apt-get)"
  sudo apt-get update -qq || true
  sudo apt-get install -y -qq --no-install-recommends \
    build-essential curl git ca-certificates \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev
elif command -v dnf >/dev/null 2>&1; then
  echo "[1/5] Build dependencies (sudo dnf)"
  sudo dnf install -y -q gcc gcc-c++ make patch zlib-devel bzip2 bzip2-devel \
    readline-devel sqlite sqlite-devel openssl-devel tk-devel libffi-devel \
    xz-devel libuuid-devel gdbm-libs libnsl2 git curl
else
  echo "[1/5] Unknown package manager — skipping build deps. If the pyenv"
  echo "      build fails, install the CPython build dependencies by hand."
fi

# ── pyenv ─────────────────────────────────────────────────────────────────────
export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
if [ ! -d "$PYENV_ROOT" ]; then
  echo "[2/5] Installing pyenv → $PYENV_ROOT"
  git clone --depth 1 https://github.com/pyenv/pyenv.git "$PYENV_ROOT"
else
  echo "[2/5] Reusing pyenv at $PYENV_ROOT"
fi
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Make it work in future login shells too, once.
SHELL_RC="$HOME/.bashrc"
if ! grep -q 'PYENV_ROOT' "$SHELL_RC" 2>/dev/null; then
  cat >> "$SHELL_RC" <<'RC'

# pyenv (added by mdg/hpc/setup_poetry.sh)
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$HOME/.local/bin:$PATH"
eval "$(pyenv init -)"
RC
fi

if ! pyenv versions --bare | grep -qx "$PY_VERSION"; then
  echo "[3/5] Building CPython $PY_VERSION (several minutes)"
  pyenv install -s "$PY_VERSION"
else
  echo "[3/5] CPython $PY_VERSION already built"
fi
pyenv local "$PY_VERSION"
PY_BIN="$PYENV_ROOT/versions/$PY_VERSION/bin/python"

# ── Poetry ────────────────────────────────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"
if ! command -v poetry >/dev/null 2>&1; then
  echo "[4/5] Installing Poetry"
  curl -fsSL https://install.python-poetry.org | "$PY_BIN" -
else
  echo "[4/5] Reusing $(poetry --version)"
fi

# ── Project ───────────────────────────────────────────────────────────────────
echo "[5/5] Installing the project"
poetry config virtualenvs.in-project true --local
poetry env use "$PY_BIN"
poetry install --no-interaction

"$PROJECT/.venv/bin/python" - <<'PYCHECK'
from importlib.metadata import version
import torch, transformers, textattack, sklearn
print(f"  torch {torch.__version__} | transformers {transformers.__version__} "
      f"| textattack {version('textattack')}")
print(f"  CUDA available: {torch.cuda.is_available()}  threads: {torch.get_num_threads()}")
import mdg.adv_attack.attack as A
import textattack.attack_recipes as R
missing = [c for _, c in A.RECIPES if not hasattr(R, c)]
assert not missing, f"recipes missing from this textattack: {missing}"
print(f"  {len(A.RECIPES)} attack recipes available")
PYCHECK

cat <<MSG

Environment ready — ./.venv, found automatically by mdg/hpc/env.sh.

  HuggingFace token (needed to push trained models):
    echo 'hf_xxx' > ~/.hf_token && chmod 600 ~/.hf_token

  Build the full-dataset splits:
    bash mdg/hpc/02_build_all_splits.sh

  Smoke test one cell:
    RECIPES=Pruthi2019 NUM_EXAMPLES=5 bash mdg/hpc/run_one.sh attack 0

MSG
