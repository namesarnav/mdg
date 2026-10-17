#!/usr/bin/env bash
# One-time setup on this Mac. Run before any dataset script.
#
#   bash mac-scripts/setup.sh
#
# The Linux equivalent is mdg/hpc/bootstrap_vm.sh, which also installs pyenv
# and Poetry from scratch. On a Mac those come from Homebrew, so this only
# builds the project environment.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

need() { command -v "$1" >/dev/null 2>&1 || { echo "[FAIL] $1 not found — brew install $2"; exit 1; }; }
need pyenv pyenv
need poetry poetry
need git-lfs git-lfs

PY_VERSION="$(cat .python-version)"
echo "── python $PY_VERSION ──"
pyenv versions --bare | grep -qx "$PY_VERSION" || pyenv install "$PY_VERSION"

echo "── data (Git LFS) ──"
git lfs install
git lfs pull

# A 3-line pointer file parses as JSON garbage 20 minutes into an attack.
missing=0
for f in mdg/finetune/data/*__test.jsonl; do
  [ "$(head -c 9 "$f")" = "version h" ] && { echo "  [POINTER] $f"; missing=1; }
done
[ "$missing" = "1" ] && { echo "[FAIL] git lfs pull did not fetch everything"; exit 1; }
echo "  $(ls mdg/finetune/data/*__test.jsonl | wc -l | tr -d ' ') test splits, all real data"

echo "── dependencies → ./.venv ──"
poetry config virtualenvs.in-project true --local
poetry env use "$(pyenv root)/versions/$PY_VERSION/bin/python"
poetry install

echo "── verifying all 18 recipes import ──"
.venv/bin/python - <<'PY'
import textattack.attack_recipes as R
import mdg.adv_attack.attack as A
missing = [c for _, c in A.RECIPES if not hasattr(R, c)]
if missing:
    raise SystemExit(f"[FAIL] recipes not found: {missing}")
print(f"  {len(A.RECIPES)} attack recipes available")
PY

echo
echo "Ready. Next:"
echo "    caffeinate -i bash mac-scripts/counterbench.sh"
