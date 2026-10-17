#!/usr/bin/env bash
# Build the environment on a plain GPU VM — GCP, RunPod, Lambda, Vast, a lab
# box, anything with an NVIDIA driver. No Slurm, no Singularity, no overlay.
#
#   git clone https://github.com/namesarnav/mdg.git ~/mdg && cd ~/mdg
#   bash mdg/hpc/setup_vm.sh
#
# Then run the grid on this machine's GPUs:
#   bash mdg/hpc/run_parallel.sh all
set -euo pipefail

HPC_DIR="$(cd "$(dirname "$0")" && pwd)"
export MDG_RUNTIME=direct
source "$HPC_DIR/env.sh"

echo "========================================================"
echo "  PROJECT : $PROJECT"
echo "  ENV     : $MDG_VENV"
echo "  GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd, - || echo 'none detected')"
echo "========================================================"

# ── Miniforge ─────────────────────────────────────────────────────────────────
if [ ! -d "$MDG_VENV" ]; then
  echo "[1/2] Installing Miniforge → $MDG_VENV"
  TMP_SH="$(mktemp /tmp/miniforge-XXXX.sh)"
  curl -fsSL -o "$TMP_SH" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
  bash "$TMP_SH" -b -p "$MDG_VENV"
  rm -f "$TMP_SH"
else
  echo "[1/2] Reusing existing env at $MDG_VENV"
fi

# shellcheck disable=SC1091
source "$MDG_VENV/bin/activate"

# ── Packages ──────────────────────────────────────────────────────────────────
# transformers pinned <5: TextAttack does not support the v5 API.
echo "[2/2] Installing packages"
"$MDG_VENV/bin/python" -m pip install --upgrade pip
"$MDG_VENV/bin/python" -m pip install --no-cache-dir \
  "torch==2.5.1" \
  "transformers>=4.46,<5" \
  "accelerate>=1.0" \
  "peft>=0.13" \
  "bitsandbytes>=0.44" \
  "datasets>=3.0,<4" \
  "sentencepiece" \
  "scikit-learn>=1.5" \
  "textattack>=0.3.10" \
  "huggingface_hub>=0.26" \
  "pandas" "tqdm" "pyyaml" "python-dotenv" "pydantic>=2"

"$MDG_VENV/bin/python" - <<'PYCHECK'
import torch, transformers, textattack, peft
print(f"torch {torch.__version__} | transformers {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}  devices: {torch.cuda.device_count()}")
PYCHECK

cat <<MSG

Environment ready.

  HuggingFace token (gated Llama + pushing models):
    echo 'hf_xxx' > ~/.hf_token && chmod 600 ~/.hf_token

  Smoke test one pair:
    bash mdg/hpc/run_one.sh train 0

  Run the whole grid across this machine's GPUs:
    bash mdg/hpc/run_parallel.sh all

MSG
