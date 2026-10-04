#!/usr/bin/env bash
# ONE-TIME environment build on NYU Greene. Run this on a compute node, not the
# login node (it compiles wheels):
#
#   ssh <netid>@greene.hpc.nyu.edu
#   git clone git@github.com:namesarnav/mdg.git /scratch/$USER/mdg
#   srun --cpus-per-task=4 --mem=32G --time=2:00:00 --pty /bin/bash
#   bash /scratch/$USER/mdg/mdg/hpc/00_setup_env.sh
#
# Builds a 15GB ext3 overlay holding a conda env with torch, transformers,
# peft, bitsandbytes and textattack.
set -euo pipefail

source "$(dirname "$0")/env.sh"

ENV_DIR="$(dirname "$OVERLAY")"
mkdir -p "$ENV_DIR" "$HF_HOME" "$TA_CACHE_DIR"

# ── Overlay image ─────────────────────────────────────────────────────────────
if [ ! -f "$OVERLAY" ]; then
  echo "[1/3] Creating overlay at $OVERLAY"
  # Verify the source with: ls /scratch/work/public/overlay-fs-ext3/
  cp -rp /scratch/work/public/overlay-fs-ext3/overlay-15GB-500K.ext3.gz "$OVERLAY.gz"
  gunzip "$OVERLAY.gz"
else
  echo "[1/3] Overlay already exists — reusing $OVERLAY"
fi

# ── Install conda + packages INSIDE the overlay (read-write) ───────────────────
echo "[2/3] Installing conda environment inside the overlay"
singularity exec --overlay "${OVERLAY}:rw" "$SIF" /bin/bash <<'INNER'
set -euo pipefail

if [ ! -d /ext3/miniforge3 ]; then
  cd /tmp
  wget -q --no-check-certificate \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3
fi

cat > /ext3/env.sh <<'WRAP'
#!/bin/bash
unset -f which
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:$PATH
WRAP

source /ext3/env.sh
conda install -y python=3.11 pip

# transformers is pinned <5: TextAttack does not support the v5 API, and the
# Trainer kwargs used in mdg/models/ changed there too.
pip install --no-cache-dir \
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

python -c "import torch, transformers, textattack, peft; print('torch', torch.__version__, '| transformers', transformers.__version__)"
INNER

# ── Verify GPU visibility ─────────────────────────────────────────────────────
echo "[3/3] Verifying the environment (GPU check needs a GPU node)"
source "$(dirname "$0")/env.sh"
in_container "python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\"" || \
  echo "[NOTE] No GPU on this node — fine, the batch jobs request one."

echo
echo "Environment ready: $OVERLAY"
echo "Next: bash $PROJECT/mdg/hpc/submit_all.sh"
