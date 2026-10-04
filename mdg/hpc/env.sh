#!/usr/bin/env bash
# Shared environment for all NYU Greene jobs. Sourced by every sbatch script.
#
# Edit PROJECT if you clone the repo somewhere other than /scratch/$USER/mdg.

# ── Paths ─────────────────────────────────────────────────────────────────────
export PROJECT="${PROJECT:-/scratch/$USER/mdg}"
export OVERLAY="${OVERLAY:-/scratch/$USER/mdg-env/overlay-15GB-500K.ext3}"

# CUDA container image. Verify what your cluster has with:
#   ls /scratch/work/public/singularity/ | grep cuda
export SIF="${SIF:-/scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif}"

# Keep every cache on /scratch — /home has a hard inode quota that HuggingFace
# model caches blow through immediately.
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TA_CACHE_DIR="${TA_CACHE_DIR:-/scratch/$USER/textattack_cache}"
export TOKENIZERS_PARALLELISM=false

# ── HuggingFace token (needed to download gated Llama + to push models) ───────
# Put your token in ~/.hf_token (chmod 600). Never commit it.
if [ -f "$HOME/.hf_token" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.hf_token")"
  export HUGGINGFACE_TOKEN="$HF_TOKEN"
fi

# ── Run a command inside the container ────────────────────────────────────────
# Overlay is mounted READ-ONLY so concurrent array tasks can share it safely.
in_container() {
  singularity exec --nv \
    --overlay "${OVERLAY}:ro" \
    "$SIF" \
    /bin/bash -c "source /ext3/env.sh; cd $PROJECT; $*"
}
