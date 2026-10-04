#!/usr/bin/env bash
# Shared environment for all NYU Greene jobs. Sourced by every sbatch script.
#
# Edit PROJECT if you clone the repo somewhere other than /scratch/$USER/mdg.

# ── Paths ─────────────────────────────────────────────────────────────────────
export PROJECT="${PROJECT:-/scratch/$USER/mdg}"
export OVERLAY="${OVERLAY:-/scratch/$USER/mdg-env/overlay-15GB-500K.ext3}"

# ── Cluster image discovery ───────────────────────────────────────────────────
# Greene keeps images under /scratch/work/public, Torch under /share/apps.
# Both are searched, so the same scripts run on either cluster. Override by
# exporting SIF / OVERLAY_SRC before sourcing this file.
_mdg_find_sif() {
  local d cand
  for d in /share/apps/images /scratch/work/public/singularity; do
    [ -d "$d" ] || continue
    cand=$(ls -1 "$d"/cuda*.sif 2>/dev/null | sort -V | tail -1)
    [ -n "$cand" ] && { echo "$cand"; return 0; }
  done
  return 1
}

_mdg_find_overlay_src() {
  local d cand
  for d in /share/apps/overlay-fs-ext3 /scratch/work/public/overlay-fs-ext3; do
    [ -d "$d" ] || continue
    # Prefer the 15GB/500K image; fall back to any overlay in that directory.
    cand=$(ls -1 "$d"/overlay-15GB-500K.ext3.gz 2>/dev/null | head -1)
    [ -z "$cand" ] && cand=$(ls -1 "$d"/overlay-*.ext3.gz 2>/dev/null | head -1)
    [ -n "$cand" ] && { echo "$cand"; return 0; }
  done
  return 1
}

export SIF="${SIF:-$(_mdg_find_sif || true)}"
export OVERLAY_SRC="${OVERLAY_SRC:-$(_mdg_find_overlay_src || true)}"

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
