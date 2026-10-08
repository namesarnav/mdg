#!/usr/bin/env bash
# Shared environment for all NYU Greene jobs. Sourced by every sbatch script.
#
# Edit PROJECT if you clone the repo somewhere other than /scratch/$USER/mdg.

# ── Paths ─────────────────────────────────────────────────────────────────────
# Repo root. Default to the checkout this file lives in (mdg/hpc/env.sh ->
# two levels up), which is right everywhere; /scratch/$USER/mdg is only used
# when that checkout cannot be located.
_mdg_repo_root() {
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"
  if [ -n "$here" ] && [ -d "$here/mdg/hpc" ]; then
    echo "$here"
  else
    echo "/scratch/$USER/mdg"
  fi
}
export PROJECT="${PROJECT:-$(_mdg_repo_root)}"
export OVERLAY="${OVERLAY:-/scratch/$USER/mdg-env/overlay-15GB-500K.ext3}"

# ── Cluster image discovery ───────────────────────────────────────────────────
# Greene keeps images under /scratch/work/public, Torch under /share/apps.
# Both are searched, so the same scripts run on either cluster. Override by
# exporting SIF / OVERLAY_SRC before sourcing this file.
# Known-good images first (torch 2.5.1 wheels are built against CUDA 12.x; the
# newest image on a cluster is often CUDA 13, which is not what we want here).
_MDG_PREFERRED_SIFS=(
  cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
  cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
  cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif
  cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
)

_mdg_find_sif() {
  local d p cand
  for d in /share/apps/images /scratch/work/public/singularity; do
    [ -d "$d" ] || continue
    for p in "${_MDG_PREFERRED_SIFS[@]}"; do
      [ -f "$d/$p" ] && { echo "$d/$p"; return 0; }
    done
  done
  # Nothing known — take the newest CUDA 12 image that also ships cuDNN.
  for d in /share/apps/images /scratch/work/public/singularity; do
    [ -d "$d" ] || continue
    cand=$(ls -1 "$d"/cuda12*cudnn*.sif 2>/dev/null | sort -V | tail -1)
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

# Caches belong on /scratch on a cluster (/home has a hard inode quota that
# HuggingFace caches blow through). Off-cluster there is no /scratch, so fall
# back to $HOME — writing to a non-existent /scratch makes TextAttack die with
# "Read-only file system" before the first attack.
_mdg_cache_root() {
  if [ -n "${MDG_CACHE_ROOT:-}" ]; then echo "$MDG_CACHE_ROOT"; return; fi
  if [ -d "/scratch/$USER" ] && [ -w "/scratch/$USER" ]; then
    echo "/scratch/$USER"
  elif mkdir -p "/scratch/$USER" 2>/dev/null; then
    echo "/scratch/$USER"
  else
    echo "$HOME/.cache/mdg"
  fi
}
MDG_CACHE_ROOT="$(_mdg_cache_root)"
mkdir -p "$MDG_CACHE_ROOT" 2>/dev/null || true
export MDG_CACHE_ROOT
export HF_HOME="${HF_HOME:-$MDG_CACHE_ROOT/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TA_CACHE_DIR="${TA_CACHE_DIR:-$MDG_CACHE_ROOT/textattack_cache}"
export TOKENIZERS_PARALLELISM=false

# ── Slurm ─────────────────────────────────────────────────────────────────────
# Torch requires an explicit GPU partition (sinfo -s lists them: l40s, a100,
# h100, h200, b200, rtx6000, and *_public / *_plus variants). Greene does not
# need one — set PARTITION="" there.
export PARTITION="${PARTITION:-a100}"
# Torch requires a project account (--account=torch_pr_xxx_yyy); without one
# every srun/sbatch fails with "Invalid Slurm account". Your PI registers it at
# https://projects.hpc.nyu.edu, then put it in ~/.slurm_account or export ACCOUNT.
# List the accounts you belong to with:  sacctmgr -nP show assoc user=$USER format=account
if [ -z "${ACCOUNT:-}" ] && [ -f "$HOME/.slurm_account" ]; then
  ACCOUNT="$(tr -d '[:space:]' < "$HOME/.slurm_account")"
fi
export ACCOUNT="${ACCOUNT:-}"
# Consolidation needs no GPU; defaults to the same partition so it always has a
# valid one. On Torch you can send it to CPU nodes with CPU_PARTITION=cpu_short.
export CPU_PARTITION="${CPU_PARTITION:-$PARTITION}"

# ── HuggingFace token (needed to download gated Llama + to push models) ───────
# Put your token in ~/.hf_token (chmod 600). Never commit it.
if [ -f "$HOME/.hf_token" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.hf_token")"
  export HUGGINGFACE_TOKEN="$HF_TOKEN"
fi

# ── Run a command in the project environment ──────────────────────────────────
# Two runtimes, picked automatically:
#   container — HPC clusters: singularity + an ext3 overlay holding the conda env
#   direct    — a plain GPU VM (GCP, RunPod, Lambda, a lab box): a normal conda
#               env, no container. Built by setup_vm.sh.
# Force one with MDG_RUNTIME=container|direct.
_mdg_detect_runtime() {
  if [ -n "${MDG_RUNTIME:-}" ]; then echo "$MDG_RUNTIME"; return; fi
  if command -v singularity >/dev/null 2>&1 && [ -f "${OVERLAY:-}" ] && [ -f "${SIF:-}" ]; then
    echo container
  else
    echo direct
  fi
}
export MDG_RUNTIME="$(_mdg_detect_runtime)"

# Conda env used by the direct runtime (setup_vm.sh creates it here).
export MDG_VENV="${MDG_VENV:-$HOME/mdg-env}"

in_container() {
  if [ "$MDG_RUNTIME" = "container" ]; then
    # Overlay mounted READ-ONLY so concurrent tasks can share it safely.
    singularity exec --nv \
      --overlay "${OVERLAY}:ro" \
      "$SIF" \
      /bin/bash -c "source /ext3/env.sh; cd $PROJECT; $*"
  else
    bash -c "
      if [ -f '$MDG_VENV/bin/activate' ]; then source '$MDG_VENV/bin/activate'; fi
      cd '$PROJECT'; $*"
  fi
}
