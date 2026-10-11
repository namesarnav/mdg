#!/usr/bin/env bash
# Train on a free GPU (Colab, Kaggle, or any CUDA box) and push to the Hub.
#
#   HF_TOKEN=hf_xxx bash mdg/hpc/train_gpu.sh "6 17"
#
# On Colab/Kaggle, where the repo is not cloned yet, one cell does everything:
#   !git clone -q https://github.com/namesarnav/mdg.git /content/mdg
#   %env HF_TOKEN=hf_xxx
#   !cd /content/mdg && bash mdg/hpc/train_gpu.sh "6 17"
#
# Why GPU for training but not for attacks: training is one big batched matmul
# stream, which a T4 does ~50x faster than 64 CPU cores. The attacks are 13M
# sequential adversarial searches — one T4 would take months, while a CPU fleet
# runs them 192 at a time. Train here, attack on the cluster.
#
# Uses the SAME code path as the cluster (run_parallel.sh train -> run_one.sh),
# so the models are identical to CPU-trained ones apart from fp16 arithmetic.
set -euo pipefail

TASKS="${1:?usage: train_gpu.sh \"<task ids>\"   e.g. \"6 17\"   (see mdg/hpc/README.md)}"
HPC_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(cd "$HPC_DIR/../.." && pwd)"
cd "$PROJECT"

echo "========================================================"
echo "  PROJECT : $PROJECT"
echo "  TASKS   : $TASKS"
python - <<'PY'
import torch
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}   : {p.name}, {p.total_memory/1e9:.0f}GB")
    if n > 1:
        print(f"  -> {n} GPUs: training {n} tasks at once, one per GPU.")
else:
    print("  GPU     : NONE DETECTED — this will be as slow as the CPU boxes.")
    print("            On Colab: Runtime > Change runtime type > T4 GPU.")
    raise SystemExit(1)
PY
echo "========================================================"

# ── Credentials ───────────────────────────────────────────────────────────────
# Checked up front: the Trainer creates the Hub repo at startup and uploads at
# the end, so a missing token otherwise wastes the whole run.
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.hf_token" ]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.hf_token")"
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "[FAIL] Set HF_TOKEN first — the trained model has to be pushed somewhere."
  echo "       Colab:  %env HF_TOKEN=hf_xxx"
  echo "       Kaggle: Add-ons > Secrets, then os.environ['HF_TOKEN']=..."
  exit 1
fi
export HF_TOKEN
export HUGGINGFACE_TOKEN="$HF_TOKEN"
echo "$HF_TOKEN" > "$HOME/.hf_token" && chmod 600 "$HOME/.hf_token"

# ── Dependencies ──────────────────────────────────────────────────────────────
# torch is deliberately NOT installed: the hosted image ships a build matched to
# its CUDA driver, and replacing it usually breaks GPU support. textattack is
# not needed either — this script only trains.
echo "[1/3] Dependencies"
pip install -q "transformers==4.50.3" "accelerate>=1.0" "datasets>=3.0,<4" \
  "scikit-learn>=1.5" sentencepiece "huggingface_hub>=0.26"

# ── Data ──────────────────────────────────────────────────────────────────────
echo "[2/3] Data (Git LFS, ~840MB)"
command -v git-lfs >/dev/null 2>&1 || (apt-get -qq update && apt-get -qq install -y git-lfs)
git lfs install --local >/dev/null 2>&1 || git lfs install >/dev/null 2>&1 || true
git lfs pull
# A pointer file is 3 lines; real data is not. Catch it before training.
for f in mdg/finetune/data/*__train.jsonl; do
  if head -c 60 "$f" | grep -q "git-lfs.github.com/spec"; then
    echo "[FAIL] $f is still an LFS pointer. 'git lfs pull' did not complete."
    exit 1
  fi
done

# ── Train ─────────────────────────────────────────────────────────────────────
# fp16 switches on automatically under CUDA (base.py). Larger batches than the
# CPU path because VRAM, not bandwidth, is the limit here.
echo "[3/3] Training"
export MDG_RUNTIME=direct
# One task per GPU. More than that on a single GPU gains nothing — a bert-base
# job already saturates it — and risks running out of VRAM, so JOBS is left to
# run_parallel.sh's GPU count unless explicitly overridden.
ONLY_TASKS="$TASKS" JOBS="${JOBS:-}" THREADS_PER_JOB=2 \
  TRAIN_ARGS="${TRAIN_ARGS:---epochs 3 --batch 32 --lr 2e-5}" \
  bash mdg/hpc/run_parallel.sh train

echo ""
echo "  Per-task logs: $PROJECT/mdg/logs/train-task*.log"
echo "  Verify the pushes landed real weights:"
echo "    bash mdg/hpc/preflight.sh"
