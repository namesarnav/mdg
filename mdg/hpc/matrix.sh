#!/usr/bin/env bash
# The full experiment grid: every model × every dataset.
# Sourced by train.sbatch and attack.sbatch so both iterate the same grid.

# "stem|label_space"  — stem matches mdg/finetune/data/<stem>__{train,test}.jsonl
DATASETS=(
  "namesarnav_counterbench|YES,NO"
  "namesarnav_ac-reason|YES,NO"
  "namesarnav_bbh-causal-judgement|YES,NO"
  "namesarnav_causalbench_code|YES,NO"
  "namesarnav_causalbench_math|YES,NO"
  "namesarnav_causalbench_text|YES,NO"
  "namesarnav_corr2cause|0,1"
  "namesarnav_e-care|0,1"
  "namesarnav_fincausal-task1|0,1"
  "namesarnav_natquest|YES,NO"
  "namesarnav_Quriosity|YES,NO"
)

# "module|checkpoint_dir_name|hf_model_id|extra_train_args"
# checkpoint_dir_name must equal basename(hf_model_id) — that is the directory
# mdg/models/base*.py writes into.
MODELS=(
  "mdg.models.bert|bert-base-uncased|bert-base-uncased|--epochs 5 --batch 16 --lr 2e-5"
  "mdg.models.roberta|roberta-base|roberta-base|--epochs 5 --batch 16 --lr 2e-5"
)

# Previously run; re-enable by moving back into MODELS above.
#   "mdg.models.t5|t5-base|t5-base|--epochs 5 --batch 16 --lr 3e-4"
#   "mdg.models.llama|Llama-3.2-1B|meta-llama/Llama-3.2-1B|--epochs 3 --batch 4 --lr 2e-4 --lora-r 16 --no-4bit"

N_DATASETS=${#DATASETS[@]}
N_MODELS=${#MODELS[@]}
N_JOBS=$(( N_DATASETS * N_MODELS ))

# Map a flat SLURM_ARRAY_TASK_ID onto (model, dataset) and export the pieces.
# Sets: MODULE MODEL_NAME HF_MODEL EXTRA_ARGS STEM LABELS DATASET_STEM HUB_ID
resolve_task() {
  local idx="$1"
  local m_idx=$(( idx / N_DATASETS ))
  local d_idx=$(( idx % N_DATASETS ))

  IFS='|' read -r MODULE MODEL_NAME HF_MODEL EXTRA_ARGS <<< "${MODELS[$m_idx]}"
  IFS='|' read -r STEM LABELS <<< "${DATASETS[$d_idx]}"

  DATASET_STEM="${STEM#namesarnav_}"
  HUB_ID="namesarnav/${DATASET_STEM}-${MODEL_NAME}"
  export MODULE MODEL_NAME HF_MODEL EXTRA_ARGS STEM LABELS DATASET_STEM HUB_ID
}
