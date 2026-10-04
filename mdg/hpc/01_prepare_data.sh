#!/usr/bin/env bash
# Regenerate mdg/finetune/data/*.jsonl — the train/test splits every job reads.
#
# You normally do NOT need this: the splits are committed to the repo. Run it
# only to refresh them from the Hub, or after adding a new dataset. Downloads
# datasets, so run it on the login node (which has internet) or a compute node.
#
#   bash mdg/hpc/01_prepare_data.sh
set -euo pipefail

source "$(dirname "$0")/env.sh"
cd "$PROJECT"

in_container "python -m mdg.scripts.prepare_finetune_data \
  --datasets \
    'namesarnav/causalbench:code' \
    'namesarnav/causalbench:math' \
    'namesarnav/causalbench:text' \
    namesarnav/corr2cause \
    namesarnav/e-care \
    namesarnav/fincausal-task1 \
    namesarnav/natquest \
    namesarnav/Quriosity \
  --local-files \
    'namesarnav_counterbench:mdg/synthetic/data/counterbench_task2_v2.jsonl' \
    'namesarnav_ac-reason:mdg/synthetic/data/ac_reason_task2.jsonl' \
    'namesarnav_bbh-causal-judgement:mdg/synthetic/data/bbh_causal_judgement_task2.jsonl' \
  --output-dir mdg/finetune/data"

echo
echo "Splits in mdg/finetune/data:"
ls -1 mdg/finetune/data | wc -l
