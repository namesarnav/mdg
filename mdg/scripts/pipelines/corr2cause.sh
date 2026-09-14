#!/usr/bin/env bash
# Pipeline: namesarnav/corr2cause
# Fine-tune BERT + RoBERTa, then run all TextAttack recipes.
# Run independently in parallel with other dataset scripts.

export DS="namesarnav/corr2cause"
export LABELS="0,1"
export NUM_ATTACK=200

source "$(dirname "$0")/_template.sh"
