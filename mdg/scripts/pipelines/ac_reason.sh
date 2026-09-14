#!/usr/bin/env bash
# Pipeline: namesarnav/ac-reason
# Fine-tune BERT + RoBERTa, then run all TextAttack recipes.
# Run independently in parallel with other dataset scripts.

export DS="namesarnav/ac-reason"
export LABELS="YES,NO"
export NUM_ATTACK=200

source "$(dirname "$0")/_template.sh"
