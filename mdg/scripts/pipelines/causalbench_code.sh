#!/usr/bin/env bash
# Pipeline: namesarnav/causalbench:code
# Fine-tune BERT + RoBERTa, then run all TextAttack recipes.
# Run independently in parallel with other dataset scripts.

export DS="namesarnav/causalbench:code"
export LABELS="YES,NO"
export NUM_ATTACK=100000

source "$(dirname "$0")/_template.sh"
