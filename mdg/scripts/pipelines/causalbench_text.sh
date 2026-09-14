#!/usr/bin/env bash
# Pipeline: namesarnav/causalbench:text
# Fine-tune BERT + RoBERTa, then run all TextAttack recipes.
# Run independently in parallel with other dataset scripts.

export DS="namesarnav/causalbench:text"
export LABELS="YES,NO"
export NUM_ATTACK=200

source "$(dirname "$0")/_template.sh"
