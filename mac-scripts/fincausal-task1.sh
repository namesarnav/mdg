#!/usr/bin/env bash
# fincausal-task1 — full attack pipeline on this Mac.
#
#   3 models x 18 recipes, on all 3750 test rows. No cap, no query budget.
#   Task ids: 8 19 30
#
# Usage:
#   bash mac-scripts/fincausal-task1.sh
#
# Safe to re-run: every finished recipe is skipped via its done-marker, so an
# interrupted run picks up where it stopped. Prevent sleep on a laptop with:
#   caffeinate -i bash mac-scripts/fincausal-task1.sh

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_dataset 8
