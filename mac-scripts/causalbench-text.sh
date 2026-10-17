#!/usr/bin/env bash
# causalbench_text — full attack pipeline on this Mac.
#
#   3 models x 18 recipes, on all 7530 test rows. No cap, no query budget.
#   Task ids: 5 16 27
#
# Usage:
#   bash mac-scripts/causalbench-text.sh
#
# Safe to re-run: every finished recipe is skipped via its done-marker, so an
# interrupted run picks up where it stopped. Prevent sleep on a laptop with:
#   caffeinate -i bash mac-scripts/causalbench-text.sh

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_dataset 5
