#!/usr/bin/env bash
# counterbench — full attack pipeline on this Mac.
#
#   3 models x 18 recipes, on all 330 test rows. No cap, no query budget.
#   Task ids: 0 11 22
#
# Usage:
#   bash mac-scripts/counterbench.sh
#
# Safe to re-run: every finished recipe is skipped via its done-marker, so an
# interrupted run picks up where it stopped. Prevent sleep on a laptop with:
#   caffeinate -i bash mac-scripts/counterbench.sh

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_dataset 0
