#!/usr/bin/env bash
# natquest — full attack pipeline on this Mac.
#
#   3 models x 18 recipes, on all 13500 test rows. No cap, no query budget.
#   Task ids: 9 20 31
#
# Usage:
#   bash mac-scripts/natquest.sh
#
# Safe to re-run: every finished recipe is skipped via its done-marker, so an
# interrupted run picks up where it stopped. Prevent sleep on a laptop with:
#   caffeinate -i bash mac-scripts/natquest.sh

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_dataset 9
