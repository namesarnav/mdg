#!/usr/bin/env bash
# e-care — full attack pipeline on this Mac.
#
#   3 models x 18 recipes, on all 2122 test rows. No cap, no query budget.
#   Task ids: 7 18 29
#
# Usage:
#   bash mac-scripts/e-care.sh
#
# Safe to re-run: every finished recipe is skipped via its done-marker, so an
# interrupted run picks up where it stopped. Prevent sleep on a laptop with:
#   caffeinate -i bash mac-scripts/e-care.sh

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_dataset 7
