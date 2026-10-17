#!/usr/bin/env bash
# Quriosity — full attack pipeline on this Mac.
#
#   3 models x 18 recipes, on all 13500 test rows. No cap, no query budget.
#   Task ids: 10 21 32
#
# Usage:
#   bash mac-scripts/quriosity.sh
#
# Safe to re-run: every finished recipe is skipped via its done-marker, so an
# interrupted run picks up where it stopped. Prevent sleep on a laptop with:
#   caffeinate -i bash mac-scripts/quriosity.sh

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_dataset 10
