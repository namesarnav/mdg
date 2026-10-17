#!/usr/bin/env bash
# Progress across every dataset, plus what this Mac is doing right now.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

python "$ROOT/mdg/hpc/status.py"

echo
echo "── this machine ──────────────────────────────────────────────"
n=$(pgrep -fc run_one.sh 2>/dev/null || echo 0)
echo "  attack workers : $n"
echo "  load           : $(sysctl -n vm.loadavg | tr -d '{}')"
if [ "$n" -gt 0 ]; then
  echo "  slowest recent :"
  grep -ho "[0-9.]*s/it" "$ROOT"/mac-scripts/logs/cells-task*.log 2>/dev/null \
    | sort -rn | head -3 | sed 's/^/      /'
fi
