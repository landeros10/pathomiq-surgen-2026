#!/usr/bin/env bash
# Run Phase 3 baseline: 200-epoch training on combined SurGen dataset,
# followed by automated report generation.
#
# Usage:
#   ./scripts/run_phase3.sh
#   ./scripts/run_phase3.sh --config configs/config_gcp.yaml
#   ./scripts/run_phase3.sh --config configs/config_gcp.yaml --run-suffix 2
#
# Ctrl+C exits the tail — training and report generation keep running.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# ── Parse --config (consume it; forward remaining args to train.py) ───────────
CONFIG="configs/config_gcp.yaml"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

mkdir -p logs reports
touch reports/.gitkeep

LOG="logs/phase3.log"

echo "Starting Phase 3 baseline run..."
echo "Config : $CONFIG"
echo "Log    : $LOG"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "Extra  : ${EXTRA_ARGS[*]}"
echo "────────────────────────────────────────"

# Launch training + report as a single backgrounded chain that survives
# SSH disconnect, tmux kill, and Ctrl+C.
nohup bash -c "
    python3 scripts/train.py --config '$CONFIG' ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"} \
        && python3 scripts/make_phase3_report.py --config '$CONFIG'
" >> "$LOG" 2>&1 &

echo "PID    : $!"
echo "────────────────────────────────────────"
tail -f "$LOG"
