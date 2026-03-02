#!/usr/bin/env bash
# Launch a training run on the GCP server in the background.
#
# Usage:
#   ./scripts/run_train.sh                          # use config defaults
#   ./scripts/run_train.sh --run-name my-run        # override mlflow run name
#   ./scripts/run_train.sh --config configs/other.yaml --run-name my-run
#
# Output is streamed to logs/train.log and tailed automatically.
# Ctrl+c exits the tail — training keeps running in the background.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
mkdir -p logs

LOG="logs/train.log"

echo "Starting training run..."
echo "Log: $LOG"
echo "Args: $*"
echo "────────────────────────────────────────"

nohup python3 scripts/train.py --config configs/config_gcp.yaml "$@" \
    > "$LOG" 2>&1 &

echo "PID: $!"
echo "────────────────────────────────────────"
tail -f "$LOG"
