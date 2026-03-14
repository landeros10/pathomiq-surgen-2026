#!/usr/bin/env bash
# Run Phase 8 server-side inference extraction on GCP.
# Survives SSH disconnect via nohup.
#
# Usage on GCP:
#   bash scripts/run_phase8_extract.sh
#
# After completion, download and extract locally:
#   zip -r phase8-data.zip phase8-data/
#   scp chris@136.109.153.16:~/surgen/phase8-data.zip ~/
#   unzip phase8-data.zip

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

mkdir -p logs/phase8

LOG="logs/phase8/extract.log"
OUT_DIR="phase8-data"
MLFLOW_DB="mlflow.db"
EMB_DIR="/mnt/data-surgen/embeddings"

echo "Phase 8 extraction starting …"
echo "  Output dir : $OUT_DIR"
echo "  MLflow DB  : $MLFLOW_DB"
echo "  Embeddings : $EMB_DIR"
echo "  Log        : $LOG"
echo ""

nohup python3 scripts/orchestrators/phase8_extract.py \
    --mlflow-db "$MLFLOW_DB" \
    --embeddings-dir "$EMB_DIR" \
    --out-dir "$OUT_DIR" \
    > "$LOG" 2>&1 &

PID=$!
echo "PID: $PID"
echo "Tailing log (Ctrl+C to detach — extraction continues in background):"
echo "────────────────────────────────────────"
tail -f "$LOG"
