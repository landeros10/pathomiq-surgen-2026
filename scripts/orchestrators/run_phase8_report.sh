#!/usr/bin/env bash
# Run Phase 8 ablation report locally after downloading phase8-data/.
#
# Usage (after scp and unzip of phase8-data.zip):
#   bash scripts/run_phase8_report.sh --data-dir phase8-data/
#   bash scripts/run_phase8_report.sh --data-dir phase8-data/ --n-bootstrap 5000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Defaults
DATA_DIR=""
N_BOOTSTRAP=2000
MLFLOW_DB="mlflow.db"

while [ $# -gt 0 ]; do
    case "$1" in
        --data-dir)
            [ $# -gt 1 ] || { echo "Error: --data-dir requires a value"; exit 1; }
            DATA_DIR="$2"
            shift 2
            ;;
        --n-bootstrap)
            [ $# -gt 1 ] || { echo "Error: --n-bootstrap requires a value"; exit 1; }
            N_BOOTSTRAP="$2"
            shift 2
            ;;
        --mlflow-db)
            [ $# -gt 1 ] || { echo "Error: --mlflow-db requires a value"; exit 1; }
            MLFLOW_DB="$2"
            shift 2
            ;;
        *)
            echo "Error: unknown option $1"
            echo "Usage: $0 --data-dir <dir> [--n-bootstrap N] [--mlflow-db path]"
            exit 1
            ;;
    esac
done

if [ -z "$DATA_DIR" ]; then
    echo "Error: --data-dir is required"
    echo "Usage: $0 --data-dir <dir> [--n-bootstrap N]"
    exit 1
fi

echo "Phase 8 report"
echo "  MLflow DB  : $MLFLOW_DB"
echo "  Data dir   : $DATA_DIR"
echo "  N bootstrap: $N_BOOTSTRAP"
echo "────────────────────────────────────────"

source surgen-env/bin/activate 2>/dev/null || true

python3 scripts/studies/phase8_ablation.py \
    --mlflow-db "$MLFLOW_DB" \
    --data-dir "$DATA_DIR" \
    --n-bootstrap "$N_BOOTSTRAP"
