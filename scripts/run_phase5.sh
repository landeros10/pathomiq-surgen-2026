#!/usr/bin/env bash
# Run Phase 5 multitask configs sequentially on GCP.
#
# Usage:
#   # Preflight: 1 epoch each to verify configs load and train
#   ./scripts/run_phase5.sh --preflight
#
#   # Full runs (background, survives SSH disconnect)
#   nohup ./scripts/run_phase5.sh > logs/phase5/orchestrator.log 2>&1 &
#   tail -f logs/phase5/orchestrator.log
#
# Each config is run sequentially; the outer loop survives SSH disconnect.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
mkdir -p logs/phase5

# Parse --preflight flag
PREFLIGHT=0
EXTRA_ARGS=()
LOG_SUFFIX=""
if [ "${1:-}" == "--preflight" ]; then
    PREFLIGHT=1
    EXTRA_ARGS=(--max-epochs 1 --run-suffix preflight)
    LOG_SUFFIX="_preflight"
    shift
    echo "════════════════════════════════════════"
    echo "  PREFLIGHT MODE: 1 epoch per config"
    echo "════════════════════════════════════════"
fi

CONFIGS=(
    configs/phase5/config_multitask_base.yaml
    configs/phase5/config_multitask_cosine_accum16.yaml
)

TOTAL=${#CONFIGS[@]}
echo "Phase 5: ${TOTAL} config(s) to run sequentially"
echo "────────────────────────────────────────"

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    stem="$(basename "$cfg" .yaml)"
    log="logs/phase5/${stem}${LOG_SUFFIX}.log"
    run_num=$((i + 1))

    echo ""
    echo "[${run_num}/${TOTAL}] Starting: ${cfg}"
    echo "  Log: ${log}"

    start_ts=$(date +%s)

    nohup python3 scripts/train.py --config "$cfg" "${EXTRA_ARGS[@]}" \
        > "$log" 2>&1 &
    pid=$!
    echo "  PID: ${pid}"
    echo "────────────────────────────────────────"

    # Wait for this run to finish before starting the next
    wait "$pid"
    exit_code=$?

    end_ts=$(date +%s)
    elapsed=$(( end_ts - start_ts ))
    mins=$(( elapsed / 60 ))
    secs=$(( elapsed % 60 ))

    if [ "$exit_code" -eq 0 ]; then
        echo "[${run_num}/${TOTAL}] Done: ${stem}  (${mins}m ${secs}s)"
    else
        echo "[${run_num}/${TOTAL}] FAILED (exit ${exit_code}): ${stem}  (${mins}m ${secs}s)"
        echo "  See ${log} for details"
    fi
done

echo ""
echo "All ${TOTAL} run(s) complete."
