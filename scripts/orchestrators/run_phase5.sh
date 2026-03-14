#!/usr/bin/env bash
# Run Phase 5 multitask configs sequentially on GCP.
#
# Usage:
#   # Preflight: 1 epoch each to verify configs load and train
#   ./scripts/run_phase5.sh --preflight
#
#   # Weighted variants (per-head class weighting)
#   nohup ./scripts/run_phase5.sh --weighted > logs/phase5/orchestrator_weighted.log 2>&1 &
#   tail -f logs/phase5/orchestrator_weighted.log
#
#   # Weighted with early-stopping patience override
#   ./scripts/run_phase5.sh --weighted --patience 50
#
#   # Weighted preflight
#   ./scripts/run_phase5.sh --weighted --preflight
#
#   # Full runs (background, survives SSH disconnect)
#   nohup ./scripts/run_phase5.sh > logs/phase5/orchestrator.log 2>&1 &
#   tail -f logs/phase5/orchestrator.log
#
# Flags are order-independent; each config runs sequentially.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
mkdir -p logs/phase5

# Parse flags: --preflight, --weighted, --patience N (order-independent)
PREFLIGHT=0
WEIGHTED=0
EXTRA_ARGS=()
LOG_SUFFIX=""
while [ $# -gt 0 ]; do
    case "$1" in
        --preflight)
            PREFLIGHT=1
            EXTRA_ARGS+=(--max-epochs 1 --run-suffix preflight)
            LOG_SUFFIX="_preflight"
            shift
            ;;
        --weighted)
            WEIGHTED=1
            shift
            ;;
        --patience)
            [ $# -gt 1 ] || { echo "Error: --patience requires a value"; exit 1; }
            EXTRA_ARGS+=(--patience "$2")
            shift 2
            ;;
        *)
            echo "Error: unknown option $1"
            exit 1
            ;;
    esac
done

if [ "$PREFLIGHT" -eq 1 ]; then
    echo "════════════════════════════════════════"
    echo "  PREFLIGHT MODE: 1 epoch per config"
    echo "════════════════════════════════════════"
fi

if [ "$WEIGHTED" -eq 1 ]; then
    CONFIGS=(
        configs/phase5/config_multitask_base_weighted.yaml
        configs/phase5/config_multitask_cosine_accum16_weighted.yaml
    )
    echo "════════════════════════════════════════"
    echo "  WEIGHTED MODE: per-head class weighting"
    echo "════════════════════════════════════════"
else
    CONFIGS=(
        configs/phase5/config_multitask_base.yaml
        configs/phase5/config_multitask_cosine_accum16.yaml
    )
fi

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
