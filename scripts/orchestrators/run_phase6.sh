#!/usr/bin/env bash
# Run Phase 6 configs sequentially on GCP.
#
# Usage:
#   # Preflight: 1 epoch each to verify configs load and train
#   ./scripts/run_phase6.sh --preflight
#
#   # Full runs (background, survives SSH disconnect)
#   nohup ./scripts/run_phase6.sh > logs/phase6/orchestrator.log 2>&1 &
#   tail -f logs/phase6/orchestrator.log
#
#   # With early-stopping patience override
#   ./scripts/run_phase6.sh --patience 50
#
# Flags are order-independent; each config runs sequentially.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
mkdir -p logs/phase6

# Parse flags: --preflight, --patience N (order-independent)
PREFLIGHT=0
EXTRA_ARGS=()
LOG_SUFFIX=""
while [ $# -gt 0 ]; do
    case "$1" in
        --preflight)
            PREFLIGHT=1
            EXTRA_ARGS+=(--max-epochs 1 --max-samples 20 --run-suffix preflight)
            LOG_SUFFIX="_preflight"
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

CONFIGS=(
    configs/phase6/config_multitask_mean.yaml
    configs/phase6/config_multitask_mean_pe.yaml
    configs/phase6/config_multitask_abmil_nope.yaml
    configs/phase6/config_multitask_abmil.yaml
    configs/phase6/config_multitask_abmil_joined.yaml
    configs/phase6/config_multitask_abmil_joined_pe.yaml
    configs/phase6/config_singletask_mmr_abmil.yaml
)

if [ "$PREFLIGHT" -eq 1 ]; then
    echo "════════════════════════════════════════"
    echo "  PREFLIGHT MODE: 1 epoch per config"
    echo "════════════════════════════════════════"
fi

TOTAL=${#CONFIGS[@]}
echo "Phase 6: ${TOTAL} config(s) to run sequentially"
echo "────────────────────────────────────────"

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    stem="$(basename "$cfg" .yaml)"
    log="logs/phase6/${stem}${LOG_SUFFIX}.log"
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

    wait "$pid" && exit_code=0 || exit_code=$?

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
