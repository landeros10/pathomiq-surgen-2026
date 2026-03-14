#!/usr/bin/env bash
# Run Phase 4 stability-study configs sequentially on GCP.
#
# Usage:
#   # All 8 Round 1 configs (default)
#   nohup ./scripts/run_stability_study.sh > logs/stability/orchestrator.log 2>&1 &
#
#   # Specific subset (e.g. Round 2 top-4)
#   ./scripts/run_stability_study.sh configs/studies/config_surgen_cosine.yaml configs/studies/config_surgen_all.yaml
#
# Each config is run sequentially; the outer loop survives SSH disconnect.
# Ctrl+C during the tail exits the tail but leaves the current run alive.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
mkdir -p logs/stability

# Default: all 8 Round 1 configs
DEFAULT_CONFIGS=(
    configs/studies/config_surgen_none.yaml
    configs/studies/config_surgen_cosine.yaml
    configs/studies/config_surgen_weighted.yaml
    configs/studies/config_surgen_accum.yaml
    configs/studies/config_surgen_cosine_weighted.yaml
    configs/studies/config_surgen_cosine_accum.yaml
    configs/studies/config_surgen_weighted_accum.yaml
    configs/studies/config_surgen_all.yaml
)

if [ "$#" -gt 0 ]; then
    CONFIGS=("$@")
else
    CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

TOTAL=${#CONFIGS[@]}
echo "Stability study: ${TOTAL} config(s) to run sequentially"
echo "────────────────────────────────────────"

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    stem="$(basename "$cfg" .yaml)"
    log="logs/stability/${stem}.log"
    run_num=$((i + 1))

    echo ""
    echo "[${run_num}/${TOTAL}] Starting: ${cfg}"
    echo "  Log: ${log}"

    start_ts=$(date +%s)

    nohup python3 scripts/train.py --config "$cfg" --run-suffix round_1 \
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
