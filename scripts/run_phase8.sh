#!/usr/bin/env bash
# Run Phase 8 ablation configs sequentially on GCP.
# 6 configs × 3 seeds = 18 total runs.
#
# Usage:
#   # Preflight: 1 epoch each, seed 0 only (6 runs)
#   ./scripts/run_phase8.sh --preflight --seeds "0"
#
#   # Full runs (background, survives SSH disconnect)
#   nohup ./scripts/run_phase8.sh > logs/phase8/orchestrator.log 2>&1 &
#   tail -f logs/phase8/orchestrator.log
#
#   # Custom seeds
#   ./scripts/run_phase8.sh --seeds "0 1 2"
#
#   # With early-stopping patience override
#   ./scripts/run_phase8.sh --patience 50
#
# Flags are order-independent; configs run sequentially within each seed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
mkdir -p logs/phase8

# Parse flags: --preflight, --patience N, --seeds "0 1 2" (order-independent)
PREFLIGHT=0
EXTRA_ARGS=()
LOG_SUFFIX=""
SEEDS="0 1 2"

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
        --seeds)
            [ $# -gt 1 ] || { echo "Error: --seeds requires a value"; exit 1; }
            SEEDS="$2"
            shift 2
            ;;
        *)
            echo "Error: unknown option $1"
            exit 1
            ;;
    esac
done

CONFIGS=(
    configs/phase8/config_phase8_baseline.yaml
    configs/phase8/config_phase8_mlp_rpb.yaml
    configs/phase8/config_phase8_dropout10.yaml
    configs/phase8/config_phase8_dropout25.yaml
    configs/phase8/config_phase8_mlp_rpb_dropout10.yaml
    configs/phase8/config_phase8_mlp_rpb_dropout25.yaml
)

# Convert seeds string to array
read -ra SEED_ARR <<< "$SEEDS"

TOTAL_RUNS=$(( ${#CONFIGS[@]} * ${#SEED_ARR[@]} ))

if [ "$PREFLIGHT" -eq 1 ]; then
    echo "════════════════════════════════════════"
    echo "  PREFLIGHT MODE: 1 epoch per run"
    echo "════════════════════════════════════════"
fi

echo "Phase 8: ${#CONFIGS[@]} config(s) × ${#SEED_ARR[@]} seed(s) = ${TOTAL_RUNS} total runs"
echo "Seeds: ${SEEDS}"
echo "────────────────────────────────────────"

run_num=0

for cfg in "${CONFIGS[@]}"; do
    stem="$(basename "$cfg" .yaml)"
    for seed in "${SEED_ARR[@]}"; do
        run_num=$(( run_num + 1 ))
        run_name="${stem}-s${seed}"
        log="logs/phase8/${run_name}${LOG_SUFFIX}.log"

        echo ""
        echo "[${run_num}/${TOTAL_RUNS}] Starting: ${cfg}  seed=${seed}"
        echo "  Run name: ${run_name}"
        echo "  Log: ${log}"

        start_ts=$(date +%s)

        nohup python3 scripts/train.py \
            --config "$cfg" \
            --seed "$seed" \
            --run-name "$run_name" \
            "${EXTRA_ARGS[@]}" \
            > "$log" 2>&1 &
        pid=$!
        echo "  PID: ${pid}"
        echo "────────────────────────────────────────"

        wait "$pid"
        exit_code=$?

        end_ts=$(date +%s)
        elapsed=$(( end_ts - start_ts ))
        mins=$(( elapsed / 60 ))
        secs=$(( elapsed % 60 ))

        if [ "$exit_code" -eq 0 ]; then
            echo "[${run_num}/${TOTAL_RUNS}] Done: ${run_name}  (${mins}m ${secs}s)"
        else
            echo "[${run_num}/${TOTAL_RUNS}] FAILED (exit ${exit_code}): ${run_name}  (${mins}m ${secs}s)"
            echo "  See ${log} for details"
        fi
    done
done

echo ""
echo "All ${TOTAL_RUNS} run(s) complete."
