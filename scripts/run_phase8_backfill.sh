#!/usr/bin/env bash
# Backfill Phase 6 multitask configs into multitask-surgen-phase8 experiment.
# 5 configs × 3 seeds = 15 runs. Skips any run already FINISHED in MLflow.
#
# Usage:
#   # Preview (no runs launched)
#   ./scripts/run_phase8_backfill.sh --dry-run
#
#   # Full backfill (background, survives SSH disconnect)
#   nohup ./scripts/run_phase8_backfill.sh > logs/phase8/backfill/orchestrator.log 2>&1 &
#   tail -f logs/phase8/backfill/orchestrator.log
#
#   # Custom seeds
#   ./scripts/run_phase8_backfill.sh --seeds "0 1 2"
#
#   # With early-stopping patience override
#   ./scripts/run_phase8_backfill.sh --patience 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
mkdir -p logs/phase8/backfill

# ── Parse flags ──────────────────────────────────────────────────────────────
EXTRA_ARGS=()
SEEDS="0 1 2"
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
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
    configs/phase8/backfill/config_phase6_abmil_nope.yaml
    configs/phase8/backfill/config_phase6_abmil.yaml
    configs/phase8/backfill/config_phase6_abmil_joined_pe.yaml
    configs/phase8/backfill/config_phase6_mean.yaml
    configs/phase8/backfill/config_phase6_mean_pe.yaml
)

# ── MLflow skip-check (query once at start) ──────────────────────────────────
MLFLOW_DB="$PROJECT_ROOT/mlflow.db"
FINISHED_RUNS=""

if [ -f "$MLFLOW_DB" ]; then
    EXP_ID=$(sqlite3 "$MLFLOW_DB" \
        "SELECT experiment_id FROM experiments WHERE name='multitask-surgen-phase8';" 2>/dev/null || true)
    if [ -n "$EXP_ID" ]; then
        FINISHED_RUNS=$(sqlite3 "$MLFLOW_DB" \
            "SELECT name FROM runs WHERE experiment_id='$EXP_ID' AND status='FINISHED' \
             AND name NOT LIKE '%preflight%';" 2>/dev/null || true)
    fi
fi

# ── Summary ──────────────────────────────────────────────────────────────────
read -ra SEED_ARR <<< "$SEEDS"
TOTAL_RUNS=$(( ${#CONFIGS[@]} * ${#SEED_ARR[@]} ))

if [ "$DRY_RUN" -eq 1 ]; then
    echo "════════════════════════════════════════"
    echo "  DRY RUN — no runs will be launched"
    echo "════════════════════════════════════════"
fi

echo "Phase 8 backfill: ${#CONFIGS[@]} config(s) × ${#SEED_ARR[@]} seed(s) = ${TOTAL_RUNS} total runs"
echo "Seeds: ${SEEDS}"
echo "────────────────────────────────────────"

run_num=0
launched=0
skipped=0

for cfg in "${CONFIGS[@]}"; do
    stem="$(basename "$cfg" .yaml)"
    for seed in "${SEED_ARR[@]}"; do
        run_num=$(( run_num + 1 ))
        run_name="${stem}-s${seed}"
        log="logs/phase8/backfill/${run_name}.log"

        echo ""
        echo "[${run_num}/${TOTAL_RUNS}] ${cfg}  seed=${seed}"
        echo "  Run name: ${run_name}"

        # Skip if already FINISHED in MLflow
        if echo "$FINISHED_RUNS" | grep -qx "${run_name}"; then
            echo "  [SKIP] ${run_name} — already FINISHED in MLflow"
            skipped=$(( skipped + 1 ))
            continue
        fi

        if [ "$DRY_RUN" -eq 1 ]; then
            echo "  [DRY RUN] would launch: ${run_name}"
            continue
        fi

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

        wait "$pid" && exit_code=0 || exit_code=$?

        end_ts=$(date +%s)
        elapsed=$(( end_ts - start_ts ))
        mins=$(( elapsed / 60 ))
        secs=$(( elapsed % 60 ))

        if [ "$exit_code" -eq 0 ]; then
            launched=$(( launched + 1 ))
            echo "[${run_num}/${TOTAL_RUNS}] Done: ${run_name}  (${mins}m ${secs}s)"
        else
            echo "[${run_num}/${TOTAL_RUNS}] FAILED (exit ${exit_code}): ${run_name}  (${mins}m ${secs}s)"
            echo "  See ${log} for details"
        fi
    done
done

echo ""
if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run complete — 0 runs launched."
else
    echo "All done: ${launched} launched, ${skipped} skipped (already FINISHED)."
fi
