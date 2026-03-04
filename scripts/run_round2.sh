#!/usr/bin/env bash
# Successive Halving — Round 2 launcher.
#
# Accepts MLflow run names from Round 1 (paste directly from MLflow UI),
# derives config paths, and relaunches training with Round 2 settings.
#
# Usage:
#   nohup ./scripts/run_round2.sh \
#       mmr-surgen-s1-cosine-round_1 \
#       mmr-surgen-s1-all-round_1 \
#       mmr-surgen-s1-weighted-round_1 \
#       mmr-surgen-s1-none-round_1 \
#       > logs/stability/round2_orchestrator.log 2>&1 &

# ── Round 2 settings ──────────────────────────────────────────────────────────
MAX_EPOCHS=50
PATIENCE=15
RUN_SUFFIX="round_2"
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
mkdir -p logs/stability

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <mlflow_run_name> [<mlflow_run_name> ...]"
    echo "  Example: $0 mmr-surgen-s1-cosine-round_1 mmr-surgen-s1-all-round_1"
    exit 1
fi

# ── Validate all run names → config paths before starting any training ────────
CONFIGS=()
STEMS=()
for run_name in "$@"; do
    # Find config whose mlflow run_name matches (handles filename/run_name mismatches)
    cfg=$(grep -rl "run_name: \"${run_name}\"" configs/studies/ 2>/dev/null | head -1)
    if [ -z "$cfg" ]; then
        echo "ERROR: no config found with run_name '${run_name}' in configs/studies/"
        exit 1
    fi
    CONFIGS+=("$cfg")
    STEMS+=("$(basename "$cfg" .yaml)")
done

TOTAL=${#CONFIGS[@]}
echo "Round 2  --max-epochs ${MAX_EPOCHS}  --patience ${PATIENCE}  --run-suffix ${RUN_SUFFIX}"
echo "${TOTAL} config(s) to run sequentially"
echo "────────────────────────────────────────"

PASS=0
FAIL=0

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    stem="${STEMS[$i]}"
    log="logs/stability/${stem}-round_2.log"
    run_num=$((i + 1))

    echo ""
    echo "[${run_num}/${TOTAL}] Starting: ${cfg}"
    echo "  Log: ${log}"

    start_ts=$(date +%s)

    nohup python3 scripts/train.py \
        --config "$cfg" \
        --max-epochs "$MAX_EPOCHS" \
        --patience "$PATIENCE" \
        --run-suffix "$RUN_SUFFIX" \
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
        echo "[${run_num}/${TOTAL}] PASS: ${stem}  (${mins}m ${secs}s)"
        PASS=$(( PASS + 1 ))
    else
        echo "[${run_num}/${TOTAL}] FAIL (exit ${exit_code}): ${stem}  (${mins}m ${secs}s)"
        echo "  See ${log} for details"
        FAIL=$(( FAIL + 1 ))
    fi
done

echo ""
echo "Round 2 complete — ${PASS} PASS  ${FAIL} FAIL  (${TOTAL} total)"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
