#!/usr/bin/env bash
# Rerun the 4 cosine configs with corrected T_max=100.
#
# Changes vs original Round 1:
#   - lr_scheduler_T_max: 100  (was defaulting to n_epochs=25)
#   - early_stopping_patience: 5
#   - run suffix: cosine-tmax100  → e.g. mmr-surgen-s1-cosine-cosine-tmax100
#
# Usage (on GCP):
#   nohup bash scripts/run_cosine_rerun.sh > logs/stability/cosine_rerun_orchestrator.log 2>&1 &
#   tail -f logs/stability/cosine_rerun_orchestrator.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
mkdir -p logs/stability

CONFIGS=(
    configs/studies/config_surgen_cosine.yaml
    configs/studies/config_surgen_cosine_accum.yaml
)

RUN_SUFFIX="cosine-tmax100"
TOTAL=${#CONFIGS[@]}
PASS=0
FAIL=0

echo "Cosine rerun — suffix=${RUN_SUFFIX}  (T_max=100, patience=5)"
echo "${TOTAL} configs to run sequentially"
echo "────────────────────────────────────────"

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    stem="$(basename "$cfg" .yaml)"
    log="logs/stability/${stem}-${RUN_SUFFIX}.log"
    run_num=$((i + 1))

    echo ""
    echo "[${run_num}/${TOTAL}] Starting: ${cfg}"
    echo "  Log: ${log}"

    start_ts=$(date +%s)

    nohup python3 scripts/train.py \
        --config "$cfg" \
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
echo "Cosine rerun complete — ${PASS} PASS  ${FAIL} FAIL  (${TOTAL} total)"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
