#!/usr/bin/env bash
# Stability study S2 — cosine+warmup, no class weighting, accum sweep (1/16/32).
#
# Usage:
#   nohup ./scripts/run_stability_s2.sh > logs/stability/s2_orchestrator.log 2>&1 &
#   tail -f logs/stability/s2_orchestrator.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
mkdir -p logs/stability

CONFIGS=(
    configs/studies/config_surgen_s2_accum1.yaml
    configs/studies/config_surgen_s2_accum16.yaml
    configs/studies/config_surgen_s2_accum32.yaml
)

TOTAL=${#CONFIGS[@]}
echo "Stability study S2: ${TOTAL} configs — accum sweep with cosine+warmup"
echo "────────────────────────────────────────"

PASS=0
FAIL=0

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    stem="$(basename "$cfg" .yaml)"
    log="logs/stability/${stem}.log"
    run_num=$((i + 1))

    echo ""
    echo "[${run_num}/${TOTAL}] Starting: ${cfg}"
    echo "  Log: ${log}"

    start_ts=$(date +%s)

    nohup python3 scripts/train.py --config "$cfg" \
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
        echo "[${run_num}/${TOTAL}] FAIL (exit ${exit_code}): ${stem}"
        echo "  See ${log} for details"
        FAIL=$(( FAIL + 1 ))
    fi
done

echo ""
echo "S2 complete — ${PASS} PASS  ${FAIL} FAIL  (${TOTAL} total)"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
