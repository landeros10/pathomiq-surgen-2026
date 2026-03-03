#!/usr/bin/env bash
# Round 1 unified launcher — 2×4 grid
#
# Variables:  cosine schedule (on/off) × accum steps (1/4/8/16)
# Settings:   lr=5e-5, epochs=25, patience=5, T_max=100 (cosine configs)
#
# Usage (on GCP):
#   nohup bash scripts/run_round1.sh > logs/stability/round1_orchestrator.log 2>&1 &
#   tail -f logs/stability/round1_orchestrator.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
mkdir -p logs/stability

# No cosine
CONFIGS=(
    configs/studies/config_surgen_none.yaml           # accum=1
    configs/studies/config_surgen_accum4.yaml         # accum=4
    configs/studies/config_surgen_accum.yaml          # accum=8
    configs/studies/config_surgen_accum16.yaml        # accum=16
    configs/studies/config_surgen_cosine.yaml         # cosine, accum=1
    configs/studies/config_surgen_cosine_accum4.yaml  # cosine, accum=4
    configs/studies/config_surgen_cosine_accum.yaml   # cosine, accum=8
    configs/studies/config_surgen_s2_accum16.yaml     # cosine, accum=16
)

TOTAL=${#CONFIGS[@]}
PASS=0
FAIL=0

echo "Round 1 — cosine (on/off) × accum (1/4/8/16)  lr=5e-5  epochs=25  patience=5"
echo "${TOTAL} configs to run sequentially"
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

    nohup python3 scripts/train.py \
        --config "$cfg" \
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
echo "Round 1 complete — ${PASS} PASS  ${FAIL} FAIL  (${TOTAL} total)"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
