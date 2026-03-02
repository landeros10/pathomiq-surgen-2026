#!/usr/bin/env bash
# Run a 1-epoch preflight check on each study config.
#
# Usage:
#   # All 8 Round 1 configs (default)
#   ./scripts/preflight_study.sh
#
#   # Specific subset
#   ./scripts/preflight_study.sh configs/studies/config_surgen_cosine.yaml configs/studies/config_surgen_all.yaml
#
# Each config runs foreground with --max-epochs 1 --run-suffix preflight.
# A PASS/FAIL summary table is printed at the end.
# Exit code is 0 only if every config passed.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
mkdir -p logs/preflight

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
echo "Preflight check: ${TOTAL} config(s)  (1 epoch each)"
echo "────────────────────────────────────────"

# Parallel arrays to accumulate results
RESULT_STEMS=()
RESULT_STATUS=()
RESULT_ELAPSED=()

any_failed=0

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    stem="$(basename "$cfg" .yaml)"
    log="logs/preflight/${stem}.log"
    run_num=$((i + 1))

    echo ""
    echo "[${run_num}/${TOTAL}] ${cfg}"
    echo "  Log: ${log}"

    start_ts=$(date +%s)

    # Run foreground — preflight is fast and interactive
    python3 scripts/train.py \
        --config "$cfg" \
        --max-epochs 1 \
        --run-suffix preflight \
        > "$log" 2>&1
    exit_code=$?

    end_ts=$(date +%s)
    elapsed=$(( end_ts - start_ts ))
    mins=$(( elapsed / 60 ))
    secs=$(( elapsed % 60 ))

    RESULT_STEMS+=("$stem")
    RESULT_ELAPSED+=("${mins}m ${secs}s")

    if [ "$exit_code" -eq 0 ]; then
        RESULT_STATUS+=("PASS")
        echo "  [PASS]  (${mins}m ${secs}s)"
    else
        RESULT_STATUS+=("FAIL (exit ${exit_code})")
        echo "  [FAIL]  exit=${exit_code}  (${mins}m ${secs}s)  — see ${log}"
        any_failed=1
    fi
done

# ── Summary table ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo " Preflight Summary"
echo "════════════════════════════════════════"
printf "  %-45s  %-20s  %s\n" "Config" "Status" "Time"
printf "  %-45s  %-20s  %s\n" "──────" "──────" "────"
for i in "${!RESULT_STEMS[@]}"; do
    printf "  %-45s  %-20s  %s\n" \
        "${RESULT_STEMS[$i]}" \
        "${RESULT_STATUS[$i]}" \
        "${RESULT_ELAPSED[$i]}"
done
echo "════════════════════════════════════════"

if [ "$any_failed" -eq 0 ]; then
    echo "All ${TOTAL} config(s) PASSED — safe to launch overnight run."
    exit 0
else
    echo "One or more configs FAILED — fix before launching overnight run."
    exit 1
fi
