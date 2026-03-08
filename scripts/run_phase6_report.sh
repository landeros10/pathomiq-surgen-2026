#!/usr/bin/env bash
# Generate the Phase 6 results report.
#
# Workflow:
#   1. SSH to GCP and run phase6_extract.py  → serialises metrics + inference
#      to a temp dir on the server
#   2. scp that temp dir to a local staging area
#   3. rm the temp dir from the server
#   4. Run phase6_report.py locally against the downloaded data
#      → writes reports/YYYY-MM-DD-phase6-results.md  +  reports/figures/
#
# GCP password is read from the SSHPASS environment variable so it never
# appears in process listings or shell history.
#
# Usage:
#   export SSHPASS="<gcp-password>"
#   ./scripts/run_phase6_report.sh
#
#   # Skip model inference (MLflow metrics only, much faster):
#   ./scripts/run_phase6_report.sh --skip-inference
#
#   # Override bootstrap sample count:
#   ./scripts/run_phase6_report.sh --n-bootstrap 500
#
#   # Override number of attention heatmap slides:
#   ./scripts/run_phase6_report.sh --n-slides 3

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
GCP_USER="chris"
GCP_HOST="136.109.153.16"
GCP_PROJECT_DIR="~/surgen"               # confirmed from mlflow.md (mlflow.db lives at ~/surgen/mlflow.db)
GCP_REMOTE_TMP="/tmp/phase6-report-data" # wiped from server after download

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOCAL_TMP="${PROJECT_ROOT}/tmp/phase6-report-data"
LOCAL_PYTHON="${PROJECT_ROOT}/surgen-env/bin/python"

# ── Parse flags ───────────────────────────────────────────────────────────────
SKIP_INFERENCE=""
N_BOOTSTRAP="1000"
N_SLIDES="5"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-inference)
            SKIP_INFERENCE="--skip-inference"
            shift
            ;;
        --n-bootstrap)
            [[ $# -gt 1 ]] || { echo "Error: --n-bootstrap requires a value"; exit 1; }
            N_BOOTSTRAP="$2"
            shift 2
            ;;
        --n-slides)
            [[ $# -gt 1 ]] || { echo "Error: --n-slides requires a value"; exit 1; }
            N_SLIDES="$2"
            shift 2
            ;;
        *)
            echo "Error: unknown option '$1'"
            echo "Usage: $0 [--skip-inference] [--n-bootstrap N] [--n-slides N]"
            exit 1
            ;;
    esac
done

# ── Validate environment ───────────────────────────────────────────────────────
if [[ -z "${SSHPASS:-}" ]]; then
    echo "Error: SSHPASS environment variable is not set."
    echo "  export SSHPASS=\"<gcp-password>\""
    exit 1
fi

if ! command -v sshpass &> /dev/null; then
    echo "Error: sshpass is not installed."
    echo "  brew install sshpass   (or equivalent)"
    exit 1
fi

SSH="sshpass -e ssh -o StrictHostKeyChecking=no ${GCP_USER}@${GCP_HOST}"
SCP="sshpass -e scp -o StrictHostKeyChecking=no -r"

# ── Step 1: Run server-side extraction ────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "  Step 1 — Server-side extraction (GCP)"
echo "════════════════════════════════════════════"

# GCP has no `python` alias — only `python3` (see docs/GCP_setup.md).
# Activate the venv first so the correct interpreter and packages are used.
REMOTE_CMD="cd ${GCP_PROJECT_DIR} \
    && source surgen-env/bin/activate \
    && python3 scripts/studies/phase6_extract.py \
        --out-dir ${GCP_REMOTE_TMP} \
        ${SKIP_INFERENCE}"

echo "  Running: ${REMOTE_CMD}"
echo "────────────────────────────────────────────"
$SSH "$REMOTE_CMD"
echo "  Extraction complete on server."

# ── Step 2: Download results ──────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "  Step 2 — Downloading data from GCP"
echo "════════════════════════════════════════════"

mkdir -p "$(dirname "${LOCAL_TMP}")"
# Remove stale local copy if it exists
rm -rf "${LOCAL_TMP}"

$SCP "${GCP_USER}@${GCP_HOST}:${GCP_REMOTE_TMP}" "$(dirname "${LOCAL_TMP}")"
echo "  Downloaded → ${LOCAL_TMP}"

# ── Step 3: Delete from server ────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "  Step 3 — Cleaning up server"
echo "════════════════════════════════════════════"

$SSH "rm -rf ${GCP_REMOTE_TMP}"
echo "  Deleted ${GCP_REMOTE_TMP} from server."

# ── Step 4: Generate report locally ───────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "  Step 4 — Generating report locally"
echo "════════════════════════════════════════════"

"${LOCAL_PYTHON}" \
    "${SCRIPT_DIR}/studies/phase6_report.py" \
    --data-dir "${LOCAL_TMP}" \
    --n-bootstrap "${N_BOOTSTRAP}" \
    --n-slides "${N_SLIDES}"

echo ""
echo "════════════════════════════════════════════"
echo "  Done."
echo "  Report:  ${PROJECT_ROOT}/reports/$(date +%Y-%m-%d)-phase6-results.md"
echo "  Figures: ${PROJECT_ROOT}/reports/figures/phase6_*.png"
echo "  Data:    ${LOCAL_TMP}  (local only, safe to delete)"
echo "════════════════════════════════════════════"
