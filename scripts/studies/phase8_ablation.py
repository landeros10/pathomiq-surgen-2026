#!/usr/bin/env python3
"""Phase 8 ablation report — thin entry point.

All analysis logic lives in scripts/eval/performance/ablation.py.

Usage (run from project root after scp of mlflow.db):
    python scripts/studies/phase8_ablation.py
    python scripts/studies/phase8_ablation.py --mlflow-db /path/to/mlflow.db
    python scripts/studies/phase8_ablation.py --data-dir phase8-data/ --n-bootstrap 2000
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from eval.performance.ablation import main  # noqa: E402

if __name__ == "__main__":
    main()
