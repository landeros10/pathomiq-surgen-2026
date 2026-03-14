"""Phase 5 multitask analysis functions.

Thin wrapper — delegates to scripts/studies/phase5_multitask_report.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    """Run the Phase 5 multitask report."""
    sys.path.insert(0, str(ROOT / "scripts" / "studies"))
    from phase5_multitask_report import main as _main  # noqa: F401
    _main()
