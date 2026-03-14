"""Phase 6 ABMIL analysis functions.

Thin wrapper — delegates to scripts/studies/phase6_report.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    """Run the Phase 6 ABMIL report."""
    sys.path.insert(0, str(ROOT / "scripts" / "studies"))
    from phase6_report import main as _main  # noqa: F401
    _main()
