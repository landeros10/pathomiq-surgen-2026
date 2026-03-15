"""Phase 7 orchestrator — runs all six study steps in dependency order.

Usage examples:
    # Dry-run (no GCP): validates local state, runs local steps only
    python scripts/orchestrators/phase7_run_all.py

    # Full run
    python scripts/orchestrators/phase7_run_all.py \\
        --gcp-host 136.109.153.16 --gcp-user chris --gcp-pass 'chris@7yz'

    # Re-run a single step (force clears only that step's sentinel)
    python scripts/orchestrators/phase7_run_all.py --steps 4 --force \\
        --gcp-host 136.109.153.16 --gcp-user chris --gcp-pass 'chris@7yz'

    # Visualise-only for a single step (no GCP)
    python scripts/orchestrators/phase7_run_all.py --steps 4 --visualise-only

    # Force re-run all steps
    python scripts/orchestrators/phase7_run_all.py --force \\
        --gcp-host 136.109.153.16 --gcp-user chris --gcp-pass 'chris@7yz'
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ── Import shared constants ────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from utils.eval_utils import DEFAULT_FIGURES_DIR as FIGURES_DIR  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "studies"
STATUS_JSON = FIGURES_DIR / "phase7_status.json"

PYTHON = sys.executable


# ── Step registry ──────────────────────────────────────────────────────────────

@dataclass
class StepDef:
    name: str
    script: str
    gcp: bool
    hard_deps: list = field(default_factory=list)
    soft_deps: list = field(default_factory=list)


STEPS: dict[int, StepDef] = {
    1: StepDef("heatmap",     "phase7_heatmap.py",     gcp=False, hard_deps=[],  soft_deps=[]),
    2: StepDef("attribution", "phase7_attribution.py", gcp=True,  hard_deps=[1], soft_deps=[]),
    3: StepDef("mc_dropout",  "phase7_mc_dropout.py",  gcp=True,  hard_deps=[1], soft_deps=[]),
    4: StepDef("deletion",    "phase7_deletion.py",    gcp=True,  hard_deps=[1], soft_deps=[2]),
    5: StepDef("top_patches", "phase7_top_patches.py", gcp=True,  hard_deps=[4], soft_deps=[]),
    6: StepDef("entropy",     "phase7_entropy.py",     gcp=False, hard_deps=[],  soft_deps=[]),
}

EXECUTION_ORDER = [1, 2, 3, 4, 5, 6]


# ── Sentinel helpers ───────────────────────────────────────────────────────────

def sentinel_path(n: int) -> Path:
    return FIGURES_DIR / f".step{n}.done"


def sentinel_exists(n: int) -> bool:
    return sentinel_path(n).exists()


def write_sentinel(n: int) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sentinel_path(n).touch()


def clear_sentinel(n: int) -> None:
    p = sentinel_path(n)
    if p.exists():
        p.unlink()


# ── Command builder ────────────────────────────────────────────────────────────

def build_command(
    n: int,
    step: StepDef,
    gcp_host: str | None,
    gcp_user: str | None,
    gcp_pass: str | None,
    visualise_only: bool,
) -> list[str]:
    script = str(SCRIPTS_DIR / step.script)
    cmd = [PYTHON, script]

    if not step.gcp:
        # Local steps: no GCP args ever
        return cmd

    # GCP step
    if visualise_only:
        # All GCP steps support --visualise-only; pass the flag directly
        cmd.append("--visualise-only")
        return cmd

    # Normal GCP run
    if gcp_host:
        cmd += ["--gcp-host", gcp_host, "--gcp-user", gcp_user, "--gcp-pass", gcp_pass]

    return cmd


# ── Per-step runner ────────────────────────────────────────────────────────────

def run_step(
    n: int,
    step: StepDef,
    statuses: dict[int, dict],
    gcp_host: str | None,
    gcp_user: str | None,
    gcp_pass: str | None,
    visualise_only: bool,
    active_steps: set[int],
) -> None:
    """Execute one step, updating *statuses[n]* in place."""
    status_entry = statuses[n]

    # ── 1. Sentinel skip ──────────────────────────────────────────────────────
    if sentinel_exists(n):
        status_entry.update(status="skipped", error="sentinel exists (already done)")
        print(f"  — SKIPPED (sentinel .step{n}.done exists)")
        return

    # ── 2. Hard dependency check ──────────────────────────────────────────────
    for dep in step.hard_deps:
        # A hard dep is satisfied only if it was in active_steps and succeeded,
        # OR if it was NOT in active_steps but its sentinel already existed before
        # this run started (i.e. it completed in a prior run).
        dep_status = statuses.get(dep, {}).get("status", "not_run")
        dep_sentinel_ok = sentinel_exists(dep) and dep not in active_steps
        if dep_status not in ("success", "skipped") and not dep_sentinel_ok:
            reason = f"hard dep step {dep} did not succeed (status={dep_status})"
            status_entry.update(status="skipped", error=reason)
            print(f"  — SKIPPED ({reason})")
            return

    # ── 3. Build & run command ────────────────────────────────────────────────
    cmd = build_command(n, step, gcp_host, gcp_user, gcp_pass, visualise_only)
    print(f"  $ {' '.join(cmd)}")

    start = time.monotonic()
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), timeout=7200)
    except subprocess.TimeoutExpired:
        duration = time.monotonic() - start
        status_entry.update(status="failed", duration_s=round(duration, 1), error="timeout after 7200s")
        print(f"  ✗ FAILED (timeout)")
        return
    except Exception as exc:
        duration = time.monotonic() - start
        status_entry.update(status="failed", duration_s=round(duration, 1), error=str(exc))
        print(f"  ✗ FAILED ({exc})")
        return

    duration = time.monotonic() - start
    status_entry["duration_s"] = round(duration, 1)

    if result.returncode == 0:
        write_sentinel(n)
        status_entry.update(status="success", error=None)
        print(f"  ✓ done in {duration:.1f}s")
    else:
        status_entry.update(status="failed", error=f"returncode={result.returncode}")
        print(f"  ✗ FAILED (returncode={result.returncode})")


# ── Status JSON ────────────────────────────────────────────────────────────────

def write_status(statuses: dict[int, dict], visualise_only: bool) -> None:
    counts = {"success": 0, "skipped": 0, "failed": 0, "not_run": 0}
    for s in statuses.values():
        counts[s.get("status", "not_run")] = counts.get(s.get("status", "not_run"), 0) + 1

    payload = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "visualise_only": visualise_only,
        "steps": {str(n): s for n, s in statuses.items()},
        "summary": {
            "total": len(statuses),
            "success": sum(1 for s in statuses.values() if s.get("status") == "success"),
            "skipped": sum(1 for s in statuses.values() if s.get("status") == "skipped"),
            "failed":  sum(1 for s in statuses.values() if s.get("status") == "failed"),
        },
    }
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(json.dumps(payload, indent=2))


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(statuses: dict[int, dict]) -> None:
    print("\n" + "=" * 50)
    print(f"{'Step':<6} {'Name':<14} {'Status':<10} {'Duration':>10}  Error")
    print("-" * 50)
    icons = {"success": "✓", "failed": "✗", "skipped": "—", "not_run": "·"}
    for n, s in sorted(statuses.items()):
        icon = icons.get(s.get("status", "not_run"), "·")
        dur = f"{s.get('duration_s', 0):.1f}s" if s.get("duration_s") else "—"
        err = (s.get("error") or "")[:60]
        print(f"  {n}    {s['name']:<14} {icon} {s.get('status','not_run'):<8}  {dur:>8}  {err}")
    print("=" * 50)
    summary_counts = {
        "success": sum(1 for s in statuses.values() if s.get("status") == "success"),
        "failed":  sum(1 for s in statuses.values() if s.get("status") == "failed"),
        "skipped": sum(1 for s in statuses.values() if s.get("status") == "skipped"),
    }
    print(f"  {summary_counts['success']} succeeded  |  {summary_counts['failed']} failed  |  {summary_counts['skipped']} skipped")
    print(f"  Status log → {STATUS_JSON.relative_to(ROOT)}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 7 orchestrator — run all (or a subset of) study steps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--gcp-host", default=None, help="GCP server hostname/IP")
    p.add_argument("--gcp-user", default=None, help="GCP SSH username")
    p.add_argument("--gcp-pass", default=None, help="GCP SSH password")
    p.add_argument(
        "--visualise-only",
        action="store_true",
        help="Skip GCP; pass --visualise-only to all GCP steps (2–5), regenerating plots from saved data.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Clear sentinels for the selected steps before running, forcing re-execution.",
    )
    p.add_argument(
        "--steps",
        default=None,
        help="Comma-separated step numbers to run (default: all). E.g. --steps 4,5",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Resolve which steps to run ─────────────────────────────────────────────
    if args.steps:
        try:
            requested = [int(s.strip()) for s in args.steps.split(",")]
        except ValueError:
            print(f"ERROR: --steps must be comma-separated integers, got: {args.steps}")
            sys.exit(1)
        invalid = [n for n in requested if n not in STEPS]
        if invalid:
            print(f"ERROR: unknown step numbers: {invalid}. Valid: {list(STEPS)}")
            sys.exit(1)
        active_steps: set[int] = set(requested)
    else:
        active_steps = set(EXECUTION_ORDER)

    # Keep execution order canonical
    run_order = [n for n in EXECUTION_ORDER if n in active_steps]

    # ── Force: clear sentinels for active steps only ───────────────────────────
    if args.force:
        for n in run_order:
            if sentinel_exists(n):
                clear_sentinel(n)
                print(f"[force] cleared .step{n}.done")

    # ── Initialise status entries for ALL steps (for status JSON completeness) ─
    statuses: dict[int, dict] = {}
    for n, step in STEPS.items():
        statuses[n] = {
            "name": step.name,
            "status": "not_run",
            "duration_s": 0,
            "error": None,
        }

    # ── Run ────────────────────────────────────────────────────────────────────
    print(f"\nPhase 7 — running steps {run_order}")
    if args.visualise_only:
        print("  mode: visualise-only (no GCP; regenerates plots from saved data)\n")
    elif not args.gcp_host:
        print("  mode: dry-run (no --gcp-host; GCP steps will run without remote args)\n")
    else:
        print(f"  GCP: {args.gcp_user}@{args.gcp_host}\n")

    for n in run_order:
        step = STEPS[n]
        print(f"\n[STEP {n}/{max(run_order)} — {step.name}]")
        run_step(
            n=n,
            step=step,
            statuses=statuses,
            gcp_host=args.gcp_host,
            gcp_user=args.gcp_user,
            gcp_pass=args.gcp_pass,
            visualise_only=args.visualise_only,
            active_steps=active_steps,
        )

    write_status(statuses, args.visualise_only)
    print_summary(statuses)

    # Exit non-zero if any step failed
    if any(s.get("status") == "failed" for s in statuses.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
