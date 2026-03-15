#!/usr/bin/env python3
"""Phase 8 interpretability orchestrator.

Runs all four interpretability steps in dependency order after the Phase 8
ablation is complete and a winning configuration has been identified.

Steps (with sentinels in reports/figures/phase8/):
  1. Study Set curation     (local, ~1 min)     → .step1_interp.done
  2. Deletion analysis      (GCP, ~15 min)      → .step2_interp.done
  3. Cross-head heatmaps    (GCP, ~10 min)      → .step3_interp.done
  4. MC Dropout uncertainty (GCP, ~20 min)      → .step4_interp.done

Step 3 reads canonical_method.txt written by step 2.

Usage examples:

    # Dry-run (no GCP): validates study set, runs step 1 locally
    python scripts/orchestrators/phase8_interpretability.py \\
        --best-run config_phase8_mlp_rpb_dropout10_seed1 \\
        --data-dir phase8-data/

    # Full run (GCP required)
    python scripts/orchestrators/phase8_interpretability.py \\
        --best-run config_phase8_mlp_rpb_dropout10_seed1 \\
        --data-dir phase8-data/ \\
        --gcp-host 136.109.153.16 --gcp-user chris --gcp-pass 'PASS'

    # Re-visualise all steps from already-downloaded results (no GCP)
    python scripts/orchestrators/phase8_interpretability.py \\
        --best-run config_phase8_mlp_rpb_dropout10_seed1 \\
        --data-dir phase8-data/ --visualise-only

    # Force re-run a single step
    python scripts/orchestrators/phase8_interpretability.py \\
        --best-run config_phase8_mlp_rpb_dropout10_seed1 \\
        --data-dir phase8-data/ --steps 3 --force \\
        --gcp-host 136.109.153.16 --gcp-user chris --gcp-pass 'PASS'
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ── Root setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

FIGURES_DIR = ROOT / "reports" / "figures" / "phase8"
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "studies"
STATUS_JSON = FIGURES_DIR / "interpretability_status.json"
PYTHON      = sys.executable


# ── Step registry ──────────────────────────────────────────────────────────────

@dataclass
class StepDef:
    name:      str
    script:    str
    gcp:       bool
    hard_deps: list = field(default_factory=list)


STEPS: dict[int, StepDef] = {
    1: StepDef("study_set",  "phase8_study_set.py",   gcp=False, hard_deps=[]),
    2: StepDef("deletion",   "phase8_deletion.py",    gcp=True,  hard_deps=[1]),
    3: StepDef("heatmaps",   "phase8_heatmaps.py",    gcp=True,  hard_deps=[1]),
    4: StepDef("mc_dropout", "phase8_mc_dropout.py",  gcp=True,  hard_deps=[1]),
}

EXECUTION_ORDER = [1, 2, 3, 4]


# ── Sentinel helpers ───────────────────────────────────────────────────────────

def sentinel_path(n: int) -> Path:
    return FIGURES_DIR / f".step{n}_interp.done"


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
    best_run: str,
    data_dir: str,
    baseline_run: str,
    canonical_metric: str,
    gcp_host: str | None,
    gcp_user: str | None,
    gcp_pass: str | None,
    visualise_only: bool,
) -> list[str]:
    script = str(SCRIPTS_DIR / step.script)
    cmd = [PYTHON, script, "--best-run", best_run, "--data-dir", data_dir]

    if n == 2:
        cmd += ["--baseline-run", baseline_run]

    if n == 3 and canonical_metric != "auto":
        cmd += ["--canonical-metric", canonical_metric]

    if step.gcp:
        if visualise_only:
            cmd.append("--visualise-only")
        elif gcp_host:
            cmd += ["--gcp-host", gcp_host, "--gcp-user", gcp_user, "--gcp-pass", gcp_pass]

    return cmd


# ── Per-step runner ────────────────────────────────────────────────────────────

def run_step(
    n: int,
    step: StepDef,
    statuses: dict[int, dict],
    best_run: str,
    data_dir: str,
    baseline_run: str,
    canonical_metric: str,
    gcp_host: str | None,
    gcp_user: str | None,
    gcp_pass: str | None,
    visualise_only: bool,
    active_steps: set[int],
) -> None:
    status_entry = statuses[n]

    # ── Sentinel skip ─────────────────────────────────────────────────────────
    if sentinel_exists(n):
        status_entry.update(status="skipped", error="sentinel exists (already done)")
        print(f"  — SKIPPED (sentinel .step{n}_interp.done exists)")
        return

    # ── Hard dependency check ─────────────────────────────────────────────────
    for dep in step.hard_deps:
        dep_status     = statuses.get(dep, {}).get("status", "not_run")
        dep_sentinel_ok = sentinel_exists(dep) and dep not in active_steps
        if dep_status not in ("success", "skipped") and not dep_sentinel_ok:
            reason = f"hard dep step {dep} did not succeed (status={dep_status})"
            status_entry.update(status="skipped", error=reason)
            print(f"  — SKIPPED ({reason})")
            return

    # ── If step 3, auto-resolve canonical metric from step 2 output ──────────
    effective_canonical = canonical_metric
    if n == 3 and canonical_metric == "auto":
        canon_path = FIGURES_DIR / "deletion" / "canonical_method.txt"
        if canon_path.exists():
            effective_canonical = canon_path.read_text().strip()
            print(f"  Canonical metric from step 2: {effective_canonical}")
        else:
            effective_canonical = "attn"
            print(f"  [INFO] canonical_method.txt not found; defaulting to 'attn'")

    # ── Build & run command ───────────────────────────────────────────────────
    cmd = build_command(
        n, step, best_run, data_dir, baseline_run, effective_canonical,
        gcp_host, gcp_user, gcp_pass, visualise_only,
    )
    print(f"  $ {' '.join(cmd)}")

    start = time.monotonic()
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), timeout=7200)
    except subprocess.TimeoutExpired:
        duration = time.monotonic() - start
        status_entry.update(status="failed", duration_s=round(duration, 1),
                            error="timeout after 7200s")
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

def write_status(statuses: dict[int, dict], args_ns) -> None:
    payload = {
        "run_timestamp":  datetime.now().isoformat(timespec="seconds"),
        "best_run":       args_ns.best_run,
        "data_dir":       args_ns.data_dir,
        "visualise_only": args_ns.visualise_only,
        "steps": {str(n): s for n, s in statuses.items()},
        "summary": {
            "total":   len(statuses),
            "success": sum(1 for s in statuses.values() if s.get("status") == "success"),
            "skipped": sum(1 for s in statuses.values() if s.get("status") == "skipped"),
            "failed":  sum(1 for s in statuses.values() if s.get("status") == "failed"),
        },
    }
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(json.dumps(payload, indent=2))


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(statuses: dict[int, dict]) -> None:
    print("\n" + "=" * 55)
    print(f"{'Step':<6} {'Name':<14} {'Status':<10} {'Duration':>10}  Error")
    print("-" * 55)
    icons = {"success": "✓", "failed": "✗", "skipped": "—", "not_run": "·"}
    for n, s in sorted(statuses.items()):
        icon = icons.get(s.get("status", "not_run"), "·")
        dur  = f"{s.get('duration_s', 0):.1f}s" if s.get("duration_s") else "—"
        err  = (s.get("error") or "")[:55]
        print(f"  {n}    {s['name']:<14} {icon} {s.get('status','not_run'):<8}  {dur:>8}  {err}")
    print("=" * 55)
    sc = {
        "success": sum(1 for s in statuses.values() if s.get("status") == "success"),
        "failed":  sum(1 for s in statuses.values() if s.get("status") == "failed"),
        "skipped": sum(1 for s in statuses.values() if s.get("status") == "skipped"),
    }
    print(f"  {sc['success']} succeeded  |  {sc['failed']} failed  |  {sc['skipped']} skipped")
    print(f"  Status log → {STATUS_JSON.relative_to(ROOT)}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 8 interpretability orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--best-run",        required=True,
                   help="Winning phase8 config run name")
    p.add_argument("--data-dir",        default="reports/data/phase8/",
                   help="Path to extracted phase8 data (default: reports/data/phase8/)")
    p.add_argument("--baseline-run",    default="config_phase8_baseline_seed0",
                   help="Baseline run for deletion comparison "
                        "(default: config_phase8_baseline_seed0)")
    p.add_argument("--canonical-metric", default="auto",
                   help="Attribution method for heatmaps: auto|attn|gradnorm|ixg "
                        "(default: auto — read from deletion step output)")
    p.add_argument("--gcp-host",        default=None)
    p.add_argument("--gcp-user",        default=None)
    p.add_argument("--gcp-pass",        default=None)
    p.add_argument("--visualise-only",  action="store_true",
                   help="Skip GCP; pass --visualise-only to all GCP steps")
    p.add_argument("--force",           action="store_true",
                   help="Clear sentinels for selected steps, forcing re-execution")
    p.add_argument("--steps",           default=None,
                   help="Comma-separated step numbers to run (default: all). E.g. --steps 3,4")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Resolve which steps to run ────────────────────────────────────────────
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

    run_order = [n for n in EXECUTION_ORDER if n in active_steps]

    # ── Force: clear sentinels ─────────────────────────────────────────────────
    if args.force:
        for n in run_order:
            if sentinel_exists(n):
                clear_sentinel(n)
                print(f"[force] cleared .step{n}_interp.done")

    # ── Initialise status entries ─────────────────────────────────────────────
    statuses: dict[int, dict] = {}
    for n, step in STEPS.items():
        statuses[n] = {
            "name":       step.name,
            "status":     "not_run",
            "duration_s": 0,
            "error":      None,
        }

    # ── Run ───────────────────────────────────────────────────────────────────
    print(f"\nPhase 8 Interpretability — running steps {run_order}")
    print(f"  best-run    : {args.best_run}")
    print(f"  data-dir    : {args.data_dir}")
    print(f"  baseline    : {args.baseline_run}")
    if args.visualise_only:
        print("  mode: visualise-only (no GCP)\n")
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
            best_run=args.best_run,
            data_dir=args.data_dir,
            baseline_run=args.baseline_run,
            canonical_metric=args.canonical_metric,
            gcp_host=args.gcp_host,
            gcp_user=args.gcp_user,
            gcp_pass=args.gcp_pass,
            visualise_only=args.visualise_only,
            active_steps=active_steps,
        )

    write_status(statuses, args)
    print_summary(statuses)

    if any(s.get("status") == "failed" for s in statuses.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
