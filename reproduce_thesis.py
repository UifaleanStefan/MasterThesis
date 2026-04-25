"""
Master reproduction script — runs every thesis experiment in dependency order.

Modes:
    --quick  Smoke-sized runs (5 gens, 20-30 episodes per setting). ~10 min total.
             Useful for validating the pipeline on a fresh machine.
    --full   Canonical runs that produce the published numbers. ~24 h total
             with NeuralV2 200-gen training; run overnight.

The script invokes existing run_*.py scripts via subprocess so the canonical
entry points stay authoritative — no logic is duplicated here. Each step
fails fast on non-zero exit; on failure, the script reports which step
failed so the user can re-run from there.

Stage 3 (real LLM) is intentionally skipped — those experiments require an
OPENAI_API_KEY and a budget commitment; they are listed at the end so
``reproduce_thesis.py`` documents them but does not invoke them automatically.

Usage (PowerShell, from project root):
    python reproduce_thesis.py --quick
    python reproduce_thesis.py --full
    python reproduce_thesis.py --full --skip benchmark neural_v2
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------


@dataclass
class Step:
    name: str
    quick_cmd: str
    full_cmd: str
    description: str


_STEPS: list[Step] = [
    Step(
        "smoke",
        "python run_smoke_tests.py --no-runner",
        "python run_smoke_tests.py",
        "Quick pipeline check (grid envs + DocumentQA memory + LLM fallback path)",
    ),
    Step(
        "audit_determinism",
        "python scripts/audit_determinism.py",
        "python scripts/audit_determinism.py",
        "Determinism audit — every memory system should be reproducible",
    ),
    Step(
        "pytest",
        "python -m pytest tests/ -q",
        "python -m pytest tests/ -q",
        "Thesis invariant tests (memory contract, V4 properties, statistics, LLM judge)",
    ),
    Step(
        "benchmark",
        "python run_benchmark.py",
        "python run_benchmark.py",
        "12-system × 4-environment benchmark on grid worlds",
    ),
    Step(
        "v4_cmaes",
        "python run_graphmemory_v4_cmaes.py --quick --no-baseline",
        "python run_graphmemory_v4_cmaes.py --generations 30 --episodes 50 --eval-episodes 200",
        "CMA-ES on GraphMemoryV4 (10D theta) — the headline thesis experiment",
    ),
    Step(
        "ablation",
        "python run_ablation.py --episodes 30",
        "python run_ablation.py --episodes 100",
        "V4 ablation study (10 configs × N episodes)",
    ),
    Step(
        "w_graph_sweep",
        "python run_ablation.py --w-graph-sweep --episodes 30",
        "python run_ablation.py --w-graph-sweep --episodes 100",
        "S3: confirm w_graph contribution is vestigial (sweep over [0, 0.25, 0.5, 1, 2])",
    ),
    Step(
        "transfer",
        "python run_transfer.py --episodes 30",
        "python run_transfer.py --episodes 100",
        "Zero-shot transfer of V4 theta to GoalRoom / HardKeyDoor / MegaQuestRoom",
    ),
    Step(
        "sensitivity",
        "python run_sensitivity.py --episodes 10 --resolution 6",
        "python run_sensitivity.py --episodes 20 --resolution 12",
        "2D reward landscape sweep (theta_novel × w_recency)",
    ),
    Step(
        "docqa_memory",
        "python run_document_qa_memory.py",
        "python run_document_qa_memory.py",
        "DocumentQA memory recall@k (no LLM)",
    ),
    Step(
        "neural_v2",
        "python run_neural_controller_v2.py --generations 30 --sigma 0.05 --no-transfer",
        "python run_neural_controller_v2.py --generations 200 --sigma 0.3 --checkpoint-every 10",
        "NeuralMemoryControllerV2Small CMA-ES training "
        "(quick: 30 gens for pipeline check; full: 200 gens, ~15-20 hours)",
    ),
    Step(
        "figures",
        "python regen_all_figures.py",
        "python regen_all_figures.py",
        "Regenerate all thesis figures from the latest result JSONs",
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_step(step: Step, mode: str, dry_run: bool) -> tuple[bool, float]:
    cmd = step.quick_cmd if mode == "quick" else step.full_cmd
    print(f"\n{'=' * 70}\n  [{step.name}] {step.description}\n{'=' * 70}")
    print(f"  $ {cmd}")
    if dry_run:
        return True, 0.0
    t0 = time.time()
    proc = subprocess.run(shlex.split(cmd, posix=False), check=False)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"\n  [{step.name}] FAILED (exit {proc.returncode}) after {elapsed:.1f}s")
        return False, elapsed
    print(f"\n  [{step.name}] OK in {elapsed:.1f}s")
    return True, elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smoke-sized runs (~10 min total). Mutually exclusive with --full.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use canonical runs that produce published numbers (~24 h total).",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        metavar="STEP",
        help=f"Step names to skip. Available: {[s.name for s in _STEPS]}",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        metavar="STEP",
        help="Run only these step names (overrides --skip).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command sequence without executing.",
    )
    args = parser.parse_args(argv)

    if args.quick and args.full:
        print("ERROR: --quick and --full are mutually exclusive.")
        return 2
    if not args.quick and not args.full:
        print("ERROR: Specify --quick or --full.")
        return 2
    mode = "quick" if args.quick else "full"

    selected = _STEPS
    if args.only:
        selected = [s for s in _STEPS if s.name in args.only]
        unknown = set(args.only) - {s.name for s in _STEPS}
        if unknown:
            print(f"ERROR: Unknown step name(s) in --only: {unknown}")
            return 2
    elif args.skip:
        selected = [s for s in _STEPS if s.name not in args.skip]
        unknown = set(args.skip) - {s.name for s in _STEPS}
        if unknown:
            print(f"ERROR: Unknown step name(s) in --skip: {unknown}")
            return 2

    print(f"Reproducing thesis in {mode.upper()} mode "
          f"({len(selected)}/{len(_STEPS)} steps)")
    if args.dry_run:
        print("DRY RUN — commands will be printed but not executed.")

    t_total = 0.0
    failures: list[str] = []
    for step in selected:
        ok, elapsed = run_step(step, mode=mode, dry_run=args.dry_run)
        t_total += elapsed
        if not ok:
            failures.append(step.name)
            # Default policy: continue running remaining steps so the user
            # gets a full picture of what's broken. If you want fail-fast,
            # change to ``break`` here.
    print("\n" + "=" * 70)
    print(f"  Total elapsed: {t_total:.1f}s ({t_total / 60:.1f} min)")
    if failures:
        print(f"  FAILED steps: {failures}")
        print(f"  Re-run with: python reproduce_thesis.py --{mode} --only {' '.join(failures)}")
        return 1
    print("  All steps OK.")
    print("\nStage 3 (deferred — requires OPENAI_API_KEY and budget):")
    print("  python runner.py --config experiments/document_qa_v4.yaml")
    print("  python runner.py --config experiments/document_qa_neural_v2.yaml")
    print("  python runner.py --config experiments/document_qa_episodic.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
