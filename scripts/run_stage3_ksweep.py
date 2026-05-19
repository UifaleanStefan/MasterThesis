"""
Tier C k-sweep — Stage 3 Pareto frontier across retrieval budgets.

Sweeps k in {4, 8, 16, 32} on the two strongest configs (v4-tuned,
v4-canonical) per benchmark, 1 seed only (variance estimates already
land via the Tier B 3-seed sweep at k=8). Produces the cost-vs-quality
Pareto curve that thesis Section 5.5 needs.

This is a wrapper around `scripts/run_stage3_full.py --mode full`. It
loops k values, saves each k's per-cell JSONs into a per-k subdirectory
(cells_k4/, cells_k8/, cells_k16/, cells_k32/), and emits a sweep
summary at `results/stage3/k_sweep.json`.

Cost estimate (gpt-4o-mini, 100 q per cell, 4 k values, 2 configs,
6 benchmarks = 4,800 questions): ~$1.50.

Usage:
    python scripts/run_stage3_ksweep.py
    python scripts/run_stage3_ksweep.py --k-values 4 8 16 32 --n-questions 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE3_DIR = ROOT / "results" / "stage3"
CELLS_DIR = STAGE3_DIR / "cells"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k-values", nargs="+", type=int, default=[4, 8, 16, 32],
        help="Top-k retrieval budgets to sweep.",
    )
    parser.add_argument(
        "--benchmarks", nargs="*",
        default=["cuad", "qasper", "hotpotqa", "longmemeval", "financebench", "narrativeqa"],
    )
    parser.add_argument(
        "--configs", nargs="*", default=["v4-canonical", "v4-tuned"],
    )
    parser.add_argument("--n-questions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("[FAIL] OPENAI_API_KEY not set")
        return 1

    # CRITICAL: snapshot the existing cells/ to cells_tier_b/ before sweeping,
    # so the Tier B baseline (k=8) isn't lost when we run k=4, 16, 32 (which
    # all overwrite the same cells/{bench}__{cfg}__seed{seed}.json paths).
    tier_b_backup = ROOT / "results" / "stage3" / "cells_tier_b"
    cells_src = ROOT / "results" / "stage3" / "cells"
    if cells_src.exists() and any(cells_src.glob("*.json")):
        if not tier_b_backup.exists() or not any(tier_b_backup.glob("*.json")):
            tier_b_backup.mkdir(parents=True, exist_ok=True)
            n_copied = 0
            for src in cells_src.glob("*.json"):
                shutil.copy(src, tier_b_backup / src.name)
                n_copied += 1
            print(f"  [snapshot] backed up {n_copied} Tier B cells to {tier_b_backup}")
        else:
            print(f"  [snapshot] cells_tier_b/ already populated — preserving existing snapshot")
    else:
        print(f"  [WARN] no cells in {cells_src} to back up — running k-sweep from clean state")

    sweep_summary: dict = {
        "config": vars(args),
        "results": {},
    }

    t_total = time.time()
    for k in args.k_values:
        print()
        print("=" * 78)
        print(f"  K-SWEEP: k={k}")
        print("=" * 78)

        # Run the orchestrator with this k.
        cmd = [
            "python", "scripts/run_stage3_full.py",
            "--mode", "full",
            "--benchmarks", *args.benchmarks,
            "--configs", *args.configs,
            "--n-questions", str(args.n_questions),
            "--k", str(k),
            "--seed", str(args.seed),
            "--model", args.model,
        ]
        print(f"  cmd: {' '.join(cmd)}")
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        elapsed = time.time() - t0
        if proc.returncode != 0:
            print(f"  [FAIL] orchestrator exited {proc.returncode}")
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            sweep_summary["results"][k] = {"status": "error", "elapsed": elapsed}
            continue

        # Save the k-specific outputs to a per-k subdirectory.
        per_k_cells = STAGE3_DIR / f"cells_k{k}"
        per_k_cells.mkdir(parents=True, exist_ok=True)
        for cell_path in CELLS_DIR.glob("*.json"):
            shutil.copy(cell_path, per_k_cells / cell_path.name)
        # Move the summary to a per-k file.
        summary_src = STAGE3_DIR / "stage3_runs.json"
        if summary_src.exists():
            shutil.move(str(summary_src), str(STAGE3_DIR / f"k_sweep_k{k}_runs.json"))

        # Load the per-k summary for the sweep aggregate.
        per_k_summary_path = STAGE3_DIR / f"k_sweep_k{k}_runs.json"
        per_k = json.loads(per_k_summary_path.read_text())
        sweep_summary["results"][k] = {
            "status": "ok",
            "elapsed": elapsed,
            "total_prompt_tokens": per_k.get("total_prompt_tokens"),
            "total_actual_cost_usd": per_k.get("total_actual_cost_usd"),
            "total_projected_cost_usd": per_k.get("total_projected_cost_usd"),
            "cells": per_k.get("cells", []),
        }
        print(f"  k={k} done in {elapsed:.1f}s — "
              f"cost ${per_k.get('total_actual_cost_usd', 0.0):.4f}")

    elapsed_total = time.time() - t_total
    sweep_summary["elapsed_total_seconds"] = elapsed_total
    sweep_summary["total_cost_usd"] = sum(
        r.get("total_actual_cost_usd", 0.0) or 0.0
        for r in sweep_summary["results"].values() if r.get("status") == "ok"
    )

    out_path = STAGE3_DIR / "k_sweep.json"
    out_path.write_text(json.dumps(sweep_summary, indent=2, default=str))

    # Pretty-print sweep summary.
    print()
    print("=" * 100)
    print(f"  K-SWEEP COMPLETE — {elapsed_total:.1f}s total, ${sweep_summary['total_cost_usd']:.4f} total")
    print("=" * 100)
    print()
    print(f"  {'k':>4} | {'benchmark':<14} | {'config':<14} | {'recall':>8} | {'judge':>8} | {'cost/q':>10}")
    for k, r in sweep_summary["results"].items():
        if r.get("status") != "ok":
            continue
        for cell in r.get("cells", []):
            cost_per_q = (cell.get("actual_cost_usd") or 0.0) / max(cell.get("n_questions_total", 1), 1)
            print(
                f"  {k:>4} | {cell['benchmark']:<14} | {cell['config']:<14} "
                f"| {cell.get('mean_recall_at_k', 0):>8.3f} | {cell.get('mean_judge_score', 0):>8.3f} "
                f"| ${cost_per_q:>9.5f}"
            )

    print(f"\n  Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
