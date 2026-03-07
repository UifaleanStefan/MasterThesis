"""
Regenerate all figures: thesis publication figures, benchmark fig5 variants, and extended (Fig 8–15).

Order:
  1. generate_thesis_figures.py --allow-missing  (so missing JSONs don't abort)
  2. regen_benchmark_figs.py if results/benchmark_results.json exists
  3. Extended figures (Fig 8–15) with real data when results/*.json exist, else synthetic

Usage (PowerShell):
  python regen_all_figures.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGS_DIR = PROJECT_ROOT / "docs" / "figures"


def main() -> int:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Regenerate all figures")
    print("=" * 60)

    # 1. Thesis publication figures (allow missing so we don't exit)
    print("\n[1/3] Thesis figures (generate_thesis_figures.py --allow-missing)")
    r = subprocess.run(
        [sys.executable, "generate_thesis_figures.py", "--allow-missing"],
        cwd=PROJECT_ROOT,
    )
    if r.returncode != 0:
        print("Thesis figures failed.")
        return r.returncode

    # 2. Benchmark fig5 variants
    print("\n[2/3] Benchmark fig5 variants")
    if (RESULTS_DIR / "benchmark_results.json").exists():
        r = subprocess.run(
            [sys.executable, "regen_benchmark_figs.py"],
            cwd=PROJECT_ROOT,
        )
        if r.returncode != 0:
            print("Benchmark figures failed.")
            return r.returncode
    else:
        print("  Skipped (results/benchmark_results.json not found; run run_benchmark.py first)")

    # 3. Extended figures (Fig 8–15) with real data when available
    print("\n[3/3] Extended figures (Fig 8–15)")
    ablation = None
    landscape = None
    transfer_matrix = None
    benchmark_results = None

    if (RESULTS_DIR / "ablation_results.json").exists():
        with open(RESULTS_DIR / "ablation_results.json") as f:
            raw = json.load(f)
        ablation = raw.get("results", raw)
    if (RESULTS_DIR / "sensitivity_results.json").exists():
        with open(RESULTS_DIR / "sensitivity_results.json") as f:
            sens = json.load(f)
        # plot_reward_landscape expects theta_store_values, theta_entity_values
        landscape = {
            "reward_grid": sens["reward_grid"],
            "theta_store_values": sens["dim1_values"],
            "theta_entity_values": sens["dim2_values"],
        }
    if (RESULTS_DIR / "transfer_results.json").exists():
        with open(RESULTS_DIR / "transfer_results.json") as f:
            trans = json.load(f)
        transfer_matrix = trans.get("matrix", trans)
    if (RESULTS_DIR / "benchmark_results.json").exists():
        with open(RESULTS_DIR / "benchmark_results.json") as f:
            benchmark_results = json.load(f)

    from viz import generate_extended_figures
    saved = generate_extended_figures(
        ablation_results=ablation,
        landscape=landscape,
        transfer_matrix=transfer_matrix,
        benchmark_results=benchmark_results,
        output_dir=FIGS_DIR,
    )
    print(f"  Extended figures saved: {len(saved)}")

    print("\n" + "=" * 60)
    print("Done. Figures in docs/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
