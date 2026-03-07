"""
Generate the extended figure suite (Fig 8–15).

When results/*.json exist, loads real data and passes it to viz.generate_extended_figures.
Otherwise uses synthetic data (viz default).

Usage:
  python generate_extended_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR = Path("docs/figures")


def main() -> None:
    ablation_results = None
    landscape = None
    transfer_matrix = None
    benchmark_results = None

    if (RESULTS_DIR / "ablation_results.json").exists():
        with open(RESULTS_DIR / "ablation_results.json") as f:
            raw = json.load(f)
        ablation_results = raw.get("results", raw)
    if (RESULTS_DIR / "sensitivity_results.json").exists():
        with open(RESULTS_DIR / "sensitivity_results.json") as f:
            sens = json.load(f)
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
        ablation_results=ablation_results,
        landscape=landscape,
        transfer_matrix=transfer_matrix,
        benchmark_results=benchmark_results,
        output_dir=OUT_DIR,
    )
    print(f"\nGenerated {len(saved)} extended figures.")


if __name__ == "__main__":
    main()
