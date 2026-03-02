"""
Sensitivity analysis for GraphMemoryV4 (10D theta) on MultiHopKeyDoor.

Grids over theta_novel x w_recency (the two dimensions the CMA-ES optimizer
pushed hardest) while fixing all other V4 dims at the learned values.

12x12 grid = 144 cells x 20 episodes = 2,880 total episodes (~14 min).

Output:
    results/sensitivity_results.json
    docs/figures/fig09_landscape_v4.png

Usage:
    python run_sensitivity.py
    python run_sensitivity.py --resolution 8 --episodes 10   (faster, ~3 min)
    python run_sensitivity.py --dim1 theta_novel --dim2 theta_surprise
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def load_v4_params():
    from memory.graph_memory_v4 import MemoryParamsV4
    data = json.loads(Path("results/graphmemory_v4_cmaes_results.json").read_text())
    bp = data["v4"]["best_params"]
    return MemoryParamsV4(
        theta_store=bp["theta_store"], theta_novel=bp["theta_novel"],
        theta_erich=bp["theta_erich"], theta_surprise=bp["theta_surprise"],
        theta_entity=bp["theta_entity"], theta_temporal=bp["theta_temporal"],
        theta_decay=bp["theta_decay"], w_graph=bp["w_graph"],
        w_embed=bp["w_embed"], w_recency=bp["w_recency"], mode="learnable",
    )


# Dimension ranges — theta dims are [0,1], retrieval weights are [0,4]
DIM_RANGES = {
    "theta_store": (0.0, 1.0),
    "theta_novel": (0.0, 1.0),
    "theta_erich": (0.0, 1.0),
    "theta_surprise": (0.0, 1.0),
    "theta_entity": (0.0, 1.0),
    "theta_temporal": (0.0, 1.0),
    "theta_decay": (0.0, 1.0),
    "w_graph": (0.0, 4.0),
    "w_embed": (0.0, 4.0),
    "w_recency": (0.0, 4.0),
}


def main():
    parser = argparse.ArgumentParser(description="V4 Sensitivity Analysis")
    parser.add_argument("--dim1", type=str, default="theta_novel",
                        help="First dimension to vary (default: theta_novel)")
    parser.add_argument("--dim2", type=str, default="w_recency",
                        help="Second dimension to vary (default: w_recency)")
    parser.add_argument("--resolution", type=int, default=12,
                        help="Grid resolution per axis (default: 12)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episodes per grid cell (default: 20)")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from environment.env import MultiHopKeyDoor
    from agent.policy import ExplorationPolicy
    from evaluation.sensitivity import compute_sensitivity_v4
    from viz.landscape_viz import plot_reward_landscape_v4

    env = MultiHopKeyDoor(seed=0)
    policy = ExplorationPolicy(seed=args.seed)
    base_params = load_v4_params()

    dim1_range = DIM_RANGES.get(args.dim1, (0.0, 1.0))
    dim2_range = DIM_RANGES.get(args.dim2, (0.0, 4.0))
    total_eps = args.resolution * args.resolution * args.episodes

    print("=" * 65)
    print("  GraphMemoryV4 Sensitivity Analysis")
    print("=" * 65)
    print(f"  Environment   : MultiHopKeyDoor")
    print(f"  Dim 1 (x-axis): {args.dim1}  range={dim1_range}")
    print(f"  Dim 2 (y-axis): {args.dim2}  range={dim2_range}")
    print(f"  Grid          : {args.resolution}x{args.resolution} = {args.resolution**2} cells")
    print(f"  Episodes/cell : {args.episodes}")
    print(f"  Total episodes: {total_eps}")
    print(f"  Est. runtime  : ~{total_eps * 0.288 / 60:.1f} min")
    print(f"  Fixed dims    : all other V4 dims at learned values")
    print("=" * 65)
    print(f"\n  Learned values: {args.dim1}={getattr(base_params, args.dim1):.3f}, "
          f"{args.dim2}={getattr(base_params, args.dim2):.3f}")
    print()

    t0 = time.time()
    landscape = compute_sensitivity_v4(
        env=env,
        policy=policy,
        base_params=base_params,
        dim1=args.dim1,
        dim2=args.dim2,
        dim1_range=dim1_range,
        dim2_range=dim2_range,
        resolution=args.resolution,
        n_episodes_per_cell=args.episodes,
        k=args.k,
        seed_offset=4000,
        verbose=True,
    )
    elapsed = time.time() - t0

    analysis = landscape["analysis"]
    print(f"\n  Sensitivity analysis complete in {elapsed:.1f}s")
    print(f"\n  Results:")
    print(f"    Best reward       : {analysis['best_reward']:.4f}")
    print(f"    Best {args.dim1:<16}: {analysis['best_dim1']:.4f}  (learned: {getattr(base_params, args.dim1):.4f})")
    print(f"    Best {args.dim2:<16}: {analysis['best_dim2']:.4f}  (learned: {getattr(base_params, args.dim2):.4f})")
    print(f"    Mean reward       : {analysis['mean_reward']:.4f}")
    print(f"    Reward std        : {analysis['std_reward']:.4f}")
    print(f"    Reward range      : {analysis['reward_range']:.4f}")
    print(f"    Top-10% mean      : {analysis['top10pct_mean']:.4f}")
    print(f"    Top-10% std       : {analysis['top10pct_std']:.4f}")
    print(f"    Sharp peak        : {analysis['is_sharp_peak']}")

    if analysis["is_sharp_peak"]:
        print(f"\n  Interpretation: SHARP PEAK — the optimum is narrow.")
        print(f"    The optimizer must find a precise value of {args.dim1}/{args.dim2}.")
        print(f"    This suggests the memory system is sensitive to these parameters.")
    else:
        print(f"\n  Interpretation: BROAD PLATEAU — the optimum is robust.")
        print(f"    Many values of {args.dim1}/{args.dim2} achieve near-optimal performance.")
        print(f"    The learned theta is not fragile.")

    # --- Figure ---
    plot_reward_landscape_v4(
        landscape,
        env_name="MultiHopKeyDoor",
        output_path="docs/figures/fig09_landscape_v4.png",
    )

    # --- Save JSON (without full reward_grid for compactness if large) ---
    out = {
        "experiment": "graphmemory_v4_sensitivity",
        "config": {
            "dim1": args.dim1,
            "dim2": args.dim2,
            "dim1_range": list(dim1_range),
            "dim2_range": list(dim2_range),
            "resolution": args.resolution,
            "n_episodes_per_cell": args.episodes,
            "environment": "MultiHopKeyDoor",
        },
        "dim1_values": landscape["dim1_values"],
        "dim2_values": landscape["dim2_values"],
        "reward_grid": landscape["reward_grid"],
        "precision_grid": landscape["precision_grid"],
        "best_params_dict": landscape["best_params_dict"],
        "best_reward": landscape["best_reward"],
        "learned_dim1": landscape["learned_dim1"],
        "learned_dim2": landscape["learned_dim2"],
        "analysis": analysis,
        "elapsed_s": elapsed,
    }
    out_path = Path("results/sensitivity_results.json")
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  Results saved to {out_path}")
    print(f"  Figure saved to docs/figures/fig09_landscape_v4.png")


if __name__ == "__main__":
    main()
