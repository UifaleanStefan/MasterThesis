"""
Ablation study for GraphMemoryV4 (10D theta) on MultiHopKeyDoor.

Loads the best V4 theta from results/graphmemory_v4_cmaes_results.json,
runs 10 ablation configurations (each resetting one theta group to its
uninformative default), and measures the performance drop.

Output:
    results/ablation_results.json
    docs/figures/fig08_ablation_v4.png
    docs/ABLATION_RESULTS.md (written by run, stub filled in later)

Usage:
    python run_ablation.py
    python run_ablation.py --episodes 50   (faster, less accurate)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def load_v4_params():
    """Load best V4 params from the CMA-ES results JSON."""
    from memory.graph_memory_v4 import MemoryParamsV4
    data = json.loads(Path("results/graphmemory_v4_cmaes_results.json").read_text())
    bp = data["v4"]["best_params"]
    return MemoryParamsV4(
        theta_store=bp["theta_store"],
        theta_novel=bp["theta_novel"],
        theta_erich=bp["theta_erich"],
        theta_surprise=bp["theta_surprise"],
        theta_entity=bp["theta_entity"],
        theta_temporal=bp["theta_temporal"],
        theta_decay=bp["theta_decay"],
        w_graph=bp["w_graph"],
        w_embed=bp["w_embed"],
        w_recency=bp["w_recency"],
        mode="learnable",
    )


def main():
    parser = argparse.ArgumentParser(description="V4 Ablation Study")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodes per ablation config (default: 100)")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from environment.env import MultiHopKeyDoor
    from agent.policy import ExplorationPolicy
    from evaluation.ablation import run_ablation_study_v4, print_ablation_report_v4
    from viz.ablation_viz import plot_ablation_results_v4

    env = MultiHopKeyDoor(seed=0)
    policy = ExplorationPolicy(seed=args.seed)
    params = load_v4_params()

    print("=" * 65)
    print("  GraphMemoryV4 Ablation Study")
    print("=" * 65)
    print(f"  Environment  : MultiHopKeyDoor")
    print(f"  Episodes/cfg : {args.episodes}")
    print(f"  Configs      : 10 (full + 9 ablations)")
    print(f"  Seed offset  : 2000 (held-out, no training overlap)")
    print("=" * 65)
    print(f"\n  Learned theta:")
    print(f"    theta_store={params.theta_store:.3f}  theta_novel={params.theta_novel:.3f}")
    print(f"    theta_erich={params.theta_erich:.3f}  theta_surprise={params.theta_surprise:.3f}")
    print(f"    theta_entity={params.theta_entity:.3f}  theta_temporal={params.theta_temporal:.3f}")
    print(f"    theta_decay={params.theta_decay:.3f}")
    print(f"    w_graph={params.w_graph:.3f}  w_embed={params.w_embed:.3f}  w_recency={params.w_recency:.3f}")
    print()

    t0 = time.time()
    results = run_ablation_study_v4(
        env, policy, params,
        n_episodes=args.episodes,
        k=args.k,
        seed_offset=2000,
        verbose=True,
    )
    elapsed = time.time() - t0

    print_ablation_report_v4(results)

    # --- Figure ---
    plot_ablation_results_v4(
        results,
        output_path="docs/figures/fig08_ablation_v4.png",
        title="GraphMemoryV4 Ablation: 10D Theta Component Contributions\n(MultiHopKeyDoor, 100 held-out episodes)",
    )

    # --- Save JSON ---
    out = {
        "experiment": "graphmemory_v4_ablation",
        "config": {
            "n_episodes": args.episodes,
            "k": args.k,
            "seed_offset": 2000,
            "environment": "MultiHopKeyDoor",
        },
        "learned_params": {
            "theta_store": params.theta_store,
            "theta_novel": params.theta_novel,
            "theta_erich": params.theta_erich,
            "theta_surprise": params.theta_surprise,
            "theta_entity": params.theta_entity,
            "theta_temporal": params.theta_temporal,
            "theta_decay": params.theta_decay,
            "w_graph": params.w_graph,
            "w_embed": params.w_embed,
            "w_recency": params.w_recency,
        },
        "results": {
            name: {k: v for k, v in res.items() if k != "rewards"}
            for name, res in results.items()
        },
        "elapsed_s": elapsed,
    }
    out_path = Path("results/ablation_results.json")
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  Results saved to {out_path}")
    print(f"  Figure saved to docs/figures/fig08_ablation_v4.png")
    print(f"  Elapsed: {elapsed:.1f}s")

    # --- Print ranked summary for quick analysis ---
    print("\n  Ranked by mean_reward (degradation from full):")
    sorted_items = sorted(results.items(), key=lambda x: -x[1]["mean_reward"])
    full_r = results["full"]["mean_reward"]
    for name, res in sorted_items:
        prec = res.get("mean_precision")
        prec_str = f"{prec:.4f}" if prec is not None else "  N/A"
        deg = res.get("degradation", 0.0)
        print(f"    {name:<18}  reward={res['mean_reward']:.4f}  prec={prec_str}  "
              f"mem={res['mean_memory_size']:.1f}  degradation={deg:.1%}")


if __name__ == "__main__":
    main()
