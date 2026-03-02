"""
Zero-shot transfer test for GraphMemoryV4 (10D theta).

Takes the V4 theta learned on MultiHopKeyDoor and evaluates it without
retraining on 3 other environments. Also runs quick per-env CMA-ES
(10 gens x 20 eps) to find the in-distribution optimal theta for comparison.

This tests the task-dependence hypothesis:
  - If diagonal >> off-diagonal: optimal theta is task-specific (thesis claim)
  - If off-diagonal is competitive: the learned theta generalizes

Environments:
  - MultiHopKeyDoor  (in-distribution, training env)
  - GoalRoom         (different task: navigation to goal, no hints)
  - HardKeyDoor      (similar but simpler: fewer doors, no multi-hop)
  - MegaQuestRoom    (harder, larger: 20x20, 6 doors, 1000 steps) -- key OOD test

Output:
    results/transfer_results.json
    docs/figures/fig10_transfer_v4.png

Usage:
    python run_transfer.py                     # zero-shot only (~2 min)
    python run_transfer.py --in-distribution   # + per-env CMA-ES (~30 min extra)
    python run_transfer.py --episodes 50       # faster
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


def run_quick_cmaes_v4(env, policy, n_generations=10, n_episodes=20, k=8, seed=42):
    """Run quick CMA-ES to find in-distribution optimal V4 params for an env."""
    from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4
    from optimization.cma_es import run_cmaes_optimization
    from agent.loop import run_episode_with_any_memory

    def eval_fn(v):
        v = np.clip(v, 0.0, 1.0)
        params = MemoryParamsV4(
            theta_store=float(v[0]), theta_novel=float(v[1]),
            theta_erich=float(v[2]), theta_surprise=float(v[3]),
            theta_entity=float(v[4]), theta_temporal=float(v[5]),
            theta_decay=float(v[6]),
            w_graph=float(v[7]) * 4.0, w_embed=float(v[8]) * 4.0,
            w_recency=float(v[9]) * 4.0, mode="learnable",
        )
        rewards = []
        for ep in range(n_episodes):
            mem = GraphMemoryV4(params)
            _, _, stats = run_episode_with_any_memory(env, policy, mem, k=k, episode_seed=ep)
            rewards.append(stats.get("reward", 0.0))
        return float(np.mean(rewards))

    best_v, history = run_cmaes_optimization(
        eval_fn, n_params=10, n_generations=n_generations, sigma=0.3, seed=seed, verbose=False
    )
    best_v = np.clip(best_v, 0.0, 1.0)
    return MemoryParamsV4(
        theta_store=float(best_v[0]), theta_novel=float(best_v[1]),
        theta_erich=float(best_v[2]), theta_surprise=float(best_v[3]),
        theta_entity=float(best_v[4]), theta_temporal=float(best_v[5]),
        theta_decay=float(best_v[6]),
        w_graph=float(best_v[7]) * 4.0, w_embed=float(best_v[8]) * 4.0,
        w_recency=float(best_v[9]) * 4.0, mode="learnable",
    ), history


def main():
    parser = argparse.ArgumentParser(description="V4 Zero-Shot Transfer Test")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodes per (source, target) pair (default: 100)")
    parser.add_argument("--in-distribution", action="store_true",
                        help="Also run per-env CMA-ES for in-distribution comparison")
    parser.add_argument("--indist-gens", type=int, default=10,
                        help="CMA-ES generations for in-distribution runs (default: 10)")
    parser.add_argument("--indist-eps", type=int, default=20,
                        help="Episodes per candidate for in-distribution CMA-ES (default: 20)")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from environment.env import MultiHopKeyDoor, GoalRoom, HardKeyDoor
    from environment.mega_quest import MegaQuestRoom
    from agent.policy import ExplorationPolicy
    from evaluation.transfer import evaluate_v4_theta_on_task
    from viz.transfer_viz import plot_transfer_matrix

    # --- Setup environments ---
    envs = {
        "MultiHopKeyDoor": MultiHopKeyDoor(seed=0),
        "GoalRoom": GoalRoom(seed=0),
        "HardKeyDoor": HardKeyDoor(seed=0),
        "MegaQuestRoom": MegaQuestRoom(seed=0),
    }
    policies = {name: ExplorationPolicy(seed=args.seed) for name in envs}

    multihop_params = load_v4_params()

    print("=" * 65)
    print("  GraphMemoryV4 Zero-Shot Transfer Test")
    print("=" * 65)
    print(f"  Source theta : MultiHopKeyDoor (V4 CMA-ES best)")
    print(f"  Target envs  : {list(envs.keys())}")
    print(f"  Episodes/pair: {args.episodes}")
    print(f"  In-dist CMA-ES: {args.in_distribution}")
    print("=" * 65)

    t0 = time.time()
    matrix = {}

    # --- Phase 1: Zero-shot evaluation of MultiHop theta on all envs ---
    print("\n[Phase 1] Zero-shot evaluation (MultiHop V4 theta -> all envs)")
    matrix["MultiHop_V4_zeroshot"] = {}
    for env_name, env in envs.items():
        print(f"  Evaluating on {env_name}...")
        result = evaluate_v4_theta_on_task(
            params=multihop_params,
            env=env,
            policy=policies[env_name],
            n_episodes=args.episodes,
            k=args.k,
            seed_offset=3000,
        )
        matrix["MultiHop_V4_zeroshot"][env_name] = result
        prec = result.get("mean_precision")
        prec_str = f"{prec:.4f}" if prec is not None else "N/A"
        print(f"    reward={result['mean_reward']:.4f}  precision={prec_str}  "
              f"mem_size={result['mean_memory_size']:.1f}")

    # --- Phase 2 (optional): In-distribution CMA-ES per env ---
    indist_params = {}
    if args.in_distribution:
        print(f"\n[Phase 2] In-distribution CMA-ES ({args.indist_gens} gens x {args.indist_eps} eps per env)")
        for env_name, env in envs.items():
            print(f"  Optimizing for {env_name}...")
            best_params, history = run_quick_cmaes_v4(
                env, policies[env_name],
                n_generations=args.indist_gens,
                n_episodes=args.indist_eps,
                k=args.k,
                seed=args.seed,
            )
            indist_params[env_name] = best_params
            best_fitness = max(h["best_fitness"] for h in history) if history else 0.0
            print(f"    Best training fitness: {best_fitness:.4f}")

            # Evaluate in-distribution
            result = evaluate_v4_theta_on_task(
                params=best_params,
                env=env,
                policy=policies[env_name],
                n_episodes=args.episodes,
                k=args.k,
                seed_offset=3500,
            )
            matrix[f"{env_name}_V4_indist"] = {env_name: result}
            prec = result.get("mean_precision")
            prec_str = f"{prec:.4f}" if prec is not None else "N/A"
            print(f"    In-dist reward={result['mean_reward']:.4f}  precision={prec_str}")

    elapsed = time.time() - t0

    # --- Results report ---
    print("\n" + "=" * 65)
    print("  ZERO-SHOT TRANSFER RESULTS")
    print("=" * 65)
    print(f"\n  MultiHop V4 theta evaluated zero-shot on all environments:")
    zs = matrix["MultiHop_V4_zeroshot"]
    for env_name, res in zs.items():
        prec = res.get("mean_precision")
        prec_str = f"{prec:.4f}" if prec is not None else "  N/A"
        in_dist = " (in-distribution)" if env_name == "MultiHopKeyDoor" else ""
        print(f"    {env_name:<20}  reward={res['mean_reward']:.4f}  "
              f"precision={prec_str}  mem={res['mean_memory_size']:.1f}{in_dist}")

    if args.in_distribution:
        print(f"\n  In-distribution comparison (theta optimized per env):")
        for env_name in envs:
            key = f"{env_name}_V4_indist"
            if key in matrix:
                res = matrix[key][env_name]
                zs_res = zs[env_name]
                delta = res["mean_reward"] - zs_res["mean_reward"]
                prec = res.get("mean_precision")
                prec_str = f"{prec:.4f}" if prec is not None else "  N/A"
                print(f"    {env_name:<20}  indist={res['mean_reward']:.4f}  "
                      f"zeroshot={zs_res['mean_reward']:.4f}  delta={delta:+.4f}  prec={prec_str}")

    # --- Figure ---
    # Build a flat matrix for the transfer viz
    viz_matrix = {}
    for source_key, target_dict in matrix.items():
        viz_matrix[source_key] = target_dict

    try:
        plot_transfer_matrix(
            viz_matrix,
            metric="mean_reward",
            output_path="docs/figures/fig10_transfer_v4.png",
            title="V4 Zero-Shot Transfer Matrix\n(MultiHop theta -> all environments)",
        )
    except Exception as e:
        print(f"  [Warning] Figure generation failed: {e}")

    # --- Save JSON ---
    out = {
        "experiment": "graphmemory_v4_transfer",
        "config": {
            "n_episodes": args.episodes,
            "k": args.k,
            "seed": args.seed,
            "in_distribution_cmaes": args.in_distribution,
        },
        "multihop_v4_params": {
            "theta_store": multihop_params.theta_store,
            "theta_novel": multihop_params.theta_novel,
            "theta_erich": multihop_params.theta_erich,
            "theta_surprise": multihop_params.theta_surprise,
            "theta_entity": multihop_params.theta_entity,
            "theta_temporal": multihop_params.theta_temporal,
            "theta_decay": multihop_params.theta_decay,
            "w_graph": multihop_params.w_graph,
            "w_embed": multihop_params.w_embed,
            "w_recency": multihop_params.w_recency,
        },
        "matrix": {
            source: {
                target: {k: v for k, v in res.items() if k != "rewards"}
                for target, res in targets.items()
            }
            for source, targets in matrix.items()
        },
        "elapsed_s": elapsed,
    }
    out_path = Path("results/transfer_results.json")
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  Results saved to {out_path}")
    print(f"  Figure saved to docs/figures/fig10_transfer_v4.png")
    print(f"  Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
