"""
CMA-ES optimization of GraphMemoryV4 (10D theta) on MultiHopKeyDoor.

This is the core experiment for the thesis: does the full 10D parameterization
push GraphMemory from its current #8 benchmark ranking (precision=0.578) up to
precision=1.000, matching the top-tier systems?

Theta vector (10D, all in [0,1] during optimization, retrieval weights scaled to [0,4]):
    v[0]  = theta_store    — importance threshold
    v[1]  = theta_novel    — novelty feature weight
    v[2]  = theta_erich    — entity richness feature weight
    v[3]  = theta_surprise — context surprise feature weight
    v[4]  = theta_entity   — entity node importance threshold
    v[5]  = theta_temporal — temporal edge probability
    v[6]  = theta_decay    — entity importance decay rate
    v[7]  = w_graph  / 4   — graph signal weight (scaled to [0,4])
    v[8]  = w_embed  / 4   — embedding similarity weight (scaled to [0,4])
    v[9]  = w_recency/ 4   — recency score weight (scaled to [0,4])

Baseline comparison:
    V1 GraphMemory (3D theta) — current benchmark #8, precision=0.578

Usage:
    python run_graphmemory_v4_cmaes.py
    python run_graphmemory_v4_cmaes.py --generations 30 --episodes 60
    python run_graphmemory_v4_cmaes.py --quick   (5 gens x 20 eps for smoke test)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Parameter scaling helpers
# ---------------------------------------------------------------------------

def vec_to_params_v4(v: np.ndarray):
    """Convert a [0,1]^10 optimization vector to MemoryParamsV4."""
    from memory.graph_memory_v4 import MemoryParamsV4
    v = np.clip(v, 0.0, 1.0)
    return MemoryParamsV4(
        theta_store=float(v[0]),
        theta_novel=float(v[1]),
        theta_erich=float(v[2]),
        theta_surprise=float(v[3]),
        theta_entity=float(v[4]),
        theta_temporal=float(v[5]),
        theta_decay=float(v[6]),
        w_graph=float(v[7]) * 4.0,
        w_embed=float(v[8]) * 4.0,
        w_recency=float(v[9]) * 4.0,
        mode="learnable",
    )


def params_v4_to_vec(p) -> np.ndarray:
    """Convert MemoryParamsV4 back to a [0,1]^10 vector."""
    return np.array([
        p.theta_store,
        p.theta_novel,
        p.theta_erich,
        p.theta_surprise,
        p.theta_entity,
        p.theta_temporal,
        p.theta_decay,
        p.w_graph / 4.0,
        p.w_embed / 4.0,
        p.w_recency / 4.0,
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

def make_eval_fn(env, policy, n_episodes: int, k: int = 8):
    """
    Returns a function that evaluates a 10D theta vector.
    Fitness = mean reward over n_episodes.
    """
    from memory.graph_memory_v4 import GraphMemoryV4
    from agent.loop import run_episode_with_any_memory

    def eval_fn(v: np.ndarray) -> float:
        params = vec_to_params_v4(v)
        rewards = []
        for ep in range(n_episodes):
            mem = GraphMemoryV4(params)
            _, _, stats = run_episode_with_any_memory(
                env, policy, mem, k=k, episode_seed=ep
            )
            rewards.append(stats.get("reward", 0.0))
        return float(np.mean(rewards))

    return eval_fn


# ---------------------------------------------------------------------------
# V1 baseline (3D theta, same eval protocol)
# ---------------------------------------------------------------------------

def run_v1_baseline(env, policy, n_episodes: int, k: int = 8, n_generations: int = 20) -> dict:
    """Run CMA-ES on V1 GraphMemory (3D theta) as a comparison baseline."""
    from memory.graph_memory import GraphMemory, MemoryParams
    from agent.loop import run_episode_with_any_memory
    from optimization.cma_es import run_cmaes_optimization

    print("\n--- V1 Baseline (3D theta) ---")

    def eval_fn_v1(v: np.ndarray) -> float:
        v = np.clip(v, 0.0, 1.0)
        rewards = []
        for ep in range(n_episodes):
            mem = GraphMemory(MemoryParams(float(v[0]), float(v[1]), float(v[2]), "learnable"))
            _, _, stats = run_episode_with_any_memory(
                env, policy, mem, k=k, episode_seed=ep
            )
            rewards.append(stats.get("reward", 0.0))
        return float(np.mean(rewards))

    best_v1, history_v1 = run_cmaes_optimization(
        eval_fn_v1, n_params=3, n_generations=n_generations, sigma=0.3, verbose=True
    )
    best_reward_v1 = float(history_v1[-1]["best_fitness"]) if history_v1 else 0.0
    return {"best_theta": best_v1.tolist(), "best_reward": best_reward_v1, "history": history_v1}


# ---------------------------------------------------------------------------
# Evaluation: precision + reward on held-out episodes
# ---------------------------------------------------------------------------

def evaluate_final(mem_factory, env, policy, n_episodes: int = 100, k: int = 8) -> dict:
    """
    Full evaluation: mean reward, retrieval precision, memory size, efficiency.
    Uses episode seeds [1000, 1000+n_episodes) to avoid overlap with training seeds.
    """
    from agent.loop import run_episode_with_any_memory

    rewards, precisions, sizes, tokens = [], [], [], []
    for ep in range(n_episodes):
        mem = mem_factory()
        _, _, stats = run_episode_with_any_memory(
            env, policy, mem, k=k, episode_seed=1000 + ep
        )
        rewards.append(stats.get("reward", 0.0))
        sizes.append(stats.get("memory_size", 0))
        tokens.append(stats.get("retrieval_tokens", 0))
        prec = stats.get("retrieval_precision")
        if prec is not None:
            precisions.append(prec)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_precision": float(np.mean(precisions)) if precisions else -1.0,
        "mean_memory_size": float(np.mean(sizes)),
        "mean_tokens": float(np.mean(tokens)),
        "efficiency": float(np.mean(rewards)) / (1 + float(np.mean(tokens))),
        "n_episodes": n_episodes,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CMA-ES on GraphMemoryV4 (10D theta)")
    parser.add_argument("--generations", type=int, default=25,
                        help="Number of CMA-ES generations (default: 25)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per candidate during optimization (default: 50)")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Episodes for final evaluation (default: 100)")
    parser.add_argument("--k", type=int, default=8, help="Retrieval top-k (default: 8)")
    parser.add_argument("--sigma", type=float, default=0.3, help="Initial CMA-ES sigma")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip V1 baseline run")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test: 5 generations x 20 episodes")
    args = parser.parse_args()

    if args.quick:
        args.generations = 5
        args.episodes = 20
        args.eval_episodes = 30
        print("[QUICK MODE: 5 gens x 20 eps]")

    np.random.seed(args.seed)

    # Setup
    from environment.env import MultiHopKeyDoor
    from agent.policy import ExplorationPolicy
    from memory.graph_memory_v4 import GraphMemoryV4
    from optimization.cma_es import run_cmaes_optimization

    env = MultiHopKeyDoor(seed=0)
    policy = ExplorationPolicy(seed=args.seed)

    print("=" * 65)
    print("  GraphMemoryV4 CMA-ES Optimization — 10D theta")
    print("=" * 65)
    print(f"  Environment : MultiHopKeyDoor (10x10, 3 doors, 250 steps)")
    print(f"  Generations : {args.generations}")
    print(f"  Episodes/candidate: {args.episodes}")
    print(f"  Eval episodes: {args.eval_episodes}")
    print(f"  Sigma       : {args.sigma}")
    print(f"  Theta dims  : 10 (store, novel, erich, surprise, entity,")
    print(f"                    temporal, decay, w_graph, w_embed, w_recency)")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # Phase 1: CMA-ES optimization of V4 (10D)
    # -----------------------------------------------------------------------
    print("\n[Phase 1] CMA-ES on GraphMemoryV4 (10D theta)")
    eval_fn = make_eval_fn(env, policy, n_episodes=args.episodes, k=args.k)

    t0 = time.time()
    best_v4, history_v4 = run_cmaes_optimization(
        eval_fn,
        n_params=10,
        n_generations=args.generations,
        sigma=args.sigma,
        seed=args.seed,
        clip_to_unit=True,   # all 10 dims kept in [0,1]; retrieval weights scaled at eval time
        verbose=True,
    )
    elapsed_opt = time.time() - t0
    best_params_v4 = vec_to_params_v4(best_v4)

    print(f"\n  Optimization complete in {elapsed_opt:.1f}s")
    print(f"  Best theta (normalized): {[round(x, 3) for x in best_v4.tolist()]}")
    print(f"  Decoded params:")
    print(f"    theta_store={best_params_v4.theta_store:.3f}  theta_novel={best_params_v4.theta_novel:.3f}")
    print(f"    theta_erich={best_params_v4.theta_erich:.3f}  theta_surprise={best_params_v4.theta_surprise:.3f}")
    print(f"    theta_entity={best_params_v4.theta_entity:.3f}  theta_temporal={best_params_v4.theta_temporal:.3f}")
    print(f"    theta_decay={best_params_v4.theta_decay:.3f}")
    print(f"    w_graph={best_params_v4.w_graph:.3f}  w_embed={best_params_v4.w_embed:.3f}  w_recency={best_params_v4.w_recency:.3f}")

    # -----------------------------------------------------------------------
    # Phase 2 (optional): V1 baseline
    # -----------------------------------------------------------------------
    v1_results = None
    if not args.no_baseline:
        v1_results = run_v1_baseline(
            env, policy,
            n_episodes=args.episodes,
            k=args.k,
            n_generations=args.generations,
        )
        print(f"\n  V1 best theta: {[round(x, 3) for x in v1_results['best_theta']]}")
        print(f"  V1 best reward (training): {v1_results['best_reward']:.4f}")

    # -----------------------------------------------------------------------
    # Phase 3: Final evaluation on held-out episodes
    # -----------------------------------------------------------------------
    print(f"\n[Phase 3] Final evaluation ({args.eval_episodes} held-out episodes)")

    print("  Evaluating V4 (10D)...")
    v4_eval = evaluate_final(
        lambda: GraphMemoryV4(best_params_v4),
        env, policy,
        n_episodes=args.eval_episodes,
        k=args.k,
    )

    v1_eval = None
    if v1_results is not None:
        from memory.graph_memory import GraphMemory, MemoryParams
        best_v1 = v1_results["best_theta"]
        print("  Evaluating V1 (3D)...")
        v1_eval = evaluate_final(
            lambda: GraphMemory(MemoryParams(best_v1[0], best_v1[1], best_v1[2], "learnable")),
            env, policy,
            n_episodes=args.eval_episodes,
            k=args.k,
        )

    # -----------------------------------------------------------------------
    # Results report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)

    print(f"\n  GraphMemoryV4 (10D theta) — held-out evaluation:")
    print(f"    mean_reward      : {v4_eval['mean_reward']:.4f}  (±{v4_eval['std_reward']:.4f})")
    print(f"    retrieval_prec   : {v4_eval['mean_precision']:.4f}")
    print(f"    mean_memory_size : {v4_eval['mean_memory_size']:.1f}")
    print(f"    mean_tokens      : {v4_eval['mean_tokens']:.1f}")
    print(f"    efficiency       : {v4_eval['efficiency']:.4f}")

    if v1_eval is not None:
        print(f"\n  GraphMemoryV1 (3D theta) — held-out evaluation:")
        print(f"    mean_reward      : {v1_eval['mean_reward']:.4f}  (±{v1_eval['std_reward']:.4f})")
        print(f"    retrieval_prec   : {v1_eval['mean_precision']:.4f}")
        print(f"    mean_memory_size : {v1_eval['mean_memory_size']:.1f}")
        print(f"    mean_tokens      : {v1_eval['mean_tokens']:.1f}")
        print(f"    efficiency       : {v1_eval['efficiency']:.4f}")

        delta_reward = v4_eval["mean_reward"] - v1_eval["mean_reward"]
        delta_prec = v4_eval["mean_precision"] - v1_eval["mean_precision"]
        print(f"\n  V4 vs V1 delta:")
        print(f"    d_reward    : {delta_reward:+.4f}")
        print(f"    d_precision : {delta_prec:+.4f}")

    # Benchmark context
    print("\n  Benchmark context (from benchmark_results.json):")
    print("    EpisodicSemantic : reward=0.173  precision=1.000  [#1]")
    print("    WorkingMemory    : reward=0.153  precision=1.000  [#2]")
    print("    AttentionMemory  : reward=0.153  precision=1.000  [#2]")
    print("    SemanticMemory   : reward=0.133  precision=1.000  [#4]")
    print("    GraphMemoryV1    : reward=0.033  precision=0.578  [#8]  << was here")
    print(f"    GraphMemoryV4    : reward={v4_eval['mean_reward']:.3f}  "
          f"precision={v4_eval['mean_precision']:.3f}  << new")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results = {
        "experiment": "graphmemory_v4_cmaes",
        "config": {
            "n_generations": args.generations,
            "n_episodes_per_candidate": args.episodes,
            "n_eval_episodes": args.eval_episodes,
            "k": args.k,
            "sigma": args.sigma,
            "seed": args.seed,
        },
        "v4": {
            "best_theta_normalized": best_v4.tolist(),
            "best_params": {
                "theta_store": best_params_v4.theta_store,
                "theta_novel": best_params_v4.theta_novel,
                "theta_erich": best_params_v4.theta_erich,
                "theta_surprise": best_params_v4.theta_surprise,
                "theta_entity": best_params_v4.theta_entity,
                "theta_temporal": best_params_v4.theta_temporal,
                "theta_decay": best_params_v4.theta_decay,
                "w_graph": best_params_v4.w_graph,
                "w_embed": best_params_v4.w_embed,
                "w_recency": best_params_v4.w_recency,
            },
            "opt_history": history_v4,
            "eval": v4_eval,
        },
        "v1_baseline": {
            "best_theta": v1_results["best_theta"] if v1_results else None,
            "opt_history": v1_results["history"] if v1_results else [],
            "eval": v1_eval,
        } if v1_results else None,
        "elapsed_optimization_s": elapsed_opt,
    }

    out_path = Path("results/graphmemory_v4_cmaes_results.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Results saved to {out_path}")
    print(f"  Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
