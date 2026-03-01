"""
Ablation suite — systematically measure the contribution of each theta component.

Standard ablation: zero out one component at a time, measure performance degradation.
This quantifies the individual value of theta_store, theta_entity, and theta_temporal.

Ablation configs:
  - full: learned theta (all components at learned values) — the upper bound.
  - no_store: theta_store = 1.0 (store everything, no filtering).
  - no_entity: theta_entity = 0.0 (no entity importance filtering, create all nodes).
  - no_temporal: theta_temporal = 1.0 (all temporal edges, dense chain).
  - baseline: theta = (1.0, 0.0, 1.0) — the fixed-memory baseline.
  - ablate_store: theta_store = 0.0 (store nothing — memory is useless).
  - ablate_entity: theta_entity = 1.0 (very high threshold — no entity nodes).
  - ablate_temporal: theta_temporal = 0.0 (no temporal edges — unlinked events).

For each ablation, run n_episodes and collect: reward, retrieval_precision, tokens.
Compare against the full learned theta to compute degradation.

Output: ablation_results dict with per-ablation stats.
Visualization: Fig 8 (ablation bar chart showing performance by ablation config).

Usage:
    learned_theta = (0.31, 0.19, 0.84)
    results = run_ablation_study(env, policy, learned_theta, n_episodes=50)
    print_ablation_report(results, learned_theta)
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

import numpy as np

from memory.graph_memory import GraphMemory, MemoryParams
from agent.loop import run_episode_with_any_memory


@dataclass
class AblationConfig:
    name: str
    description: str
    theta: tuple[float, float, float]


def get_ablation_configs(learned_theta: tuple[float, float, float]) -> list[AblationConfig]:
    """
    Return the standard set of ablation configurations.

    Parameters
    ----------
    learned_theta : (store, entity, temporal)
        The learned theta to use as the "full" config and to derive ablations.
    """
    s, e, t = learned_theta
    return [
        AblationConfig("full",           f"Learned theta ({s:.2f},{e:.2f},{t:.2f})",        learned_theta),
        AblationConfig("baseline",       "Fixed baseline (1.0, 0.0, 1.0)",                   (1.0, 0.0, 1.0)),
        AblationConfig("no_store_filt",  "No store filter (store=1.0)",                      (1.0, e, t)),
        AblationConfig("no_entity_filt", "No entity filter (entity=0.0)",                    (s, 0.0, t)),
        AblationConfig("no_temp_filt",   "No temporal filter (temporal=1.0)",                 (s, e, 1.0)),
        AblationConfig("zero_store",     "Zero storage (store=0.0) — memory disabled",        (0.0, e, t)),
        AblationConfig("zero_entity",    "Max entity threshold (entity=1.0) — no entity nodes", (s, 1.0, t)),
        AblationConfig("zero_temporal",  "No temporal edges (temporal=0.0)",                  (s, e, 0.0)),
    ]


def run_ablation_study(
    env: Any,
    policy: Any,
    learned_theta: tuple[float, float, float],
    n_episodes: int = 50,
    k: int = 8,
    seed_offset: int = 0,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Run all ablation configurations and collect per-ablation metrics.

    Returns
    -------
    dict mapping ablation_name -> {mean_reward, std_reward, mean_tokens,
                                   mean_precision, degradation_vs_full}
    """
    configs = get_ablation_configs(learned_theta)
    results: dict[str, dict] = {}

    for cfg in configs:
        if verbose:
            print(f"  Ablation '{cfg.name}': theta={cfg.theta}")
        rewards: list[float] = []
        tokens: list[float] = []
        precisions: list[float] = []

        for ep in range(n_episodes):
            mem = GraphMemory(MemoryParams(*cfg.theta, "learnable"))
            _, _, stats = run_episode_with_any_memory(
                env, policy, mem, k=k, episode_seed=seed_offset + ep
            )
            rewards.append(stats.get("reward", 0.0))
            tokens.append(stats.get("retrieval_tokens", 0))
            prec = stats.get("retrieval_precision")
            if prec is not None:
                precisions.append(prec)

        results[cfg.name] = {
            "theta": cfg.theta,
            "description": cfg.description,
            "mean_reward": statistics.mean(rewards),
            "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
            "mean_tokens": statistics.mean(tokens),
            "mean_precision": statistics.mean(precisions) if precisions else None,
            "rewards": rewards,
        }
        if verbose:
            print(f"    mean_reward={results[cfg.name]['mean_reward']:.4f}")

    # Compute degradation relative to "full"
    full_reward = results["full"]["mean_reward"]
    for name, res in results.items():
        if full_reward > 0:
            res["degradation"] = (full_reward - res["mean_reward"]) / full_reward
        else:
            res["degradation"] = 0.0

    return results


def print_ablation_report(results: dict[str, dict], learned_theta: tuple) -> None:
    """Print a formatted ablation report."""
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print(f"Learned theta: {learned_theta}")
    print("=" * 70)
    print(f"{'Config':<20} {'Mean Reward':>12} {'Std':>8} {'Tokens':>10} {'Degradation':>12}")
    print("-" * 70)

    full_reward = results["full"]["mean_reward"]
    sorted_results = sorted(results.items(), key=lambda x: -x[1]["mean_reward"])

    for name, res in sorted_results:
        prec_str = f"{res['mean_precision']:.3f}" if res["mean_precision"] is not None else "N/A"
        print(
            f"{name:<20} {res['mean_reward']:>12.4f} {res['std_reward']:>8.4f} "
            f"{res['mean_tokens']:>10.1f} {res['degradation']:>12.2%}"
        )
    print("=" * 70)
