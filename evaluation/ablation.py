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


# ---------------------------------------------------------------------------
# V4 ablation (10D theta) — added for GraphMemoryV4 experiments
# ---------------------------------------------------------------------------

@dataclass
class AblationConfigV4:
    name: str
    description: str
    params: Any  # MemoryParamsV4


def get_ablation_configs_v4(learned_params) -> list[AblationConfigV4]:
    """
    Return the standard set of V4 ablation configurations.

    Each config resets one group of theta dimensions to its "uninformative default"
    while keeping all other dimensions at the learned values.

    Parameters
    ----------
    learned_params : MemoryParamsV4
        The best params found by CMA-ES — used as the "full" upper bound.
    """
    from memory.graph_memory_v4 import MemoryParamsV4
    p = learned_params

    def _mod(**kwargs):
        """Return a copy of p with specified fields overridden."""
        return MemoryParamsV4(
            theta_store=kwargs.get("theta_store", p.theta_store),
            theta_novel=kwargs.get("theta_novel", p.theta_novel),
            theta_erich=kwargs.get("theta_erich", p.theta_erich),
            theta_surprise=kwargs.get("theta_surprise", p.theta_surprise),
            theta_entity=kwargs.get("theta_entity", p.theta_entity),
            theta_temporal=kwargs.get("theta_temporal", p.theta_temporal),
            theta_decay=kwargs.get("theta_decay", p.theta_decay),
            w_graph=kwargs.get("w_graph", p.w_graph),
            w_embed=kwargs.get("w_embed", p.w_embed),
            w_recency=kwargs.get("w_recency", p.w_recency),
            mode="learnable",
        )

    return [
        AblationConfigV4(
            "full",
            f"Learned theta (upper bound)",
            p,
        ),
        AblationConfigV4(
            "no_novelty",
            "theta_novel=0.0 — novelty feature disabled",
            _mod(theta_novel=0.0),
        ),
        AblationConfigV4(
            "no_surprise",
            "theta_surprise=0.0 — surprise feature disabled",
            _mod(theta_surprise=0.0),
        ),
        AblationConfigV4(
            "no_erich",
            "theta_erich=0.0 — entity richness feature disabled",
            _mod(theta_erich=0.0),
        ),
        AblationConfigV4(
            "no_decay",
            "theta_decay=0.0 — entity temporal decay disabled",
            _mod(theta_decay=0.0),
        ),
        AblationConfigV4(
            "no_embed",
            "w_embed=0.0 — embedding retrieval disabled",
            _mod(w_embed=0.0),
        ),
        AblationConfigV4(
            "no_recency",
            "w_recency=0.0 — recency retrieval disabled",
            _mod(w_recency=0.0),
        ),
        AblationConfigV4(
            "graph_only",
            "w_embed=0, w_recency=0, w_graph=2.0 — graph traversal only",
            _mod(w_embed=0.0, w_recency=0.0, w_graph=2.0),
        ),
        AblationConfigV4(
            "v1_equivalent",
            "V1 defaults: store=0.874, entity=0.946, temporal=0.648, all V3/V4 dims=0",
            MemoryParamsV4(
                theta_store=0.874, theta_novel=0.0, theta_erich=0.0, theta_surprise=0.0,
                theta_entity=0.946, theta_temporal=0.648, theta_decay=0.0,
                w_graph=1.5, w_embed=1.0, w_recency=0.2, mode="learnable",
            ),
        ),
        AblationConfigV4(
            "store_all",
            "theta_store=0.0, all importance weights=0 — store every event (no filter)",
            _mod(theta_store=0.0, theta_novel=0.0, theta_erich=0.0, theta_surprise=0.0),
        ),
    ]


def run_ablation_study_v4(
    env: Any,
    policy: Any,
    learned_params,
    n_episodes: int = 100,
    k: int = 8,
    seed_offset: int = 2000,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Run all V4 ablation configurations and collect per-ablation metrics.

    Uses GraphMemoryV4 with each ablation's MemoryParamsV4.
    Episode seeds start at seed_offset to avoid overlap with training seeds.

    Returns
    -------
    dict mapping ablation_name -> {mean_reward, std_reward, mean_precision,
                                   mean_memory_size, degradation_vs_full}
    """
    from memory.graph_memory_v4 import GraphMemoryV4

    configs = get_ablation_configs_v4(learned_params)
    results: dict[str, dict] = {}

    for cfg in configs:
        if verbose:
            print(f"  [{cfg.name}] {cfg.description}")
        rewards, tokens, sizes, precisions = [], [], [], []

        for ep in range(n_episodes):
            mem = GraphMemoryV4(cfg.params)
            _, _, stats = run_episode_with_any_memory(
                env, policy, mem, k=k, episode_seed=seed_offset + ep
            )
            rewards.append(stats.get("reward", 0.0))
            tokens.append(stats.get("retrieval_tokens", 0))
            sizes.append(stats.get("memory_size", 0))
            prec = stats.get("retrieval_precision")
            if prec is not None:
                precisions.append(prec)

        results[cfg.name] = {
            "description": cfg.description,
            "mean_reward": statistics.mean(rewards),
            "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
            "mean_tokens": statistics.mean(tokens),
            "mean_memory_size": statistics.mean(sizes),
            "mean_precision": statistics.mean(precisions) if precisions else None,
            "rewards": rewards,
        }
        if verbose:
            prec_str = f"{results[cfg.name]['mean_precision']:.4f}" if results[cfg.name]["mean_precision"] is not None else "N/A"
            print(f"    reward={results[cfg.name]['mean_reward']:.4f}  precision={prec_str}  mem_size={results[cfg.name]['mean_memory_size']:.1f}")

    # Compute degradation relative to "full"
    full_reward = results["full"]["mean_reward"]
    for name, res in results.items():
        if full_reward > 0:
            res["degradation"] = (full_reward - res["mean_reward"]) / full_reward
        else:
            res["degradation"] = 0.0

    return results


def print_ablation_report_v4(results: dict[str, dict]) -> None:
    """Print a formatted V4 ablation report."""
    print("\n" + "=" * 80)
    print("V4 ABLATION STUDY RESULTS")
    print("=" * 80)
    print(f"{'Config':<18} {'Reward':>10} {'Std':>8} {'Precision':>10} {'MemSize':>9} {'Degradation':>12}")
    print("-" * 80)

    sorted_results = sorted(results.items(), key=lambda x: -x[1]["mean_reward"])
    for name, res in sorted_results:
        prec_str = f"{res['mean_precision']:.4f}" if res["mean_precision"] is not None else "  N/A  "
        print(
            f"{name:<18} {res['mean_reward']:>10.4f} {res['std_reward']:>8.4f} "
            f"{prec_str:>10} {res['mean_memory_size']:>9.1f} {res['degradation']:>12.2%}"
        )
    print("=" * 80)


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
