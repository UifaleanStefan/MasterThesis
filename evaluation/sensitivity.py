"""
Sensitivity analysis — 2D reward heatmap over theta_store × theta_entity.

Fixes theta_temporal at the learned value and grids over the other two dimensions.
This produces a reward landscape showing:
  - Where is the optimum?
  - Is the landscape convex (single peak) or multi-modal?
  - How sharp is the optimum (high sensitivity = narrow optimum = fragile learning)?
  - Does the Bayesian optimizer or ES find the true optimum?

Resolution: default 10×10 = 100 evaluations per environment.
Each cell: n_episodes_per_cell episodes.

Usage:
    landscape = compute_sensitivity(env, policy, fixed_temporal=0.84, n_episodes=20)
    # Returns: {theta_store: [float], theta_entity: [float], reward_grid: [[float]]}
"""

from __future__ import annotations

import statistics
from typing import Any

import numpy as np

from memory.graph_memory import GraphMemory, MemoryParams
from agent.loop import run_episode_with_any_memory


def compute_sensitivity(
    env: Any,
    policy: Any,
    fixed_temporal: float = 0.8,
    resolution: int = 10,
    n_episodes_per_cell: int = 20,
    k: int = 8,
    seed_offset: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Compute 2D reward landscape over theta_store × theta_entity.

    Returns
    -------
    dict with:
        theta_store_values: list of float (x axis)
        theta_entity_values: list of float (y axis)
        reward_grid: 2D list [i_store][j_entity] = mean_reward
        best_theta: (store, entity, temporal) of grid optimum
        best_reward: float
    """
    stores = np.linspace(0.0, 1.0, resolution)
    entities = np.linspace(0.0, 1.0, resolution)
    grid = np.zeros((resolution, resolution))

    for i, s in enumerate(stores):
        for j, e in enumerate(entities):
            theta = (float(s), float(e), fixed_temporal)
            rewards = []
            for ep in range(n_episodes_per_cell):
                mem = GraphMemory(MemoryParams(*theta, "learnable"))
                _, _, stats = run_episode_with_any_memory(
                    env, policy, mem, k=k, episode_seed=seed_offset + i * resolution + j + ep
                )
                rewards.append(stats.get("reward", 0.0))
            grid[i, j] = statistics.mean(rewards)

        if verbose:
            print(f"  Sensitivity: store={s:.2f} done, max_reward_so_far={grid[:i+1].max():.4f}")

    # Find best
    best_idx = np.unravel_index(np.argmax(grid), grid.shape)
    best_s = float(stores[best_idx[0]])
    best_e = float(entities[best_idx[1]])
    best_reward = float(grid[best_idx])

    return {
        "theta_store_values": stores.tolist(),
        "theta_entity_values": entities.tolist(),
        "reward_grid": grid.tolist(),
        "best_theta": (best_s, best_e, fixed_temporal),
        "best_reward": best_reward,
        "fixed_temporal": fixed_temporal,
        "resolution": resolution,
        "n_episodes_per_cell": n_episodes_per_cell,
    }


def analyze_landscape(landscape: dict) -> dict:
    """
    Compute summary statistics of the reward landscape.

    Returns
    -------
    dict with:
        is_convex: bool (single clear peak)
        sharpness: float (std of top-10% rewards, low = flat optimum, high = sharp)
        best_theta: tuple
        best_reward: float
        mean_reward: float
        reward_range: float (max - min)
    """
    grid = np.array(landscape["reward_grid"])
    flat = grid.flatten()

    # Top 10% threshold
    top_threshold = np.percentile(flat, 90)
    top_values = flat[flat >= top_threshold]

    sharpness = float(np.std(flat))
    peak_concentration = float(np.std(top_values))

    return {
        "best_theta": landscape["best_theta"],
        "best_reward": landscape["best_reward"],
        "mean_reward": float(np.mean(flat)),
        "std_reward": sharpness,
        "reward_range": float(np.max(flat) - np.min(flat)),
        "peak_concentration": peak_concentration,
        "is_sharp": peak_concentration < sharpness * 0.5,
        "top_10pct_mean": float(np.mean(top_values)),
    }


def run_multi_env_sensitivity(
    envs: dict[str, Any],
    policies: dict[str, Any],
    learned_thetas: dict[str, tuple[float, float, float]],
    resolution: int = 8,
    n_episodes_per_cell: int = 15,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Run sensitivity analysis for multiple environments.
    Uses the learned theta_temporal for each environment as the fixed value.

    Returns {env_name: landscape_dict}
    """
    results = {}
    for env_name, env in envs.items():
        policy = policies[env_name]
        fixed_t = learned_thetas.get(env_name, (0.5, 0.1, 0.8))[2]
        if verbose:
            print(f"\n[Sensitivity] {env_name} (theta_temporal={fixed_t:.2f}, {resolution}x{resolution} grid)")
        landscape = compute_sensitivity(
            env, policy,
            fixed_temporal=fixed_t,
            resolution=resolution,
            n_episodes_per_cell=n_episodes_per_cell,
            verbose=verbose,
        )
        landscape["analysis"] = analyze_landscape(landscape)
        results[env_name] = landscape
        if verbose:
            print(f"  Best theta: {landscape['best_theta']}, reward: {landscape['best_reward']:.4f}")
    return results
