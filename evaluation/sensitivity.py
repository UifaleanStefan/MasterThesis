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
from dataclasses import asdict
from typing import Any

import numpy as np

from memory.graph_memory import GraphMemory, MemoryParams
from agent.loop import run_episode_with_any_memory


# ---------------------------------------------------------------------------
# V4 sensitivity analysis — parameterized grid over any two MemoryParamsV4 dims
# ---------------------------------------------------------------------------

def compute_sensitivity_v4(
    env: Any,
    policy: Any,
    base_params,
    dim1: str = "theta_novel",
    dim2: str = "w_recency",
    dim1_range: tuple[float, float] = (0.0, 1.0),
    dim2_range: tuple[float, float] = (0.0, 4.0),
    resolution: int = 12,
    n_episodes_per_cell: int = 20,
    k: int = 8,
    seed_offset: int = 4000,
    verbose: bool = True,
) -> dict:
    """
    Compute 2D reward landscape over any two MemoryParamsV4 dimensions.

    All other dimensions are fixed at base_params values.
    Uses GraphMemoryV4 for evaluation.

    Parameters
    ----------
    base_params : MemoryParamsV4
        Fixed values for all dimensions not being varied.
    dim1 : str
        Name of the first dimension to vary (x-axis). Must be a MemoryParamsV4 field.
    dim2 : str
        Name of the second dimension to vary (y-axis). Must be a MemoryParamsV4 field.
    dim1_range : (min, max) for dim1
    dim2_range : (min, max) for dim2
    resolution : int
        Grid resolution per axis (resolution x resolution cells).
    n_episodes_per_cell : int
        Episodes per grid cell.
    seed_offset : int
        Episode seeds start here to avoid overlap with training/eval seeds.

    Returns
    -------
    dict with:
        dim1_name, dim2_name, dim1_values, dim2_values,
        reward_grid (resolution x resolution),
        precision_grid (resolution x resolution),
        best_params_dict, best_reward, analysis
    """
    from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4

    dim1_vals = np.linspace(dim1_range[0], dim1_range[1], resolution)
    dim2_vals = np.linspace(dim2_range[0], dim2_range[1], resolution)
    reward_grid = np.zeros((resolution, resolution))
    precision_grid = np.zeros((resolution, resolution))

    # Build base params dict for easy override
    base_dict = {
        "theta_store": base_params.theta_store,
        "theta_novel": base_params.theta_novel,
        "theta_erich": base_params.theta_erich,
        "theta_surprise": base_params.theta_surprise,
        "theta_entity": base_params.theta_entity,
        "theta_temporal": base_params.theta_temporal,
        "theta_decay": base_params.theta_decay,
        "w_graph": base_params.w_graph,
        "w_embed": base_params.w_embed,
        "w_recency": base_params.w_recency,
    }

    for i, v1 in enumerate(dim1_vals):
        for j, v2 in enumerate(dim2_vals):
            cell_dict = dict(base_dict)
            cell_dict[dim1] = float(v1)
            cell_dict[dim2] = float(v2)
            params = MemoryParamsV4(**cell_dict, mode="learnable")

            rewards, precisions = [], []
            for ep in range(n_episodes_per_cell):
                mem = GraphMemoryV4(params)
                _, _, stats = run_episode_with_any_memory(
                    env, policy, mem, k=k,
                    episode_seed=seed_offset + i * resolution * n_episodes_per_cell + j * n_episodes_per_cell + ep
                )
                rewards.append(stats.get("reward", 0.0))
                prec = stats.get("retrieval_precision")
                if prec is not None:
                    precisions.append(prec)

            reward_grid[i, j] = statistics.mean(rewards)
            precision_grid[i, j] = statistics.mean(precisions) if precisions else 0.0

        if verbose:
            best_so_far = reward_grid[:i + 1].max()
            print(f"  Sensitivity [{dim1}={v1:.3f}] done — best reward so far: {best_so_far:.4f}")

    # Find best cell
    best_idx = np.unravel_index(np.argmax(reward_grid), reward_grid.shape)
    best_v1 = float(dim1_vals[best_idx[0]])
    best_v2 = float(dim2_vals[best_idx[1]])
    best_reward = float(reward_grid[best_idx])

    best_params_dict = dict(base_dict)
    best_params_dict[dim1] = best_v1
    best_params_dict[dim2] = best_v2

    # Landscape analysis
    flat = reward_grid.flatten()
    top_threshold = np.percentile(flat, 90)
    top_values = flat[flat >= top_threshold]

    analysis = {
        "best_reward": best_reward,
        "best_dim1": best_v1,
        "best_dim2": best_v2,
        "mean_reward": float(np.mean(flat)),
        "std_reward": float(np.std(flat)),
        "reward_range": float(np.max(flat) - np.min(flat)),
        "top10pct_mean": float(np.mean(top_values)),
        "top10pct_std": float(np.std(top_values)),
        "is_sharp_peak": float(np.std(top_values)) < float(np.std(flat)) * 0.5,
    }

    return {
        "dim1_name": dim1,
        "dim2_name": dim2,
        "dim1_values": dim1_vals.tolist(),
        "dim2_values": dim2_vals.tolist(),
        "reward_grid": reward_grid.tolist(),
        "precision_grid": precision_grid.tolist(),
        "best_params_dict": best_params_dict,
        "best_reward": best_reward,
        "learned_dim1": float(getattr(base_params, dim1)),
        "learned_dim2": float(getattr(base_params, dim2)),
        "resolution": resolution,
        "n_episodes_per_cell": n_episodes_per_cell,
        "analysis": analysis,
    }


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
