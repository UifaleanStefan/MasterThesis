"""
Fig 9 — Reward landscape heatmap (theta_store x theta_entity, or any two V4 dims).

Shows the 2D optimization surface for the main environments.
Each cell = mean reward over n_episodes at that (dim1, dim2) point.
Overlays: learned theta (star), ES trajectory (path), optimum (circle).

This visualization answers: is the landscape convex? Is the optimum sharp or broad?
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def plot_reward_landscape_v4(
    landscape: dict,
    env_name: str = "MultiHopKeyDoor",
    output_path: str | Path = "docs/figures/fig09_landscape_v4.png",
) -> None:
    """
    Plot 2D reward heatmap from compute_sensitivity_v4 output (Fig 9 V4 variant).

    Uses the dim1_name / dim2_name fields from the landscape dict for axis labels.
    Marks the learned optimum (star) and the grid optimum (cross).
    Dual panel: left = reward, right = precision.

    Parameters
    ----------
    landscape : dict from evaluation.sensitivity.compute_sensitivity_v4
    env_name : str -- used in title
    output_path : path to save PNG
    """
    grid = np.array(landscape["reward_grid"])
    dim1_vals = np.array(landscape["dim1_values"])
    dim2_vals = np.array(landscape["dim2_values"])
    dim1_name = landscape.get("dim1_name", "dim1")
    dim2_name = landscape.get("dim2_name", "dim2")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left: reward heatmap ---
    im = ax1.pcolormesh(dim2_vals, dim1_vals, grid, cmap="RdYlGn", shading="auto",
                        vmin=grid.min(), vmax=max(grid.max(), 0.01))
    plt.colorbar(im, ax=ax1, label="Mean Reward")
    try:
        ax1.contour(dim2_vals, dim1_vals, grid, levels=6, colors="black", alpha=0.25, linewidths=0.7)
    except Exception:
        pass

    learned_d1 = landscape.get("learned_dim1")
    learned_d2 = landscape.get("learned_dim2")
    if learned_d1 is not None and learned_d2 is not None:
        ax1.plot(learned_d2, learned_d1, "b*", markersize=18,
                 label=f"Learned ({dim1_name}={learned_d1:.3f}, {dim2_name}={learned_d2:.3f})",
                 markeredgecolor="white", markeredgewidth=1)

    analysis = landscape.get("analysis", {})
    best_d1 = analysis.get("best_dim1")
    best_d2 = analysis.get("best_dim2")
    if best_d1 is not None and best_d2 is not None:
        ax1.plot(best_d2, best_d1, "r+", markersize=16, markeredgewidth=2.5,
                 label=f"Grid optimum ({dim1_name}={best_d1:.3f}, {dim2_name}={best_d2:.3f})")

    ax1.set_xlabel(dim2_name, fontsize=11)
    ax1.set_ylabel(dim1_name, fontsize=11)
    ax1.set_title(f"Reward Landscape: {dim1_name} x {dim2_name}\nEnv: {env_name}", fontsize=12)
    ax1.legend(fontsize=8, loc="upper right")

    # --- Right: precision heatmap ---
    prec_grid = np.array(landscape.get("precision_grid", np.zeros_like(grid)))
    im2 = ax2.pcolormesh(dim2_vals, dim1_vals, prec_grid, cmap="Blues", shading="auto",
                         vmin=0.0, vmax=1.0)
    plt.colorbar(im2, ax=ax2, label="Retrieval Precision")
    if learned_d1 is not None and learned_d2 is not None:
        ax2.plot(learned_d2, learned_d1, "r*", markersize=18, markeredgecolor="white",
                 markeredgewidth=1, label="Learned optimum")
    ax2.set_xlabel(dim2_name, fontsize=11)
    ax2.set_ylabel(dim1_name, fontsize=11)
    ax2.set_title(f"Precision Landscape: {dim1_name} x {dim2_name}", fontsize=12)
    ax2.legend(fontsize=8, loc="upper right")

    if analysis:
        summary = (f"Best reward: {analysis.get('best_reward', 0):.4f}  "
                   f"Mean: {analysis.get('mean_reward', 0):.4f}  "
                   f"Range: {analysis.get('reward_range', 0):.4f}  "
                   f"Sharp peak: {analysis.get('is_sharp_peak', False)}")
        fig.text(0.5, 0.01, summary, ha="center", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.suptitle(f"Sensitivity Analysis -- GraphMemoryV4 on {env_name}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig 9 V4] Saved: {output_path}")


def plot_reward_landscape(
    landscape: dict,
    learned_theta: tuple[float, float, float] | None = None,
    es_trajectory: list[tuple] | None = None,
    env_name: str = "",
    output_path: str | Path = "docs/figures/fig09_landscape.png",
) -> None:
    """
    Plot 2D reward heatmap over theta_store × theta_entity (Fig 9).

    Parameters
    ----------
    landscape : dict from evaluation.sensitivity.compute_sensitivity
    learned_theta : (store, entity, temporal) — plotted as a star
    es_trajectory : list of (store, entity) tuples — ES optimization path
    env_name : str — used in title
    output_path : path to save PNG
    """
    grid = np.array(landscape["reward_grid"])
    stores = np.array(landscape["theta_store_values"])
    entities = np.array(landscape["theta_entity_values"])

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.pcolormesh(
        entities, stores, grid,
        cmap="RdYlGn", shading="auto",
        vmin=grid.min(), vmax=grid.max(),
    )
    plt.colorbar(im, ax=ax, label="Mean Reward")

    # Contour lines
    try:
        ax.contour(entities, stores, grid, levels=6, colors="black", alpha=0.3, linewidths=0.7)
    except Exception:
        pass

    # ES trajectory
    if es_trajectory:
        traj_e = [t[1] for t in es_trajectory]
        traj_s = [t[0] for t in es_trajectory]
        ax.plot(traj_e, traj_s, "w--o", markersize=5, linewidth=1.5, label="ES trajectory", alpha=0.8)
        ax.plot(traj_e[0], traj_s[0], "ws", markersize=8, label="ES start")
        ax.plot(traj_e[-1], traj_s[-1], "w^", markersize=10, label="ES end")

    # Learned theta
    if learned_theta:
        ax.plot(
            learned_theta[1], learned_theta[0],
            "b*", markersize=18, label=f"Learned θ ({learned_theta[0]:.2f},{learned_theta[1]:.2f})",
            markeredgecolor="white", markeredgewidth=1,
        )

    # Grid optimum
    best_theta = landscape.get("best_theta")
    if best_theta:
        ax.plot(
            best_theta[1], best_theta[0],
            "r+", markersize=14, markeredgewidth=2,
            label=f"Grid optimum ({best_theta[0]:.2f},{best_theta[1]:.2f})",
        )

    ax.set_xlabel("theta_entity (entity importance threshold)", fontsize=11)
    ax.set_ylabel("theta_store (storage probability)", fontsize=11)
    ax.set_title(
        f"Reward Landscape: θ_store × θ_entity\n"
        f"Environment: {env_name}, θ_temporal fixed={landscape.get('fixed_temporal', 0.8):.2f}",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig 9] Saved: {output_path}")


def plot_landscape_comparison(
    landscapes: dict[str, dict],
    output_path: str | Path = "docs/figures/fig09_landscape_comparison.png",
) -> None:
    """Plot multiple environment landscapes side-by-side."""
    n = len(landscapes)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (env_name, landscape) in zip(axes, landscapes.items()):
        grid = np.array(landscape["reward_grid"])
        stores = landscape["theta_store_values"]
        entities = landscape["theta_entity_values"]
        im = ax.pcolormesh(entities, stores, grid, cmap="RdYlGn", shading="auto")
        plt.colorbar(im, ax=ax)
        best = landscape.get("best_theta")
        if best:
            ax.plot(best[1], best[0], "b*", markersize=14, markeredgecolor="white")
        ax.set_title(f"{env_name}\nBest: {best}", fontsize=10)
        ax.set_xlabel("theta_entity")
        ax.set_ylabel("theta_store")

    plt.suptitle("Reward Landscapes: All Environments", fontsize=13)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig 9 multi] Saved: {output_path}")
