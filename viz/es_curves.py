"""Figures 2 & 3 — ES Learning Curves and Theta Trajectory."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_ENV_COLORS = {
    "Key-Door": "#2196F3",
    "Goal-Room": "#4CAF50",
    "MultiHop-KeyDoor": "#FF5722",
}

_THETA_COLORS = {
    "theta_store": "#1976D2",
    "theta_entity": "#388E3C",
    "theta_temporal": "#F57C00",
}


def plot_es_learning_curves(env_data: dict, output_dir: str | Path) -> Path:
    """
    Figure 2: one subplot per environment showing mean_j (left axis) and
    efficiency (right axis) over all ES generations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    envs = [e for e in ["Key-Door", "Goal-Room", "MultiHop-KeyDoor"] if e in env_data]
    n = len(envs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    fig.suptitle("ES Learning Curves — Mean Reward and Efficiency per Generation",
                 fontsize=13, fontweight="bold")

    for ax, env_name in zip(axes, envs):
        gens = env_data[env_name].get("phase7_generations", [])
        if not gens:
            ax.set_title(env_name)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        gen_nums = [g["generation"] for g in gens]
        rewards = [g["mean_j"] for g in gens]
        efficiencies = [g.get("efficiency", 0.0) for g in gens]

        color = _ENV_COLORS.get(env_name, "#333333")

        ax.plot(gen_nums, rewards, color=color, marker="o", linewidth=2,
                markersize=5, label="Mean reward")
        ax.fill_between(gen_nums, rewards, alpha=0.12, color=color)
        ax.set_xlabel("Generation", fontsize=10)
        ax.set_ylabel("Mean reward (J)", fontsize=10, color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.set_ylim(bottom=0)
        ax.set_xticks(gen_nums)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(gen_nums, efficiencies, color="#9C27B0", marker="s",
                 linewidth=1.5, linestyle="--", markersize=4, label="Efficiency")
        ax2.set_ylabel("Efficiency = reward/(1+tokens)", fontsize=9, color="#9C27B0")
        ax2.tick_params(axis="y", labelcolor="#9C27B0")
        ax2.set_ylim(bottom=0)

        baseline_reward = env_data[env_name].get("baseline_es", {}).get("mean_j", None)
        if baseline_reward is not None:
            ax.axhline(baseline_reward, color="grey", linestyle=":", linewidth=1.2,
                       label=f"Baseline ({baseline_reward:.3f})")

        ax.set_title(env_name, fontsize=11, fontweight="bold")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")

    plt.tight_layout()
    out_path = output_dir / "fig2_es_learning_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_theta_trajectory(env_data: dict, output_dir: str | Path) -> Path:
    """
    Figure 3: one subplot per environment showing how theta_store, theta_entity,
    theta_temporal evolve over all ES generations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    envs = [e for e in ["Key-Door", "Goal-Room", "MultiHop-KeyDoor"] if e in env_data]
    n = len(envs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    fig.suptitle("Theta Trajectory over ES Generations",
                 fontsize=13, fontweight="bold")

    for ax, env_name in zip(axes, envs):
        gens = env_data[env_name].get("phase7_generations", [])
        if not gens:
            ax.set_title(env_name)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        gen_nums = [g["generation"] for g in gens]
        stores = [g["best_theta"][0] for g in gens]
        entities = [g["best_theta"][1] for g in gens]
        temporals = [g["best_theta"][2] for g in gens]

        ax.plot(gen_nums, stores, color=_THETA_COLORS["theta_store"],
                marker="o", linewidth=2, markersize=5, label="theta_store")
        ax.plot(gen_nums, entities, color=_THETA_COLORS["theta_entity"],
                marker="s", linewidth=2, markersize=5, label="theta_entity")
        ax.plot(gen_nums, temporals, color=_THETA_COLORS["theta_temporal"],
                marker="^", linewidth=2, markersize=5, label="theta_temporal")

        ax.set_xlabel("Generation", fontsize=10)
        ax.set_ylabel("Theta value [0, 1]", fontsize=10)
        ax.set_ylim(-0.05, 1.10)
        ax.set_xticks(gen_nums)
        ax.grid(True, alpha=0.3)
        ax.set_title(env_name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    out_path = output_dir / "fig3_theta_trajectory.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
