"""Figure 4 — Phase 6 Bandit Landscape in theta-space."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_bandit_landscape(env_data: dict, output_dir: str | Path) -> Path:
    """
    Scatter plot of the 15 Phase 6 theta configs in (theta_store, theta_entity) space.
    Dot size proportional to theta_temporal.
    Dot color represents mean_j (reward).
    Best theta marked with a gold star.
    One subplot per environment.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    envs = [e for e in ["Key-Door", "Goal-Room", "MultiHop-KeyDoor"] if e in env_data]
    n = len(envs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    fig.suptitle("Phase 6 Bandit Landscape — Theta Search Space",
                 fontsize=13, fontweight="bold")

    for ax, env_name in zip(axes, envs):
        phase6 = env_data[env_name].get("phase6_results", [])
        best_theta = env_data[env_name].get("best_theta", None)
        if not phase6:
            ax.set_title(env_name)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        xs = [r["theta"][0] for r in phase6]   # theta_store
        ys = [r["theta"][1] for r in phase6]   # theta_entity
        sizes = [50 + 250 * r["theta"][2] for r in phase6]  # theta_temporal → size
        colors = [r["mean_j"] for r in phase6]

        vmin = min(colors)
        vmax = max(colors) if max(colors) > vmin else vmin + 1e-6

        sc = ax.scatter(xs, ys, s=sizes, c=colors, cmap="YlOrRd",
                        vmin=vmin, vmax=vmax, alpha=0.85, edgecolors="grey", linewidths=0.5)

        if best_theta is not None:
            ax.scatter([best_theta[0]], [best_theta[1]],
                       s=300, marker="*", c="gold", edgecolors="black",
                       linewidths=1.0, zorder=5, label="Best theta")
            ax.legend(fontsize=8, loc="upper right")

        plt.colorbar(sc, ax=ax, label="Mean reward")
        ax.set_xlabel("theta_store", fontsize=10)
        ax.set_ylabel("theta_entity", fontsize=10)
        ax.set_xlim(-0.05, 1.1)
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(env_name, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)

        # Size legend
        for tval in [0.2, 0.6, 1.0]:
            ax.scatter([], [], s=50 + 250 * tval, c="grey", alpha=0.6,
                       label=f"t_temp={tval:.1f}")
        ax.legend(fontsize=7, loc="lower right", title="dot size=theta_temporal")

    plt.tight_layout()
    out_path = output_dir / "fig4_bandit_landscape.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
