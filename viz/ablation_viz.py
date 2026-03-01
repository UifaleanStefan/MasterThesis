"""
Fig 8 — Ablation study bar chart.

Shows performance degradation when each theta component is zeroed out.
Bar chart: x = ablation config, y = mean_reward.
Color: green for "full learned", gray for baselines, red for ablated configs.
Error bars: ± 1 std (over n_episodes).
Annotation: degradation % relative to full learned.

This is the core ablation visualization for Chapter 6.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_ablation_results(
    ablation_results: dict[str, dict],
    learned_theta: tuple[float, float, float],
    output_path: str | Path = "docs/figures/fig08_ablation.png",
    title: str = "Ablation Study: theta Component Contributions",
) -> None:
    """
    Plot ablation bar chart (Fig 8).

    Parameters
    ----------
    ablation_results : dict from evaluation.ablation.run_ablation_study
    learned_theta : tuple
    output_path : path to save PNG
    """
    names = list(ablation_results.keys())
    means = [ablation_results[n]["mean_reward"] for n in names]
    stds = [ablation_results[n]["std_reward"] for n in names]
    degradations = [ablation_results[n].get("degradation", 0.0) for n in names]

    # Colors: full = green, baseline = steelblue, zero_* = red, others = orange
    colors = []
    for n in names:
        if n == "full":
            colors.append("#2ecc71")
        elif n == "baseline":
            colors.append("#3498db")
        elif n.startswith("zero"):
            colors.append("#e74c3c")
        else:
            colors.append("#e67e22")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    bars = ax.bar(x, means, color=colors, alpha=0.85, edgecolor="black", linewidth=0.7)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black", capsize=4, linewidth=1.2)

    # Annotate degradation
    full_reward = ablation_results.get("full", {}).get("mean_reward", 1.0)
    for bar, deg, mean in zip(bars, degradations, means):
        if deg > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.15,
                f"-{deg:.0%}",
                ha="center", va="bottom", fontsize=9, color="#e74c3c", fontweight="bold"
            )

    # Labels
    labels = [n.replace("_", " ").title() for n in names]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.set_title(
        f"{title}\nLearned θ=({learned_theta[0]:.2f}, {learned_theta[1]:.2f}, {learned_theta[2]:.2f})",
        fontsize=13
    )
    ax.set_ylim(0, max(means) * 1.3)

    # Legend
    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="Full learned theta"),
        mpatches.Patch(color="#3498db", label="Baseline (fixed theta)"),
        mpatches.Patch(color="#e67e22", label="Partial ablation"),
        mpatches.Patch(color="#e74c3c", label="Full component zeroed"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig 8] Saved: {output_path}")
