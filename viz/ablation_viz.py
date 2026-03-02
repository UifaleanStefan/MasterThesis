"""
Fig 8 — Ablation study bar chart.

Shows performance degradation when each theta component is zeroed out.
Bar chart: x = ablation config, y = mean_reward.
Color: green for "full learned", gray for baselines, red for ablated configs.
Error bars: +/- 1 std (over n_episodes).
Annotation: degradation % relative to full learned.

This is the core ablation visualization for Chapter 6.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_ablation_results_v4(
    ablation_results: dict[str, dict],
    output_path: str | Path = "docs/figures/fig08_ablation_v4.png",
    title: str = "V4 Ablation Study: 10D Theta Component Contributions",
) -> None:
    """
    Plot V4 ablation dual-panel bar chart (Fig 8 variant).

    Top panel: mean reward per ablation config with error bars and degradation %.
    Bottom panel: retrieval precision per ablation config.

    Parameters
    ----------
    ablation_results : dict from evaluation.ablation.run_ablation_study_v4
    output_path : path to save PNG
    title : figure title
    """
    names = list(ablation_results.keys())
    means = [ablation_results[n]["mean_reward"] for n in names]
    stds = [ablation_results[n]["std_reward"] for n in names]
    degradations = [ablation_results[n].get("degradation", 0.0) for n in names]
    precisions = [ablation_results[n].get("mean_precision") or 0.0 for n in names]

    # Color scheme
    colors = []
    for n in names:
        if n == "full":
            colors.append("#2ecc71")       # green
        elif n in ("v1_equivalent",):
            colors.append("#3498db")       # blue
        elif n in ("store_all",):
            colors.append("#e74c3c")       # red
        elif n.startswith("no_") or n == "graph_only":
            colors.append("#e67e22")       # orange
        else:
            colors.append("#95a5a6")       # gray

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    x = np.arange(len(names))

    # --- Panel 1: Reward ---
    bars = ax1.bar(x, means, color=colors, alpha=0.85, edgecolor="black", linewidth=0.7)
    ax1.errorbar(x, means, yerr=stds, fmt="none", color="black", capsize=4, linewidth=1.2)

    full_reward = ablation_results.get("full", {}).get("mean_reward", 1.0)
    for bar, deg, mean in zip(bars, degradations, means):
        if deg > 0.001:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.15 + 0.002,
                f"-{deg:.0%}",
                ha="center", va="bottom", fontsize=8, color="#c0392b", fontweight="bold"
            )

    ax1.axhline(full_reward, color="#2ecc71", linestyle="--", linewidth=1.2, alpha=0.6, label=f"Full ({full_reward:.3f})")
    ax1.set_ylabel("Mean Reward", fontsize=12)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.set_ylim(0, max(means) * 1.35)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # --- Panel 2: Precision ---
    prec_colors = ["#27ae60" if p >= 0.99 else "#f39c12" if p >= 0.7 else "#e74c3c" for p in precisions]
    ax2.bar(x, precisions, color=prec_colors, alpha=0.85, edgecolor="black", linewidth=0.7)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.set_ylabel("Retrieval Precision", fontsize=12)
    ax2.set_ylim(0, 1.15)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    for i, (prec, name) in enumerate(zip(precisions, names)):
        ax2.text(i, prec + 0.02, f"{prec:.3f}", ha="center", va="bottom", fontsize=8)

    labels = [n.replace("_", "\n") for n in names]
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_xlabel("Ablation Configuration", fontsize=12)

    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="Full learned theta"),
        mpatches.Patch(color="#3498db", label="V1 equivalent"),
        mpatches.Patch(color="#e67e22", label="Single dim ablated"),
        mpatches.Patch(color="#e74c3c", label="Store all (no filter)"),
    ]
    ax1.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color="#2ecc71", linestyle="--", label=f"Full reward ({full_reward:.3f})")
    ], fontsize=9, loc="upper right")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig 8 V4] Saved: {output_path}")


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
