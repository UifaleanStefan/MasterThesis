"""
Fig 11 — Token cost vs. reward Pareto frontier scatter plot.

Each point = one memory system.
x-axis = mean retrieval tokens (proxy for LLM cost).
y-axis = mean reward.
Pareto frontier = systems that are not dominated by any other.
Label: system name, efficiency annotation.

This is the key cost-efficiency visualization for Chapter 5.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compute_pareto_frontier(
    costs: list[float],
    rewards: list[float],
) -> list[int]:
    """
    Return indices of Pareto-optimal points (maximize reward, minimize cost).
    A point is Pareto-optimal if no other point has both higher reward AND lower cost.
    """
    n = len(costs)
    is_pareto = [True] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                if costs[j] <= costs[i] and rewards[j] >= rewards[i]:
                    if costs[j] < costs[i] or rewards[j] > rewards[i]:
                        is_pareto[i] = False
                        break
    return [i for i in range(n) if is_pareto[i]]


def plot_pareto_frontier(
    systems: list[str],
    costs: list[float],
    rewards: list[float],
    ci_rewards: list[tuple[float, float]] | None = None,
    output_path: str | Path = "docs/figures/fig11_pareto.png",
    title: str = "Token Cost vs. Reward Pareto Frontier",
    cost_label: str = "Mean Retrieval Tokens (proxy for LLM cost)",
) -> None:
    """
    Plot Pareto frontier for all memory systems (Fig 11).

    Parameters
    ----------
    systems : list of system names
    costs : list of mean retrieval tokens (lower = better)
    rewards : list of mean rewards (higher = better)
    ci_rewards : optional list of (ci_lower, ci_upper) for error bars
    """
    pareto_indices = compute_pareto_frontier(costs, rewards)
    pareto_set = set(pareto_indices)

    # Colors: Pareto-optimal = gold star, others = gray circle
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {
        "FlatWindow(50)":       "#95a5a6",
        "GraphMemory+Theta":    "#2ecc71",
        "SemanticMemory":       "#3498db",
        "SummaryMemory":        "#9b59b6",
        "EpisodicSemantic":     "#e67e22",
        "RAGMemory":            "#e74c3c",
        "HierarchicalMemory":   "#1abc9c",
        "WorkingMemory(7)":     "#f39c12",
        "CausalMemory":         "#d35400",
        "AttentionMemory":      "#8e44ad",
    }

    for i, (sys, cost, reward) in enumerate(zip(systems, costs, rewards)):
        color = colors.get(sys, "#7f8c8d")
        marker = "*" if i in pareto_set else "o"
        size = 200 if i in pareto_set else 80

        if ci_rewards:
            ci_lo, ci_hi = ci_rewards[i]
            ax.errorbar(cost, reward, yerr=[[reward - ci_lo], [ci_hi - reward]],
                        fmt="none", color=color, capsize=5, linewidth=1.5, alpha=0.7)

        ax.scatter(cost, reward, c=color, marker=marker, s=size,
                   edgecolors="black", linewidths=0.8, zorder=5,
                   label=sys + (" ✓" if i in pareto_set else ""))

        # Label
        offset_x = (max(costs) - min(costs)) * 0.015
        ax.annotate(
            sys.replace("Memory", "Mem").replace("(50)", "").replace("(7)", ""),
            (cost + offset_x, reward),
            fontsize=8.5, va="center",
        )

    # Draw Pareto frontier
    if len(pareto_indices) > 1:
        pareto_sorted = sorted(pareto_indices, key=lambda i: costs[i])
        pf_costs = [costs[i] for i in pareto_sorted]
        pf_rewards = [rewards[i] for i in pareto_sorted]
        ax.step(pf_costs, pf_rewards, "k--", linewidth=1.5, alpha=0.6, label="Pareto frontier",
                where="post")

    ax.set_xlabel(cost_label, fontsize=12)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.set_title(title, fontsize=13)

    # Shade Pareto-optimal region
    if pareto_indices:
        ax.axhline(
            y=max(rewards[i] for i in pareto_indices),
            color="green", linestyle=":", alpha=0.2, linewidth=1,
        )

    handles, labels = ax.get_legend_handles_labels()
    # Deduplicate
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=8, loc="lower right",
              ncol=2, framealpha=0.8)

    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Efficiency annotation
    for i, (sys, cost, reward) in enumerate(zip(systems, costs, rewards)):
        if i in pareto_set:
            eff = reward / (1 + cost)
            ax.annotate(
                f"eff={eff:.3f}",
                (costs[i], rewards[i]),
                xytext=(0, -18),
                textcoords="offset points",
                fontsize=7.5,
                color="#27ae60",
                ha="center",
            )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig 11] Saved: {output_path}")
