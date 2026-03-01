"""
Figs 12–15 — Cost and adaptation visualizations.

Fig 12: Online adaptation curves — theta_store over steps for StatisticsAdapter.
Fig 13: Memory size over episode — how stored events grow/plateau/evict.
Fig 14: LLM cost breakdown — prompt vs. completion vs. memory tokens per system.
Fig 15: Multi-session memory persistence — what's retained across sessions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------
# Fig 12: Online adaptation theta curves
# ------------------------------------------------------------------

def plot_online_adaptation(
    theta_histories: dict[str, list[tuple]],
    output_path: str | Path = "docs/figures/fig12_online_adaptation.png",
    title: str = "Online θ Adaptation During Episode",
) -> None:
    """
    Plot theta component values over adaptation steps (Fig 12).

    Parameters
    ----------
    theta_histories : {adapter_name: [(store, entity, temporal), ...]}
        From StatisticsAdapter.get_theta_history() or GradientAdapter.get_theta_history().
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    component_names = ["theta_store", "theta_entity", "theta_temporal"]
    colors = {"StatisticsAdapter": "#e74c3c", "GradientAdapter": "#3498db", "Baseline": "#95a5a6"}

    for name, history in theta_histories.items():
        color = colors.get(name, "#2ecc71")
        for ax_idx, (ax, comp) in enumerate(zip(axes, component_names)):
            vals = [h[ax_idx] for h in history]
            steps = list(range(len(vals)))
            ax.plot(steps, vals, "-o", color=color, label=name, markersize=3, linewidth=1.8)
            ax.set_xlabel("Adaptation Step", fontsize=11)
            ax.set_ylabel(comp, fontsize=11)
            ax.set_title(comp, fontsize=12)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.axhline(y=0.5, color="black", linestyle=":", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(theta_histories),
               fontsize=10, bbox_to_anchor=(0.5, 1.0))
    plt.suptitle(title, fontsize=13, y=1.03)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig 12] Saved: {output_path}")


# ------------------------------------------------------------------
# Fig 13: Memory size over episode steps
# ------------------------------------------------------------------

def plot_memory_size_over_episode(
    memory_size_curves: dict[str, list[int]],
    output_path: str | Path = "docs/figures/fig13_memory_size.png",
    title: str = "Memory Size Over Episode Steps",
) -> None:
    """
    Plot memory size (n_events) over steps for each system (Fig 13).

    Parameters
    ----------
    memory_size_curves : {system_name: [size_at_step_0, size_at_step_1, ...]}
    """
    colors = [
        "#e74c3c", "#2ecc71", "#3498db", "#9b59b6",
        "#e67e22", "#1abc9c", "#f39c12", "#d35400", "#8e44ad", "#95a5a6"
    ]
    fig, ax = plt.subplots(figsize=(11, 6))

    for (name, sizes), color in zip(memory_size_curves.items(), colors):
        steps = list(range(len(sizes)))
        ax.plot(steps, sizes, label=name, color=color, linewidth=1.8)

    ax.set_xlabel("Episode Step", fontsize=12)
    ax.set_ylabel("Number of Stored Events", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.25, linestyle="--")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig 13] Saved: {output_path}")


# ------------------------------------------------------------------
# Fig 14: LLM cost breakdown per system
# ------------------------------------------------------------------

def plot_cost_breakdown(
    cost_data: dict[str, dict],
    output_path: str | Path = "docs/figures/fig14_cost_breakdown.png",
    title: str = "LLM Token Cost Breakdown by Memory System",
) -> None:
    """
    Stacked bar chart: prompt vs. memory tokens per system (Fig 14).

    Parameters
    ----------
    cost_data : {system_name: {mean_prompt_tokens, mean_memory_tokens,
                               mean_completion_tokens, mean_reward, total_cost_usd}}
    """
    systems = list(cost_data.keys())
    prompt_tokens = [cost_data[s].get("mean_prompt_tokens", 0) - cost_data[s].get("mean_memory_tokens", 0)
                     for s in systems]
    memory_tokens = [cost_data[s].get("mean_memory_tokens", 0) for s in systems]
    completion_tokens = [cost_data[s].get("mean_completion_tokens", 5) for s in systems]

    x = np.arange(len(systems))
    width = 0.6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Token stacked bar
    p1 = ax1.bar(x, memory_tokens, width, label="Memory context tokens", color="#3498db", alpha=0.85)
    p2 = ax1.bar(x, prompt_tokens, width, bottom=memory_tokens,
                 label="System+obs tokens", color="#e67e22", alpha=0.85)
    p3 = ax1.bar(x, completion_tokens, width,
                 bottom=[m + p for m, p in zip(memory_tokens, prompt_tokens)],
                 label="Completion tokens", color="#2ecc71", alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(systems, rotation=35, ha="right", fontsize=9)
    ax1.set_ylabel("Mean Tokens per Episode", fontsize=11)
    ax1.set_title("Token Breakdown", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Reward vs. cost scatter
    total_costs = [cost_data[s].get("total_cost_usd", 0) for s in systems]
    rewards = [cost_data[s].get("mean_reward", 0) for s in systems]
    scatter_colors = plt.cm.tab10(np.linspace(0, 1, len(systems)))
    for i, (sys, cost, reward) in enumerate(zip(systems, total_costs, rewards)):
        ax2.scatter(cost, reward, c=[scatter_colors[i]], s=80, label=sys,
                    edgecolors="black", linewidths=0.6, zorder=5)
        ax2.annotate(sys[:12], (cost, reward), xytext=(5, 0), textcoords="offset points",
                     fontsize=7.5)

    ax2.set_xlabel("Total Cost USD (all episodes)", fontsize=11)
    ax2.set_ylabel("Mean Reward", fontsize=11)
    ax2.set_title("Cost vs. Reward", fontsize=12)
    ax2.grid(True, alpha=0.25, linestyle="--")

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig 14] Saved: {output_path}")


# ------------------------------------------------------------------
# Fig 15: Multi-session memory persistence
# ------------------------------------------------------------------

def plot_multi_session_persistence(
    session_scores: list[float],
    session_memory_sizes: list[int],
    session_names: list[str] | None = None,
    output_path: str | Path = "docs/figures/fig15_multi_session.png",
    title: str = "Multi-Session Memory Persistence",
) -> None:
    """
    Plot per-session score and memory size over 20 sessions (Fig 15).

    Parameters
    ----------
    session_scores : list of partial_score per session
    session_memory_sizes : list of total memory events at end of each session
    session_names : optional list of session labels
    """
    n = len(session_scores)
    sessions = list(range(1, n + 1))
    labels = session_names or [f"S{i}" for i in sessions]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Score over sessions
    ax1.bar(sessions, session_scores, color="#2ecc71", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax1.plot(sessions, session_scores, "r-o", markersize=5, linewidth=1.5,
             label="Per-session score")

    # Trend line
    z = np.polyfit(sessions, session_scores, 1)
    p = np.poly1d(z)
    ax1.plot(sessions, p(sessions), "k--", linewidth=1.2, alpha=0.6, label=f"Trend (slope={z[0]:+.4f})")

    ax1.set_ylabel("Session Score", fontsize=11)
    ax1.set_title(title, fontsize=13)
    ax1.legend(fontsize=9)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.25, linestyle="--")

    # Memory size over sessions
    ax2.plot(sessions, session_memory_sizes, "-o", markersize=5, linewidth=1.8,
             label="Memory size (events)", color="#3498db")
    ax2.fill_between(sessions, session_memory_sizes, alpha=0.2, color="#3498db")
    ax2.set_xlabel("Session Number", fontsize=11)
    ax2.set_ylabel("Memory Events Retained", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25, linestyle="--")

    if len(labels) <= 20:
        ax2.set_xticks(sessions)
        ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig 15] Saved: {output_path}")
