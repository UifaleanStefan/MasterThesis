"""Figure 5 — Memory System Comparison Bar Charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_SYSTEM_COLORS = {
    "EpisodicSemantic": "#2196F3",
    "SemanticMemory": "#4CAF50",
    "GraphMemory+Theta": "#FF9800",
    "RAGMemory": "#9C27B0",
    "FlatWindow(50)": "#F44336",
    "SummaryMemory": "#795548",
}


def plot_memory_comparison(comparison_results: dict, output_dir: str | Path) -> Path:
    """
    Two grouped bar charts:
    - Left: mean_partial_score per system
    - Right: mean_retrieval_precision per system
    Systems sorted by partial score descending.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not comparison_results:
        return output_dir / "fig5_memory_comparison.png"

    # Sort systems by partial score
    sorted_systems = sorted(
        comparison_results.keys(),
        key=lambda s: comparison_results[s].get("mean_partial_score", 0),
        reverse=True,
    )

    scores = [comparison_results[s].get("mean_partial_score", 0) for s in sorted_systems]
    precisions = [comparison_results[s].get("mean_retrieval_precision") or 0 for s in sorted_systems]
    tokens = [comparison_results[s].get("mean_retrieval_tokens", 0) for s in sorted_systems]
    efficiencies = [comparison_results[s].get("efficiency", 0) for s in sorted_systems]

    colors = [_SYSTEM_COLORS.get(s, "#607D8B") for s in sorted_systems]
    short_names = [s.replace("(50)", "") for s in sorted_systems]
    x = np.arange(len(sorted_systems))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Memory System Comparison — MultiHopKeyDoor",
                 fontsize=14, fontweight="bold")

    # --- Panel 1: Partial score ---
    bars1 = axes[0].bar(x, scores, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("Mean partial score (doors/3)", fontsize=10)
    axes[0].set_title("Task Performance", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, max(scores) * 1.3 + 0.01)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, scores):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # --- Panel 2: Retrieval precision ---
    bars2 = axes[1].bar(x, precisions, color=colors, edgecolor="white", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("Retrieval precision", fontsize=10)
    axes[1].set_title("Memory Quality\n(Hint Retrieval Precision)", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0, 1.15)
    axes[1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, precisions):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # --- Panel 3: Efficiency ---
    efficiencies_scaled = [e * 1e4 for e in efficiencies]  # scale for readability
    bars3 = axes[2].bar(x, efficiencies_scaled, color=colors, edgecolor="white", linewidth=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short_names, rotation=30, ha="right", fontsize=9)
    axes[2].set_ylabel("Efficiency × 10⁴  (score / (1+tokens))", fontsize=10)
    axes[2].set_title("Token Efficiency", fontsize=11, fontweight="bold")
    axes[2].set_ylim(0, max(efficiencies_scaled) * 1.3 + 0.001)
    axes[2].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars3, efficiencies_scaled):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = output_dir / "fig5_memory_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
