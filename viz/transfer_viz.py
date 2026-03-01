"""
Fig 10 — Cross-task transfer matrix heatmap.

Rows = theta source (trained on), Columns = evaluation task.
Cell value = mean_reward on evaluation task using source theta.
Diagonal = in-distribution performance.
Off-diagonal = transfer performance.

Color: green = high reward, red = low reward.
Annotation: reward value in each cell.
Diagonal highlight: dashed border to distinguish in-distribution cells.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_transfer_matrix(
    matrix: dict[str, dict[str, dict]],
    metric: str = "mean_reward",
    output_path: str | Path = "docs/figures/fig10_transfer.png",
    title: str = "Cross-Task Transfer Matrix",
) -> None:
    """
    Plot the cross-task transfer matrix as a heatmap (Fig 10).

    Parameters
    ----------
    matrix : dict from evaluation.transfer.run_transfer_matrix
        {source_task: {target_task: result_dict}}
    metric : str
        Metric to visualize (default: mean_reward)
    output_path : path to save PNG
    """
    source_tasks = list(matrix.keys())
    target_tasks = list(matrix[source_tasks[0]].keys()) if source_tasks else []
    n_src, n_tgt = len(source_tasks), len(target_tasks)

    # Build value matrix
    values = np.zeros((n_src, n_tgt))
    for i, src in enumerate(source_tasks):
        for j, tgt in enumerate(target_tasks):
            values[i, j] = matrix[src][tgt].get(metric, 0.0)

    fig, ax = plt.subplots(figsize=(max(6, n_tgt * 1.8), max(5, n_src * 1.5)))
    im = ax.imshow(values, cmap="RdYlGn", aspect="auto",
                   vmin=0.0, vmax=max(values.max(), 0.01))
    plt.colorbar(im, ax=ax, label=metric)

    # Annotate cells
    for i in range(n_src):
        for j in range(n_tgt):
            val = values[i, j]
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    # Highlight diagonal (in-distribution)
    for k in range(min(n_src, n_tgt)):
        src_idx = k
        tgt_name = source_tasks[k]
        if tgt_name in target_tasks:
            tgt_idx = target_tasks.index(tgt_name)
            rect = patches.Rectangle(
                (tgt_idx - 0.5, src_idx - 0.5), 1, 1,
                linewidth=2.5, edgecolor="blue", facecolor="none", linestyle="--"
            )
            ax.add_patch(rect)

    ax.set_xticks(range(n_tgt))
    ax.set_yticks(range(n_src))
    ax.set_xticklabels(target_tasks, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(source_tasks, fontsize=10)
    ax.set_xlabel("Evaluation Task", fontsize=12)
    ax.set_ylabel("Theta Source (Trained On)", fontsize=12)
    ax.set_title(f"{title}\n(dashed border = in-distribution)", fontsize=13)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig 10] Saved: {output_path}")
