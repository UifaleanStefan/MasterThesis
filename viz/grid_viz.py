"""Figure 6 — Grid Trajectory Visualization for MultiHopKeyDoor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np


_KEY_COLORS_MAP = {
    "red": "#F44336",
    "blue": "#2196F3",
    "green": "#4CAF50",
    "yellow": "#FFEB3B",
    "purple": "#9C27B0",
    "orange": "#FF9800",
    "cyan": "#00BCD4",
    "magenta": "#E91E63",
    "white": "#EEEEEE",
}

_DISTRACTOR_SUFFIXES = ("_distractor", "_spare", "_fake")


def run_episode_with_trajectory(env: Any, policy: Any, memory: Any, k: int = 8,
                                 episode_seed: int = 42) -> tuple[list, list, list, float]:
    """
    Run one episode while recording agent positions at each step.
    Returns: (trajectory, events, hint_steps, partial_score)
    """
    from agent.loop import run_episode_with_any_memory
    import copy

    trajectory: list[tuple[int, int]] = []

    # We need step-by-step positions; we run the loop manually here
    memory.clear()
    obs = env.reset()
    from memory.event import Event

    env_hints = getattr(env, "hint_observations", [])
    step = 0
    events = []

    while not env.done:
        trajectory.append(env.agent_pos)
        past = memory.get_relevant_events(obs, step, k=k)
        action = policy.decide(obs, past)
        is_hint = any(h in obs for h in env_hints) if env_hints else False
        event = Event(step=step, observation=obs, action=action, is_hint=is_hint)
        memory.add_event(event, episode_seed=episode_seed)
        events.append(event)
        obs, done, success = env.step(action)
        step += 1

    # final position
    trajectory.append(env.agent_pos)

    partial = getattr(env, "partial_score", float(env.success))
    hint_steps = [i for i, e in enumerate(events) if e.is_hint]
    return trajectory, events, hint_steps, partial


def plot_grid_trajectory(
    env: Any,
    policy: Any,
    memory: Any,
    output_dir: str | Path,
    episode_seed: int = 42,
    k: int = 8,
) -> Path:
    """
    Render the MultiHopKeyDoor grid after running one episode:
    - Visit-count heatmap (background)
    - Key positions (colored squares)
    - Door positions (triangles)
    - Agent start (green circle) and end (red circle)
    - Hint annotation text box
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory, events, hint_steps, partial_score = run_episode_with_trajectory(
        env, policy, memory, k=k, episode_seed=episode_seed
    )

    grid_w = getattr(env, "width", 10)
    grid_h = getattr(env, "height", 10)

    # Build visit count grid
    visit_grid = np.zeros((grid_h, grid_w), dtype=float)
    for (x, y) in trajectory:
        if 0 <= x < grid_w and 0 <= y < grid_h:
            visit_grid[y, x] += 1

    fig, ax = plt.subplots(figsize=(8, 8))

    # Heatmap
    cmap = plt.cm.Blues
    masked = np.ma.masked_where(visit_grid == 0, visit_grid)
    ax.imshow(masked, cmap=cmap, origin="lower", aspect="equal",
              extent=[-0.5, grid_w - 0.5, -0.5, grid_h - 0.5],
              vmin=0, vmax=max(visit_grid.max(), 1))

    # Unvisited cells light grey
    unvisited = np.ma.masked_where(visit_grid > 0, np.ones_like(visit_grid))
    ax.imshow(unvisited, cmap="Greys", origin="lower", aspect="equal",
              extent=[-0.5, grid_w - 0.5, -0.5, grid_h - 0.5],
              vmin=0, vmax=1, alpha=0.2)

    # Draw grid lines
    for xi in range(grid_w + 1):
        ax.axvline(xi - 0.5, color="grey", linewidth=0.4, alpha=0.5)
    for yi in range(grid_h + 1):
        ax.axhline(yi - 0.5, color="grey", linewidth=0.4, alpha=0.5)

    # Key positions
    key_positions = getattr(env, "key_positions", {})
    door_key_map = getattr(env, "door_key_map", [])
    real_key_colors = set(door_key_map) if door_key_map else set()

    for key_name, (kx, ky) in key_positions.items():
        base_color_name = key_name.split("_")[0]
        color = _KEY_COLORS_MAP.get(base_color_name, "#AAAAAA")
        is_real = base_color_name in real_key_colors
        edgecolor = "black" if is_real else "red"
        lw = 1.5 if is_real else 2.5
        rect = mpatches.FancyBboxPatch(
            (kx - 0.35, ky - 0.35), 0.7, 0.7,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor=edgecolor, linewidth=lw, zorder=3
        )
        ax.add_patch(rect)
        label = base_color_name[:3]
        ax.text(kx, ky, label, ha="center", va="center", fontsize=6,
                fontweight="bold", color="black", zorder=4)

    # Door positions
    door_positions = getattr(env, "door_positions", [])
    doors_opened = getattr(env, "doors_opened", [False] * len(door_positions))
    door_names = ["north", "east", "south"]

    for i, (dx, dy) in enumerate(door_positions):
        opened = doors_opened[i] if i < len(doors_opened) else False
        needed_color = door_key_map[i] if i < len(door_key_map) else "grey"
        fill = _KEY_COLORS_MAP.get(needed_color, "#AAAAAA")
        marker_color = "white" if opened else fill
        edge_color = "#1A237E" if opened else "black"
        triangle = plt.Polygon(
            [[dx, dy + 0.45], [dx - 0.4, dy - 0.35], [dx + 0.4, dy - 0.35]],
            closed=True, facecolor=marker_color, edgecolor=edge_color,
            linewidth=2, zorder=3
        )
        ax.add_patch(triangle)
        label = door_names[i][:1].upper() if i < len(door_names) else str(i)
        ax.text(dx, dy - 0.05, label, ha="center", va="center", fontsize=7,
                fontweight="bold", color="black", zorder=4)

    # Agent start and end
    if trajectory:
        sx, sy = trajectory[0]
        ex, ey = trajectory[-1]
        ax.plot(sx, sy, "o", color="limegreen", markersize=14, zorder=5,
                markeredgecolor="black", markeredgewidth=1.2, label="Start")
        ax.plot(ex, ey, "o", color="crimson", markersize=14, zorder=5,
                markeredgecolor="black", markeredgewidth=1.2, label="End")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, max(visit_grid.max(), 1)))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label("Visit count", fontsize=9)

    # Hint annotation
    hint_obs = getattr(env, "hint_observations", [])
    if hint_obs:
        hint_text = "Hints (steps 0-2):\n" + "\n".join(f"  {h}" for h in hint_obs)
    else:
        hint_text = "No hints"
    ax.text(0.01, 0.99, hint_text, transform=ax.transAxes, fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="goldenrod", alpha=0.9))

    ax.set_xlim(-0.5, grid_w - 0.5)
    ax.set_ylim(-0.5, grid_h - 0.5)
    ax.set_xticks(range(grid_w))
    ax.set_yticks(range(grid_h))
    ax.tick_params(labelsize=7)

    legend_elements = [
        mpatches.Patch(facecolor="#4C9BE8", edgecolor="black", label="Real key"),
        mpatches.Patch(facecolor="#AAAAAA", edgecolor="red", linewidth=2.5, label="Distractor key"),
        mpatches.Patch(facecolor="#AAAAAA", edgecolor="#1A237E", label="Door (closed)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="limegreen",
                   markersize=10, markeredgecolor="black", label="Agent start"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson",
                   markersize=10, markeredgecolor="black", label="Agent end"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7,
              framealpha=0.9)

    ax.set_title(
        f"MultiHopKeyDoor — Agent Trajectory (episode seed={episode_seed})\n"
        f"Steps: {len(trajectory)-1}  |  Partial score: {partial_score:.2f}",
        fontsize=11, fontweight="bold"
    )

    plt.tight_layout()
    out_path = output_dir / "fig6_grid_trajectory.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
