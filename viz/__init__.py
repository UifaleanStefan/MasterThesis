"""Visualization package for the Learnable Memory thesis."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def generate_all_figures(
    env_data: dict,
    comparison_results: dict,
    output_dir: str | Path = "docs/figures",
) -> list[Path]:
    """
    Generate all 7 figures from the thesis experiment data.

    Parameters
    ----------
    env_data : dict
        Output from main.py's env_data dict:
          env_data[env_name] = {
            "best_theta", "phase6_results", "baseline_fixed",
            "learned_theta", "phase7_generations", "baseline_es"
          }
    comparison_results : dict
        Output from run_memory_comparison — keyed by system name.
    output_dir : str | Path
        Directory where PNGs will be saved.

    Returns
    -------
    list[Path]
        Paths of all successfully saved figures.
    """
    import traceback

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATION FIGURES")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Figure 1 — Memory Graph: Fixed vs. Learned theta (Key-Door)
    # ------------------------------------------------------------------ #
    print("\n[Fig 1] Memory graph: fixed vs. learned theta ...")
    try:
        from viz.graph_viz import plot_memory_graphs
        from memory.graph_memory import GraphMemory, MemoryParams
        from environment import ToyEnvironment
        from agent import ExplorationPolicy
        from agent.loop import run_episode_with_any_memory

        _env_g = ToyEnvironment(seed=42)
        _policy_g = ExplorationPolicy(seed=42)

        fixed_mem = GraphMemory(MemoryParams(1.0, 0.0, 1.0, "learnable"))
        run_episode_with_any_memory(_env_g, _policy_g, fixed_mem, episode_seed=42)

        learned_theta_kd = env_data.get("Key-Door", {}).get("learned_theta", (0.116, 0.0, 0.819))
        learned_mem = GraphMemory(MemoryParams(*learned_theta_kd, "learnable"))
        run_episode_with_any_memory(_env_g, _policy_g, learned_mem, episode_seed=42)

        p = plot_memory_graphs(
            fixed_mem.get_graph(),
            learned_mem.get_graph(),
            fixed_theta=(1.0, 0.0, 1.0),
            learned_theta=learned_theta_kd,
            env_name="Key-Door",
            output_dir=output_dir,
        )
        saved.append(p)
        print(f"  Saved: {p}")
    except Exception:
        print("  ERROR generating Fig 1:")
        traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Figure 2 — ES Learning Curves
    # ------------------------------------------------------------------ #
    print("\n[Fig 2] ES learning curves ...")
    try:
        from viz.es_curves import plot_es_learning_curves
        p = plot_es_learning_curves(env_data, output_dir)
        saved.append(p)
        print(f"  Saved: {p}")
    except Exception:
        print("  ERROR generating Fig 2:")
        traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Figure 3 — Theta Trajectory
    # ------------------------------------------------------------------ #
    print("\n[Fig 3] Theta trajectory over ES generations ...")
    try:
        from viz.es_curves import plot_theta_trajectory
        p = plot_theta_trajectory(env_data, output_dir)
        saved.append(p)
        print(f"  Saved: {p}")
    except Exception:
        print("  ERROR generating Fig 3:")
        traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Figure 4 — Bandit Landscape
    # ------------------------------------------------------------------ #
    print("\n[Fig 4] Phase 6 bandit landscape ...")
    try:
        from viz.bandit_landscape import plot_bandit_landscape
        p = plot_bandit_landscape(env_data, output_dir)
        saved.append(p)
        print(f"  Saved: {p}")
    except Exception:
        print("  ERROR generating Fig 4:")
        traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Figure 5 — Memory System Comparison
    # ------------------------------------------------------------------ #
    print("\n[Fig 5] Memory system comparison bar charts ...")
    try:
        from viz.memory_comparison import plot_memory_comparison
        p = plot_memory_comparison(comparison_results, output_dir)
        saved.append(p)
        print(f"  Saved: {p}")
    except Exception:
        print("  ERROR generating Fig 5:")
        traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Figure 6 — Grid Trajectory (MultiHopKeyDoor)
    # ------------------------------------------------------------------ #
    print("\n[Fig 6] Grid trajectory visualization ...")
    try:
        from viz.grid_viz import plot_grid_trajectory
        from environment import MultiHopKeyDoor
        from agent import ExplorationPolicy
        from memory.episodic_semantic_memory import EpisodicSemanticMemory

        _env_traj = MultiHopKeyDoor(seed=42)
        _policy_traj = ExplorationPolicy(seed=42)
        _mem_traj = EpisodicSemanticMemory(episodic_size=30)

        p = plot_grid_trajectory(
            _env_traj, _policy_traj, _mem_traj,
            output_dir=output_dir, episode_seed=42, k=8
        )
        saved.append(p)
        print(f"  Saved: {p}")
    except Exception:
        print("  ERROR generating Fig 6:")
        traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Figure 7 — Per-Episode Metric Curves (EpisodicSemantic on MultiHop)
    # ------------------------------------------------------------------ #
    print("\n[Fig 7] Per-episode metric curves (20 evaluation episodes) ...")
    try:
        from viz.episode_curves import collect_episode_metrics, plot_episode_metrics
        from environment import MultiHopKeyDoor
        from agent import ExplorationPolicy
        from memory.episodic_semantic_memory import EpisodicSemanticMemory

        _env_ep = MultiHopKeyDoor(seed=77)
        _policy_ep = ExplorationPolicy(seed=42)

        records = collect_episode_metrics(
            _env_ep, _policy_ep,
            memory_factory=lambda: EpisodicSemanticMemory(episodic_size=30),
            n_episodes=20, k=8, base_seed=200,
        )
        p = plot_episode_metrics(
            records,
            env_name="MultiHop-KeyDoor",
            system_name="EpisodicSemantic",
            output_dir=output_dir,
        )
        saved.append(p)
        print(f"  Saved: {p}")
    except Exception:
        print("  ERROR generating Fig 7:")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Figures generated: {len(saved)}/{7}")
    for p in saved:
        print(f"  {p}")
    print("=" * 60)
    return saved
