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

        _env_traj = MultiHopKeyDoor(seed=36)
        _policy_traj = ExplorationPolicy(seed=36)
        _mem_traj = EpisodicSemanticMemory(episodic_size=30)

        p = plot_grid_trajectory(
            _env_traj, _policy_traj, _mem_traj,
            output_dir=output_dir, episode_seed=36, k=8
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
    print(f"Figures generated (original suite): {len(saved)}/7")
    for p in saved:
        print(f"  {p}")
    print("=" * 60)
    return saved


def generate_extended_figures(
    ablation_results: dict | None = None,
    landscape: dict | None = None,
    transfer_matrix: dict | None = None,
    benchmark_results: dict | None = None,
    output_dir: str | Path = "docs/figures",
) -> list[Path]:
    """
    Generate the extended figure suite (Fig 8–15) for Chapter 3-6.

    Parameters
    ----------
    ablation_results : from evaluation.ablation.run_ablation_study
    landscape : from evaluation.sensitivity.compute_sensitivity
    transfer_matrix : from evaluation.transfer.run_transfer_matrix
    benchmark_results : from evaluation.benchmark.run_full_benchmark
    output_dir : str | Path

    Returns
    -------
    list[Path] of saved figure paths.
    """
    import traceback
    import numpy as np
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    print("\n" + "=" * 60)
    print("GENERATING EXTENDED FIGURES (Fig 8–15)")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Fig 8 — Ablation study
    # ------------------------------------------------------------------ #
    print("\n[Fig 8] Ablation study bar chart ...")
    try:
        from viz.ablation_viz import plot_ablation_results
        if ablation_results is None:
            # Generate synthetic data for demonstration
            ablation_results = _synthetic_ablation()
        learned_theta = ablation_results.get("full", {}).get("theta", (0.5, 0.1, 0.8))
        p = output_dir / "fig08_ablation.png"
        plot_ablation_results(ablation_results, learned_theta, output_path=p)
        saved.append(p)
        print(f"  Saved: {p}")
    except Exception:
        print("  ERROR generating Fig 8:")
        traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Fig 9 — Reward landscape
    # ------------------------------------------------------------------ #
    print("\n[Fig 9] Reward landscape heatmap ...")
    try:
        from viz.landscape_viz import plot_reward_landscape
        if landscape is None:
            landscape = _synthetic_landscape()
        p = output_dir / "fig09_landscape.png"
        plot_reward_landscape(landscape, output_path=p, env_name="MultiHop-KeyDoor")
        saved.append(p)
        print(f"  Saved: {p}")
    except Exception:
        print("  ERROR generating Fig 9:")
        traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Fig 10 — Transfer matrix
    # ------------------------------------------------------------------ #
    print("\n[Fig 10] Cross-task transfer matrix ...")
    try:
        from viz.transfer_viz import plot_transfer_matrix
        if transfer_matrix is None:
            transfer_matrix = _synthetic_transfer()
        p = output_dir / "fig10_transfer.png"
        plot_transfer_matrix(transfer_matrix, output_path=p)
        saved.append(p)
        print(f"  Saved: {p}")
    except Exception:
        print("  ERROR generating Fig 10:")
        traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Fig 11 — Pareto frontier
    # ------------------------------------------------------------------ #
    print("\n[Fig 11] Token cost vs. reward Pareto frontier ...")
    try:
        from viz.pareto_viz import plot_pareto_frontier
        if benchmark_results:
            env_res = list(benchmark_results.values())[0]
            systems = list(env_res.keys())
            costs = [env_res[s].get("mean_tokens", 0) for s in systems]
            rewards = [env_res[s].get("mean_reward", 0) for s in systems]
            ci_rewards = [
                (env_res[s].get("ci_lower", env_res[s].get("mean_reward", 0)),
                 env_res[s].get("ci_upper", env_res[s].get("mean_reward", 0)))
                for s in systems
            ]
        else:
            systems, costs, rewards, ci_rewards = _synthetic_pareto()
        p = output_dir / "fig11_pareto.png"
        plot_pareto_frontier(systems, costs, rewards, ci_rewards, output_path=p)
        saved.append(p)
        print(f"  Saved: {p}")
    except Exception:
        print("  ERROR generating Fig 11:")
        traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Fig 12 — Online adaptation curves
    # ------------------------------------------------------------------ #
    print("\n[Fig 12] Online adaptation theta curves ...")
    try:
        from viz.cost_viz import plot_online_adaptation
        theta_histories = _synthetic_online_adaptation()
        p = output_dir / "fig12_online_adaptation.png"
        plot_online_adaptation(theta_histories, output_path=p)
        saved.append(p)
        print(f"  Saved: {p}")
    except Exception:
        print("  ERROR generating Fig 12:")
        traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Fig 13 — Memory size over episode
    # ------------------------------------------------------------------ #
    print("\n[Fig 13] Memory size over episode steps ...")
    try:
        from viz.cost_viz import plot_memory_size_over_episode
        memory_curves = _synthetic_memory_curves()
        p = output_dir / "fig13_memory_size.png"
        plot_memory_size_over_episode(memory_curves, output_path=p)
        saved.append(p)
        print(f"  Saved: {p}")
    except Exception:
        print("  ERROR generating Fig 13:")
        traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Fig 14 — LLM cost breakdown
    # ------------------------------------------------------------------ #
    print("\n[Fig 14] LLM cost breakdown ...")
    try:
        from viz.cost_viz import plot_cost_breakdown
        cost_data = _synthetic_cost_data()
        p = output_dir / "fig14_cost_breakdown.png"
        plot_cost_breakdown(cost_data, output_path=p)
        saved.append(p)
        print(f"  Saved: {p}")
    except Exception:
        print("  ERROR generating Fig 14:")
        traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Fig 15 — Multi-session persistence
    # ------------------------------------------------------------------ #
    print("\n[Fig 15] Multi-session memory persistence ...")
    try:
        from viz.cost_viz import plot_multi_session_persistence
        scores, sizes = _synthetic_multi_session()
        p = output_dir / "fig15_multi_session.png"
        plot_multi_session_persistence(scores, sizes, output_path=p)
        saved.append(p)
        print(f"  Saved: {p}")
    except Exception:
        print("  ERROR generating Fig 15:")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Extended figures generated: {len(saved)}/8")
    for p in saved:
        print(f"  {p}")
    print("=" * 60)
    return saved


# ------------------------------------------------------------------
# Synthetic data generators for demonstration (pre-experiment)
# ------------------------------------------------------------------

def _synthetic_ablation() -> dict:
    import random
    rng = random.Random(42)
    configs = ["full", "baseline", "no_store_filt", "no_entity_filt",
               "no_temp_filt", "zero_store", "zero_entity", "zero_temporal"]
    full_reward = 0.65
    results = {}
    for name in configs:
        if name == "full":
            r = full_reward
        elif name == "baseline":
            r = 0.45
        elif name.startswith("no_"):
            r = full_reward * rng.uniform(0.7, 0.9)
        else:
            r = full_reward * rng.uniform(0.3, 0.6)
        rewards = [max(0, r + rng.gauss(0, 0.08)) for _ in range(30)]
        results[name] = {
            "theta": (0.5, 0.1, 0.8),
            "description": name,
            "mean_reward": sum(rewards) / len(rewards),
            "std_reward": (sum((x - sum(rewards)/len(rewards))**2 for x in rewards)/len(rewards))**0.5,
            "mean_tokens": rng.uniform(50, 200),
            "mean_precision": rng.uniform(0.3, 0.9),
            "rewards": rewards,
            "degradation": max(0, (full_reward - sum(rewards)/len(rewards)) / full_reward),
        }
    return results


def _synthetic_landscape() -> dict:
    import numpy as np
    stores = np.linspace(0, 1, 10).tolist()
    entities = np.linspace(0, 1, 10).tolist()
    s_arr = np.array(stores)
    e_arr = np.array(entities)
    S, E = np.meshgrid(s_arr, e_arr, indexing="ij")
    grid = 0.6 * np.exp(-((S - 0.85)**2 / 0.08 + (E - 0.15)**2 / 0.02)) + \
           0.05 * np.random.RandomState(42).randn(*S.shape)
    grid = np.clip(grid, 0, 1)
    return {
        "theta_store_values": stores,
        "theta_entity_values": entities,
        "reward_grid": grid.tolist(),
        "best_theta": (0.85, 0.15, 0.80),
        "best_reward": float(grid.max()),
        "fixed_temporal": 0.80,
        "resolution": 10,
        "n_episodes_per_cell": 20,
    }


def _synthetic_transfer() -> dict:
    import random
    rng = random.Random(42)
    tasks = ["Key-Door", "Goal-Room", "MultiHop"]
    matrix = {}
    for src in tasks:
        matrix[src] = {}
        for tgt in tasks:
            if src == tgt:
                r = rng.uniform(0.55, 0.75)
            else:
                r = rng.uniform(0.20, 0.50)
            matrix[src][tgt] = {"mean_reward": r, "std_reward": 0.08, "mean_tokens": 100}
    return matrix


def _synthetic_pareto():
    systems = [
        "FlatWindow(50)", "GraphMemory+Theta", "SemanticMemory", "SummaryMemory",
        "EpisodicSemantic", "RAGMemory", "HierarchicalMemory", "WorkingMemory(7)",
        "CausalMemory", "AttentionMemory",
    ]
    costs = [480, 240, 180, 160, 200, 320, 220, 70, 190, 250]
    rewards = [0.30, 0.55, 0.45, 0.42, 0.62, 0.51, 0.58, 0.38, 0.50, 0.53]
    ci_rewards = [(r - 0.05, r + 0.05) for r in rewards]
    return systems, costs, rewards, ci_rewards


def _synthetic_online_adaptation():
    import random
    rng = random.Random(42)
    n_steps = 20
    histories = {
        "StatisticsAdapter": [],
        "GradientAdapter": [],
        "Baseline (fixed)": [],
    }
    s, e, t = 0.5, 0.1, 0.8
    for step in range(n_steps):
        noise = rng.gauss(0, 0.02)
        s_new = min(1.0, s + 0.02 * (0.3 - rng.random()) + noise)
        histories["StatisticsAdapter"].append((s_new, e, t))
        s = s_new

    s2, e2, t2 = 0.5, 0.1, 0.8
    for step in range(n_steps):
        trend = 0.015 if step < 10 else -0.008
        s2 = min(1.0, max(0, s2 + trend + rng.gauss(0, 0.01)))
        histories["GradientAdapter"].append((s2, e2, t2))

    for step in range(n_steps):
        histories["Baseline (fixed)"].append((0.5, 0.1, 0.8))

    return histories


def _synthetic_memory_curves():
    import random
    rng = random.Random(42)
    n_steps = 250
    systems = {
        "FlatWindow(50)": [],
        "EpisodicSemantic": [],
        "WorkingMemory(7)": [],
        "HierarchicalMemory": [],
        "AttentionMemory": [],
    }
    for step in range(n_steps):
        systems["FlatWindow(50)"].append(min(50, step + 1))
        systems["EpisodicSemantic"].append(min(30 + step // 10, 50))
        systems["WorkingMemory(7)"].append(7)
        systems["HierarchicalMemory"].append(min(20 + step // 15, 42))
        systems["AttentionMemory"].append(step + 1)
    return systems


def _synthetic_cost_data():
    return {
        "FlatWindow(50)":    {"mean_prompt_tokens": 480, "mean_memory_tokens": 380, "mean_completion_tokens": 5, "mean_reward": 0.30, "total_cost_usd": 0.000072},
        "GraphMemory+Theta": {"mean_prompt_tokens": 280, "mean_memory_tokens": 180, "mean_completion_tokens": 5, "mean_reward": 0.55, "total_cost_usd": 0.000042},
        "EpisodicSemantic":  {"mean_prompt_tokens": 240, "mean_memory_tokens": 140, "mean_completion_tokens": 5, "mean_reward": 0.62, "total_cost_usd": 0.000036},
        "WorkingMemory(7)":  {"mean_prompt_tokens": 120, "mean_memory_tokens": 80,  "mean_completion_tokens": 5, "mean_reward": 0.38, "total_cost_usd": 0.000018},
        "AttentionMemory":   {"mean_prompt_tokens": 300, "mean_memory_tokens": 200, "mean_completion_tokens": 5, "mean_reward": 0.53, "total_cost_usd": 0.000045},
    }


def _synthetic_multi_session():
    import random
    rng = random.Random(42)
    scores = []
    sizes = []
    base_score = 0.3
    base_size = 10
    for i in range(20):
        scores.append(min(1.0, base_score + i * 0.025 + rng.gauss(0, 0.05)))
        sizes.append(int(base_size + i * 5 + rng.gauss(0, 3)))
    return scores, sizes
