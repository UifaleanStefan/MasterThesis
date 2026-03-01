"""Regenerate Fig 6 and Fig 7 only."""
from pathlib import Path

output_dir = Path("docs/figures")

# ── Figure 6: grid trajectory with seed=36 (opens 2 doors) ──────────────────
print("[Fig 6] Grid trajectory seed=36 ...")
from viz.grid_viz import plot_grid_trajectory
from environment import MultiHopKeyDoor
from agent import ExplorationPolicy
from memory.episodic_semantic_memory import EpisodicSemanticMemory

env6 = MultiHopKeyDoor(seed=36)
pol6 = ExplorationPolicy(seed=36)
mem6 = EpisodicSemanticMemory(episodic_size=30)
p6 = plot_grid_trajectory(env6, pol6, mem6, output_dir=output_dir, episode_seed=36, k=8)
print(f"  Saved: {p6}")

# ── Figure 7: per-episode metrics with memory_size panel ────────────────────
print("[Fig 7] Per-episode metrics (20 eps, memory_size panel) ...")
from viz.episode_curves import collect_episode_metrics, plot_episode_metrics

env7 = MultiHopKeyDoor(seed=77)
pol7 = ExplorationPolicy(seed=42)
records = collect_episode_metrics(
    env7, pol7,
    memory_factory=lambda: EpisodicSemanticMemory(episodic_size=30),
    n_episodes=20, k=8, base_seed=200,
)
p7 = plot_episode_metrics(
    records,
    env_name="MultiHop-KeyDoor",
    system_name="EpisodicSemantic",
    output_dir=output_dir,
)
print(f"  Saved: {p7}")
print("Done.")
