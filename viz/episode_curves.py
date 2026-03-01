"""Figure 7 — Per-Episode Metric Curves over evaluation episodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def collect_episode_metrics(
    env: Any,
    policy: Any,
    memory_factory,
    n_episodes: int = 20,
    k: int = 8,
    base_seed: int = 100,
) -> list[dict]:
    """
    Run n_episodes with a fresh memory each time, collect per-episode stats.
    memory_factory: callable() -> memory instance
    """
    from agent.loop import run_episode_with_any_memory

    records = []
    for ep in range(n_episodes):
        mem = memory_factory()
        seed = base_seed + ep
        success, events, stats = run_episode_with_any_memory(
            env, policy, mem, k=k, episode_seed=seed
        )
        records.append({
            "episode": ep + 1,
            "reward": stats.get("reward", float(success)),
            "retrieval_tokens": stats.get("retrieval_tokens", 0),
            "retrieval_precision": stats.get("retrieval_precision"),
        })
    return records


def plot_episode_metrics(
    records: list[dict],
    env_name: str,
    system_name: str,
    output_dir: str | Path,
) -> Path:
    """
    Three-panel figure:
    - Top: reward per episode (bar) + rolling mean (line)
    - Middle: retrieval_tokens per episode
    - Bottom: retrieval_precision per episode (scatter)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eps = [r["episode"] for r in records]
    rewards = [r["reward"] for r in records]
    tokens = [r["retrieval_tokens"] for r in records]
    precisions = [r["retrieval_precision"] if r["retrieval_precision"] is not None else float("nan")
                  for r in records]

    window = max(1, len(eps) // 5)
    rolling_reward = np.convolve(rewards, np.ones(window) / window, mode="same")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(
        f"Per-Episode Metrics — {env_name} / {system_name}",
        fontsize=13, fontweight="bold"
    )

    # Panel 1: Reward
    axes[0].bar(eps, rewards, color="#90CAF9", alpha=0.7, label="Reward")
    axes[0].plot(eps, rolling_reward, color="#1565C0", linewidth=2,
                 label=f"Rolling mean (w={window})")
    mean_r = np.nanmean(rewards)
    axes[0].axhline(mean_r, color="grey", linestyle=":", linewidth=1.2,
                    label=f"Mean={mean_r:.3f}")
    axes[0].set_ylabel("Reward", fontsize=10)
    axes[0].set_ylim(0, max(max(rewards) * 1.2, 0.1))
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: Retrieval tokens
    axes[1].plot(eps, tokens, color="#FF7043", marker=".", linewidth=1.5,
                 markersize=4, label="Retrieval tokens")
    axes[1].fill_between(eps, tokens, alpha=0.15, color="#FF7043")
    mean_t = np.mean(tokens)
    axes[1].axhline(mean_t, color="grey", linestyle=":", linewidth=1.2,
                    label=f"Mean={mean_t:.0f}")
    axes[1].set_ylabel("Retrieval tokens", fontsize=10)
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(axis="y", alpha=0.3)

    # Panel 3: Retrieval precision
    valid_eps = [e for e, p in zip(eps, precisions) if not np.isnan(p)]
    valid_prec = [p for p in precisions if not np.isnan(p)]
    if valid_prec:
        axes[2].scatter(valid_eps, valid_prec, color="#7B1FA2", s=40, zorder=3,
                        label="Retrieval precision")
        axes[2].plot(valid_eps, valid_prec, color="#7B1FA2", linewidth=1,
                     alpha=0.5)
        mean_p = np.nanmean(valid_prec)
        axes[2].axhline(mean_p, color="grey", linestyle=":", linewidth=1.2,
                        label=f"Mean={mean_p:.3f}")
        axes[2].set_ylim(-0.05, 1.1)
    else:
        axes[2].text(0.5, 0.5, "No precision data\n(no door-query steps)",
                     ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set_ylabel("Retrieval precision", fontsize=10)
    axes[2].set_xlabel("Episode", fontsize=10)
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    safe_name = system_name.lower().replace("+", "").replace("(", "").replace(")", "").replace(" ", "_")
    out_path = output_dir / f"fig7_episode_metrics_{safe_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
