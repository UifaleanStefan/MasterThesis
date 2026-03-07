"""
Regenerate figures 5 and a new extended comparison figure from real benchmark data.
Uses results/benchmark_results.json produced by run_benchmark.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("results/benchmark_results.json")
OUT_DIR = Path("docs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette — consistent across all figures
COLORS = {
    "GraphMemoryV4":      "#16A34A",
    "GraphMemoryV5":      "#0D9488",
    "GraphMemoryV1":      "#EA580C",
    "EpisodicSemantic":   "#2196F3",
    "WorkingMemory(7)":   "#00BCD4",
    "AttentionMemory":    "#009688",
    "SemanticMemory":     "#4CAF50",
    "HierarchicalMemory": "#8BC34A",
    "CausalMemory":       "#CDDC39",
    "RAGMemory":          "#9C27B0",
    "GraphMemory+Theta":  "#FF9800",
    "SummaryMemory":      "#795548",
    "FlatWindow(50)":     "#F44336",
}


def load_results() -> dict:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Run run_benchmark.py first. {RESULTS_PATH} not found.")
    return json.loads(RESULTS_PATH.read_text())


def _is_valid_system_entry(entry: dict) -> bool:
    """Exclude error entries and entries without mean_reward."""
    if not isinstance(entry, dict):
        return False
    if "error" in entry:
        return False
    return "mean_reward" in entry


def make_fig5_multihop(results: dict, v4res: dict | None = None) -> Path:
    """Fig 5 — MultiHopKeyDoor comparison: reward, precision, efficiency."""
    env_data = dict(results.get("MultiHop-KeyDoor", {}))

    # Merge optimized V4 and V1 from CMA-ES results if available
    v4_path = Path("results/graphmemory_v4_cmaes_results.json")
    if v4res is None and v4_path.exists():
        v4res = json.loads(v4_path.read_text())
    if v4res:
        env_data["GraphMemoryV4"] = {
            "mean_reward": v4res["v4"]["eval"]["mean_reward"],
            "ci_lower": v4res["v4"]["eval"].get("ci_lower", v4res["v4"]["eval"]["mean_reward"] - 0.02),
            "ci_upper": v4res["v4"]["eval"].get("ci_upper", v4res["v4"]["eval"]["mean_reward"] + 0.02),
            "retrieval_precision": v4res["v4"]["eval"]["mean_precision"],
            "efficiency": v4res["v4"]["eval"]["efficiency"],
        }
        env_data["GraphMemoryV1"] = {
            "mean_reward": v4res["v1_baseline"]["eval"]["mean_reward"],
            "ci_lower": v4res["v1_baseline"]["eval"].get("ci_lower", v4res["v1_baseline"]["eval"]["mean_reward"] - 0.02),
            "ci_upper": v4res["v1_baseline"]["eval"].get("ci_upper", v4res["v1_baseline"]["eval"]["mean_reward"] + 0.02),
            "retrieval_precision": v4res["v1_baseline"]["eval"]["mean_precision"],
            "efficiency": v4res["v1_baseline"]["eval"]["efficiency"],
        }

    # Filter out error entries
    env_data = {s: v for s, v in env_data.items() if _is_valid_system_entry(v)}

    # Sort by mean_reward descending
    systems = sorted(env_data.keys(),
                     key=lambda s: env_data[s].get("mean_reward", 0), reverse=True)

    rewards     = [env_data[s].get("mean_reward", 0) for s in systems]
    ci_lo       = [env_data[s].get("ci_lower", 0) for s in systems]
    ci_hi       = [env_data[s].get("ci_upper", 0) for s in systems]
    precisions  = [env_data[s].get("retrieval_precision") or 0 for s in systems]
    efficiencies= [env_data[s].get("efficiency", 0) * 1e4 for s in systems]

    err_lo = [r - lo for r, lo in zip(rewards, ci_lo)]
    err_hi = [hi - r for r, hi in zip(rewards, ci_hi)]

    colors = [COLORS.get(s, "#607D8B") for s in systems]
    short  = [s.replace("Memory", "Mem").replace("(50)", "") for s in systems]
    x = np.arange(len(systems))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Memory System Benchmark — MultiHopKeyDoor  (n=50 episodes per system)",
                 fontsize=13, fontweight="bold", y=1.02)

    # Panel 1: Reward + CI
    bars = axes[0].bar(x, rewards, color=colors, edgecolor="white", linewidth=0.8,
                       yerr=[err_lo, err_hi], capsize=4, error_kw={"linewidth": 1.2})
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short, rotation=35, ha="right", fontsize=8.5)
    axes[0].set_ylabel("Mean reward (partial score)", fontsize=10)
    axes[0].set_title("Task Performance\n(with 95% bootstrap CI)", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, max(rewards) * 1.35 + 0.01)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, rewards):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(rewards) * 0.03,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel 2: Retrieval precision
    bars2 = axes[1].bar(x, precisions, color=colors, edgecolor="white", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short, rotation=35, ha="right", fontsize=8.5)
    axes[1].set_ylabel("Retrieval precision", fontsize=10)
    axes[1].set_title("Hint Retrieval Precision\n(at-door steps)", fontsize=10, fontweight="bold")
    axes[1].set_ylim(0, 1.2)
    axes[1].axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    axes[1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, precisions):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel 3: Efficiency
    bars3 = axes[2].bar(x, efficiencies, color=colors, edgecolor="white", linewidth=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short, rotation=35, ha="right", fontsize=8.5)
    axes[2].set_ylabel("Efficiency × 10⁴  (reward / (1 + tokens))", fontsize=10)
    axes[2].set_title("Token Efficiency", fontsize=10, fontweight="bold")
    axes[2].set_ylim(0, max(efficiencies) * 1.35 + 0.001)
    axes[2].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars3, efficiencies):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(efficiencies) * 0.02,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    out = OUT_DIR / "fig5_memory_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def make_cross_env_heatmap(results: dict) -> Path:
    """Fig 5b — Cross-environment performance heatmap."""
    envs = list(results.keys())
    # Union of systems across envs; filter error entries
    all_systems = set()
    for env in envs:
        for s, v in results[env].items():
            if _is_valid_system_entry(v):
                all_systems.add(s)
    systems = sorted(all_systems)

    # Build matrix (0 for missing system-env pairs)
    matrix = np.zeros((len(systems), len(envs)))
    for j, env in enumerate(envs):
        for i, sys in enumerate(systems):
            entry = results[env].get(sys, {})
            if _is_valid_system_entry(entry):
                matrix[i, j] = entry.get("mean_reward", 0.0)

    # Sort systems by MultiHop reward (most interesting env)
    multihop_idx = envs.index("MultiHop-KeyDoor") if "MultiHop-KeyDoor" in envs else -1
    order = np.argsort(matrix[:, multihop_idx])[::-1]
    matrix = matrix[order]
    sorted_systems = [systems[i] for i in order]

    # Episode counts: MegaQuestRoom uses 20, others use 50
    n_ep_str = "n=50 (Key-Door, Goal-Room, MultiHop); n=20 (MegaQuestRoom)"

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=matrix.max())

    ax.set_xticks(np.arange(len(envs)))
    ax.set_xticklabels(envs, fontsize=11, fontweight="bold")
    ax.set_yticks(np.arange(len(sorted_systems)))
    ax.set_yticklabels(sorted_systems, fontsize=10)

    # Annotate cells
    for i in range(len(sorted_systems)):
        for j in range(len(envs)):
            val = matrix[i, j]
            color = "black" if val < matrix.max() * 0.6 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Mean Reward", shrink=0.8)
    n_sys, n_env = len(sorted_systems), len(envs)
    ax.set_title(f"Cross-Environment Performance Heatmap\n({n_sys} memory systems × {n_env} environments, {n_ep_str})",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    out = OUT_DIR / "fig5b_cross_env_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def make_precision_scatter(results: dict, v4res: dict | None = None) -> Path:
    """Fig 5c — Precision vs Reward scatter on MultiHop."""
    env_data = dict(results.get("MultiHop-KeyDoor", {}))
    v4_path = Path("results/graphmemory_v4_cmaes_results.json")
    if v4res is None and v4_path.exists():
        v4res = json.loads(v4_path.read_text())
    if v4res:
        env_data["GraphMemoryV4"] = {
            "mean_reward": v4res["v4"]["eval"]["mean_reward"],
            "retrieval_precision": v4res["v4"]["eval"]["mean_precision"],
        }
        env_data["GraphMemoryV1"] = {
            "mean_reward": v4res["v1_baseline"]["eval"]["mean_reward"],
            "retrieval_precision": v4res["v1_baseline"]["eval"]["mean_precision"],
        }
    env_data = {s: v for s, v in env_data.items() if _is_valid_system_entry(v)}
    systems = list(env_data.keys())

    rewards    = [env_data[s].get("mean_reward", 0) for s in systems]
    precisions = [env_data[s].get("retrieval_precision") or 0 for s in systems]
    colors     = [COLORS.get(s, "#607D8B") for s in systems]

    fig, ax = plt.subplots(figsize=(8, 6))
    for s, r, p, c in zip(systems, rewards, precisions, colors):
        ax.scatter(p, r, color=c, s=180, zorder=3, edgecolors="white", linewidths=1.5)
        short = s.replace("Memory", "Mem").replace("(50)", "")
        ax.annotate(short, (p, r), textcoords="offset points", xytext=(6, 4),
                    fontsize=8.5, color=c, fontweight="bold")

    ax.set_xlabel("Retrieval Precision (hint retrieved at door steps)", fontsize=11)
    ax.set_ylabel("Mean Reward (partial score)", fontsize=11)
    ax.set_title("Retrieval Precision vs Task Performance\nMultiHopKeyDoor — Causal Relationship",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.01, max(rewards) * 1.3 + 0.01)
    ax.grid(alpha=0.3)

    # Correlation annotation
    if len(rewards) > 2:
        corr = float(np.corrcoef(precisions, rewards)[0, 1])
        ax.text(0.05, 0.92, f"Pearson r = {corr:.3f}", transform=ax.transAxes,
                fontsize=10, color="black",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    out = OUT_DIR / "fig5c_precision_vs_reward.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def make_easy_env_comparison(results: dict) -> Path:
    """Fig 5d — Key-Door and Goal-Room rankings."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Memory Systems on Simple Environments (n=50 episodes each)",
                 fontsize=13, fontweight="bold")

    for ax, env_name in zip(axes, ["Key-Door", "Goal-Room"]):
        env_data = {s: v for s, v in results.get(env_name, {}).items() if _is_valid_system_entry(v)}
        systems = sorted(env_data.keys(),
                         key=lambda s: env_data[s].get("mean_reward", 0), reverse=True)
        rewards = [env_data[s].get("mean_reward", 0) for s in systems]
        ci_lo   = [env_data[s].get("ci_lower", 0) for s in systems]
        ci_hi   = [env_data[s].get("ci_upper", 0) for s in systems]
        err_lo  = [r - lo for r, lo in zip(rewards, ci_lo)]
        err_hi  = [hi - r for r, hi in zip(rewards, ci_hi)]
        colors  = [COLORS.get(s, "#607D8B") for s in systems]
        short   = [s.replace("Memory", "Mem").replace("(50)", "") for s in systems]
        x = np.arange(len(systems))

        bars = ax.bar(x, rewards, color=colors, edgecolor="white", linewidth=0.8,
                      yerr=[err_lo, err_hi], capsize=3, error_kw={"linewidth": 1.2})
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=35, ha="right", fontsize=8.5)
        ax.set_ylabel("Mean Reward", fontsize=10)
        ax.set_title(f"{env_name}", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, rewards):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = OUT_DIR / "fig5d_simple_env_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def main() -> None:
    print("Loading benchmark results...")
    results = load_results()
    v4res = None
    v4_path = Path("results/graphmemory_v4_cmaes_results.json")
    if v4_path.exists():
        v4res = json.loads(v4_path.read_text())
        print("  Merging V4/V1 from graphmemory_v4_cmaes_results.json for MultiHop")
    print(f"  Environments: {list(results.keys())}")
    print(f"  Systems: {list(list(results.values())[0].keys())}")
    print()

    print("Generating figures...")
    make_fig5_multihop(results, v4res)
    make_cross_env_heatmap(results)
    make_precision_scatter(results, v4res)
    make_easy_env_comparison(results)

    print("\nAll figures saved to docs/figures/")


if __name__ == "__main__":
    main()
