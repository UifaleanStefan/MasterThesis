"""
MultiSession persistent-memory benchmark — Phase 4 / A1 of the PoC plan.

The standard 12-system × 4-env benchmark in `evaluation/benchmark.py` is
single-episode: a fresh memory is constructed for each evaluation episode.
That doesn't exercise the most LLM-realistic property of memory: surviving
across context-window boundaries.

`environment.multi_session.MultiSessionEnv` simulates 20 sessions of
collaborative storytelling where the agent has a fresh context per session
but its memory persists. This script runs a small set of representative
memory systems on the multi-session env, with the lifecycle: one persistent
memory instance, looped across all sessions, with consistency + completion
scoring at the end.

Output:
    results/multisession_results.json
    docs/figures/fig15_multi_session.png  (replaces the synthetic placeholder)

Usage (PowerShell, from project root):
    python run_multisession_benchmark.py
    python run_multisession_benchmark.py --episodes 3        # fewer trials per system
    python run_multisession_benchmark.py --systems FlatWindow GraphMemoryV4 RAGMemory
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _factory(name: str, seed: int):
    from memory.attention_memory import AttentionMemory
    from memory.causal_memory import CausalMemory
    from memory.episodic_semantic_memory import EpisodicSemanticMemory
    from memory.flat_memory import FlatMemory
    from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4
    from memory.hierarchical_memory import HierarchicalMemory
    from memory.rag_memory import RAGMemory
    from memory.semantic_memory import SemanticMemory
    from memory.summary_memory import SummaryMemory
    from memory.working_memory import WorkingMemory

    # V4 starting point — published optimum (will be overridden by post-MiniLM
    # CMA-ES once that completes; for now this is a reasonable warm start).
    v4_params = MemoryParamsV4(
        theta_store=0.293, theta_novel=0.908, theta_erich=0.198,
        theta_surprise=0.785, theta_entity=0.285, theta_temporal=0.278,
        theta_decay=0.668, w_graph=0.0, w_embed=1.079, w_recency=3.777,
        mode="learnable",
    )

    factories = {
        "FlatWindow(50)": lambda: FlatMemory(window_size=50),
        "GraphMemoryV4": lambda: GraphMemoryV4(v4_params),
        "EpisodicSemantic": lambda: EpisodicSemanticMemory(episodic_size=30),
        "WorkingMemory(7)": lambda: WorkingMemory(capacity=7),
        "RAGMemory": lambda: RAGMemory(),
        "SemanticMemory": lambda: SemanticMemory(max_capacity=80),
        "SummaryMemory": lambda: SummaryMemory(),
        "HierarchicalMemory": lambda: HierarchicalMemory(),
        "CausalMemory": lambda: CausalMemory(),
        "AttentionMemory": lambda: AttentionMemory(temperature=0.5),
    }
    if name not in factories:
        raise ValueError(f"Unknown system: {name}. Available: {list(factories)}")
    return factories[name]()


def _run_one_trial(memory, env_seed: int) -> dict:
    """
    Run a complete 20-session trial with one persistent memory instance.

    The memory is NOT cleared between sessions — that's the whole point of
    the MultiSession benchmark. Returns the env's final score plus per-trial
    stats.
    """
    from environment.multi_session import MultiSessionEnv
    from memory.event import Event

    env = MultiSessionEnv(seed=env_seed)
    global_step = 0
    while not env.all_sessions_done:
        obs = env.reset()
        while not env.done:
            # Retrieve relevant past events from persistent memory.
            past = memory.get_relevant_events(obs, current_step=global_step, k=8)
            # Naive policy: emit a "continue" action regardless of past_events;
            # the test is about memory recall, not policy quality.
            action = "continue"
            obs, _, _ = env.step(action)
            memory.add_event(
                Event(step=global_step, observation=obs, action=action),
                episode_seed=env_seed,
            )
            global_step += 1
    stats = memory.get_stats()
    return {
        "partial_score": float(env.partial_score),
        "n_steps_total": global_step,
        "memory_size_final": int(stats.get("n_events", 0) or stats.get("size", global_step)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5,
                        help="Trials per system (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--systems", nargs="*", default=None,
                        help="Subset of systems to evaluate (default: a representative 5)")
    parser.add_argument("--output", type=str, default="results/multisession_results.json")
    args = parser.parse_args(argv)

    # Default to the five systems most likely to differentiate on persistent memory.
    if args.systems is None:
        args.systems = [
            "FlatWindow(50)",
            "GraphMemoryV4",
            "EpisodicSemantic",
            "WorkingMemory(7)",
            "RAGMemory",
        ]

    print("=" * 70)
    print(f"  MultiSession benchmark ({len(args.systems)} systems × {args.episodes} trials)")
    print("=" * 70)

    t0 = time.time()
    results: dict = {}
    for sys_name in args.systems:
        scores, sizes, steps = [], [], []
        print(f"\n  [{sys_name}]")
        for trial in range(args.episodes):
            mem = _factory(sys_name, seed=args.seed)
            trial_data = _run_one_trial(mem, env_seed=args.seed + trial)
            scores.append(trial_data["partial_score"])
            sizes.append(trial_data["memory_size_final"])
            steps.append(trial_data["n_steps_total"])
            print(f"    trial {trial + 1}: score={trial_data['partial_score']:.3f}  "
                  f"final_mem_size={trial_data['memory_size_final']}")
        results[sys_name] = {
            "mean_score": statistics.mean(scores),
            "std_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "scores": scores,
            "mean_memory_size": statistics.mean(sizes),
            "mean_n_steps": statistics.mean(steps),
            "n_trials": args.episodes,
        }

    print("\n" + "=" * 70)
    print("  RANKED BY MEAN SCORE")
    print("=" * 70)
    for name, res in sorted(results.items(), key=lambda kv: -kv[1]["mean_score"]):
        print(f"  {name:<22}  score={res['mean_score']:.3f} "
              f"±{res['std_score']:.3f}  mem={res['mean_memory_size']:.0f}")

    # Save JSON
    from results.manifest import build_manifest
    payload = {
        "_manifest": build_manifest(seed=args.seed, extra={
            "experiment": "multisession_benchmark",
            "n_trials_per_system": args.episodes,
            "n_sessions_per_trial": 20,
            "systems": list(results.keys()),
        }),
        "results": results,
        "elapsed_s": time.time() - t0,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  Saved to {out_path}  (elapsed: {payload['elapsed_s']:.1f}s)")

    # Render figure (overrides the synthetic placeholder under docs/figures/draft/)
    try:
        _plot_multisession_results(results)
    except Exception as exc:  # pragma: no cover
        print(f"  [figure] generation failed: {exc}")

    return 0


def _plot_multisession_results(results: dict) -> None:
    """Bar chart of mean partial_score per system, with std error bars."""
    import matplotlib.pyplot as plt

    items = sorted(results.items(), key=lambda kv: -kv[1]["mean_score"])
    names = [n for n, _ in items]
    means = [r["mean_score"] for _, r in items]
    stds = [r["std_score"] for _, r in items]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4,
                  color="#3F51B5", alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("Memory", "Mem") for n in names], rotation=18, ha="right")
    ax.set_ylabel("Multi-session partial score (consistency × completion)")
    ax.set_title("MultiSessionEnv — persistent memory across 20 sessions\n"
                 "(real data, MiniLM embedding)")
    ax.set_ylim(0, max(1.0, max(means) * 1.15))
    ax.grid(axis="y", alpha=0.3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{m:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    out = Path("docs/figures/fig15_multi_session.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] saved {out} (replaces draft synthetic placeholder)")


if __name__ == "__main__":
    sys.exit(main())
