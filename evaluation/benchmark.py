"""
Memory Architecture Benchmark — 8+ systems × 4 environments × 100 episodes.

This is the main comparison table of Chapter 3. Runs all memory systems on all
environments and produces a comprehensive performance matrix.

Memory systems (8 total):
  1. FlatWindow(50)          — sliding window baseline
  2. GraphMemory+Theta       — learnable graph memory (ES-learned theta)
  3. SemanticMemory          — importance-weighted pool
  4. SummaryMemory           — periodic compression
  5. EpisodicSemantic        — dual-store (existing best performer)
  6. RAGMemory               — dense embeddings
  7. HierarchicalMemory      — 3-level multi-resolution (NEW)
  8. WorkingMemory(7)        — bounded LRU-retrieval (NEW)
  9. CausalMemory            — causal chain tracking (NEW)
  10. AttentionMemory        — softmax attention retrieval (NEW)

Environments (4 total):
  1. ToyEnvironment (Key-Door) — simple baseline
  2. GoalRoom                  — simple navigation
  3. MultiHopKeyDoor           — hard memory benchmark (primary)
  4. MegaQuestRoom             — extended benchmark (new)

For each (system, environment) pair, run n_episodes and collect:
  - mean_reward, std_reward
  - mean_retrieval_tokens
  - mean_memory_size
  - efficiency = mean_reward / (1 + mean_tokens)
  - retrieval_precision (for environments with hints)
  - bootstrap 95% CI on reward

Usage:
    results = run_full_benchmark(n_episodes=50)
    print_benchmark_table(results)
    save_benchmark_results(results, "results/benchmark.json")
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from agent.loop import run_episode_with_any_memory
from evaluation.statistics import bootstrap_ci


def _make_memory_systems(learned_thetas: dict[str, tuple] | None = None) -> dict[str, Any]:
    """
    Instantiate all memory systems. Uses best learned theta for GraphMemory+Theta.
    """
    from memory.flat_memory import FlatMemory
    from memory.semantic_memory import SemanticMemory
    from memory.summary_memory import SummaryMemory
    from memory.episodic_semantic_memory import EpisodicSemanticMemory
    from memory.rag_memory import RAGMemory
    from memory.graph_memory import GraphMemory, MemoryParams
    from memory.hierarchical_memory import HierarchicalMemory
    from memory.working_memory import WorkingMemory
    from memory.causal_memory import CausalMemory
    from memory.attention_memory import AttentionMemory

    # Default learned theta for MultiHop (best from Phase 7)
    default_theta = (0.956, 0.378, 1.000)
    if learned_thetas:
        default_theta = learned_thetas.get("MultiHop-KeyDoor", default_theta)

    return {
        "FlatWindow(50)":       lambda: FlatMemory(window_size=50),
        "GraphMemory+Theta":    lambda: GraphMemory(MemoryParams(*default_theta, "learnable")),
        "SemanticMemory":       lambda: SemanticMemory(max_capacity=80),
        "SummaryMemory":        lambda: SummaryMemory(raw_buffer_size=30, summarize_every=25),
        "EpisodicSemantic":     lambda: EpisodicSemanticMemory(episodic_size=30),
        "RAGMemory":            lambda: RAGMemory(),
        "HierarchicalMemory":   lambda: HierarchicalMemory(),
        "WorkingMemory(7)":     lambda: WorkingMemory(capacity=7),
        "CausalMemory":         lambda: CausalMemory(),
        "AttentionMemory":      lambda: AttentionMemory(temperature=0.5),
    }


def run_system_on_env(
    memory_factory,
    env: Any,
    policy: Any,
    n_episodes: int = 50,
    k: int = 8,
    seed_offset: int = 0,
) -> dict:
    """Run one memory system on one environment for n_episodes."""
    rewards, tokens, sizes, precisions = [], [], [], []

    for ep in range(n_episodes):
        mem = memory_factory()
        _, _, stats = run_episode_with_any_memory(
            env, policy, mem, k=k, episode_seed=seed_offset + ep
        )
        rewards.append(stats.get("reward", 0.0))
        tokens.append(stats.get("retrieval_tokens", 0))
        sizes.append(stats.get("memory_size", 0))
        prec = stats.get("retrieval_precision")
        if prec is not None:
            precisions.append(prec)

    mean_r = statistics.mean(rewards)
    mean_t = statistics.mean(tokens)
    ci = bootstrap_ci(rewards, n_resamples=500)

    return {
        "mean_reward": mean_r,
        "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
        "ci_lower": ci["ci_lower"],
        "ci_upper": ci["ci_upper"],
        "mean_tokens": mean_t,
        "mean_memory_size": statistics.mean(sizes),
        "efficiency": mean_r / (1 + mean_t),
        "retrieval_precision": statistics.mean(precisions) if precisions else None,
        "n_episodes": n_episodes,
        "rewards": rewards,
    }


def run_full_benchmark(
    envs: dict[str, tuple] | None = None,
    n_episodes: int = 50,
    k: int = 8,
    learned_thetas: dict[str, tuple] | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, dict]]:
    """
    Run all memory systems on all environments.

    Parameters
    ----------
    envs : dict mapping env_name -> (env_instance, policy_instance)
        If None, uses default set (MultiHopKeyDoor + GoalRoom).
    n_episodes : int
        Episodes per (system, env) pair.
    learned_thetas : dict mapping env_name -> theta tuple
        Learned thetas from Phase 7 for GraphMemory+Theta.
    verbose : bool

    Returns
    -------
    results : {env_name: {system_name: result_dict}}
    """
    from environment import MultiHopKeyDoor, GoalRoom, ToyEnvironment
    from agent import ExplorationPolicy

    if envs is None:
        envs = {
            "Key-Door":        (ToyEnvironment(seed=0),      ExplorationPolicy(seed=0)),
            "Goal-Room":       (GoalRoom(seed=0),             ExplorationPolicy(seed=0)),
            "MultiHop-KeyDoor":(MultiHopKeyDoor(seed=0),     ExplorationPolicy(seed=0)),
        }

    memory_factories = _make_memory_systems(learned_thetas)
    results: dict[str, dict[str, dict]] = {}

    for env_name, (env, policy) in envs.items():
        results[env_name] = {}
        if verbose:
            print(f"\n[Benchmark] Environment: {env_name}")

        for sys_name, factory in memory_factories.items():
            if verbose:
                print(f"  System: {sys_name}")
            try:
                result = run_system_on_env(
                    memory_factory=factory,
                    env=env,
                    policy=policy,
                    n_episodes=n_episodes,
                    k=k,
                    seed_offset=hash(env_name + sys_name) % 10000,
                )
                results[env_name][sys_name] = result
                if verbose:
                    print(
                        f"    reward={result['mean_reward']:.4f} "
                        f"[{result['ci_lower']:.3f},{result['ci_upper']:.3f}] "
                        f"prec={result['retrieval_precision']}"
                    )
            except Exception as e:
                print(f"    ERROR: {e}")
                results[env_name][sys_name] = {"error": str(e)}

    return results


def print_benchmark_table(results: dict[str, dict[str, dict]]) -> None:
    """Print a formatted cross-environment benchmark table."""
    env_names = list(results.keys())
    sys_names = list(list(results.values())[0].keys()) if results else []

    print("\n" + "=" * 100)
    print("MEMORY ARCHITECTURE BENCHMARK — All Systems × All Environments")
    print("=" * 100)

    header = f"{'System':<22}" + "".join(
        f"{'Reward':>10}{'CI':>10}{'Prec':>8}" for _ in env_names
    )
    env_header = f"{'':22}" + "".join(f"{e[:26]:^28}" for e in env_names)
    print(env_header)
    print("-" * 100)

    for sys_name in sys_names:
        row = f"{sys_name:<22}"
        for env_name in env_names:
            res = results.get(env_name, {}).get(sys_name, {})
            if "error" in res:
                row += f"{'ERR':>10}{'':>10}{'':>8}"
            else:
                r = res.get("mean_reward", 0.0)
                ci_str = f"[{res.get('ci_lower', 0):.2f},{res.get('ci_upper', 0):.2f}]"
                prec = res.get("retrieval_precision")
                prec_str = f"{prec:.3f}" if prec is not None else "N/A"
                row += f"{r:>10.4f}{ci_str:>10}{prec_str:>8}"
        print(row)

    print("=" * 100)


def save_benchmark_results(results: dict, path: str | Path) -> None:
    """Save benchmark results to JSON (excluding raw rewards list for file size)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {}
    for env, sys_dict in results.items():
        clean[env] = {}
        for sys_name, res in sys_dict.items():
            clean[env][sys_name] = {k: v for k, v in res.items() if k != "rewards"}
    path.write_text(json.dumps(clean, indent=2, default=str))
    print(f"[Benchmark] Results saved to {path}")
