"""
Cross-task transfer experiments — theta learned on Task A, evaluated on Task B.

This is the generalization experiment: does a theta optimized for one task produce
good memory structure on a different task?

Hypothesis 1 (task-specific): theta_Key-Door on Goal-Room performs worse than
theta_Goal-Room on Goal-Room — confirming that optimal theta is task-specific.

Hypothesis 2 (positive transfer): theta_MultiHop on Key-Door may outperform the
baseline because both tasks require remembering entity-color associations.

Hypothesis 3 (negative transfer): theta_Goal-Room (sparse entity nodes) on MultiHop
should fail badly because MultiHop requires entity nodes to distinguish keys.

Transfer matrix:
  - Rows: theta source task (where it was trained).
  - Columns: evaluation task (where it is tested).
  - Cell: mean_reward on evaluation task using source theta.

Usage:
    matrix = run_transfer_matrix(envs, policies, learned_thetas, n_episodes=50)
    print_transfer_matrix(matrix)
"""

from __future__ import annotations

import statistics
from typing import Any

from memory.graph_memory import GraphMemory, MemoryParams
from agent.loop import run_episode_with_any_memory


def evaluate_theta_on_task(
    theta: tuple[float, float, float],
    env: Any,
    policy: Any,
    n_episodes: int = 50,
    k: int = 8,
    seed_offset: int = 0,
) -> dict:
    """Evaluate a specific theta on a specific environment for n_episodes."""
    rewards: list[float] = []
    tokens: list[float] = []
    precisions: list[float] = []

    for ep in range(n_episodes):
        mem = GraphMemory(MemoryParams(*theta, "learnable"))
        _, _, stats = run_episode_with_any_memory(
            env, policy, mem, k=k, episode_seed=seed_offset + ep
        )
        rewards.append(stats.get("reward", 0.0))
        tokens.append(stats.get("retrieval_tokens", 0))
        prec = stats.get("retrieval_precision")
        if prec is not None:
            precisions.append(prec)

    return {
        "mean_reward": statistics.mean(rewards),
        "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
        "mean_tokens": statistics.mean(tokens),
        "mean_precision": statistics.mean(precisions) if precisions else None,
        "n_episodes": n_episodes,
    }


def run_transfer_matrix(
    envs: dict[str, Any],
    policies: dict[str, Any],
    learned_thetas: dict[str, tuple[float, float, float]],
    n_episodes: int = 50,
    k: int = 8,
    verbose: bool = True,
) -> dict[str, dict[str, dict]]:
    """
    Run full cross-task transfer matrix.

    Parameters
    ----------
    envs : {task_name: env_instance}
    policies : {task_name: policy_instance}
    learned_thetas : {task_name: (store, entity, temporal)}
        Thetas learned on each task (from Phase 7 ES).
    n_episodes : int
        Episodes per (source_theta, target_task) pair.

    Returns
    -------
    matrix : {source_task: {target_task: result_dict}}
        result_dict has keys: mean_reward, std_reward, mean_tokens, mean_precision.
    """
    task_names = list(envs.keys())
    matrix: dict[str, dict[str, dict]] = {}

    for source_task, theta in learned_thetas.items():
        matrix[source_task] = {}
        for target_task in task_names:
            if verbose:
                print(f"  Transfer: theta from '{source_task}' → eval on '{target_task}'")
            result = evaluate_theta_on_task(
                theta=theta,
                env=envs[target_task],
                policy=policies[target_task],
                n_episodes=n_episodes,
                k=k,
                seed_offset=hash(source_task + target_task) % 10000,
            )
            matrix[source_task][target_task] = result
            if verbose:
                print(f"    mean_reward={result['mean_reward']:.4f}")

    return matrix


def print_transfer_matrix(
    matrix: dict[str, dict[str, dict]],
    metric: str = "mean_reward",
) -> None:
    """Print the transfer matrix as a formatted table."""
    source_tasks = list(matrix.keys())
    target_tasks = list(matrix[source_tasks[0]].keys()) if source_tasks else []

    print("\n" + "=" * 80)
    print(f"CROSS-TASK TRANSFER MATRIX (metric: {metric})")
    print("Rows = theta source (trained on), Columns = evaluation task")
    print("=" * 80)

    col_w = 16
    header = f"{'Source→Target':<20}" + "".join(f"{t[:col_w]:>{col_w}}" for t in target_tasks)
    print(header)
    print("-" * 80)

    for source in source_tasks:
        row = f"{source:<20}"
        for target in target_tasks:
            val = matrix[source][target].get(metric, 0.0)
            marker = " *" if source == target else ""
            row += f"{val:>{col_w-2}.4f}{marker}"
        print(row)

    print("=" * 80)
    print("* = in-distribution (theta trained and evaluated on same task type)")
    print("\nInterpretation:")
    print("  - Diagonal > off-diagonal: task-specific theta (thesis claim confirmed)")
    print("  - Off-diagonal > baseline: positive transfer exists")
    print("  - Off-diagonal << baseline: negative transfer (incompatible memory structures)")


def compute_transfer_summary(matrix: dict, baselines: dict[str, float]) -> dict:
    """
    Compute transfer summary statistics:
      - mean in-distribution performance (diagonal)
      - mean out-of-distribution performance (off-diagonal)
      - mean degradation from in-distribution to OOD
    """
    source_tasks = list(matrix.keys())
    target_tasks = list(list(matrix.values())[0].keys()) if matrix else []

    in_dist = []
    out_dist = []
    for source in source_tasks:
        for target in target_tasks:
            val = matrix[source][target]["mean_reward"]
            if source == target:
                in_dist.append(val)
            else:
                out_dist.append(val)

    baseline_vals = list(baselines.values())
    return {
        "mean_in_distribution": statistics.mean(in_dist) if in_dist else 0.0,
        "mean_out_of_distribution": statistics.mean(out_dist) if out_dist else 0.0,
        "mean_baseline": statistics.mean(baseline_vals) if baseline_vals else 0.0,
        "transfer_degradation": (statistics.mean(in_dist) - statistics.mean(out_dist)) / max(1e-9, statistics.mean(in_dist)) if in_dist else 0.0,
    }
