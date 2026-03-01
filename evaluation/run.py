"""Evaluation utilities: per-retrieval-mode and per-memory-system comparisons."""

from typing import Any

from agent import ExplorationPolicy, run_episode_no_memory, run_episode_with_any_memory, run_episode_with_memory
from environment import ToyEnvironment
from memory import GraphMemory, MemoryParams


def run_evaluation(
    n_episodes: int = 50,
    env_seed: int | None = None,
    policy_seed: int | None = 42,
    learnable_configs: list[tuple[float, float, float]] | None = None,
    memory_params: MemoryParams | None = None,
) -> dict:
    """
    Run evaluation over multiple episodes.
    Returns dict with success rates, avg steps, and (when using memory) mean retrieval_tokens, mean memory_size.
    If memory_params is provided, GraphMemory is created with it (learnable memory creation).
    """
    env = ToyEnvironment(seed=env_seed)
    policy = ExplorationPolicy(seed=policy_seed)
    memory = GraphMemory(params=memory_params) if memory_params is not None else GraphMemory()
    base_seed = env_seed if env_seed is not None else 0

    def run_baseline(
        name: str,
        retrieval_mode: str | None,
        learnable_weights: tuple[float, float, float] | None = None,
    ) -> dict:
        successes = 0
        steps: list[int] = []
        results: list[bool] = []
        retrieval_tokens_list: list[int] = []
        memory_size_list: list[int] = []
        for ep_idx in range(n_episodes):
            episode_seed = base_seed + ep_idx
            if retrieval_mode is None:
                success, events = run_episode_no_memory(env, policy)
            else:
                success, events, stats_dict = run_episode_with_memory(
                    env,
                    policy,
                    memory,
                    retrieval_mode=retrieval_mode,
                    learnable_weights=learnable_weights,
                    episode_seed=episode_seed,
                )
                retrieval_tokens_list.append(stats_dict["retrieval_tokens"])
                memory_size_list.append(stats_dict["memory_size"])
            successes += int(success)
            steps.append(len(events))
            results.append(success)
        out: dict = {
            "success_rate": successes / n_episodes,
            "successes": successes,
            "avg_steps": sum(steps) / n_episodes if steps else 0,
            "steps_success": [s for s, r in zip(steps, results) if r],
            "results": results,
        }
        if retrieval_tokens_list:
            out["mean_retrieval_tokens"] = sum(retrieval_tokens_list) / len(retrieval_tokens_list)
            out["mean_memory_size"] = sum(memory_size_list) / len(memory_size_list)
        return out

    result_no_memory = run_baseline("no_memory", None)
    result_no_memory["mean_retrieval_tokens"] = 0.0
    result_no_memory["mean_memory_size"] = 0

    out: dict = {
        "n_episodes": n_episodes,
        "no_memory": result_no_memory,
        "graph": run_baseline("graph", "graph"),
        "embedding": run_baseline("embedding", "embedding"),
        "hybrid": run_baseline("hybrid", "hybrid"),
    }

    if learnable_configs:
        out["learnable"] = {}
        for weights in learnable_configs:
            w_g, w_e, w_r = weights
            out["learnable"][weights] = run_baseline(
                "learnable",
                "learnable",
                learnable_weights=(w_g, w_e, w_r),
            )

    return out


def run_memory_comparison(
    env: Any,
    policy: ExplorationPolicy,
    memory_systems: dict[str, Any],
    n_episodes: int = 30,
    k: int = 8,
    env_seed: int | None = None,
    lambda_penalty: float = 0.001,
) -> dict[str, dict]:
    """
    Run each memory system in memory_systems for n_episodes on env.
    Returns a dict: system_name -> result_dict with:
        success_rate, mean_partial_score, mean_retrieval_tokens,
        mean_memory_size, efficiency, mean_j.

    Each memory system must expose the uniform interface:
        add_event(event, episode_seed=None)
        get_relevant_events(observation, current_step, k) -> list[Event]
        clear()
        get_stats() -> dict
    """
    base_seed = env_seed if env_seed is not None else 0
    results: dict[str, dict] = {}

    for name, memory in memory_systems.items():
        successes = 0
        partial_scores: list[float] = []
        retrieval_tokens_list: list[int] = []
        memory_size_list: list[int] = []
        j_list: list[float] = []

        for ep_idx in range(n_episodes):
            episode_seed = base_seed + ep_idx
            success, events, stats_dict = run_episode_with_any_memory(
                env,
                policy,
                memory,
                k=k,
                episode_seed=episode_seed,
            )
            successes += int(success)
            reward = stats_dict["reward"]
            partial_scores.append(reward)
            retrieval_tokens_list.append(stats_dict["retrieval_tokens"])
            memory_size_list.append(stats_dict["memory_size"])
            j = reward - lambda_penalty * stats_dict["retrieval_tokens"]
            j_list.append(j)

        mean_partial = sum(partial_scores) / n_episodes
        mean_tokens = sum(retrieval_tokens_list) / n_episodes
        mean_size = sum(memory_size_list) / n_episodes
        mean_j = sum(j_list) / n_episodes
        efficiency = mean_partial / (1.0 + mean_tokens)

        results[name] = {
            "success_rate": successes / n_episodes,
            "successes": successes,
            "mean_partial_score": mean_partial,
            "mean_retrieval_tokens": mean_tokens,
            "mean_memory_size": mean_size,
            "efficiency": efficiency,
            "mean_j": mean_j,
        }

    return results
