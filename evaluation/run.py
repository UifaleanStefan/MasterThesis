"""Run evaluation: compare no-memory vs graph vs embedding vs hybrid vs learnable."""

from environment import ToyEnvironment
from memory import GraphMemory, MemoryParams
from agent import ExplorationPolicy, run_episode_no_memory, run_episode_with_memory


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
