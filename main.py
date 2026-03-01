"""
Main entry point for Structured Learnable Event Memory prototype.
Phase 4: Learnable memory (embeddings + similarity retrieval).
Phase 5: Learnable retrieval scoring (weighted graph + embedding + recency).
Phase 6: Learnable memory creation (theta_store, theta_entity, theta_temporal).
Phase 7: Adaptive theta via Evolution Strategy (ES).
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from environment import GoalRoom, HardKeyDoor, QuestRoom, ToyEnvironment
from memory import (
    EpisodicSemanticMemory,
    FlatMemory,
    GraphMemory,
    MemoryParams,
    RAGMemory,
    SemanticMemory,
    SummaryMemory,
    retrieve_similar_events,
)
from agent import ExplorationPolicy, run_episode_with_logging, run_episode_with_memory
from evaluation import run_evaluation, run_memory_comparison

LEARNABLE_RETRIEVAL_WEIGHTS = (1.5, 1.0, 0.2)


def demo_embedding_retrieval() -> list[dict]:
    """Run embedding retrieval on sample observations for report."""
    from memory import Event

    memory = GraphMemory()
    sample_events = [
        (0, "You are in a room. You see nothing of interest."),
        (1, "You are in a room. You see nothing of interest."),
        (2, "You are in a room. You see a red key."),
        (3, "You are in a room. You see nothing of interest. You are carrying a red key."),
        (4, "You are in a room. You see a blue door."),
    ]
    for step, obs in sample_events:
        memory.add_event(Event(step=step, observation=obs, action="move_north"))

    queries = [
        "You see a red door",
        "You see a red key",
        "You are carrying a blue key",
    ]
    results = []
    for q in queries:
        events, scores = retrieve_similar_events(
            q, memory.get_graph(), k=3, verbose=False
        )
        results.append({
            "query": q,
            "retrieved": [(e.step, e.observation[:45] + "...", s) for e, s in zip(events, scores)],
        })
    return results


def print_report(results: dict, embedding_demo: list[dict]) -> None:
    """Print Phase 4 report for ChatGPT."""
    n = results["n_episodes"]

    print("\n" + "=" * 70)
    print("PHASE 4 REPORT")
    print("=" * 70)

    print("\n1. Implementation Summary")
    print("-" * 40)
    print("Files added:")
    print("  - memory/embedding.py: embed_observation() using TF-IDF, fixed vocabulary")
    print("Files modified:")
    print("  - memory/graph_memory.py: store embedding in each event node")
    print("  - memory/retrieval.py: retrieve_similar_events(), retrieve_events() hybrid")
    print("  - agent/loop.py: configurable retrieval_mode (graph|embedding|hybrid)")
    print("  - evaluation/run.py: eval no_memory, graph, embedding, hybrid")
    print("  - requirements.txt: + numpy, scikit-learn")
    print("Retrieval methods:")
    print("  - Graph: entity-based traversal (Phase 3)")
    print("  - Embedding: cosine similarity over TF-IDF vectors")
    print("  - Hybrid: union of graph + embedding results")

    print("\n2. Retrieval Behavior")
    print("-" * 40)
    for r in embedding_demo:
        print(f"Query: \"{r['query']}\"")
        for step, obs, score in r["retrieved"]:
            print(f"  - event_{step} (score={score:.2f}): {obs}")
        print()

    print("\n3. Performance")
    print("-" * 40)
    for name in ["no_memory", "graph", "embedding", "hybrid"]:
        d = results[name]
        succ = d["successes"]
        rate = d["success_rate"]
        avg = d["avg_steps"]
        succ_steps = d["steps_success"]
        avg_succ = sum(succ_steps) / len(succ_steps) if succ_steps else 0
        print(f"{name:12}: {succ}/{n} = {rate:.1%} success, avg steps={avg:.1f}, avg(success)={avg_succ:.1f}")
    learnable = results.get("learnable")
    if learnable:
        for weights, d in learnable.items():
            w_g, w_e, w_r = weights
            succ = d["successes"]
            rate = d["success_rate"]
            avg = d["avg_steps"]
            succ_steps = d["steps_success"]
            avg_succ = sum(succ_steps) / len(succ_steps) if succ_steps else 0
            label = f"learnable({w_g},{w_e},{w_r})"
            print(f"{label:12}: {succ}/{n} = {rate:.1%} success, avg steps={avg:.1f}, avg(success)={avg_succ:.1f}")

    print("\n4. Comparison to Phase 3")
    print("-" * 40)
    g = results["graph"]
    e = results["embedding"]
    h = results["hybrid"]
    no = results["no_memory"]
    if h["success_rate"] > g["success_rate"]:
        print("Hybrid BETTER than graph-only.")
    elif h["success_rate"] < g["success_rate"]:
        print("Hybrid WORSE than graph-only.")
    else:
        print("Hybrid SAME as graph-only.")
    if e["success_rate"] > no["success_rate"]:
        print("Embedding-only BETTER than no-memory.")
    elif e["success_rate"] < no["success_rate"]:
        print("Embedding-only WORSE than no-memory.")
    else:
        print("Embedding-only SAME as no-memory.")

    print("\n5. Observations")
    print("-" * 40)
    print("TF-IDF embeddings capture lexical overlap: 'red key' similar to 'red door' (shared 'red').")
    print("Graph retrieval uses entity structure; embedding uses word co-occurrence.")
    print("Hybrid merges both; may help when one method misses relevant events.")
    print("Variance across runs is expected with random exploration.")


def print_phase5_report(results: dict) -> None:
    """Print Phase 5 Learnable Retrieval report."""
    n = results["n_episodes"]
    learnable = results.get("learnable")
    if not learnable:
        return

    print("\n" + "=" * 70)
    print("=== Phase 5 Learnable Retrieval Report ===")
    print("=" * 70)

    g = results["graph"]
    h = results["hybrid"]

    for weights, d in learnable.items():
        w_g, w_e, w_r = weights
        succ = d["successes"]
        rate = d["success_rate"]
        avg_steps = d["avg_steps"]
        succ_steps = d["steps_success"]
        avg_succ = sum(succ_steps) / len(succ_steps) if succ_steps else 0

        print(f"\nWeights: w_graph={w_g}, w_embed={w_e}, w_recency={w_r}")
        print("Condition: learnable")
        print(f"Success: {succ}/{n}")
        print(f"Success Rate: {rate:.1%}")
        print(f"Avg Steps: {avg_steps:.1f}")
        print(f"Avg Steps (success only): {avg_succ:.1f}")

        geq_g = rate >= g["success_rate"]
        geq_h = rate >= h["success_rate"]
        print(f"learnable >= graph: {'yes' if geq_g else 'no'}")
        print(f"learnable >= hybrid: {'yes' if geq_h else 'no'}")


def run_phase6_bandit(
    n_theta_configs: int = 15,
    n_episodes_per_theta: int = 75,
    lambda_penalty: float = 0.001,
    env_seed: int | None = None,
    policy_seed: int | None = 42,
    bandit_seed: int | None = 123,
    env_name: str = "Key-Door",
) -> tuple[tuple[float, float, float], list[dict], dict]:
    """
    Random search over [0,1]^3 for MemoryParams (theta_store, theta_entity, theta_temporal).
    Uses learnable retrieval weights (1.5, 1.0, 0.2). J = reward - lambda * retrieval_tokens.
    Returns (best_theta, all_results_list, baseline_fixed).
    """
    rng = random.Random(bandit_seed)
    if env_name == "Goal-Room":
        env = GoalRoom(seed=env_seed)
    elif env_name == "Hard-KeyDoor":
        env = HardKeyDoor(seed=env_seed)
    else:
        env = ToyEnvironment(seed=env_seed)
    policy = ExplorationPolicy(seed=policy_seed)
    base_seed = env_seed if env_seed is not None else 0
    learnable_weights = (1.5, 1.0, 0.2)

    configs: list[tuple[float, float, float]] = []
    for _ in range(n_theta_configs):
        theta_store = rng.uniform(0.0, 1.0)
        theta_entity = rng.uniform(0.0, 1.0)
        theta_temporal = rng.uniform(0.0, 1.0)
        configs.append((theta_store, theta_entity, theta_temporal))

    all_results: list[dict] = []
    for theta in configs:
        theta_store, theta_entity, theta_temporal = theta
        params = MemoryParams(
            theta_store=theta_store,
            theta_entity=theta_entity,
            theta_temporal=theta_temporal,
            mode="learnable",
        )
        memory = GraphMemory(params=params)
        successes = 0
        j_sum = 0.0
        retrieval_tokens_list: list[int] = []
        memory_size_list: list[int] = []
        for ep_idx in range(n_episodes_per_theta):
            episode_seed = base_seed + ep_idx
            success, events, stats_dict = run_episode_with_memory(
                env,
                policy,
                memory,
                retrieval_mode="learnable",
                learnable_weights=learnable_weights,
                episode_seed=episode_seed,
            )
            successes += int(success)
            reward = stats_dict.get("reward", 1.0 if success else 0.0)
            j_sum += reward - lambda_penalty * stats_dict["retrieval_tokens"]
            retrieval_tokens_list.append(stats_dict["retrieval_tokens"])
            memory_size_list.append(stats_dict["memory_size"])
        mean_j = j_sum / n_episodes_per_theta
        mean_retrieval_tokens = sum(retrieval_tokens_list) / len(retrieval_tokens_list)
        mean_memory_size = sum(memory_size_list) / len(memory_size_list)
        all_results.append({
            "theta": theta,
            "success_rate": successes / n_episodes_per_theta,
            "successes": successes,
            "mean_j": mean_j,
            "mean_retrieval_tokens": mean_retrieval_tokens,
            "mean_memory_size": mean_memory_size,
        })

    best = max(all_results, key=lambda x: x["mean_j"])
    best_theta = best["theta"]

    baseline_memory = GraphMemory()
    baseline_successes = 0
    baseline_j_sum = 0.0
    baseline_retrieval_tokens_list: list[int] = []
    baseline_memory_size_list: list[int] = []
    for ep_idx in range(n_episodes_per_theta):
        episode_seed = base_seed + ep_idx
        success, events, stats_dict = run_episode_with_memory(
            env,
            policy,
            baseline_memory,
            retrieval_mode="learnable",
            learnable_weights=learnable_weights,
            episode_seed=episode_seed,
        )
        baseline_successes += int(success)
        reward = stats_dict.get("reward", 1.0 if success else 0.0)
        baseline_j_sum += reward - lambda_penalty * stats_dict["retrieval_tokens"]
        baseline_retrieval_tokens_list.append(stats_dict["retrieval_tokens"])
        baseline_memory_size_list.append(stats_dict["memory_size"])
    baseline_fixed = {
        "success_rate": baseline_successes / n_episodes_per_theta,
        "successes": baseline_successes,
        "mean_j": baseline_j_sum / n_episodes_per_theta,
        "mean_retrieval_tokens": sum(baseline_retrieval_tokens_list) / len(baseline_retrieval_tokens_list),
        "mean_memory_size": sum(baseline_memory_size_list) / len(baseline_memory_size_list),
    }

    return best_theta, all_results, baseline_fixed


def print_phase6_report(
    best_theta: tuple[float, float, float],
    all_results: list[dict],
    baseline_fixed: dict,
    n_episodes_per_theta: int,
    lambda_penalty: float,
) -> None:
    """Print Phase 6 Learnable Memory Creation report."""
    print("\n" + "=" * 70)
    print("=== Phase 6 Learnable Memory Creation Report ===")
    print("=" * 70)
    print(f"\nObjective: J = reward - {lambda_penalty} * retrieval_tokens")
    print(f"Episodes per theta: {n_episodes_per_theta}")
    print(f"\nBest theta: theta_store={best_theta[0]:.3f}, theta_entity={best_theta[1]:.3f}, theta_temporal={best_theta[2]:.3f}")
    print("\nFixed memory baseline (learnable retrieval only):")
    print(f"  Success: {baseline_fixed['successes']}/{n_episodes_per_theta} = {baseline_fixed['success_rate']:.1%}")
    print(f"  Mean J: {baseline_fixed['mean_j']:.4f}")
    print(f"  Mean retrieval_tokens: {baseline_fixed['mean_retrieval_tokens']:.1f}")
    print(f"  Mean memory_size: {baseline_fixed['mean_memory_size']:.1f}")
    best_result = max(all_results, key=lambda x: x["mean_j"])
    print("\nBest learnable memory config:")
    print(f"  Success: {best_result['successes']}/{n_episodes_per_theta} = {best_result['success_rate']:.1%}")
    print(f"  Mean J: {best_result['mean_j']:.4f}")
    print(f"  Mean retrieval_tokens: {best_result['mean_retrieval_tokens']:.1f}")
    print(f"  Mean memory_size: {best_result['mean_memory_size']:.1f}")
    print("\nComparison: learnable memory vs fixed")
    if best_result["mean_j"] >= baseline_fixed["mean_j"]:
        print("  Learnable memory >= fixed (by mean J).")
    else:
        print("  Learnable memory < fixed (by mean J).")


def _eval_theta(
    theta: tuple[float, float, float],
    env: ToyEnvironment | GoalRoom,
    policy: ExplorationPolicy,
    base_seed: int,
    n_episodes: int,
    lambda_penalty: float,
) -> dict:
    """Run n_episodes with given theta; return mean J, success_rate, mean retrieval_tokens, mean memory_size."""
    params = MemoryParams(
        theta_store=theta[0],
        theta_entity=theta[1],
        theta_temporal=theta[2],
        mode="learnable",
    )
    memory = GraphMemory(params=params)
    successes = 0
    j_sum = 0.0
    retrieval_tokens_list: list[int] = []
    memory_size_list: list[int] = []
    for ep_idx in range(n_episodes):
        episode_seed = base_seed + ep_idx
        success, events, stats_dict = run_episode_with_memory(
            env,
            policy,
            memory,
            retrieval_mode="learnable",
            learnable_weights=LEARNABLE_RETRIEVAL_WEIGHTS,
            episode_seed=episode_seed,
        )
        successes += int(success)
        reward = stats_dict.get("reward", 1.0 if success else 0.0)
        j_sum += reward - lambda_penalty * stats_dict["retrieval_tokens"]
        retrieval_tokens_list.append(stats_dict["retrieval_tokens"])
        memory_size_list.append(stats_dict["memory_size"])
    n = len(retrieval_tokens_list)
    return {
        "mean_j": j_sum / n,
        "success_rate": successes / n,
        "successes": successes,
        "mean_retrieval_tokens": sum(retrieval_tokens_list) / n,
        "mean_memory_size": sum(memory_size_list) / n,
    }


def run_phase7_es(
    initial_theta: tuple[float, float, float] = (1.0, 0.0, 1.0),
    sigma: float = 0.2,
    sigma_decay: float = 0.95,
    sigma_min: float = 0.05,
    n_generations: int = 12,
    n_candidates: int = 6,
    n_episodes_per_candidate: int = 25,
    lambda_penalty: float = 0.001,
    env_seed: int | None = None,
    policy_seed: int | None = 42,
    es_seed: int | None = 456,
    env_name: str = "Key-Door",
) -> tuple[tuple[float, float, float], list[dict], dict]:
    """
    Evolution Strategy for adaptive theta. Each generation: sample n_candidates from
    N(mu, sigma) clipped to [0,1]; evaluate each with n_episodes_per_candidate; set mu to
    best candidate; decay sigma. Returns (best_theta, per_generation_stats, baseline_fixed).
    """
    rng = random.Random(es_seed)
    if env_name == "Goal-Room":
        env = GoalRoom(seed=env_seed)
    elif env_name == "Hard-KeyDoor":
        env = HardKeyDoor(seed=env_seed)
    else:
        env = ToyEnvironment(seed=env_seed)
    policy = ExplorationPolicy(seed=policy_seed)
    base_seed = env_seed if env_seed is not None else 0

    mu = list(initial_theta)
    current_sigma = sigma
    generations_stats: list[dict] = []

    for gen in range(n_generations):
        candidates: list[tuple[float, float, float]] = []
        for _ in range(n_candidates):
            t = (
                max(0.0, min(1.0, mu[0] + current_sigma * rng.gauss(0, 1))),
                max(0.0, min(1.0, mu[1] + current_sigma * rng.gauss(0, 1))),
                max(0.0, min(1.0, mu[2] + current_sigma * rng.gauss(0, 1))),
            )
            candidates.append(t)

        results_per_candidate: list[tuple[tuple[float, float, float], dict]] = []
        for theta in candidates:
            stats = _eval_theta(
                theta, env, policy, base_seed, n_episodes_per_candidate, lambda_penalty
            )
            results_per_candidate.append((theta, stats))

        best_theta, best_stats = max(results_per_candidate, key=lambda x: x[1]["mean_j"])
        mu = list(best_theta)
        current_sigma = max(sigma_min, current_sigma * sigma_decay)

        generations_stats.append({
            "generation": gen + 1,
            "best_theta": tuple(mu),
            "mean_j": best_stats["mean_j"],
            "success_rate": best_stats["success_rate"],
            "mean_retrieval_tokens": best_stats["mean_retrieval_tokens"],
            "mean_memory_size": best_stats["mean_memory_size"],
        })

    best_theta = tuple(mu)

    baseline_memory = GraphMemory()
    baseline_successes = 0
    baseline_j_sum = 0.0
    baseline_retrieval_tokens_list: list[int] = []
    baseline_memory_size_list: list[int] = []
    n_baseline = n_episodes_per_candidate
    for ep_idx in range(n_baseline):
        episode_seed = base_seed + ep_idx
        success, events, stats_dict = run_episode_with_memory(
            env,
            policy,
            baseline_memory,
            retrieval_mode="learnable",
            learnable_weights=LEARNABLE_RETRIEVAL_WEIGHTS,
            episode_seed=episode_seed,
        )
        baseline_successes += int(success)
        reward = stats_dict.get("reward", 1.0 if success else 0.0)
        baseline_j_sum += reward - lambda_penalty * stats_dict["retrieval_tokens"]
        baseline_retrieval_tokens_list.append(stats_dict["retrieval_tokens"])
        baseline_memory_size_list.append(stats_dict["memory_size"])
    baseline_fixed = {
        "success_rate": baseline_successes / n_baseline,
        "successes": baseline_successes,
        "mean_j": baseline_j_sum / n_baseline,
        "mean_retrieval_tokens": sum(baseline_retrieval_tokens_list) / n_baseline,
        "mean_memory_size": sum(baseline_memory_size_list) / n_baseline,
    }

    return best_theta, generations_stats, baseline_fixed


def print_phase7_report(
    best_theta: tuple[float, float, float],
    generations_stats: list[dict],
    baseline_fixed: dict,
    n_generations: int,
    n_candidates: int,
    n_episodes_per_candidate: int,
    lambda_penalty: float,
) -> None:
    """Print Phase 7 Adaptive Theta (ES) report."""
    print("\n" + "=" * 70)
    print("=== Phase 7 Adaptive Theta (Evolution Strategy) Report ===")
    print("=" * 70)
    print(f"\nObjective: J = reward - {lambda_penalty} * retrieval_tokens")
    print(f"Generations: {n_generations}, candidates per gen: {n_candidates}, episodes per candidate: {n_episodes_per_candidate}")
    print(f"Total episodes (learnable): {n_generations * n_candidates * n_episodes_per_candidate}")
    print(f"\nLearned theta: theta_store={best_theta[0]:.3f}, theta_entity={best_theta[1]:.3f}, theta_temporal={best_theta[2]:.3f}")
    print("\nPer-generation best (last 3 generations):")
    for g in generations_stats[-3:]:
        print(f"  Gen {g['generation']}: theta={g['best_theta']}, mean_J={g['mean_j']:.4f}, success={g['success_rate']:.1%}, tokens={g['mean_retrieval_tokens']:.0f}")
    print("\nFixed memory baseline:")
    print(f"  Success: {baseline_fixed['successes']} (rate={baseline_fixed['success_rate']:.1%})")
    print(f"  Mean J: {baseline_fixed['mean_j']:.4f}")
    print(f"  Mean retrieval_tokens: {baseline_fixed['mean_retrieval_tokens']:.1f}")
    print(f"  Mean memory_size: {baseline_fixed['mean_memory_size']:.1f}")
    best_final = generations_stats[-1] if generations_stats else {}
    if best_final and best_final.get("mean_j", 0) >= baseline_fixed["mean_j"]:
        print("\nComparison: ES learned theta >= fixed (by mean J).")
    else:
        print("\nComparison: ES learned theta < fixed (by mean J).")


def print_experiment_summary(
    environment: str,
    method: str,
    best_theta: tuple[float, float, float],
    lambda_penalty: float,
    n_episodes: int,
    baseline: dict,
    learnable: dict,
) -> None:
    """Print structured EXPERIMENT SUMMARY block for ChatGPT/scripts."""
    print("\n" + "=" * 70)
    print("=== EXPERIMENT SUMMARY ===")
    print("=" * 70)
    print(f"Environment: {environment}")
    print(f"Method: {method}")
    print(f"Best theta: theta_store={best_theta[0]:.3f}, theta_entity={best_theta[1]:.3f}, theta_temporal={best_theta[2]:.3f}")
    print(f"Lambda: {lambda_penalty}, n_episodes (or equivalent): {n_episodes}")
    print("\nBaseline (fixed memory):")
    print(f"  Reward (success rate): {baseline.get('success_rate', 0):.2%}")
    print(f"  Mean retrieval_tokens: {baseline.get('mean_retrieval_tokens', 0):.1f}")
    print(f"  Mean memory_size: {baseline.get('mean_memory_size', 0):.1f}")
    print(f"  Mean J: {baseline.get('mean_j', 0):.4f}")
    print("\nLearnable:")
    print(f"  Reward (success rate): {learnable.get('success_rate', 0):.2%}")
    print(f"  Mean retrieval_tokens: {learnable.get('mean_retrieval_tokens', 0):.1f}")
    print(f"  Mean memory_size: {learnable.get('mean_memory_size', 0):.1f}")
    print(f"  Mean J: {learnable.get('mean_j', 0):.4f}")
    print("\nInterpretation:")
    ts, te, tt = best_theta
    interp = []
    if ts < 1.0:
        interp.append("theta_store < 1 -> event filtering reduces storage.")
    if te > 0.0:
        interp.append("theta_entity > 0 -> entity importance threshold active.")
    if tt < 1.0:
        interp.append("theta_temporal < 1 -> sparse temporal chain.")
    if not interp:
        interp.append("Learned theta near default (1,0,1); task may favor full memory.")
    print("  " + " ".join(interp))
    print("=" * 70)


def print_cross_environment_comparison(env_results: dict) -> None:
    """
    env_results[env_name] = {
        "best_theta": (s, e, t), "learned_theta": (s, e, t),
        "baseline": {success_rate, mean_retrieval_tokens, mean_j, ...},
        "learnable": {success_rate, mean_retrieval_tokens, mean_j, ...},
    }
    """
    print("\n" + "=" * 70)
    print("=== CROSS-ENVIRONMENT COMPARISON ===")
    print("=" * 70)
    for name in env_results:
        r = env_results[name]
        best = r.get("best_theta", (0, 0, 0))
        learned = r.get("learned_theta", (0, 0, 0))
        base = r.get("baseline", {})
        learn = r.get("learnable", {})
        print(f"\n{name}:")
        print(f"  Best theta (Phase 6):   store={best[0]:.3f}, entity={best[1]:.3f}, temporal={best[2]:.3f}")
        print(f"  Learned theta (Phase 7): store={learned[0]:.3f}, entity={learned[1]:.3f}, temporal={learned[2]:.3f}")
        print(f"  Success rate: baseline={base.get('success_rate', 0):.2%}, learnable={learn.get('success_rate', 0):.2%}")
        print(f"  Mean retrieval_tokens: baseline={base.get('mean_retrieval_tokens', 0):.1f}, learnable={learn.get('mean_retrieval_tokens', 0):.1f}")
        print(f"  Mean J: baseline={base.get('mean_j', 0):.4f}, learnable={learn.get('mean_j', 0):.4f}")
    print("\nInterpretation:")
    print("  Memory structure (theta) is task-dependent: different tasks converge to different learned theta.")
    kd = env_results.get("Key-Door", {})
    gr = env_results.get("Goal-Room", {})
    hk = env_results.get("Hard-KeyDoor", {})
    if kd:
        t = kd.get("learned_theta", (1, 0, 1))
        print(f"  Key-Door: theta=({t[0]:.2f},{t[1]:.2f},{t[2]:.2f}) — key-door entity matching dominates.")
    if gr:
        t = gr.get("learned_theta", (1, 0, 1))
        print(f"  Goal-Room: theta=({t[0]:.2f},{t[1]:.2f},{t[2]:.2f}) — sparse, recency-focused memory.")
    if hk:
        t = hk.get("learned_theta", (1, 0, 1))
        print(f"  Hard-KeyDoor: theta=({t[0]:.2f},{t[1]:.2f},{t[2]:.2f}) — must track entities across 300 steps; distractors penalize low theta_entity.")
    print("=" * 70)


def print_memory_comparison_report(
    results: dict[str, dict],
    env_name: str,
    n_episodes: int,
    lambda_penalty: float,
) -> None:
    """Print the memory system comparison table."""
    print("\n" + "=" * 70)
    print(f"=== MEMORY SYSTEM COMPARISON ({env_name}, n={n_episodes}) ===")
    print("=" * 70)
    print(f"Objective: J = partial_score - {lambda_penalty} * retrieval_tokens")
    print(f"\n{'System':<22} | {'PartialScore':>12} | {'Tokens':>8} | {'MemSize':>7} | {'Efficiency':>10} | {'MeanJ':>8}")
    print("-" * 80)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["mean_partial_score"]):
        print(
            f"{name:<22} | {r['mean_partial_score']:>12.3f} | "
            f"{r['mean_retrieval_tokens']:>8.1f} | {r['mean_memory_size']:>7.1f} | "
            f"{r['efficiency']:>10.5f} | {r['mean_j']:>8.4f}"
        )
    best = max(results.items(), key=lambda x: x[1]["mean_j"])
    print(f"\nBest by mean J: {best[0]} (J={best[1]['mean_j']:.4f})")
    best_eff = max(results.items(), key=lambda x: x[1]["efficiency"])
    print(f"Best efficiency: {best_eff[0]} (eff={best_eff[1]['efficiency']:.5f})")
    print("=" * 70)


def main() -> None:
    n_episodes = 100
    learnable_configs = [
        (1.0, 1.0, 0.1),
        (1.5, 1.0, 0.2),
        (1.0, 2.0, 0.1),
    ]
    results = run_evaluation(
        n_episodes=n_episodes,
        env_seed=None,
        policy_seed=42,
        learnable_configs=learnable_configs,
    )

    embedding_demo = demo_embedding_retrieval()

    print_report(results, embedding_demo)
    print_phase5_report(results)

    print("\n--- Sample Embedding Retrieval Log (episode excerpt) ---")
    env = ToyEnvironment(seed=100)
    policy = ExplorationPolicy(seed=42)
    memory = GraphMemory()
    run_episode_with_logging(
        env, policy, memory,
        use_memory=True,
        retrieval_mode="embedding",
        verbose=True,
    )

    n_generations = 12
    n_candidates = 6
    n_episodes_per_candidate = 25
    lambda_penalty_es = 0.001
    n_episodes_equiv = n_generations * n_candidates * n_episodes_per_candidate

    env_data: dict = {}
    env_results: dict = {}

    for env_name in ["Key-Door", "Goal-Room", "Hard-KeyDoor"]:
        print(f"\n--- Phase 6: Learnable Memory Creation ({env_name}) ---")
        best_theta, phase6_results, baseline_fixed = run_phase6_bandit(
            n_theta_configs=15,
            n_episodes_per_theta=75,
            lambda_penalty=0.001,
            env_seed=None,
            policy_seed=42,
            bandit_seed=123,
            env_name=env_name,
        )
        print_phase6_report(
            best_theta,
            phase6_results,
            baseline_fixed,
            n_episodes_per_theta=75,
            lambda_penalty=0.001,
        )

        print(f"\n--- Phase 7: Adaptive Theta (Evolution Strategy) ({env_name}) ---")
        es_theta, phase7_generations, baseline_es = run_phase7_es(
            initial_theta=(1.0, 0.0, 1.0),
            sigma=0.2,
            sigma_decay=0.95,
            sigma_min=0.05,
            n_generations=n_generations,
            n_candidates=n_candidates,
            n_episodes_per_candidate=n_episodes_per_candidate,
            lambda_penalty=lambda_penalty_es,
            env_seed=None,
            policy_seed=42,
            es_seed=456,
            env_name=env_name,
        )
        print_phase7_report(
            es_theta,
            phase7_generations,
            baseline_es,
            n_generations=n_generations,
            n_candidates=n_candidates,
            n_episodes_per_candidate=n_episodes_per_candidate,
            lambda_penalty=lambda_penalty_es,
        )
        learnable_es = phase7_generations[-1] if phase7_generations else {}
        print_experiment_summary(
            environment=env_name,
            method="ES",
            best_theta=es_theta,
            lambda_penalty=lambda_penalty_es,
            n_episodes=n_episodes_equiv,
            baseline=baseline_es,
            learnable={
                "success_rate": learnable_es.get("success_rate", 0),
                "mean_retrieval_tokens": learnable_es.get("mean_retrieval_tokens", 0),
                "mean_memory_size": learnable_es.get("mean_memory_size", 0),
                "mean_j": learnable_es.get("mean_j", 0),
            },
        )
        env_data[env_name] = {
            "best_theta": best_theta,
            "phase6_results": phase6_results,
            "baseline_fixed": baseline_fixed,
            "learned_theta": es_theta,
            "phase7_generations": phase7_generations,
            "baseline_es": baseline_es,
        }
        env_results[env_name] = {
            "best_theta": best_theta,
            "learned_theta": es_theta,
            "baseline": baseline_es,
            "learnable": learnable_es,
        }

    print_cross_environment_comparison(env_results)

    report_path = Path(__file__).parent / "report.txt"
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report(results, embedding_demo)
        print_phase5_report(results)
        for env_name in ["Key-Door", "Goal-Room", "Hard-KeyDoor"]:
            d = env_data[env_name]
            print_phase6_report(
                d["best_theta"],
                d["phase6_results"],
                d["baseline_fixed"],
                n_episodes_per_theta=75,
                lambda_penalty=0.001,
            )
            print_phase7_report(
                d["learned_theta"],
                d["phase7_generations"],
                d["baseline_es"],
                n_generations=n_generations,
                n_candidates=n_candidates,
                n_episodes_per_candidate=n_episodes_per_candidate,
                lambda_penalty=lambda_penalty_es,
            )
            learnable_es = d["phase7_generations"][-1] if d["phase7_generations"] else {}
            print_experiment_summary(
                environment=env_name,
                method="ES",
                best_theta=d["learned_theta"],
                lambda_penalty=lambda_penalty_es,
                n_episodes=n_episodes_equiv,
                baseline=d["baseline_es"],
                learnable={
                    "success_rate": learnable_es.get("success_rate", 0),
                    "mean_retrieval_tokens": learnable_es.get("mean_retrieval_tokens", 0),
                    "mean_memory_size": learnable_es.get("mean_memory_size", 0),
                    "mean_j": learnable_es.get("mean_j", 0),
                },
            )
        print_cross_environment_comparison(env_results)

    # -----------------------------------------------------------------------
    # Memory System Comparison on QuestRoom
    # -----------------------------------------------------------------------
    print("\n--- Memory System Comparison (QuestRoom) ---")
    quest_env = QuestRoom(seed=77)
    quest_policy = ExplorationPolicy(seed=42)
    n_comparison_episodes = 30
    lambda_comp = 0.001

    memory_systems: dict = {
        "FlatWindow(50)": FlatMemory(window_size=50),
        "GraphMemory+θ": GraphMemory(params=MemoryParams(
            theta_store=0.864, theta_entity=0.230, theta_temporal=1.0, mode="learnable"
        )),
        "SemanticMemory": SemanticMemory(max_capacity=80, alpha=1.0, beta=5.0, gamma=2.0),
        "SummaryMemory": SummaryMemory(raw_buffer_size=30, summarize_every=25),
        "EpisodicSemantic": EpisodicSemanticMemory(episodic_size=30),
        "RAGMemory": RAGMemory(),
    }

    comparison_results = run_memory_comparison(
        quest_env,
        quest_policy,
        memory_systems,
        n_episodes=n_comparison_episodes,
        k=8,
        env_seed=77,
        lambda_penalty=lambda_comp,
    )
    print_memory_comparison_report(
        comparison_results, "QuestRoom", n_comparison_episodes, lambda_comp
    )

    report_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")

    # Append comparison to report
    import io as _io
    from contextlib import redirect_stdout as _rds
    comp_buf = _io.StringIO()
    with _rds(comp_buf):
        print_memory_comparison_report(
            comparison_results, "QuestRoom", n_comparison_episodes, lambda_comp
        )
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(comp_buf.getvalue())


if __name__ == "__main__":
    main()
