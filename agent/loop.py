"""Agent loop: observe, store event, retrieve memory, decide.

Supports two runner styles:
1. run_episode_with_memory  — GraphMemory with graph/embedding/hybrid/learnable retrieval (existing).
2. run_episode_with_any_memory — accepts any memory class exposing the uniform interface:
       add_event(event, episode_seed=None) -> None
       get_relevant_events(observation, current_step, k=8) -> list[Event]
       clear() -> None
       get_stats() -> dict
   Used by the memory system comparison (FlatMemory, SemanticMemory, SummaryMemory,
   EpisodicSemanticMemory, RAGMemory, and the GraphMemory adapter).
"""

from typing import Any

from environment import ToyEnvironment
from memory import Event, GraphMemory, retrieve_events, retrieve_events_learnable

from .policy import ExplorationPolicy

Action = str  # type alias

# Retrieval mode: "graph" | "embedding" | "hybrid" | "learnable"
RETRIEVAL_MODE = "hybrid"


def run_episode_no_memory(
    env: ToyEnvironment,
    policy: ExplorationPolicy,
) -> tuple[bool, list[Event]]:
    """
    Run one episode with NO memory baseline.
    Agent acts only on current observation.
    Returns (success, list of events that occurred).
    """
    obs = env.reset()
    events: list[Event] = []
    step = 0

    while not env.done:
        action = policy.decide(obs, past_events=[])  # no memory
        events.append(Event(step=step, observation=obs, action=action))
        obs, done, success = env.step(action)
        step += 1

    return env.success, events


def _retrieval_config(mode: str) -> tuple[bool, bool]:
    if mode == "graph":
        return True, False
    if mode == "embedding":
        return False, True
    if mode == "learnable":
        return True, True  # unused when learnable; branch handles it
    return True, True  # hybrid


def run_episode_with_logging(
    env: ToyEnvironment,
    policy: ExplorationPolicy,
    memory: GraphMemory | None,
    retrieve_n: int = 8,
    use_memory: bool = True,
    retrieval_mode: str = "hybrid",
    learnable_weights: tuple[float, float, float] | None = (1.0, 1.0, 0.1),
    episode_seed: int | None = None,
    verbose: bool = False,
) -> tuple[bool, list[dict]]:
    """Run one episode and return (success, list of step logs) for reporting."""
    if memory:
        memory.clear()
    obs = env.reset()
    step = 0
    logs = []
    use_graph, use_emb = _retrieval_config(retrieval_mode)

    while not env.done:
        past: list[Event] = []
        if use_memory and memory:
            if retrieval_mode == "learnable":
                w_g, w_e, w_r = learnable_weights or (1.0, 1.0, 0.1)
                past = retrieve_events_learnable(
                    memory.get_graph(),
                    obs,
                    current_step=step,
                    k=retrieve_n,
                    w_graph=w_g,
                    w_embed=w_e,
                    w_recency=w_r,
                    return_debug=False,
                )
            else:
                do_verbose = verbose and (step in (0, 3, 8))
                past = retrieve_events(
                    obs,
                    memory.get_graph(),
                    use_graph=use_graph,
                    use_embedding=use_emb,
                    last_n=retrieve_n,
                    k=retrieve_n,
                    verbose=do_verbose,
                )
        action = policy.decide(obs, past_events=past if use_memory else [])

        log = {"step": step, "obs": obs, "action": action}
        if past:
            log["retrieved"] = [e.observation[:60] + ("..." if len(e.observation) > 60 else "") for e in past[-3:]]

        logs.append(log)

        if use_memory and memory:
            memory.add_event(Event(step=step, observation=obs, action=action), episode_seed=episode_seed)

        obs, _, _ = env.step(action)
        step += 1

    return env.success, logs


def run_episode_with_memory(
    env: ToyEnvironment,
    policy: ExplorationPolicy,
    memory: GraphMemory,
    retrieve_n: int = 8,
    retrieval_mode: str = "hybrid",
    learnable_weights: tuple[float, float, float] | None = (1.0, 1.0, 0.1),
    episode_seed: int | None = None,
) -> tuple[bool, list[Event], dict]:
    """
    Run one episode with memory (graph, embedding, hybrid, or learnable).
    Returns (success, list of events that occurred, stats_dict).
    stats_dict has retrieval_tokens (sum of len(past) per step) and memory_size (n_events at end).
    """
    memory.clear()
    obs = env.reset()
    events: list[Event] = []
    step = 0
    retrieval_tokens = 0
    use_graph, use_emb = _retrieval_config(retrieval_mode)

    while not env.done:
        if retrieval_mode == "learnable":
            w_g, w_e, w_r = learnable_weights or (1.0, 1.0, 0.1)
            past = retrieve_events_learnable(
                memory.get_graph(),
                obs,
                current_step=step,
                k=retrieve_n,
                w_graph=w_g,
                w_embed=w_e,
                w_recency=w_r,
                return_debug=False,
            )
        else:
            past = retrieve_events(
                obs,
                memory.get_graph(),
                use_graph=use_graph,
                use_embedding=use_emb,
                last_n=retrieve_n,
                k=retrieve_n,
                verbose=False,
            )

        retrieval_tokens += len(past)
        action = policy.decide(obs, past_events=past)

        event = Event(step=step, observation=obs, action=action)
        memory.add_event(event, episode_seed=episode_seed)
        events.append(event)

        obs, done, success = env.step(action)
        step += 1

    stats = memory.get_stats()
    # partial_score supports HardKeyDoor (doors_opened / 3); falls back to binary for other envs
    reward = getattr(env, "partial_score", 1.0 if env.success else 0.0)
    stats_dict = {
        "retrieval_tokens": retrieval_tokens,
        "memory_size": stats["n_events"],
        "reward": reward,
    }
    return env.success, events, stats_dict


def run_episode_with_any_memory(
    env: Any,
    policy: ExplorationPolicy,
    memory: Any,
    k: int = 8,
    episode_seed: int | None = None,
) -> tuple[bool, list[Event], dict]:
    """
    Universal episode runner that works with ANY memory system exposing:
        add_event(event, episode_seed=None) -> None
        get_relevant_events(observation, current_step, k) -> list[Event]
        clear() -> None
        get_stats() -> dict

    Compatible with: FlatMemory, SemanticMemory, SummaryMemory,
    EpisodicSemanticMemory, RAGMemory, and GraphMemory (via get_relevant_events).

    Tracks retrieval_precision: when the agent is at a door and the environment
    has hint events, measures whether any hint appeared in the retrieved top-k.
    This is the direct memory quality metric independent of task success rate.

    Returns (success, events, stats_dict).
    stats_dict keys: retrieval_tokens, memory_size, reward, retrieval_precision.
    retrieval_precision is None when no door-query moments occurred.
    """
    memory.clear()
    obs = env.reset()
    events: list[Event] = []
    step = 0
    retrieval_tokens = 0

    # Retrieval precision tracking: count door-step queries and hint hits
    hint_queries = 0   # steps where agent was at a door (memory needed)
    hint_hits = 0      # of those, retrieved events contained at least one hint

    # Get hint observations from the environment if available (MultiHopKeyDoor)
    env_hints: set[str] = set()
    if hasattr(env, "hint_observations"):
        env_hints = set(env.hint_observations)

    while not env.done:
        past: list[Event] = memory.get_relevant_events(obs, current_step=step, k=k)
        retrieval_tokens += len(past)

        # Retrieval precision: if at a door step and hints exist, score this retrieval.
        # "door" obs with "requires" means agent is standing at an unopened door
        # and needs to know the correct key — the prime moment to recall a hint.
        if env_hints and "door" in obs.lower() and "requires" in obs.lower():
            hint_queries += 1
            # A hit: any retrieved event is marked as a hint OR its obs contains a hint substring
            if any(
                e.is_hint or any(h in e.observation for h in env_hints)
                for e in past
            ):
                hint_hits += 1

        action = policy.decide(obs, past_events=past)

        # Mark hint events: obs contains one of the hint substrings
        is_hint = any(h in obs for h in env_hints)
        event = Event(step=step, observation=obs, action=action, is_hint=is_hint)
        memory.add_event(event, episode_seed=episode_seed)
        events.append(event)

        obs, done, success = env.step(action)
        step += 1

    stats = memory.get_stats()
    reward = getattr(env, "partial_score", 1.0 if env.success else 0.0)
    retrieval_precision = (hint_hits / hint_queries) if hint_queries > 0 else None
    stats_dict = {
        "retrieval_tokens": retrieval_tokens,
        "memory_size": stats["n_events"],
        "reward": reward,
        "retrieval_precision": retrieval_precision,
        "hint_queries": hint_queries,
        "hint_hits": hint_hits,
    }
    return env.success, events, stats_dict
