"""Graph-based and similarity-based retrieval."""

from typing import TYPE_CHECKING

import numpy as np

from .embedding import embed_observation
from .entity_extraction import extract_entities
from .event import Event

if TYPE_CHECKING:
    import networkx as nx

USE_GRAPH = True
USE_EMBEDDING = True


def _entity_to_matching_key(entity_name: str) -> str | None:
    """red_door -> red_key, blue_door -> blue_key, etc."""
    if entity_name.endswith("_door"):
        color = entity_name.replace("_door", "")
        return f"{color}_key"
    return None


def _get_events_from_entity(graph: "nx.DiGraph", entity_name: str) -> list[Event]:
    """Get events connected to entity via graph traversal."""
    if not graph.has_node(entity_name):
        return []
    events = []
    for neighbor in graph.predecessors(entity_name):
        data = graph.nodes.get(neighbor, {})
        if data.get("type") == "event" and "event" in data:
            events.append(data["event"])
    for neighbor in graph.successors(entity_name):
        data = graph.nodes.get(neighbor, {})
        if data.get("type") == "event" and "event" in data:
            events.append(data["event"])
    return list({id(e): e for e in events}.values())


def _get_all_events_sorted(graph: "nx.DiGraph") -> list[Event]:
    """Get all event nodes sorted by step."""
    events = []
    for _, data in graph.nodes(data=True):
        if data.get("type") == "event" and "event" in data:
            events.append(data["event"])
    events.sort(key=lambda e: e.step)
    return events


def retrieve_relevant_events(
    current_observation: str,
    graph: "nx.DiGraph",
    last_n: int = 5,
) -> list[Event]:
    """
    Graph-based retrieval (no keyword matching).

    CASE 1: Obs contains DOOR -> find matching key entity, get events connected to key
    CASE 2: Obs contains KEY -> get events connected to this key (+ optional matching door)
    CASE 3: No relevant entity -> last N events
    """
    entities = extract_entities(current_observation)

    for entity_name in entities:
        if entity_name.endswith("_door"):
            matching_key = _entity_to_matching_key(entity_name)
            if matching_key:
                events = _get_events_from_entity(graph, matching_key)
                if events:
                    events.sort(key=lambda e: e.step)
                    return events[-last_n:] if last_n > 0 else events

        if entity_name.endswith("_key"):
            events = _get_events_from_entity(graph, entity_name)
            matching_door = entity_name.replace("_key", "_door")
            door_events = _get_events_from_entity(graph, matching_door)
            combined = events + door_events
            seen = set()
            unique = []
            for e in sorted(combined, key=lambda x: x.step):
                if id(e) not in seen:
                    seen.add(id(e))
                    unique.append(e)
            if unique:
                return unique[-last_n:] if last_n > 0 else unique

    all_events = _get_all_events_sorted(graph)
    return all_events[-last_n:] if last_n > 0 and all_events else all_events


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    anorm = np.linalg.norm(a)
    bnorm = np.linalg.norm(b)
    if anorm == 0 or bnorm == 0:
        return 0.0
    return float(np.dot(a, b) / (anorm * bnorm))


def retrieve_similar_events(
    observation: str,
    graph: "nx.DiGraph",
    k: int = 5,
    verbose: bool = False,
) -> tuple[list[Event], list[float]]:
    """
    Similarity-based retrieval: embed observation, compute cosine similarity
    with all past event embeddings, return top-k most similar events.
    Returns (events, scores) sorted by score descending.
    """
    query_emb = embed_observation(observation)
    events_with_scores: list[tuple[Event, float]] = []

    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "event" or "embedding" not in data:
            continue
        emb = data["embedding"]
        event = data.get("event")
        if event is None:
            continue
        score = _cosine_similarity(query_emb, emb)
        events_with_scores.append((event, score))

    events_with_scores.sort(key=lambda x: -x[1])
    top = events_with_scores[:k]
    events = [e for e, _ in top]
    scores = [s for _, s in top]

    if verbose and events:
        print("\n[Embedding Retrieval]")
        q_short = observation[:60] + "..." if len(observation) > 60 else observation
        print(f"Query: \"{q_short}\"")
        print("Top matches:")
        for e, s in top[:5]:
            obs_short = e.observation[:50] + "..." if len(e.observation) > 50 else e.observation
            print(f"  - event_{e.step} (score={s:.2f}): {obs_short}")

    return events, scores


def retrieve_events(
    observation: str,
    graph: "nx.DiGraph",
    use_graph: bool = True,
    use_embedding: bool = True,
    last_n: int = 5,
    k: int = 5,
    verbose: bool = False,
) -> list[Event]:
    """
    Hybrid retrieval: combines graph-based and similarity-based.
    Configurable via use_graph, use_embedding.
    """
    graph_results: list[Event] = []
    embedding_results: list[Event] = []
    mode_used: list[str] = []

    if verbose:
        mode_used = []
        if use_graph:
            mode_used.append("graph")
        if use_embedding:
            mode_used.append("embedding")
        if mode_used:
            print(f"[Retrieval mode: {' + '.join(mode_used)}]")

    if use_graph:
        graph_results = retrieve_relevant_events(observation, graph, last_n=last_n)

    if use_embedding:
        emb_events, _ = retrieve_similar_events(observation, graph, k=k, verbose=verbose)
        embedding_results = emb_events

    if not graph_results and not embedding_results:
        all_events = _get_all_events_sorted(graph)
        return all_events[-last_n:] if last_n > 0 and all_events else all_events

    if use_graph and not use_embedding:
        return graph_results
    if use_embedding and not use_graph:
        return embedding_results

    seen = set()
    merged: list[Event] = []
    for e in graph_results + embedding_results:
        if id(e) not in seen:
            seen.add(id(e))
            merged.append(e)
    merged.sort(key=lambda e: e.step)
    return merged[-max(last_n, k):] if merged else merged


def retrieve_events_learnable(
    graph: "nx.DiGraph",
    current_observation: str,
    current_step: int,
    k: int = 5,
    w_graph: float = 1.0,
    w_embed: float = 1.0,
    w_recency: float = 0.1,
    return_debug: bool = False,
) -> list[Event] | tuple[list[Event], dict]:
    """
    Learnable retrieval: score all candidate events by weighted combination of
    graph signal, embedding similarity, and recency; return top-k by score.
    """
    candidates = _get_all_events_sorted(graph)
    if not candidates:
        return ([], {}) if return_debug else []

    graph_relevant = retrieve_relevant_events(
        current_observation, graph, last_n=999
    )
    graph_relevant_ids = {id(e) for e in graph_relevant}

    query_emb = embed_observation(current_observation)

    scored: list[tuple[Event, float, float, float, float]] = []
    for _, data in graph.nodes(data=True):
        if data.get("type") != "event" or "event" not in data or "embedding" not in data:
            continue
        event = data["event"]
        emb = data["embedding"]

        graph_signal = 1.0 if id(event) in graph_relevant_ids else 0.0

        cos_sim = _cosine_similarity(query_emb, emb)
        embedding_similarity = (cos_sim + 1.0) / 2.0

        delta = max(0, current_step - event.step)
        recency_score = 1.0 / (1.0 + delta)

        score = (
            w_graph * graph_signal
            + w_embed * embedding_similarity
            + w_recency * recency_score
        )
        scored.append((event, score, graph_signal, embedding_similarity, recency_score))

    scored.sort(key=lambda x: -x[1])
    top = scored[:k]
    events = [e for e, *_ in top]

    if return_debug:
        if top:
            n = len(top)
            debug = {
                "avg_graph_signal": sum(t[2] for t in top) / n,
                "avg_embedding_similarity": sum(t[3] for t in top) / n,
                "avg_recency_score": sum(t[4] for t in top) / n,
            }
            return events, debug
        return events, {}

    return events
