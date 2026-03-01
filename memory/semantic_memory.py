"""
SemanticMemory — importance-weighted memory with capacity limit.

Each event is scored by importance:
    importance = alpha * entity_count
               + beta  * is_npc_hint      (obs contains "says:")
               + gamma * is_novel_entity  (first time this entity seen)

Events are stored in a capped pool (max_capacity). When the pool is full, the
lowest-importance event is evicted to make room for a new high-importance one.

At retrieval: embed current observation, score stored events by
    combined = importance(event) * cosine_similarity(query, event_embedding)
Return top-k by combined score.

Parameters alpha, beta, gamma are learnable (same ES loop as theta). Defaults
chosen so that NPC hints (beta dominates) are always retained.
"""

import numpy as np

from .embedding import embed_observation
from .entity_extraction import extract_entities
from .event import Event


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SemanticMemory:
    """
    Importance-weighted memory pool with eviction.

    Not rule-based: importance is computed from observable signals
    (entity density, NPC hint marker, novelty) rather than hardcoded keys.
    """

    def __init__(
        self,
        max_capacity: int = 80,
        alpha: float = 1.0,   # weight for entity count
        beta: float = 5.0,    # weight for NPC hint detection
        gamma: float = 2.0,   # weight for novel entity
    ) -> None:
        self._max_capacity = max_capacity
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # Stored as list of (event, importance, embedding)
        self._store: list[tuple[Event, float, np.ndarray]] = []
        self._seen_entities: set[str] = set()

    def _compute_importance(self, event: Event) -> float:
        obs = event.observation
        obs_lower = obs.lower()

        entities = extract_entities(obs)
        entity_count = len(entities)

        is_npc_hint = 1.0 if ("says:" in obs_lower or "guard says" in obs_lower or "sage says" in obs_lower) else 0.0

        novel_count = sum(1 for e in entities if e not in self._seen_entities)
        is_novel = 1.0 if novel_count > 0 else 0.0

        return (
            self.alpha * entity_count
            + self.beta * is_npc_hint
            + self.gamma * is_novel
        )

    def add_event(self, event: Event, episode_seed: int | None = None) -> None:
        entities = extract_entities(event.observation)
        importance = self._compute_importance(event)
        self._seen_entities.update(entities)

        emb = embed_observation(event.observation)

        if len(self._store) < self._max_capacity:
            self._store.append((event, importance, emb))
        else:
            # Evict the lowest-importance event
            min_idx = min(range(len(self._store)), key=lambda i: self._store[i][1])
            if importance > self._store[min_idx][1]:
                self._store[min_idx] = (event, importance, emb)
            # If new event is less important than everything stored, discard it

    def get_relevant_events(
        self,
        observation: str,
        current_step: int,
        k: int = 8,
    ) -> list[Event]:
        if not self._store:
            return []
        query_emb = embed_observation(observation)
        scored = []
        for event, importance, emb in self._store:
            sim = (_cosine_sim(query_emb, emb) + 1.0) / 2.0  # normalize to [0,1]
            combined = importance * sim
            scored.append((event, combined))
        scored.sort(key=lambda x: -x[1])
        return [e for e, _ in scored[:k]]

    def clear(self) -> None:
        self._store.clear()
        self._seen_entities.clear()

    def get_stats(self) -> dict:
        return {
            "n_events": len(self._store),
            "n_entities": len(self._seen_entities),
            "n_nodes": len(self._store),
            "n_edges": 0,
        }
