"""
GraphMemory V3 — Proposal B: Learned Event Importance Scoring.

Extends V2 by replacing the blind Bernoulli(theta_store) storage gate with
a principled, task-agnostic importance scoring function.

THE PROBLEM WITH V1/V2 STORAGE:
    if rng.random() > theta_store: skip
This drops events uniformly at random. A rare, information-rich observation
(a new entity introduced for the first time) has the same store probability
as a repeated, information-poor one ("you move north" for the 50th time).
The optimizer cannot express "store novel events, skip repetitive ones" using
a single scalar theta_store — it can only raise or lower the bar uniformly.

THE V3 SOLUTION:
    importance(obs) = theta_novel   * novelty(obs)
                    + theta_erich   * entity_richness(obs)
                    + theta_surprise * surprise(obs)
    store if importance(obs) > theta_store  (theta_store acts as a threshold)

Where the three feature functions are ALL task-agnostic:

    novelty(obs):
        1 - max_cosine_sim(embed(obs), recent_stored_embeddings)
        High for genuinely new observations. Low for repeated navigation noise.
        A hint seen for the first time scores ~1.0.
        "You move north" for the 30th time scores ~0.0.

    entity_richness(obs):
        len(extract_entities(obs)) / (max_entities_seen_so_far + 1)
        Normalized count of entities in this observation.
        An observation mentioning two entities is richer than one with none.
        Works for any domain — DocumentQA facts, NPC dialogue, game events.

    surprise(obs):
        ||embed(obs) - mean(last_k_embeddings)||  (L2 norm, normalized)
        How much does this observation deviate from recent context?
        Sudden topic changes score high. Continuation of existing theme scores low.
        This fires on domain shifts: entering a new room, encountering a new NPC,
        receiving an unexpected game event.

FULL THETA VECTOR (9D):
    θ = (theta_store,    # importance threshold [0, 1]
         theta_novel,    # weight on novelty [0, 1]
         theta_erich,    # weight on entity richness [0, 1]
         theta_surprise, # weight on context surprise [0, 1]
         theta_entity,   # entity node importance threshold [0, 1]
         theta_temporal, # temporal edge probability [0, 1]
         w_graph,        # retrieval: graph signal weight [0, 4]
         w_embed,        # retrieval: cosine similarity weight [0, 4]
         w_recency)      # retrieval: recency weight [0, 4]

WHY THIS GENERALIZES:
    None of these three features encode task-specific knowledge.
    On MultiHopKeyDoor: hints are novel + entity-rich → stored reliably.
    On DocumentQA: new facts are novel + entity-rich → stored reliably.
    On navigation: repeated movement is not novel, not entity-rich → filtered.
    The optimizer discovers the right feature weights from reward alone.

Original file preserved at: memory/graph_memory.py
V2 (retrieval weights only) preserved at: memory/graph_memory_v2.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import networkx as nx

from .embedding import embed_observation
from .entity_extraction import extract_entities
from .event import Event

# Rolling window size for novelty and surprise computation
_NOVELTY_WINDOW = 50


def _event_node(step: int) -> str:
    return f"event_{step}"


@dataclass
class MemoryParamsV3:
    """
    9D learnable memory parameters.

    Storage dims:
        theta_store    — importance threshold: store if score > this [0, 1]
        theta_novel    — weight on novelty feature [0, 1]
        theta_erich    — weight on entity richness feature [0, 1]
        theta_surprise — weight on context surprise feature [0, 1]
        theta_entity   — entity node importance threshold [0, 1]
        theta_temporal — temporal edge Bernoulli probability [0, 1]

    Retrieval dims (from V2):
        w_graph   — graph traversal signal weight [0, 4]
        w_embed   — TF-IDF cosine similarity weight [0, 4]
        w_recency — recency score weight [0, 4]

    Defaults: theta_store=0.0 (store everything by default so that
    baseline behavior is preserved when all feature weights are 0).
    When theta_novel=theta_erich=theta_surprise=0, importance=0 for all
    observations, so importance > theta_store=0 is never True — this
    reduces exactly to Bernoulli(1.0), i.e., store everything.
    """
    theta_store: float = 0.0
    theta_novel: float = 0.0
    theta_erich: float = 0.0
    theta_surprise: float = 0.0
    theta_entity: float = 0.0
    theta_temporal: float = 1.0
    w_graph: float = 1.5
    w_embed: float = 1.0
    w_recency: float = 0.2
    mode: Literal["fixed", "learnable"] = "learnable"

    def __post_init__(self) -> None:
        if self.mode == "fixed":
            return
        for attr in ("theta_store", "theta_novel", "theta_erich",
                     "theta_surprise", "theta_entity", "theta_temporal"):
            setattr(self, attr, max(0.0, min(1.0, getattr(self, attr))))
        for attr in ("w_graph", "w_embed", "w_recency"):
            setattr(self, attr, max(0.0, min(4.0, getattr(self, attr))))

    @classmethod
    def from_vector(cls, v: list[float] | tuple[float, ...]) -> "MemoryParamsV3":
        """Construct from a flat 9-element vector produced by CMA-ES."""
        if len(v) < 9:
            raise ValueError(f"Expected 9 values, got {len(v)}")
        return cls(
            theta_store=float(v[0]),
            theta_novel=float(v[1]),
            theta_erich=float(v[2]),
            theta_surprise=float(v[3]),
            theta_entity=float(v[4]),
            theta_temporal=float(v[5]),
            w_graph=float(v[6]),
            w_embed=float(v[7]),
            w_recency=float(v[8]),
        )

    def to_vector(self) -> tuple[float, ...]:
        return (self.theta_store, self.theta_novel, self.theta_erich,
                self.theta_surprise, self.theta_entity, self.theta_temporal,
                self.w_graph, self.w_embed, self.w_recency)


def _memory_rng(episode_seed: int, step: int) -> random.Random:
    h = (episode_seed * 10000 + step) % (2 ** 32)
    return random.Random(h)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _compute_importance(
    obs: str,
    params: MemoryParamsV3,
    stored_embeddings: list[np.ndarray],
    max_entities_seen: int,
) -> float:
    """
    Compute a task-agnostic importance score for an observation.
    All three component features are properties of the observation relative
    to the current memory state — no task-specific knowledge used.

    Returns a float in [0, ~3]. An observation is stored if this score
    exceeds params.theta_store.

    If all feature weights are zero (default), returns 0.0 for every
    observation, so the threshold check (0.0 > 0.0) is always False and
    every event is stored — backward compatible.
    """
    # Short-circuit: if all weights are zero, skip expensive computation
    if params.theta_novel == 0.0 and params.theta_erich == 0.0 and params.theta_surprise == 0.0:
        return 0.0

    obs_emb = embed_observation(obs)

    # --- Novelty: how different is this from stored memory? ---
    if params.theta_novel > 0.0 and stored_embeddings:
        # Use a rolling window of recent stored embeddings for efficiency
        recent = stored_embeddings[-_NOVELTY_WINDOW:]
        max_sim = max(_cosine_sim(obs_emb, e) for e in recent)
        novelty = max(0.0, 1.0 - max_sim)
    else:
        novelty = 1.0  # first observation is maximally novel

    # --- Entity richness: how many entities does this observation mention? ---
    if params.theta_erich > 0.0:
        n_entities = len(extract_entities(obs))
        denom = max(max_entities_seen, 1)
        entity_richness = min(n_entities / denom, 1.0)
    else:
        entity_richness = 0.0

    # --- Surprise: how much does this deviate from recent context? ---
    if params.theta_surprise > 0.0 and len(stored_embeddings) >= 2:
        recent = stored_embeddings[-_NOVELTY_WINDOW:]
        mean_emb = np.mean(recent, axis=0)
        # L2 distance, normalized by sqrt(dim) so it's roughly in [0, 1]
        dim = max(len(obs_emb), 1)
        surprise = float(np.linalg.norm(obs_emb - mean_emb)) / (dim ** 0.5)
        surprise = min(surprise, 1.0)
    else:
        surprise = 0.0

    return (params.theta_novel * novelty
            + params.theta_erich * entity_richness
            + params.theta_surprise * surprise)


class GraphMemoryV3:
    """
    GraphMemory with learned event importance scoring (Proposal B).

    Storage: instead of Bernoulli(theta_store), compute importance(obs) using
    novelty, entity richness, and surprise; store if importance > theta_store.
    Retrieval: learnable weights from MemoryParamsV3 (same as V2).

    Drop-in compatible with the standard memory interface.
    """

    def __init__(self, params: MemoryParamsV3 | None = None) -> None:
        self._graph = nx.DiGraph()
        self._params = params or MemoryParamsV3()
        self._entity_mention_count: dict[str, int] = {}
        # Rolling window of stored embeddings for novelty/surprise
        self._stored_embeddings: list[np.ndarray] = []
        self._max_entities_seen: int = 0

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def add_event(self, event: Event, episode_seed: int | None = None) -> None:
        params = self._params
        if params.mode == "fixed":
            self._add_event_fixed(event)
            return

        # --- Importance-based storage gate ---
        importance = _compute_importance(
            event.observation,
            params,
            self._stored_embeddings,
            self._max_entities_seen,
        )
        # When all feature weights are 0: importance=0.0 and threshold=0.0,
        # so (0.0 > 0.0) is False → store everything. Backward compatible.
        if importance <= params.theta_store and not (
            params.theta_novel == 0.0
            and params.theta_erich == 0.0
            and params.theta_surprise == 0.0
        ):
            return

        node_id = _event_node(event.step)
        embedding = embed_observation(event.observation)
        self._graph.add_node(
            node_id,
            type="event",
            step=event.step,
            observation=event.observation,
            action=event.action,
            embedding=embedding,
            event=event,
        )
        # Track stored embeddings for future novelty/surprise computation
        self._stored_embeddings.append(embedding)
        if len(self._stored_embeddings) > _NOVELTY_WINDOW * 2:
            self._stored_embeddings = self._stored_embeddings[-_NOVELTY_WINDOW:]

        # --- Entity nodes ---
        entities = extract_entities(event.observation)
        self._max_entities_seen = max(self._max_entities_seen, len(entities))
        for entity_name in entities:
            self._entity_mention_count[entity_name] = (
                self._entity_mention_count.get(entity_name, 0) + 1
            )
        total_mentions = sum(self._entity_mention_count.values())
        for entity_name in entities:
            count = self._entity_mention_count.get(entity_name, 0)
            ent_importance = (count / total_mentions) if total_mentions > 0 else 0.0
            if ent_importance > params.theta_entity:
                if not self._graph.has_node(entity_name):
                    self._graph.add_node(entity_name, type="entity", name=entity_name)
                self._graph.add_edge(node_id, entity_name, edge_type="mentions")
                self._graph.add_edge(entity_name, node_id, edge_type="mentioned_in")

        # --- Temporal edge ---
        rng = _memory_rng(episode_seed if episode_seed is not None else 0, event.step)
        if event.step > 0:
            prev_id = _event_node(event.step - 1)
            if self._graph.has_node(prev_id) and rng.random() < params.theta_temporal:
                self._graph.add_edge(prev_id, node_id, edge_type="temporal")

    def _add_event_fixed(self, event: Event) -> None:
        node_id = _event_node(event.step)
        embedding = embed_observation(event.observation)
        self._graph.add_node(
            node_id,
            type="event",
            step=event.step,
            observation=event.observation,
            action=event.action,
            embedding=embedding,
            event=event,
        )
        self._stored_embeddings.append(embedding)
        if event.step > 0:
            prev_id = _event_node(event.step - 1)
            if self._graph.has_node(prev_id):
                self._graph.add_edge(prev_id, node_id, edge_type="temporal")
        for entity_name in extract_entities(event.observation):
            if not self._graph.has_node(entity_name):
                self._graph.add_node(entity_name, type="entity", name=entity_name)
            self._graph.add_edge(node_id, entity_name, edge_type="mentions")
            self._graph.add_edge(entity_name, node_id, edge_type="mentioned_in")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_relevant_events(
        self,
        observation: str,
        current_step: int,
        k: int = 8,
    ) -> list[Event]:
        from .retrieval import retrieve_events_learnable
        p = self._params
        return retrieve_events_learnable(
            self._graph,
            observation,
            current_step=current_step,
            k=k,
            w_graph=p.w_graph,
            w_embed=p.w_embed,
            w_recency=p.w_recency,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_graph(self) -> nx.DiGraph:
        return self._graph

    def get_all_events(self) -> list[Event]:
        events = []
        for _, data in self._graph.nodes(data=True):
            if data.get("type") == "event" and "event" in data:
                events.append(data["event"])
        events.sort(key=lambda e: e.step)
        return events

    def get_stats(self) -> dict:
        n_events = sum(1 for _, d in self._graph.nodes(data=True) if d.get("type") == "event")
        n_entities = sum(1 for _, d in self._graph.nodes(data=True) if d.get("type") == "entity")
        return {
            "n_nodes": self._graph.number_of_nodes(),
            "n_edges": self._graph.number_of_edges(),
            "n_events": n_events,
            "n_entities": n_entities,
        }

    def clear(self) -> None:
        self._graph.clear()
        self._entity_mention_count.clear()
        self._stored_embeddings.clear()
        self._max_entities_seen = 0
