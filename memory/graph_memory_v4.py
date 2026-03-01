"""
GraphMemory V4 — Proposal C: Bayesian Entity Importance with Temporal Decay.

Extends V3 by replacing the unstable count/total entity importance formula
with a Bayesian smoothed model plus a learnable temporal decay parameter.

THE PROBLEM WITH V1-V3 ENTITY IMPORTANCE:
    importance = count / total_mentions

    This has two failure modes:

    1. Early-episode instability: at step 0, total_mentions=1.
       The first entity has importance=1.0, which immediately drops to 0.5
       when the second mention arrives. This creates a sharp, noisy importance
       curve at the start of episodes — exactly when hints arrive (steps 0-2).

    2. No notion of time: an entity last seen at step 5 and one seen at step
       200 have the same importance if their mention counts are equal.
       For tasks where entities become irrelevant over time (navigation, role
       changes, multi-session), this is wrong.

THE V4 SOLUTION — Bayesian smoothing:
    importance(entity) = (count + alpha) / (total + beta * n_entities)
                       * decay(delta_t, theta_decay)

    alpha (prior pseudo-count) = 0.5:
        A new entity with 1 mention gets importance (1+0.5)/(total+...) instead
        of 1/total. The prior prevents extreme values at small counts.

    beta (concentration parameter) = 1.0:
        Scales how strongly the prior shrinks estimates toward 1/n_entities.

    decay = exp(-theta_decay * delta_t):
        delta_t = current_step - entity_last_step
        theta_decay = 0.0 → no decay (entity stays relevant forever)
        theta_decay = 0.05 → entity importance halves every ~14 steps
        theta_decay = 0.5  → entity importance halves every ~1.4 steps

    theta_decay is a LEARNABLE PARAMETER. The optimizer discovers the right
    decay rate from reward:
        - MultiHopKeyDoor: theta_decay → 0.0 (hints never decay)
        - Navigation with changing rooms: theta_decay → high (stale rooms irrelevant)
        - DocumentQA: theta_decay → 0.0 (facts stay relevant throughout)
        - Multi-session with shifting topics: theta_decay → moderate

FULL THETA VECTOR (10D):
    θ = (theta_store,    # importance threshold [0, 1]
         theta_novel,    # novelty feature weight [0, 1]
         theta_erich,    # entity richness feature weight [0, 1]
         theta_surprise, # context surprise feature weight [0, 1]
         theta_entity,   # entity node importance threshold [0, 1]
         theta_temporal, # temporal edge probability [0, 1]
         theta_decay,    # entity importance decay rate [0, 1]
         w_graph,        # retrieval: graph signal weight [0, 4]
         w_embed,        # retrieval: embedding similarity weight [0, 4]
         w_recency)      # retrieval: recency score weight [0, 4]

BACKWARD COMPATIBILITY:
    theta_decay=0.0 → no decay → entity importance = Bayesian smoothed count.
    With alpha=0.5, beta=1.0, this is slightly different from the original
    count/total but converges to the same value as counts grow. The change
    is most significant at small counts (early episode), which is where the
    improvement matters.

Original files preserved:
    memory/graph_memory.py   — V1, original
    memory/graph_memory_v2.py — V2, +retrieval weights
    memory/graph_memory_v3.py — V3, +importance scoring
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import networkx as nx

from .embedding import embed_observation
from .entity_extraction import extract_entities
from .event import Event

_NOVELTY_WINDOW = 50
_ALPHA_PRIOR = 0.5   # pseudo-count for Bayesian smoothing
_BETA_PRIOR = 1.0    # concentration parameter


def _event_node(step: int) -> str:
    return f"event_{step}"


@dataclass
class MemoryParamsV4:
    """
    10D learnable memory parameters — the full parameterized system.

    Storage (6 dims):
        theta_store    — importance threshold [0, 1]
        theta_novel    — novelty feature weight [0, 1]
        theta_erich    — entity richness feature weight [0, 1]
        theta_surprise — context surprise feature weight [0, 1]
        theta_entity   — entity node importance threshold [0, 1]
        theta_temporal — temporal edge Bernoulli probability [0, 1]

    Entity decay (1 dim):
        theta_decay    — exponential decay rate for entity importance [0, 1]
                         0 = no decay, 1 = very fast decay

    Retrieval (3 dims):
        w_graph   — graph traversal signal weight [0, 4]
        w_embed   — TF-IDF cosine similarity weight [0, 4]
        w_recency — recency score weight [0, 4]
    """
    theta_store: float = 0.0
    theta_novel: float = 0.0
    theta_erich: float = 0.0
    theta_surprise: float = 0.0
    theta_entity: float = 0.0
    theta_temporal: float = 1.0
    theta_decay: float = 0.0
    w_graph: float = 1.5
    w_embed: float = 1.0
    w_recency: float = 0.2
    mode: Literal["fixed", "learnable"] = "learnable"

    def __post_init__(self) -> None:
        if self.mode == "fixed":
            return
        for attr in ("theta_store", "theta_novel", "theta_erich", "theta_surprise",
                     "theta_entity", "theta_temporal", "theta_decay"):
            setattr(self, attr, max(0.0, min(1.0, getattr(self, attr))))
        for attr in ("w_graph", "w_embed", "w_recency"):
            setattr(self, attr, max(0.0, min(4.0, getattr(self, attr))))

    @classmethod
    def from_vector(cls, v: list[float] | tuple[float, ...]) -> "MemoryParamsV4":
        """Construct from a flat 10-element vector produced by CMA-ES."""
        if len(v) < 10:
            raise ValueError(f"Expected 10 values, got {len(v)}")
        return cls(
            theta_store=float(v[0]),
            theta_novel=float(v[1]),
            theta_erich=float(v[2]),
            theta_surprise=float(v[3]),
            theta_entity=float(v[4]),
            theta_temporal=float(v[5]),
            theta_decay=float(v[6]),
            w_graph=float(v[7]),
            w_embed=float(v[8]),
            w_recency=float(v[9]),
        )

    def to_vector(self) -> tuple[float, ...]:
        return (self.theta_store, self.theta_novel, self.theta_erich,
                self.theta_surprise, self.theta_entity, self.theta_temporal,
                self.theta_decay, self.w_graph, self.w_embed, self.w_recency)


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
    params: MemoryParamsV4,
    stored_embeddings: list[np.ndarray],
    max_entities_seen: int,
) -> float:
    """Task-agnostic importance score — same logic as V3."""
    if params.theta_novel == 0.0 and params.theta_erich == 0.0 and params.theta_surprise == 0.0:
        return 0.0

    obs_emb = embed_observation(obs)

    if params.theta_novel > 0.0 and stored_embeddings:
        recent = stored_embeddings[-_NOVELTY_WINDOW:]
        max_sim = max(_cosine_sim(obs_emb, e) for e in recent)
        novelty = max(0.0, 1.0 - max_sim)
    else:
        novelty = 1.0

    if params.theta_erich > 0.0:
        n_entities = len(extract_entities(obs))
        entity_richness = min(n_entities / max(max_entities_seen, 1), 1.0)
    else:
        entity_richness = 0.0

    if params.theta_surprise > 0.0 and len(stored_embeddings) >= 2:
        recent = stored_embeddings[-_NOVELTY_WINDOW:]
        mean_emb = np.mean(recent, axis=0)
        dim = max(len(obs_emb), 1)
        surprise = min(float(np.linalg.norm(obs_emb - mean_emb)) / (dim ** 0.5), 1.0)
    else:
        surprise = 0.0

    return (params.theta_novel * novelty
            + params.theta_erich * entity_richness
            + params.theta_surprise * surprise)


def _bayesian_importance(
    entity: str,
    current_step: int,
    mention_count: dict[str, int],
    last_step: dict[str, int],
    n_entities: int,
    theta_decay: float,
) -> float:
    """
    Bayesian-smoothed entity importance with optional temporal decay.

    importance = (count + alpha) / (total + beta * n_entities)
               * exp(-theta_decay * delta_t)

    This is more stable than count/total at small counts (early episode)
    and allows the optimizer to express "recent entities matter more"
    by setting theta_decay > 0.
    """
    count = mention_count.get(entity, 0)
    total = sum(mention_count.values())
    denom = total + _BETA_PRIOR * max(n_entities, 1)
    base_importance = (count + _ALPHA_PRIOR) / denom if denom > 0 else 0.0

    if theta_decay > 0.0:
        delta_t = max(0, current_step - last_step.get(entity, current_step))
        decay = math.exp(-theta_decay * delta_t)
        return base_importance * decay

    return base_importance


class GraphMemoryV4:
    """
    GraphMemory with learned importance scoring + Bayesian entity decay (Proposals B+C).

    This is the full parameterized system with 10D theta.
    Storage: importance-scored gate + Bayesian entity importance with decay.
    Retrieval: learnable weights from MemoryParamsV4.

    Drop-in compatible with the standard memory interface.
    """

    def __init__(self, params: MemoryParamsV4 | None = None) -> None:
        self._graph = nx.DiGraph()
        self._params = params or MemoryParamsV4()
        self._entity_mention_count: dict[str, int] = {}
        self._entity_last_step: dict[str, int] = {}
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

        # --- Importance-based storage gate (from V3) ---
        importance = _compute_importance(
            event.observation,
            params,
            self._stored_embeddings,
            self._max_entities_seen,
        )
        all_weights_zero = (
            params.theta_novel == 0.0
            and params.theta_erich == 0.0
            and params.theta_surprise == 0.0
        )
        if not all_weights_zero and importance <= params.theta_store:
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
        self._stored_embeddings.append(embedding)
        if len(self._stored_embeddings) > _NOVELTY_WINDOW * 2:
            self._stored_embeddings = self._stored_embeddings[-_NOVELTY_WINDOW:]

        # --- Entity nodes with Bayesian importance + decay (V4 change) ---
        entities = extract_entities(event.observation)
        self._max_entities_seen = max(self._max_entities_seen, len(entities))

        # Update mention counts and last-seen step
        for entity_name in entities:
            self._entity_mention_count[entity_name] = (
                self._entity_mention_count.get(entity_name, 0) + 1
            )
            self._entity_last_step[entity_name] = event.step

        n_unique_entities = len(self._entity_mention_count)

        for entity_name in entities:
            ent_importance = _bayesian_importance(
                entity_name,
                event.step,
                self._entity_mention_count,
                self._entity_last_step,
                n_unique_entities,
                params.theta_decay,
            )
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
            "theta_decay": self._params.theta_decay,
        }

    def clear(self) -> None:
        self._graph.clear()
        self._entity_mention_count.clear()
        self._entity_last_step.clear()
        self._stored_embeddings.clear()
        self._max_entities_seen = 0
