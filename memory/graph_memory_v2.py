"""
GraphMemory V2 — Proposal A: Learnable Retrieval Weights.

Extends the original GraphMemory by making retrieval weights part of theta.

Original GraphMemory hardcodes w_graph=1.5, w_embed=1.0, w_recency=0.2 in
get_relevant_events(). These values are never touched by the optimizer.
This means the optimizer can only tune how memory is *constructed* (what gets
stored and which entities are promoted), but never how it is *retrieved*.

V2 adds three new learnable fields to MemoryParams:
    w_graph   — weight on the graph traversal signal (0..4)
    w_embed   — weight on TF-IDF cosine similarity (0..4)
    w_recency — weight on temporal recency 1/(1+Δt) (0..4)

The optimizer (CMA-ES) now jointly tunes all 6 dimensions:
    θ = (θ_store, θ_entity, θ_temporal, w_graph, w_embed, w_recency)

On a hint-heavy long-horizon task the optimizer will learn:
    w_graph ↑ (graph traversal is the reliable signal)
    w_recency ↓ (hints are old; penalising recency hurts)

On a recency-dominant task (navigation) it will learn the opposite.
This is purely data-driven — no task knowledge encoded anywhere.

Backward compatibility: MemoryParamsV2 defaults match the original hardcoded
values, so existing experiments that pass MemoryParamsV2 with defaults are
identical to original GraphMemory behaviour.

Original file preserved at: memory/graph_memory.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

import networkx as nx

from .embedding import embed_observation
from .entity_extraction import extract_entities
from .event import Event


def _event_node(step: int) -> str:
    return f"event_{step}"


@dataclass
class MemoryParamsV2:
    """
    6D learnable memory parameters.

    First 3 dims (inherited from original):
        theta_store    — Bernoulli probability of storing an event [0, 1]
        theta_entity   — entity importance threshold [0, 1]
        theta_temporal — Bernoulli probability of adding a temporal edge [0, 1]

    New dims (retrieval weights, clamped to [0, 4]):
        w_graph   — weight on graph traversal signal
        w_embed   — weight on TF-IDF cosine similarity
        w_recency — weight on recency score 1/(1+delta_step)

    Default values match the original hardcoded constants, so this is a
    drop-in replacement with identical default behavior.
    """
    theta_store: float = 1.0
    theta_entity: float = 0.0
    theta_temporal: float = 1.0
    w_graph: float = 1.5
    w_embed: float = 1.0
    w_recency: float = 0.2
    mode: Literal["fixed", "learnable"] = "learnable"

    def __post_init__(self) -> None:
        if self.mode == "fixed":
            return
        self.theta_store = max(0.0, min(1.0, self.theta_store))
        self.theta_entity = max(0.0, min(1.0, self.theta_entity))
        self.theta_temporal = max(0.0, min(1.0, self.theta_temporal))
        # Retrieval weights are positive reals; clamp to [0, 4]
        self.w_graph = max(0.0, min(4.0, self.w_graph))
        self.w_embed = max(0.0, min(4.0, self.w_embed))
        self.w_recency = max(0.0, min(4.0, self.w_recency))

    @classmethod
    def from_vector(cls, v: list[float] | tuple[float, ...]) -> "MemoryParamsV2":
        """Construct from a flat 6-element vector produced by CMA-ES."""
        if len(v) < 6:
            raise ValueError(f"Expected 6 values, got {len(v)}")
        return cls(
            theta_store=float(v[0]),
            theta_entity=float(v[1]),
            theta_temporal=float(v[2]),
            w_graph=float(v[3]),
            w_embed=float(v[4]),
            w_recency=float(v[5]),
        )

    def to_vector(self) -> tuple[float, ...]:
        """Serialize to a flat tuple for logging/saving."""
        return (self.theta_store, self.theta_entity, self.theta_temporal,
                self.w_graph, self.w_embed, self.w_recency)


def _memory_rng(episode_seed: int, step: int) -> random.Random:
    """Reproducible per-(episode, step) RNG — same logic as original."""
    h = (episode_seed * 10000 + step) % (2 ** 32)
    return random.Random(h)


class GraphMemoryV2:
    """
    GraphMemory with learnable retrieval weights (Proposal A).

    Storage logic is identical to the original GraphMemory.
    The only change: get_relevant_events() uses params.w_graph,
    params.w_embed, params.w_recency instead of hardcoded constants.

    Drop-in compatible with the standard memory interface:
        add_event(event, episode_seed) -> None
        get_relevant_events(observation, current_step, k) -> list[Event]
        clear() -> None
        get_stats() -> dict
    """

    def __init__(self, params: MemoryParamsV2 | None = None) -> None:
        self._graph = nx.DiGraph()
        self._params = params or MemoryParamsV2()
        self._entity_mention_count: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def add_event(self, event: Event, episode_seed: int | None = None) -> None:
        """Identical storage logic to original GraphMemory."""
        params = self._params
        if params.mode == "fixed":
            self._add_event_fixed(event)
            return

        rng = _memory_rng(episode_seed if episode_seed is not None else 0, event.step)
        if rng.random() > params.theta_store:
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

        entities = extract_entities(event.observation)
        for entity_name in entities:
            self._entity_mention_count[entity_name] = (
                self._entity_mention_count.get(entity_name, 0) + 1
            )
        total_mentions = sum(self._entity_mention_count.values())
        for entity_name in entities:
            count = self._entity_mention_count.get(entity_name, 0)
            importance = (count / total_mentions) if total_mentions > 0 else 0.0
            if importance > params.theta_entity:
                if not self._graph.has_node(entity_name):
                    self._graph.add_node(entity_name, type="entity", name=entity_name)
                self._graph.add_edge(node_id, entity_name, edge_type="mentions")
                self._graph.add_edge(entity_name, node_id, edge_type="mentioned_in")

        if event.step > 0:
            prev_id = _event_node(event.step - 1)
            if self._graph.has_node(prev_id) and rng.random() < params.theta_temporal:
                self._graph.add_edge(prev_id, node_id, edge_type="temporal")

    def _add_event_fixed(self, event: Event) -> None:
        """Original fixed behavior: store everything."""
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
    # Retrieval — KEY CHANGE: uses params.w_graph/w_embed/w_recency
    # ------------------------------------------------------------------

    def get_relevant_events(
        self,
        observation: str,
        current_step: int,
        k: int = 8,
    ) -> list[Event]:
        """
        Retrieval using *learnable* weights from MemoryParamsV2.
        w_graph, w_embed, w_recency are part of theta and optimized by CMA-ES.
        """
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
            "w_graph": self._params.w_graph,
            "w_embed": self._params.w_embed,
            "w_recency": self._params.w_recency,
        }

    def clear(self) -> None:
        self._graph.clear()
        self._entity_mention_count.clear()
