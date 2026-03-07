"""
GraphMemory V5 — Attention-based storage gating.

Extends V4 by adding an attention mechanism at storage time: compute attention
over the last W stored observations (cosine similarity with current embedding),
then gate storage using a combination of V4 importance and attention-weighted
relevance. Observations that strongly "attend to" recent stored content get
a bonus to be stored (contextually relevant).

Storage gate: store if (importance + alpha_attn * attention_score) > theta_store.
- importance: same as V4 (novelty, entity richness, surprise).
- attention_score: softmax(cosine_sims) then max or mean; high when current obs
  is similar to some recent stored obs (contextually relevant).
alpha_attn = 0.5 (fixed) so V5 uses same 10D params as V4 for benchmark parity.

Original files preserved: graph_memory.py (V1) through graph_memory_v4.py (V4).
"""

from __future__ import annotations

import random

import numpy as np
import networkx as nx

from .embedding import embed_observation
from .entity_extraction import extract_entities
from .event import Event
from .graph_memory_v4 import (
    MemoryParamsV4,
    _compute_importance,
    _bayesian_importance,
    _event_node,
    _memory_rng,
    _cosine_sim,
    _NOVELTY_WINDOW,
)

_ALPHA_ATTN = 0.5  # weight of attention score in storage gate


def _attention_score(obs_embedding: np.ndarray, stored_embeddings: list[np.ndarray]) -> float:
    """
    Compute attention-based relevance: softmax over cosine sims to last W stored,
    then return max attention weight. High when current obs is similar to some stored.
    """
    if not stored_embeddings:
        return 0.0
    recent = stored_embeddings[-_NOVELTY_WINDOW:]
    sims = np.array([_cosine_sim(obs_embedding, e) for e in recent], dtype=np.float64)
    # Softmax with temperature 1.0
    exp_sims = np.exp(sims - sims.max())
    weights = exp_sims / exp_sims.sum()
    return float(weights.max())


class GraphMemoryV5:
    """
    GraphMemory with V4 importance scoring plus attention-based storage gating.

    Storage: store if (importance + alpha_attn * attention_score) > theta_store.
    Entity/temporal/retrieval: same as V4. Same 4-method interface.
    """

    def __init__(self, params: MemoryParamsV4 | None = None) -> None:
        self._graph = nx.DiGraph()
        self._params = params or MemoryParamsV4()
        self._entity_mention_count: dict[str, int] = {}
        self._entity_last_step: dict[str, int] = {}
        self._stored_embeddings: list[np.ndarray] = []
        self._max_entities_seen: int = 0

    def add_event(self, event: Event, episode_seed: int | None = None) -> None:
        params = self._params
        if params.mode == "fixed":
            self._add_event_fixed(event)
            return

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
        if not all_weights_zero:
            obs_emb = embed_observation(event.observation)
            attn_score = _attention_score(obs_emb, self._stored_embeddings)
            combined = importance + _ALPHA_ATTN * attn_score
            if combined <= params.theta_store:
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

        entities = extract_entities(event.observation)
        self._max_entities_seen = max(self._max_entities_seen, len(entities))
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
        self._entity_last_step.clear()
        self._stored_embeddings.clear()
        self._max_entities_seen = 0
