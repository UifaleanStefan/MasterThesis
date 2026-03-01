"""Structured graph memory: event nodes + entity nodes + embeddings."""

import random
from dataclasses import dataclass
from typing import Literal

import networkx as nx

from .embedding import embed_observation
from .entity_extraction import extract_entities
from .event import Event


def _event_node(step: int) -> str:
    return f"event_{step}"


@dataclass
class MemoryParams:
    """
    Learnable memory creation parameters.
    Default (1.0, 0.0, 1.0) gives backward-compatible behavior.
    """
    theta_store: float = 1.0
    theta_entity: float = 0.0
    theta_temporal: float = 1.0
    mode: Literal["fixed", "learnable"] = "learnable"

    def __post_init__(self) -> None:
        if self.mode == "fixed":
            return
        self.theta_store = max(0.0, min(1.0, self.theta_store))
        self.theta_entity = max(0.0, min(1.0, self.theta_entity))
        self.theta_temporal = max(0.0, min(1.0, self.theta_temporal))


def _memory_rng(episode_seed: int, step: int) -> random.Random:
    """Reproducible RNG for memory construction decisions (deterministic across runs)."""
    h = (episode_seed * 10000 + step) % (2**32)
    return random.Random(h)


class GraphMemory:
    """
    Structured graph:
    - Event nodes: event_{step} with {type, step, observation, action}
    - Entity nodes: red_key, blue_door, etc. with {type, name}
    - Temporal edges: event_t -> event_t+1
    - Event-entity edges: event <-> entity (bidirectional)
    With MemoryParams (learnable mode): optional store, entity filter, temporal edge.
    """

    def __init__(self, params: MemoryParams | None = None) -> None:
        self._graph = nx.DiGraph()
        self._params = params
        self._entity_mention_count: dict[str, int] = {}

    def add_event(self, event: Event, episode_seed: int | None = None) -> None:
        """
        1. If learnable: Bernoulli(theta_store) -> maybe skip storing.
        2. Create event node event_{step}
        3. Extract entities; if learnable, filter by importance > theta_entity (frequency-based).
        4. Add temporal edge to previous event; if learnable, Bernoulli(theta_temporal).
        """
        params = self._params
        if params is None or params.mode == "fixed":
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
            self._entity_mention_count[entity_name] = self._entity_mention_count.get(entity_name, 0) + 1
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
        """Original fixed behavior: always store, all entities, all temporal edges."""
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

    def get_graph(self) -> nx.DiGraph:
        return self._graph

    def get_all_events(self) -> list[Event]:
        """Return all events in temporal order."""
        events = []
        for node_id, data in self._graph.nodes(data=True):
            if data.get("type") == "event" and "event" in data:
                events.append(data["event"])
        events.sort(key=lambda e: e.step)
        return events

    def get_stats(self) -> dict:
        """Return graph stats for reporting."""
        n_events = sum(1 for _, d in self._graph.nodes(data=True) if d.get("type") == "event")
        n_entities = sum(1 for _, d in self._graph.nodes(data=True) if d.get("type") == "entity")
        return {
            "n_nodes": self._graph.number_of_nodes(),
            "n_edges": self._graph.number_of_edges(),
            "n_events": n_events,
            "n_entities": n_entities,
        }

    def get_relevant_events(
        self,
        observation: str,
        current_step: int,
        k: int = 8,
    ) -> list[Event]:
        """
        Uniform interface for the memory comparison framework.
        Uses learnable retrieval with default weights (equal graph + embedding, low recency).
        """
        from .retrieval import retrieve_events_learnable
        return retrieve_events_learnable(
            self._graph,
            observation,
            current_step=current_step,
            k=k,
            w_graph=1.5,
            w_embed=1.0,
            w_recency=0.2,
        )

    def clear(self) -> None:
        self._graph.clear()
        self._entity_mention_count.clear()
