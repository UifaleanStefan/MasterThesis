"""
GraphMemoryV6 — V4 + a Reflexion-style verbal lesson buffer.

V6 extends V4 with one new event type and one new retrieval channel. The
storage gate, entity decay, and base retrieval scoring of V4 are preserved
verbatim. V4 is *never* modified; V6 inherits and extends.

What's new:
  * `MemoryParamsV6` adds two dimensions to V4's 10-D theta:
      - `w_lesson`           : retrieval weight on lesson similarity
      - `theta_lesson_decay` : exponential decay rate for lesson freshness
  * `record_lesson(lesson, episode_seed)` stores a Lesson as a graph node
    with `node_type="lesson"`. Lessons are linked to (a) the entities they
    mention (via the existing entity nodes V4 already maintains) and
    (b) the events they cite (via "derived_from" edges).
  * `get_relevant_events(...)` returns events ranked by V4's existing
    learned scoring; `get_relevant_lessons(...)` is a parallel method that
    surfaces lessons. The Reflexion policy queries both per step.
  * `get_relevant_context(observation, current_step, k_events, k_lessons)`
    returns both as one structured dict so the policy can read them
    together without two graph traversals.

Critical invariant: with `w_lesson = 0.0` AND no lessons recorded, V6
retrieval is bit-identical to V4 retrieval. The pytest suite locks this
in `tests/test_v6_v7_invariants.py`.

Reference: Reflexion (Shinn et al., NeurIPS 2023).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Literal

import numpy as np

from .embedding import embed_observation
from .event import Event
from .graph_memory_v4 import (
    GraphMemoryV4,
    MemoryParamsV4,
    _NOVELTY_WINDOW,  # noqa: F401  (kept for parity with V4)
)
from .lesson import Lesson


# =============================================================================
# Parameters
# =============================================================================


@dataclass
class MemoryParamsV6(MemoryParamsV4):
    """
    12-D theta = V4's 10D + (w_lesson, theta_lesson_decay).

    w_lesson:
        Retrieval weight applied to a lesson's score. With w_lesson = 0
        the lesson channel is fully disabled and V6 reduces to V4 exactly.
    theta_lesson_decay:
        Per-episode exponential decay rate. Lessons from N episodes ago
        contribute exp(-theta_lesson_decay * N) of their raw score. With
        theta_lesson_decay = 0 lessons never decay.
    """

    w_lesson: float = 0.0
    theta_lesson_decay: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mode == "fixed":
            return
        # w_lesson on the same scale as the other retrieval weights ([0, 4]).
        self.w_lesson = max(0.0, min(4.0, self.w_lesson))
        # decay rate in [0, 1]
        self.theta_lesson_decay = max(0.0, min(1.0, self.theta_lesson_decay))

    @classmethod
    def from_vector(cls, v):
        if len(v) < 12:
            raise ValueError(f"MemoryParamsV6 needs 12 values, got {len(v)}")
        return cls(
            theta_store=float(v[0]), theta_novel=float(v[1]),
            theta_erich=float(v[2]), theta_surprise=float(v[3]),
            theta_entity=float(v[4]), theta_temporal=float(v[5]),
            theta_decay=float(v[6]),
            w_graph=float(v[7]), w_embed=float(v[8]), w_recency=float(v[9]),
            w_lesson=float(v[10]), theta_lesson_decay=float(v[11]),
        )

    def to_vector(self) -> tuple[float, ...]:
        return (
            *super().to_vector(),
            self.w_lesson,
            self.theta_lesson_decay,
        )


# =============================================================================
# Memory class
# =============================================================================


def _lesson_node(seed: int, idx: int) -> str:
    """Stable id for a lesson node."""
    return f"lesson_{seed}_{idx}"


class GraphMemoryV6(GraphMemoryV4):
    """
    V4 + Lesson event type. All event-storage, entity, and event-retrieval
    behavior is inherited from V4 unchanged.
    """

    def __init__(self, params: MemoryParamsV6 | None = None) -> None:
        # V4 expects a MemoryParamsV4 for typing; V6 params satisfies it via
        # subclassing.
        super().__init__(params or MemoryParamsV6())
        # Per-episode tracker for assigning unique lesson ids.
        self._lessons_recorded: int = 0
        # Track the current "episode index" (number of episodes completed).
        # Lessons from older episodes get exponentially smaller weight.
        self._episode_index: int = 0
        # In-memory cache of lessons (parallel to graph storage).
        self._lessons: list[Lesson] = []
        self._lesson_embeddings: list[np.ndarray] = []
        self._lesson_episode_idx: list[int] = []  # parallel to _lessons

    # ------------------------------------------------------------------
    # Lesson lifecycle
    # ------------------------------------------------------------------

    def record_lesson(
        self,
        lesson: Lesson,
        episode_seed: int | None = None,
    ) -> None:
        """
        Persist a verbal lesson into the V6 graph. Wires edges to:
          - entity nodes the lesson mentions (so graph traversal can surface
            it when the agent sees those entities again),
          - events the lesson was derived from (provenance, optional).
        """
        if lesson is None:
            return
        idx = self._lessons_recorded
        self._lessons_recorded += 1
        seed = episode_seed if episode_seed is not None else lesson.episode_seed
        node_id = _lesson_node(seed, idx)

        embedding = embed_observation(lesson.text)
        self._graph.add_node(
            node_id,
            type="lesson",
            text=lesson.text,
            embedding=embedding,
            success_marker=lesson.success_marker,
            episode_index=self._episode_index,
            lesson=lesson,
        )

        # Link lesson to entity nodes it mentions (creating entity nodes if
        # they don't already exist — same V4 entity node format).
        for ent in lesson.relevant_entities:
            if not self._graph.has_node(ent):
                self._graph.add_node(ent, type="entity", name=ent)
            self._graph.add_edge(node_id, ent, edge_type="lesson_mentions")
            self._graph.add_edge(ent, node_id, edge_type="mentioned_in_lesson")

        # Provenance edges to source events (if those events still exist).
        for step in lesson.relevant_event_steps:
            evt_id = f"event_{step}"
            if self._graph.has_node(evt_id):
                self._graph.add_edge(node_id, evt_id, edge_type="lesson_derived_from")

        self._lessons.append(lesson)
        self._lesson_embeddings.append(embedding)
        self._lesson_episode_idx.append(self._episode_index)

    def end_episode(self) -> None:
        """
        Advance the episode counter so new lessons get higher recency
        than old ones. Call this from the run loop after each episode
        completes (after `record_lesson` has been called).
        """
        self._episode_index += 1

    def clear(self) -> None:
        """
        Reset event/entity state — but NOT lessons.

        V6 deliberately diverges from V4's `clear()` contract here. The whole
        point of the lesson buffer is to persist verbal reflections *across*
        episodes; if `clear()` wiped them, every episode would start cold and
        Reflexion couldn't learn anything across episodes.

        Use `reset_lesson_buffer()` to wipe lessons explicitly (e.g. between
        unrelated benchmark runs).
        """
        # Snapshot the lesson nodes before V4's clear nukes the graph.
        lesson_node_ids = [
            n for n, d in self._graph.nodes(data=True) if d.get("type") == "lesson"
        ]
        lesson_nodes_data = {
            n: dict(self._graph.nodes[n]) for n in lesson_node_ids
        }
        super().clear()
        # Restore lesson nodes to the freshly-cleared graph.
        for node_id, data in lesson_nodes_data.items():
            self._graph.add_node(node_id, **data)
        # Note: _lessons / _lesson_embeddings / _lesson_episode_idx /
        # _lessons_recorded / _episode_index are all preserved.

    def reset_lesson_buffer(self) -> None:
        """
        Wipe lessons + episode counter (used by tests; not by the run loop).
        """
        # Remove lesson nodes from the graph.
        to_remove = [
            n for n, d in self._graph.nodes(data=True) if d.get("type") == "lesson"
        ]
        self._graph.remove_nodes_from(to_remove)
        self._lessons.clear()
        self._lesson_embeddings.clear()
        self._lesson_episode_idx.clear()
        self._lessons_recorded = 0
        self._episode_index = 0

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_relevant_lessons(
        self,
        observation: str,
        k: int = 4,
    ) -> list[Lesson]:
        """
        Rank stored lessons by `w_lesson * cosine(query, lesson_text) *
        decay`, then return the top-k.
        """
        params: MemoryParamsV6 = self._params  # type: ignore[assignment]
        if params.w_lesson <= 0.0 or not self._lessons:
            return []

        q = embed_observation(observation)
        scores: list[tuple[float, Lesson]] = []
        for lesson, emb, ep_idx in zip(
            self._lessons, self._lesson_embeddings, self._lesson_episode_idx
        ):
            denom = (np.linalg.norm(q) * np.linalg.norm(emb)) or 1e-9
            sim = float(np.dot(q, emb) / denom)
            decay = math.exp(
                -params.theta_lesson_decay * (self._episode_index - ep_idx)
            )
            score = params.w_lesson * sim * decay
            scores.append((score, lesson))

        scores.sort(key=lambda t: -t[0])
        return [l for _, l in scores[:k]]

    def get_relevant_context(
        self,
        observation: str,
        current_step: int,
        k_events: int = 8,
        k_lessons: int = 4,
    ) -> dict:
        """
        Single-call helper for the Reflexion policy: returns both events
        (V4's existing scoring) and lessons (V6's new channel) in one dict.
        """
        events = self.get_relevant_events(
            observation, current_step=current_step, k=k_events
        )
        lessons = self.get_relevant_lessons(observation, k=k_lessons)
        return {"events": events, "lessons": lessons}

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        base = super().get_stats()
        n_lessons = sum(
            1 for _, d in self._graph.nodes(data=True) if d.get("type") == "lesson"
        )
        n_lessons_success = sum(
            1
            for _, d in self._graph.nodes(data=True)
            if d.get("type") == "lesson" and d.get("success_marker")
        )
        base.update(
            {
                "n_lessons": n_lessons,
                "n_lessons_success": n_lessons_success,
                "episode_index": self._episode_index,
                "w_lesson": self._params.w_lesson,
            }
        )
        return base
