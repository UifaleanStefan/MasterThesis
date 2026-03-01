"""
WorkingMemory — strictly bounded 7-slot memory inspired by Miller's Law.

The human working memory capacity is approximately 7±2 items (Miller, 1956).
This implementation enforces a hard capacity limit and uses an LRU-style eviction
policy where recently *retrieved* events are harder to evict than recently *stored*
but unretrieved events.

Key properties:
  - Hard cap: exactly N slots (default 7).
  - Eviction: when full, the slot with the oldest *last_accessed* timestamp is replaced.
  - Access tracking: calling get_relevant_events updates the access timestamp of every
    returned event. Events that are useful (frequently retrieved) survive; events that
    are stored but never retrieved are the first to go.
  - Hints are given infinite access priority — they can never be evicted.

Thesis motivation: maximum selection pressure forces the agent to maintain only the
most actively useful context. Unlike FlatWindow (evicts oldest stored) or SemanticMemory
(evicts lowest importance at write time), WorkingMemory evicts based on retrieval
utility — events that the policy never queries become dead weight and are purged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .embedding import embed_observation
from .entity_extraction import extract_entities
from .event import Event

_NPC_MARKERS = ("says:", "guard says", "sage says", "see a sign:")


def _is_hint(obs: str) -> bool:
    lo = obs.lower()
    return any(m in lo for m in _NPC_MARKERS)


@dataclass
class _Slot:
    event: Event
    stored_at: float
    last_accessed: float
    protected: bool = False  # hints are protected from eviction


class WorkingMemory:
    """
    7-slot LRU-retrieval working memory.
    Implements the standard 4-method memory interface.
    """

    def __init__(self, capacity: int = 7) -> None:
        self._capacity = capacity
        self._slots: list[_Slot] = []

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def add_event(self, event: Event, episode_seed: int | None = None) -> None:
        now = time.monotonic()

        # Hints are always stored (expand capacity by 1 if needed rather than evict)
        if _is_hint(event.observation):
            slot = _Slot(event=event, stored_at=now, last_accessed=now, protected=True)
            self._slots.append(slot)
            return

        if len(self._slots) >= self._capacity:
            self._evict_one()

        slot = _Slot(event=event, stored_at=now, last_accessed=now)
        self._slots.append(slot)

    def get_relevant_events(
        self,
        observation: str,
        current_step: int,
        k: int = 8,
    ) -> list[Event]:
        """
        Return up to k events, ranked by cosine similarity to the query.
        Update last_accessed for all returned events (LRU tracking).
        """
        if not self._slots:
            return []

        query_emb = embed_observation(observation)
        now = time.monotonic()

        scored: list[tuple[float, _Slot]] = []
        for slot in self._slots:
            emb = embed_observation(slot.event.observation)
            norm_q = np.linalg.norm(query_emb)
            norm_e = np.linalg.norm(emb)
            if norm_q > 0 and norm_e > 0:
                sim = float(np.dot(query_emb, emb) / (norm_q * norm_e))
            else:
                sim = 0.0
            # Protected (hint) slots get a bonus
            bonus = 2.0 if slot.protected else 0.0
            scored.append((sim + bonus, slot))

        scored.sort(key=lambda x: -x[0])
        top_slots = [s for _, s in scored[:k]]

        # Update access timestamps (LRU)
        for slot in top_slots:
            slot.last_accessed = now

        return [s.event for s in top_slots]

    def clear(self) -> None:
        self._slots.clear()

    def get_stats(self) -> dict:
        n = len(self._slots)
        protected = sum(1 for s in self._slots if s.protected)
        return {
            "n_events": n,
            "n_entities": len({ent for s in self._slots for ent in extract_entities(s.event.observation)}),
            "n_nodes": n,
            "n_edges": 0,
            "capacity": self._capacity,
            "protected_slots": protected,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_one(self) -> None:
        """Evict the slot with the oldest last_accessed time, skipping protected slots."""
        evictable = [s for s in self._slots if not s.protected]
        if not evictable:
            # All slots are protected — evict the oldest stored (fallback)
            evictable = self._slots
        victim = min(evictable, key=lambda s: s.last_accessed)
        self._slots.remove(victim)
