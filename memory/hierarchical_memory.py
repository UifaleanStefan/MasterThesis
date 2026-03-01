"""
HierarchicalMemory — multi-resolution memory with three distinct levels.

Level 0 — Raw buffer (last 20 events): fine-grained, recent, detailed.
Level 1 — Episode summaries (compressed every 25 steps): medium-term, entity-focused.
Level 2 — Long-term facts (persistent across the episode): hints, first-seen entities.

Each level has its own retention policy:
  - Level 0: sliding window, evicts by age.
  - Level 1: bounded pool (max 10 summaries), evicts lowest-importance when full.
  - Level 2: never evicts; only stores unique, high-value facts.

Retrieval merges all three levels: long-term facts first, then recent raw events,
then summaries if slots remain. This mirrors multi-resolution human memory where
facts are immediately accessible, recency provides context, and summaries bridge
the gap for medium-term history.

Thesis motivation: tests whether multi-resolution storage outperforms single-level
systems on tasks with both short-horizon (navigation) and long-horizon (recall hints)
memory requirements.
"""

from __future__ import annotations

from collections import deque

from .entity_extraction import extract_entities
from .event import Event

_NPC_MARKERS = ("says:", "guard says", "sage says", "see a sign:")
_MAX_SUMMARIES = 10
_RAW_BUFFER_SIZE = 20
_SUMMARIZE_EVERY = 25


def _is_hint(obs: str) -> bool:
    lo = obs.lower()
    return any(m in lo for m in _NPC_MARKERS)


def _importance(event: Event) -> float:
    """Heuristic importance for Level-1 summary eviction."""
    score = 0.0
    if _is_hint(event.observation):
        score += 5.0
    entities = extract_entities(event.observation)
    score += len(entities) * 1.0
    if "opened" in event.observation.lower():
        score += 3.0
    return score


def _summarize(events: list[Event]) -> Event:
    """Compress a list of events into a single summary Event."""
    if not events:
        raise ValueError("Cannot summarize empty event list")
    steps = [e.step for e in events]
    entities: set[str] = set()
    hint_text: str | None = None
    last_action = events[-1].action

    for e in events:
        entities.update(extract_entities(e.observation))
        if _is_hint(e.observation) and hint_text is None:
            hint_text = e.observation

    entity_str = ", ".join(sorted(entities)) if entities else "nothing notable"
    hint_part = f" NPC hint: {hint_text}." if hint_text else ""
    obs = (
        f"[Summary steps {min(steps)}-{max(steps)}]: "
        f"saw {entity_str}.{hint_part} Last action: {last_action}."
    )
    is_h = any(e.is_hint for e in events)
    return Event(step=max(steps), observation=obs, action=last_action, is_hint=is_h)


class HierarchicalMemory:
    """
    Three-level memory: raw (L0) → summaries (L1) → long-term facts (L2).
    Implements the standard 4-method memory interface.
    """

    def __init__(
        self,
        raw_size: int = _RAW_BUFFER_SIZE,
        max_summaries: int = _MAX_SUMMARIES,
        summarize_every: int = _SUMMARIZE_EVERY,
    ) -> None:
        self._raw_size = raw_size
        self._max_summaries = max_summaries
        self._summarize_every = summarize_every

        # Level 0: raw sliding window
        self._raw: deque[Event] = deque(maxlen=raw_size)
        # Level 1: episode summaries (bounded importance pool)
        self._summaries: list[Event] = []
        # Level 2: long-term facts (persistent, never evicted)
        self._long_term: list[Event] = []

        # Pending events accumulating toward next summary
        self._pending: list[Event] = []
        self._seen_hints: set[str] = set()
        self._seen_entities: set[str] = set()

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def add_event(self, event: Event, episode_seed: int | None = None) -> None:
        obs = event.observation

        # Level 2: extract long-term facts immediately
        if _is_hint(obs) and obs not in self._seen_hints:
            self._seen_hints.add(obs)
            fact = Event(
                step=event.step,
                observation=obs,
                action=event.action,
                is_hint=event.is_hint,
            )
            self._long_term.append(fact)

        for ent in extract_entities(obs):
            if ent not in self._seen_entities:
                self._seen_entities.add(ent)
                fact_obs = f"[Fact] First saw {ent} at step {event.step}."
                self._long_term.append(Event(step=event.step, observation=fact_obs, action=event.action))

        # Level 0: add to raw sliding window
        self._raw.append(event)

        # Level 1: accumulate toward next summary
        self._pending.append(event)
        if len(self._pending) >= self._summarize_every:
            self._flush_summary()

    def get_relevant_events(
        self,
        observation: str,
        current_step: int,
        k: int = 8,
    ) -> list[Event]:
        """
        Merge all three levels: L2 facts first, then L0 recent raw, then L1 summaries.
        Deduplicate by step index.
        """
        result: list[Event] = []
        seen_steps: set[int] = set()

        def _add(e: Event) -> None:
            if e.step not in seen_steps and len(result) < k:
                seen_steps.add(e.step)
                result.append(e)

        # Long-term facts always first
        for e in self._long_term:
            _add(e)

        # Recent raw events (most recent first within remaining slots)
        for e in reversed(self._raw):
            _add(e)

        # Fill remaining slots with summaries (most recent first)
        for s in reversed(self._summaries):
            _add(s)

        return result[:k]

    def clear(self) -> None:
        self._raw.clear()
        self._summaries.clear()
        self._long_term.clear()
        self._pending.clear()
        self._seen_hints.clear()
        self._seen_entities.clear()

    def get_stats(self) -> dict:
        total = len(self._raw) + len(self._summaries) + len(self._long_term)
        return {
            "n_events": total,
            "n_entities": len(self._seen_entities),
            "n_nodes": total,
            "n_edges": 0,
            "n_raw": len(self._raw),
            "n_summaries": len(self._summaries),
            "n_long_term": len(self._long_term),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush_summary(self) -> None:
        """Compress pending events into a Level-1 summary."""
        if not self._pending:
            return
        summary = _summarize(self._pending)
        self._pending = []

        if len(self._summaries) >= self._max_summaries:
            # Evict least-important summary
            worst_idx = min(range(len(self._summaries)), key=lambda i: _importance(self._summaries[i]))
            self._summaries.pop(worst_idx)

        self._summaries.append(summary)
