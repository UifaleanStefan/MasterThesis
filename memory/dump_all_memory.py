"""
DumpAllMemory — the "no selection, just give the LLM everything" baseline.

Implements the 4-method memory contract by storing every event verbatim
and returning ALL of them at retrieval time. This is the limit case of
any selective-memory system: when the entire corpus fits in the LLM's
context window, what's the point of a selection mechanism?

Why this exists (Stage 3 Phase 2 / critique #9): The headline V4t corpus-
mode claim is "selective memory beats grid-world θ." But a stronger
baseline question is: does selective memory beat NO selection at all?
For corpora that fit in 128K tokens (FinanceBench, LongMemEval), we can
literally dump every paragraph into the LLM prompt and measure judge
score. If V4t doesn't beat this baseline, the selection itself isn't
contributing.

This baseline is FEASIBLE only on small corpora. For CUAD (~46K
paragraphs ~10M tokens) or NarrativeQA (~2.88M paragraphs ~700M tokens)
it would overflow gpt-4o-mini's 128K context. The runner uses
estimate_prompt_tokens() to skip questions where dump-everything would
overflow.

Token cost projection: at 50K tokens per question × 200 questions =
10M tokens = $1.50 per benchmark at gpt-4o-mini pricing. Expensive
compared to k=8 selective retrieval (~$0.02 per benchmark), but a
real upper bound on what we're competing against.
"""

from __future__ import annotations

from typing import Any

from memory.event import Event


class DumpAllMemory:
    """Stores every event; returns ALL events at retrieval time.

    Satisfies the standard memory contract. The `k` parameter on
    `get_relevant_events` is ignored — we return everything. For
    consistency with the existing memory protocol the events are
    returned in INSERTION ORDER (no scoring, no ranking).

    This is intentionally simple. The whole point is the absence of
    selection.
    """

    def __init__(self) -> None:
        self._events: list[Event] = []

    def add_event(self, event: Event, episode_seed: int | None = None) -> None:
        self._events.append(event)

    def get_relevant_events(
        self, observation: str, current_step: int, k: int = 8,
    ) -> list[Event]:
        # Return ALL events. The `k` parameter is intentionally ignored
        # — this is the no-selection baseline.
        return list(self._events)

    def clear(self) -> None:
        self._events = []

    def get_stats(self) -> dict[str, Any]:
        return {
            "n_events": len(self._events),
            "n_entities": 0,
            "n_nodes": len(self._events),
            "n_edges": 0,
        }
