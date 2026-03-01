"""
FlatMemory — sliding window baseline.

Keeps the last `window_size` events in a list. No structure, no filtering,
no entity tracking. Always returns the most recent k events.

Expected behavior: cheap token cost but loses all distant information.
Fails on QuestRoom NPC hints given 300+ steps ago.
"""

from collections import deque

from .event import Event


class FlatMemory:
    """Sliding window over raw events. The simplest possible memory baseline."""

    def __init__(self, window_size: int = 50) -> None:
        self._window_size = window_size
        self._events: deque[Event] = deque(maxlen=window_size)

    def add_event(self, event: Event, episode_seed: int | None = None) -> None:
        self._events.append(event)

    def get_relevant_events(
        self,
        observation: str,
        current_step: int,
        k: int = 8,
    ) -> list[Event]:
        """Return the k most recent events."""
        events = list(self._events)
        return events[-k:] if k > 0 else events

    def clear(self) -> None:
        self._events.clear()

    def get_stats(self) -> dict:
        return {
            "n_events": len(self._events),
            "n_entities": 0,
            "n_nodes": len(self._events),
            "n_edges": 0,
        }
