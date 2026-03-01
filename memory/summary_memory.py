"""
SummaryMemory — periodic compression of old events into summary nodes.

Every `summarize_every` steps, all raw events older than the recent window
are compressed into a single summary Event whose observation string contains:
  - all unique entities seen in that window
  - the most recent NPC hint (if any)
  - the last action taken

Old raw events are discarded; the summary is kept. Retrieval works on
summaries + recent raw events.

This mimics human chunking of episodic memories. Memory size stays bounded
even over very long episodes.
"""

from .entity_extraction import extract_entities
from .event import Event

_NPC_MARKERS = ("says:", "guard says", "sage says")


def _is_npc_hint(obs: str) -> bool:
    obs_l = obs.lower()
    return any(m in obs_l for m in _NPC_MARKERS)


def _make_summary(events: list[Event], window_start: int, window_end: int) -> Event:
    """Compress a window of events into one summary Event."""
    entities: list[str] = []
    seen: set[str] = set()
    npc_hint: str | None = None
    last_action = events[-1].action if events else "none"

    for e in events:
        if _is_npc_hint(e.observation):
            npc_hint = e.observation  # keep last hint text
        for ent in extract_entities(e.observation):
            if ent not in seen:
                seen.add(ent)
                entities.append(ent)

    entity_str = ", ".join(entities) if entities else "nothing notable"
    hint_str = f" NPC hint: {npc_hint}." if npc_hint else ""
    obs = (
        f"[Summary steps {window_start}-{window_end}]: "
        f"saw {entity_str}.{hint_str} "
        f"Last action: {last_action}."
    )
    # Use the midpoint step so retrieval by recency is meaningful
    mid_step = (window_start + window_end) // 2
    return Event(step=mid_step, observation=obs, action=last_action)


class SummaryMemory:
    """
    Sliding raw buffer + compressed summaries of older events.

    raw_buffer_size controls how many recent events are kept raw.
    summarize_every controls compression frequency for older events.
    """

    def __init__(
        self,
        raw_buffer_size: int = 30,
        summarize_every: int = 25,
    ) -> None:
        self._raw_buffer_size = raw_buffer_size
        self._summarize_every = summarize_every
        self._raw: list[Event] = []      # recent raw events
        self._summaries: list[Event] = []  # compressed summaries
        self._pending: list[Event] = []  # events accumulating toward next summary
        self._next_summary_at = summarize_every

    def add_event(self, event: Event, episode_seed: int | None = None) -> None:
        self._raw.append(event)
        self._pending.append(event)

        # Keep raw buffer bounded
        if len(self._raw) > self._raw_buffer_size:
            self._raw.pop(0)

        # Compress pending events into a summary when threshold reached
        if event.step >= self._next_summary_at:
            # Only summarize events that have aged out of the raw buffer
            to_compress = [e for e in self._pending if e not in self._raw]
            if to_compress:
                start = to_compress[0].step
                end = to_compress[-1].step
                summary = _make_summary(to_compress, start, end)
                self._summaries.append(summary)
            self._pending = [e for e in self._pending if e in self._raw]
            self._next_summary_at += self._summarize_every

    def get_relevant_events(
        self,
        observation: str,
        current_step: int,
        k: int = 8,
    ) -> list[Event]:
        """Return recent raw events + all summaries (summaries always included)."""
        recent = self._raw[-max(1, k // 2):]
        summaries = self._summaries
        combined = summaries + [e for e in recent if e not in summaries]
        # Deduplicate by step
        seen_steps: set[int] = set()
        unique: list[Event] = []
        for e in combined:
            if e.step not in seen_steps:
                seen_steps.add(e.step)
                unique.append(e)
        return unique[:k]

    def clear(self) -> None:
        self._raw.clear()
        self._summaries.clear()
        self._pending.clear()
        self._next_summary_at = self._summarize_every

    def get_stats(self) -> dict:
        return {
            "n_events": len(self._raw) + len(self._summaries),
            "n_entities": 0,
            "n_nodes": len(self._raw) + len(self._summaries),
            "n_edges": 0,
        }
