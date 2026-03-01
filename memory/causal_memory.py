"""
CausalMemory — memory that tracks causal chains: event → action → outcome.

Standard memory systems store events in isolation. CausalMemory explicitly links
events that have a causal relationship: "picking up the orange key" caused
"the north door opened". These chains are the most retrieval-useful structure for
sequential decision-making tasks.

Causal link construction:
  - A causal link is formed when action A at step T produces a meaningful outcome O at step T+1.
  - "Meaningful outcomes": door opened, key picked up, NPC hint received.
  - Links are stored as (cause_event, action, effect_event) triples.

Retrieval:
  - When the agent sees a door, retrieve all causal chains ending with "open door" for that color.
  - When the agent sees a key, retrieve chains where picking up that key type was useful.
  - Fall back to recency-based retrieval for unmatched queries.

Storage:
  - Raw event buffer (last 40 events) for recency fallback.
  - Causal chain pool (unbounded, deduped).
  - Hint facts (persistent, never evicted) — same as EpisodicSemantic.

Thesis motivation: tests whether explicitly encoding the "why" (causal chains) is more
useful than just the "what" (event sequences) for long-horizon tasks with delayed rewards.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass

from .entity_extraction import extract_entities
from .event import Event

_NPC_MARKERS = ("says:", "guard says", "sage says", "see a sign:")
_DOOR_RE = re.compile(r"\b(north|east|south|west|red|blue|green|yellow|orange|purple|cyan|magenta|white)\b.*door", re.I)
_KEY_RE = re.compile(r"\b(red|blue|green|yellow|orange|purple|cyan|magenta|white)\b.*key", re.I)
_OPENED_RE = re.compile(r"opened|open.*door|door.*open", re.I)
_PICKUP_RE = re.compile(r"pick.*up|picked|carrying|have.*key", re.I)
_HINT_RE = re.compile(r"see a sign.*|says:.*|hint.*", re.I)


def _is_hint(obs: str) -> bool:
    return any(m in obs.lower() for m in _NPC_MARKERS)


def _is_meaningful_outcome(obs: str) -> bool:
    """An outcome worth anchoring a causal chain to."""
    return bool(
        _OPENED_RE.search(obs)
        or _PICKUP_RE.search(obs)
        or _is_hint(obs)
    )


@dataclass
class CausalLink:
    """A (cause, action, effect) triple representing a causal relationship."""
    cause: Event
    action: str
    effect: Event

    def to_event(self) -> Event:
        """Linearize the chain into a single Event for retrieval return."""
        obs = (
            f"[Causal] At step {self.cause.step}: '{self.cause.observation[:60]}' "
            f"→ action '{self.action}' → at step {self.effect.step}: "
            f"'{self.effect.observation[:60]}'."
        )
        return Event(
            step=self.effect.step,
            observation=obs,
            action=self.action,
            is_hint=self.cause.is_hint or self.effect.is_hint,
        )


class CausalMemory:
    """
    Memory that stores and retrieves causal event chains.
    Implements the standard 4-method memory interface.
    """

    def __init__(self, raw_buffer_size: int = 40) -> None:
        self._raw: deque[Event] = deque(maxlen=raw_buffer_size)
        self._chains: list[CausalLink] = []
        self._hints: list[Event] = []
        self._seen_hints: set[str] = set()
        self._prev_event: Event | None = None

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def add_event(self, event: Event, episode_seed: int | None = None) -> None:
        obs = event.observation

        # Always store hints persistently
        if _is_hint(obs) and obs not in self._seen_hints:
            self._seen_hints.add(obs)
            self._hints.append(Event(
                step=event.step,
                observation=obs,
                action=event.action,
                is_hint=event.is_hint,
            ))

        # Form causal link if prev event's action caused a meaningful outcome
        if self._prev_event is not None and _is_meaningful_outcome(obs):
            link = CausalLink(
                cause=self._prev_event,
                action=self._prev_event.action,
                effect=event,
            )
            self._chains.append(link)

        self._raw.append(event)
        self._prev_event = event

    def get_relevant_events(
        self,
        observation: str,
        current_step: int,
        k: int = 8,
    ) -> list[Event]:
        """
        Return hints + matching causal chains + recent raw events.
        """
        result: list[Event] = []
        seen_steps: set[int] = set()

        def _add(e: Event) -> None:
            if e.step not in seen_steps and len(result) < k:
                seen_steps.add(e.step)
                result.append(e)

        # Always include persistent hints
        for h in self._hints:
            _add(h)

        # Find relevant causal chains
        chains = self._match_chains(observation)
        for link in chains:
            _add(link.cause)
            _add(link.effect)

        # Fill remaining with recent raw events
        for e in reversed(self._raw):
            _add(e)

        return result[:k]

    def clear(self) -> None:
        self._raw.clear()
        self._chains.clear()
        self._hints.clear()
        self._seen_hints.clear()
        self._prev_event = None

    def get_stats(self) -> dict:
        return {
            "n_events": len(self._raw) + len(self._hints),
            "n_entities": 0,
            "n_nodes": len(self._raw) + len(self._hints),
            "n_edges": len(self._chains),
            "n_chains": len(self._chains),
            "n_hints": len(self._hints),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _match_chains(self, observation: str) -> list[CausalLink]:
        """Find causal chains whose effect matches the current observation context."""
        obs_lo = observation.lower()
        matched: list[CausalLink] = []

        # Extract color tokens from current observation for targeted matching
        door_m = _DOOR_RE.search(observation)
        key_m = _KEY_RE.search(observation)
        current_color = None
        if door_m:
            current_color = door_m.group(1).lower()
        elif key_m:
            current_color = key_m.group(1).lower()

        for link in self._chains:
            effect_lo = link.effect.observation.lower()
            cause_lo = link.cause.observation.lower()
            # Match: current query is about a door → find chains that opened a door
            if "door" in obs_lo and "door" in effect_lo:
                if current_color is None or current_color in cause_lo or current_color in effect_lo:
                    matched.append(link)
            # Match: current query is about picking up a key → find chains where key led to outcome
            elif "key" in obs_lo and "key" in cause_lo:
                if current_color is None or current_color in cause_lo:
                    matched.append(link)
            # Match: chain involves a hint
            elif link.cause.is_hint or link.effect.is_hint:
                matched.append(link)

        # Return most recent chains first
        return sorted(matched, key=lambda l: -l.effect.step)
