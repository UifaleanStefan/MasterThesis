"""
ContextFormatter — converts retrieved memory events into LLM prompt sections.

The format of memory context in the LLM prompt matters for:
  1. Token efficiency: how many tokens does the context consume?
  2. Information fidelity: does the format preserve the key facts?
  3. LLM comprehension: which format does the LLM parse best?

Three format styles are implemented and compared experimentally:

FormatStyle.FLAT:
  The simplest format. Concatenates observation strings separated by newlines.
  Lowest overhead but no structure — the LLM must infer what's important.
  Token cost: ~(avg_obs_len * n_events) tokens.

  Example:
    Memory context (8 events):
    You see a sign: the orange key opens the north door.
    You are carrying nothing. You see a green key.
    ...

FormatStyle.STRUCTURED:
  Labels each event by type (hint, fact, episode) and formats with bullet points.
  Higher token overhead but clearer signal — the LLM can immediately focus on hints.
  Token cost: ~(avg_obs_len * n_events + 5 * n_events) tokens (label overhead).

  Example:
    Memory context:
    [HINT] You see a sign: the orange key opens the north door.
    [FACT] First saw orange_key at step 3.
    [EPISODE step=47] You are in a corridor. You see a blue door (requires blue key).
    ...

FormatStyle.COMPRESSED:
  Generates a single-sentence summary of all retrieved events using simple rules.
  Lowest token cost (fixed ~50 tokens regardless of n_events).
  May lose detail but maximizes token efficiency.

  Example:
    Memory summary: Hints say orange→north, yellow→east. Carrying nothing.
    Doors seen: north, east. Keys seen: orange, green, blue.

All formatters preserve hint events at the top regardless of input order.

Usage:
    formatter = ContextFormatter(style=FormatStyle.STRUCTURED)
    prompt_section = formatter.format(retrieved_events)
    # prompt_section is a string to include in the LLM user message
"""

from __future__ import annotations

import re
from enum import Enum

from memory.event import Event

_NPC_MARKERS = ("says:", "guard says", "sage says", "see a sign:")
_KEY_COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "cyan", "magenta", "white"]
_DOOR_NAMES = ["north", "east", "south", "west"]
_HINT_RE = re.compile(
    r"(?:see a sign|says?).*?(?:the\s+)?(\w+)\s+key\s+opens?\s+(?:the\s+)?(\w+)\s+door",
    re.I,
)


def _is_hint(obs: str) -> bool:
    lo = obs.lower()
    return any(m in lo for m in _NPC_MARKERS)


def _is_fact(obs: str) -> bool:
    return obs.strip().startswith("[Fact]") or obs.strip().startswith("[Summary")


class FormatStyle(Enum):
    FLAT = "flat"
    STRUCTURED = "structured"
    COMPRESSED = "compressed"


class ContextFormatter:
    """
    Formats retrieved memory events into an LLM prompt section.
    """

    def __init__(self, style: FormatStyle = FormatStyle.STRUCTURED) -> None:
        self._style = style

    def format(self, events: list[Event], max_events: int = 12) -> str:
        """Convert a list of events to a formatted prompt string."""
        if not events:
            return "Memory context: (empty)"

        # Sort: hints first, then facts, then episodic (reverse step order)
        hints = [e for e in events if _is_hint(e.observation)]
        facts = [e for e in events if _is_fact(e.observation) and not _is_hint(e.observation)]
        episodic = [e for e in events if not _is_hint(e.observation) and not _is_fact(e.observation)]
        episodic.sort(key=lambda e: -e.step)  # most recent first
        ordered = (hints + facts + episodic)[:max_events]

        if self._style == FormatStyle.FLAT:
            return self._format_flat(ordered)
        elif self._style == FormatStyle.STRUCTURED:
            return self._format_structured(ordered)
        elif self._style == FormatStyle.COMPRESSED:
            return self._format_compressed(ordered)
        else:
            return self._format_structured(ordered)

    def count_tokens(self, events: list[Event]) -> int:
        """Rough token count estimate (4 chars per token)."""
        text = self.format(events)
        return len(text) // 4

    # ------------------------------------------------------------------
    # Format implementations
    # ------------------------------------------------------------------

    def _format_flat(self, events: list[Event]) -> str:
        lines = [f"Memory context ({len(events)} events):"]
        for e in events:
            lines.append(e.observation)
        return "\n".join(lines)

    def _format_structured(self, events: list[Event]) -> str:
        lines = ["Memory context:"]
        for e in events:
            if _is_hint(e.observation):
                tag = "[HINT]"
            elif _is_fact(e.observation):
                tag = "[FACT]"
            else:
                tag = f"[EPISODE step={e.step}]"
            lines.append(f"  {tag} {e.observation}")
        return "\n".join(lines)

    def _format_compressed(self, events: list[Event]) -> str:
        """
        Compress all events into a concise summary (~50 tokens).
        Extracts: hint mappings, keys seen, doors seen, current carry state.
        """
        hint_map: dict[str, str] = {}  # key_color -> door_name
        keys_seen: set[str] = set()
        doors_seen: set[str] = set()
        carrying: str | None = None
        opened_doors: set[str] = set()

        for e in events:
            obs = e.observation.lower()

            # Parse hints
            m = _HINT_RE.search(obs)
            if m:
                hint_map[m.group(1).lower()] = m.group(2).lower()

            # Keys seen
            for color in _KEY_COLORS:
                if f"{color} key" in obs:
                    keys_seen.add(color)

            # Doors seen
            for name in _DOOR_NAMES:
                if f"{name} door" in obs or f"door" in obs:
                    for dn in _DOOR_NAMES:
                        if dn in obs and "door" in obs:
                            doors_seen.add(dn)

            # Current carry
            if "carrying" in obs and "nothing" not in obs:
                for color in _KEY_COLORS:
                    if f"{color} key" in obs:
                        carrying = color

            # Opened doors
            if "opened" in obs or "open" in obs:
                for dn in _DOOR_NAMES:
                    if dn in obs:
                        opened_doors.add(dn)

        parts: list[str] = ["Memory summary:"]
        if hint_map:
            hint_str = ", ".join(f"{k}→{v}" for k, v in hint_map.items())
            parts.append(f"Hints: {hint_str}.")
        if carrying:
            parts.append(f"Carrying: {carrying} key.")
        else:
            parts.append("Carrying: nothing.")
        if keys_seen:
            parts.append(f"Keys seen: {', '.join(sorted(keys_seen))}.")
        if doors_seen:
            parts.append(f"Doors seen: {', '.join(sorted(doors_seen))}.")
        if opened_doors:
            parts.append(f"Opened: {', '.join(sorted(opened_doors))}.")

        return " ".join(parts)
