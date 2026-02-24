"""Phase 2: Simple exploration policy. No hardcoded navigation."""

import random
from typing import Literal

from memory.event import Event

Action = Literal["move_north", "move_south", "move_east", "move_west", "pickup", "use_door"]

MOVES = ["move_north", "move_south", "move_east", "move_west"]
KEY_COLORS = ["red", "blue", "green"]


class ExplorationPolicy:
    """
    Simple exploration: move randomly, pickup key if present, use door if have key.
    With memory: only pickup key that matches door color (from past observations).
    Without memory: pickup any key (random, often wrong).
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def _get_door_color_from_memory(self, past_events: list[Event]) -> str | None:
        """Extract door color from past observations."""
        for e in reversed(past_events):
            obs = e.observation.lower()
            if "door" in obs:
                for c in KEY_COLORS:
                    if c in obs and "door" in obs:
                        return c
        return None

    def _get_key_color_from_obs(self, observation: str) -> str | None:
        """Extract key color from current observation."""
        obs = observation.lower()
        for c in KEY_COLORS:
            if f"{c} key" in obs:
                return c
        return None

    def _get_carried_key(self, observation: str) -> str | None:
        """Extract carried key color from observation."""
        obs = observation.lower()
        for c in KEY_COLORS:
            if f"carrying a {c} key" in obs:
                return c
        return None

    def _get_door_color_from_obs(self, observation: str) -> str | None:
        """Extract door color from current observation."""
        obs = observation.lower()
        for c in KEY_COLORS:
            if f"{c} door" in obs:
                return c
        return None

    def decide(
        self,
        observation: str,
        past_events: list[Event] | None = None,
    ) -> Action:
        """
        With memory: only pickup key matching door; use door when match.
        Without memory: pickup first key seen; use door when at door with any key.
        Goal-Room: if "goal" in obs, use_door to reach goal.
        """
        past = past_events or []
        obs_lower = observation.lower()

        if "goal" in obs_lower:
            return "use_door"

        at_door = "door" in obs_lower
        at_key = any(f"{c} key" in obs_lower for c in KEY_COLORS)
        carried = self._get_carried_key(observation)
        door_color_here = self._get_door_color_from_obs(observation)
        door_color_mem = self._get_door_color_from_memory(past)
        door_color = door_color_here or door_color_mem
        key_color_here = self._get_key_color_from_obs(observation)

        # Use door when at door with matching key
        if at_door and carried and door_color_here and carried == door_color_here:
            return "use_door"

        # At a key
        if at_key and key_color_here:
            if past:
                # With memory: only pickup matching key. If we don't know door yet, pickup (explore).
                if door_color:
                    if key_color_here == door_color:
                        return "pickup"
                    if carried and carried != door_color and key_color_here == door_color:
                        return "pickup"
                else:
                    if not carried:
                        return "pickup"
            else:
                # No memory: pickup any key; rarely replace when have key (hard to recover from wrong key)
                if not carried:
                    return "pickup"
                if self._rng.random() < 0.10:
                    return "pickup"

        return self._rng.choice(MOVES)
