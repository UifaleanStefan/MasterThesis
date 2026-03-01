"""Simple exploration policy compatible with ToyEnvironment, GoalRoom, and HardKeyDoor."""

import random
from typing import Literal

from memory.event import Event

Action = Literal["move_north", "move_south", "move_east", "move_west", "pickup", "use_door"]

MOVES = ["move_north", "move_south", "move_east", "move_west"]
KEY_COLORS = ["red", "blue", "green"]
ALL_KEY_COLORS = ["red", "blue", "green", "yellow", "purple"]


class ExplorationPolicy:
    """
    Simple exploration: move randomly, pickup key if present, use door if have key.
    With memory: only pickup key that matches door color (from past observations).
    Without memory: pickup any key (random, often wrong).
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def _get_door_color_from_memory(self, past_events: list[Event]) -> str | None:
        """Extract most recently seen door color from past observations."""
        for e in reversed(past_events):
            obs = e.observation.lower()
            if "door" in obs:
                for c in ALL_KEY_COLORS:
                    if f"{c} door" in obs:
                        return c
        return None

    def _get_all_door_colors_from_memory(self, past_events: list[Event]) -> set[str]:
        """Extract all door colors ever seen from past observations."""
        colors: set[str] = set()
        for e in past_events:
            obs = e.observation.lower()
            if "door" in obs:
                for c in ALL_KEY_COLORS:
                    if f"{c} door" in obs:
                        colors.add(c)
        return colors

    def _get_key_color_from_obs(self, observation: str) -> str | None:
        """Extract key color from current observation."""
        obs = observation.lower()
        for c in ALL_KEY_COLORS:
            if f"{c} key" in obs and "carrying" not in obs.split(f"{c} key")[0].split(".")[-1]:
                return c
        return None

    def _get_carried_key(self, observation: str) -> str | None:
        """Extract carried key color from observation."""
        obs = observation.lower()
        for c in ALL_KEY_COLORS:
            if f"carrying a {c} key" in obs:
                return c
        return None

    def _get_door_color_from_obs(self, observation: str) -> str | None:
        """Extract door color from current observation."""
        obs = observation.lower()
        for c in ALL_KEY_COLORS:
            if f"{c} door" in obs:
                return c
        return None

    def decide(
        self,
        observation: str,
        past_events: list[Event] | None = None,
    ) -> Action:
        """
        Works for ToyEnvironment, GoalRoom, and HardKeyDoor.

        - GoalRoom: if at goal, use_door.
        - At a door: use_door if carrying the matching key (color known from obs).
        - At a key: pick up if it matches a door we've seen and aren't already carrying
          the right key; avoid distractor keys when we know what we need.
        - Otherwise: move randomly.
        """
        past = past_events or []
        obs_lower = observation.lower()

        # GoalRoom shortcut
        if "goal" in obs_lower and "door" not in obs_lower:
            return "use_door"

        at_door = "door" in obs_lower
        carried = self._get_carried_key(observation)
        door_color_here = self._get_door_color_from_obs(observation)
        key_color_here = self._get_key_color_from_obs(observation)

        # Use door if at door with correct key
        if at_door and carried and door_color_here and carried == door_color_here:
            return "use_door"

        # At a key
        if key_color_here:
            if past:
                # Know which doors exist from memory
                known_door_colors = self._get_all_door_colors_from_memory(past)
                door_color_mem = self._get_door_color_from_memory(past)

                # If we're carrying the right key for a known door, don't swap
                if carried and known_door_colors and carried in known_door_colors:
                    # Already have a useful key — don't swap unless this key is also useful
                    # and we haven't matched it yet; for simplicity, skip distractors
                    if key_color_here not in known_door_colors:
                        pass  # distractor — ignore
                    else:
                        # Both useful; pick up if not already carrying something better
                        if not carried:
                            return "pickup"
                else:
                    # Not carrying anything useful yet
                    if key_color_here in known_door_colors or not known_door_colors:
                        return "pickup"
                    # Key is distractor — ignore
            else:
                # No memory: pick up any key we're not carrying
                if not carried:
                    return "pickup"
                if self._rng.random() < 0.10:
                    return "pickup"

        return self._rng.choice(MOVES)
