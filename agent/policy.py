"""Simple exploration policy compatible with ToyEnvironment, GoalRoom, HardKeyDoor,
and MultiHopKeyDoor."""

import re
import random
from typing import Literal

from memory.event import Event

Action = Literal["move_north", "move_south", "move_east", "move_west", "pickup", "use_door"]

MOVES = ["move_north", "move_south", "move_east", "move_west"]
KEY_COLORS = ["red", "blue", "green"]
ALL_KEY_COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta", "white"]

# Pattern: "the X key opens the Y door"  (from MultiHopKeyDoor sign hints)
_HINT_RE = re.compile(r"the (\w+) key opens the (\w+) door", re.IGNORECASE)


class ExplorationPolicy:
    """
    Rule-based exploration policy.

    Core decision logic:
    1. Parse hint events from memory to build a key->door_name map (MultiHopKeyDoor).
    2. If at a door: use_door when carrying the matching key.
    3. If at a key: pick it up if it matches a known door we haven't opened yet,
       and we're not already carrying a better key.
    4. Otherwise: move randomly.

    The policy is *memory-dependent* for MultiHopKeyDoor: without hint observations
    in past_events, the key->door map is empty and the policy cannot act optimally.
    This is the designed-in memory dependency that makes the benchmark valid.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Hint parsing — MultiHopKeyDoor specific
    # ------------------------------------------------------------------

    def _parse_hint_events(self, past_events: list[Event]) -> dict[str, str]:
        """
        Scan past events for sign hints of the form:
          "the red key opens the north door"
        Returns a dict mapping key_color -> door_name.
        Only hint events (is_hint=True) or observations containing the hint
        pattern are considered.
        """
        key_to_door: dict[str, str] = {}
        for e in past_events:
            m = _HINT_RE.search(e.observation)
            if m:
                key_color = m.group(1).lower()
                door_name = m.group(2).lower()
                key_to_door[key_color] = door_name
        return key_to_door

    def _get_door_name_from_obs(self, observation: str) -> str | None:
        """Extract door name (north/east/south) from current observation."""
        obs = observation.lower()
        for name in ["north", "east", "south"]:
            if f"{name} door" in obs:
                return name
        return None

    def _get_key_needed_for_door(
        self, door_name: str, key_to_door: dict[str, str]
    ) -> str | None:
        """Given a door name, return the key color needed (reverse lookup of hint map)."""
        for key_color, d_name in key_to_door.items():
            if d_name == door_name:
                return key_color
        return None

    # ------------------------------------------------------------------
    # Standard helpers (ToyEnvironment / HardKeyDoor)
    # ------------------------------------------------------------------

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
        """Extract key color from current observation (a key on the ground)."""
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
        """Extract door color from current observation (for ToyEnv / HardKeyDoor)."""
        obs = observation.lower()
        for c in ALL_KEY_COLORS:
            if f"{c} door" in obs:
                return c
        return None

    # ------------------------------------------------------------------
    # Main decision method
    # ------------------------------------------------------------------

    def decide(
        self,
        observation: str,
        past_events: list[Event] | None = None,
    ) -> Action:
        """
        Works for ToyEnvironment, GoalRoom, HardKeyDoor, and MultiHopKeyDoor.

        MultiHopKeyDoor path (hint-based):
        - Parse hint events from memory to get key->door_name map.
        - At a door: if carrying the key specified in the hint for this door, use_door.
        - At a key: pick it up if it's the key needed for any unopened door and
          we're not already carrying the right key for an imminent door.

        Standard path (color-matching):
        - GoalRoom: use_door at goal.
        - At a color-labeled door: use_door if carrying matching key.
        - At a key: pick up if it matches a known door color from memory.
        """
        past = past_events or []
        obs_lower = observation.lower()

        # GoalRoom shortcut
        if "goal" in obs_lower and "door" not in obs_lower:
            return "use_door"

        carried = self._get_carried_key(observation)
        at_door = "door" in obs_lower
        key_color_here = self._get_key_color_from_obs(observation)

        # ------------------------------------------------------------------
        # MultiHopKeyDoor path: door has a name (north/east/south), not a color
        # ------------------------------------------------------------------
        door_name_here = self._get_door_name_from_obs(observation)
        if door_name_here is not None or (past and any(
            _HINT_RE.search(e.observation) for e in past
        )):
            key_to_door = self._parse_hint_events(past)

            if door_name_here is not None and key_to_door:
                needed_key = self._get_key_needed_for_door(door_name_here, key_to_door)
                if needed_key and carried == needed_key:
                    return "use_door"

            if key_color_here and key_to_door:
                # Pick up this key only if it maps to a door we need
                if key_color_here in key_to_door:
                    # Don't swap if already carrying a useful key for a different door
                    if carried is None:
                        return "pickup"
                    if carried not in key_to_door:
                        return "pickup"  # carrying a useless key — swap
                    # Both useful: prefer current
                # Distractor key — ignore
                return self._rng.choice(MOVES)

            if key_color_here and not key_to_door:
                # No hints in memory — pick up any key we're not carrying (random walk fallback)
                if not carried:
                    return "pickup"

            return self._rng.choice(MOVES)

        # ------------------------------------------------------------------
        # Standard path: ToyEnvironment / HardKeyDoor (color-labeled doors)
        # ------------------------------------------------------------------
        door_color_here = self._get_door_color_from_obs(observation)

        # Use door if at door with correct key
        if at_door and carried and door_color_here and carried == door_color_here:
            return "use_door"

        # At a key
        if key_color_here:
            if past:
                known_door_colors = self._get_all_door_colors_from_memory(past)
                if carried and known_door_colors and carried in known_door_colors:
                    if key_color_here not in known_door_colors:
                        pass  # distractor — ignore
                    else:
                        if not carried:
                            return "pickup"
                else:
                    if key_color_here in known_door_colors or not known_door_colors:
                        return "pickup"
            else:
                if not carried:
                    return "pickup"
                if self._rng.random() < 0.10:
                    return "pickup"

        return self._rng.choice(MOVES)
