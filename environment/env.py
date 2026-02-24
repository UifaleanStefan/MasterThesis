"""
Phase 2: Partial observability, randomized layout, multiple keys, delayed dependencies.

Agent sees only local information. No coordinates or global layout.
2 keys (red, blue), 1 door (matches one key). Grid 6x6.
"""

import random
from dataclasses import dataclass, field
from typing import Literal

Action = Literal["move_north", "move_south", "move_east", "move_west", "pickup", "use_door"]

KEY_COLORS = ["red", "blue", "green"]


@dataclass
class ToyEnvironment:
    """
    Grid environment with partial observability.
    - Agent, 2 keys (red, blue), 1 door (red or blue) at random positions
    - Only matching key opens door
    - Local-only observations
    """

    width: int = 6
    height: int = 6
    max_steps: int = 80
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset and randomize positions."""
        positions = list(
            (x, y) for x in range(self.width) for y in range(self.height)
        )
        self._rng.shuffle(positions)

        self.agent_pos = positions[0]
        self.key_positions: dict[str, tuple[int, int]] = {
            "red": positions[1],
            "blue": positions[2],
            "green": positions[3],
        }
        self.door_pos = positions[4]
        self.door_color = self._rng.choice(KEY_COLORS)
        self.carried_key: str | None = None  # "red" or "blue" or None
        self.step_count = 0
        self.done = False
        self.success = False

    def reset(self) -> str:
        """Reset with new random layout, return initial observation."""
        self._reset_state()
        return self._get_observation()

    def _get_observation(self) -> str:
        """Local-only observation. No coordinates or grid info."""
        parts = ["You are in a room."]

        if self.agent_pos == self.door_pos:
            parts.append(f"You see a {self.door_color} door.")
        else:
            for color, pos in self.key_positions.items():
                if self.agent_pos == pos and self.carried_key != color:
                    parts.append(f"You see a {color} key.")
                    break
            else:
                parts.append("You see nothing of interest.")

        if self.carried_key:
            parts.append(f"You are carrying a {self.carried_key} key.")

        return " ".join(parts)

    def step(self, action: Action) -> tuple[str, bool, bool]:
        """Execute action, return (observation, done, success)."""
        if self.done:
            return self._get_observation(), self.done, self.success

        self.step_count += 1

        if action == "pickup":
            for color, pos in self.key_positions.items():
                if self.agent_pos == pos and self.carried_key != color:
                    self.carried_key = color
                    break
        elif action == "use_door":
            if self.agent_pos == self.door_pos and self.carried_key == self.door_color:
                self.done = True
                self.success = True
                return self._get_observation(), self.done, self.success
            # Wrong key or no key: no effect
        elif action == "move_north":
            x, y = self.agent_pos
            self.agent_pos = (x, min(self.height - 1, y + 1))
        elif action == "move_south":
            x, y = self.agent_pos
            self.agent_pos = (x, max(0, y - 1))
        elif action == "move_east":
            x, y = self.agent_pos
            self.agent_pos = (min(self.width - 1, x + 1), y)
        elif action == "move_west":
            x, y = self.agent_pos
            self.agent_pos = (max(0, x - 1), y)

        if self.step_count >= self.max_steps:
            self.done = True
            self.success = False

        return self._get_observation(), self.done, self.success

    def get_actions(self) -> list[Action]:
        """Return available actions."""
        return [
            "move_north", "move_south", "move_east", "move_west",
            "pickup", "use_door"
        ]


@dataclass
class GoalRoom:
    """
    Minimal second task: 6x6 grid, one goal cell.
    Success = use_door when on goal (reach goal). No keys or doors.
    Same action space as ToyEnvironment for pipeline compatibility.
    """

    width: int = 6
    height: int = 6
    max_steps: int = 80
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._reset_state()

    def _reset_state(self) -> None:
        positions = list(
            (x, y) for x in range(self.width) for y in range(self.height)
        )
        self._rng.shuffle(positions)
        self.agent_pos = positions[0]
        self.goal_pos = positions[1]
        self.step_count = 0
        self.done = False
        self.success = False

    def reset(self) -> str:
        self._reset_state()
        return self._get_observation()

    def _get_observation(self) -> str:
        if self.agent_pos == self.goal_pos:
            return "You are in a room. You see the goal."
        return "You are in a room. You see nothing of interest."

    def step(self, action: Action) -> tuple[str, bool, bool]:
        if self.done:
            return self._get_observation(), self.done, self.success

        self.step_count += 1

        if action == "use_door":
            if self.agent_pos == self.goal_pos:
                self.done = True
                self.success = True
                return self._get_observation(), self.done, self.success
        elif action == "move_north":
            x, y = self.agent_pos
            self.agent_pos = (x, min(self.height - 1, y + 1))
        elif action == "move_south":
            x, y = self.agent_pos
            self.agent_pos = (x, max(0, y - 1))
        elif action == "move_east":
            x, y = self.agent_pos
            self.agent_pos = (min(self.width - 1, x + 1), y)
        elif action == "move_west":
            x, y = self.agent_pos
            self.agent_pos = (max(0, x - 1), y)
        # pickup is no-op

        if self.step_count >= self.max_steps:
            self.done = True
            self.success = False

        return self._get_observation(), self.done, self.success

    def get_actions(self) -> list[Action]:
        return [
            "move_north", "move_south", "move_east", "move_west",
            "pickup", "use_door"
        ]
