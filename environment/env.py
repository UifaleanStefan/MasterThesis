"""
Environments for learnable memory thesis experiments.

ToyEnvironment: 6x6, 3 keys, 1 door, partial observability.
GoalRoom: 6x6, reach-the-goal task.
HardKeyDoor: 10x10, 5 keys, 3 doors, distractors, 300 steps. Creates genuine memory pressure.
QuestRoom: 12x12, 500 steps, 4 chained doors, 2 NPCs giving one-time hints, distractor obs.
           Genuine long-horizon memory test: NPC hints arrive early, must be recalled 300+ steps later.
"""

import random
from dataclasses import dataclass, field
from typing import Literal

Action = Literal["move_north", "move_south", "move_east", "move_west", "pickup", "use_door"]

KEY_COLORS = ["red", "blue", "green"]
ALL_KEY_COLORS = ["red", "blue", "green", "yellow", "purple"]


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


@dataclass
class HardKeyDoor:
    """
    Hard multi-door task creating genuine memory pressure.

    - 10x10 grid, 300 max steps.
    - 5 keys: red, blue, green (real) + yellow, purple (distractors).
    - 3 doors: each requires exactly one specific real key.
    - Carrying capacity: 1 key. Must swap to get the right key.
    - Goal: open all 3 doors (partial success counted).
    - Memory matters: agent must remember which color opens which door
      across hundreds of steps, and must learn to ignore distractors.
    """

    width: int = 10
    height: int = 10
    max_steps: int = 300
    seed: int | None = None

    # Which real keys are assigned to which doors (fixed mapping, random assignment per reset)
    _REAL_KEYS: tuple = ("red", "blue", "green")
    _DISTRACTOR_KEYS: tuple = ("yellow", "purple")
    _N_DOORS: int = 3

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._reset_state()

    def _reset_state(self) -> None:
        cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        self._rng.shuffle(cells)

        # Assign positions: agent + 5 keys + 3 doors = 9 objects
        self.agent_pos = cells[0]

        self.key_positions: dict[str, tuple[int, int]] = {
            "red": cells[1],
            "blue": cells[2],
            "green": cells[3],
            "yellow": cells[4],   # distractor
            "purple": cells[5],   # distractor
        }

        # 3 doors at random positions
        self.door_positions: list[tuple[int, int]] = [cells[6], cells[7], cells[8]]

        # Randomly assign one real key to each door (shuffled each episode)
        keys_shuffled = list(self._REAL_KEYS)
        self._rng.shuffle(keys_shuffled)
        # door_key_map[i] = color of key that opens door i
        self.door_key_map: list[str] = keys_shuffled

        self.carried_key: str | None = None
        self.doors_opened: list[bool] = [False, False, False]
        self.step_count = 0
        self.done = False
        self.success = False

    def reset(self) -> str:
        self._reset_state()
        return self._get_observation()

    def _get_observation(self) -> str:
        parts = ["You are in a room."]
        pos = self.agent_pos

        # Check if at a door
        door_here: int | None = None
        for i, dpos in enumerate(self.door_positions):
            if pos == dpos and not self.doors_opened[i]:
                door_here = i
                break

        if door_here is not None:
            required_color = self.door_key_map[door_here]
            parts.append(f"You see a {required_color} door.")
        else:
            # Check if at a key (and not already carrying it)
            key_here: str | None = None
            for color, kpos in self.key_positions.items():
                if pos == kpos and self.carried_key != color:
                    key_here = color
                    break
            if key_here:
                parts.append(f"You see a {key_here} key.")
            else:
                parts.append("You see nothing of interest.")

        if self.carried_key:
            parts.append(f"You are carrying a {self.carried_key} key.")

        opened = sum(self.doors_opened)
        if opened > 0:
            parts.append(f"You have opened {opened} door(s).")

        return " ".join(parts)

    def step(self, action: Action) -> tuple[str, bool, bool]:
        if self.done:
            return self._get_observation(), self.done, self.success

        self.step_count += 1
        pos = self.agent_pos

        if action == "pickup":
            for color, kpos in self.key_positions.items():
                if pos == kpos:
                    self.carried_key = color  # swaps if already carrying
                    break

        elif action == "use_door":
            for i, dpos in enumerate(self.door_positions):
                if pos == dpos and not self.doors_opened[i]:
                    if self.carried_key == self.door_key_map[i]:
                        self.doors_opened[i] = True
                        self.carried_key = None  # key consumed
                    break  # only interact with one door per step

        elif action == "move_north":
            x, y = pos
            self.agent_pos = (x, min(self.height - 1, y + 1))
        elif action == "move_south":
            x, y = pos
            self.agent_pos = (x, max(0, y - 1))
        elif action == "move_east":
            x, y = pos
            self.agent_pos = (min(self.width - 1, x + 1), y)
        elif action == "move_west":
            x, y = pos
            self.agent_pos = (max(0, x - 1), y)

        # Success = all 3 doors opened
        if all(self.doors_opened):
            self.done = True
            self.success = True

        if self.step_count >= self.max_steps:
            self.done = True
            # Partial success: at least 1 door opened counts as partial
            self.success = any(self.doors_opened)

        return self._get_observation(), self.done, self.success

    def get_actions(self) -> list[Action]:
        return [
            "move_north", "move_south", "move_east", "move_west",
            "pickup", "use_door"
        ]

    @property
    def partial_score(self) -> float:
        """Fraction of doors opened (0.0 to 1.0). Use as reward for J(θ)."""
        return sum(self.doors_opened) / self._N_DOORS


# ---------------------------------------------------------------------------
# Distractor observations injected randomly (not tied to agent position)
# ---------------------------------------------------------------------------
_DISTRACTOR_OBS = [
    "You hear the wind howling.",
    "You see a painting on the wall.",
    "You notice some dust on the floor.",
    "You smell something faintly sweet.",
    "You hear distant footsteps.",
    "You see a torch flickering.",
    "You notice a crack in the ceiling.",
    "You hear water dripping.",
]

# Colors for QuestRoom keys/doors
_QUEST_KEY_COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]
_QUEST_REAL_KEYS = ["red", "blue", "green", "yellow"]   # 4 real keys → 4 doors
_QUEST_DISTRACTOR_KEYS = ["purple", "orange"]            # 2 distractors


@dataclass
class QuestRoom:
    """
    Hard long-horizon memory benchmark.

    - 12x12 grid, 500 max steps.
    - 4 doors in a fixed chain: Door 0 must be opened before Door 1 becomes accessible, etc.
    - 4 real keys (red, blue, green, yellow) — one per door, assigned randomly each episode.
    - 2 distractor keys (purple, orange) — open nothing.
    - 2 NPCs: NPC A reveals which key opens Door 0. NPC B reveals which key opens Door 2.
      Hints are text observations (e.g. "The guard says: the blue key opens the first door.").
      Each NPC gives its hint only the FIRST time the agent visits.
    - Random distractor observations injected ~10% of steps (noise).
    - Reward: partial_score = doors_opened / 4.
    - Interface identical to ToyEnvironment / HardKeyDoor.

    Memory pressure:
    - NPC hints arrive at random early steps; agent must recall them 200-400 steps later.
    - Door chain means agent must track *which* doors are open to know what to do next.
    - Distractors force memory systems to filter noise vs. signal.
    - 500 steps × k retrieved events = high token cost → J(θ) penalty has real bite.
    """

    width: int = 12
    height: int = 12
    max_steps: int = 500
    seed: int | None = None
    distractor_prob: float = 0.10   # probability of distractor obs on any step

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._reset_state()

    def _reset_state(self) -> None:
        cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        self._rng.shuffle(cells)

        idx = 0
        self.agent_pos: tuple[int, int] = cells[idx]; idx += 1

        # 4 real keys + 2 distractor keys
        self.key_positions: dict[str, tuple[int, int]] = {}
        for color in _QUEST_KEY_COLORS:
            self.key_positions[color] = cells[idx]; idx += 1

        # 4 doors
        self.door_positions: list[tuple[int, int]] = [cells[idx + i] for i in range(4)]; idx += 4

        # 2 NPCs
        self.npc_positions: list[tuple[int, int]] = [cells[idx], cells[idx + 1]]; idx += 2

        # Random assignment: which real key opens which door
        keys_shuffled = list(_QUEST_REAL_KEYS)
        self._rng.shuffle(keys_shuffled)
        self.door_key_map: list[str] = keys_shuffled  # door_key_map[i] = color that opens door i

        self.carried_key: str | None = None
        self.doors_opened: list[bool] = [False, False, False, False]
        self.npc_visited: list[bool] = [False, False]  # has agent received each hint?
        self.step_count: int = 0
        self.done: bool = False
        self.success: bool = False
        self._pending_distractor: str | None = None  # injected on next obs

    def reset(self) -> str:
        self._reset_state()
        return self._get_observation()

    def _door_accessible(self, door_idx: int) -> bool:
        """Door chain: door i requires all doors 0..i-1 to be open."""
        return all(self.doors_opened[:door_idx])

    def _get_observation(self) -> str:
        parts = ["You are in a room."]
        pos = self.agent_pos

        # Inject distractor with some probability (set before step, cleared after)
        if self._pending_distractor:
            parts.append(self._pending_distractor)
            self._pending_distractor = None
            return " ".join(parts)

        # NPC hint (one-time, on first visit)
        for i, npc_pos in enumerate(self.npc_positions):
            if pos == npc_pos and not self.npc_visited[i]:
                self.npc_visited[i] = True
                if i == 0:
                    key = self.door_key_map[0]
                    parts.append(f"A guard says: the {key} key opens the first door.")
                else:
                    key = self.door_key_map[2]
                    parts.append(f"A sage says: the {key} key opens the third door.")
                if self.carried_key:
                    parts.append(f"You are carrying a {self.carried_key} key.")
                return " ".join(parts)

        # Door
        for i, dpos in enumerate(self.door_positions):
            if pos == dpos and not self.doors_opened[i]:
                if self._door_accessible(i):
                    required = self.door_key_map[i]
                    parts.append(f"You see door {i + 1} (requires {required} key).")
                else:
                    parts.append(f"You see door {i + 1} (locked, previous door not open).")
                if self.carried_key:
                    parts.append(f"You are carrying a {self.carried_key} key.")
                return " ".join(parts)

        # Key
        for color, kpos in self.key_positions.items():
            if pos == kpos and self.carried_key != color:
                parts.append(f"You see a {color} key.")
                if self.carried_key:
                    parts.append(f"You are carrying a {self.carried_key} key.")
                return " ".join(parts)

        # Default
        parts.append("You see nothing of interest.")
        if self.carried_key:
            parts.append(f"You are carrying a {self.carried_key} key.")
        opened = sum(self.doors_opened)
        if opened:
            parts.append(f"You have opened {opened} door(s).")
        return " ".join(parts)

    def step(self, action: Action) -> tuple[str, bool, bool]:
        if self.done:
            return self._get_observation(), self.done, self.success

        self.step_count += 1

        # Maybe inject distractor on NEXT observation
        if self._rng.random() < self.distractor_prob:
            self._pending_distractor = self._rng.choice(_DISTRACTOR_OBS)

        pos = self.agent_pos

        if action == "pickup":
            for color, kpos in self.key_positions.items():
                if pos == kpos:
                    self.carried_key = color
                    break

        elif action == "use_door":
            for i, dpos in enumerate(self.door_positions):
                if pos == dpos and not self.doors_opened[i] and self._door_accessible(i):
                    if self.carried_key == self.door_key_map[i]:
                        self.doors_opened[i] = True
                        self.carried_key = None
                    break

        elif action == "move_north":
            x, y = pos; self.agent_pos = (x, min(self.height - 1, y + 1))
        elif action == "move_south":
            x, y = pos; self.agent_pos = (x, max(0, y - 1))
        elif action == "move_east":
            x, y = pos; self.agent_pos = (min(self.width - 1, x + 1), y)
        elif action == "move_west":
            x, y = pos; self.agent_pos = (max(0, x - 1), y)

        if all(self.doors_opened):
            self.done = True
            self.success = True
        elif self.step_count >= self.max_steps:
            self.done = True
            self.success = False

        return self._get_observation(), self.done, self.success

    def get_actions(self) -> list[Action]:
        return [
            "move_north", "move_south", "move_east", "move_west",
            "pickup", "use_door"
        ]

    @property
    def partial_score(self) -> float:
        return sum(self.doors_opened) / 4.0
