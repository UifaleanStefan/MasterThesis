"""
MegaQuestRoom — 20×20 grid, 1000 steps, 6 doors, 4 NPCs, 10 real keys, 5 distractors.

This is the hardest grid-world environment in the thesis. It creates genuine memory
pressure across multiple dimensions:

1. Long horizon (1000 steps): hints arrive in the first 20 steps; doors may not be
   reached until step 500+. Any memory system with a finite window will forget hints.

2. NPC hints (4 NPCs, one-time each): each NPC gives a one-time clue mapping a key
   color to a named door. NPCs disappear after their hint (no second chance).

3. Many keys (10 real + 5 distractor): agent picks up one at a time. Must recall
   which key opens which door from memory (NPC hints).

4. Chained doors (6 doors): doors are numbered D1-D6 and must be opened in sequence
   (D1 before D2, etc.). Agent must track which doors it has opened.

5. Distractor observations (25% of steps): random irrelevant observations inject noise
   that can displace useful events from fixed-size memory systems.

6. Token cost real: at 1000 steps × 8 retrieved events per step, even compact memory
   systems will produce 8000 retrieval tokens per episode. This makes the efficiency
   metric meaningful.

Partial score = doors_opened / 6.

Interface: same as all other environments (reset, step, get_actions, done, success,
partial_score).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

Action = Literal["move_north", "move_south", "move_east", "move_west", "pickup", "use_door"]

_GRID = 20
_MAX_STEPS = 1000
_N_DOORS = 6
_N_REAL_KEYS = 10    # 6 needed (one per door) + 4 spares for confusion
_N_DISTRACTOR_KEYS = 5
_N_NPCS = 4

_DOOR_NAMES = ["north", "east", "south", "west", "inner", "outer"]
_ALL_COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple",
    "cyan", "magenta", "white", "silver",
    # distractor colors
    "pink", "brown", "gray", "lime", "teal",
]
_DISTRACTOR_OBS = [
    "You hear footsteps in the distance.",
    "A cool breeze passes through the corridor.",
    "You see a painting on the wall.",
    "You smell something faint.",
    "The floor creaks beneath you.",
    "You notice a shadow moving.",
    "There is graffiti on the wall.",
    "You see a torch flickering.",
    "You hear distant water dripping.",
    "The air feels heavy here.",
]


@dataclass
class MegaQuestRoom:
    """
    20×20, 1000-step, 6-door chained quest environment with 4 NPCs.
    """

    seed: int | None = None
    width: int = _GRID
    height: int = _GRID
    max_steps: int = _MAX_STEPS

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._reset_state()

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def reset(self) -> str:
        self._reset_state()
        return self._get_observation()

    def step(self, action: Action) -> tuple[str, bool, bool]:
        if self.done:
            return self._get_observation(), True, self.success

        self._step_count += 1

        # Emit distractor ~25% of steps (not at NPC hint steps)
        if self._step_count not in self._npc_steps and self._rng.random() < 0.25:
            self._pending_distractor = self._rng.choice(_DISTRACTOR_OBS)

        if action == "move_north":
            self._move(0, 1)
        elif action == "move_south":
            self._move(0, -1)
        elif action == "move_east":
            self._move(1, 0)
        elif action == "move_west":
            self._move(-1, 0)
        elif action == "pickup":
            self._try_pickup()
        elif action == "use_door":
            self._try_use_door()

        if self._step_count >= self.max_steps:
            self.done = True
            self.success = self._doors_opened >= _N_DOORS

        return self._get_observation(), self.done, self.success

    def get_actions(self) -> list[Action]:
        return ["move_north", "move_south", "move_east", "move_west", "pickup", "use_door"]

    @property
    def partial_score(self) -> float:
        return self._doors_opened / _N_DOORS

    @property
    def hint_observations(self) -> list[str]:
        """All NPC hint observation strings for this episode (for retrieval_precision tracking)."""
        return [
            f"You see {name} (an NPC): \"{name.capitalize()} says: the {color} key opens the {door} door.\""
            for name, (color, door) in self._npc_hints.items()
        ]

    # ------------------------------------------------------------------
    # Internal state
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        positions = [(x, y) for x in range(self.width) for y in range(self.height)]
        self._rng.shuffle(positions)
        idx = 0

        self._agent_pos = positions[idx]; idx += 1
        self._carried_key: str | None = None
        self._step_count = 0
        self.done = False
        self.success = False
        self._doors_opened = 0
        self._pending_distractor: str | None = None

        # Assign colors: first N_REAL_KEYS are real, rest are distractors
        colors = _ALL_COLORS.copy()
        self._rng.shuffle(colors)
        real_colors = colors[:_N_REAL_KEYS]
        distractor_colors = colors[_N_REAL_KEYS: _N_REAL_KEYS + _N_DISTRACTOR_KEYS]

        # Key positions
        self._key_positions: dict[str, tuple[int, int]] = {}
        self._collected_keys: set[str] = set()
        for color in real_colors:
            self._key_positions[color] = positions[idx]; idx += 1
        for color in distractor_colors:
            self._key_positions[color] = positions[idx]; idx += 1
        self._distractor_colors: set[str] = set(distractor_colors)

        # Door positions and key assignments
        # Each door requires one of the first 6 real colors
        door_colors = real_colors[:_N_DOORS]
        self._door_positions: dict[str, tuple[int, int]] = {}
        self._door_key_map: dict[str, str] = {}   # door_name -> required color
        self._doors_unlocked: set[str] = set()
        names = _DOOR_NAMES[:_N_DOORS]
        for name, color in zip(names, door_colors):
            self._door_positions[name] = positions[idx]; idx += 1
            self._door_key_map[name] = color

        # NPC positions and hint assignments (4 NPCs, each covers one door)
        npc_names = ["Aria", "Borin", "Cara", "Dex"]
        npc_doors = self._rng.sample(names, k=_N_NPCS)  # which doors the NPCs know about
        self._npc_positions: dict[str, tuple[int, int]] = {}
        self._npc_hints: dict[str, tuple[str, str]] = {}  # name -> (color, door)
        self._npc_spoken: set[str] = set()

        for npc_name, door_name in zip(npc_names, npc_doors):
            self._npc_positions[npc_name] = positions[idx]; idx += 1
            color = self._door_key_map[door_name]
            self._npc_hints[npc_name] = (color, door_name)

        # NPC hint delivery steps (first 20 steps, spaced)
        self._npc_steps: dict[int, str] = {}
        for i, npc_name in enumerate(npc_names):
            step = 2 + i * 4  # steps 2, 6, 10, 14
            self._npc_steps[step] = npc_name

    def _get_observation(self) -> str:
        # Distractor takes priority (one per step)
        if self._pending_distractor:
            obs = self._pending_distractor
            self._pending_distractor = None
            return obs

        # NPC hints at designated steps
        if self._step_count in self._npc_steps:
            npc_name = self._npc_steps[self._step_count]
            if npc_name not in self._npc_spoken:
                self._npc_spoken.add(npc_name)
                color, door = self._npc_hints[npc_name]
                return (
                    f"You see {npc_name} (an NPC): \"{npc_name.capitalize()} says: "
                    f"the {color} key opens the {door} door.\""
                )

        pos = self._agent_pos
        parts = []

        # Doors
        for door_name, door_pos in self._door_positions.items():
            if pos == door_pos and door_name not in self._doors_unlocked:
                req = self._door_key_map[door_name]
                parts.append(f"You see the {door_name} door (requires {req} key).")
                break

        # Keys
        if not parts:
            for color, key_pos in self._key_positions.items():
                if pos == key_pos and color not in self._collected_keys:
                    parts.append(f"You see a {color} key.")
                    break

        if not parts:
            parts.append("You are in a corridor.")

        if self._carried_key:
            parts.append(f"You are carrying the {self._carried_key} key.")
        else:
            parts.append("You are carrying nothing.")

        parts.append(f"Doors opened: {self._doors_opened}/{_N_DOORS}.")
        return " ".join(parts)

    def _move(self, dx: int, dy: int) -> None:
        x, y = self._agent_pos
        nx = max(0, min(self.width - 1, x + dx))
        ny = max(0, min(self.height - 1, y + dy))
        self._agent_pos = (nx, ny)

    def _try_pickup(self) -> None:
        pos = self._agent_pos
        for color, key_pos in self._key_positions.items():
            if pos == key_pos and color not in self._collected_keys:
                if self._carried_key is None:
                    self._carried_key = color
                    self._collected_keys.add(color)
                break

    def _try_use_door(self) -> None:
        pos = self._agent_pos
        for door_name, door_pos in self._door_positions.items():
            if pos == door_pos and door_name not in self._doors_unlocked:
                req = self._door_key_map[door_name]
                if self._carried_key == req:
                    self._doors_unlocked.add(door_name)
                    self._carried_key = None
                    self._doors_opened += 1
                    if self._doors_opened >= _N_DOORS:
                        self.done = True
                        self.success = True
                break
