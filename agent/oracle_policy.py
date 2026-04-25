"""
Oracle policy — Phase 0 of the Reflexion plan.

The hypothesis being tested: V4's MegaQuestRoom failure (reward = 0.0 despite
retrieval precision = 0.94+) is a *policy* bottleneck, not a memory one. To
falsify or confirm, we need a policy that:

  (a) consumes the same retrieved hints as ExplorationPolicy,
  (b) explores the 20×20 grid systematically rather than via random walk
      (random walk's coupon-collector cost on 400 cells is ~2400 steps,
      exceeding the 1000-step budget; snake-sweep is ~440 steps),
  (c) plans key/door visits against the hint-derived map.

If this oracle achieves reward >> 0 on MegaQuest, the bottleneck is provably
the random-walk exploration in `agent/policy.py`, and the Reflexion plan is
justified. If even this oracle returns 0.0, something deeper is broken
(env mechanics, hint parsing, etc.) and the plan needs to redirect.

Crucially the oracle uses ONLY the observation channel — no peeking at the
env's `_door_positions` / `_key_positions`. It tracks its own self-coords
by accumulating move deltas from the start of the episode.
"""

from __future__ import annotations

import random
import re
from typing import Literal

from memory.event import Event

Action = Literal[
    "move_north", "move_south", "move_east", "move_west", "pickup", "use_door"
]

_HINT_RE = re.compile(r"the (\w+) key opens the (\w+) door", re.IGNORECASE)
_DOOR_REQ_RE = re.compile(r"requires (\w+) key", re.IGNORECASE)
_DOOR_NAME_RE = re.compile(
    r"the (north|east|south|west|inner|outer) door", re.IGNORECASE
)
_KEY_COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple",
    "cyan", "magenta", "white", "silver",
    "pink", "brown", "gray", "lime", "teal",
]


class OraclePolicy:
    """
    Snake-sweep + hint-aware policy intended as a falsification check.

    Per-episode lifecycle: instantiate fresh; the policy maintains a
    self-coordinate frame (initial position = (0, 0); grid bounds are
    relative to discovered extent). It records (color, self_pos) for
    every key seen and (door_name, required_color, self_pos) for every
    door seen. After completing one full sweep it switches to a planner
    that visits the right key, then the right door, until all hinted
    doors are open or the budget runs out.
    """

    def __init__(
        self,
        seed: int | None = None,
        grid_w: int = 20,
        grid_h: int = 20,
    ) -> None:
        self._rng = random.Random(seed)
        self._grid_w = grid_w
        self._grid_h = grid_h
        self.reset_episode()

    # ------------------------------------------------------------------
    # Per-episode state
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        # Self-coordinate frame: (0, 0) at episode start; we extend bounds as
        # the snake-sweep discovers them.
        self._x = 0
        self._y = 0
        # Discovered objects in self-coord frame.
        self._key_positions: dict[str, tuple[int, int]] = {}  # color -> (x, y)
        self._door_positions: dict[str, tuple[int, int]] = {}  # name -> (x, y)
        self._door_required: dict[str, str] = {}              # name -> color
        self._opened_doors: set[str] = set()
        self._collected_keys: set[str] = set()
        # Snake-sweep state.
        self._sweep_phase = "sweep"  # "sweep" -> "plan" -> "exhausted"
        self._sweep_idx = 0
        # Plan execution.
        self._plan_path: list[Action] = []
        # Track last action for self-coord tracking.
        self._last_action: Action | None = None
        # Last obs for blocked-detection.
        self._last_obs: str | None = None
        self._last_self_pos: tuple[int, int] = (0, 0)

    # ------------------------------------------------------------------
    # Observation parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_hints(events: list[Event]) -> dict[str, str]:
        """Build color -> door_name map from past hint events."""
        m: dict[str, str] = {}
        for e in events:
            for match in _HINT_RE.finditer(e.observation):
                m[match.group(1).lower()] = match.group(2).lower()
        return m

    @staticmethod
    def _get_carried_key(obs: str) -> str | None:
        low = obs.lower()
        for c in _KEY_COLORS:
            if f"carrying the {c} key" in low or f"carrying a {c} key" in low:
                return c
        return None

    @staticmethod
    def _get_key_here(obs: str) -> str | None:
        low = obs.lower()
        for c in _KEY_COLORS:
            if f"see a {c} key" in low or f"see the {c} key" in low:
                return c
        return None

    @staticmethod
    def _get_door_at_pos(obs: str) -> tuple[str | None, str | None]:
        """Return (door_name, required_color) when standing on an unopened door."""
        name_m = _DOOR_NAME_RE.search(obs)
        req_m = _DOOR_REQ_RE.search(obs)
        name = name_m.group(1).lower() if name_m else None
        req = req_m.group(1).lower() if req_m else None
        return name, req

    # ------------------------------------------------------------------
    # Self-coordinate tracking
    # ------------------------------------------------------------------

    def _on_observe(self, obs: str) -> None:
        """
        Update self-coords given the action just taken.

        We can't always be sure a move actually moved us (env clamps at the
        boundary). Detect a no-op: same observation as the previous step.
        When we suspect a wall, *don't* update self-coords — the planner
        will then avoid recording phantom positions.
        """
        if self._last_action is None:
            self._last_obs = obs
            return

        moved = obs != self._last_obs
        if self._last_action == "move_north" and moved:
            self._y += 1
        elif self._last_action == "move_south" and moved:
            self._y -= 1
        elif self._last_action == "move_east" and moved:
            self._x += 1
        elif self._last_action == "move_west" and moved:
            self._x -= 1
        # pickup / use_door / no-op: position unchanged

        self._last_obs = obs

    def _record_observed_objects(self, obs: str) -> None:
        """Save (color, self_pos) for keys / (name, color, self_pos) for doors here."""
        pos = (self._x, self._y)
        key_here = self._get_key_here(obs)
        if key_here and key_here not in self._collected_keys:
            self._key_positions[key_here] = pos
        door_name, door_req = self._get_door_at_pos(obs)
        if door_name and door_req:
            self._door_positions[door_name] = pos
            self._door_required[door_name] = door_req

    # ------------------------------------------------------------------
    # Snake sweep
    # ------------------------------------------------------------------

    def _next_sweep_action(self) -> Action:
        """
        Visit every cell of a grid_w × grid_h grid in a snake pattern.

        Pattern (for 20x20):
          row 0: 19 north steps                       (idx 0..18)
          row 0->1: 1 east step                       (idx 19)
          row 1: 19 south steps                       (idx 20..38)
          row 1->2: 1 east step                       (idx 39)
          ...
          total = grid_h * (grid_h - 1) + (grid_w - 1) east transitions
                = 20 * 19 + 19 = 399 actions for a 20x20 grid.
        """
        # 19 vertical moves + 1 east transition per row
        per_row = self._grid_h - 1 + 1  # 19 + 1 = 20
        col = self._sweep_idx // per_row
        within = self._sweep_idx % per_row

        if col >= self._grid_w:
            # Sweep complete.
            self._sweep_phase = "plan"
            return self._next_plan_action() or self._random_move()

        self._sweep_idx += 1

        if within == per_row - 1:
            # End-of-row transition: move east (or west on last col).
            return "move_east" if col < self._grid_w - 1 else "move_north"

        # Vertical move within the row.
        going_up = (col % 2 == 0)
        return "move_north" if going_up else "move_south"

    # ------------------------------------------------------------------
    # Planner
    # ------------------------------------------------------------------

    def _path_to(self, target: tuple[int, int]) -> list[Action]:
        """Manhattan path from current self-coord to target."""
        dx = target[0] - self._x
        dy = target[1] - self._y
        path: list[Action] = []
        path.extend(["move_east"] * dx if dx > 0 else ["move_west"] * (-dx))
        path.extend(["move_north"] * dy if dy > 0 else ["move_south"] * (-dy))
        return path

    def _next_plan_action(self) -> Action | None:
        """
        Pick the next door we can address: one whose required-key location
        we know AND whose own location we know, prioritizing the closest.
        Plan: walk to key -> pickup -> walk to door -> use_door.
        """
        if self._plan_path:
            act = self._plan_path.pop(0)
            return act

        candidates: list[tuple[int, str, str]] = []  # (manhattan_cost, door_name, color)
        carried_color: str | None = None  # set later
        for name, color in self._door_required.items():
            if name in self._opened_doors:
                continue
            if name not in self._door_positions:
                continue
            if color not in self._key_positions and color not in self._collected_keys:
                continue
            cost = (
                abs(self._door_positions[name][0] - self._x)
                + abs(self._door_positions[name][1] - self._y)
            )
            candidates.append((cost, name, color))

        if not candidates:
            return None

        candidates.sort()
        _, door_name, color = candidates[0]

        # Plan path: (key -> door) if not already carrying, else direct to door.
        plan: list[Action] = []
        if color not in self._collected_keys and color in self._key_positions:
            plan.extend(self._path_to(self._key_positions[color]))
            plan.append("pickup")
        plan.extend(self._path_to(self._door_positions[door_name]))
        plan.append("use_door")

        if not plan:
            return None
        self._plan_path = plan
        return self._plan_path.pop(0)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _random_move(self) -> Action:
        return self._rng.choice(
            ["move_north", "move_south", "move_east", "move_west"]
        )

    # ------------------------------------------------------------------
    # Public decide() — same signature as ExplorationPolicy
    # ------------------------------------------------------------------

    def decide(
        self,
        observation: str,
        past_events: list[Event] | None = None,
    ) -> Action:
        # Update self-coords from previous action's effect.
        self._on_observe(observation)
        # Update what we know about this cell.
        self._record_observed_objects(observation)
        # Merge any new hints from memory into door requirements.
        hint_map = self._parse_hints(past_events or [])
        for color, door_name in hint_map.items():
            self._door_required.setdefault(door_name, color)

        carried = self._get_carried_key(observation)
        key_here = self._get_key_here(observation)
        door_name, door_req = self._get_door_at_pos(observation)

        # ---- Top-priority opportunistic actions ----

        # If standing on the right door + holding the right key: use it.
        if door_name and carried and door_req and carried == door_req:
            self._opened_doors.add(door_name)
            self._last_action = "use_door"
            return "use_door"

        # If standing on a key we need (or any key, if not carrying anything)
        # and not currently mid-plan towards a different target.
        if key_here and not carried:
            wanted = key_here in self._door_required.values()
            if wanted or self._sweep_phase == "sweep":
                self._collected_keys.add(key_here)
                # Drop our plan since pickup changes carrying state.
                self._plan_path = []
                self._last_action = "pickup"
                return "pickup"

        # ---- Otherwise: sweep, then plan ----

        if self._sweep_phase == "sweep":
            act = self._next_sweep_action()
        elif self._sweep_phase == "plan":
            act = self._next_plan_action()
            if act is None:
                # Nothing planable — go random.
                act = self._random_move()
                self._sweep_phase = "exhausted"
        else:
            act = self._random_move()

        self._last_action = act
        return act


# =============================================================================
# OmniscientOraclePolicy — env-cheating upper bound
# =============================================================================

class OmniscientOraclePolicy:
    """
    Cheating oracle for diagnosis ONLY: reads env._agent_pos / _key_positions /
    _door_positions / _door_key_map directly. This removes all observation-
    parsing and self-coord-tracking errors and gives the absolute upper bound
    of "what fraction of doors are physically reachable in 1000 steps".

    If even THIS oracle returns reward ~0, the env design itself is the
    bottleneck (1000 steps insufficient for 6 keys × 6 doors of Manhattan
    paths in a 20×20 grid). If it returns near-1, the bottleneck is
    "policy + planning under partial observability" — exactly what the
    Reflexion plan addresses.

    Crucially this is NOT a memory-using policy — it ignores `past_events`.
    It exists solely to falsify-or-confirm env solvability.
    """

    def __init__(self, env, seed: int | None = None) -> None:
        self._env = env
        self._rng = random.Random(seed)
        self._plan: list[Action] = []
        self._opened: set[str] = set()

    def reset_episode(self) -> None:
        self._plan = []
        self._opened = set()

    def _path_between(
        self,
        src: tuple[int, int],
        dst: tuple[int, int],
    ) -> list[Action]:
        dx = dst[0] - src[0]
        dy = dst[1] - src[1]
        path: list[Action] = []
        path.extend(["move_east"] * dx if dx > 0 else ["move_west"] * (-dx))
        path.extend(["move_north"] * dy if dy > 0 else ["move_south"] * (-dy))
        return path

    def _build_plan(self) -> list[Action]:
        env = self._env
        pos = env._agent_pos  # type: ignore[attr-defined]
        carried = env._carried_key  # type: ignore[attr-defined]

        # Find all unopened doors whose key is still available (or carried).
        door_pos: dict[str, tuple[int, int]] = env._door_positions  # type: ignore[attr-defined]
        door_keymap: dict[str, str] = env._door_key_map  # type: ignore[attr-defined]
        unlocked: set[str] = env._doors_unlocked  # type: ignore[attr-defined]
        key_pos: dict[str, tuple[int, int]] = env._key_positions  # type: ignore[attr-defined]
        collected: set[str] = env._collected_keys  # type: ignore[attr-defined]

        # Enumerate viable next-targets: (cost, door_name, key_color)
        candidates: list[tuple[int, str, str]] = []
        for name, color in door_keymap.items():
            if name in unlocked:
                continue
            if color in collected and carried != color:
                continue  # already collected the key but it's not carried — shouldn't happen
            if color not in collected and color not in key_pos:
                continue
            # Compute total cost: pos -> key -> door (or pos -> door if carrying).
            if carried == color:
                cost = abs(door_pos[name][0] - pos[0]) + abs(door_pos[name][1] - pos[1])
            else:
                kp = key_pos[color]
                cost = (
                    abs(kp[0] - pos[0]) + abs(kp[1] - pos[1])
                    + abs(door_pos[name][0] - kp[0])
                    + abs(door_pos[name][1] - kp[1])
                )
            candidates.append((cost, name, color))

        if not candidates:
            return ["move_north"]  # nothing to do

        candidates.sort()
        _, door_name, color = candidates[0]
        plan: list[Action] = []
        if carried != color and color in key_pos:
            plan.extend(self._path_between(pos, key_pos[color]))
            plan.append("pickup")
            plan.extend(self._path_between(key_pos[color], door_pos[door_name]))
        else:
            plan.extend(self._path_between(pos, door_pos[door_name]))
        plan.append("use_door")
        return plan

    def decide(
        self,
        observation: str,
        past_events: list[Event] | None = None,
    ) -> Action:
        # If we have a plan, follow it.
        if not self._plan:
            self._plan = self._build_plan()
        if not self._plan:
            return self._rng.choice(["move_north", "move_south", "move_east", "move_west"])

        # If the previous step opened a door, rebuild the plan.
        env = self._env
        unlocked: set[str] = env._doors_unlocked  # type: ignore[attr-defined]
        if unlocked != self._opened:
            self._opened = set(unlocked)
            self._plan = self._build_plan()

        return self._plan.pop(0)
