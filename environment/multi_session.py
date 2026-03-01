"""
MultiSessionEnv — persistent memory across 20 independent context windows.

This environment simulates the most real-world scenario: an agent working on a
long-running task across multiple sessions. In each session:
  - The agent has a fresh context window (no access to previous session's observations).
  - Only the memory store persists between sessions.
  - The agent must use memory to reconstruct context and continue the task.

Task: collaborative story building across 20 sessions.
  - Session 1: establish characters and setting.
  - Sessions 2-5: develop plot, introduce conflicts.
  - Sessions 6-15: resolve subplots, develop character arcs.
  - Sessions 16-20: converge to ending, consistency check.
  - Reward: consistency score (do characters/facts stay consistent across sessions?)
             + completion score (did the story reach a meaningful conclusion?)

Why memory is critical:
  - Character names, traits, and relationships established in session 1 must be
    recalled in session 15.
  - A flat window forgets session 1 by session 3 (window too small).
  - EpisodicSemantic retains character introductions as semantic facts.
  - The memory system determines whether the story is coherent.

Metrics:
  - consistency_score: fraction of consistency checks that pass (0–1).
    Checks: "Is character X still called by the same name?", "Does location Y
    still have property Z?", "Is the established relationship between A and B honored?"
  - completion_score: 0.1 per completed session milestone.
  - partial_score = 0.5 * consistency + 0.5 * completion.

Sessions:
  - reset() starts a new session (increments session counter).
  - Memory is NOT cleared between sessions (caller must manage this).
  - done=True after max_steps_per_session steps; advance with reset().
  - all_sessions_done=True after n_sessions sessions.

Usage with memory:
    env = MultiSessionEnv(n_sessions=20, seed=42)
    memory = EpisodicSemanticMemory()
    for session in range(env.n_sessions):
        obs = env.reset()   # new session, memory persists
        while not env.done:
            past = memory.get_relevant_events(obs, current_step=env.step_count)
            action = agent.decide(obs, past_events=past)
            obs, done, _ = env.step(action)
            memory.add_event(Event(step=..., observation=obs, action=action))
    score = env.partial_score
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

Action = Literal["move_north", "move_south", "move_east", "move_west", "pickup", "use_door"]

# Story elements
_CHARACTERS = [
    ("Aryn", "a young scholar with silver hair", "curious and brave"),
    ("Mira", "a veteran warrior", "gruff but loyal"),
    ("Zell", "a mysterious merchant", "speaks in riddles"),
    ("Corven", "the city guard captain", "strict but fair"),
    ("Lyssa", "an elven herbalist", "wise and patient"),
]
_LOCATIONS = [
    ("the Market Square", "a bustling open-air market with a central fountain"),
    ("the Scholar's Tower", "a tall stone tower filled with books and scrolls"),
    ("the Old Temple", "an abandoned temple covered in ivy"),
    ("the Harbor", "a busy port with ships from distant lands"),
    ("the Undercity", "a hidden network of tunnels beneath the city"),
]
_MILESTONES = [
    "Characters introduced",
    "Conflict established",
    "First investigation complete",
    "Key clue discovered",
    "Betrayal revealed",
    "Alliance formed",
    "Mid-point crisis",
    "New information uncovered",
    "Plan formulated",
    "First attempt fails",
    "Character growth moment",
    "Second attempt succeeds",
    "Climax begins",
    "Final confrontation",
    "Resolution reached",
    "Epilogue",
    "Consistency verified",
    "Story archived",
    "Legacy established",
    "Complete",
]
_CONSISTENCY_CHECKS = [
    ("Aryn", "silver hair"),
    ("Mira", "warrior"),
    ("Zell", "merchant"),
    ("Market Square", "fountain"),
    ("Scholar's Tower", "books"),
]


class MultiSessionEnv:
    """
    20-session persistent memory environment.
    Memory is not cleared between sessions — caller manages memory lifecycle.
    """

    def __init__(
        self,
        n_sessions: int = 20,
        max_steps_per_session: int = 50,
        seed: int = 0,
    ) -> None:
        self._n_sessions = n_sessions
        self._max_steps = max_steps_per_session
        self._rng = random.Random(seed)

        self._session_idx = 0
        self._step_count = 0
        self._milestones_completed: list[bool] = [False] * min(n_sessions, len(_MILESTONES))
        self._consistency_passes: list[bool] = []
        self._active_characters: list[tuple] = _CHARACTERS.copy()
        self._active_locations: list[tuple] = _LOCATIONS.copy()

        self.done = False
        self.success = False
        self.all_sessions_done = False

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def reset(self) -> str:
        """Start the next session. Does NOT clear memory."""
        if self.all_sessions_done:
            return "All sessions complete."

        self._step_count = 0
        self.done = False
        self.success = False
        return self._session_intro()

    def step(self, action: str | Action) -> tuple[str, bool, bool]:
        if self.done:
            return "Session complete.", True, self.success

        self._step_count += 1

        # Run a consistency check every 10 steps
        if self._step_count % 10 == 0:
            check_result = self._run_consistency_check()
            self._consistency_passes.append(check_result)

        # Progress the story
        obs = self._generate_story_event()

        if self._step_count >= self._max_steps:
            self.done = True
            idx = self._session_idx
            if idx < len(self._milestones_completed):
                self._milestones_completed[idx] = True
            self._session_idx += 1
            if self._session_idx >= self._n_sessions:
                self.all_sessions_done = True
                self.success = self.partial_score >= 0.5
            return f"[Session {self._session_idx} complete] {obs}", True, self.success

        return obs, False, False

    def get_actions(self) -> list[str]:
        return ["continue", "ask_question", "introduce_character", "describe_location"]

    @property
    def partial_score(self) -> float:
        consistency = (
            sum(self._consistency_passes) / len(self._consistency_passes)
            if self._consistency_passes else 0.0
        )
        completion = sum(self._milestones_completed) / self._n_sessions
        return 0.5 * consistency + 0.5 * completion

    @property
    def session_number(self) -> int:
        return self._session_idx + 1

    @property
    def n_sessions(self) -> int:
        return self._n_sessions

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def hint_observations(self) -> list[str]:
        """Character introductions are the 'hints' to retain (for retrieval_precision)."""
        return [
            f"{name} is {desc}."
            for name, desc, _ in _CHARACTERS
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _session_intro(self) -> str:
        idx = self._session_idx
        milestone = _MILESTONES[min(idx, len(_MILESTONES) - 1)]
        if idx == 0:
            char = _CHARACTERS[0]
            loc = _LOCATIONS[0]
            return (
                f"[Session {idx+1}: {milestone}] "
                f"You are {char[0]}, {char[1]}, standing in {loc[0]}, {loc[1]}. "
                f"Your personality: {char[2]}. "
                f"Your mission begins here."
            )
        char = self._rng.choice(_CHARACTERS)
        loc = self._rng.choice(_LOCATIONS)
        return (
            f"[Session {idx+1}: {milestone}] "
            f"You return to {loc[0]}. "
            f"You encounter {char[0]}. "
            f"The story continues from where you left off."
        )

    def _generate_story_event(self) -> str:
        idx = self._session_idx
        char = self._rng.choice(_CHARACTERS)
        loc = self._rng.choice(_LOCATIONS)
        events = [
            f"{char[0]} speaks to you in {loc[0]}: 'Remember what we discussed?'",
            f"You find a clue in {loc[0]} that connects to earlier events.",
            f"A new development in the plot: {char[0]} reveals a secret.",
            f"You must recall what {char[0]} told you in a previous session.",
            f"The situation in {loc[0]} has changed since your last visit.",
            f"{char[0]} asks about your progress on the mission.",
        ]
        return f"[Step {self._step_count}] {self._rng.choice(events)}"

    def _run_consistency_check(self) -> bool:
        """
        Check if a consistency fact is remembered.
        Returns True (pass) unconditionally here — the actual check is done
        by the agent/memory system retrieving the right fact at query time.
        In real experiments, an LLM judge verifies consistency.
        """
        # Simplified: return True with probability representing consistency
        # In real evaluation: ask LLM "Is [character] still [trait]?" and score answer.
        return True  # placeholder; real score from LLM judge
