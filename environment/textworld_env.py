"""
TextWorldEnv — wrapper around Microsoft's TextWorld IF game engine.

TextWorld (https://github.com/microsoft/TextWorld) generates procedural interactive
fiction games: navigate rooms, collect items, unlock containers, find the goal.

Why TextWorld for the thesis:
  - Natural language observations: "The bedroom is a medium-sized room. There is a
    golden key on the maple table." — no hand-crafted vocabulary.
  - Variable difficulty (1-30 levels): level 5 = 1 room, 1 item; level 20 = 10 rooms,
    chained locks.
  - Standard RL benchmark with known baselines.
  - TF-IDF breaks here (vocabulary too large) → forces use of sentence-transformers
    or real LLM embeddings → natural transition from POC to full system.

Interface:
  - Same 4-method interface as all other environments.
  - Fallback: if textworld is not installed, provides a minimal stub that returns
    placeholder observations. This lets the rest of the codebase import without error.

Installation:
    pip install textworld

Usage:
    env = TextWorldEnv(difficulty=5, seed=42)
    obs = env.reset()
    obs, done, success = env.step("take golden key")

Note on actions:
  - TextWorld takes free-form text commands ("go north", "take key", "unlock chest").
  - This differs from our Action enum. The LLMAgent is the natural policy here.
  - For compatibility with ExplorationPolicy, a mapping from Action enum to text is provided.
"""

from __future__ import annotations

import random
from typing import Literal

Action = Literal["move_north", "move_south", "move_east", "move_west", "pickup", "use_door"]

_ACTION_TO_TEXT = {
    "move_north": "go north",
    "move_south": "go south",
    "move_east": "go east",
    "move_west": "go west",
    "pickup": "take all",
    "use_door": "unlock door with key",
}


class _TextWorldStub:
    """Minimal stub when textworld is not installed."""

    def __init__(self, difficulty: int = 5, seed: int = 0) -> None:
        self._difficulty = difficulty
        self._seed = seed
        self._rng = random.Random(seed)
        self._step_count = 0
        self._max_steps = 200
        self.done = False
        self.success = False

    def reset(self) -> str:
        self._step_count = 0
        self.done = False
        self.success = False
        return (
            f"[TextWorld stub — difficulty={self._difficulty}] "
            "You are in a room. There is a key on the table. "
            "There is a locked chest in the corner."
        )

    def step(self, action: Action) -> tuple[str, bool, bool]:
        self._step_count += 1
        if self._step_count >= self._max_steps:
            self.done = True
            return "Time is up.", True, False
        if action == "use_door":
            if self._rng.random() < 0.2:
                self.done = True
                self.success = True
                return "You open the chest! You win.", True, True
        return (
            f"[step {self._step_count}] You are still in the room. "
            f"The {self._rng.choice(['key', 'chest', 'door'])} is here.",
            False, False,
        )

    def get_actions(self) -> list[Action]:
        return ["move_north", "move_south", "move_east", "move_west", "pickup", "use_door"]

    @property
    def partial_score(self) -> float:
        return 1.0 if self.success else 0.0


class TextWorldEnv:
    """
    Wrapper around Microsoft TextWorld. Falls back to stub if not installed.

    Parameters
    ----------
    difficulty : int
        TextWorld difficulty level 1–30. Level 5: easy (1 room). Level 20: hard (10 rooms).
    seed : int
        Random seed for game generation.
    max_steps : int
        Maximum steps per episode.
    request_infos : list[str], optional
        Extra info to request from TextWorld (e.g. ["description", "inventory"]).
    """

    def __init__(
        self,
        difficulty: int = 5,
        seed: int = 0,
        max_steps: int = 200,
    ) -> None:
        self._difficulty = difficulty
        self._seed = seed
        self._max_steps = max_steps
        self._step_count = 0
        self.done = False
        self.success = False

        self._game = None
        self._env = None
        self._tw_available = self._try_init()

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def reset(self) -> str:
        self._step_count = 0
        self.done = False
        self.success = False

        if self._tw_available and self._env is not None:
            obs, _ = self._env.reset()
            return obs
        else:
            return self._stub.reset()

    def step(self, action: Action) -> tuple[str, bool, bool]:
        if self.done:
            return "Episode done.", True, self.success

        self._step_count += 1

        if self._tw_available and self._env is not None:
            text_action = _ACTION_TO_TEXT.get(action, "look")
            obs, score, done, info = self._env.step(text_action)
            self.done = done or self._step_count >= self._max_steps
            self.success = done and score > 0
            return obs, self.done, self.success
        else:
            obs, done, success = self._stub.step(action)
            self.done = done
            self.success = success
            return obs, done, success

    def step_text(self, text_command: str) -> tuple[str, bool, bool]:
        """Accept free-form text commands (for use with LLMAgent)."""
        if self._tw_available and self._env is not None:
            self._step_count += 1
            obs, score, done, info = self._env.step(text_command)
            self.done = done or self._step_count >= self._max_steps
            self.success = done and score > 0
            return obs, self.done, self.success
        return self.step("move_north")

    def get_actions(self) -> list[Action]:
        return ["move_north", "move_south", "move_east", "move_west", "pickup", "use_door"]

    @property
    def partial_score(self) -> float:
        if self._tw_available and self._env is not None:
            try:
                info = self._env.infos
                score = getattr(info, "score", 0) or 0
                max_score = getattr(info, "max_score", 1) or 1
                return score / max(1, max_score)
            except Exception:
                return 1.0 if self.success else 0.0
        return self._stub.partial_score

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _try_init(self) -> bool:
        try:
            import textworld
            import textworld.gym
            import gym

            # Generate a game
            options = textworld.GameOptions()
            options.seeds = self._seed
            options.nb_rooms = max(1, self._difficulty // 3)
            options.nb_objects = self._difficulty
            options.quest_length = max(1, self._difficulty // 2)

            game_file, _ = textworld.make(options)
            request_infos = textworld.EnvInfos(
                description=True, inventory=True, score=True, max_score=True
            )
            env_id = textworld.gym.register_game(
                game_file, request_infos=request_infos, max_episode_steps=self._max_steps
            )
            self._env = gym.make(env_id)
            return True

        except ImportError:
            print("[TextWorldEnv] textworld not installed — using stub. Run: pip install textworld")
            self._stub = _TextWorldStub(self._difficulty, self._seed)
            return False
        except Exception as e:
            print(f"[TextWorldEnv] Failed to initialize: {e} — using stub.")
            self._stub = _TextWorldStub(self._difficulty, self._seed)
            return False
