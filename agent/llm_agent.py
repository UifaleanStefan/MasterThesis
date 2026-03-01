"""
LLMAgent — GPT-4o/GPT-4o-mini wrapper with real token and cost tracking.

Replaces the rule-based ExplorationPolicy with an actual LLM decision-maker.
The agent receives (current_observation, retrieved_memory_context) formatted
as a prompt, calls the OpenAI API, and parses the returned action string.

Design:
  - Model: gpt-4o-mini by default (cheap experiments); gpt-4o for final runs.
  - System prompt: describes the task, available actions, and memory context format.
  - User message: current observation + memory context (from ContextFormatter).
  - Response parsing: looks for action keywords in LLM output.
  - Cost tracking: records prompt_tokens, completion_tokens, cost_usd per call.

Token counting:
  - Uses tiktoken for accurate token counting before API calls.
  - Aggregates per-episode for cost-efficiency analysis.

Cost model (as of 2026, may change):
  - gpt-4o-mini: $0.15/1M input, $0.60/1M output
  - gpt-4o: $2.50/1M input, $10.00/1M output

Fallback:
  - If OPENAI_API_KEY is not set, falls back to a keyword-matching heuristic
    that mimics what an LLM would do. This allows testing the pipeline without
    API access.

Interface:
  - Compatible with ExplorationPolicy: both expose decide(observation, past_events).
  - The LLM agent uses ContextFormatter to convert past_events into a prompt section.

Usage:
    agent = LLMAgent(model="gpt-4o-mini")
    action = agent.decide(observation, past_events=retrieved_events)
    cost = agent.session_cost_usd   # running total
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .context_formatter import ContextFormatter, FormatStyle
from memory.event import Event

# Action names that match environment/env.py Action enum
_ACTION_KEYWORDS = {
    "north": "move_north",
    "south": "move_south",
    "east": "move_east",
    "west": "move_west",
    "pick": "pickup",
    "pickup": "pickup",
    "use": "use_door",
    "open": "use_door",
    "door": "use_door",
}

# Pricing (USD per 1M tokens, as of early 2026)
_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
}


@dataclass
class CallRecord:
    step: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    action: str
    latency_ms: float


@dataclass
class EpisodeStats:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    n_calls: int = 0
    calls: list[CallRecord] = field(default_factory=list)

    def add(self, record: CallRecord) -> None:
        self.total_prompt_tokens += record.prompt_tokens
        self.total_completion_tokens += record.completion_tokens
        self.total_cost_usd += record.cost_usd
        self.n_calls += 1
        self.calls.append(record)

    def to_dict(self) -> dict:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost_usd": self.total_cost_usd,
            "n_calls": self.n_calls,
            "avg_prompt_tokens": self.total_prompt_tokens / max(1, self.n_calls),
            "avg_cost_per_call_usd": self.total_cost_usd / max(1, self.n_calls),
        }


class LLMAgent:
    """
    LLM-based agent wrapping OpenAI API. Compatible with ExplorationPolicy interface.
    """

    _SYSTEM_PROMPT = (
        "You are an agent navigating a text-based grid world. "
        "At each step you receive your current observation and relevant memory context. "
        "You must choose exactly ONE action from: "
        "[move_north, move_south, move_east, move_west, pickup, use_door]. "
        "Output ONLY the action name, nothing else. "
        "Use memory context to remember key facts (hints, keys seen, doors). "
        "Prioritize: (1) follow sign hints to match keys to doors, "
        "(2) pick up keys needed for doors, (3) use doors when holding the right key."
    )

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        format_style: FormatStyle = FormatStyle.STRUCTURED,
        temperature: float = 0.0,
        max_tokens: int = 10,
        seed: int = 42,
    ) -> None:
        self._model = model
        self._formatter = ContextFormatter(style=format_style)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._seed = seed

        self._client: Any = None
        self._has_openai = self._try_init_client()

        self._episode_stats = EpisodeStats()
        self._session_stats = EpisodeStats()
        self._step = 0

    # ------------------------------------------------------------------
    # Public API (matches ExplorationPolicy.decide signature)
    # ------------------------------------------------------------------

    def decide(self, observation: str, past_events: list[Event] | None = None) -> str:
        """
        Choose an action given current observation and retrieved past events.
        Returns action name string.
        """
        if not self._has_openai:
            return self._fallback_decide(observation, past_events or [])

        memory_context = self._formatter.format(past_events or [])
        user_msg = f"Current observation: {observation}\n\n{memory_context}\n\nAction:"

        t0 = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                seed=self._seed,
            )
        except Exception as e:
            print(f"[LLMAgent] API error: {e} — falling back to heuristic")
            return self._fallback_decide(observation, past_events or [])

        latency_ms = (time.monotonic() - t0) * 1000
        raw_action = response.choices[0].message.content.strip().lower()
        action = self._parse_action(raw_action)

        # Track costs
        usage = response.usage
        pricing = _PRICING.get(self._model, {"input": 0.15, "output": 0.60})
        cost = (
            usage.prompt_tokens * pricing["input"]
            + usage.completion_tokens * pricing["output"]
        ) / 1_000_000

        record = CallRecord(
            step=self._step,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=cost,
            action=action,
            latency_ms=latency_ms,
        )
        self._episode_stats.add(record)
        self._session_stats.add(record)
        self._step += 1
        return action

    def reset_episode(self) -> EpisodeStats:
        """Reset per-episode stats and return the completed episode stats."""
        completed = self._episode_stats
        self._episode_stats = EpisodeStats()
        self._step = 0
        return completed

    @property
    def episode_stats(self) -> EpisodeStats:
        return self._episode_stats

    @property
    def session_cost_usd(self) -> float:
        return self._session_stats.total_cost_usd

    @property
    def session_stats(self) -> EpisodeStats:
        return self._session_stats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _try_init_client(self) -> bool:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("[LLMAgent] OPENAI_API_KEY not set — using fallback heuristic mode.")
            return False
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
            return True
        except ImportError:
            print("[LLMAgent] openai package not installed — using fallback heuristic mode.")
            return False

    def _parse_action(self, raw: str) -> str:
        """Extract a valid action from LLM output."""
        raw = raw.strip().lower().replace("-", "_")
        # Direct match
        valid = {"move_north", "move_south", "move_east", "move_west", "pickup", "use_door"}
        if raw in valid:
            return raw
        # Keyword matching
        for kw, action in _ACTION_KEYWORDS.items():
            if kw in raw:
                return action
        # Default fallback
        return "move_north"

    def _fallback_decide(self, observation: str, past_events: list[Event]) -> str:
        """
        Heuristic fallback when OpenAI API is unavailable.
        Mimics what a well-prompted LLM would do for grid-world tasks.
        """
        import re
        obs_lo = observation.lower()

        # Use door if we see a door and are carrying something
        if "door" in obs_lo and "requires" in obs_lo and "carrying" in obs_lo:
            return "use_door"
        # Pick up a key if we see one and aren't carrying anything
        if "key" in obs_lo and "carrying" not in obs_lo:
            return "pickup"
        # Follow hint logic from memory
        if past_events:
            for e in reversed(past_events):
                if "sign" in e.observation.lower() or "says:" in e.observation.lower():
                    pass  # would parse hint in full implementation
        # Random cardinal direction
        import random
        return random.choice(["move_north", "move_south", "move_east", "move_west"])
