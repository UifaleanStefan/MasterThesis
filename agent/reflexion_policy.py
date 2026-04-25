"""
Reflexion policy — wraps any policy with a memory-driven lesson injector.

The wrapped policy never sees the lesson buffer directly. Instead the
ReflexionPolicy retrieves the top-k lessons relevant to the current
observation (via memory.get_relevant_lessons), wraps each as a synthetic
Event, and prepends them to past_events before calling the underlying
policy.decide(). The underlying policy then parses lesson text exactly as
if it were a past observation — the existing ExplorationPolicy's
"the X key opens the Y door" regex picks them up directly with no policy
changes.

This is the simplest possible Reflexion integration: one wrapper, no
forking of the rule-based policy logic.
"""

from __future__ import annotations

from typing import Any, Protocol

from memory.event import Event


class _PolicyLike(Protocol):
    def decide(self, observation: str, past_events: list[Event] | None = None) -> str: ...


class ReflexionPolicy:
    """
    Decorate any base policy with retrieved-lesson context.

    Constructor:
        base_policy : the policy that actually emits actions
                      (ExplorationPolicy, OraclePolicy, LLMAgent, ...)
        memory      : the V6+ memory instance — must expose
                      get_relevant_lessons(observation, k) -> list[Lesson]
        k_lessons   : how many lessons to surface per step (default 3)

    decide() merges retrieved lessons into past_events as synthetic Events
    with `is_hint=True` and a negative step (so they sort before real
    events without colliding with real step numbers).
    """

    def __init__(
        self,
        base_policy: _PolicyLike,
        memory: Any,
        k_lessons: int = 3,
    ) -> None:
        self._base = base_policy
        self._memory = memory
        self._k_lessons = k_lessons

    def decide(
        self,
        observation: str,
        past_events: list[Event] | None = None,
    ) -> str:
        past = list(past_events or [])
        # Skip cleanly when memory doesn't support lessons (e.g. wrapping a
        # FlatMemory by mistake).
        if not hasattr(self._memory, "get_relevant_lessons"):
            return self._base.decide(observation, past_events=past)

        lessons = self._memory.get_relevant_lessons(observation, k=self._k_lessons)
        if not lessons:
            return self._base.decide(observation, past_events=past)

        synthetic = [
            Event(
                step=-(i + 1),  # negative so they sort before all real events
                observation=lesson.text,
                action="reflect",
                is_hint=True,
            )
            for i, lesson in enumerate(lessons)
        ]
        # Prepend so hint parsing sees lessons first; underlying policy then
        # sees the latest *real* hint observations afterwards (real ones win
        # in case of ties because the hint map is built by iteration and
        # later assignments overwrite earlier ones).
        merged = synthetic + past
        return self._base.decide(observation, past_events=merged)
