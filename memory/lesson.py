"""
Lesson primitive — verbal post-episode reflections, the second memory tier
introduced in V6 (the Reflexion-augmented variant of V4).

A `Lesson` is what V4 lacks: a structured note that *generalizes* across
episodes. Where Events store "I saw X at step Y, I did Z", lessons store
"when you see X, do Z first to maximize reward". They live in the same
V6 graph as Events but with `node_type="lesson"`.

Two reflection generators behind a uniform interface:

  * heuristic_lesson — template-based, no API dependency. Sufficient to
    demonstrate the architecture and provide a deterministic baseline for
    pytest invariants.

  * llm_lesson — calls the existing `agent.llm_agent.LLMAgent` with a
    one-sentence reflection prompt. Falls back to the heuristic when no
    OPENAI_API_KEY is set, mirroring the LLM-judge fallback pattern in
    evaluation/document_qa_llm_judge.py.

Lessons are deterministic given (episode_seed, trajectory) for the
heuristic backend; the LLM backend uses temperature=0 + a fixed seed but is
only deterministic up to model snapshot drift.

References:
  * Reflexion (Shinn et al., NeurIPS 2023)  https://arxiv.org/abs/2303.11366
  * Voyager   (Wang  et al., TMLR 2024)     https://arxiv.org/abs/2305.16291
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .event import Event


@dataclass(frozen=False)
class Lesson:
    """A verbal reflection on a completed episode."""

    episode_seed: int
    final_reward: float
    text: str
    relevant_entities: tuple[str, ...] = field(default_factory=tuple)
    relevant_event_steps: tuple[int, ...] = field(default_factory=tuple)
    success_marker: bool = False
    """True when reward was 'high enough' (per the writer's threshold)."""

    def __hash__(self) -> int:
        return hash((self.episode_seed, self.text))


# --- Heuristic reflection generator -----------------------------------------

_HINT_RE = re.compile(r"the (\w+) key opens the (\w+) door", re.IGNORECASE)
_KEY_RE = re.compile(r"see a (\w+) key", re.IGNORECASE)
_DOOR_RE = re.compile(r"see the (\w+) door", re.IGNORECASE)
_DOOR_REQ_RE = re.compile(r"requires (\w+) key", re.IGNORECASE)


def heuristic_lesson(
    events: list[Event],
    final_reward: float,
    *,
    episode_seed: int,
    success_threshold: float = 0.5,
    failure_threshold: float = 0.05,
) -> Lesson | None:
    """
    Build a one-sentence lesson summarising what happened in the episode.

    Logic:
      * If reward >= success_threshold:
          "When {hint_text}, picking up {key} and using {door} works."
      * If reward <= failure_threshold AND a hint was visible:
          "Hint '{hint_text}' was retrieved but not acted on within {N} steps."
      * Otherwise: no lesson (returns None).

    Returning None is meaningful: the lesson buffer should not fill with
    uninformative middle-ground episodes.
    """
    hints: list[tuple[str, str, int]] = []  # (color, door_name, step)
    for e in events:
        m = _HINT_RE.search(e.observation)
        if m:
            hints.append((m.group(1).lower(), m.group(2).lower(), e.step))

    relevant_entities: set[str] = set()
    relevant_steps: set[int] = set()
    for color, door, step in hints:
        relevant_entities.update([f"{color}_key", f"{door}_door"])
        relevant_steps.add(step)

    if final_reward >= success_threshold and hints:
        # Pick the hint most likely to have driven the success — the latest
        # one observed. (More sophisticated: trace the door that was opened.)
        color, door, step = hints[-1]
        text = (
            f"Episode succeeded (reward={final_reward:.2f}). "
            f"Following the hint '{color} key opens {door} door' "
            f"(step {step}) led to success — "
            f"prefer picking up the {color} key when targeting the {door} door."
        )
        return Lesson(
            episode_seed=episode_seed,
            final_reward=final_reward,
            text=text,
            relevant_entities=tuple(sorted(relevant_entities)),
            relevant_event_steps=tuple(sorted(relevant_steps)),
            success_marker=True,
        )

    if final_reward <= failure_threshold and hints:
        n = len(events)
        # Combine all hints into one lesson — failure carries forward.
        hint_text = "; ".join(f"{c} key -> {d} door" for c, d, _ in hints)
        text = (
            f"Episode failed (reward={final_reward:.2f}) over {n} steps despite "
            f"observing hints ({hint_text}). The agent saw the hints but did not "
            f"reach the corresponding doors — prioritize navigating to "
            f"{', '.join(sorted({d + ' door' for _, d, _ in hints}))} after "
            f"hint observations."
        )
        return Lesson(
            episode_seed=episode_seed,
            final_reward=final_reward,
            text=text,
            relevant_entities=tuple(sorted(relevant_entities)),
            relevant_event_steps=tuple(sorted(relevant_steps)),
            success_marker=False,
        )

    return None


# --- LLM-judge reflection generator -----------------------------------------


def llm_lesson(
    events: list[Event],
    final_reward: float,
    *,
    episode_seed: int,
    model: str = "gpt-4o-mini",
    success_threshold: float = 0.5,
) -> Lesson | None:
    """
    Generate a lesson by asking GPT-4o-mini to summarize the episode in one
    sentence. Falls back to ``heuristic_lesson`` when no API key is set
    (so CI / test runs exercise the codepath without API cost).

    The LLM is given the trajectory in compact form: each step's observation
    + action, plus the final reward.
    """
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        return heuristic_lesson(
            events,
            final_reward,
            episode_seed=episode_seed,
            success_threshold=success_threshold,
        )

    try:
        from openai import OpenAI
    except Exception:
        return heuristic_lesson(
            events,
            final_reward,
            episode_seed=episode_seed,
            success_threshold=success_threshold,
        )

    client = OpenAI()
    trajectory_lines = []
    for e in events[-20:]:  # tail of trajectory; LLM doesn't need everything
        trajectory_lines.append(f"step {e.step}: obs='{e.observation}' action={e.action}")
    trajectory = "\n".join(trajectory_lines)

    rubric = (
        "You are reviewing a single agent episode. Output one short sentence "
        "(under 30 words) describing the most useful generalizable insight an "
        "agent could carry into future similar episodes. Focus on cause-effect, "
        "not narration. If nothing useful happened, output literally 'NONE'."
    )
    user_msg = (
        f"Episode trajectory (last 20 steps):\n{trajectory}\n\n"
        f"Final reward: {final_reward:.3f}.\n\n"
        f"One-sentence lesson:"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": rubric},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            seed=episode_seed,
            max_tokens=64,
        )
    except Exception as exc:
        print(f"[llm_lesson] API error: {exc} — heuristic fallback")
        return heuristic_lesson(
            events,
            final_reward,
            episode_seed=episode_seed,
            success_threshold=success_threshold,
        )

    text = response.choices[0].message.content.strip()
    if not text or text.upper().startswith("NONE"):
        return None

    # Pull out entities mentioned (best-effort).
    relevant_entities: set[str] = set()
    for m in _HINT_RE.finditer(text):
        relevant_entities.update([f"{m.group(1).lower()}_key", f"{m.group(2).lower()}_door"])

    return Lesson(
        episode_seed=episode_seed,
        final_reward=final_reward,
        text=text,
        relevant_entities=tuple(sorted(relevant_entities)),
        relevant_event_steps=tuple(),
        success_marker=final_reward >= success_threshold,
    )


# A type alias for any lesson generator.
LessonGenerator = Callable[[list[Event], float], Lesson | None]


def make_generator(
    name: str = "heuristic",
    *,
    episode_seed: int | None = None,
    **kwargs,
) -> LessonGenerator:
    """
    Build a callable closure suitable for plugging into the run loop.

    Usage:
        gen = make_generator("heuristic", episode_seed=42)
        lesson = gen(events, reward)   # returns Lesson | None
    """
    if name == "heuristic":
        def _gen(events: list[Event], final_reward: float) -> Lesson | None:
            return heuristic_lesson(
                events,
                final_reward,
                episode_seed=episode_seed if episode_seed is not None else 0,
                **kwargs,
            )
        return _gen
    if name == "llm":
        def _gen(events: list[Event], final_reward: float) -> Lesson | None:
            return llm_lesson(
                events,
                final_reward,
                episode_seed=episode_seed if episode_seed is not None else 0,
                **kwargs,
            )
        return _gen
    raise ValueError(f"Unknown lesson generator: {name!r}. Choose 'heuristic' or 'llm'.")


def merge_lesson_text(lessons: Iterable[Lesson]) -> str:
    """Join multiple lesson texts into one block for inclusion in a prompt."""
    parts = [f"- {l.text}" for l in lessons]
    return "Past lessons:\n" + "\n".join(parts) if parts else ""
