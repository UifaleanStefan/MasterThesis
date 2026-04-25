"""
Invariants for V6 (GraphMemoryV6 — V4 + Reflexion lesson buffer).

The load-bearing test: with `w_lesson = 0` and no lessons recorded, V6's
retrieval is bit-identical to V4's. This is the strict-generalization
guarantee that lets V6 plug into the existing benchmark without regressing
V4's results.
"""

from __future__ import annotations

import numpy as np
import pytest

from memory.event import Event
from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4
from memory.graph_memory_v6 import GraphMemoryV6, MemoryParamsV6
from memory.lesson import (
    Lesson,
    heuristic_lesson,
    llm_lesson,
    make_generator,
)


# --- Strict-generalization tests --------------------------------------------


class TestV6IsStrictGeneralizationOfV4:
    def _events(self) -> list[Event]:
        return [
            Event(0, "you are in a room. you see a red key.", "pickup"),
            Event(1, "you are in a room. you see a sign: blue key opens north door.",
                  "move_north", is_hint=True),
            Event(2, "you are in a room. you see a blue key.", "pickup"),
            Event(3, "you are in a room. you see a blue door requires blue key.",
                  "use_door"),
            Event(4, "you are in a room. you see the goal.", "move_east"),
        ]

    def test_default_v6_params_equal_v4_params_on_first_10_dims(self):
        v4 = MemoryParamsV4()
        v6 = MemoryParamsV6()
        assert v6.to_vector()[:10] == v4.to_vector()
        assert v6.w_lesson == 0.0
        assert v6.theta_lesson_decay == 0.0

    def test_w_lesson_zero_no_lessons_matches_v4_retrieval(self):
        """V6 with w_lesson=0 + no lessons recorded retrieves identically to V4."""
        v4_params = MemoryParamsV4(
            theta_store=0.3, theta_novel=0.9, theta_surprise=0.7,
            w_recency=3.0, w_embed=1.0, mode="learnable",
        )
        v6_params = MemoryParamsV6(
            theta_store=0.3, theta_novel=0.9, theta_surprise=0.7,
            w_recency=3.0, w_embed=1.0, w_lesson=0.0,
            mode="learnable",
        )
        events = self._events()
        query = "you see a blue door"
        seed = 42

        v4_mem = GraphMemoryV4(v4_params)
        v6_mem = GraphMemoryV6(v6_params)
        for ev in events:
            v4_mem.add_event(ev, episode_seed=seed)
            v6_mem.add_event(ev, episode_seed=seed)

        v4_retrieved = v4_mem.get_relevant_events(query, current_step=10, k=3)
        v6_retrieved = v6_mem.get_relevant_events(query, current_step=10, k=3)
        assert [(e.step, e.observation) for e in v4_retrieved] == [
            (e.step, e.observation) for e in v6_retrieved
        ], "V6 with w_lesson=0 must match V4 retrieval exactly"

    def test_w_lesson_zero_with_lessons_still_recovers_v4_retrieval(self):
        """Even when lessons exist, w_lesson=0 means they don't influence event retrieval."""
        params = MemoryParamsV6(
            theta_store=0.3, theta_novel=0.9,
            w_recency=3.0, w_embed=1.0, w_lesson=0.0,
            mode="learnable",
        )
        events = self._events()
        v4_params = MemoryParamsV4(
            theta_store=0.3, theta_novel=0.9,
            w_recency=3.0, w_embed=1.0,
            mode="learnable",
        )
        v4_mem = GraphMemoryV4(v4_params)
        v6_mem = GraphMemoryV6(params)
        for ev in events:
            v4_mem.add_event(ev, episode_seed=42)
            v6_mem.add_event(ev, episode_seed=42)
        v6_mem.record_lesson(
            Lesson(
                episode_seed=42,
                final_reward=1.0,
                text="Picking up the blue key opens the north door.",
                relevant_entities=("blue_key", "north_door"),
                success_marker=True,
            )
        )
        v4_out = v4_mem.get_relevant_events("you see a blue door", 10, k=3)
        v6_out = v6_mem.get_relevant_events("you see a blue door", 10, k=3)
        assert [(e.step, e.observation) for e in v4_out] == [
            (e.step, e.observation) for e in v6_out
        ]

    def test_w_lesson_positive_returns_lessons_via_dedicated_channel(self):
        """When w_lesson>0 and we query for a relevant observation, lessons surface."""
        params = MemoryParamsV6(
            theta_store=0.3, theta_novel=0.9,
            w_recency=3.0, w_embed=1.0,
            w_lesson=2.0, theta_lesson_decay=0.0,
            mode="learnable",
        )
        v6_mem = GraphMemoryV6(params)
        v6_mem.record_lesson(
            Lesson(
                episode_seed=42,
                final_reward=1.0,
                text="Picking up the blue key opens the north door.",
                relevant_entities=("blue_key", "north_door"),
                success_marker=True,
            )
        )
        lessons = v6_mem.get_relevant_lessons("you see the blue door", k=4)
        assert len(lessons) == 1
        assert "blue key" in lessons[0].text


# --- Param roundtrip --------------------------------------------------------


class TestMemoryParamsV6Roundtrip:
    def test_to_from_vector_identity(self):
        p = MemoryParamsV6(
            theta_store=0.293, theta_novel=0.908, theta_erich=0.198,
            theta_surprise=0.785, theta_entity=0.285, theta_temporal=0.278,
            theta_decay=0.668, w_graph=0.0, w_embed=1.079, w_recency=3.777,
            w_lesson=1.5, theta_lesson_decay=0.3,
        )
        roundtrip = MemoryParamsV6.from_vector(p.to_vector())
        assert roundtrip.to_vector() == p.to_vector()
        assert roundtrip.w_lesson == p.w_lesson
        assert roundtrip.theta_lesson_decay == p.theta_lesson_decay

    def test_from_vector_rejects_short_vector(self):
        with pytest.raises(ValueError):
            MemoryParamsV6.from_vector([0.1] * 11)

    def test_w_lesson_clipping(self):
        p = MemoryParamsV6(w_lesson=10.0, theta_lesson_decay=2.0, mode="learnable")
        assert 0.0 <= p.w_lesson <= 4.0
        assert 0.0 <= p.theta_lesson_decay <= 1.0


# --- Lesson lifecycle -------------------------------------------------------


class TestLessonLifecycle:
    def test_record_then_retrieve_roundtrip(self):
        params = MemoryParamsV6(w_lesson=2.0)
        v6 = GraphMemoryV6(params)
        L = Lesson(
            episode_seed=7,
            final_reward=0.9,
            text="When the agent sees the red key, picking it up unlocks the south door.",
            relevant_entities=("red_key", "south_door"),
        )
        v6.record_lesson(L)
        retrieved = v6.get_relevant_lessons("you see the red key", k=4)
        assert len(retrieved) == 1
        assert retrieved[0].text == L.text

    def test_decay_reduces_old_episode_lessons(self):
        """
        With identical lesson text, the lesson from a more recent episode
        must score strictly higher than the older one. Tests the decay term
        in isolation (no similarity confounds).
        """
        params = MemoryParamsV6(w_lesson=2.0, theta_lesson_decay=0.5)
        v6 = GraphMemoryV6(params)

        text = "Pick up the red key first."
        # Old lesson (episode 0)
        v6.record_lesson(Lesson(
            episode_seed=1, final_reward=1.0,
            text=text, relevant_entities=("red_key",),
        ))
        v6.end_episode()
        v6.end_episode()
        v6.end_episode()
        # Newer copy of the same text (episode 3)
        v6.record_lesson(Lesson(
            episode_seed=2, final_reward=1.0,
            text=text + " (again)", relevant_entities=("red_key",),
        ))

        # Both score the same on similarity (text is essentially identical)
        # but the newer one wins on the decay term.
        retrieved = v6.get_relevant_lessons("you see the red key", k=4)
        assert len(retrieved) == 2
        assert retrieved[0].episode_seed == 2  # newer one ranks first
        assert retrieved[1].episode_seed == 1  # older one second

    def test_clear_resets_lesson_buffer(self):
        params = MemoryParamsV6(w_lesson=2.0)
        v6 = GraphMemoryV6(params)
        v6.record_lesson(Lesson(
            episode_seed=1, final_reward=1.0,
            text="Open the door.", relevant_entities=("door",),
        ))
        assert v6.get_stats()["n_lessons"] == 1
        v6.clear()
        assert v6.get_stats()["n_lessons"] == 0

    def test_episode_index_persists_across_clear(self):
        """clear() wipes lessons but NOT the episode counter — lessons recorded
        before a clear still get older-episode decay if added back later."""
        params = MemoryParamsV6(w_lesson=2.0, theta_lesson_decay=0.5)
        v6 = GraphMemoryV6(params)
        v6.end_episode()
        v6.end_episode()
        ep_before = v6._episode_index
        v6.clear()
        assert v6._episode_index == ep_before


# --- Heuristic generator ----------------------------------------------------


class TestHeuristicLesson:
    def _success_events(self) -> list[Event]:
        # Phrasing matches MultiHopKeyDoor / MegaQuestRoom NPC hints:
        # "the {color} key opens the {door} door."
        return [
            Event(0, "you see Aria (an NPC): the red key opens the north door.",
                  "move_north", is_hint=True),
            Event(1, "you see a red key.", "pickup"),
            Event(2, "you see the north door.", "use_door"),
        ]

    def test_high_reward_emits_success_lesson(self):
        L = heuristic_lesson(self._success_events(), final_reward=0.8, episode_seed=1)
        assert L is not None
        assert L.success_marker is True
        assert "red key" in L.text and "north door" in L.text

    def test_zero_reward_with_hint_emits_failure_lesson(self):
        L = heuristic_lesson(self._success_events(), final_reward=0.0, episode_seed=1)
        assert L is not None
        assert L.success_marker is False
        assert "failed" in L.text.lower()

    def test_middle_reward_emits_no_lesson(self):
        L = heuristic_lesson(self._success_events(), final_reward=0.3, episode_seed=1)
        assert L is None  # neither success nor clear failure

    def test_no_hints_no_lesson(self):
        events = [Event(0, "you see a corridor.", "move_north")]
        assert heuristic_lesson(events, final_reward=1.0, episode_seed=1) is None

    def test_make_generator_factory(self):
        gen = make_generator("heuristic", episode_seed=42)
        L = gen(self._success_events(), 0.9)
        assert L is not None and L.success_marker

    def test_unknown_generator_raises(self):
        with pytest.raises(ValueError):
            make_generator("magic", episode_seed=1)


class TestLLMLesson:
    """LLM-judge fallback path (no API key required)."""

    @pytest.fixture(autouse=True)
    def _no_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def test_falls_back_to_heuristic_without_api_key(self):
        events = [
            Event(0, "you see a sign: the blue key opens the north door.",
                  "move", is_hint=True),
            Event(1, "you see a blue key.", "pickup"),
            Event(2, "you see the north door.", "use_door"),
        ]
        L = llm_lesson(events, final_reward=1.0, episode_seed=42)
        assert L is not None
        assert L.success_marker is True
