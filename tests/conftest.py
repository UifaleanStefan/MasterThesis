"""Shared pytest fixtures for thesis invariant tests."""

from __future__ import annotations

import pytest

from memory.event import Event


@pytest.fixture
def sample_events() -> list[Event]:
    """A small, varied sequence of events covering colors, doors, and signs."""
    return [
        Event(step=0, observation="you are in a room. you see a red key.", action="pickup"),
        Event(step=1, observation="you are in a room. you see a sign: blue key opens north door.",
              action="move_north", is_hint=True),
        Event(step=2, observation="you are in a room. you see a blue key.", action="pickup"),
        Event(step=3, observation="you are in a room. you see a blue door requires blue key.",
              action="use_door"),
        Event(step=4, observation="you are in a room. you see the goal.", action="move_east"),
    ]


@pytest.fixture
def query_observation() -> str:
    """A retrieval query that should match the door-related hints in sample_events."""
    return "you see a blue door"
