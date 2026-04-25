"""
Memory interface contract tests.

Every memory system must implement the four-method interface:
    add_event(event, episode_seed=None) -> None
    get_relevant_events(observation, current_step, k=8) -> list[Event]
    clear() -> None
    get_stats() -> dict

This file enumerates each public memory class and verifies the contract holds.
If a new memory variant is added, list it here so the regression net catches
silent contract drift.
"""

from __future__ import annotations

import inspect

import pytest

from memory import (
    AttentionMemory,
    CausalMemory,
    EpisodicSemanticMemory,
    FlatMemory,
    GraphMemory,
    GraphMemoryV2,
    GraphMemoryV3,
    GraphMemoryV4,
    GraphMemoryV5,
    GraphMemoryV6,
    HierarchicalMemory,
    MemoryParams,
    MemoryParamsV2,
    MemoryParamsV3,
    MemoryParamsV4,
    MemoryParamsV6,
    SemanticMemory,
    SummaryMemory,
    WorkingMemory,
)


def _factories():
    """Return (name, factory) pairs for every memory system used in benchmarks."""
    return [
        ("FlatMemory", lambda: FlatMemory(window_size=20)),
        ("SemanticMemory", lambda: SemanticMemory(max_capacity=20)),
        ("SummaryMemory", lambda: SummaryMemory(raw_buffer_size=10, summarize_every=5)),
        ("EpisodicSemanticMemory", lambda: EpisodicSemanticMemory(episodic_size=10)),
        ("HierarchicalMemory", lambda: HierarchicalMemory()),
        ("WorkingMemory", lambda: WorkingMemory(capacity=7)),
        ("CausalMemory", lambda: CausalMemory()),
        ("AttentionMemory", lambda: AttentionMemory(temperature=0.5)),
        ("GraphMemory_V1", lambda: GraphMemory(MemoryParams(0.5, 0.1, 0.8, "learnable"))),
        ("GraphMemoryV2", lambda: GraphMemoryV2(MemoryParamsV2())),
        ("GraphMemoryV3", lambda: GraphMemoryV3(MemoryParamsV3())),
        ("GraphMemoryV4", lambda: GraphMemoryV4(MemoryParamsV4())),
        ("GraphMemoryV5", lambda: GraphMemoryV5(MemoryParamsV4())),
        ("GraphMemoryV6", lambda: GraphMemoryV6(MemoryParamsV6())),
    ]


@pytest.mark.parametrize("name,factory", _factories(), ids=[n for n, _ in _factories()])
def test_implements_four_method_contract(name, factory, sample_events, query_observation):
    """Every memory system must support add_event/get_relevant_events/clear/get_stats."""
    mem = factory()

    # Required methods exist
    for required in ("add_event", "get_relevant_events", "clear", "get_stats"):
        assert callable(getattr(mem, required, None)), (
            f"{name} is missing required method {required!r}"
        )

    # add_event accepts an Event (and optional episode_seed)
    for ev in sample_events:
        sig = inspect.signature(mem.add_event)
        if "episode_seed" in sig.parameters:
            mem.add_event(ev, episode_seed=42)
        else:
            mem.add_event(ev)

    # get_relevant_events returns a list of events of length <= k
    relevant = mem.get_relevant_events(query_observation, current_step=10, k=3)
    assert isinstance(relevant, list)
    assert len(relevant) <= 3

    # get_stats returns a dict
    stats = mem.get_stats()
    assert isinstance(stats, dict)

    # clear empties the store (subsequent retrieval returns 0 events)
    mem.clear()
    relevant_after_clear = mem.get_relevant_events(query_observation, current_step=10, k=3)
    assert isinstance(relevant_after_clear, list)
    assert len(relevant_after_clear) == 0
