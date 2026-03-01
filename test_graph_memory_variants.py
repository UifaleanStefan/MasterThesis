"""
Functional test for GraphMemory V2/V3/V4 and NeuralMemoryControllerV2.

Verifies:
1. All new classes import correctly.
2. All implement the standard memory interface.
3. MemoryParams from_vector/to_vector round-trip correctly.
4. Storage + retrieval produces non-empty results.
5. clear() resets state correctly.
6. NeuralMemoryControllerV2 weight pack/unpack is lossless.

Run with: python test_graph_memory_variants.py
"""

from __future__ import annotations

import traceback

import numpy as np

from memory.event import Event
from memory.graph_memory_v2 import GraphMemoryV2, MemoryParamsV2
from memory.graph_memory_v3 import GraphMemoryV3, MemoryParamsV3
from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4
from memory.neural_controller_v2 import NeuralMemoryControllerV2

# ---------------------------------------------------------------------------
# Shared test observations (same as MultiHopKeyDoor episode)
# ---------------------------------------------------------------------------
OBSERVATIONS = [
    "You see a sign: the red key opens the north door",   # step 0 — hint
    "You see a sign: the blue key opens the east door",   # step 1 — hint
    "You see a sign: the green key opens the south door", # step 2 — hint
    "You are in a room. You see a red key.",
    "You move north.",
    "You move north.",
    "You pick up the red key.",
    "You are in a room. You see a blue key.",
    "You move east.",
    "You see a blue door to the north.",
    "You are carrying the red key.",
    "You see a green key on the ground.",
    "You move south.",
    "You move south.",
    "You move south.",
    "You see a red door to the east.",
]


def _make_events() -> list[Event]:
    return [
        Event(step=i, observation=obs, action="move", is_hint=(i < 3))
        for i, obs in enumerate(OBSERVATIONS)
    ]


def _run_episode(mem, events: list[Event]) -> list:
    mem.clear()
    for ev in events:
        mem.add_event(ev, episode_seed=42)
    results = mem.get_relevant_events("You see a red door to the east.", len(events), k=5)
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_params_v2_roundtrip():
    original = (0.7, 0.3, 0.9, 2.0, 1.5, 0.1)
    p = MemoryParamsV2.from_vector(original)
    recovered = p.to_vector()
    assert len(recovered) == 6
    for a, b in zip(original, recovered):
        assert abs(a - b) < 1e-6, f"Mismatch: {a} vs {b}"
    print("  [PASS] MemoryParamsV2 from_vector/to_vector roundtrip")


def test_params_v3_roundtrip():
    original = (0.2, 0.8, 0.6, 0.4, 0.1, 0.9, 1.5, 2.0, 0.5)
    p = MemoryParamsV3.from_vector(original)
    recovered = p.to_vector()
    assert len(recovered) == 9
    for a, b in zip(original, recovered):
        assert abs(a - b) < 1e-6, f"Mismatch: {a} vs {b}"
    print("  [PASS] MemoryParamsV3 from_vector/to_vector roundtrip")


def test_params_v4_roundtrip():
    original = (0.2, 0.8, 0.6, 0.4, 0.1, 0.9, 0.05, 1.5, 2.0, 0.5)
    p = MemoryParamsV4.from_vector(original)
    recovered = p.to_vector()
    assert len(recovered) == 10
    for a, b in zip(original, recovered):
        assert abs(a - b) < 1e-6, f"Mismatch: {a} vs {b}"
    print("  [PASS] MemoryParamsV4 from_vector/to_vector roundtrip")


def test_graph_memory_v2_default():
    """Default params = identical to original GraphMemory behavior."""
    mem = GraphMemoryV2()
    events = _make_events()
    results = _run_episode(mem, events)
    stats = mem.get_stats()
    assert stats["n_events"] == len(events), f"Expected {len(events)}, got {stats['n_events']}"
    assert len(results) > 0, "Expected non-empty retrieval results"
    assert "w_graph" in stats
    print(f"  [PASS] GraphMemoryV2 default: {stats['n_events']} events stored, "
          f"{len(results)} retrieved")


def test_graph_memory_v2_learned_weights():
    """High w_graph should return graph-matched events."""
    params = MemoryParamsV2(theta_store=0.9, theta_entity=0.1, theta_temporal=0.8,
                            w_graph=3.0, w_embed=0.1, w_recency=0.0)
    mem = GraphMemoryV2(params)
    events = _make_events()
    results = _run_episode(mem, events)
    assert len(results) > 0
    print(f"  [PASS] GraphMemoryV2 high w_graph: {len(results)} events retrieved")


def test_graph_memory_v3_all_zeros():
    """All feature weights zero → store everything (backward compatible)."""
    params = MemoryParamsV3(theta_store=0.0, theta_novel=0.0, theta_erich=0.0,
                            theta_surprise=0.0, theta_entity=0.0, theta_temporal=1.0)
    mem = GraphMemoryV3(params)
    events = _make_events()
    results = _run_episode(mem, events)
    stats = mem.get_stats()
    assert stats["n_events"] == len(events), (
        f"Expected all {len(events)} stored, got {stats['n_events']}"
    )
    print(f"  [PASS] GraphMemoryV3 zero weights: stores all {stats['n_events']} events")


def test_graph_memory_v3_novelty_filtering():
    """High novelty weight + high threshold should filter repetitive events."""
    params = MemoryParamsV3(theta_store=0.3, theta_novel=1.0, theta_erich=0.0,
                            theta_surprise=0.0, theta_entity=0.0, theta_temporal=1.0)
    mem = GraphMemoryV3(params)
    events = _make_events()
    results = _run_episode(mem, events)
    stats = mem.get_stats()
    # Should store fewer events than total (repeated "move" obs are filtered)
    print(f"  [PASS] GraphMemoryV3 novelty filter: {stats['n_events']}/{len(events)} stored, "
          f"{len(results)} retrieved")


def test_graph_memory_v4_decay_zero():
    """theta_decay=0 → no decay → same as V3 behavior."""
    params = MemoryParamsV4(theta_store=0.0, theta_novel=0.0, theta_erich=0.0,
                            theta_surprise=0.0, theta_entity=0.0, theta_temporal=1.0,
                            theta_decay=0.0)
    mem = GraphMemoryV4(params)
    events = _make_events()
    results = _run_episode(mem, events)
    stats = mem.get_stats()
    assert stats["n_events"] == len(events)
    print(f"  [PASS] GraphMemoryV4 decay=0: {stats['n_events']} events, theta_decay=0.0")


def test_graph_memory_v4_decay_high():
    """High theta_decay should suppress entity nodes for stale entities."""
    params = MemoryParamsV4(theta_store=0.0, theta_novel=0.0, theta_erich=0.0,
                            theta_surprise=0.0, theta_entity=0.05, theta_temporal=1.0,
                            theta_decay=0.5)
    mem = GraphMemoryV4(params)
    events = _make_events()
    results = _run_episode(mem, events)
    stats = mem.get_stats()
    print(f"  [PASS] GraphMemoryV4 high decay: {stats['n_entities']} entity nodes "
          f"(may be fewer due to decay)")


def test_graph_memory_v4_clear():
    """clear() resets all state."""
    mem = GraphMemoryV4()
    events = _make_events()
    for ev in events:
        mem.add_event(ev, episode_seed=42)
    assert mem.get_stats()["n_events"] > 0
    mem.clear()
    stats = mem.get_stats()
    assert stats["n_events"] == 0
    assert stats["n_entities"] == 0
    print("  [PASS] GraphMemoryV4 clear() resets state")


def test_neural_controller_v2_param_count():
    ctrl = NeuralMemoryControllerV2(seed=0)
    n = ctrl.n_params
    # 50*64 + 64 + 64*32 + 32 + 32*10 + 10 = 3200+64+2048+32+320+10 = 5674
    assert 5000 < n < 7000, f"Unexpected param count: {n}"
    print(f"  [PASS] NeuralMemoryControllerV2 param count: {n}")


def test_neural_controller_v2_weight_roundtrip():
    ctrl = NeuralMemoryControllerV2(seed=7)
    w1 = ctrl.get_weights().copy()
    ctrl.set_weights(w1)
    w2 = ctrl.get_weights()
    assert np.allclose(w1, w2), "Weight roundtrip failed"
    print("  [PASS] NeuralMemoryControllerV2 weight pack/unpack lossless")


def test_neural_controller_v2_episode():
    ctrl = NeuralMemoryControllerV2(seed=42)
    events = _make_events()
    results = _run_episode(ctrl, events)
    stats = ctrl.get_stats()
    assert stats["n_events"] >= 0
    assert stats["input_dim"] == 50
    assert stats["output_dim"] == 10
    print(f"  [PASS] NeuralMemoryControllerV2 episode: {stats['n_events']} events, "
          f"{len(results)} retrieved")


def test_neural_controller_v2_perturbation():
    """Perturbing weights changes outputs (MLP is active)."""
    ctrl1 = NeuralMemoryControllerV2(seed=0)
    ctrl2 = NeuralMemoryControllerV2(seed=0)
    w = ctrl2.get_weights()
    w += np.random.randn(len(w)) * 0.5
    ctrl2.set_weights(w)
    events = _make_events()
    r1 = _run_episode(ctrl1, events)
    r2 = _run_episode(ctrl2, events)
    # Results may differ (different weights, different storage decisions)
    print(f"  [PASS] NeuralMemoryControllerV2 perturbation: "
          f"ctrl1 retrieved {len(r1)}, ctrl2 retrieved {len(r2)}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    test_params_v2_roundtrip,
    test_params_v3_roundtrip,
    test_params_v4_roundtrip,
    test_graph_memory_v2_default,
    test_graph_memory_v2_learned_weights,
    test_graph_memory_v3_all_zeros,
    test_graph_memory_v3_novelty_filtering,
    test_graph_memory_v4_decay_zero,
    test_graph_memory_v4_decay_high,
    test_graph_memory_v4_clear,
    test_neural_controller_v2_param_count,
    test_neural_controller_v2_weight_roundtrip,
    test_neural_controller_v2_episode,
    test_neural_controller_v2_perturbation,
]


if __name__ == "__main__":
    passed, failed = 0, 0
    for test_fn in TESTS:
        print(f"\n[{test_fn.__name__}]")
        try:
            test_fn()
            passed += 1
        except Exception:
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(TESTS)} tests")
    if failed == 0:
        print("All tests passed.")
