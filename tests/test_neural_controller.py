"""
Neural controller invariants — weight pack/unpack roundtrip and deterministic
forward pass via the public memory interface.
"""

from __future__ import annotations

import numpy as np

from memory.event import Event
from memory.neural_controller_v2 import NeuralMemoryControllerV2


def test_get_set_weights_roundtrip():
    """get_weights -> set_weights restores identical state."""
    ctrl = NeuralMemoryControllerV2(seed=42)
    original = ctrl.get_weights().copy()

    perturbed = original + 0.5
    ctrl.set_weights(perturbed)
    assert np.allclose(ctrl.get_weights(), perturbed)

    ctrl.set_weights(original)
    assert np.allclose(ctrl.get_weights(), original)


def test_n_params_matches_get_weights_length():
    ctrl = NeuralMemoryControllerV2(seed=0)
    assert ctrl.n_params == len(ctrl.get_weights())


def test_n_params_is_5674():
    """The published thesis architecture has 5,674 parameters."""
    ctrl = NeuralMemoryControllerV2(seed=0)
    assert ctrl.n_params == 5674


def test_deterministic_retrieval_under_fixed_seed():
    """Same weights + same event sequence + same episode_seed => same retrieval."""
    events = [
        Event(step=0, observation="you see a red key.", action="pickup"),
        Event(step=1, observation="you see a sign: blue key opens north.",
              action="move", is_hint=True),
        Event(step=2, observation="you see a blue key.", action="pickup"),
    ]

    def run() -> list[tuple[int, str]]:
        m = NeuralMemoryControllerV2(seed=42)
        for ev in events:
            m.add_event(ev, episode_seed=7)
        out = m.get_relevant_events("you see a blue door", current_step=10, k=2)
        return [(e.step, e.observation) for e in out]

    a, b = run(), run()
    assert a == b

