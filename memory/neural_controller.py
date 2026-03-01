"""
NeuralMemoryController — a small MLP that computes context-dependent theta.

Instead of a fixed scalar theta = (store, entity, temporal), this controller
takes the current observation embedding and memory statistics as input, and
outputs three continuous values in [0, 1] that are used as theta.

This means memory construction parameters adapt *dynamically per observation*:
  - The store probability can be high when an important observation arrives
    (entity-rich, hint, door) and low for irrelevant movement steps.
  - The entity threshold can tighten when the memory is crowded.
  - The temporal probability can be high when sequential structure matters.

Architecture:
  - Input: [obs_embedding (31-dim) + memory_stats (5-dim)] = 36-dim
  - Hidden layers: 64 → 32
  - Output: 3 scalars via sigmoid → (theta_store, theta_entity, theta_temporal)
  - Total params: 36*64 + 64 + 64*32 + 32 + 32*3 + 3 = ~4,451 params
  - Easily optimized by CMA-ES (which handles 1000+ parameter spaces).

Weight packing:
  - All weights are packed into a flat numpy array (self.get_weights()).
  - CMA-ES optimizes this flat array.
  - set_weights() restores the network from the flat array.

This class also wraps GraphMemory and uses the neural output as MemoryParams,
so it is drop-in compatible with the existing episode runner.

Thesis motivation: scalar theta is a fixed point in parameter space; neural theta
is a learned mapping from context to parameters. This allows memory construction
to react to what the agent is currently perceiving — the most principled form
of adaptive memory construction.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from .embedding import embed_observation, VOCAB
from .event import Event
from .graph_memory import GraphMemory, MemoryParams

_INPUT_DIM = len(VOCAB) + 5   # obs embedding + 5 memory stats
_H1 = 64
_H2 = 32
_OUTPUT_DIM = 3


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


class NeuralMemoryController:
    """
    MLP-based memory controller. Wraps GraphMemory; theta is computed per observation.
    Implements the standard 4-method memory interface.
    """

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.RandomState(seed)
        # Layer 1: (input → H1)
        self._W1 = rng.randn(_INPUT_DIM, _H1).astype(np.float32) * 0.1
        self._b1 = np.zeros(_H1, dtype=np.float32)
        # Layer 2: (H1 → H2)
        self._W2 = rng.randn(_H1, _H2).astype(np.float32) * 0.1
        self._b2 = np.zeros(_H2, dtype=np.float32)
        # Output layer: (H2 → 3)
        self._W3 = rng.randn(_H2, _OUTPUT_DIM).astype(np.float32) * 0.1
        self._b3 = np.zeros(_OUTPUT_DIM, dtype=np.float32)

        # Internal GraphMemory, re-created with fresh params each episode
        self._graph: GraphMemory = GraphMemory(MemoryParams(0.5, 0.1, 0.8, "learnable"))
        self._step_rng = random.Random(seed)
        self._episode_seed: int | None = None

    # ------------------------------------------------------------------
    # Weight I/O (for CMA-ES optimization)
    # ------------------------------------------------------------------

    def get_weights(self) -> np.ndarray:
        """Pack all weights into a flat 1-D array."""
        return np.concatenate([
            self._W1.flatten(), self._b1,
            self._W2.flatten(), self._b2,
            self._W3.flatten(), self._b3,
        ])

    def set_weights(self, w: np.ndarray) -> None:
        """Restore all weights from a flat 1-D array."""
        idx = 0
        def _slice(size: int) -> np.ndarray:
            nonlocal idx
            chunk = w[idx: idx + size]
            idx += size
            return chunk.astype(np.float32)

        self._W1 = _slice(_INPUT_DIM * _H1).reshape(_INPUT_DIM, _H1)
        self._b1 = _slice(_H1)
        self._W2 = _slice(_H1 * _H2).reshape(_H1, _H2)
        self._b2 = _slice(_H2)
        self._W3 = _slice(_H2 * _OUTPUT_DIM).reshape(_H2, _OUTPUT_DIM)
        self._b3 = _slice(_OUTPUT_DIM)

    @property
    def n_params(self) -> int:
        return len(self.get_weights())

    # ------------------------------------------------------------------
    # Forward pass: observation → theta
    # ------------------------------------------------------------------

    def _forward(self, observation: str) -> tuple[float, float, float]:
        """Compute (theta_store, theta_entity, theta_temporal) for an observation."""
        obs_emb = embed_observation(observation)  # shape: (31,)
        stats = self._graph.get_stats()
        mem_feats = np.array([
            min(stats["n_events"] / 100.0, 1.0),
            min(stats["n_entities"] / 20.0, 1.0),
            min(stats["n_edges"] / 200.0, 1.0),
            min(stats["n_nodes"] / 150.0, 1.0),
            0.0,   # reserved for future stat
        ], dtype=np.float32)
        x = np.concatenate([obs_emb, mem_feats])  # shape: (36,)

        h1 = _relu(x @ self._W1 + self._b1)
        h2 = _relu(h1 @ self._W2 + self._b2)
        out = _sigmoid(h2 @ self._W3 + self._b3)   # shape: (3,)
        return float(out[0]), float(out[1]), float(out[2])

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def add_event(self, event: Event, episode_seed: int | None = None) -> None:
        self._episode_seed = episode_seed
        theta = self._forward(event.observation)
        params = MemoryParams(theta[0], theta[1], theta[2], "learnable")
        self._graph._params = params
        self._graph.add_event(event, episode_seed=episode_seed)

    def get_relevant_events(
        self,
        observation: str,
        current_step: int,
        k: int = 8,
    ) -> list[Event]:
        return self._graph.get_relevant_events(observation, current_step, k)

    def clear(self) -> None:
        self._graph.clear()

    def get_stats(self) -> dict:
        s = self._graph.get_stats()
        s["n_params"] = self.n_params
        return s
