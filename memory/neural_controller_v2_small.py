"""
NeuralMemoryControllerV2Small — compact MLP meta-controller for CMA-ES training.

Same interface and input features as NeuralMemoryControllerV2, but with a
smaller architecture that makes CMA-ES training practical:

    Full V2:  50 -> 64 -> 32 -> 10  (~5,578 params, lambda~30, ~2500 min)
    Small V2: 50 -> 32 -> 10        (~1,962 params, lambda~27, ~47 min)

The smaller architecture is used for the first training run. Results are
documented and compared to the scalar V4 theta baseline. The full V2 can
be trained overnight if the small version shows promise.

Architecture:
    Input:  50-dim (31 TF-IDF + 10 task-agnostic features + 9 reserved zeros)
    Hidden: 32 (ReLU)
    Output: 10 (Sigmoid -> MemoryParamsV4 per observation)
    Params: 50*32 + 32 + 32*10 + 10 = 1,962

Output interpretation (same as V2):
    dims 0-6: sigmoid -> [0, 1]  (theta parameters)
    dims 7-9: sigmoid * 4 -> [0, 4]  (retrieval weights)

CMA-ES training:
    clip_to_unit=False  (weights are unbounded)
    sigma=0.05          (small perturbations in weight space)
    n_generations=30
    n_episodes_per_candidate=20

Original V2 preserved at: memory/neural_controller_v2.py
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .embedding import embed_observation, VOCAB
from .entity_extraction import extract_entities
from .event import Event
from .graph_memory_v4 import GraphMemoryV4, MemoryParamsV4
from .neural_controller_v2 import (
    _compute_input_features, _sigmoid, _relu,
    _VOCAB_DIM, _EXTRA_FEATURES, _INPUT_DIM, _OUTPUT_DIM, _RETRIEVAL_WEIGHT_SCALE, _WINDOW,
)

# Small architecture constants
_H_SMALL = 32   # single hidden layer


class NeuralMemoryControllerV2Small:
    """
    Compact MLP meta-controller: 50 -> 32 -> 10.

    Wraps GraphMemoryV4. At every step, computes the full MemoryParamsV4
    vector from the current observation and memory state using a smaller MLP.

    Implements the standard 4-method memory interface:
        add_event(event, episode_seed) -> None
        get_relevant_events(observation, current_step, k) -> list[Event]
        clear() -> None
        get_stats() -> dict

    CMA-ES interface:
        get_weights() -> np.ndarray  (flat 1962-dim)
        set_weights(w: np.ndarray) -> None
        n_params: int (property)
    """

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.RandomState(seed)
        scale = 0.05
        self._W1 = rng.randn(_INPUT_DIM, _H_SMALL).astype(np.float32) * scale
        self._b1 = np.zeros(_H_SMALL, dtype=np.float32)
        self._W2 = rng.randn(_H_SMALL, _OUTPUT_DIM).astype(np.float32) * scale
        self._b2 = np.zeros(_OUTPUT_DIM, dtype=np.float32)

        self._graph = GraphMemoryV4(MemoryParamsV4())
        self._stored_embeddings: list[np.ndarray] = []
        self._entity_names_in_graph: set[str] = set()
        self._max_entities_seen: int = 0

    # ------------------------------------------------------------------
    # Weight I/O
    # ------------------------------------------------------------------

    def get_weights(self) -> np.ndarray:
        return np.concatenate([
            self._W1.flatten(), self._b1,
            self._W2.flatten(), self._b2,
        ])

    def set_weights(self, w: np.ndarray) -> None:
        idx = 0

        def _slice(size: int) -> np.ndarray:
            nonlocal idx
            chunk = w[idx: idx + size]
            idx += size
            return chunk.astype(np.float32)

        self._W1 = _slice(_INPUT_DIM * _H_SMALL).reshape(_INPUT_DIM, _H_SMALL)
        self._b1 = _slice(_H_SMALL)
        self._W2 = _slice(_H_SMALL * _OUTPUT_DIM).reshape(_H_SMALL, _OUTPUT_DIM)
        self._b2 = _slice(_OUTPUT_DIM)

    @property
    def n_params(self) -> int:
        return len(self.get_weights())

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, observation: str, current_step: int) -> MemoryParamsV4:
        graph_stats = self._graph.get_stats()
        x = _compute_input_features(
            observation, current_step, graph_stats,
            self._stored_embeddings, self._entity_names_in_graph,
            self._max_entities_seen,
        )
        h1 = _relu(x @ self._W1 + self._b1)
        out = _sigmoid(h1 @ self._W2 + self._b2)

        return MemoryParamsV4(
            theta_store=float(out[0]),
            theta_novel=float(out[1]),
            theta_erich=float(out[2]),
            theta_surprise=float(out[3]),
            theta_entity=float(out[4]),
            theta_temporal=float(out[5]),
            theta_decay=float(out[6]),
            w_graph=float(out[7]) * _RETRIEVAL_WEIGHT_SCALE,
            w_embed=float(out[8]) * _RETRIEVAL_WEIGHT_SCALE,
            w_recency=float(out[9]) * _RETRIEVAL_WEIGHT_SCALE,
            mode="learnable",
        )

    # ------------------------------------------------------------------
    # Memory interface
    # ------------------------------------------------------------------

    def add_event(self, event: Event, episode_seed: int | None = None) -> None:
        obs_entities = extract_entities(event.observation)
        self._max_entities_seen = max(self._max_entities_seen, len(obs_entities))

        params = self._forward(event.observation, event.step)
        self._graph._params = params
        self._graph.add_event(event, episode_seed=episode_seed)

        obs_emb = embed_observation(event.observation)
        self._stored_embeddings.append(obs_emb)
        if len(self._stored_embeddings) > _WINDOW * 2:
            self._stored_embeddings = self._stored_embeddings[-_WINDOW:]

        for e in obs_entities:
            self._entity_names_in_graph.add(e)

    def get_relevant_events(self, observation: str, current_step: int, k: int = 8) -> list[Event]:
        return self._graph.get_relevant_events(observation, current_step, k)

    def clear(self) -> None:
        self._graph.clear()
        self._stored_embeddings.clear()
        self._entity_names_in_graph.clear()
        self._max_entities_seen = 0

    def get_stats(self) -> dict:
        s = self._graph.get_stats()
        s["n_params"] = self.n_params
        s["architecture"] = f"{_INPUT_DIM}->{_H_SMALL}->{_OUTPUT_DIM}"
        return s
