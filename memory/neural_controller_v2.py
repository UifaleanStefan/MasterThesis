"""
NeuralMemoryController V2 — Proposal D: Richer Input + Full 10D Output.

Extends the original NeuralMemoryController (neural_controller.py) in two ways:

1. RICHER INPUT REPRESENTATION (36-dim → 50-dim):
   The original controller uses [TF-IDF(obs) (31-dim), 5 graph stats].
   This misses several task-agnostic signals that are predictive of whether
   an observation should be stored aggressively or filtered.

   New input features (all task-agnostic, no task identity exposed to MLP):

   Original (31-dim): TF-IDF bag-of-words over fixed vocabulary
   Added (19-dim):
     novelty_score       (1) — cosine dist from mean stored embedding
     entity_count_norm   (1) — entities in obs / max entities seen
     step_normalized     (1) — t / 250 (proxy for episode progress)
     n_events_norm       (1) — n_events / 100 (memory fill level)
     n_entities_norm     (1) — n_entities / 20 (entity graph size)
     n_edges_norm        (1) — n_edges / 200 (graph density)
     mean_recency        (1) — mean 1/(1+Δt) over stored events
     vocab_entropy       (1) — entropy of token dist in obs (information density)
     entity_repeat_rate  (1) — fraction of obs entities already in graph
     surprise_score      (1) — L2(embed(obs), mean(recent_embeds)) / sqrt(dim)
     [reserved × 9]     (9) — zero-padded for future features (keeps dim stable)

   Total input: 31 + 10 active + 9 reserved = 50 dims.
   All features are normalized to [0, 1] and require no task knowledge.

2. FULL 10D OUTPUT (3-dim → 10-dim):
   The original controller outputs (theta_store, theta_entity, theta_temporal).
   V2 outputs the full MemoryParamsV4 vector:
     (theta_store, theta_novel, theta_erich, theta_surprise,
      theta_entity, theta_temporal, theta_decay,
      w_graph, w_embed, w_recency)

   The MLP becomes a meta-controller: given what the agent currently sees
   and what memory currently contains, it decides ALL aspects of memory
   construction and retrieval at every step.

   w_graph/w_embed/w_recency outputs are passed through sigmoid and then
   scaled to [0, 4] before being used as retrieval weights.

ARCHITECTURE:
   Input:  50-dim
   Layer 1: 50 → 64 (ReLU)
   Layer 2: 64 → 32 (ReLU)
   Output:  32 → 10 (Sigmoid → output-specific scaling)
   Total params: 50*64 + 64 + 64*32 + 32 + 32*10 + 10 = 5,578

TRAINING:
   CMA-ES on the flat weight vector (~5,578 parameters).
   Objective: J(weights) = mean_reward over N episodes.
   Clip: theta dims are in [0,1] (sigmoid), w dims scaled to [0,4].

   python runner.py --config experiments/neural_controller_v2_cmaes.yaml

THE GENERALIZATION EXPERIMENT:
   Train on MultiHopKeyDoor (n=50 generations × 20 candidates × 40 episodes).
   Then evaluate zero-shot on MegaQuestRoom — no weight update.
   If precision stays near 1.000, the learned policy generalizes.
   If precision drops, it confirms that task-specific θ is necessary
   (also a valid thesis finding).

Original controller preserved at: memory/neural_controller.py
Original GraphMemory preserved at: memory/graph_memory.py
This controller wraps GraphMemoryV4 (memory/graph_memory_v4.py).
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from .embedding import embed_observation, VOCAB
from .entity_extraction import extract_entities
from .event import Event
from .graph_memory_v4 import GraphMemoryV4, MemoryParamsV4

# -----------------------------------------------------------------------
# Architecture constants
# -----------------------------------------------------------------------
_VOCAB_DIM = len(VOCAB)            # 31
_EXTRA_FEATURES = 19               # 10 active + 9 reserved
_INPUT_DIM = _VOCAB_DIM + _EXTRA_FEATURES   # 50
_H1 = 64
_H2 = 32
_OUTPUT_DIM = 10                   # full MemoryParamsV4 vector

# Scaling for retrieval weight outputs (sigmoid output * scale → [0, scale])
_RETRIEVAL_WEIGHT_SCALE = 4.0

# Rolling window for novelty / surprise
_WINDOW = 50


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _cosine_sim_np(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _vocab_entropy(obs_emb: np.ndarray) -> float:
    """Shannon entropy of the TF-IDF token distribution (normalized to [0,1])."""
    p = obs_emb / (obs_emb.sum() + 1e-9)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    h = -float(np.sum(p * np.log(p + 1e-9)))
    max_h = math.log(len(obs_emb))
    return min(h / max_h, 1.0) if max_h > 0 else 0.0


def _compute_input_features(
    observation: str,
    current_step: int,
    graph_stats: dict,
    stored_embeddings: list[np.ndarray],
    entity_names_in_graph: set[str],
    max_entities_seen: int,
    max_steps: int = 250,
) -> np.ndarray:
    """
    Build the 50-dim input vector for the MLP.
    All features are task-agnostic and normalized to [0, 1].
    """
    obs_emb = embed_observation(observation)   # (31,)

    # --- 1. Novelty: cosine distance from mean stored embedding ---
    if stored_embeddings:
        recent = stored_embeddings[-_WINDOW:]
        mean_emb = np.mean(recent, axis=0)
        novelty = max(0.0, 1.0 - _cosine_sim_np(obs_emb, mean_emb))
    else:
        novelty = 1.0

    # --- 2. Entity richness ---
    obs_entities = extract_entities(observation)
    entity_count_norm = min(len(obs_entities) / max(max_entities_seen, 1), 1.0)

    # --- 3. Episode progress ---
    step_norm = min(current_step / max_steps, 1.0)

    # --- 4. Memory fill stats ---
    n_events_norm = min(graph_stats.get("n_events", 0) / 100.0, 1.0)
    n_entities_norm = min(graph_stats.get("n_entities", 0) / 20.0, 1.0)
    n_edges_norm = min(graph_stats.get("n_edges", 0) / 200.0, 1.0)

    # --- 5. Mean recency of stored events ---
    # Approximate: if we have N events stored, mean delta_t ≈ step/2
    n_ev = max(graph_stats.get("n_events", 1), 1)
    approx_mean_delta = current_step / (2 * n_ev)
    mean_recency = 1.0 / (1.0 + approx_mean_delta)

    # --- 6. Vocabulary entropy (information density of obs) ---
    entropy = _vocab_entropy(obs_emb)

    # --- 7. Entity repeat rate: fraction already in graph ---
    if obs_entities and entity_names_in_graph:
        repeat_rate = sum(1 for e in obs_entities if e in entity_names_in_graph) / len(obs_entities)
    else:
        repeat_rate = 0.0

    # --- 8. Surprise: L2 dist from recent context mean ---
    if len(stored_embeddings) >= 2:
        recent = stored_embeddings[-_WINDOW:]
        mean_emb_r = np.mean(recent, axis=0)
        dim = max(len(obs_emb), 1)
        surprise = min(float(np.linalg.norm(obs_emb - mean_emb_r)) / (dim ** 0.5), 1.0)
    else:
        surprise = 0.0

    # Pack active extra features (10) + reserved zeros (9)
    extra = np.array([
        novelty,           # 0
        entity_count_norm, # 1
        step_norm,         # 2
        n_events_norm,     # 3
        n_entities_norm,   # 4
        n_edges_norm,      # 5
        mean_recency,      # 6
        entropy,           # 7
        repeat_rate,       # 8
        surprise,          # 9
        # reserved:
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 10-18
    ], dtype=np.float32)

    return np.concatenate([obs_emb, extra])   # (50,)


class NeuralMemoryControllerV2:
    """
    MLP meta-controller with richer inputs and full 10D theta output.

    Wraps GraphMemoryV4. At every step, computes the full MemoryParamsV4
    vector from the current observation and memory state.

    Implements the standard 4-method memory interface:
        add_event(event, episode_seed) -> None
        get_relevant_events(observation, current_step, k) -> list[Event]
        clear() -> None
        get_stats() -> dict

    CMA-ES interface:
        get_weights() -> np.ndarray  (flat ~5578-dim)
        set_weights(w: np.ndarray) -> None
        n_params: int (property)
    """

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.RandomState(seed)
        scale = 0.05   # small init keeps sigmoid outputs near 0.5
        self._W1 = rng.randn(_INPUT_DIM, _H1).astype(np.float32) * scale
        self._b1 = np.zeros(_H1, dtype=np.float32)
        self._W2 = rng.randn(_H1, _H2).astype(np.float32) * scale
        self._b2 = np.zeros(_H2, dtype=np.float32)
        self._W3 = rng.randn(_H2, _OUTPUT_DIM).astype(np.float32) * scale
        self._b3 = np.zeros(_OUTPUT_DIM, dtype=np.float32)

        self._graph = GraphMemoryV4(MemoryParamsV4())
        self._stored_embeddings: list[np.ndarray] = []
        self._entity_names_in_graph: set[str] = set()
        self._max_entities_seen: int = 0
        self._episode_seed: int | None = None

    # ------------------------------------------------------------------
    # Weight I/O
    # ------------------------------------------------------------------

    def get_weights(self) -> np.ndarray:
        return np.concatenate([
            self._W1.flatten(), self._b1,
            self._W2.flatten(), self._b2,
            self._W3.flatten(), self._b3,
        ])

    def set_weights(self, w: np.ndarray) -> None:
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
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, observation: str, current_step: int) -> MemoryParamsV4:
        """
        observation → 50-dim features → MLP → MemoryParamsV4

        Output interpretation:
            dims 0-6: sigmoid → [0, 1]  (theta parameters)
            dims 7-9: sigmoid * 4 → [0, 4]  (retrieval weights)
        """
        graph_stats = self._graph.get_stats()
        x = _compute_input_features(
            observation,
            current_step,
            graph_stats,
            self._stored_embeddings,
            self._entity_names_in_graph,
            self._max_entities_seen,
        )

        h1 = _relu(x @ self._W1 + self._b1)
        h2 = _relu(h1 @ self._W2 + self._b2)
        out = _sigmoid(h2 @ self._W3 + self._b3)   # (10,) all in [0,1]

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
        self._episode_seed = episode_seed

        # Track entities for entity_repeat_rate feature
        obs_entities = extract_entities(event.observation)
        self._max_entities_seen = max(self._max_entities_seen, len(obs_entities))

        # Forward pass → context-dependent params for this observation
        params = self._forward(event.observation, event.step)
        self._graph._params = params
        self._graph.add_event(event, episode_seed=episode_seed)

        # Track stored embedding for future novelty/surprise computation
        obs_emb = embed_observation(event.observation)
        self._stored_embeddings.append(obs_emb)
        if len(self._stored_embeddings) > _WINDOW * 2:
            self._stored_embeddings = self._stored_embeddings[-_WINDOW:]

        # Update entity tracking
        for e in obs_entities:
            self._entity_names_in_graph.add(e)

    def get_relevant_events(
        self,
        observation: str,
        current_step: int,
        k: int = 8,
    ) -> list[Event]:
        return self._graph.get_relevant_events(observation, current_step, k)

    def clear(self) -> None:
        self._graph.clear()
        self._stored_embeddings.clear()
        self._entity_names_in_graph.clear()
        self._max_entities_seen = 0

    def get_stats(self) -> dict:
        s = self._graph.get_stats()
        s["n_params"] = self.n_params
        s["input_dim"] = _INPUT_DIM
        s["output_dim"] = _OUTPUT_DIM
        return s
