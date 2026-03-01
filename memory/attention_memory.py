"""
AttentionMemory — differentiable soft-attention retrieval over stored events.

Standard memory systems use hard top-k selection (graph traversal, cosine cutoff,
sliding window). AttentionMemory computes a soft attention distribution over all
stored event embeddings using the current observation as query:

    e_i = query · key_i / sqrt(d)        (raw attention logit)
    a_i = softmax(e_1, ..., e_N)         (attention weights)
    top-k events ranked by a_i

Properties:
  - Every stored event contributes a differentiable score.
  - Gradients can flow from task reward through attention weights back to embeddings
    (enabling future gradient-based memory optimization).
  - No hard thresholds, no entity rules — purely similarity-driven.
  - Temperature parameter controls sharpness of attention distribution.

Temperature (τ):
  - τ = 1.0: standard softmax
  - τ < 1.0: sharper, more focused retrieval
  - τ > 1.0: softer, broader retrieval
  - Default τ = 0.5 for focused retrieval on short vocabularies.

Storage:
  - All events stored (no eviction). This is the "store everything, retrieve selectively"
    paradigm — closest analog to production RAG but with learned attention vs. cosine lookup.
  - Hint events stored with a priority flag that adds a constant attention bonus.

Thesis motivation: AttentionMemory is the bridge between retrieval (what we have now)
and attention-based reasoning (what modern transformers do). It allows a fair comparison:
can a simple attention mechanism over stored events match or beat rule-based retrieval?
It also opens the door to gradient-based optimization of the memory system's parameters.
"""

from __future__ import annotations

import numpy as np

from .embedding import embed_observation
from .event import Event

_NPC_MARKERS = ("says:", "guard says", "sage says", "see a sign:")
_HINT_BONUS = 3.0   # added to attention logit for hint events


def _is_hint(obs: str) -> bool:
    lo = obs.lower()
    return any(m in lo for m in _NPC_MARKERS)


class AttentionMemory:
    """
    Soft-attention retrieval memory. Stores all events; retrieves by attention weight.
    Implements the standard 4-method memory interface.
    """

    def __init__(self, temperature: float = 0.5) -> None:
        self._temperature = temperature
        self._events: list[Event] = []
        self._embeddings: list[np.ndarray] = []   # parallel to _events
        self._is_hint_flags: list[bool] = []

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def add_event(self, event: Event, episode_seed: int | None = None) -> None:
        emb = embed_observation(event.observation)
        self._events.append(event)
        self._embeddings.append(emb)
        self._is_hint_flags.append(_is_hint(event.observation) or event.is_hint)

    def get_relevant_events(
        self,
        observation: str,
        current_step: int,
        k: int = 8,
    ) -> list[Event]:
        """
        Compute scaled dot-product attention of observation vs. all stored events.
        Return top-k events by attention weight.
        """
        if not self._events:
            return []

        query = embed_observation(observation)
        d = max(query.shape[0], 1)

        # Stack embeddings into matrix [N x d]
        K = np.stack(self._embeddings, axis=0)  # (N, d)

        # Scaled dot-product attention logits
        logits = K @ query / np.sqrt(d)          # (N,)

        # Add hint bonus to logits for protected events
        for i, is_h in enumerate(self._is_hint_flags):
            if is_h:
                logits[i] += _HINT_BONUS

        # Softmax with temperature
        logits = logits / self._temperature
        logits -= logits.max()   # numerical stability
        weights = np.exp(logits)
        weights /= weights.sum() + 1e-9

        # Top-k by attention weight
        k_actual = min(k, len(self._events))
        top_indices = np.argpartition(weights, -k_actual)[-k_actual:]
        top_indices = top_indices[np.argsort(-weights[top_indices])]

        return [self._events[i] for i in top_indices]

    def clear(self) -> None:
        self._events.clear()
        self._embeddings.clear()
        self._is_hint_flags.clear()

    def get_stats(self) -> dict:
        n = len(self._events)
        return {
            "n_events": n,
            "n_entities": 0,
            "n_nodes": n,
            "n_edges": 0,
            "temperature": self._temperature,
            "n_hints": sum(self._is_hint_flags),
        }

    # ------------------------------------------------------------------
    # Gradient access (for future differentiable optimization)
    # ------------------------------------------------------------------

    def attention_weights(self, observation: str) -> np.ndarray:
        """
        Return the full attention weight distribution over stored events.
        Useful for visualization and gradient-based optimization.
        """
        if not self._events:
            return np.array([])
        query = embed_observation(observation)
        d = max(query.shape[0], 1)
        K = np.stack(self._embeddings, axis=0)
        logits = (K @ query / np.sqrt(d)) / self._temperature
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum() + 1e-9
        return weights
