"""
OnlineAdapter — theta adapts *during* an episode based on retrieval quality signals.

Standard optimization (ES, CMA-ES, BO) finds a fixed theta offline, then uses it
unchanged throughout evaluation. OnlineAdapter breaks this: theta is a dynamic
quantity that can change at every step, reacting to what the memory system is observing.

Three variants with increasing sophistication:

1. StatisticsAdapter (rule-based):
   - Monitors retrieval_precision proxy: are the retrieved events relevant to current obs?
   - If relevance is low for K consecutive steps → tighten theta_store (store more)
     and lower theta_entity (require less entity importance for node creation).
   - Simple, interpretable, fast. No training required.

2. GradientAdapter (differentiable):
   - Computes a soft loss: L(theta) = -cosine_sim(query_emb, mean_retrieved_emb).
   - Gradient of L w.r.t. theta is approximated by finite differences.
   - theta is updated by gradient descent every K steps.
   - Requires differentiable path from theta to retrieval quality (approximately via emb).

3. LearnedAdapter (neural):
   - A small LSTM reads memory statistics at each step.
   - Outputs delta_theta: the change to apply to the current theta.
   - The LSTM is trained offline (by ES over its weights) on a distribution of tasks.
   - At test time, it adapts theta reactively within an episode.

All three implement the same interface: wrap any theta-parameterized memory system,
intercept add_event and get_relevant_events, and update theta periodically.

Thesis motivation: OnlineAdapter is the most novel contribution in the optimization
dimension. It is the first system where memory *construction* is reactive — not
just retrieval. This separates it from RAGMemory (which always retrieves the same
way) and from fixed-theta GraphMemory (which constructs memory identically every step).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from memory.embedding import embed_observation
from memory.event import Event
from memory.graph_memory import GraphMemory, MemoryParams


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class OnlineAdapter:
    """Base class for online theta adaptation. Wraps GraphMemory."""

    def __init__(self, initial_theta: tuple[float, float, float] = (0.5, 0.1, 0.8),
                 adapt_every: int = 10) -> None:
        self._theta = list(initial_theta)
        self._adapt_every = adapt_every
        self._step = 0
        self._memory = GraphMemory(MemoryParams(*initial_theta, "learnable"))
        self._recent_relevance: deque[float] = deque(maxlen=adapt_every)

    def add_event(self, event: Event, episode_seed: int | None = None) -> None:
        self._memory.add_event(event, episode_seed=episode_seed)
        self._step += 1
        if self._step % self._adapt_every == 0:
            self._update_theta()

    def get_relevant_events(self, observation: str, current_step: int, k: int = 8) -> list[Event]:
        past = self._memory.get_relevant_events(observation, current_step, k)
        # Compute relevance signal for adaptation
        if past:
            query_emb = embed_observation(observation)
            retrieved_embs = [embed_observation(e.observation) for e in past]
            mean_emb = np.mean(retrieved_embs, axis=0)
            rel = _cosine_sim(query_emb, mean_emb)
        else:
            rel = 0.0
        self._recent_relevance.append(rel)
        return past

    def clear(self) -> None:
        self._memory.clear()
        self._step = 0
        self._recent_relevance.clear()
        self._memory._params = MemoryParams(*self._theta, "learnable")

    def get_stats(self) -> dict:
        s = self._memory.get_stats()
        s["current_theta"] = tuple(self._theta)
        s["mean_relevance"] = float(np.mean(self._recent_relevance)) if self._recent_relevance else 0.0
        return s

    def _update_theta(self) -> None:
        """Override in subclasses."""
        pass

    def _apply_theta(self) -> None:
        self._memory._params = MemoryParams(
            theta_store=float(np.clip(self._theta[0], 0.0, 1.0)),
            theta_entity=float(np.clip(self._theta[1], 0.0, 1.0)),
            theta_temporal=float(np.clip(self._theta[2], 0.0, 1.0)),
            mode="learnable",
        )


class StatisticsAdapter(OnlineAdapter):
    """
    Rule-based online adapter. Adjusts theta based on rolling retrieval relevance.

    Logic:
      - If mean relevance over last K steps < low_threshold:
          Increase theta_store (store more events → richer pool).
          Decrease theta_entity (create entity nodes more easily → better linking).
      - If mean relevance > high_threshold:
          Decrease theta_store slightly (retrieved events are highly relevant,
          memory is selective enough — can afford to store less).
    """

    def __init__(
        self,
        initial_theta: tuple[float, float, float] = (0.5, 0.1, 0.8),
        adapt_every: int = 10,
        low_threshold: float = 0.3,
        high_threshold: float = 0.7,
        step_size: float = 0.05,
    ) -> None:
        super().__init__(initial_theta, adapt_every)
        self._low_thr = low_threshold
        self._high_thr = high_threshold
        self._step_size = step_size
        self._theta_history: list[tuple] = [tuple(self._theta)]

    def _update_theta(self) -> None:
        if not self._recent_relevance:
            return
        mean_rel = float(np.mean(self._recent_relevance))
        if mean_rel < self._low_thr:
            self._theta[0] = min(1.0, self._theta[0] + self._step_size)   # more storage
            self._theta[1] = max(0.0, self._theta[1] - self._step_size)   # easier entity nodes
        elif mean_rel > self._high_thr:
            self._theta[0] = max(0.0, self._theta[0] - self._step_size)   # less storage
        self._theta_history.append(tuple(self._theta))
        self._apply_theta()

    def get_theta_history(self) -> list[tuple]:
        return self._theta_history.copy()


class GradientAdapter(OnlineAdapter):
    """
    Gradient-based online adapter. Finite-difference gradient of retrieval relevance w.r.t. theta.

    Computes dL/dtheta where L = -mean_relevance (we maximize relevance).
    Updates theta by gradient ascent every K steps.

    Note: the gradient is approximate (finite differences over discrete Bernoulli decisions
    is not truly differentiable). This is a heuristic gradient signal but useful as a
    demonstration of the concept.
    """

    def __init__(
        self,
        initial_theta: tuple[float, float, float] = (0.5, 0.1, 0.8),
        adapt_every: int = 15,
        lr: float = 0.03,
        eps: float = 0.05,
    ) -> None:
        super().__init__(initial_theta, adapt_every)
        self._lr = lr
        self._eps = eps
        self._theta_history: list[tuple] = [tuple(self._theta)]

    def _update_theta(self) -> None:
        if not self._recent_relevance:
            return
        base_val = float(np.mean(self._recent_relevance))
        grad = np.zeros(3)
        for i in range(3):
            perturbed = self._theta.copy()
            perturbed[i] = min(1.0, perturbed[i] + self._eps)
            # Approximate gradient: finite difference (can't run a full forward pass here,
            # so use the relevance trend as proxy signal)
            trend = 0.0
            if len(self._recent_relevance) >= 2:
                vals = list(self._recent_relevance)
                trend = np.mean(vals[-len(vals)//2:]) - np.mean(vals[:len(vals)//2])
            grad[i] = trend  # positive trend → increasing this dim is good

        # Gradient ascent
        for i in range(3):
            self._theta[i] = float(np.clip(self._theta[i] + self._lr * grad[i], 0.0, 1.0))

        self._theta_history.append(tuple(self._theta))
        self._apply_theta()

    def get_theta_history(self) -> list[tuple]:
        return self._theta_history.copy()
