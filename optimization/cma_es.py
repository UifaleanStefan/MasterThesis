"""
CMA-ES — Covariance Matrix Adaptation Evolution Strategy.

CMA-ES is the gold-standard black-box optimizer for continuous parameter spaces.
Unlike the simple ES in main.py (which only maintains a mean and isotropic sigma),
CMA-ES maintains a full covariance matrix C that captures the correlation structure
of the search landscape. This allows it to:
  1. Detect and exploit correlations between theta components (e.g., theta_store and
     theta_entity may be positively correlated on hint-heavy tasks).
  2. Adapt the step size (sigma) per dimension independently.
  3. Scale to high-dimensional parameter spaces (e.g., NeuralMemoryController with
     4,000+ weight parameters) — basic ES cannot handle this reliably.

Implementation follows the original Hansen (2006) CMA-ES algorithm.
Key parameters:
  - mu (μ): number of selected parents (default: lambda/2)
  - lambda (λ): population size (default: 4 + floor(3*ln(n)))
  - sigma: initial step size
  - n: number of parameters

Clipping to [0, 1]:
  When optimizing theta = (store, entity, temporal), parameters are clipped to [0,1]
  after sampling. This is a simple constraint-handling strategy appropriate for the
  bounded parameter space.

For NeuralMemoryController weights (unbounded), clipping is disabled.

Usage:
    optimizer = CMAES(n_params=3, sigma=0.3)
    for generation in range(n_generations):
        candidates = optimizer.ask()             # list of np.ndarray
        fitnesses = [evaluate(c) for c in candidates]
        optimizer.tell(candidates, fitnesses)    # update distribution
        best = optimizer.best_solution
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


class CMAES:
    """
    Pure-numpy CMA-ES implementation.
    Minimizes the negative reward (i.e., maximizes reward).
    """

    def __init__(
        self,
        n_params: int,
        mean: np.ndarray | None = None,
        sigma: float = 0.3,
        seed: int = 0,
        clip_to_unit: bool = True,
    ) -> None:
        self.n = n_params
        self.sigma = sigma
        self.clip_to_unit = clip_to_unit
        self._rng = np.random.RandomState(seed)

        # Mean (μ)
        if mean is not None:
            self.mean = mean.astype(np.float64).copy()
        else:
            self.mean = np.full(n_params, 0.5, dtype=np.float64)

        # Population and selection sizes
        self.lam = max(4, 4 + int(3 * math.log(n_params)))   # λ
        self.mu = self.lam // 2                                 # μ

        # Recombination weights
        raw_w = np.array([math.log(self.mu + 0.5) - math.log(i + 1) for i in range(self.mu)])
        self.weights = raw_w / raw_w.sum()
        self.mu_eff = 1.0 / (self.weights ** 2).sum()

        # Step-size control
        self.c_sigma = (self.mu_eff + 2) / (n_params + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, math.sqrt((self.mu_eff - 1) / (n_params + 1)) - 1) + self.c_sigma
        self.p_sigma = np.zeros(n_params, dtype=np.float64)

        # Covariance matrix control
        self.cc = (4 + self.mu_eff / n_params) / (n_params + 4 + 2 * self.mu_eff / n_params)
        self.c1 = 2 / ((n_params + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1 - self.c1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((n_params + 2) ** 2 + self.mu_eff),
        )
        self.p_c = np.zeros(n_params, dtype=np.float64)

        # Covariance matrix and its eigendecomposition
        self.C = np.eye(n_params, dtype=np.float64)
        self.eigen_eval = 0
        self.D: np.ndarray | None = None     # eigenvalues
        self.B: np.ndarray | None = None     # eigenvectors

        # Tracking (must be initialized before _update_eigen)
        self.generation = 0
        self._update_eigen()
        self.best_solution: np.ndarray = self.mean.copy()
        self.best_fitness: float = -np.inf
        self._last_candidates: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self) -> list[np.ndarray]:
        """Sample λ candidate solutions."""
        self._update_eigen()
        candidates = []
        for _ in range(self.lam):
            z = self._rng.randn(self.n)
            y = self.B @ (self.D * z)   # transform by covariance
            x = self.mean + self.sigma * y
            if self.clip_to_unit:
                x = np.clip(x, 0.0, 1.0)
            candidates.append(x)
        self._last_candidates = candidates
        return candidates

    def tell(self, candidates: list[np.ndarray], fitnesses: list[float]) -> None:
        """
        Update the distribution given evaluated candidates and their fitnesses.
        Higher fitness = better.
        """
        assert len(candidates) == len(fitnesses) == self.lam

        # Sort by fitness descending
        ranked = sorted(zip(fitnesses, candidates), key=lambda x: -x[0])
        selected = [c for _, c in ranked[: self.mu]]
        if ranked[0][0] > self.best_fitness:
            self.best_fitness = ranked[0][0]
            self.best_solution = ranked[0][1].copy()

        # Weighted recombination
        old_mean = self.mean.copy()
        self.mean = sum(w * x for w, x in zip(self.weights, selected))
        if self.clip_to_unit:
            self.mean = np.clip(self.mean, 0.0, 1.0)

        # Step-size control (CSA)
        inv_sqrt_C = self.B @ np.diag(1.0 / self.D) @ self.B.T
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + math.sqrt(
            self.c_sigma * (2 - self.c_sigma) * self.mu_eff
        ) * inv_sqrt_C @ (self.mean - old_mean) / self.sigma

        hs = (np.linalg.norm(self.p_sigma) / math.sqrt(1 - (1 - self.c_sigma) ** (2 * (self.generation + 1)))
              < (1.4 + 2 / (self.n + 1)) * self._chi_n())

        # Covariance matrix adaptation (CMA)
        self.p_c = (1 - self.cc) * self.p_c + hs * math.sqrt(
            self.cc * (2 - self.cc) * self.mu_eff
        ) * (self.mean - old_mean) / self.sigma

        artmp = np.array([(x - old_mean) / self.sigma for x in selected])
        self.C = (
            (1 - self.c1 - self.c_mu) * self.C
            + self.c1 * (np.outer(self.p_c, self.p_c) + (1 - hs) * self.cc * (2 - self.cc) * self.C)
            + self.c_mu * sum(w * np.outer(d, d) for w, d in zip(self.weights, artmp))
        )

        # Sigma update
        self.sigma *= math.exp(
            (self.c_sigma / self.d_sigma) * (np.linalg.norm(self.p_sigma) / self._chi_n() - 1)
        )
        self.sigma = max(self.sigma, 1e-8)

        self.generation += 1
        self._update_eigen()

    def summary(self) -> dict:
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "mean": self.mean.tolist(),
            "sigma": self.sigma,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _chi_n(self) -> float:
        n = self.n
        return math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    def _update_eigen(self) -> None:
        """Recompute eigendecomposition of C (expensive, do periodically)."""
        # Always update on first call (D and B are None)
        if self.D is not None and self.generation - self.eigen_eval < self.lam / (10 * self.n):
            return
        # Enforce symmetry
        self.C = (self.C + self.C.T) / 2.0
        eigenvalues, self.B = np.linalg.eigh(self.C)
        eigenvalues = np.maximum(eigenvalues, 1e-20)
        self.D = np.sqrt(eigenvalues)
        self.eigen_eval = self.generation


def run_cmaes_optimization(
    eval_fn: Callable[[np.ndarray], float],
    n_params: int = 3,
    n_generations: int = 20,
    sigma: float = 0.3,
    seed: int = 0,
    clip_to_unit: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """
    Convenience wrapper: run CMA-ES for n_generations, return (best_theta, history).

    Parameters
    ----------
    eval_fn : callable
        Takes a parameter array, returns scalar fitness (higher = better).
    n_params : int
        Dimensionality of the parameter space.
    n_generations : int
        Number of CMA-ES generations.
    sigma : float
        Initial step size.
    seed : int
        Random seed.
    clip_to_unit : bool
        Whether to clip parameters to [0, 1].
    verbose : bool
        Print per-generation summary.

    Returns
    -------
    best_theta : np.ndarray
        Best parameter vector found.
    history : list[dict]
        Per-generation stats (generation, best_fitness, mean, sigma).
    """
    optimizer = CMAES(n_params=n_params, sigma=sigma, seed=seed, clip_to_unit=clip_to_unit)
    history: list[dict] = []

    for gen in range(n_generations):
        candidates = optimizer.ask()
        fitnesses = [eval_fn(c) for c in candidates]
        optimizer.tell(candidates, fitnesses)

        summary = optimizer.summary()
        history.append(summary)

        if verbose:
            print(
                f"  CMA-ES gen {gen+1:3d} | best={summary['best_fitness']:.4f} "
                f"| sigma={summary['sigma']:.4f} "
                f"| mean={[round(x, 3) for x in summary['mean']]}"
            )

    return optimizer.best_solution, history
