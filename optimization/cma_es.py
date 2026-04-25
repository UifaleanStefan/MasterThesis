"""
CMA-ES — public entry point.

This module preserves the historical API (``CMAES`` class and
``run_cmaes_optimization`` function) while transparently picking the best
available backend:

  * **pycma** (`cma` library, default when installed) — the reference
    implementation by Hansen et al. Provides BIPOP-CMA-ES restarts,
    automatic stop-condition handling, and well-tested eigendecomposition.
  * **Minimal** (``optimization.cma_es_minimal``) — pure-numpy fallback
    used when pycma is unavailable. Kept as the thesis pedagogy reference.

Override the backend with the ``CMA_ES_BACKEND`` env var:
  * ``CMA_ES_BACKEND=pycma`` (default when installed)
  * ``CMA_ES_BACKEND=minimal``

Public API (unchanged from the minimal implementation):

    optimizer = CMAES(n_params=10, sigma=0.3, seed=42, clip_to_unit=True)
    candidates = optimizer.ask()
    optimizer.tell(candidates, fitnesses)
    best_x, best_f = optimizer.best_solution, optimizer.best_fitness
    info = optimizer.summary()

    best_theta, history = run_cmaes_optimization(
        eval_fn, n_params=10, n_generations=30, sigma=0.3, seed=42,
    )
"""

from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import Callable

import numpy as np

from optimization.cma_es_minimal import CMAES as CMAESMinimal
from optimization.cma_es_minimal import run_cmaes_optimization as _run_cmaes_minimal

try:
    import cma as _pycma
    _HAS_PYCMA = True
except ImportError:
    _HAS_PYCMA = False


def _pick_backend() -> str:
    """Return either 'pycma' or 'minimal' based on availability and env override."""
    requested = os.environ.get("CMA_ES_BACKEND", "").lower()
    if requested == "minimal":
        return "minimal"
    if requested == "pycma":
        if not _HAS_PYCMA:
            raise RuntimeError(
                "CMA_ES_BACKEND=pycma but the `cma` package is not installed. "
                "Install with `pip install cma>=3.3` or unset CMA_ES_BACKEND."
            )
        return "pycma"
    # Default: prefer pycma when available, fall back to minimal otherwise.
    return "pycma" if _HAS_PYCMA else "minimal"


class CMAESPyCma:
    """
    Thin wrapper over ``cma.CMAEvolutionStrategy`` exposing the same API as
    the minimal implementation. Pycma minimizes a cost; this wrapper negates
    the user's fitness so the convention "higher is better" is preserved
    end-to-end.
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
        self.clip_to_unit = clip_to_unit

        x0 = (np.full(n_params, 0.5, dtype=np.float64)
              if mean is None
              else np.asarray(mean, dtype=np.float64).copy())

        # pycma seed=0 is reserved (means "use a random seed"); coerce to a positive int.
        opts: dict = {
            "seed": int(seed) if seed > 0 else 1,
            "verbose": -9,  # silence pycma's per-generation prints
        }
        if clip_to_unit:
            opts["bounds"] = [[0.0] * n_params, [1.0] * n_params]

        self._es = _pycma.CMAEvolutionStrategy(x0.tolist(), float(sigma), opts)
        self._last_candidates: list[np.ndarray] = []

        # best_solution / best_fitness are kept in sync with es.best
        self.best_solution: np.ndarray = x0.copy()
        self.best_fitness: float = -np.inf

    # ------------------------------------------------------------------
    # Public API mirroring the minimal CMAES
    # ------------------------------------------------------------------

    def ask(self) -> list[np.ndarray]:
        candidates = [np.asarray(c, dtype=np.float64) for c in self._es.ask()]
        self._last_candidates = candidates
        return candidates

    def tell(self, candidates: list[np.ndarray], fitnesses: list[float]) -> None:
        # pycma minimizes; we maximize, so flip sign for the optimizer.
        costs = [-float(f) for f in fitnesses]
        self._es.tell([np.asarray(c, dtype=np.float64) for c in candidates], costs)

        # Keep best-so-far in sync (note: es.best.f is the negated cost)
        if self._es.best is not None and self._es.best.x is not None:
            best_x = np.asarray(self._es.best.x, dtype=np.float64)
            best_f = -float(self._es.best.f)
            if best_f > self.best_fitness:
                self.best_fitness = best_f
                self.best_solution = best_x

    def summary(self) -> dict:
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "mean": self.mean.tolist(),
            "sigma": self.sigma,
        }

    # ------------------------------------------------------------------
    # Properties exposing pycma's internal state under the legacy names
    # ------------------------------------------------------------------

    @property
    def generation(self) -> int:
        return int(self._es.countiter)

    @property
    def mean(self) -> np.ndarray:
        return np.asarray(self._es.mean, dtype=np.float64)

    @property
    def sigma(self) -> float:
        return float(self._es.sigma)

    @property
    def lam(self) -> int:
        return int(self._es.popsize)


_BACKEND = _pick_backend()


def CMAES(*args, **kwargs):
    """Construct a CMA-ES optimizer using the active backend."""
    if _BACKEND == "pycma":
        return CMAESPyCma(*args, **kwargs)
    return CMAESMinimal(*args, **kwargs)


def get_active_backend() -> str:
    """Return the active CMA-ES backend ('pycma' or 'minimal')."""
    return _BACKEND


# ----------------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------------

def save_checkpoint(
    optimizer,
    path: str | Path,
    history: list[dict] | None = None,
    extra: dict | None = None,
) -> None:
    """
    Save CMA-ES optimizer state plus per-generation history to ``path``.

    The payload format is a versioned dict so future optimizer additions
    can be carried forward without breaking older checkpoints.

    Parameters
    ----------
    optimizer : CMAESPyCma | CMAESMinimal
        The optimizer to checkpoint.
    path : str | Path
        Destination filename. Parent directory is created if needed.
    history : list[dict], optional
        Per-generation summary dicts produced by ``optimizer.summary()``.
    extra : dict, optional
        Caller-provided fields (CLI args, run_id, etc.) preserved alongside
        the optimizer state for context.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "version": 1,
        "history": list(history or []),
        "extra": dict(extra or {}),
    }
    if isinstance(optimizer, CMAESPyCma):
        payload["backend"] = "pycma"
        payload["pycma_pickle"] = optimizer._es.pickle_dumps()
        payload["best_solution"] = optimizer.best_solution.tolist()
        payload["best_fitness"] = float(optimizer.best_fitness)
        payload["n"] = optimizer.n
        payload["clip_to_unit"] = optimizer.clip_to_unit
    else:  # CMAESMinimal — pickleable directly
        payload["backend"] = "minimal"
        payload["optimizer_pickle"] = pickle.dumps(optimizer)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path: str | Path) -> tuple[object, list[dict], dict]:
    """
    Restore an optimizer + history + extra context from a checkpoint file.

    Returns
    -------
    (optimizer, history, extra)
        ``optimizer`` is a reconstructed CMAES instance (backend matches the
        one originally used). ``history`` is the per-generation list saved
        with it. ``extra`` is the caller-provided context dict.
    """
    path = Path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("version") != 1:
        raise ValueError(f"Unsupported checkpoint version: {payload.get('version')}")
    history = list(payload.get("history", []))
    extra = dict(payload.get("extra", {}))
    backend = payload["backend"]
    if backend == "pycma":
        if not _HAS_PYCMA:
            raise RuntimeError(
                "Checkpoint was saved with the pycma backend but the `cma` "
                "package is not installed in the current environment."
            )
        optimizer = CMAESPyCma.__new__(CMAESPyCma)
        # pycma exposes ``pickle_dumps()`` (returns bytes) but no matching
        # classmethod; the standard pickle.loads round-trips the instance.
        optimizer._es = pickle.loads(payload["pycma_pickle"])
        optimizer.best_solution = np.asarray(payload["best_solution"], dtype=np.float64)
        optimizer.best_fitness = float(payload["best_fitness"])
        optimizer.n = int(payload["n"])
        optimizer.clip_to_unit = bool(payload["clip_to_unit"])
        optimizer._last_candidates = []
        return optimizer, history, extra
    if backend == "minimal":
        optimizer = pickle.loads(payload["optimizer_pickle"])
        return optimizer, history, extra
    raise ValueError(f"Unrecognized backend in checkpoint: {backend!r}")


def run_cmaes_optimization(
    eval_fn: Callable[[np.ndarray], float],
    n_params: int = 3,
    n_generations: int = 20,
    sigma: float = 0.3,
    seed: int = 0,
    clip_to_unit: bool = True,
    verbose: bool = True,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 0,
    resume_from: str | Path | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Convenience wrapper: run CMA-ES for ``n_generations``, return
    ``(best_theta, history)``.

    When pycma is the active backend, this delegates to the same loop the
    minimal implementation uses (so caller-visible behavior is unchanged).
    The minimal pure-numpy version is invoked when pycma is unavailable or
    explicitly disabled via ``CMA_ES_BACKEND=minimal``.

    Checkpointing:
        - ``checkpoint_path``: file where per-generation state is written.
        - ``checkpoint_every``: write checkpoint every K generations (0 = never).
        - ``resume_from``: load checkpoint and continue from saved generation.
    """
    if resume_from is not None:
        optimizer, history, _extra = load_checkpoint(resume_from)
        already_done = len(history)
        if verbose:
            print(f"  [resume] loaded {already_done} gens from {resume_from}")
    elif _BACKEND == "minimal":
        # Minimal backend doesn't support checkpoint mid-run via this entry
        # point; fall through to its native loop unless checkpointing requested.
        if checkpoint_every <= 0 and checkpoint_path is None:
            return _run_cmaes_minimal(
                eval_fn,
                n_params=n_params,
                n_generations=n_generations,
                sigma=sigma,
                seed=seed,
                clip_to_unit=clip_to_unit,
                verbose=verbose,
            )
        optimizer = CMAESMinimal(n_params=n_params, sigma=sigma, seed=seed,
                                 clip_to_unit=clip_to_unit)
        history = []
    else:
        optimizer = CMAESPyCma(
            n_params=n_params,
            sigma=sigma,
            seed=seed,
            clip_to_unit=clip_to_unit,
        )
        history = []

    for gen in range(len(history), n_generations):
        candidates = optimizer.ask()
        fitnesses = [eval_fn(c) for c in candidates]
        optimizer.tell(candidates, fitnesses)
        info = optimizer.summary()
        history.append(info)
        if verbose:
            print(
                f"  CMA-ES gen {gen+1:3d} | best={info['best_fitness']:.4f} "
                f"| sigma={info['sigma']:.4f} "
                f"| mean={[round(x, 3) for x in info['mean']]}"
            )
        if checkpoint_path and checkpoint_every > 0 and (gen + 1) % checkpoint_every == 0:
            save_checkpoint(optimizer, checkpoint_path, history=history)
            if verbose:
                print(f"  [checkpoint] saved gen {gen+1} -> {checkpoint_path}")
    # Always save final checkpoint if a path was specified
    if checkpoint_path:
        save_checkpoint(optimizer, checkpoint_path, history=history)
    return optimizer.best_solution, history
