"""
Event embedding — semantic by default, TF-IDF as legacy fallback.

The thesis originally used a 31-word TF-IDF vocabulary covering grid-world
tokens (colors, doors, signs, directions). That vocab is task-specific by
construction, which contradicts the "memory must learn from reward alone"
principle and silently zeroes out embeddings on natural-language inputs
like DocumentQA paragraphs.

This module provides two backends behind the same public API:

  * **sentence-transformers** (default) — uses ``all-MiniLM-L6-v2``, a 384-dim
    pretrained sentence encoder. Same model already used by RAGMemory, so no
    new dependency is introduced. Embeddings are LRU-cached because
    deterministic-seed runs revisit identical observation strings frequently.

  * **tfidf** (legacy) — the original 31-token TF-IDF over a fixed grid-world
    vocabulary. Selected via ``EMBEDDING_BACKEND=tfidf``; produces the
    pre-PoC-hardening numbers verbatim.

Public API (unchanged):

    vec = embed_observation("you see a red key")
    # Returns np.ndarray of shape (D,) where D depends on backend.
    # Cosine similarity is the only consumer-side operation, so the dim is
    # opaque to memory variants.

Selection precedence (high → low):
  1. ``EMBEDDING_BACKEND`` env var (case-insensitive: "sentence-transformers"
     or "tfidf").
  2. Default: try sentence-transformers; if the model can't load, fall back
     to TF-IDF and emit a one-time warning.

Reproducibility note: switching backends invalidates every numeric result
that touches embeddings (V4 CMA-ES optimum, ablation, transfer, sensitivity,
neural V2 weights). The active backend is recorded in the per-run manifest
(see ``results/manifest.py``) so legacy reproductions remain identifiable.
"""

from __future__ import annotations

import functools
import os
import warnings
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

# Public for backwards compatibility — the neural controllers use this list
# directly to size their input layer (they intentionally keep a small,
# deterministic TF-IDF feature vector regardless of the active backend).
VOCAB = [
    "you", "are", "in", "a", "room", "see", "red", "blue", "green",
    "key", "door", "nothing", "interest", "carrying", "of", "goal",
    "yellow", "purple", "opened", "have", "doors",
    "orange", "cyan", "magenta", "white",
    "sign", "opens", "north", "east", "south", "requires",
]
_VOCAB = VOCAB  # internal alias kept for clarity below

_st_model: Any = None       # sentence-transformers model (lazy)
_tfidf_vec: Any = None      # TfidfVectorizer (lazy)
_active_backend: str | None = None
_st_load_failed: bool = False


def _resolve_backend() -> str:
    """Return the active backend name ('sentence-transformers' or 'tfidf')."""
    global _active_backend
    if _active_backend is not None:
        return _active_backend
    requested = os.environ.get("EMBEDDING_BACKEND", "").strip().lower()
    if requested == "tfidf":
        _active_backend = "tfidf"
        return _active_backend
    if requested in ("sentence-transformers", "sentence_transformers", "minilm"):
        _active_backend = "sentence-transformers"
        return _active_backend
    # Default: prefer sentence-transformers; the lazy loader will fall back
    # if the model fails to initialize.
    _active_backend = "sentence-transformers"
    return _active_backend


def get_active_backend() -> str:
    """Return the embedding backend currently in use."""
    return _resolve_backend()


def _get_tfidf():
    """Build the legacy TF-IDF vectorizer the first time it's requested."""
    global _tfidf_vec
    if _tfidf_vec is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        _tfidf_vec = TfidfVectorizer(
            vocabulary={w: i for i, w in enumerate(_VOCAB)},
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        _tfidf_vec.fit([" ".join(_VOCAB)])
    return _tfidf_vec


def _get_st_model():
    """Lazily load the sentence-transformers model. On failure, fall back to TF-IDF."""
    global _st_model, _st_load_failed, _active_backend
    if _st_model is not None:
        return _st_model
    if _st_load_failed:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        return _st_model
    except Exception as exc:
        _st_load_failed = True
        warnings.warn(
            f"sentence-transformers unavailable ({exc!r}); "
            "falling back to TF-IDF embedding. Set EMBEDDING_BACKEND=tfidf "
            "explicitly to silence this warning.",
            stacklevel=2,
        )
        _active_backend = "tfidf"
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8192)
def _embed_st(observation: str) -> np.ndarray:
    """Cached sentence-transformers encode."""
    model = _get_st_model()
    if model is None:
        return _embed_tfidf(observation)
    vec = model.encode([observation], convert_to_numpy=True, show_progress_bar=False)
    return vec[0].astype(np.float32)


@functools.lru_cache(maxsize=8192)
def _embed_tfidf(observation: str) -> np.ndarray:
    """Cached TF-IDF encode (legacy)."""
    vec = _get_tfidf()
    X = vec.transform([observation])
    return X.toarray().flatten().astype(np.float32)


def embed_observation(observation: str) -> np.ndarray:
    """
    Embed an observation string into a fixed-size dense vector.

    Default backend: ``sentence-transformers/all-MiniLM-L6-v2`` (384-dim).
    Legacy backend (``EMBEDDING_BACKEND=tfidf``): 31-dim TF-IDF.

    The output is a numpy float32 1-D array. Downstream consumers use only
    cosine similarity, so the exact dimensionality is opaque to memory
    variants and neural controllers.

    Embeddings are LRU-cached on the input string. Most observations repeat
    across deterministic-seed reruns, so the cache typically eliminates >90%
    of redundant encoder calls in CMA-ES eval loops.
    """
    backend = _resolve_backend()
    if backend == "tfidf":
        return _embed_tfidf(observation)
    return _embed_st(observation)


def embed_observation_tfidf(observation: str) -> np.ndarray:
    """
    Force-TF-IDF embedding regardless of backend selection.

    Used by the neural controllers (memory/neural_controller*.py), which take
    a small fixed-size feature vector and would otherwise need to be retrained
    with a much larger input layer when the global backend switches to
    sentence-transformers. Keeping their feature stream on TF-IDF preserves
    their parameter count and decouples controller training from the storage
    embedding choice.
    """
    return _embed_tfidf(observation)


def reset_caches() -> None:
    """Drop the LRU caches. Useful in tests that switch backends mid-process."""
    _embed_st.cache_clear()
    _embed_tfidf.cache_clear()


def reset_backend() -> None:
    """
    Reset the cached backend selection so a subsequent call re-reads
    ``EMBEDDING_BACKEND``. Test-only utility.
    """
    global _active_backend, _st_model, _st_load_failed
    _active_backend = None
    _st_model = None
    _st_load_failed = False
    reset_caches()
