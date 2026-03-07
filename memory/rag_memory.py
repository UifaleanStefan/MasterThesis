"""
RAGMemory — vector-database style retrieval using real sentence embeddings.

Uses sentence-transformers `all-MiniLM-L6-v2` (~80MB, CPU-friendly).
Stores all events with their dense embeddings.
At retrieval: embed current observation, return top-k by cosine similarity.

No graph structure, no rules, no entity extraction.
This is the closest analog to production LLM agents using Pinecone / Chroma / FAISS.

The model is loaded once as a module-level singleton to avoid re-loading cost.
Falls back to the TF-IDF embedder from memory/embedding.py if sentence-transformers
is not installed, so the rest of the pipeline still runs.
"""

import numpy as np

from .event import Event

# Best-effort import of sentence-transformers. In some environments this can
# fail with runtime errors (e.g. Keras/tf-keras incompatibilities) even when
# the package is installed, so we treat *any* exception as a signal to fall
# back to the lightweight TF-IDF embedder.
_HAS_ST = False
_MODEL = None  # type: ignore[attr-defined]
_ST_ERROR: str | None = None

try:
    from sentence_transformers import SentenceTransformer as _ST  # type: ignore[assignment]
    try:
        _MODEL = _ST("all-MiniLM-L6-v2")
        _HAS_ST = True
    except Exception as e:  # pragma: no cover - defensive against env issues
        _MODEL = None  # type: ignore[assignment]
        _HAS_ST = False
        _ST_ERROR = repr(e)
except Exception as e:  # pragma: no cover - defensive against env issues
    _HAS_ST = False
    _MODEL = None  # type: ignore[assignment]
    _ST_ERROR = repr(e)


def _embed(text: str) -> np.ndarray:
    global _HAS_ST, _ST_ERROR
    if _HAS_ST and _MODEL is not None:
        try:
            return _MODEL.encode(text, convert_to_numpy=True).astype(np.float32)
        except Exception as e:  # pragma: no cover - defensive
            _HAS_ST = False
            _ST_ERROR = repr(e)
    # Fallback to TF-IDF embedder
    from .embedding import embed_observation
    return embed_observation(text)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class RAGMemory:
    """
    Vector-store memory: dense embeddings + cosine retrieval.

    Stores every event with its sentence embedding.
    Retrieval is purely similarity-based — no graph, no rules, no recency bias.
    """

    def __init__(self) -> None:
        self._store: list[tuple[Event, np.ndarray]] = []

    def add_event(self, event: Event, episode_seed: int | None = None) -> None:
        emb = _embed(event.observation)
        self._store.append((event, emb))

    def get_relevant_events(
        self,
        observation: str,
        current_step: int,
        k: int = 8,
    ) -> list[Event]:
        if not self._store:
            return []
        query_emb = _embed(observation)
        scored = [(event, _cosine_sim(query_emb, emb)) for event, emb in self._store]
        scored.sort(key=lambda x: -x[1])
        return [e for e, _ in scored[:k]]

    def clear(self) -> None:
        self._store.clear()

    def get_stats(self) -> dict:
        return {
            "n_events": len(self._store),
            "n_entities": 0,
            "n_nodes": len(self._store),
            "n_edges": 0,
        }

    @property
    def using_sentence_transformers(self) -> bool:
        return _HAS_ST
