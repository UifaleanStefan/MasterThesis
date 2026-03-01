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

try:
    from sentence_transformers import SentenceTransformer as _ST
    _MODEL = _ST("all-MiniLM-L6-v2")
    _HAS_ST = True
except ImportError:
    _HAS_ST = False
    _MODEL = None  # type: ignore


def _embed(text: str) -> np.ndarray:
    if _HAS_ST and _MODEL is not None:
        return _MODEL.encode(text, convert_to_numpy=True).astype(np.float32)
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
