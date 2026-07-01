"""
LettaMemory - faithful reduction of the MemGPT / Letta tiered memory model
for a STATIC corpus-QA head-to-head.

Real system (MemGPT, Packer et al. 2023, arXiv:2310.08560; Letta, 2024-2026)
---------------------------------------------------------------------------
MemGPT/Letta wrap an LLM in an OS-style "virtual context manager" with a
hierarchical, tiered memory (https://docs.letta.com):

  * Main context (working memory / "core memory") - the bounded slice of
    tokens the model sees every turn: system prompt, editable core-memory
    blocks, and the most recent messages. Always surfaced.
  * Recall memory - the searchable log of prior conversation messages
    (recent history), queried on demand.
  * Archival memory - an external vector store of long-term facts, embedded
    (Letta's default local encoder is BAAI/bge-small-en-v1.5) and retrieved by
    semantic similarity via the `archival_memory_search` tool.

The agent itself pages information between these tiers with function calls
(`archival_memory_insert`, `archival_memory_search`, core-memory edits), i.e.
it self-manages what to store, retrieve, and evict - the defining feature of
MemGPT.

Faithful reduction to our retrieval-only interface
--------------------------------------------------
Our head-to-head holds the answerer (gpt-4o-mini) FIXED and only swaps the
retriever, so we cannot and do not replicate the *agentic* self-editing loop.
What we DO replicate faithfully is Letta's tiered retrieval surface for a
static corpus:

  1. A small recency "core" tier: the last ``core_window`` ingested passages
     are ALWAYS surfaced (mirrors the always-present recent-message /
     core-memory slice of main context).
  2. An embedding-retrieved archival tier: everything else is ranked by MiniLM
     cosine similarity to the query (mirrors `archival_memory_search`).

A retrieval of size ``k`` returns the ``core_window`` most-recent events UNION
the top archival events needed to reach ``k`` (deduplicated). This is exactly
the union of "what main context always holds" and "what the agent would page
in from archival for this query".

Relationship to our other configs (disclosed in the thesis)
-----------------------------------------------------------
This is deliberately close to the pure embedding-kNN retriever (our
``semantic`` / RAGMemory config): with ``core_window = 0`` it reduces exactly
to top-k cosine retrieval. The ONLY structural difference is the always-on
recency core, which is the faithful trace of Letta's main-context tier. We
measure Letta's ARCHIVAL RETRIEVAL mechanism, not its agentic memory
management (which is what makes MemGPT novel and which we deliberately do NOT
replicate).

Cost / speed
------------
One MiniLM embedding per passage at ingest (LRU-cached, no LLM call). Query
time is a single embedding + one vectorized cosine over the stored matrix -
O(N*384) - which is fine for 5k-20k events on CPU. No per-event LLM calls at
any point. Entities are extracted with the non-LLM ``extract_entities`` only
to populate ``get_stats`` (kept cheap; not used for ranking), matching the
stats contract of the other adapters.
"""

from __future__ import annotations

import numpy as np

from .embedding import embed_observation
from .entity_extraction import extract_entities
from .event import Event


class LettaMemory:
    """Two-tier (recency-core + embedding-archival) retriever, MemGPT/Letta style.

    Parameters
    ----------
    core_window:
        Number of most-recently-ingested passages that are ALWAYS surfaced,
        modelling Letta's main-context recent-message / core-memory slice.
        Letta keeps a bounded recent window in main context; a small value
        (default 4) reflects that only a handful of recent items co-reside with
        the archival hits under a realistic token budget. Set to 0 to recover a
        pure embedding-kNN retriever.
    track_entities:
        If True, run the cheap non-LLM entity extractor at ingest so
        ``get_stats`` reports a meaningful ``n_entities`` (parity with the graph
        adapters' stats). Ranking never uses entities. Disable to shave ingest
        cost on very large corpora.
    """

    def __init__(
        self,
        core_window: int = 4,
        track_entities: bool = True,
    ) -> None:
        self._core_window = max(0, int(core_window))
        self._track_entities = bool(track_entities)

        self._events: list[Event] = []
        # Embeddings kept as a single (N, D) matrix, L2-normalized rows, so the
        # archival search is one matmul. Rebuilt lazily when invalidated.
        self._emb_rows: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None  # (N, D) normalized, lazy
        self._seen_entities: set[str] = set()

    # ------------------------------------------------------------------ ingest
    def add_event(self, event: Event, episode_seed: int | None = None) -> None:
        """Ingest one passage into archival storage (append-only).

        ``episode_seed`` is part of the memory contract but this retriever has
        no per-event randomness, so it is ignored.
        """
        emb = embed_observation(event.observation).astype(np.float32)
        self._events.append(event)
        self._emb_rows.append(emb)
        self._matrix = None  # invalidate the cached normalized matrix

        if self._track_entities:
            self._seen_entities.update(extract_entities(event.observation))

    def _ensure_matrix(self) -> None:
        """Materialize the L2-normalized embedding matrix if stale."""
        if self._matrix is not None or not self._emb_rows:
            return
        mat = np.vstack(self._emb_rows).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0  # avoid divide-by-zero for empty embeddings
        self._matrix = mat / norms

    # --------------------------------------------------------------- retrieval
    def get_relevant_events(
        self,
        observation: str,
        current_step: int,
        k: int = 8,
    ) -> list[Event]:
        """Return the recency-core union top-archival events, capped at ``k``.

        Tier 1 (always-on core): the ``min(core_window, k)`` most-recent events.
        Tier 2 (archival search): fill the remaining budget with the highest
        cosine-similarity events not already taken by the core tier.

        The returned list is ordered core-first (most-recent -> older), then
        archival hits by descending similarity - so the always-present context
        leads, exactly as it would sit in Letta's main context ahead of the
        paged-in archival results.
        """
        n = len(self._events)
        if n == 0 or k <= 0:
            return []

        # ---- Tier 1: recency core (last core_window ingested, newest first) --
        core_budget = min(self._core_window, k)
        core_idx: list[int] = []
        if core_budget > 0:
            core_idx = list(range(n - 1, n - 1 - core_budget, -1))
        core_set = set(core_idx)

        remaining = k - len(core_idx)
        if remaining <= 0:
            return [self._events[i] for i in core_idx]

        # ---- Tier 2: archival embedding search over the non-core events ------
        self._ensure_matrix()
        q = embed_observation(observation).astype(np.float32)
        qn = np.linalg.norm(q)
        if qn == 0.0 or self._matrix is None:
            # Degenerate query (e.g. empty/OOV under TF-IDF fallback): mirror
            # the other adapters and back-fill with the next most-recent events.
            archival_idx = [i for i in range(n - 1, -1, -1) if i not in core_set]
            picked = archival_idx[:remaining]
            order = core_idx + picked
            return [self._events[i] for i in order]

        sims = self._matrix @ (q / qn)  # (N,) cosine similarity

        # Rank all events by similarity; skip those already in the core tier.
        # argsort is stable, so equal-similarity ties break by earlier index.
        ranked = np.argsort(-sims, kind="stable")
        archival_pick: list[int] = []
        for i in ranked:
            ii = int(i)
            if ii in core_set:
                continue
            archival_pick.append(ii)
            if len(archival_pick) >= remaining:
                break

        order = core_idx + archival_pick
        return [self._events[i] for i in order]

    # -------------------------------------------------------------- lifecycle
    def clear(self) -> None:
        self._events = []
        self._emb_rows = []
        self._matrix = None
        self._seen_entities = set()

    def get_stats(self) -> dict:
        return {
            "n_events": len(self._events),
            "n_entities": len(self._seen_entities),
            "n_nodes": len(self._events),
            "n_edges": 0,
        }
