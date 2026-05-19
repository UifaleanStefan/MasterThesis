"""
BM25Memory — sparse-retrieval baseline (Phase 1.7 honesty pass).

Implements the 4-method memory contract (`add_event`, `get_relevant_events`,
`clear`, `get_stats`) using rank-bm25's Okapi BM25 over event observations.

Why this exists: the Phase 1.5 retrieval study compared V4-tuned against
12 *memory-system* baselines (FlatWindow, RAGMemory, AttentionMemory,
etc.), but none of them are the sparse-retrieval industry standard. BM25
is. Without a BM25 comparison, the "V4-tuned wins on long-haystack"
claim is restricted to "V4-tuned beats this curated set of in-house
memory systems" — not "V4-tuned advances the SOTA for long-context QA".

BM25 is also a strong baseline historically: on factual/extractive QA
it often matches or beats dense retrievers when the question shares
lexical vocabulary with the gold passage. We expect it to be
particularly strong on:
  * FinanceBench (numeric + named-entity-heavy questions)
  * HotpotQA (Wikipedia-style entity overlap)
And to lag on:
  * QASPER (paraphrased scientific questions vs technical paragraphs)
  * NarrativeQA (multi-sentence reasoning over long fiction)

Storage cost is O(N) text + O(N × |vocab|) inverted index; rebuild on
every retrieval call is O(N) which matches our small per-doc N (under
2000 paragraphs for NarrativeQA, the largest case).
"""

from __future__ import annotations

import re
from typing import Any

from memory.event import Event

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Cheap, deterministic tokenizer — lowercase alphanumeric only.

    BM25 quality is fairly robust to tokenizer choice; we avoid heavier
    options (spaCy, transformers) to keep this baseline transparent and
    fast to recompute.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Memory:
    """Sparse-retrieval memory using Okapi BM25 over event observations.

    Satisfies the 4-method memory contract identical to other memory
    systems in this repo (see `memory/rag_memory.py` for the canonical
    template).
    """

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._tokens: list[list[str]] = []
        self._bm25 = None  # built lazily on first retrieval

    def add_event(self, event: Event, episode_seed: int | None = None) -> None:
        """Append the event; invalidate the BM25 index (rebuild on next query).

        The `episode_seed` parameter is part of the memory contract but
        BM25 has no per-event randomness, so we ignore it.
        """
        self._events.append(event)
        self._tokens.append(_tokenize(event.observation))
        self._bm25 = None  # invalidate

    def _rebuild_index(self) -> None:
        from rank_bm25 import BM25Okapi
        # rank_bm25 doesn't gracefully handle empty corpora; guard.
        if not self._tokens:
            self._bm25 = None
            return
        # Avoid all-empty token lists (would crash BM25).
        safe = [t if t else ["__empty__"] for t in self._tokens]
        self._bm25 = BM25Okapi(safe)

    def get_relevant_events(
        self, observation: str, current_step: int, k: int = 8,
    ) -> list[Event]:
        """Return top-k events by BM25 score against the query observation."""
        if self._bm25 is None:
            self._rebuild_index()
        if self._bm25 is None or not self._events:
            return []
        query = _tokenize(observation)
        if not query:
            # No content in query — fall back to the k most recent events,
            # matching how SemanticMemory and others handle this edge case.
            return self._events[-k:] if self._events else []
        scores = self._bm25.get_scores(query)
        # Sort by score descending; ties broken by insertion order (stable).
        ranked = sorted(
            range(len(self._events)),
            key=lambda i: (-scores[i], i),
        )
        return [self._events[i] for i in ranked[:k]]

    def clear(self) -> None:
        self._events = []
        self._tokens = []
        self._bm25 = None

    def get_stats(self) -> dict[str, Any]:
        return {
            "n_events": len(self._events),
            "n_entities": 0,
            "n_nodes": len(self._events),
            "n_edges": 0,
        }
