"""Tests for the corpus-mode baseline tuners (audit B6 fairness pass).

Covers: BM25Memory's new (k1, b) hyperparameters, and the recall@k eval
helpers used by tune_bm25_corpus / tune_attention_corpus. Uses a tiny
hand-built ingestion stream so the tests need no benchmark data.
"""
from __future__ import annotations

from memory.bm25_memory import BM25Memory
from memory.event import Event


def _ingest():
    # (global_step, observation) — step 0 and 2 are about cats/felines.
    return [
        (0, "the cat sat on the mat"),
        (1, "dogs run very fast in the park"),
        (2, "feline animals such as cats purr softly"),
        (3, "stock prices rose sharply this quarter"),
    ]


def test_bm25_accepts_k1_b_and_retrieves():
    mem = BM25Memory(k1=0.5, b=0.5)
    assert mem._k1 == 0.5 and mem._b == 0.5
    for step, obs in _ingest():
        mem.add_event(Event(step=step, observation=obs, action="read"))
    hits = mem.get_relevant_events("cat feline purr", current_step=4, k=2)
    steps = {e.step for e in hits}
    # The two cat/feline docs (0, 2) should be the top hits.
    assert steps == {0, 2}


def test_bm25_defaults_unchanged():
    # Default construction must still match rank-bm25 defaults (back-compat).
    mem = BM25Memory()
    assert mem._k1 == 1.5 and mem._b == 0.75


def test_bm25_k1_b_change_scores():
    # Different (k1, b) should generally produce a different score vector,
    # confirming the hyperparameters are actually threaded into BM25Okapi.
    ing = _ingest()
    q = "cats purr"

    def scores(k1, b):
        m = BM25Memory(k1=k1, b=b)
        for s, o in ing:
            m.add_event(Event(step=s, observation=o, action="read"))
        m._rebuild_index()
        from memory.bm25_memory import _tokenize
        return list(m._bm25.get_scores(_tokenize(q)))

    assert scores(0.5, 0.0) != scores(3.0, 1.0)


def test_bm25_recall_helper():
    from tuning.tune_bm25_corpus import recall_at_k
    ingest = _ingest()
    eval_tasks = [("cat feline purr", {0, 2}, 4)]
    assert recall_at_k(ingest, eval_tasks, k1=1.5, b=0.75, k=4) == 1.0
    # k=1 may still hit one of {0,2}; recall is hit-if-any-overlap.
    assert recall_at_k(ingest, eval_tasks, k1=1.5, b=0.75, k=1) in (0.0, 1.0)
