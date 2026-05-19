"""
Layer 2 — V4 retrieval smoke for the six Stage 3 benchmark adapters.

For each adapter:
  1. Pull 1 document via iter_documents(limit=1).
  2. Build a canonical GraphMemoryV4 (post-MiniLM thesis-default θ).
  3. Run the reading phase: each paragraph → Event with step = paragraph_index.
  4. For each qa_pair, retrieve top-k=8 and compute recall@k against
     relevant_paragraphs.
  5. Assert the median recall@k clears the per-benchmark threshold AND
     that total chars retrieved per question stays under the prompt-budget
     ceiling (early-warning probe for Phase-4 token blowups).

Thresholds: mean recall@k=8 across qa_pairs WITH non-empty
relevant_paragraphs, aggregated across `limit=3` documents per
benchmark. Calibrated conservatively against initial V4-with-MiniLM
observations — the V4 theta was tuned for grid-world tasks, not science
QA, so retrieval on QASPER/CUAD is necessarily weaker than on HotpotQA
where Wikipedia content shares vocabulary with the questions.

Why filter to non-empty gold: qa_pairs with empty relevant_paragraphs
(QASPER's evidence-substring miss path, NarrativeQA always) cannot
meaningfully contribute to recall — empty gold = 0 recall by
construction, not a retrieval failure.

Why mean (not median): recall@k is a 0/1 binary metric per question.
Median is brittle (flips with one question's outcome); mean across
multiple qa_pairs is a stable retrieval-quality estimate.

Thresholds:
  * HotpotQA:     0.40   (Wikipedia, strong vocab overlap)
  * QASPER:       0.15   (sparse evidence, scientific abstract questions)
  * CUAD:         0.20   (templated questions, "highlight parts related to X")
  * NarrativeQA:  SKIP   (no gold; LLM judge handles this benchmark)
  * FinanceBench: 0.80   (every paragraph is gold by construction)
  * LongMemEval:  0.30   (long dialogue sessions, sparse gold)

Also asserts: total_chars_retrieved_per_question < 50_000 (prompt-budget guard).

NB: This test exercises the DocumentQA + V4 retrieval path on real
benchmark data WITHOUT calling any LLM. Run time ~30-60s on warm HF
cache. Should be CI-fast.
"""

from __future__ import annotations

import os
import statistics

import pytest

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from environment.benchmarks import get_adapter
from environment.document_qa import DocumentQA
from evaluation.document_qa_memory import _recall_at_k_for_qa, _run_reading_phase
from memory.event import Event
from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4


# Recall thresholds — mean recall@k=8 across qa_pairs WITH non-empty
# relevant_paragraphs, aggregated over `SMOKE_DOC_LIMIT` documents per
# benchmark. NarrativeQA = None means SKIP recall assertion (no gold).
RECALL_THRESHOLD: dict[str, float | None] = {
    "hotpotqa":     0.40,
    "qasper":       0.15,
    "cuad":         0.20,
    "narrativeqa":  None,
    "financebench": 0.80,
    "longmemeval":  0.30,
}

# Number of documents to aggregate per benchmark in the smoke test.
# More docs = more stable mean; cap at 3 to keep CI fast.
SMOKE_DOC_LIMIT = 3

# Char-budget guard — if any qa_pair retrieves > this much across top-k,
# the LLM prompt cost will explode in Phase 4. Fail early.
MAX_RETRIEVED_CHARS_PER_QUESTION = 50_000


# V4 smoke-test params — embedding-only baseline.
#
# Why not the canonical post-MiniLM theta from
# `evaluation/document_qa_memory._make_document_qa_memory_systems`?
# That theta has w_recency=3.777, w_embed=1.079, theta_store=0.293 — tuned
# for grid-world (MultiHop-KeyDoor) where recent observations carry the
# task signal. On static document QA where all paragraphs are read in
# order, that bias collapses retrieval onto the last-k paragraphs
# regardless of question (verified empirically — see
# `scripts/debug_qasper_retrieval.py`).
#
# The smoke test's job is to validate the ADAPTER + DocumentQA pipeline,
# not to relitigate V4's theta tuning. Hence: V4 with all events stored
# (theta_store=0) and pure-embedding retrieval (w_recency=0, w_embed=1).
# Phase 4 will compare multiple thetas + memory systems on these adapters.
_V4_SMOKE_PARAMS = MemoryParamsV4(
    theta_store=0.0,   # store every paragraph (no novelty filter)
    theta_novel=0.0, theta_erich=0.0, theta_surprise=0.0,
    theta_entity=0.0, theta_temporal=0.0, theta_decay=0.0,
    w_graph=0.0, w_embed=1.0, w_recency=0.0,
    mode="learnable",
)


def _build_v4_memory() -> GraphMemoryV4:
    return GraphMemoryV4(_V4_SMOKE_PARAMS)


def _run_episode_for_doc(
    doc: dict, memory: GraphMemoryV4, k: int = 8,
) -> tuple[list[float], list[int], list[int]]:
    """Run reading phase + retrieval loop for one document.

    Returns
    -------
    recalls : list[float]
        Per-qa_pair recall@k value (0.0 or 1.0 in the current discrete metric).
    chars_per_q : list[int]
        Sum of len(event.observation) across the top-k retrievals per qa_pair.
    gold_counts : list[int]
        len(relevant_paragraphs) per qa_pair. Used to filter to qa_pairs
        with gold when computing the threshold.
    """
    env = DocumentQA(document=doc, seed=42, question_shuffle=False)
    _run_reading_phase(env, memory, episode_seed=42)
    n_paragraphs = len(doc["paragraphs"])
    qa_pairs = doc["qa_pairs"]

    recalls: list[float] = []
    chars_per_q: list[int] = []
    gold_counts: list[int] = []
    for qa_idx, qa in enumerate(qa_pairs):
        question = qa["question"]
        relevant = qa.get("relevant_paragraphs", [])
        current_step = n_paragraphs + qa_idx
        r = _recall_at_k_for_qa(memory, question, relevant, k=k, current_step=current_step)
        recalls.append(r)
        retrieved = memory.get_relevant_events(question, current_step=current_step, k=k)
        chars_per_q.append(sum(len(e.observation) for e in retrieved))
        gold_counts.append(len(relevant))
    return recalls, chars_per_q, gold_counts


@pytest.mark.parametrize("name", sorted(RECALL_THRESHOLD.keys()))
def test_v4_retrieval_smoke(name: str) -> None:
    """V4 retrieval mean recall@k clears the per-benchmark threshold.

    Aggregates across SMOKE_DOC_LIMIT docs, filtered to qa_pairs with
    non-empty gold relevance.
    """
    adapter = get_adapter(name)
    docs = list(adapter.iter_documents(limit=SMOKE_DOC_LIMIT))
    assert len(docs) >= 1, f"[{name}] no docs returned"

    all_recalls: list[float] = []
    all_chars: list[int] = []
    all_gold_counts: list[int] = []
    for doc in docs:
        memory = _build_v4_memory()  # fresh memory per doc
        r, c, g = _run_episode_for_doc(doc, memory)
        all_recalls.extend(r)
        all_chars.extend(c)
        all_gold_counts.extend(g)

    # Char-budget guard always asserted (catches Phase-4 token blowups early).
    max_chars = max(all_chars) if all_chars else 0
    assert max_chars < MAX_RETRIEVED_CHARS_PER_QUESTION, (
        f"[{name}] retrieval pulled {max_chars:,} chars on one qa_pair "
        f"(budget = {MAX_RETRIEVED_CHARS_PER_QUESTION:,}). "
        f"Adapter paragraphs are too long — split more aggressively."
    )

    threshold = RECALL_THRESHOLD[name]
    if threshold is None:
        # NarrativeQA: no gold relevance; only the char-budget guard runs.
        for r in all_recalls:
            assert r == 0.0, f"[{name}] recall should be 0 (no gold), got {r}"
        return

    # Filter to qa_pairs with non-empty gold — only those test retrieval.
    eligible_recalls = [
        r for r, g in zip(all_recalls, all_gold_counts) if g > 0
    ]
    n_eligible = len(eligible_recalls)
    n_total = len(all_recalls)

    if n_eligible == 0:
        pytest.fail(
            f"[{name}] no qa_pairs across {len(docs)} docs had non-empty "
            f"relevant_paragraphs — adapter is not producing gold relevance."
        )

    mean_recall = statistics.mean(eligible_recalls)
    median_recall = statistics.median(eligible_recalls)
    print(
        f"\n  [{name}] n_docs={len(docs)} n_qa_total={n_total} "
        f"n_with_gold={n_eligible}  mean_recall@8={mean_recall:.3f}  "
        f"median={median_recall:.3f}  max_chars={max_chars:,}"
    )
    assert mean_recall >= threshold, (
        f"[{name}] mean recall@k=8 = {mean_recall:.3f} < threshold {threshold} "
        f"across {n_eligible}/{n_total} qa_pairs with gold. "
        f"Recalls: {[f'{r:.1f}' for r in eligible_recalls[:15]]}"
    )


def test_event_step_equals_paragraph_index() -> None:
    """Contract invariant: Event.step == paragraph index.

    The eval path depends on this — relevant_paragraphs indices are
    compared against Event.step values. Verify directly on the simplest
    adapter (FinanceBench).
    """
    adapter = get_adapter("financebench")
    doc = next(adapter.iter_documents(limit=1))
    memory = _build_v4_memory()
    env = DocumentQA(document=doc, seed=42, question_shuffle=False)
    _run_reading_phase(env, memory, episode_seed=42)

    # Inspect the events stored in memory — their step values should
    # cover [0, n_paragraphs) (subject to V4's theta_store filtering;
    # some events may be filtered out, but the steps that ARE stored
    # must be valid paragraph indices).
    stats = memory.get_stats()
    n_stored = stats.get("n_events", 0)
    n_paragraphs = len(doc["paragraphs"])
    # We don't assert exact equality (theta_store filters some events);
    # we just confirm the stored steps are all in valid range.
    retrieved = memory.get_relevant_events(
        doc["qa_pairs"][0]["question"], current_step=n_paragraphs, k=n_paragraphs
    )
    for ev in retrieved:
        assert isinstance(ev, Event)
        assert 0 <= ev.step < n_paragraphs, (
            f"Event.step={ev.step} out of paragraph range [0, {n_paragraphs})"
        )
