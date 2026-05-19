"""
Layer 1 — Adapter unit tests for the six Stage 3 benchmark adapters.

For each adapter we verify:
  * Schema correctness (required keys, types).
  * Non-empty content (title, paragraphs, qa_pairs).
  * Index validity (every relevant_paragraphs entry in
    [0, len(paragraphs))).
  * Trigram-overlap sanity: when relevant_paragraphs is non-empty, at
    least one indexed paragraph shares trigrams with the question or
    answer. Catches off-by-one and post-hoc-filter bugs that schema
    checks miss.
  * Determinism: repeated iter_documents(seed=42) calls yield identical
    output (byte-identical via document_fingerprint).
  * Shuffle determinism: iter_documents(seed=42, shuffle=True) is also
    stable across rerun.

The tests are PARAMETRIZED over the six benchmarks. NarrativeQA's
trigram check is relaxed (no gold relevance → no expected match).

These tests require the prefetch and verifier to have run successfully
(``data/benchmarks/`` populated). On a fresh checkout, run:

    python scripts/prefetch_benchmarks.py
    python scripts/verify_benchmarks.py
    python -m pytest tests/test_benchmark_adapters.py -v
"""

from __future__ import annotations

import os
import re

import pytest

# Ensure HF stays offline — these tests should never hit the network.
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from environment.benchmarks import ADAPTERS, document_fingerprint, get_adapter

# Limit pulled per test — keeps the whole file fast (target < 60 s).
SAMPLE_LIMIT = 3

# Benchmarks where NO gold relevance is provided. Trigram-overlap check
# is skipped for these (relevant_paragraphs is always empty by design).
BENCHMARKS_WITHOUT_GOLD = {"narrativeqa"}

ALL_NAMES = sorted(ADAPTERS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trigrams(text: str) -> set[str]:
    """Lowercase content trigrams."""
    text = re.sub(r"\s+", " ", text.lower())
    return {text[i:i + 3] for i in range(len(text) - 2)} if len(text) >= 3 else set()


def _check_schema(doc: dict, name: str) -> None:
    """Verify the Document contract — duplicate of base.validate_document
    but lives here as the externally-asserted spec.
    """
    assert isinstance(doc, dict), f"[{name}] doc is {type(doc)}, want dict"
    # Title
    assert "title" in doc and isinstance(doc["title"], str) and doc["title"].strip(), \
        f"[{name}] doc missing/empty title: {doc.get('title')!r}"
    # Paragraphs
    assert "paragraphs" in doc and isinstance(doc["paragraphs"], list), \
        f"[{name}] doc missing/invalid paragraphs"
    paragraphs = doc["paragraphs"]
    assert len(paragraphs) >= 1, f"[{name}] doc has 0 paragraphs"
    for i, p in enumerate(paragraphs):
        assert isinstance(p, str), \
            f"[{name}] paragraphs[{i}] is {type(p)}, want str"
    # QA pairs
    assert "qa_pairs" in doc and isinstance(doc["qa_pairs"], list), \
        f"[{name}] doc missing/invalid qa_pairs"
    qa_pairs = doc["qa_pairs"]
    assert len(qa_pairs) >= 1, f"[{name}] doc has 0 qa_pairs"
    for j, qa in enumerate(qa_pairs):
        assert isinstance(qa, dict), f"[{name}] qa_pairs[{j}] is {type(qa)}"
        # Question
        assert "question" in qa and isinstance(qa["question"], str), \
            f"[{name}] qa_pairs[{j}] missing question"
        assert qa["question"].strip(), f"[{name}] qa_pairs[{j}] question empty"
        # Answer (str OR list[str])
        assert "answer" in qa, f"[{name}] qa_pairs[{j}] missing answer"
        ans = qa["answer"]
        if isinstance(ans, list):
            assert any(str(a).strip() for a in ans), \
                f"[{name}] qa_pairs[{j}] all-empty answer list"
        else:
            assert isinstance(ans, str), \
                f"[{name}] qa_pairs[{j}] answer type {type(ans)}, want str|list"
            assert ans.strip(), f"[{name}] qa_pairs[{j}] empty answer string"
        # Relevant paragraphs — type + index range
        rel = qa.get("relevant_paragraphs", [])
        assert isinstance(rel, list), \
            f"[{name}] qa_pairs[{j}] relevant_paragraphs not a list"
        for pidx in rel:
            assert isinstance(pidx, int), \
                f"[{name}] qa_pairs[{j}].relevant_paragraphs has non-int {pidx!r}"
            assert 0 <= pidx < len(paragraphs), \
                f"[{name}] qa_pairs[{j}].relevant_paragraphs[{pidx}] out of [0, {len(paragraphs)})"


def _check_trigram_sanity(doc: dict, name: str) -> None:
    """For every (qa, pidx) with non-empty relevant_paragraphs, assert that
    paragraphs[pidx] shares ≥ 3 content trigrams with the question or answer.

    Skipped for benchmarks without gold relevance (NarrativeQA).
    """
    if name in BENCHMARKS_WITHOUT_GOLD:
        return
    paragraphs = doc["paragraphs"]
    for j, qa in enumerate(doc["qa_pairs"]):
        rel = qa.get("relevant_paragraphs", [])
        if not rel:
            # Empty is allowed (some benchmarks have items where no evidence
            # match succeeded — e.g. QASPER substring miss). Don't penalize.
            continue
        # Build expected-overlap set: trigrams of question + answer text.
        q_tri = _trigrams(qa["question"])
        if isinstance(qa["answer"], list):
            ans_text = " ".join(str(a) for a in qa["answer"])
        else:
            ans_text = qa["answer"]
        a_tri = _trigrams(ans_text)
        target = q_tri | a_tri
        # At least one indexed paragraph must share ≥ 3 trigrams.
        hit = False
        for pidx in rel:
            p_tri = _trigrams(paragraphs[pidx])
            if len(target & p_tri) >= 3:
                hit = True
                break
        if not hit:
            # Soft fail — only QASPER's evidence-substring path is realistically
            # vulnerable here, and even there a 3-trigram floor is reasonable.
            # Pretty-print the failing example for debug.
            preview_paras = [paragraphs[pidx][:80] for pidx in rel[:3]]
            pytest.fail(
                f"[{name}] qa_pairs[{j}] gold paragraphs share <3 trigrams "
                f"with question/answer.\n  Q: {qa['question'][:80]!r}\n  "
                f"A: {ans_text[:80]!r}\n  relevant paras: {preview_paras!r}"
            )


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_NAMES)
def test_iter_documents_yields_correct_count(name: str) -> None:
    """iter_documents(limit=N) yields exactly N docs (or fewer if dataset is smaller).
    """
    adapter = get_adapter(name)
    docs = list(adapter.iter_documents(limit=SAMPLE_LIMIT))
    # Each adapter should be able to produce at least SAMPLE_LIMIT docs from
    # the canonical split; if not, the prefetched data is incomplete.
    assert 1 <= len(docs) <= SAMPLE_LIMIT, \
        f"[{name}] yielded {len(docs)} docs (expected 1..{SAMPLE_LIMIT})"


@pytest.mark.parametrize("name", ALL_NAMES)
def test_document_schema(name: str) -> None:
    """Every yielded doc satisfies the Document contract."""
    adapter = get_adapter(name)
    for doc in adapter.iter_documents(limit=SAMPLE_LIMIT):
        _check_schema(doc, name)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_relevant_paragraphs_trigram_sanity(name: str) -> None:
    """For benchmarks with gold relevance, indexed paragraphs share content
    trigrams with the question/answer. Catches index-drift bugs.
    """
    adapter = get_adapter(name)
    for doc in adapter.iter_documents(limit=SAMPLE_LIMIT):
        _check_trigram_sanity(doc, name)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_deterministic_without_shuffle(name: str) -> None:
    """Two consecutive iter_documents(seed=42) calls yield byte-identical docs."""
    a = get_adapter(name)
    fp1 = [document_fingerprint(d) for d in a.iter_documents(limit=SAMPLE_LIMIT, seed=42)]
    b = get_adapter(name)
    fp2 = [document_fingerprint(d) for d in b.iter_documents(limit=SAMPLE_LIMIT, seed=42)]
    assert fp1 == fp2, \
        f"[{name}] non-deterministic iter_documents (seed=42, no shuffle)\n  run1: {fp1}\n  run2: {fp2}"


@pytest.mark.parametrize("name", ALL_NAMES)
def test_deterministic_with_shuffle(name: str) -> None:
    """Shuffle is seeded — same seed yields same order."""
    a = get_adapter(name)
    fp1 = [document_fingerprint(d)
           for d in a.iter_documents(limit=SAMPLE_LIMIT, seed=42, shuffle=True)]
    b = get_adapter(name)
    fp2 = [document_fingerprint(d)
           for d in b.iter_documents(limit=SAMPLE_LIMIT, seed=42, shuffle=True)]
    assert fp1 == fp2, \
        f"[{name}] non-deterministic iter_documents (seed=42, shuffle=True)"


@pytest.mark.parametrize("name", ALL_NAMES)
def test_dataset_fingerprint_stable(name: str) -> None:
    """dataset_fingerprint() is stable across calls and adapter instances."""
    a = get_adapter(name)
    fp1 = a.dataset_fingerprint()
    fp2 = a.dataset_fingerprint()
    b = get_adapter(name)
    fp3 = b.dataset_fingerprint()
    assert fp1 == fp2 == fp3, \
        f"[{name}] dataset_fingerprint unstable: {fp1} {fp2} {fp3}"
    assert fp1.startswith("sha256:"), \
        f"[{name}] dataset_fingerprint doesn't start with 'sha256:': {fp1!r}"


@pytest.mark.parametrize("name", ALL_NAMES)
def test_schema_version_set(name: str) -> None:
    """Every adapter exposes a SCHEMA_VERSION string."""
    a = get_adapter(name)
    assert isinstance(a.SCHEMA_VERSION, str), f"[{name}] SCHEMA_VERSION missing or non-str"
    assert a.SCHEMA_VERSION.startswith(name), \
        f"[{name}] SCHEMA_VERSION should start with adapter name: {a.SCHEMA_VERSION!r}"


def test_registry_completeness() -> None:
    """All 6 adapters are registered."""
    expected = {"hotpotqa", "qasper", "cuad", "narrativeqa", "financebench", "longmemeval"}
    assert set(ADAPTERS.keys()) == expected, \
        f"Registry mismatch: have {set(ADAPTERS)}, expected {expected}"


def test_get_adapter_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        get_adapter("does_not_exist")
