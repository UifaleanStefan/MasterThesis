"""Tests for Phase 1.9 Protocol B (calibration mode) in `scripts/run_corpus_qa.py`.

Covers:
  * `_build_all_qa_inventory` produces complete, doc-ordered inventory with
    correct `doc_start_step` global indices.
  * Calibration sampling at each doc-end is deterministic given seed.
  * `expected_behavior` is tagged correctly based on whether source_doc_idx
    has been ingested at the moment the question is asked.
  * Cross-config sampling reproducibility: same seed → same qids picked at
    each doc-end (allows paired analysis across configs).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_corpus_qa import _build_all_qa_inventory  # noqa: E402


# ----------------------------------------------------------------------
# Inventory shape
# ----------------------------------------------------------------------


def _make_doc(doc_idx: int, n_para: int, qa_pairs):
    return {
        "title": f"doc_{doc_idx}",
        "paragraphs": [f"para {doc_idx}.{i}" for i in range(n_para)],
        "qa_pairs": qa_pairs,
    }


class TestBuildInventory:
    def test_empty_corpus(self):
        assert _build_all_qa_inventory([]) == []

    def test_single_doc_single_q(self):
        docs = [_make_doc(0, 3, [{"question": "q0", "answer": "a0", "relevant_paragraphs": [1]}])]
        inv = _build_all_qa_inventory(docs)
        assert len(inv) == 1
        assert inv[0]["doc_idx"] == 0
        assert inv[0]["qa_idx_in_doc"] == 0
        assert inv[0]["question"] == "q0"
        assert inv[0]["gold_answer"] == "a0"
        assert inv[0]["doc_start_step"] == 0
        assert inv[0]["n_paragraphs"] == 3
        # relevant_global_steps = doc_start_step + local_idx
        assert inv[0]["relevant_global_steps"] == [1]

    def test_multiple_docs_global_step_offsets(self):
        docs = [
            _make_doc(0, 2, [{"question": "q0", "answer": "a0", "relevant_paragraphs": [0, 1]}]),
            _make_doc(1, 3, [{"question": "q1", "answer": "a1", "relevant_paragraphs": [0]}]),
            _make_doc(2, 1, [{"question": "q2", "answer": "a2", "relevant_paragraphs": [0]}]),
        ]
        inv = _build_all_qa_inventory(docs)
        assert len(inv) == 3
        # doc 0 occupies global steps 0..1
        assert inv[0]["doc_start_step"] == 0
        assert inv[0]["relevant_global_steps"] == [0, 1]
        # doc 1 occupies global steps 2..4
        assert inv[1]["doc_start_step"] == 2
        assert inv[1]["relevant_global_steps"] == [2]
        # doc 2 occupies global step 5
        assert inv[2]["doc_start_step"] == 5
        assert inv[2]["relevant_global_steps"] == [5]

    def test_multiple_qa_pairs_per_doc(self):
        docs = [_make_doc(0, 5, [
            {"question": "q0", "answer": "a0", "relevant_paragraphs": [0]},
            {"question": "q1", "answer": "a1", "relevant_paragraphs": [4]},
        ])]
        inv = _build_all_qa_inventory(docs)
        assert len(inv) == 2
        assert [e["qa_idx_in_doc"] for e in inv] == [0, 1]
        assert inv[0]["relevant_global_steps"] == [0]
        assert inv[1]["relevant_global_steps"] == [4]

    def test_oob_relevant_paragraphs_filtered(self):
        """A relevant index beyond doc length should be filtered, not raised."""
        docs = [_make_doc(0, 2, [
            {"question": "q0", "answer": "a0", "relevant_paragraphs": [0, 1, 99]}
        ])]
        inv = _build_all_qa_inventory(docs)
        assert inv[0]["relevant_global_steps"] == [0, 1]  # 99 dropped


# ----------------------------------------------------------------------
# Calibration sampling reproducibility
# ----------------------------------------------------------------------


class TestCalibrationSampling:
    """Verifies the sampling logic embedded inline in `run_corpus_qa`.

    We replicate the per-doc-end sampler here as a small helper so the test
    is decoupled from the full QA loop (no LLM calls).
    """

    @staticmethod
    def _sample_for_doc(seed: int, doc_idx: int, n_total: int, k: int) -> list[int]:
        """Matches the inline call in run_corpus_qa:
            rng = random.Random(seed * 31 + doc_idx)
            rng.sample(range(n_total), min(k, n_total))
        """
        rng = random.Random(seed * 31 + doc_idx)
        return rng.sample(range(n_total), min(k, n_total))

    def test_sampling_deterministic_per_seed_doc(self):
        a = self._sample_for_doc(seed=42, doc_idx=5, n_total=150, k=10)
        b = self._sample_for_doc(seed=42, doc_idx=5, n_total=150, k=10)
        assert a == b

    def test_sampling_differs_across_doc_ends(self):
        a = self._sample_for_doc(seed=42, doc_idx=5, n_total=150, k=10)
        b = self._sample_for_doc(seed=42, doc_idx=6, n_total=150, k=10)
        assert a != b

    def test_sampling_differs_across_seeds(self):
        a = self._sample_for_doc(seed=42, doc_idx=5, n_total=150, k=10)
        b = self._sample_for_doc(seed=7, doc_idx=5, n_total=150, k=10)
        assert a != b

    def test_sampling_unique_within_a_doc_end(self):
        """rng.sample is without replacement within one call."""
        sampled = self._sample_for_doc(seed=42, doc_idx=0, n_total=150, k=10)
        assert len(sampled) == 10
        assert len(set(sampled)) == 10  # no duplicates

    def test_sampling_caps_at_pool_size(self):
        """If we ask for more samples than the pool, get only pool size."""
        sampled = self._sample_for_doc(seed=42, doc_idx=0, n_total=5, k=10)
        assert len(sampled) == 5
        assert set(sampled) == {0, 1, 2, 3, 4}

    def test_sampling_reproducible_across_configs(self):
        """Same seed across two 'config runs' must pick same qids at each
        doc-end (this is the property that makes per-question paired
        analysis across configs meaningful)."""
        for doc_idx in [0, 1, 50, 149]:
            a = self._sample_for_doc(seed=42, doc_idx=doc_idx, n_total=150, k=10)
            b = self._sample_for_doc(seed=42, doc_idx=doc_idx, n_total=150, k=10)
            assert a == b, f"sampling not reproducible at doc_idx={doc_idx}"


# ----------------------------------------------------------------------
# expected_behavior tagging
# ----------------------------------------------------------------------


class TestExpectedBehaviorTagging:
    """The tag is:
        'answer' if source_doc_idx <= asked_after_doc_idx else 'acknowledge_missing'
    """

    @staticmethod
    def _tag(source_doc_idx: int, asked_after_doc_idx: int) -> str:
        return "answer" if source_doc_idx <= asked_after_doc_idx else "acknowledge_missing"

    def test_source_before_ask_is_answer(self):
        assert self._tag(source_doc_idx=0, asked_after_doc_idx=5) == "answer"

    def test_source_equals_ask_is_answer(self):
        """When doc N is just ingested and we ask doc N's own q, that's
        the standard online case — should answer."""
        assert self._tag(source_doc_idx=5, asked_after_doc_idx=5) == "answer"

    def test_source_after_ask_is_acknowledge_missing(self):
        """Question is about a doc not yet ingested — model should refuse."""
        assert self._tag(source_doc_idx=10, asked_after_doc_idx=5) == "acknowledge_missing"

    def test_boundary_first_doc_no_questions_yet_ingested_except_own(self):
        """At doc_idx=0 (only doc 0 ingested), only doc-0 questions are answerable."""
        assert self._tag(0, 0) == "answer"
        assert self._tag(1, 0) == "acknowledge_missing"
        assert self._tag(149, 0) == "acknowledge_missing"

    def test_boundary_last_doc_all_questions_answerable(self):
        """At doc_idx=149 (full corpus ingested), every question is answerable."""
        for src in range(150):
            assert self._tag(src, 149) == "answer"


# ----------------------------------------------------------------------
# End-to-end sampling-distribution sanity
# ----------------------------------------------------------------------


class TestSamplingDistributionSanity:
    """For a corpus of 150 docs sampled 10/doc-end with seed=42, the total
    expected mix should be ~50/50 answer/acknowledge_missing on average
    (since at doc N, fraction ingested = (N+1)/150).
    """

    def test_total_mix_roughly_balanced(self):
        n_total = 150
        per_doc = 10
        seed = 42
        n_answer = 0
        n_ack = 0
        for doc_idx in range(n_total):
            rng = random.Random(seed * 31 + doc_idx)
            sampled = rng.sample(range(n_total), per_doc)
            for src in sampled:
                if src <= doc_idx:
                    n_answer += 1
                else:
                    n_ack += 1
        total = n_answer + n_ack
        assert total == n_total * per_doc == 1500
        # Expected: each fraction ≈ 50% by symmetry of uniform sampling
        # over the triangle. Allow ±5% tolerance for finite-sample noise.
        ratio_answer = n_answer / total
        assert 0.45 < ratio_answer < 0.55, (
            f"answer fraction {ratio_answer:.3f} out of expected [0.45, 0.55]")
