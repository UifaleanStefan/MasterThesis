"""
LLM-judge plumbing smoke tests.

These tests verify that the A4 LLM-judge codepath is wired correctly without
requiring an OPENAI_API_KEY: when the key is unset, ``llm_judge_score`` falls
back to the keyword-overlap heuristic and DocumentQA's score_fn injection
still produces sensible numbers.

The real LLM-judged grading is exercised in Stage 3 runs only (deferred from
the PoC plan).
"""

from __future__ import annotations

import os

import pytest

from environment.document_qa import DocumentQA
from evaluation.document_qa_llm_judge import (
    _heuristic_score,
    llm_judge_score,
    get_judge_stats,
    reset_judge_stats,
)


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """Force the no-API-key path for every test in this module."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_heuristic_score_perfect_match():
    score = _heuristic_score(
        "Petra is the descendant of Jorin, a farmer in Mosswater.",
        "Petra is Jorin's descendant, a farmer in Mosswater village.",
    )
    assert score >= 0.5  # Most key nouns overlap


def test_heuristic_score_no_match():
    score = _heuristic_score("nothing relevant here", "Sera is a spy for Ashfall.")
    assert score < 0.3


def test_llm_judge_falls_back_without_api_key():
    """Without OPENAI_API_KEY, llm_judge_score must use the heuristic and not crash."""
    score = llm_judge_score("the cat sat on the mat", "the cat sat on the mat")
    assert 0.0 <= score <= 1.0
    # No API call was made, so judge stats should remain zero
    stats = get_judge_stats()
    assert stats["total_judge_calls"] == 0
    assert stats["total_judge_cost_usd"] == 0.0


def test_documentqa_uses_custom_score_fn():
    """When score_fn is provided, DocumentQA delegates to it for scoring."""
    calls = []

    def stub_score_fn(pred: str, gt: str) -> float:
        calls.append((pred, gt))
        return 0.42

    env = DocumentQA(document_name="fantasy_lore", seed=0,
                     question_shuffle=False, score_fn=stub_score_fn)
    env.reset()
    # Skip through the reading phase
    for _ in range(len(env._paragraphs)):
        env.step("next")
    # Answer the first question
    obs, done, success = env.step("any predicted answer text")
    assert calls, "score_fn was not invoked"
    last_pred, last_gt = calls[0]
    assert last_pred == "any predicted answer text"
    assert "Petra" in last_gt or len(last_gt) > 0  # ground truth populated
    # The recorded score should reflect the stub return value
    assert env._scores[0] == pytest.approx(0.42)


def test_documentqa_score_fn_clipped_to_unit_interval():
    """Custom score_fn returning out-of-range values must be clipped to [0, 1]."""
    env = DocumentQA(document_name="fantasy_lore", seed=0,
                     question_shuffle=False, score_fn=lambda p, g: 5.0)
    env.reset()
    for _ in range(len(env._paragraphs)):
        env.step("next")
    env.step("any answer")
    assert 0.0 <= env._scores[0] <= 1.0


def test_documentqa_score_fn_exception_falls_back_to_keyword_overlap():
    """If score_fn raises, DocumentQA must fall back to the keyword overlap."""
    def broken(pred, gt):
        raise RuntimeError("scorer is broken")

    env = DocumentQA(document_name="fantasy_lore", seed=0,
                     question_shuffle=False, score_fn=broken)
    env.reset()
    for _ in range(len(env._paragraphs)):
        env.step("next")
    env.step("Petra is a farmer in Mosswater")
    # Heuristic score is in [0,1] and non-NaN
    assert 0.0 <= env._scores[0] <= 1.0


def test_judge_stats_reset():
    reset_judge_stats()
    stats = get_judge_stats()
    assert stats["total_judge_calls"] == 0
    assert stats["total_judge_cost_usd"] == 0.0
