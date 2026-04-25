"""
LLM-as-judge scorer for DocumentQA.

The default DocumentQA scorer (`environment.document_qa.DocumentQA._score_answer`)
matches keyword overlap of nouns. That under-credits semantically-correct
paraphrases — a problem when the agent's answer is rephrased differently from
the ground truth. This module provides an LLM-judged alternative that grades
answers on a 0–1 scale.

Public surface:
    score = llm_judge_score(predicted, ground_truth)

Behaviour:
  * Without ``OPENAI_API_KEY`` (or with the openai SDK unavailable), falls back
    to the same keyword-overlap heuristic the environment uses by default. The
    ``score_fn`` plumbing in DocumentQA can therefore be wired in production
    code without forcing real API calls in CI / smoke tests.
  * With an API key, calls ``gpt-4o-mini`` with a structured rubric and
    extracts the numeric grade from the response.

Usage:
    from environment.document_qa import DocumentQA
    from evaluation.document_qa_llm_judge import llm_judge_score
    env = DocumentQA(document_name="fantasy_lore", score_fn=llm_judge_score)

This module is implemented as part of A4 in the PoC plan; it is NOT exercised
against a real API in PoC runs. Activation requires ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import os
import re
from typing import Optional

_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

# Module-level cost tracker — callers can read this to roll up grading cost
# alongside agent inference cost.
_total_judge_cost_usd: float = 0.0
_total_judge_calls: int = 0


def get_judge_stats() -> dict:
    """Return cumulative LLM-judge cost since process start."""
    return {
        "total_judge_cost_usd": _total_judge_cost_usd,
        "total_judge_calls": _total_judge_calls,
    }


def reset_judge_stats() -> None:
    global _total_judge_cost_usd, _total_judge_calls
    _total_judge_cost_usd = 0.0
    _total_judge_calls = 0


_RUBRIC = (
    "You are grading a question-answering response against ground truth. "
    "Output a single number between 0.0 and 1.0 representing how completely "
    "the predicted answer captures the key facts in the ground truth. "
    "1.0 = all key facts present, paraphrasing OK. "
    "0.5 = some key facts present, but missing or contradicting others. "
    "0.0 = no relevant facts captured. "
    "Respond with ONLY the number, no other text."
)


def _heuristic_score(predicted: str, ground_truth: str) -> float:
    """
    Keyword-overlap fallback (same logic as DocumentQA._score_answer:335-355).
    Used when no API key is available — keeps the codepath testable in CI.
    """
    gt = ground_truth.lower()
    pred = predicted.lower()
    words = re.findall(r"\b[a-z]{4,}\b", gt)
    key_words = [w for w in words
                 if w not in {"that", "this", "from", "with",
                              "have", "been", "they", "their", "were"}]
    if not key_words:
        return 1.0 if gt[:30] in pred else 0.0
    matches = sum(1 for w in key_words if w in pred)
    return matches / len(key_words)


def _try_init_client() -> Optional[object]:
    """Return an OpenAI client when the SDK + API key are present, else None."""
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
        return OpenAI()
    except Exception:
        return None


def llm_judge_score(
    predicted: str,
    ground_truth: str,
    model: str = "gpt-4o-mini",
    seed: int = 42,
    max_tokens: int = 10,
) -> float:
    """
    Grade ``predicted`` against ``ground_truth`` on a 0–1 scale.

    Returns a float in [0, 1]. When no OpenAI client is available, falls back
    to ``_heuristic_score`` so callers always get a number.
    """
    global _total_judge_cost_usd, _total_judge_calls

    client = _try_init_client()
    if client is None:
        return _heuristic_score(predicted, ground_truth)

    user_msg = (
        f"Ground truth: {ground_truth}\n\n"
        f"Predicted: {predicted}\n\n"
        f"Score (0.0-1.0):"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _RUBRIC},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            seed=seed,
        )
    except Exception as exc:
        print(f"[llm_judge] API error: {exc} — heuristic fallback")
        return _heuristic_score(predicted, ground_truth)

    raw = response.choices[0].message.content.strip()
    # Track cost
    usage = response.usage
    pricing = _PRICING.get(model, {"input": 0.15, "output": 0.60})
    cost = (
        usage.prompt_tokens * pricing["input"]
        + usage.completion_tokens * pricing["output"]
    ) / 1_000_000
    _total_judge_cost_usd += cost
    _total_judge_calls += 1

    # Extract a float in [0, 1]; on parse failure, fall back to heuristic.
    match = re.search(r"\d+\.?\d*", raw)
    if not match:
        return _heuristic_score(predicted, ground_truth)
    try:
        score = float(match.group())
    except ValueError:
        return _heuristic_score(predicted, ground_truth)
    return max(0.0, min(1.0, score))
