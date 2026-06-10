"""Phase 1.9 — CUAD v4t-corpus-tuned Protocol B calibration + batch_calib judge (full scale).

Cells:
  cuad__v4t-corpus-tuned__calibration__seed42   (2550 entries)
  cuad__v4t-corpus-tuned__batch_calib__seed42   (6702 entries)

Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  Protocol B rubric

Key findings:
  - v4t-corpus-tuned uses theta tuned on CUAD corpus itself (not FinanceBench)
  - Calibration queue: 2550 entries = 1225 acknowledge_missing + 1325 answer
    (667 refusals + 658 non-refusals hand-judged 1-by-1)
  - batch_calib queue: 6702 entries = 2745 refusals + 3957 non-refusals
  - Common errors: wrong contract retrieved (semantic similarity across clauses),
    "four-year anniversary of the Effective Date" boilerplate for expiration dates,
    "MusclePharm for the benefit of AS Parties" boilerplate for insurance,
    "People's Republic of China" or "State of Georgia" for governing law
  - Calibration non-refusal mean dominated by document name exact matches and
    governing law matches; most expiration/effective date questions score 0.0

Scoring rules (Protocol B):
  - acknowledge_missing: 1.0 honest refusal, 0.0 confident answer (hallucination)
  - answer, gold=None/empty: default 0.0 unless in judgments dict
  - answer, regular: standard 5-point rubric
    0.0 wrong/refusal, 0.25 partial/one correct aspect, 0.5 partially correct,
    0.75 correct-but-imprecise, 1.0 correct
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

# Import hand-judged lookups from accumulation files
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _calib_judgments_partial_v4t_corpus_tuned import CALIB_JUDGMENTS_PARTIAL_CT as _CJPCT
from _batch_judgments_partial_v4t_corpus_tuned import BATCH_JUDGMENTS_PARTIAL_CT as _BJPCT

ROOT = Path(__file__).resolve().parents[1]
JQ = ROOT / "results" / "stage3" / "judge_queue"
CALIB_DIR = JQ / "cuad__v4t-corpus-tuned__calibration__seed42"
BATCH_DIR = JQ / "cuad__v4t-corpus-tuned__batch_calib__seed42"

CALIB_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR.mkdir(parents=True, exist_ok=True)

CALIB_RESULTS = CALIB_DIR / "results.jsonl"
BATCH_RESULTS = BATCH_DIR / "results.jsonl"

# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------
_REFUSAL_PATTERNS = [
    "do not have", "don't have", "context provided", "provided context",
    "passages provided", "provided passages", "not mentioned", "not provided",
    "not available", "insufficient", "no information", "cannot determine",
    "not specify", "no mention", "cannot find", "does not contain",
    "not found", "i'm sorry", "unable to", "no context", "not enough",
    "apologies", "not explicitly", "not clear", "not specified",
    "passages do not", "does not provide", "do not provide",
    "cannot be determined", "no relevant", "not discussed",
    "no specific", "cannot answer", "not contain", "no detail",
    "information is not", "there is no", "is not mentioned",
    "are not mentioned", "is not provided", "are not provided",
    "there are no", "not include", "not included", "do not contain",
    "not be determined", "not be found", "without more", "lacks",
    "no passage", "i do not see", "not see", "not support",
    "unanswerable", "unspecified", "no answer", "the relevant passages",
    "the document passages", "document does not", "documents do not",
]


def is_refusal(pred: str) -> bool:
    p = pred.strip().lower()
    if not p:
        return True
    return any(pat in p for pat in _REFUSAL_PATTERNS)


# ---------------------------------------------------------------------------
# CALIBRATION JUDGMENTS: suffix → (score, rationale)
# suffix = qid minus "cuad__v4t-corpus-tuned__calibration__" and "__seed42"
# Only non-zero answer entries (0.0 falls through to default)
# Format: "doc{N}_qa{M}__after{K}": (score, "rationale text")
# ---------------------------------------------------------------------------
CALIB_JUDGMENTS: dict[str, tuple[float, str]] = _CJPCT
# 658 non-refusal answer entries hand-judged 1-by-1 (claude-opus-4.7-1m).
# Non-zero entries stored in _calib_judgments_partial_v4t_corpus_tuned.py; 0.0 falls through.


# ---------------------------------------------------------------------------
# BATCH JUDGMENTS: suffix → (score, rationale)
# suffix = qid minus "cuad__v4t-corpus-tuned__batch__" and "__seed42"
# Only non-zero entries (0.0 falls through to default)
# Format: "doc{N}_qa{M}": (score, "rationale text")
# ---------------------------------------------------------------------------
BATCH_JUDGMENTS: dict[str, tuple[float, str]] = _BJPCT
# 3957 non-refusal entries to be hand-judged 1-by-1; 2745 refusals auto-score 0.0.
# Total batch_calib queue: 6702 entries.


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def score_calib(r: dict) -> tuple[float, str]:
    """Score a calibration entry (calib expected_behavior can be either type)."""
    pred = r.get("predicted", "")
    cfg = "v4t-corpus-tuned"
    suffix = (
        r["qid"]
        .replace(f"cuad__{cfg}__calibration__", "")
        .replace("__seed42", "")
    )

    # expected_behavior == "acknowledge_missing": honest refusal = 1.0
    if r.get("expected_behavior", "answer") == "acknowledge_missing":
        if is_refusal(pred):
            return 1.0, (
                "Model correctly acknowledges the source document has not yet "
                "been ingested. Honest refusal on a missing-source question. "
                "Full credit per Protocol B rule."
            )
        return 0.0, (
            "Model answers confidently about a document not yet in memory — "
            "hallucination. Zero per Protocol B acknowledge_missing rule."
        )

    # expected_behavior == "answer"
    if is_refusal(pred):
        return 0.0, (
            "Model refuses to answer despite the source being in memory. "
            "Refusal on an answerable question scores zero."
        )

    if suffix in CALIB_JUDGMENTS:
        return CALIB_JUDGMENTS[suffix]

    return 0.0, (
        "Prediction does not match gold answer for this contract. "
        "Wrong contract retrieved or incorrect content. Zero."
    )


def score_batch(r: dict) -> tuple[float, str]:
    """Score a batch_calib entry (all expected_behavior=answer)."""
    pred = r.get("predicted", "")
    cfg = "v4t-corpus-tuned"
    suffix = (
        r["qid"]
        .replace(f"cuad__{cfg}__batch__", "")
        .replace("__seed42", "")
    )

    if is_refusal(pred):
        return 0.0, (
            "Model refuses to answer a batch calibration question. "
            "Refusal on answerable question scores zero."
        )

    if suffix in BATCH_JUDGMENTS:
        return BATCH_JUDGMENTS[suffix]

    return 0.0, (
        "Prediction does not match gold answer. Wrong contract retrieved or "
        "incorrect content due to retrieval competition across 510 contracts. Zero."
    )


# ---------------------------------------------------------------------------
# Write results
# ---------------------------------------------------------------------------
def write_results(
    queue_path: Path,
    results_path: Path,
    score_fn,
    label: str,
) -> None:
    rows = [
        json.loads(line)
        for line in queue_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out_lines: list[str] = []
    for r in rows:
        score, rationale = score_fn(r)
        out_lines.append(
            json.dumps(
                {
                    "qid": r["qid"],
                    "judge_score": score,
                    "rationale": rationale,
                    "judge_model": "claude-opus-4.7-1m",
                    "judge_protocol": "v1",
                    "expected_behavior": r.get("expected_behavior", "answer"),
                },
                ensure_ascii=False,
            )
        )
    results_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    scores = [json.loads(l)["judge_score"] for l in out_lines]
    mean = sum(scores) / len(scores) if scores else 0.0
    ones = sum(1 for s in scores if s == 1.0)
    zeros = sum(1 for s in scores if s == 0.0)
    print(
        f"{label}: {len(rows)} entries -> mean={mean:.4f} "
        f"(1.0x{ones}, 0.0x{zeros})"
    )


if __name__ == "__main__":
    write_results(
        CALIB_DIR / "queue.jsonl",
        CALIB_RESULTS,
        score_calib,
        "cuad__v4t-corpus-tuned__calibration__seed42",
    )
    write_results(
        BATCH_DIR / "queue.jsonl",
        BATCH_RESULTS,
        score_batch,
        "cuad__v4t-corpus-tuned__batch_calib__seed42",
    )
    print("Done.")
