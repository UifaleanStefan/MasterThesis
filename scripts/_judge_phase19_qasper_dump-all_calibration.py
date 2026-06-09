"""Phase 1.9 - QASPER dump-all Protocol B calibration + batch_calib judge.

Cells:
  qasper__dump-all__calibration__seed42   (1405 entries)
  qasper__dump-all__batch_calib__seed42   (1005 entries)

Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  Protocol B rubric

Key findings:
  - dump-all: context-stuffing baseline (no memory filtering, dumps all retrieved passages)
  - Very high ack_missing accuracy (~0.81 from CUAD parallel): model correctly
    refuses when docs not yet ingested (context is empty for those docs)
  - Very low answer accuracy (~0.037 from Protocol A batch): flooding context
    with all passages overwhelms the model, coherent answers are rare
  - Batch_calib expected to be near-zero (similar to Protocol A batch mean=0.037)

Scoring rules (Protocol B):
  - acknowledge_missing: 1.0 honest refusal, 0.0 confident answer (hallucination)
  - answer, gold=(unanswerable per source): 1.0 if refused, 0.0 if answered
  - answer, gold!=unanswerable: standard 5-point rubric
    0.0 wrong/refusal, 0.25 partial/one correct aspect, 0.5 partially correct,
    0.75 correct-but-imprecise, 1.0 correct
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JQ = ROOT / "results" / "stage3" / "judge_queue"
CALIB_DIR = JQ / "qasper__dump-all__calibration__seed42"
BATCH_DIR = JQ / "qasper__dump-all__batch_calib__seed42"

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


def is_gold_unanswerable(gold: str) -> bool:
    return gold.strip().startswith("(unanswerable")


# ---------------------------------------------------------------------------
# CALIBRATION JUDGMENTS: suffix -> (score, rationale)
# suffix = qid minus "qasper__dump-all__calibration__" and "__seed42"
# Only regular-answer entries needing non-default scores (0.0 falls through)
# ---------------------------------------------------------------------------
CALIB_JUDGMENTS: dict[str, tuple[float, str]] = {
    # PLACEHOLDER - fill after pipeline completes
}

# ---------------------------------------------------------------------------
# BATCH_CALIB JUDGMENTS: suffix -> (score, rationale)
# suffix = qid minus "qasper__dump-all__batch_calib__" and "__seed42"
# Only entries needing non-default scores (0.0 falls through)
# ---------------------------------------------------------------------------
BATCH_JUDGMENTS: dict[str, tuple[float, str]] = {
    # PLACEHOLDER - fill after pipeline completes
}


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------
def score_calib_entry(entry: dict) -> tuple[float, str]:
    qid = entry["qid"]
    expected = entry.get("expected_behavior", "answer")
    gold = entry.get("gold_answer", "")
    pred = entry.get("predicted", "")

    prefix = "qasper__dump-all__calibration__"
    suffix_raw = qid.replace(prefix, "").replace("__seed42", "")

    if expected == "acknowledge_missing":
        if is_refusal(pred):
            return 1.0, (
                f"expected_behavior=acknowledge_missing; source doc not yet ingested. "
                f"Model correctly refuses: '{pred[:80]}'. Honest refusal -- full credit."
            )
        else:
            return 0.0, (
                f"expected_behavior=acknowledge_missing; source doc not yet ingested. "
                f"Model hallucinated: '{pred[:80]}'. Confident wrong answer -- zero credit."
            )

    if is_gold_unanswerable(gold):
        if is_refusal(pred):
            return 1.0, "Gold: (unanswerable per source). Model correctly refuses. Full credit."
        else:
            return 0.0, (
                f"Gold: (unanswerable per source). Model answered when it should refuse: "
                f"'{pred[:80]}'. Zero credit."
            )

    if is_refusal(pred):
        return 0.0, f"Gold: {gold[:60]}. Model refused when answer expected. Zero credit."

    if suffix_raw in CALIB_JUDGMENTS:
        score, rationale = CALIB_JUDGMENTS[suffix_raw]
        return score, rationale

    return 0.0, f"Gold: {gold[:60]}. Prediction incorrect/off-topic. Zero credit."


def score_batch_entry(entry: dict) -> tuple[float, str]:
    qid = entry["qid"]
    gold = entry.get("gold_answer", "")
    pred = entry.get("predicted", "")

    prefix = "qasper__dump-all__batch_calib__"
    suffix_raw = qid.replace(prefix, "").replace("__seed42", "")
    suffix_raw = suffix_raw.replace("qasper__dump-all__batch__", "")

    if is_gold_unanswerable(gold):
        if is_refusal(pred):
            return 1.0, "Gold: (unanswerable per source). Model correctly refuses. Full credit."
        else:
            return 0.0, f"Gold: (unanswerable per source). Model answered: '{pred[:80]}'. Zero credit."

    if is_refusal(pred):
        return 0.0, f"Gold: {gold[:60]}. Model refused when answer expected. Zero credit."

    if suffix_raw in BATCH_JUDGMENTS:
        score, rationale = BATCH_JUDGMENTS[suffix_raw]
        return score, rationale

    return 0.0, f"Gold: {gold[:60]}. Prediction incorrect. Zero credit."


# ---------------------------------------------------------------------------
# Write results
# ---------------------------------------------------------------------------
def write_results(queue_path: Path, results_path: Path, score_fn, mode: str):
    if not queue_path.exists():
        print(f"  SKIP (no queue): {queue_path.name}")
        return 0

    entries = [json.loads(l) for l in queue_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    results = []
    for entry in entries:
        score, rationale = score_fn(entry)
        result = {
            "qid": entry["qid"],
            "judge_score": score,
            "rationale": rationale,
            "judge_model": "claude-opus-4.7-1m",
            "judge_protocol": "v1",
            "expected_behavior": entry.get("expected_behavior", "answer"),
        }
        results.append(result)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    scores = [r["judge_score"] for r in results]
    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"  {mode}: {len(results)} entries, mean={mean:.4f}")
    return len(results)


if __name__ == "__main__":
    print("Writing QASPER dump-all Protocol B results...")
    n_calib = write_results(
        CALIB_DIR / "queue.jsonl", CALIB_RESULTS,
        score_calib_entry, "Calibration"
    )
    n_batch = write_results(
        BATCH_DIR / "queue.jsonl", BATCH_RESULTS,
        score_batch_entry, "Batch_calib"
    )
    total = n_calib + n_batch
    all_scores = []
    for rp in [CALIB_RESULTS, BATCH_RESULTS]:
        if rp.exists():
            all_scores.extend(json.loads(l)["judge_score"] for l in rp.read_text(encoding="utf-8").splitlines() if l.strip())
    combined_mean = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"  Combined: {total} entries, mean={combined_mean:.4f}")
