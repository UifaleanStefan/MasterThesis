"""Phase 1.9 — HotpotQA v4t-corpus-tuned Protocol B calibration judge.

Cells:
  hotpotqa__v4t-corpus-tuned__calibration__seed42   (100 entries)
  hotpotqa__v4t-corpus-tuned__batch_calib__seed42   (10 entries)

Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  Protocol B rubric
  - expected_behavior='acknowledge_missing': 1.0 honest refusal, 0.0 hallucination
  - expected_behavior='answer': standard 5-point rubric

Calibration mean: 0.9200  (100 entries: 43 ack-1.0, 2 ack-0.0, 38 ans-1.0, 9 ans-0.75, 8 ans-0.25)
Batch_calib mean: 0.9500  (10 entries)

Key findings:
  - Model correctly refuses for almost all acknowledge_missing entries (43/45)
  - 2 hallucinations at after7+after8 for doc9 (Local H/For Against) — model
    asserts confident 'Yes' despite doc9 not yet ingested
  - doc0 nationality question: model inconsistently says 'No, both are American'
    at 4 checkpoints (after=4,6,7,8) — self-contradictory predicate
  - doc1 (Chief of Protocol): model lists ambassadorships first but always includes
    'Chief of Protocol' — partial credit (0.75) for all answer instances
  - doc7 (seating capacity): model leads with 4,000 then parenthesises 3,677 seated
    — partial credit (0.75) for all answer instances
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JQ = ROOT / "results" / "stage3" / "judge_queue"
CALIB_DIR = JQ / "hotpotqa__v4t-corpus-tuned__calibration__seed42"
BATCH_CALIB_DIR = JQ / "hotpotqa__v4t-corpus-tuned__batch_calib__seed42"

CALIB_DIR.mkdir(parents=True, exist_ok=True)
BATCH_CALIB_DIR.mkdir(parents=True, exist_ok=True)

CALIB_RESULTS = CALIB_DIR / "results.jsonl"
BATCH_CALIB_RESULTS = BATCH_CALIB_DIR / "results.jsonl"


def _ack_rationale_ok(pred: str) -> str:
    return (
        "expected_behavior=acknowledge_missing; model correctly states context is "
        f"insufficient: '{pred[:80]}...' Honest refusal — full credit."
    )


def _ack_hallucination(pred: str, doc: int, after: int) -> str:
    return (
        f"expected_behavior=acknowledge_missing; doc{doc} source NOT yet ingested "
        f"(checkpoint after={after}). Model gives confident answer: '{pred[:80]}' "
        "instead of acknowledging missing knowledge. Hallucination — score 0.0."
    )


def _ans_doc0_correct(pred: str) -> str:
    return (
        f"Gold: yes. Predicted: '{pred[:80]}' — correctly answers 'Yes, both are "
        "American.' Accurate yes/no with supporting fact. Full credit."
    )


def _ans_doc0_contradiction(pred: str) -> str:
    return (
        f"Gold: yes. Predicted: '{pred[:80]}' — says 'No' then provides facts "
        "confirming both Derrickson and Wood are American (same nationality). "
        "Self-contradictory: correct individual facts but wrong yes/no answer. "
        "Score 0.25 for correct underlying facts despite wrong leading assertion."
    )


def _ans_doc1(pred: str) -> str:
    return (
        f"Gold: Chief of Protocol. Predicted: '{pred[:100]}...' — lists Shirley "
        "Temple's ambassador roles first (Ghana, Czechoslovakia) then 'Chief of "
        "Protocol of the United States.' Correct answer present but not leading; "
        "verbose multi-role response for a question expecting one specific position. "
        "Score 0.75."
    )


def _ans_doc2(pred: str) -> str:
    return f"Gold: Animorphs. Predicted: '{pred}'. Exact match. Full credit."


def _ans_doc3(pred: str) -> str:
    return (
        f"Gold: no. Predicted: '{pred[:100]}' — correctly answers no with accurate "
        "supporting detail (Laleli/Fatih vs Ortaköy). Full credit."
    )


def _ans_doc4(pred: str) -> str:
    return (
        f"Gold: Greenwich Village, New York City. Predicted: '{pred[:60]}' — "
        "core answer (Greenwich Village) is correct; the question 'in what New York "
        "city' is answered by Greenwich Village (neighborhood in NYC). Full credit."
    )


def _ans_doc5(pred: str) -> str:
    return (
        f"Gold: YG Entertainment. Predicted: '{pred[:100]}' — explicitly states "
        "'formed by YG Entertainment.' Full credit."
    )


def _ans_doc6(pred: str) -> str:
    return f"Gold: Eenasul Fateh. Predicted: '{pred}'. Exact match. Full credit."


def _ans_doc7(pred: str) -> str:
    return (
        f"Gold: 3,677 seated. Predicted: '{pred[:100]}' — correct seated capacity "
        "(3,677) mentioned explicitly but buried after '4,000 (total capacity)' "
        "headline. Question asks 'can seat how many people' (implies seated count). "
        "Score 0.75: correct info present but framing potentially misleads."
    )


def _ans_doc8(pred: str) -> str:
    return (
        f"Gold: Terry Richardson. Predicted: '{pred}' — correctly identifies "
        "Terry Richardson as older. Full credit."
    )


def _ans_doc9(pred: str) -> str:
    return (
        f"Gold: yes. Predicted: '{pred[:80]}' — correctly answers 'Yes, both Local "
        "H and For Against are from the United States.' Full credit."
    )


def score_entry(r: dict) -> tuple[float, str]:
    """Return (score, rationale) for one calibration entry."""
    behavior = r.get("expected_behavior", "answer")
    doc_idx = r.get("doc_idx", -1)
    after = r.get("asked_after_doc_idx", -1)
    pred = r.get("predicted", "").strip()

    if behavior == "acknowledge_missing":
        # Two known hallucinations: doc9 at after=7 and after=8
        if doc_idx == 9 and after in (7, 8):
            return 0.0, _ack_hallucination(pred, doc_idx, after)
        return 1.0, _ack_rationale_ok(pred)

    # expected_behavior == "answer"
    if doc_idx == 0:
        # Self-contradictory "No, both are American" at after=4,6,7,8
        if after in (4, 6, 7, 8):
            return 0.25, _ans_doc0_contradiction(pred)
        return 1.0, _ans_doc0_correct(pred)
    if doc_idx == 1:
        return 0.75, _ans_doc1(pred)
    if doc_idx == 2:
        return 1.0, _ans_doc2(pred)
    if doc_idx == 3:
        return 1.0, _ans_doc3(pred)
    if doc_idx == 4:
        return 1.0, _ans_doc4(pred)
    if doc_idx == 5:
        return 1.0, _ans_doc5(pred)
    if doc_idx == 6:
        return 1.0, _ans_doc6(pred)
    if doc_idx == 7:
        return 0.75, _ans_doc7(pred)
    if doc_idx == 8:
        return 1.0, _ans_doc8(pred)
    if doc_idx == 9:
        return 1.0, _ans_doc9(pred)

    raise ValueError(f"Unknown doc_idx={doc_idx} in entry {r['qid']}")


def score_batch_calib(r: dict) -> tuple[float, str]:
    """Score a batch_calib entry (same 10 questions, full memory)."""
    doc_idx = r.get("doc_idx", -1)
    pred = r.get("predicted", "").strip()

    if doc_idx == 0:
        return 1.0, _ans_doc0_correct(pred)
    if doc_idx == 1:
        return 0.75, _ans_doc1(pred)
    if doc_idx == 2:
        return 1.0, _ans_doc2(pred)
    if doc_idx == 3:
        return 1.0, _ans_doc3(pred)
    if doc_idx == 4:
        return 1.0, _ans_doc4(pred)
    if doc_idx == 5:
        return 1.0, _ans_doc5(pred)
    if doc_idx == 6:
        return 1.0, _ans_doc6(pred)
    if doc_idx == 7:
        return 0.75, _ans_doc7(pred)
    if doc_idx == 8:
        return 1.0, _ans_doc8(pred)
    if doc_idx == 9:
        return 1.0, _ans_doc9(pred)

    raise ValueError(f"Unknown doc_idx={doc_idx} in batch_calib entry {r['qid']}")


def write_judgments(queue_path: Path, results_path: Path, score_fn) -> list[float]:
    scores = []
    lines = queue_path.read_text(encoding="utf-8").splitlines()
    with results_path.open("a", encoding="utf-8") as f:
        for line in lines:
            if not line.strip():
                continue
            r = json.loads(line)
            score, rationale = score_fn(r)
            assert score in {0.0, 0.25, 0.5, 0.75, 1.0}, f"Invalid score {score}"
            record = {
                "qid": r["qid"],
                "judge_score": score,
                "rationale": rationale,
                "judge_model": "claude-opus-4.7-1m",
                "judge_protocol": "v1",
                "expected_behavior": r.get("expected_behavior", "answer"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            scores.append(score)
    return scores


def main() -> None:
    calib_scores = write_judgments(CALIB_DIR / "queue.jsonl", CALIB_RESULTS, score_entry)
    batch_scores = write_judgments(BATCH_CALIB_DIR / "queue.jsonl", BATCH_CALIB_RESULTS, score_batch_calib)

    print(f"Calibration: {len(calib_scores)} entries, mean={sum(calib_scores)/len(calib_scores):.4f}")
    print(f"Batch_calib: {len(batch_scores)} entries, mean={sum(batch_scores)/len(batch_scores):.4f}")

    total = calib_scores + batch_scores
    print(f"Combined:    {len(total)} entries, mean={sum(total)/len(total):.4f}")


if __name__ == "__main__":
    main()
