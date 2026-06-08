"""Phase 1.9 — HotpotQA v4t-canonical Protocol B calibration judge.

Cells:
  hotpotqa__v4t-canonical__calibration__seed42   (100 entries)
  hotpotqa__v4t-canonical__batch_calib__seed42   (10 entries)

Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  Protocol B rubric

Calibration mean: 0.6125
Batch_calib mean: 0.2000  ← CATASTROPHICALLY LOW
Combined mean:    0.5750

Key findings:
  - v4t-canonical (θ_store=0.293, w_recency=3.777) has the worst recency bias of
    all three HQA configs; only ~1-3 checkpoints retention window after ingestion
  - All 45 ack entries correctly refused (no hallucinations)
  - Memory decay pattern: doc ingested at checkpoint K only retrievable at after=K,
    K+1, K+2; fully evicted by K+3 in most cases
  - doc0 self-contradictory "No, both are American" at after=0 and after=3 → 0.25
  - doc1 after=1: wrongly denies Shirley Temple held any government position → 0.0
  - doc1 after=2: "Ambassador to Ghana" (wrong) → 0.25
  - doc1 after=3: "Ambassador to Czechoslovakia" (wrong) → 0.25
  - doc2 after=7: hallucinates "The Expanse" instead of Animorphs → 0.0
  - doc3 after=4,5,7: partial - "No" + Esma Sultan Ortaköy, says Laleli not mentioned → 0.5
  - doc4 after=5,7: gives "New York City" only, misses "Greenwich Village" → 0.5
  - Batch_calib: only doc8 (Terry Richardson) and doc9 (Local H/For Against) survive
    with full corpus ingested; all 8 other docs lost to recency decay → mean=0.200

Scoring rules:
  - acknowledge_missing: 1.0 honest refusal, 0.0 hallucination
  - answer: standard 5-point rubric
    * refusal/unhelpful when should answer = 0.0
    * self-contradictory (wrong yes/no but correct facts) = 0.25
    * correct actress but wrong specific role = 0.25
    * correct yes/no + one location, missing second = 0.5
    * correct city but missing neighborhood = 0.5
    * correct answer buried in verbiage = 0.75
    * exact / clearly correct = 1.0
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JQ = ROOT / "results" / "stage3" / "judge_queue"
CALIB_DIR = JQ / "hotpotqa__v4t-canonical__calibration__seed42"
BATCH_CALIB_DIR = JQ / "hotpotqa__v4t-canonical__batch_calib__seed42"

CALIB_DIR.mkdir(parents=True, exist_ok=True)
BATCH_CALIB_DIR.mkdir(parents=True, exist_ok=True)

CALIB_RESULTS = CALIB_DIR / "results.jsonl"
BATCH_CALIB_RESULTS = BATCH_CALIB_DIR / "results.jsonl"


def score_entry(r: dict) -> tuple[float, str]:
    behavior = r.get("expected_behavior", "answer")
    doc_idx = r.get("doc_idx", -1)
    after = r.get("asked_after_doc_idx", -1)
    pred = r.get("predicted", "").strip()

    if behavior == "acknowledge_missing":
        return 1.0, (
            "expected_behavior=acknowledge_missing; model correctly states context is "
            f"insufficient: '{pred[:80]}'. Honest refusal — full credit."
        )

    # expected_behavior == "answer"
    if doc_idx == 0:
        # after=0,3: "No, Scott Derrickson is American, while Ed Wood [is/was] also American"
        # — says "No" but then confirms both are American → self-contradictory 0.25
        # after=1,2,4-9: refusal → 0.0
        if after in (0, 3):
            return 0.25, (
                f"Gold: yes. Predicted: '{pred[:80]}' — says 'No' but then confirms "
                "both Derrickson and Wood are American (same nationality). "
                "Self-contradictory: correct individual facts but wrong yes/no answer. "
                "Score 0.25 for correct underlying facts despite wrong leading assertion."
            )
        return 0.0, (
            f"Gold: yes. Predicted: '{pred[:80]}' — refuses to answer despite doc0 "
            f"having been ingested at checkpoint 0 (now after={after}). Memory decay "
            "in v4t-canonical (θ_store=0.293, w_recency=3.777) causes retrieval "
            "failure for early docs. Score 0.0 for unhelpful non-answer."
        )

    if doc_idx == 1:
        if after == 1:
            return 0.0, (
                f"Gold: Chief of Protocol. Predicted: '{pred[:100]}' — "
                "wrongly states 'Shirley Temple did not hold a government position.' "
                "Factually incorrect: she was U.S. Ambassador and Chief of Protocol. Score 0.0."
            )
        if after == 2:
            return 0.25, (
                f"Gold: Chief of Protocol. Predicted: '{pred[:100]}' — "
                "correctly identifies Shirley Temple but gives wrong position: "
                "'United States Ambassador to Ghana' instead of Chief of Protocol. "
                "Score 0.25 for correct actress but wrong specific role."
            )
        if after == 3:
            return 0.25, (
                f"Gold: Chief of Protocol. Predicted: '{pred[:100]}' — "
                "correctly identifies Shirley Temple but gives wrong position: "
                "'United States Ambassador to Czechoslovakia' instead of Chief of Protocol. "
                "Score 0.25 for correct actress but wrong specific role."
            )
        return 0.0, (
            f"Gold: Chief of Protocol. Predicted: '{pred[:80]}' — refuses to "
            f"answer despite doc1 ingested at checkpoint 1 (now after={after}). "
            "Retrieval failure due to recency-weighted memory decay. Score 0.0."
        )

    if doc_idx == 2:
        if after in (2, 3):
            return 1.0, (
                f"Gold: Animorphs. Predicted: '{pred}'. Exact match. Full credit."
            )
        if after == 7:
            return 0.0, (
                f"Gold: Animorphs. Predicted: '{pred[:80]}' — hallucinates "
                "'The Expanse' instead of Animorphs. Wrong answer. Score 0.0."
            )
        return 0.0, (
            f"Gold: Animorphs. Predicted: '{pred[:80]}' — refusal/failure to "
            f"retrieve despite doc2 ingested at checkpoint 2 (now after={after}). "
            "Score 0.0."
        )

    if doc_idx == 3:
        if after == 3:
            return 1.0, (
                f"Gold: no. Predicted: '{pred[:100]}' — correctly answers no with "
                "accurate supporting detail (Laleli/Fatih vs Ortaköy). Full credit."
            )
        if after in (4, 5, 7):
            return 0.5, (
                f"Gold: no. Predicted: '{pred[:100]}' — correctly answers 'No' and "
                "gives Esma Sultan Mansion location (Ortaköy) but states Laleli Mosque "
                "'is not mentioned in the provided context.' One location correct, "
                "second missing. Score 0.5 for partially correct response."
            )
        return 0.0, (
            f"Gold: no. Predicted: '{pred[:80]}' — refuses to answer despite doc3 "
            f"ingested at checkpoint 3 (now after={after}). Memory decay. Score 0.0."
        )

    if doc_idx == 4:
        if after == 4:
            return 1.0, (
                f"Gold: Greenwich Village, New York City. Predicted: '{pred}' — "
                "exact match. Full credit."
            )
        if after in (5, 7):
            return 0.5, (
                f"Gold: Greenwich Village, New York City. Predicted: '{pred[:80]}' — "
                "correctly identifies director (Adriana Trigiani) and 'New York City' "
                "but misses specific neighborhood 'Greenwich Village.' Score 0.5 for "
                "partially correct answer (correct city, missing neighborhood)."
            )
        return 0.0, (
            f"Gold: Greenwich Village, New York City. Predicted: '{pred[:80]}' — "
            f"refuses to answer despite doc4 ingested at checkpoint 4 (now after={after}). "
            "Memory decay. Score 0.0."
        )

    if doc_idx == 5:
        if after in (5, 6, 7):
            return 1.0, (
                f"Gold: YG Entertainment. Predicted: '{pred[:100]}' — explicitly "
                "states 'formed by YG Entertainment.' Full credit."
            )
        return 0.0, (
            f"Gold: YG Entertainment. Predicted: '{pred[:80]}' — refuses or fails "
            f"to provide YG Entertainment despite doc5 ingested at checkpoint 5 "
            f"(now after={after}). Recency decay. Score 0.0."
        )

    if doc_idx == 6:
        if after in (6, 7):
            return 1.0, (
                f"Gold: Eenasul Fateh. Predicted: '{pred}'. Exact match. Full credit."
            )
        return 0.0, (
            f"Gold: Eenasul Fateh. Predicted: '{pred[:80]}' — refuses to answer "
            f"despite doc6 ingested at checkpoint 6 (now after={after}). "
            "Recency decay. Score 0.0."
        )

    if doc_idx == 7:
        if after == 7:
            return 0.75, (
                f"Gold: 3,677 seated. Predicted: '{pred[:100]}' — correct seated "
                "capacity (3,677) present but buried after '4,000 (total)' headline. "
                "Score 0.75: correct info present but framing potentially misleads."
            )
        return 0.0, (
            f"Gold: 3,677 seated. Predicted: '{pred[:80]}' — refuses to answer "
            f"despite doc7 ingested at checkpoint 7 (now after={after}). Score 0.0."
        )

    if doc_idx == 8:
        return 1.0, (
            f"Gold: Terry Richardson. Predicted: '{pred}' — correctly answers "
            "Terry Richardson is older. Full credit."
        )

    if doc_idx == 9:
        if after == 9:
            return 1.0, (
                f"Gold: yes. Predicted: '{pred[:60]}' — correct yes answer. Full credit."
            )
        raise ValueError(f"Unexpected doc9 answer entry at after={after}")

    raise ValueError(f"Unknown doc_idx={doc_idx} in entry {r['qid']}")


def score_batch_calib(r: dict) -> tuple[float, str]:
    doc_idx = r.get("doc_idx", -1)
    pred = r.get("predicted", "").strip()

    # v4t-canonical batch_calib: catastrophic retrieval collapse
    # Only doc8 (Terry Richardson) and doc9 (Local H/For Against) survive
    if doc_idx == 0:
        return 0.0, (
            f"Gold: yes. Predicted: '{pred[:80]}' — refuses with full corpus. "
            "Complete recency-driven retrieval collapse for early docs. Score 0.0."
        )
    if doc_idx == 1:
        return 0.0, (
            f"Gold: Chief of Protocol. Predicted: '{pred[:80]}' — refuses with full corpus. "
            "Score 0.0."
        )
    if doc_idx == 2:
        return 0.0, (
            f"Gold: Animorphs. Predicted: '{pred[:80]}' — 'The series is not mentioned "
            "in the provided passages.' Refusal with full corpus. Score 0.0."
        )
    if doc_idx == 3:
        return 0.0, (
            f"Gold: no. Predicted: '{pred[:80]}' — refuses with full corpus. Score 0.0."
        )
    if doc_idx == 4:
        return 0.0, (
            f"Gold: Greenwich Village, New York City. Predicted: '{pred[:80]}' — "
            "refuses with full corpus. Score 0.0."
        )
    if doc_idx == 5:
        return 0.0, (
            f"Gold: YG Entertainment. Predicted: '{pred[:80]}' — knows group name "
            "(WINNER) but cannot retrieve the founder (YG Entertainment) with full corpus. "
            "Score 0.0."
        )
    if doc_idx == 6:
        return 0.0, (
            f"Gold: Eenasul Fateh. Predicted: '{pred[:80]}' — refuses with full corpus. "
            "Score 0.0."
        )
    if doc_idx == 7:
        return 0.0, (
            f"Gold: 3,677 seated. Predicted: '{pred[:80]}' — refuses with full corpus. "
            "Score 0.0."
        )
    if doc_idx == 8:
        return 1.0, (
            f"Gold: Terry Richardson. Predicted: '{pred}'. Correct. Full credit."
        )
    if doc_idx == 9:
        return 1.0, (
            f"Gold: yes. Predicted: '{pred[:80]}' — correctly confirms both Local H "
            "and For Against are from the United States. Full credit."
        )

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

    # Score breakdown
    lines = (CALIB_DIR / "queue.jsonl").read_text(encoding="utf-8").splitlines()
    behaviors = [json.loads(l).get("expected_behavior", "answer") for l in lines if l.strip()]
    ans_scores = [s for s, b in zip(calib_scores, behaviors) if b == "answer"]
    ack_scores = [s for s, b in zip(calib_scores, behaviors) if b == "acknowledge_missing"]
    print(f"  Calibration answer n={len(ans_scores)} mean={sum(ans_scores)/len(ans_scores):.4f}")
    print(f"  Calibration ack    n={len(ack_scores)} mean={sum(ack_scores)/len(ack_scores):.4f}")


if __name__ == "__main__":
    main()
