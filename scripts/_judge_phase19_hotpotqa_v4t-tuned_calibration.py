"""Phase 1.9 — HotpotQA v4t-tuned Protocol B calibration judge.

Cells:
  hotpotqa__v4t-tuned__calibration__seed42   (100 entries)
  hotpotqa__v4t-tuned__batch_calib__seed42   (10 entries)

Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  Protocol B rubric

Calibration mean: 0.6950
Batch_calib mean: 0.3750  ← VERY LOW (model fails to retrieve most answers at corpus end)

Key findings:
  - v4t-tuned (θ_store=0.306, w_embed=0.424, w_recency=1.121) stores fewer memories
    and is recency-biased; early docs (0-3) are effectively forgotten by after=5+
  - ALL 45 acknowledge_missing entries correctly refused (no hallucinations unlike
    v4t-corpus-tuned which hallucinated doc9 at after=7,8)
  - doc0 nationality: refusal at after=2 through after=9 (7/10 answer instances fail)
  - doc1 actress: wrong actress 'Janet Waldo' at after=1; only after=3 correct (0.75);
    after=7 gives correct actress but wrong role 'Ambassador to Ghana'
  - doc2 Animorphs: only after=2,3 correct; after=7 hallucinates 'The Illuminae Files'
  - doc7 seating: only after=7 correct (0.75); after=8 hallucinates 1,400 seats
  - Batch_calib: model fails on docs 0,1,2,5,7 at full corpus (retrieval collapse)

Scoring rules applied:
  - acknowledge_missing: 1.0 honest refusal, 0.0 confident hallucination
  - answer: standard 5-point rubric;
    * refusal when should answer = 0.0
    * correct = 1.0
    * self-contradictory yes/no = 0.25
    * correct but buried = 0.75
    * partial (correct city, wrong neighborhood) = 0.5
    * completely wrong = 0.0
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JQ = ROOT / "results" / "stage3" / "judge_queue"
CALIB_DIR = JQ / "hotpotqa__v4t-tuned__calibration__seed42"
BATCH_CALIB_DIR = JQ / "hotpotqa__v4t-tuned__batch_calib__seed42"

CALIB_DIR.mkdir(parents=True, exist_ok=True)
BATCH_CALIB_DIR.mkdir(parents=True, exist_ok=True)

CALIB_RESULTS = CALIB_DIR / "results.jsonl"
BATCH_CALIB_RESULTS = BATCH_CALIB_DIR / "results.jsonl"

# Per-entry scoring table: (doc_idx, asked_after_doc_idx, mode) -> (score, rationale_key)
# mode: 'cal'=calibration, 'bc'=batch_calib

# ---- Acknowledge-missing entries (all 1.0 for v4t-tuned; no hallucinations) ----
# These are scored uniformly by the score_entry function; no special-casing needed.

# ---- Answer entries with non-trivial scoring ----

def score_entry(r: dict) -> tuple[float, str]:
    behavior = r.get("expected_behavior", "answer")
    doc_idx = r.get("doc_idx", -1)
    after = r.get("asked_after_doc_idx", -1)
    pred = r.get("predicted", "").strip()

    if behavior == "acknowledge_missing":
        # v4t-tuned: all ack entries are proper refusals
        return 1.0, (
            "expected_behavior=acknowledge_missing; model correctly states context is "
            f"insufficient: '{pred[:80]}'. Honest refusal — full credit."
        )

    # expected_behavior == "answer"
    if doc_idx == 0:
        # after=0: correct "Yes, both are American"
        # after=1,3: self-contradictory "No, [both are American]"
        # after=2,4-9: refusal when should answer (memory decay of early doc)
        if after == 0:
            return 1.0, (
                f"Gold: yes. Predicted: '{pred[:80]}' — correctly answers 'Yes, both "
                "Scott Derrickson and Ed Wood are American.' Full credit."
            )
        if after in (1, 3):
            return 0.25, (
                f"Gold: yes. Predicted: '{pred[:80]}' — says 'No' but then confirms "
                "both Derrickson and Wood are American (same nationality). "
                "Self-contradictory: correct individual facts but wrong yes/no. "
                "Score 0.25 for correct underlying facts."
            )
        # after=2,4,5,6,7,8,9: refusal
        return 0.0, (
            f"Gold: yes. Predicted: '{pred[:80]}' — refuses to answer despite doc0 "
            f"having been ingested at checkpoint 0 (now after={after}). Memory decay "
            "of early docs in v4t-tuned (θ_store=0.306, w_recency=1.121) causes "
            "retrieval failure. Score 0.0 for unhelpful non-answer on answer-expected entry."
        )

    if doc_idx == 1:
        if after == 1:
            return 0.0, (
                f"Gold: Chief of Protocol. Predicted: '{pred[:100]}' — "
                "wrong actress identification ('Janet Waldo' instead of Shirley Temple) "
                "and claims no government position was held. Completely wrong answer. Score 0.0."
            )
        if after == 2:
            return 0.25, (
                f"Gold: Chief of Protocol. Predicted: '{pred[:100]}' — "
                "correctly identifies Shirley Temple but gives wrong position: "
                "'U.S. Ambassador to Czechoslovakia' instead of Chief of Protocol. "
                "Score 0.25 for correct actress but wrong specific role."
            )
        if after == 3:
            return 0.75, (
                f"Gold: Chief of Protocol. Predicted: '{pred[:100]}' — "
                "correct answer present: '...and also served as Chief of Protocol of "
                "the United States.' Also lists ambassador roles. Score 0.75 for "
                "correct but verbose/padded response."
            )
        if after == 7:
            return 0.25, (
                f"Gold: Chief of Protocol. Predicted: '{pred[:100]}' — "
                "correctly identifies Shirley Temple but gives wrong position: "
                "'United States Ambassador to Ghana' (not Chief of Protocol). "
                "Score 0.25 for correct actress but wrong specific role."
            )
        # after=4,5,6,8,9: refusal
        return 0.0, (
            f"Gold: Chief of Protocol. Predicted: '{pred[:80]}' — refuses to "
            f"answer despite doc1 ingested at checkpoint 1 (now after={after}). "
            "Retrieval failure due to memory decay. Score 0.0."
        )

    if doc_idx == 2:
        if after in (2, 3):
            return 1.0, (
                f"Gold: Animorphs. Predicted: '{pred}'. Exact match. Full credit."
            )
        if after == 7:
            return 0.0, (
                f"Gold: Animorphs. Predicted: '{pred[:80]}' — wrong answer: "
                "'The Illuminae Files' (hallucinated incorrect series). Score 0.0."
            )
        return 0.0, (
            f"Gold: Animorphs. Predicted: '{pred[:80]}' — refusal/failure to "
            f"retrieve despite doc2 ingested at checkpoint 2 (now after={after}). Score 0.0."
        )

    if doc_idx == 3:
        if after == 6:
            return 0.0, (
                f"Gold: no. Predicted: '{pred[:80]}' — refuses to answer despite "
                f"doc3 ingested at checkpoint 3 (now after={after}). Retrieval failure. Score 0.0."
            )
        return 1.0, (
            f"Gold: no. Predicted: '{pred[:100]}' — correctly answers no with "
            "accurate location detail (Laleli/Fatih vs Ortaköy). Full credit."
        )

    if doc_idx == 4:
        if after == 4:
            return 1.0, (
                f"Gold: Greenwich Village, New York City. Predicted: '{pred}' — "
                "exact match with full neighborhood and city. Full credit."
            )
        if after == 8:
            return 0.0, (
                f"Gold: Greenwich Village, New York City. Predicted: '{pred[:80]}' — "
                f"refuses to answer despite doc4 ingested (now after={after}). Score 0.0."
            )
        # after=5,6,7,9: gives "New York City" but not Greenwich Village specifically
        return 0.5, (
            f"Gold: Greenwich Village, New York City. Predicted: '{pred[:80]}' — "
            "correctly identifies director (Adriana Trigiani) and city (New York City) "
            "but misses the specific neighborhood 'Greenwich Village.' The question "
            "asks 'in what New York city?' expecting a specific place. Score 0.5 for "
            "partially correct answer (correct city, missing neighborhood)."
        )

    if doc_idx == 5:
        if after in (5, 6, 7):
            return 1.0, (
                f"Gold: YG Entertainment. Predicted: '{pred[:80]}' — explicitly "
                "states 'formed by YG Entertainment.' Full credit."
            )
        # after=8,9: refusal
        return 0.0, (
            f"Gold: YG Entertainment. Predicted: '{pred[:80]}' — refuses to "
            f"answer despite doc5 ingested at checkpoint 5 (now after={after}). Score 0.0."
        )

    if doc_idx == 6:
        return 1.0, (
            f"Gold: Eenasul Fateh. Predicted: '{pred}'. Exact match. Full credit."
        )

    if doc_idx == 7:
        if after == 7:
            return 0.75, (
                f"Gold: 3,677 seated. Predicted: '{pred[:100]}' — correct seated "
                "capacity (3,677) present but buried after '4,000 (total)' headline. "
                "Score 0.75: correct info but framing potentially misleads."
            )
        if after == 8:
            return 0.0, (
                f"Gold: 3,677 seated. Predicted: '{pred[:80]}' — completely wrong: "
                "says '1,400 people' (hallucinated). Correct answer is 3,677 seated. Score 0.0."
            )
        # after=9: refusal
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
        return 1.0, (
            f"Gold: yes. Predicted: '{pred[:60]}' — correct yes answer. Full credit."
        )

    raise ValueError(f"Unknown doc_idx={doc_idx} in entry {r['qid']}")


def score_batch_calib(r: dict) -> tuple[float, str]:
    doc_idx = r.get("doc_idx", -1)
    pred = r.get("predicted", "").strip()

    # v4t-tuned batch_calib: model fails to retrieve most answers at full corpus
    if doc_idx == 0:
        return 0.0, (
            f"Gold: yes. Predicted: '{pred[:80]}' — refuses to answer despite full "
            "corpus ingested. Retrieval collapse of early doc (doc0). Score 0.0."
        )
    if doc_idx == 1:
        return 0.0, (
            f"Gold: Chief of Protocol. Predicted: '{pred[:80]}' — refuses to answer "
            "with full corpus. Retrieval failure. Score 0.0."
        )
    if doc_idx == 2:
        return 0.0, (
            f"Gold: Animorphs. Predicted: '{pred[:80]}' — refuses to answer with "
            "full corpus. Score 0.0."
        )
    if doc_idx == 3:
        return 0.25, (
            f"Gold: no. Predicted: '{pred[:120]}' — gives Esma Sultan Mansion "
            "location (Ortaköy, Istanbul) correctly but states Laleli Mosque location "
            "is unknown, so 'cannot be determined if they are in the same neighborhood.' "
            "Partial credit: one location correct, but question unanswered. Score 0.25."
        )
    if doc_idx == 4:
        return 0.5, (
            f"Gold: Greenwich Village, New York City. Predicted: '{pred[:80]}' — "
            "correctly identifies director and 'New York City' but misses specific "
            "neighborhood (Greenwich Village). Score 0.5."
        )
    if doc_idx == 5:
        return 0.0, (
            f"Gold: YG Entertainment. Predicted: '{pred[:80]}' — refuses despite "
            "full corpus. Score 0.0."
        )
    if doc_idx == 6:
        return 1.0, (
            f"Gold: Eenasul Fateh. Predicted: '{pred}'. Exact match. Full credit."
        )
    if doc_idx == 7:
        return 0.0, (
            f"Gold: 3,677 seated. Predicted: '{pred[:80]}' — refuses despite full "
            "corpus ingested. Score 0.0."
        )
    if doc_idx == 8:
        return 1.0, (
            f"Gold: Terry Richardson. Predicted: '{pred}'. Correct. Full credit."
        )
    if doc_idx == 9:
        return 1.0, (
            f"Gold: yes. Predicted: '{pred[:60]}'. Correct. Full credit."
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
    ans = [s for s, b in zip(calib_scores,
                              [json.loads(l).get("expected_behavior","answer")
                               for l in (CALIB_DIR/"queue.jsonl").read_text(encoding="utf-8").splitlines()
                               if l.strip()])
           if b == "answer"]
    ack = [s for s, b in zip(calib_scores,
                              [json.loads(l).get("expected_behavior","answer")
                               for l in (CALIB_DIR/"queue.jsonl").read_text(encoding="utf-8").splitlines()
                               if l.strip()])
           if b == "acknowledge_missing"]
    print(f"  Calibration answer n={len(ans)} mean={sum(ans)/len(ans):.4f}")
    print(f"  Calibration ack    n={len(ack)} mean={sum(ack)/len(ack):.4f}")


if __name__ == "__main__":
    main()
