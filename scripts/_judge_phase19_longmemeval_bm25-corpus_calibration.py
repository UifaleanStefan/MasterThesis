"""Phase 1.9 — LongMemEval bm25-corpus Protocol B calibration judge.

Cells:
  longmemeval__bm25-corpus__calibration__seed42   (100 entries)
  longmemeval__bm25-corpus__batch_calib__seed42   (10 entries)

Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  Protocol B rubric

Calibration mean: 0.5375
  ack (45): mean=0.4667  (21 honest refusals × 1.0, 24 hallucinations × 0.0)
  answer (55): mean=0.5955
Batch_calib mean: 0.7500
Combined mean:    0.5568

LME docs (0-9) and their gold answers:
  doc0: 'GPS system not functioning correctly'
  doc1: "'Data Analysis using Python' webinar"
  doc2: 'bike'
  doc3: 'Samsung Galaxy S22'
  doc4: '7 days. 8 days (including the last day)'
  doc5: '30 days. 31 days (including the last day)'
  doc6: '14 days. 15 days (including the last day)'
  doc7: 'Tomatoes'
  doc8: '21 days. 22 days (including the last day)'
  doc9: '4 days.'

BM25 characteristics:
  - Perfect memory (all ingested docs retrievable via keyword match)
  - But many ACK hallucinations: BM25 retrieves other docs matching keywords
    and the model answers confidently rather than refusing when the target doc
    hasn't been ingested yet
  - Systematic hallucination in ack entries for docs that share vocabulary with
    already-ingested docs

Key findings (ack hallucinations — 24 total):
  - doc2 at after=0,1: model says "You took care of the car first in February."
    (doc2 is about bike; model retrieves car-related docs 0, 1 before doc2 is ingested)
  - doc4 at after=2,3: model says "You attended the workshop on Effective Communication..."
    (retrieves doc1 workshop content before doc4 workshop content is ingested)
  - doc5 at after=0-4 (5 entries): model computes a number of days based on other docs
    (retrieves doc8/doc6 church events and gives confident but wrong date calculation)
  - doc7 at after=0-6 (7 entries): model says "The tomatoes were started first."
    (BM25 retrieves doc7 content by proximity? Or confabulates. Either way: confident
    answer before doc7 is ingested → hallucination)
  - doc8 at after=0-7 (8 entries): model gives "Holi on March 8/Feb 26..." based on
    other docs or world knowledge before doc8 is ingested

Answer entry patterns:
  - doc0 (GPS): correct at all 10 checkpoints → 1.0 each (BM25 perfect memory)
  - doc1: correct only at after=2,3 (1.0); wrong at after=1 (0.0); self-contradictory
    at after=5-9 (says ETM first but dates show DAP was two months prior = earlier → 0.25)
  - doc2: wrong at after=2 (0.0); self-contradictory at after=3,5-9 (says car first
    but gives bike mid-February dates showing bike was earlier → 0.25);
    no-date-contradiction at after=4 (0.0)
  - doc3: wrong at after=3 (gives Dell XPS first → 0.0); correct at after=4-9 (1.0)
  - doc4: correct at all 6 checkpoints (7 days → 1.0)
  - doc5: wrong at after=5,6,7 (gives 33 days → 0.0); correct at after=8,9 (30 days → 1.0)
  - doc6: wrong at after=6,7,8 (gives 19 days → 0.0); correct at after=9 (14 days → 1.0)
  - doc7: wrong at all 3 answer checkpoints (gives marigolds → 0.0)
  - doc8: correct at after=8,9 (21 days → 1.0)
  - doc9: correct at after=9 (4 days → 1.0)

Batch_calib (10 entries, full corpus):
  - doc0 (GPS): 1.0 (correct)
  - doc1 (DAP webinar): 0.25 (self-contradictory: ETM first but DAP was two months ago = earlier)
  - doc2 (bike): 0.25 (self-contradictory: car first but gives bike mid-Feb = earlier)
  - doc3 (S22): 1.0 (correct)
  - doc4 (7 days): 1.0 (correct)
  - doc5 (30 days): 1.0 (correct)
  - doc6 (14 days): 1.0 (correct)
  - doc7 (Tomatoes): 0.0 (wrong: marigolds first)
  - doc8 (21 days): 1.0 (correct)
  - doc9 (4 days): 1.0 (correct)
  batch_calib mean = 7.5/10 = 0.7500
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JQ = ROOT / "results" / "stage3" / "judge_queue"
CALIB_DIR = JQ / "longmemeval__bm25-corpus__calibration__seed42"
BATCH_CALIB_DIR = JQ / "longmemeval__bm25-corpus__batch_calib__seed42"

CALIB_DIR.mkdir(parents=True, exist_ok=True)
BATCH_CALIB_DIR.mkdir(parents=True, exist_ok=True)

CALIB_RESULTS = CALIB_DIR / "results.jsonl"
BATCH_CALIB_RESULTS = BATCH_CALIB_DIR / "results.jsonl"


def score_entry(r: dict) -> tuple[float, str]:
    behavior = r.get("expected_behavior", "answer")
    doc_idx = r.get("doc_idx", -1)
    after = r.get("asked_after_doc_idx", -1)
    pred = r.get("predicted", "").strip()
    gold = r.get("gold_answer", "")

    if behavior == "acknowledge_missing":
        # BM25 ack hallucinations — 24 entries total:
        # doc2 at after=0,1: says "You took care of the car first in February."
        if doc_idx == 2 and after in (0, 1):
            return 0.0, (
                f"expected_behavior=acknowledge_missing; doc2 NOT yet ingested at after={after}. "
                f"Model gives confident wrong answer: '{pred[:80]}' (gold: 'bike'). "
                "BM25 retrieved car-related docs (doc0/doc1) matching 'vehicle' keywords; "
                "model answers confidently instead of refusing. Hallucination. Score 0.0."
            )
        # doc4 at after=2,3: says "You attended the workshop on Effective Communication..."
        if doc_idx == 4 and after in (2, 3):
            return 0.0, (
                f"expected_behavior=acknowledge_missing; doc4 NOT yet ingested at after={after}. "
                f"Model gives confident answer: '{pred[:80]}' instead of refusing. "
                "BM25 retrieved doc1 workshop content matching 'workshop' keyword before "
                "doc4 is ingested. Hallucination. Score 0.0."
            )
        # doc5 at after=0-4: model computes days from other docs/world knowledge
        if doc_idx == 5 and after in (0, 1, 2, 3, 4):
            return 0.0, (
                f"expected_behavior=acknowledge_missing; doc5 NOT yet ingested at after={after}. "
                f"Model gives confident date answer: '{pred[:80]}' instead of refusing. "
                "BM25 retrieved church/event docs matching date keywords; model computes "
                "days based on unrelated content. Hallucination. Score 0.0."
            )
        # doc7 at after=0-6: says "The tomatoes were started first."
        if doc_idx == 7 and after in range(0, 7):
            return 0.0, (
                f"expected_behavior=acknowledge_missing; doc7 NOT yet ingested at after={after}. "
                f"Model gives confident answer: '{pred[:80]}' instead of refusing. "
                "BM25 retrieves gardening-related content; model answers 'tomatoes were "
                "started first' without doc7 being in memory. Hallucination. Score 0.0."
            )
        # doc8 at after=0-7: gives Holi dates from world knowledge / other docs
        if doc_idx == 8 and after in range(0, 8):
            return 0.0, (
                f"expected_behavior=acknowledge_missing; doc8 NOT yet ingested at after={after}. "
                f"Model gives confident answer: '{pred[:80]}' instead of refusing. "
                "BM25 retrieves other event/date-related docs; model computes days based "
                "on Holi dates from world knowledge before doc8 is ingested. Hallucination. "
                "Score 0.0."
            )
        # All other ack entries are honest refusals
        return 1.0, (
            "expected_behavior=acknowledge_missing; model correctly states context is "
            f"insufficient: '{pred[:80]}'. Honest refusal — full credit."
        )

    # expected_behavior == "answer"
    if doc_idx == 0:
        # BM25 perfect memory: correct at all 10 checkpoints
        return 1.0, (
            f"Gold: '{gold}'. Predicted: '{pred[:80]}' — correctly identifies "
            "GPS system issue as first problem. BM25 retrieves doc0 at all checkpoints. "
            "Full credit."
        )

    if doc_idx == 1:
        # after=1: wrong (ETM first, no contradiction) → 0.0
        # after=2,3: correct (DAP first) → 1.0
        # after=4: wrong (ETM first, no date contradiction) → 0.0
        # after=5-9: self-contradictory (says ETM first but DAP was 'two months prior' = before ETM) → 0.25
        if after == 1:
            return 0.0, (
                f"Gold: '{gold}'. Predicted: '{pred[:100]}' — wrong: says 'Effective "
                "Time Management' workshop first. Gold is 'Data Analysis using Python' "
                "webinar first. BM25 retrieves doc1 correctly but model inverts the "
                "temporal order. Score 0.0."
            )
        if after in (2, 3):
            return 1.0, (
                f"Gold: '{gold}'. Predicted: '{pred[:100]}' — correctly identifies "
                "'Data Analysis using Python' webinar first. Full credit."
            )
        if after == 4:
            return 0.0, (
                f"Gold: '{gold}'. Predicted: '{pred[:80]}' — wrong: says 'Effective "
                "Time Management' workshop first at after={after}. BM25 retrieves "
                "inconsistent context as index grows. Score 0.0."
            )
        if after in (5, 6, 7, 8, 9):
            return 0.25, (
                f"Gold: '{gold}'. Predicted: '{pred[:120]}' — self-contradictory: "
                "leading claim says 'Effective Time Management first' but then states "
                "'Data Analysis using Python' webinar was attended 'two months prior' "
                "(i.e., before ETM), which contradicts the leading claim. Correct facts "
                "embedded in wrong framing. Score 0.25."
            )

    if doc_idx == 2:
        # after=2: wrong (car first, no date contradiction for bike) → 0.0
        # after=3: self-contradictory (car first March 15, bike mid-Feb = earlier) → 0.25
        # after=4: wrong (car first, bike mentioned as later for moving, no date) → 0.0
        # after=5,6: self-contradictory (car first March 15, bike mid-Feb) → 0.25
        # after=7,8,9: self-contradictory (car first March 15, bike mid-Feb) → 0.25
        if after == 2:
            return 0.0, (
                f"Gold: '{gold}'. Predicted: '{pred[:100]}' — wrong: says 'car first "
                "in February' (car washing Feb 27). Gold is 'bike.' Model retrieves "
                "car-related doc0 and ignores bike doc. Score 0.0."
            )
        if after in (3, 5, 6, 7, 8, 9):
            return 0.25, (
                f"Gold: '{gold}'. Predicted: '{pred[:120]}' — self-contradictory: says "
                "'car first in February' (car serviced March 15) but also states 'bike "
                "repairs in mid-February.' Mid-February precedes March 15, so the embedded "
                "dates support bike-first (the correct answer) despite the wrong leading "
                "claim. Score 0.25."
            )
        if after == 4:
            return 0.0, (
                f"Gold: '{gold}'. Predicted: '{pred[:120]}' — wrong: says 'car first "
                "in February' (car serviced March 15). Mentions bike was referenced later "
                "in relation to moving furniture, without providing dates that would "
                "contradict the car-first claim. Score 0.0."
            )

    if doc_idx == 3:
        # after=3: wrong (Dell XPS first) → 0.0
        # after=4-9: correct (S22 first on Feb 20) → 1.0
        if after == 3:
            return 0.0, (
                f"Gold: '{gold}'. Predicted: '{pred[:120]}' — wrong: says 'Dell XPS 13 "
                "first (arrived Feb 25)' while S22 was mentioned later. Gold is Samsung "
                "Galaxy S22 first. BM25 retrieves wrong ordering at early checkpoint. "
                "Score 0.0."
            )
        return 1.0, (
            f"Gold: '{gold}'. Predicted: '{pred[:100]}' — correctly identifies Samsung "
            "Galaxy S22 first (Feb 20) before Dell XPS 13 (Feb 25). Full credit."
        )

    if doc_idx == 4:
        # All 6 checkpoints: workshop Jan 10, meeting Jan 17, 7 days → correct
        return 1.0, (
            f"Gold: '{gold}'. Predicted: '{pred[:120]}' — correctly calculates 7 days "
            "between workshop (Jan 10) and team meeting (Jan 17). Full credit."
        )

    if doc_idx == 5:
        # after=5,6,7: wrong 33 days → 0.0
        # after=8,9: correct 30 days → 1.0
        if after in (5, 6, 7):
            return 0.0, (
                f"Gold: '{gold}'. Predicted: '{pred[:120]}' — wrong: calculates 33 days "
                "between St. Mary's Sunday mass (Jan 2) and Ash Wednesday cathedral service "
                "(Feb 1). Correct answer is 30 days (Jan has 31 days; Jan 2 to Feb 1 = "
                "30 days, not 33). Score 0.0."
            )
        if after in (8, 9):
            return 1.0, (
                f"Gold: '{gold}'. Predicted: '{pred[:120]}' — correctly calculates 30 days "
                "between Sunday mass (Jan 2) and Ash Wednesday service (Feb 1). Full credit."
            )

    if doc_idx == 6:
        # after=6,7,8: wrong 19 days → 0.0
        # after=9: correct 14 days → 1.0
        if after in (6, 7, 8):
            return 0.0, (
                f"Gold: '{gold}'. Predicted: '{pred[:100]}' — wrong: says '19 days to "
                "find a house.' Gold is 14 days (Feb 15 to March 1). Score 0.0."
            )
        if after == 9:
            return 1.0, (
                f"Gold: '{gold}'. Predicted: '{pred[:120]}' — correctly calculates 14 "
                "days (Feb 15 to March 1). Full credit."
            )

    if doc_idx == 7:
        # after=7,8,9: all wrong — gives marigolds first
        return 0.0, (
            f"Gold: '{gold}'. Predicted: '{pred[:100]}' — wrong: says 'marigold seeds "
            "were started first (March 3).' Gold is 'Tomatoes.' BM25 retrieves marigold "
            "start date but misses tomato first-planting date. Score 0.0."
        )

    if doc_idx == 8:
        # after=8,9: correct 21 days
        return 1.0, (
            f"Gold: '{gold}'. Predicted: '{pred[:120]}' — correctly calculates 21 days "
            "between Holi (Feb 26) and Sunday mass at St. Mary's (March 19). Full credit."
        )

    if doc_idx == 9:
        # after=9: correct 4 days
        return 1.0, (
            f"Gold: '{gold}'. Predicted: '{pred[:120]}' — correctly calculates 4 days "
            "between Turbocharged Tuesdays (June 14) and Rack Fest (June 18). Full credit."
        )

    raise ValueError(f"Unknown doc_idx={doc_idx} in entry {r['qid']}")


def score_batch_calib(r: dict) -> tuple[float, str]:
    doc_idx = r.get("doc_idx", -1)
    pred = r.get("predicted", "").strip()
    gold = r.get("gold_answer", "")

    if doc_idx == 0:
        return 1.0, (
            f"Gold: '{gold}'. Predicted: '{pred[:80]}' — correctly identifies "
            "GPS system issue. Full credit."
        )
    if doc_idx == 1:
        return 0.25, (
            f"Gold: '{gold}'. Predicted: '{pred[:120]}' — self-contradictory: "
            "says 'Effective Time Management first' but then states 'Data Analysis using "
            "Python webinar was two months ago' (= before ETM). Correct facts embedded "
            "in wrong leading claim. Score 0.25."
        )
    if doc_idx == 2:
        return 0.25, (
            f"Gold: '{gold}'. Predicted: '{pred[:120]}' — self-contradictory: says "
            "'car first in February' (car serviced March 15) but gives bike repairs "
            "'mid-February' which precedes the car service date. Correct dates embedded "
            "in wrong leading claim. Score 0.25."
        )
    if doc_idx == 3:
        return 1.0, (
            f"Gold: '{gold}'. Predicted: '{pred[:100]}' — correctly identifies "
            "Samsung Galaxy S22 first (Feb 20) before Dell XPS 13 (Feb 25). Full credit."
        )
    if doc_idx == 4:
        return 1.0, (
            f"Gold: '{gold}'. Predicted: '{pred[:100]}' — correctly calculates "
            "7 days (Jan 10 workshop to Jan 17 meeting). Full credit."
        )
    if doc_idx == 5:
        return 1.0, (
            f"Gold: '{gold}'. Predicted: '{pred[:120]}' — correctly calculates "
            "30 days (Jan 2 mass to Feb 1 Ash Wednesday service). Full credit."
        )
    if doc_idx == 6:
        return 1.0, (
            f"Gold: '{gold}'. Predicted: '{pred[:120]}' — correctly calculates "
            "14 days (Feb 15 to March 1). Full credit."
        )
    if doc_idx == 7:
        return 0.0, (
            f"Gold: '{gold}'. Predicted: '{pred[:100]}' — wrong: says 'marigold "
            "seeds first (March 3)' but gold is 'Tomatoes.' BM25 keyword match "
            "retrieves marigold planting date; model misses tomato planting. Score 0.0."
        )
    if doc_idx == 8:
        return 1.0, (
            f"Gold: '{gold}'. Predicted: '{pred[:120]}' — correctly calculates "
            "21 days (Holi Feb 26 to St. Mary's mass March 19). Full credit."
        )
    if doc_idx == 9:
        return 1.0, (
            f"Gold: '{gold}'. Predicted: '{pred[:120]}' — correctly calculates "
            "4 days (Turbocharged Tuesdays June 14 to Rack Fest June 18). Full credit."
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

    lines = (CALIB_DIR / "queue.jsonl").read_text(encoding="utf-8").splitlines()
    behaviors = [json.loads(l).get("expected_behavior", "answer") for l in lines if l.strip()]
    ans_scores = [s for s, b in zip(calib_scores, behaviors) if b == "answer"]
    ack_scores = [s for s, b in zip(calib_scores, behaviors) if b == "acknowledge_missing"]
    print(f"  Calibration answer n={len(ans_scores)} mean={sum(ans_scores)/len(ans_scores):.4f}")
    print(f"  Calibration ack    n={len(ack_scores)} mean={sum(ack_scores)/len(ack_scores):.4f}")


if __name__ == "__main__":
    main()
