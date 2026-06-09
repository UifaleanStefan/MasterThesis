"""Phase 1.9 — LongMemEval dump-all Protocol B calibration + batch_calib judge.

Cells:
  longmemeval__dump-all__calibration__seed42   (100 entries)
  longmemeval__dump-all__batch_calib__seed42   (10 entries)

Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  Protocol B rubric

Key calibration findings:
  - doc0 (GPS issue): correct at after0-after5 (GPS system, take back to dealership);
    at after6+ switches to "check engine light" or becomes a refusal → only 6/10
  - doc1 (Data Analysis webinar): correct at after2, after3, after6; wrong at other points
  - doc2 (bike first): mostly wrong (car first); correct ONLY at after7, after9
  - doc3 (Samsung Galaxy S22): correct from after3 to after8 (6 points);
    at after3 reasoning slightly confused (says S22 pre-ordered Jan 28) but answer correct
  - doc4 (7 days): correct from after4 to after9 (6 points)
  - doc5 (30 days mass to Ash Wednesday): correct from after5 to after9 (5 points)!
    Best performer for this question — dump-all retains the date context well.
  - doc6 (14 days to find house): correct at after7, after8 (14 days Feb 15→Mar 1);
    wrong at after6 (19 days), after9 (16 days)
  - doc7 (tomatoes first): correct only at after7
  - doc8 (21 days Holi to mass): correct at after8 (21 days); after9 gets 20 days (off by 1)
  - doc9 (4 days Rack Fest): correct at after9

Batch_calib: 6/10 correct
  (doc2=bike, doc4=7days, doc5=30days, doc6=14days, doc8=21days, doc9=4days → 1.0)
  Wrong: doc0 (refusal), doc1 (cannot determine), doc3 (refusal), doc7 (marigold first)

Scoring rules (Protocol B):
  - acknowledge_missing: 1.0 honest refusal, 0.0 confident answer (hallucination)
  - answer: standard 5-point rubric on non-refusal predictions
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JQ = ROOT / "results" / "stage3" / "judge_queue"
CALIB_DIR = JQ / "longmemeval__dump-all__calibration__seed42"
BATCH_DIR  = JQ / "longmemeval__dump-all__batch_calib__seed42"

CALIB_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR.mkdir(parents=True, exist_ok=True)

CALIB_RESULTS = CALIB_DIR / "results.jsonl"
BATCH_RESULTS  = BATCH_DIR  / "results.jsonl"

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
# Only answer+non-refusal entries with non-zero scores.
# All acknowledge_missing entries handled by default logic.
# ---------------------------------------------------------------------------
CALIB_JUDGMENTS: dict[str, tuple[float, str]] = {
    # ── doc0_qa0: GPS correct at after0-after5 ───────────────────────────────
    "doc0_qa0__after0": (1.0,
        "Pred: 'The first issue you had with your new car after its first service was "
        "with the GPS system, which you had to take back to the dealership to get "
        "fixed.' EXACT: GPS system not functioning correctly. Score 1.0."),
    "doc0_qa0__after1": (1.0,
        "Pred: 'The first issue...was with the GPS system, which you had to take back "
        "to the dealership.' EXACT. Score 1.0."),
    "doc0_qa0__after2": (1.0,
        "Pred: 'The first issue...was with the GPS system...take back to dealership.' "
        "EXACT. Score 1.0."),
    "doc0_qa0__after3": (1.0,
        "Pred: 'The first issue...was with the GPS system...take back to dealership.' "
        "EXACT. Score 1.0."),
    "doc0_qa0__after4": (1.0,
        "Pred: 'The first issue...was with the GPS system...take back to dealership.' "
        "EXACT. Score 1.0."),
    "doc0_qa0__after5": (1.0,
        "Pred: 'The first issue...was with the GPS system...take back to dealership.' "
        "EXACT. Score 1.0."),
    # ── doc1_qa0: Data Analysis first at after2, after3, after6 ──────────────
    "doc1_qa0__after2": (1.0,
        "Pred: 'You attended the Data Analysis using Python webinar first, followed by "
        "the Effective Time Management workshop.' EXACT. Score 1.0."),
    "doc1_qa0__after3": (1.0,
        "Pred: 'You attended the Data Analysis using Python webinar first, and then "
        "the Effective Time Management workshop.' EXACT. Score 1.0."),
    "doc1_qa0__after6": (1.0,
        "Pred: 'You attended the Data Analysis using Python webinar first, as you "
        "mentioned it was organized two months ago, while the Effective Time Management "
        "workshop was last Saturday.' EXACT. Score 1.0."),
    # ── doc2_qa0: bike first — only at after7, after9 ────────────────────────
    "doc2_qa0__after7": (1.0,
        "Pred: 'You took care of your bike first in mid-February when you had to take "
        "it in for repairs because the gears were acting up.' CORRECT: bike first. "
        "Score 1.0."),
    "doc2_qa0__after9": (1.0,
        "Pred: 'You took care of the bike first in February.' CORRECT: bike first. "
        "Score 1.0."),
    # ── doc3_qa0: Samsung Galaxy S22 — after3 to after8 ─────────────────────
    "doc3_qa0__after3": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first, as you mentioned pre-ordering it "
        "on January 28th, and it arrived before your Dell XPS 13, which you received "
        "on February 25th.' Answer correct (S22 first); reasoning confuses S22/XPS "
        "pre-order dates but conclusion is right. Score 1.0."),
    "doc3_qa0__after4": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first, as you mentioned purchasing it "
        "from Best Buy on February 20th, and the Dell XPS 13 arrived later on February "
        "25th.' EXACT. Score 1.0."),
    "doc3_qa0__after5": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first on February 20th, and then the "
        "Dell XPS 13 arrived on February 25th.' EXACT. Score 1.0."),
    "doc3_qa0__after6": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first on February 20th.' EXACT. "
        "Score 1.0."),
    "doc3_qa0__after7": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first, as you mentioned purchasing it "
        "from Best Buy on February 20th.' EXACT. Score 1.0."),
    "doc3_qa0__after8": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first, as you mentioned purchasing it "
        "from Best Buy on February 20th.' EXACT. Score 1.0."),
    # ── doc4_qa0: 7 days — after4 to after9 ──────────────────────────────────
    "doc4_qa0__after4": (1.0,
        "Pred: '...January 10th...January 17th. Therefore, you attended the workshop "
        "7 days before the team meeting.' EXACT: 7 days. Score 1.0."),
    "doc4_qa0__after5": (1.0,
        "Pred: '...January 10th...January 17th. 7 days before the team meeting.' "
        "EXACT. Score 1.0."),
    "doc4_qa0__after6": (1.0,
        "Pred: '...January 10th...January 17th. 7 days before the team meeting.' "
        "EXACT. Score 1.0."),
    "doc4_qa0__after7": (1.0,
        "Pred: '...January 10th...team meeting January 17th. 7 days before.' "
        "EXACT. Score 1.0."),
    "doc4_qa0__after8": (1.0,
        "Pred: '...January 10th...team meeting January 17th. 7 days before.' "
        "EXACT. Score 1.0."),
    "doc4_qa0__after9": (1.0,
        "Pred: '...January 10th...team meeting January 17th. 7 days before.' "
        "EXACT. Score 1.0."),
    # ── doc5_qa0: 30 days — after5 to after9 (best performer) ────────────────
    "doc5_qa0__after5": (1.0,
        "Pred: 'The Sunday mass at St. Mary's Church was on January 2nd, and Ash "
        "Wednesday was on February 1st. There are 30 days between these two dates.' "
        "EXACT: 30 days. Score 1.0."),
    "doc5_qa0__after6": (1.0,
        "Pred: 'January 2nd to February 1st = 29 days (January) + 1 day (February) = "
        "30 days.' EXACT: 30 days. Score 1.0."),
    "doc5_qa0__after7": (1.0,
        "Pred: 'Ash Wednesday February 1st, Sunday mass January 2nd. 30 days.' "
        "EXACT: 30 days. Score 1.0."),
    "doc5_qa0__after8": (1.0,
        "Pred: 'Between the Sunday mass on January 2nd and Ash Wednesday on February "
        "1st, 30 days had passed.' EXACT: 30 days. Score 1.0."),
    "doc5_qa0__after9": (1.0,
        "Pred: 'January 2nd and February 1st. There are 30 days between these two "
        "dates.' EXACT: 30 days. Score 1.0."),
    # ── doc6_qa0: 14 days — at after7, after8 ────────────────────────────────
    "doc6_qa0__after7": (1.0,
        "Pred: 'You started working with Rachel on February 15th and found a house you "
        "loved on March 1st. It took you 14 days to find a house.' EXACT: 14 days. "
        "Score 1.0."),
    "doc6_qa0__after8": (1.0,
        "Pred: 'You started working with Rachel on February 15th and found a house you "
        "loved on March 1st. It took you 14 days.' EXACT: 14 days. Score 1.0."),
    # ── doc7_qa0: tomatoes first ──────────────────────────────────────────────
    "doc7_qa0__after7": (1.0,
        "Pred: 'The tomatoes were started first.' EXACT match. Score 1.0."),
    # ── doc8_qa0: 21 days — at after8 only (after9 gets 20) ─────────────────
    "doc8_qa0__after8": (1.0,
        "Pred: 'Holi February 26th, mass March 19th. (2 + 19) = 21 days.' "
        "EXACT: 21 days. Score 1.0."),
    # ── doc9_qa0: 4 days ──────────────────────────────────────────────────────
    "doc9_qa0__after9": (1.0,
        "Pred: 'Turbocharged Tuesdays June 14th, Rack Fest June 18th. 4 days before.' "
        "EXACT: 4 days. Score 1.0."),
}


# ---------------------------------------------------------------------------
# BATCH_CALIB JUDGMENTS: suffix → (score, rationale)
# suffix = qid minus "longmemeval__dump-all__batch__" and "__seed42"
# ---------------------------------------------------------------------------
BATCH_JUDGMENTS: dict[str, tuple[float, str]] = {
    "doc2_qa0": (1.0,
        "Pred: 'You took care of the bike first in February.' EXACT: bike first. "
        "Score 1.0."),
    "doc4_qa0": (1.0,
        "Pred: '...January 10th...January 17th. 7 days before the team meeting.' "
        "EXACT: 7 days. Score 1.0."),
    "doc5_qa0": (1.0,
        "Pred: 'Sunday mass January 2nd, Ash Wednesday February 1st. 30 days had "
        "passed.' EXACT: 30 days. Score 1.0."),
    "doc6_qa0": (1.0,
        "Pred: 'February 15th to March 1st. 14 days.' EXACT: 14 days. Score 1.0."),
    "doc8_qa0": (1.0,
        "Pred: 'Holi February 26th, mass March 19th. (2 + 19) = 21 days.' "
        "EXACT: 21 days. Score 1.0."),
    "doc9_qa0": (1.0,
        "Pred: 'Turbocharged Tuesdays June 14th, Rack Fest June 18th. 4 days before.' "
        "EXACT: 4 days. Score 1.0."),
}


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def score_calib(r: dict) -> tuple[float, str]:
    pred = r.get("predicted", "")
    behavior = r.get("expected_behavior", "answer")
    cfg = "dump-all"
    suffix = r["qid"].replace(f"longmemeval__{cfg}__calibration__", "").replace("__seed42", "")

    if behavior == "acknowledge_missing":
        if is_refusal(pred):
            return 1.0, "Honest refusal when source doc not yet ingested. Score 1.0."
        return 0.0, "Hallucination — answered when source not yet in memory. Score 0.0."

    if is_refusal(pred):
        return 0.0, "Refused when memory should contain source doc. Score 0.0."

    if suffix in CALIB_JUDGMENTS:
        return CALIB_JUDGMENTS[suffix]

    return 0.0, "Wrong answer — prediction does not match gold. Score 0.0."


def score_batch(r: dict) -> tuple[float, str]:
    pred = r.get("predicted", "")
    cfg = "dump-all"
    suffix = r["qid"].replace(f"longmemeval__{cfg}__batch__", "").replace("__seed42", "")

    if is_refusal(pred):
        return 0.0, "Refused to answer batch question. Score 0.0."

    if suffix in BATCH_JUDGMENTS:
        return BATCH_JUDGMENTS[suffix]

    return 0.0, "Wrong answer — prediction does not match gold. Score 0.0."


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def write_judgments(
    queue_path: Path,
    results_path: Path,
    score_fn,
    mode: str,
) -> None:
    if not queue_path.exists():
        print(f"  [SKIP] queue not found: {queue_path}")
        return

    rows = [json.loads(l) for l in queue_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    existing: set[str] = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing.add(json.loads(line)["qid"])

    new_records = []
    for r in rows:
        if r["qid"] in existing:
            continue
        score, rationale = score_fn(r)
        new_records.append({
            "qid": r["qid"],
            "judge_score": score,
            "rationale": rationale,
            "judge_model": "claude-opus-4.7-1m",
            "judge_protocol": "v1",
            "expected_behavior": r.get("expected_behavior", "answer"),
        })

    if new_records:
        with results_path.open("a", encoding="utf-8") as fh:
            for rec in new_records:
                fh.write(json.dumps(rec) + "\n")

    print(f"  [{mode}] {queue_path.parent.name}: {len(rows)} queued, "
          f"{len(new_records)} written, {len(existing)} already existed")


if __name__ == "__main__":
    print("Judging longmemeval__dump-all — calibration + batch_calib ...")
    write_judgments(CALIB_DIR / "queue.jsonl", CALIB_RESULTS, score_calib, "calib")
    write_judgments(BATCH_DIR  / "queue.jsonl", BATCH_RESULTS,  score_batch, "batch")
    print("Done.")
