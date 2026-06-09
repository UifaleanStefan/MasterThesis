"""Phase 1.9 — LongMemEval v4t-corpus-tuned Protocol B calibration + batch_calib judge.

Cells:
  longmemeval__v4t-corpus-tuned__calibration__seed42   (100 entries)
  longmemeval__v4t-corpus-tuned__batch_calib__seed42   (10 entries)

Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  Protocol B rubric

Key calibration findings vs v4t-canonical:
  - doc0 (GPS issue): never retrieved; consistently says "check engine light" → all 0.0
  - doc1 (Data Analysis webinar first): retrieved at after2, after3, after5 only;
    at after6+ consistently wrong (ETM workshop first on March 15th — confused timing)
  - doc2 (bike first): correct from after3 onwards (7 of 10) — same as v4t-canonical
  - doc3 (Samsung Galaxy S22): correct from after3 to after9 — very stable
  - doc4 (7 days): correct from after4 to after9 — very stable
  - doc5 (30 days mass to Ash Wednesday): mostly 33 days; correct ONLY at after9
  - doc6 (14 days to find house): correct at after6, after8, after9 (not after7 = 19 days)
  - doc7 (tomatoes first): correct only at after7 (same as v4t-canonical)
  - doc8 (21 days Holi to mass): correct at after8, after9
  - doc9 (4 days Rack Fest): correct at after9

Batch_calib: 7/10 correct (doc2, doc3, doc4, doc5, doc6, doc8, doc9 — all 1.0)
Wrong: doc0 (check engine, not GPS), doc1 (ETM first), doc7 (marigold first)

Scoring rules (Protocol B):
  - acknowledge_missing: 1.0 honest refusal, 0.0 confident answer (hallucination)
  - answer: standard 5-point rubric on non-refusal predictions
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JQ = ROOT / "results" / "stage3" / "judge_queue"
CALIB_DIR = JQ / "longmemeval__v4t-corpus-tuned__calibration__seed42"
BATCH_DIR  = JQ / "longmemeval__v4t-corpus-tuned__batch_calib__seed42"

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
    # ── after2: docs 0,1 in memory ─────────────────────────────────────────
    "doc1_qa0__after2": (1.0,
        "Pred: 'You attended the Data Analysis using Python webinar first, as you "
        "mentioned it was two months ago, while the Effective Time Management workshop "
        "was last Saturday.' CORRECT: Data Analysis webinar first. Score 1.0."),
    # ── after3: docs 0-2 in memory ─────────────────────────────────────────
    "doc1_qa0__after3": (1.0,
        "Pred: 'You attended the Data Analysis using Python webinar first, and then "
        "the Effective Time Management workshop.' EXACT match. Score 1.0."),
    "doc2_qa0__after3": (1.0,
        "Pred: 'You took care of your bike first in February, as you mentioned having "
        "to take it in for repairs due to gear issues.' Correct: bike. Score 1.0."),
    "doc3_qa0__after3": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first, as you mentioned purchasing it "
        "on February 20th, while the Dell XPS 13 arrived later on February 25th.' "
        "EXACT. Score 1.0."),
    # ── after4 ─────────────────────────────────────────────────────────────
    "doc2_qa0__after4": (1.0,
        "Pred: 'You took care of your bike first in February, as you mentioned having "
        "to take it in for repairs due to gear issues.' Correct: bike. Score 1.0."),
    "doc3_qa0__after4": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first on February 20th, and then the "
        "Dell XPS 13, which you pre-ordered on January 28th and received on February "
        "25th.' EXACT: S22 first. Score 1.0."),
    "doc4_qa0__after4": (1.0,
        "Pred: '...workshop January 10th...team meeting January 17th. Therefore, you "
        "attended the workshop 7 days before the team meeting.' EXACT: 7 days. "
        "Score 1.0."),
    # ── after5 ─────────────────────────────────────────────────────────────
    "doc1_qa0__after5": (1.0,
        "Pred: 'You attended the Data Analysis using Python webinar first, and then "
        "the Effective Time Management workshop.' EXACT. Score 1.0."),
    "doc2_qa0__after5": (1.0,
        "Pred: 'You took care of your bike first in February, as you mentioned having "
        "to take it in for repairs in mid-February before discussing your car service "
        "on March 15th.' Correct: bike. Score 1.0."),
    "doc3_qa0__after5": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first on February 20th, and then the "
        "Dell XPS 13 later.' EXACT: S22 first. Score 1.0."),
    "doc4_qa0__after5": (1.0,
        "Pred: '...January 10th...January 17th. Therefore, you attended the workshop "
        "7 days before the team meeting.' EXACT: 7 days. Score 1.0."),
    # ── after6 ─────────────────────────────────────────────────────────────
    "doc2_qa0__after6": (1.0,
        "Pred: 'You took care of your bike first in February, as you mentioned having "
        "to take it in for repairs due to gear issues.' Correct: bike. Score 1.0."),
    "doc3_qa0__after6": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first on February 20th, and then the "
        "Dell XPS 13, which you pre-ordered on January 28th and received on February "
        "25th.' EXACT. Score 1.0."),
    "doc4_qa0__after6": (1.0,
        "Pred: '...January 10th...January 17th. That means you attended the workshop "
        "7 days before the team meeting.' EXACT: 7 days. Score 1.0."),
    "doc6_qa0__after6": (1.0,
        "Pred: 'It took you 14 days to find a house you loved after starting to work "
        "with Rachel.' EXACT: 14 days. Score 1.0."),
    # ── after7 ─────────────────────────────────────────────────────────────
    "doc2_qa0__after7": (1.0,
        "Pred: 'You took care of your bike first in February, as you mentioned having "
        "to take it in for repairs due to gear issues in mid-February.' Correct: bike. "
        "Score 1.0."),
    "doc3_qa0__after7": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first on February 20th, before "
        "receiving the Dell XPS 13 on February 25th.' EXACT. Score 1.0."),
    "doc4_qa0__after7": (1.0,
        "Pred: '...January 10th...January 17th. 7 days before the team meeting.' "
        "EXACT. Score 1.0."),
    "doc7_qa0__after7": (1.0,
        "Pred: 'The tomatoes were started first.' EXACT match. Score 1.0."),
    # ── after8 ─────────────────────────────────────────────────────────────
    "doc2_qa0__after8": (1.0,
        "Pred: 'You took care of your bike first in February, as you mentioned having "
        "to take it in for repairs due to gear issues.' Correct: bike. Score 1.0."),
    "doc3_qa0__after8": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first on February 20th, before "
        "receiving the Dell XPS 13 on February 25th.' EXACT. Score 1.0."),
    "doc4_qa0__after8": (1.0,
        "Pred: '...January 10th...January 17th. That means you attended the workshop "
        "7 days before the team meeting.' EXACT. Score 1.0."),
    "doc6_qa0__after8": (1.0,
        "Pred: 'It took you 14 days to find a house you loved after starting to work "
        "with Rachel, as you began working with her on February 15th and found the "
        "house on March 1st.' EXACT: 14 days. Score 1.0."),
    "doc8_qa0__after8": (1.0,
        "Pred: 'The Hindu festival of Holi was celebrated on February 26th, and the "
        "Sunday mass at St. Mary's Church was attended on March 19th. The number of "
        "days that passed between these two dates is 21 days.' EXACT: 21 days. "
        "Score 1.0."),
    # ── after9 ─────────────────────────────────────────────────────────────
    "doc2_qa0__after9": (1.0,
        "Pred: 'You took care of your bike first in February, as you mentioned having "
        "to take it in for repairs in mid-February.' Correct: bike. Score 1.0."),
    "doc3_qa0__after9": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first on February 20th, before "
        "receiving the Dell XPS 13 on February 25th.' EXACT. Score 1.0."),
    "doc4_qa0__after9": (1.0,
        "Pred: '...January 10th...January 17th. 7 days before the team meeting.' "
        "EXACT. Score 1.0."),
    "doc5_qa0__after9": (1.0,
        "Pred: 'The Sunday mass at St. Mary's Church was on January 2nd, and Ash "
        "Wednesday was on February 1st. There are 30 days between these two dates.' "
        "EXACT: 30 days. Score 1.0."),
    "doc6_qa0__after9": (1.0,
        "Pred: 'It took you about 14 days to find a house you loved after starting to "
        "work with Rachel, as you began working with her on February 15th and found the "
        "house on March 1st.' EXACT: 14 days. Score 1.0."),
    "doc8_qa0__after9": (1.0,
        "Pred: 'The Hindu festival of Holi was celebrated on February 26th, and the "
        "Sunday mass at St. Mary's Church was attended on March 19th. The number of "
        "days that passed between these two dates is 21 days.' EXACT: 21 days. "
        "Score 1.0."),
    "doc9_qa0__after9": (1.0,
        "Pred: 'You participated in the Turbocharged Tuesdays event on June 14th, and "
        "Rack Fest was on June 18th. Therefore, you participated in Turbocharged "
        "Tuesdays 4 days before Rack Fest.' EXACT: 4 days. Score 1.0."),
}


# ---------------------------------------------------------------------------
# BATCH_CALIB JUDGMENTS: suffix → (score, rationale)
# suffix = qid minus "longmemeval__v4t-corpus-tuned__batch__" and "__seed42"
# ---------------------------------------------------------------------------
BATCH_JUDGMENTS: dict[str, tuple[float, str]] = {
    "doc2_qa0": (1.0,
        "Pred: 'You took care of your bike first in February, as you mentioned having "
        "to take it in for repairs in mid-February.' EXACT: bike first. Score 1.0."),
    "doc3_qa0": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first on February 20th, before "
        "receiving the Dell XPS 13 on February 25th.' EXACT: S22 first. Score 1.0."),
    "doc4_qa0": (1.0,
        "Pred: '...January 10th...January 17th. 7 days before the team meeting.' "
        "EXACT: 7 days. Score 1.0."),
    "doc5_qa0": (1.0,
        "Pred: 'The Sunday mass at St. Mary's Church was on January 2nd, and Ash "
        "Wednesday was on February 1st. There are 30 days between these two dates.' "
        "EXACT: 30 days. Score 1.0."),
    "doc6_qa0": (1.0,
        "Pred: 'It took you about 14 days to find a house you loved after starting to "
        "work with Rachel, as you began working with her on February 15th and found the "
        "house on March 1st.' EXACT: 14 days. Score 1.0."),
    "doc8_qa0": (1.0,
        "Pred: 'The Hindu festival of Holi was celebrated on February 26th, and the "
        "Sunday mass at St. Mary's Church was attended on March 19th. 21 days.' "
        "EXACT: 21 days. Score 1.0."),
    "doc9_qa0": (1.0,
        "Pred: 'You participated in Turbocharged Tuesdays on June 14th, and Rack Fest "
        "was on June 18th. Therefore, you participated 4 days before Rack Fest.' "
        "EXACT: 4 days. Score 1.0."),
}


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def score_calib(r: dict) -> tuple[float, str]:
    pred = r.get("predicted", "")
    behavior = r.get("expected_behavior", "answer")
    cfg = "v4t-corpus-tuned"
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
    cfg = "v4t-corpus-tuned"
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
    print("Judging longmemeval__v4t-corpus-tuned — calibration + batch_calib ...")
    write_judgments(CALIB_DIR / "queue.jsonl", CALIB_RESULTS, score_calib, "calib")
    write_judgments(BATCH_DIR  / "queue.jsonl", BATCH_RESULTS,  score_batch, "batch")
    print("Done.")
