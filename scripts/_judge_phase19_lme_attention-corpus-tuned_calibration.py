"""Phase 1.9 — LongMemEval attention-corpus-tuned Protocol B calibration + batch_calib judge.

Cells:
  longmemeval__attention-corpus-tuned__calibration__seed42   (100 entries)
  longmemeval__attention-corpus-tuned__batch_calib__seed42   (10 entries)

Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  Protocol B rubric

Key calibration findings vs other configs:
  - doc0 (GPS issue): BEST config for doc0 — retrieves GPS correctly at ALL 10 time
    points (after0 through after9). Unique strength: "GPS system, had to take back
    to the dealership to get fixed. They replaced the entire system."
  - doc1 (Data Analysis webinar first): only correct at after3; ETM-first wrong at
    after1-2, after4-9
  - doc2 (bike first): mostly wrong (car first); correct ONLY at after6. The GPS
    narrative bleeds over and causes car-first errors at most time points.
  - doc3 (Samsung Galaxy S22): correct from after3 to after9 — stable
  - doc4 (7 days): correct from after4 to after9
  - doc5 (30 days mass to Ash Wednesday): correct at after6, after7; wrong at
    after5 (33), after8 (32), after9 (32)
  - doc6 (14 days to find house): consistently wrong (18 or 19 days); never correct
  - doc7 (tomatoes first): correct only at after7
  - doc8 (21 days Holi to mass): correct at after8, after9
  - doc9 (4 days Rack Fest): wrong at after9 (says 14 days on wrong dates)

Batch_calib: 4/10 correct (doc0=GPS 1.0, doc3=S22 1.0, doc4=7days 1.0, doc8=21days 1.0)
Wrong: doc1 (ETM first), doc2 (car first), doc5 (32 days), doc6 (18 days),
       doc7 (marigold first), doc9 (14 days on wrong dates)

Scoring rules (Protocol B):
  - acknowledge_missing: 1.0 honest refusal, 0.0 confident answer (hallucination)
  - answer: standard 5-point rubric on non-refusal predictions
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JQ = ROOT / "results" / "stage3" / "judge_queue"
CALIB_DIR = JQ / "longmemeval__attention-corpus-tuned__calibration__seed42"
BATCH_DIR  = JQ / "longmemeval__attention-corpus-tuned__batch_calib__seed42"

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
# Notable: doc0 GPS correct at ALL after-points (unique among configs).
# ---------------------------------------------------------------------------
CALIB_JUDGMENTS: dict[str, tuple[float, str]] = {
    # ── doc0_qa0: GPS system — correct at ALL 10 time points ─────────────────
    "doc0_qa0__after0": (1.0,
        "Pred: 'The first issue you had with your new car after its first service was "
        "with the GPS system, which you had to take back to the dealership to get "
        "fixed.' EXACT: GPS system not functioning correctly. Score 1.0."),
    "doc0_qa0__after1": (1.0,
        "Pred: 'The first issue...was with the GPS system, which you had to take back "
        "to the dealership to get fixed.' EXACT. Score 1.0."),
    "doc0_qa0__after2": (1.0,
        "Pred: 'The first issue...was with the GPS system, which you had to take back "
        "to the dealership to get fixed on March 22nd. They replaced the entire system.' "
        "EXACT: GPS system not functioning correctly. Score 1.0."),
    "doc0_qa0__after3": (1.0,
        "Pred: 'The first issue...was with the GPS system, which you had to take back "
        "to the dealership to get fixed. They replaced the entire system.' "
        "EXACT. Score 1.0."),
    "doc0_qa0__after4": (1.0,
        "Pred: 'The first issue...was with the GPS system...replaced the entire system, "
        "and it is now working flawlessly.' EXACT. Score 1.0."),
    "doc0_qa0__after5": (1.0,
        "Pred: 'The first issue...was with the GPS system...replaced the entire system.' "
        "EXACT. Score 1.0."),
    "doc0_qa0__after6": (1.0,
        "Pred: 'The first issue...was with the GPS system...now working flawlessly.' "
        "EXACT. Score 1.0."),
    "doc0_qa0__after7": (1.0,
        "Pred: 'The first issue...was with the GPS system...now working flawlessly.' "
        "EXACT. Score 1.0."),
    "doc0_qa0__after8": (1.0,
        "Pred: 'The first issue...was with the GPS system...replaced the entire system, "
        "now working flawlessly.' EXACT. Score 1.0."),
    "doc0_qa0__after9": (1.0,
        "Pred: 'The first issue...was with the GPS system...replaced the entire system, "
        "now working flawlessly.' EXACT. Score 1.0."),
    # ── doc1_qa0: Data Analysis webinar first ────────────────────────────────
    "doc1_qa0__after3": (1.0,
        "Pred: 'You attended the Data Analysis using Python webinar first, as you "
        "mentioned it was two months ago, while the Effective Time Management workshop "
        "was held last Saturday.' CORRECT: Data Analysis first. Score 1.0."),
    # ── doc2_qa0: bike first — only correct at after6 ────────────────────────
    "doc2_qa0__after6": (1.0,
        "Pred: 'You took care of your bike first in February, as you mentioned having "
        "to take it in for repairs due to gear issues.' CORRECT: bike first. Score 1.0."),
    # ── doc3_qa0: Samsung Galaxy S22 ─────────────────────────────────────────
    "doc3_qa0__after3": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first, as you mentioned purchasing it "
        "on February 20th, while the Dell XPS 13 arrived later on February 25th.' "
        "EXACT. Score 1.0."),
    "doc3_qa0__after4": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first on February 20th, and then the "
        "Dell XPS 13 arrived on February 25th.' EXACT. Score 1.0."),
    "doc3_qa0__after5": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first on February 20th, before "
        "receiving the Dell XPS 13 on February 25th.' EXACT. Score 1.0."),
    "doc3_qa0__after6": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first, as you mentioned purchasing it "
        "from Best Buy on February 20th, while the Dell XPS 13 was pre-ordered on "
        "January 28th and arrived on February 25th.' EXACT. Score 1.0."),
    "doc3_qa0__after7": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first on February 20th, before "
        "receiving the Dell XPS 13 on February 25th.' EXACT. Score 1.0."),
    "doc3_qa0__after8": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first on February 20th, before "
        "receiving the Dell XPS 13 on February 25th.' EXACT. Score 1.0."),
    "doc3_qa0__after9": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first on February 20th, before "
        "receiving the Dell XPS 13 on February 25th.' EXACT. Score 1.0."),
    # ── doc4_qa0: 7 days ─────────────────────────────────────────────────────
    "doc4_qa0__after4": (1.0,
        "Pred: '...January 10th...team meeting January 17th. Therefore, you attended "
        "the workshop 7 days before the team meeting.' EXACT: 7 days. Score 1.0."),
    "doc4_qa0__after5": (1.0,
        "Pred: '...January 10th...January 17th. Therefore, you attended the workshop "
        "7 days before the team meeting.' EXACT. Score 1.0."),
    "doc4_qa0__after6": (1.0,
        "Pred: '...January 10th...January 17th. That means you attended the workshop "
        "7 days before the team meeting.' EXACT. Score 1.0."),
    "doc4_qa0__after7": (1.0,
        "Pred: '...January 10th...January 17th. 7 days before the team meeting.' "
        "EXACT. Score 1.0."),
    "doc4_qa0__after8": (1.0,
        "Pred: '...January 10th...January 17th. 7 days before the team meeting.' "
        "EXACT. Score 1.0."),
    "doc4_qa0__after9": (1.0,
        "Pred: '...January 10th...January 17th. 7 days before the team meeting.' "
        "EXACT. Score 1.0."),
    # ── doc5_qa0: 30 days — correct at after6, after7 only ───────────────────
    "doc5_qa0__after6": (1.0,
        "Pred: 'The Sunday mass at St. Mary's Church was on January 2nd, and Ash "
        "Wednesday was on February 1st. Therefore, 30 days had passed between the "
        "two events.' EXACT: 30 days. Score 1.0."),
    "doc5_qa0__after7": (1.0,
        "Pred: 'The Sunday mass at St. Mary's Church was on January 2nd, and Ash "
        "Wednesday was on February 1st. Therefore, 30 days had passed between the "
        "two events.' EXACT: 30 days. Score 1.0."),
    # ── doc7_qa0: tomatoes first ──────────────────────────────────────────────
    "doc7_qa0__after7": (1.0,
        "Pred: 'The tomatoes were started first.' EXACT match. Score 1.0."),
    # ── doc8_qa0: 21 days ─────────────────────────────────────────────────────
    "doc8_qa0__after8": (1.0,
        "Pred: 'The Hindu festival of Holi was celebrated on February 26th, and the "
        "Sunday mass at St. Mary's Church was attended on March 19th. The number of "
        "days that passed between these two dates is 21 days.' EXACT. Score 1.0."),
    "doc8_qa0__after9": (1.0,
        "Pred: 'The Hindu festival of Holi was celebrated on February 26th, and the "
        "Sunday mass at St. Mary's Church was attended on March 19th. The number of "
        "days that passed between these two dates is 21 days.' EXACT. Score 1.0."),
}


# ---------------------------------------------------------------------------
# BATCH_CALIB JUDGMENTS: suffix → (score, rationale)
# suffix = qid minus "longmemeval__attention-corpus-tuned__batch__" and "__seed42"
# ---------------------------------------------------------------------------
BATCH_JUDGMENTS: dict[str, tuple[float, str]] = {
    "doc0_qa0": (1.0,
        "Pred: 'The first issue you had with your new car after its first service was "
        "with the GPS system, which you had to take back to the dealership to get "
        "fixed.' EXACT: GPS system not functioning correctly. Score 1.0."),
    "doc3_qa0": (1.0,
        "Pred: 'You got the Samsung Galaxy S22 first on February 20th, before "
        "receiving the Dell XPS 13 on February 25th.' EXACT: S22 first. Score 1.0."),
    "doc4_qa0": (1.0,
        "Pred: '...January 10th...January 17th. That means you attended the workshop "
        "7 days before the team meeting.' EXACT: 7 days. Score 1.0."),
    "doc8_qa0": (1.0,
        "Pred: 'The Hindu festival of Holi was celebrated on February 26th, and the "
        "Sunday mass at St. Mary's Church was attended on March 19th. The number of "
        "days that passed between these two dates is 21 days.' EXACT: 21 days. "
        "Score 1.0."),
}


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def score_calib(r: dict) -> tuple[float, str]:
    pred = r.get("predicted", "")
    behavior = r.get("expected_behavior", "answer")
    cfg = "attention-corpus-tuned"
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
    cfg = "attention-corpus-tuned"
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
    print("Judging longmemeval__attention-corpus-tuned — calibration + batch_calib ...")
    write_judgments(CALIB_DIR / "queue.jsonl", CALIB_RESULTS, score_calib, "calib")
    write_judgments(BATCH_DIR  / "queue.jsonl", BATCH_RESULTS,  score_batch, "batch")
    print("Done.")
