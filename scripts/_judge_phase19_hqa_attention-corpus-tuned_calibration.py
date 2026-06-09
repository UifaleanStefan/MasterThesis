"""Phase 1.9 — HotpotQA attention-corpus-tuned Protocol B calibration + batch_calib judge.

Cells:
  hotpotqa__attention-corpus-tuned__calibration__seed42   (100 entries)
  hotpotqa__attention-corpus-tuned__batch_calib__seed42   (10 entries)

Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  Protocol B rubric

Benchmark: HotpotQA (multi-hop factual Wikipedia QA)
Structure: 10 docs × 10 questions = 100 calib entries + 10 batch_calib entries
  expected_answer=55, expected_acknowledge_missing=45

Key calibration findings:
  - doc0 (Scott Derrickson + Ed Wood same nationality? gold=yes):
      after0-3: YES, both American → 1.0
      after4: "No, ...Ed Wood was also American" → contradictory — says No but both American → 0.25
      after5: YES → 1.0
      after6-8: "No, ...both American" → contradictory → 0.25
      after9: YES → 1.0
  - doc1 (Corliss Archer actress govt position? gold=Chief of Protocol):
      All after1-after9 explicitly state "also served as Chief of Protocol of the United States" → 1.0
  - doc2 (science fantasy YA series? gold=Animorphs): all after2-after9 → "Animorphs" → 1.0
  - doc3 (Laleli Mosque + Esma Sultan same neighborhood? gold=no):
      all after3-after9 → "No, Laleli Mosque in Laleli, Fatih…" → 1.0
  - doc4 (Big Stone Gap director New York neighborhood? gold=Greenwich Village, NYC):
      all after4-after9 → "Greenwich Village" or "Greenwich Village, NYC" → 1.0
  - doc5 (2014 S/S debut album formed by? gold=YG Entertainment):
      all after5-after9 → "formed by YG Entertainment" → 1.0
  - doc6 (stage name Aladin performance consultant? gold=Eenasul Fateh):
      all after6-after9 → "Eenasul Fateh" → 1.0
  - doc7 (Lewiston Maineiacs arena capacity? gold=3,677 seated):
      all after7-after9 → "capacity of 4,000 (3,677 seated)" → 1.0
  - doc8 (Annie Morton or Terry Richardson older? gold=Terry Richardson):
      after8-after9 → "Terry Richardson is older" → 1.0
  - doc9 (Local H and For Against both US? gold=yes):
      after9 → "Yes, both...from the United States" → 1.0
      after7, after8 → acknowledge_missing + non-refusal → 0.0 (hallucination, auto-handled)

Batch_calib: 10/10 correct → all 1.0 (perfect recall)
  doc0=yes, doc1=Chief of Protocol, doc2=Animorphs, doc3=no, doc4=Greenwich Village,
  doc5=YG Entertainment, doc6=Eenasul Fateh, doc7=3677, doc8=Terry Richardson, doc9=yes

Scoring rules (Protocol B):
  - acknowledge_missing: 1.0 honest refusal, 0.0 confident answer (hallucination)
  - answer: standard 5-point rubric on non-refusal predictions
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JQ = ROOT / "results" / "stage3" / "judge_queue"
CALIB_DIR = JQ / "hotpotqa__attention-corpus-tuned__calibration__seed42"
BATCH_DIR  = JQ / "hotpotqa__attention-corpus-tuned__batch_calib__seed42"

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
# All 55 answer entries have non-zero scores — all included here.
# ---------------------------------------------------------------------------
CALIB_JUDGMENTS: dict[str, tuple[float, str]] = {
    # ── doc0_qa0: same nationality? gold=yes ─────────────────────────────────
    "doc0_qa0__after0": (1.0,
        "Pred: 'Yes, both Scott Derrickson and Ed Wood are American.' EXACT: "
        "gold=yes, both confirmed same nationality (American). Score 1.0."),
    "doc0_qa0__after1": (1.0,
        "Pred: 'Yes, both Scott Derrickson and Ed Wood were American.' EXACT. "
        "Score 1.0."),
    "doc0_qa0__after2": (1.0,
        "Pred: 'Yes, both Scott Derrickson and Ed Wood were American.' EXACT. "
        "Score 1.0."),
    "doc0_qa0__after3": (1.0,
        "Pred: 'Yes, both Scott Derrickson and Ed Wood were American.' EXACT. "
        "Score 1.0."),
    "doc0_qa0__after4": (0.25,
        "Pred: 'No, Scott Derrickson is American, and Ed Wood was also American.' "
        "Self-contradictory: opens with 'No' (wrong final verdict) but correctly "
        "identifies both as American (which implies yes). Gold=yes. Wrong answer "
        "verdict despite correct underlying facts. Score 0.25."),
    "doc0_qa0__after5": (1.0,
        "Pred: 'Yes, both Scott Derrickson and Ed Wood were American.' EXACT. "
        "Score 1.0."),
    "doc0_qa0__after6": (0.25,
        "Pred: 'No, Scott Derrickson is American, while Ed Wood was also American.' "
        "Self-contradictory: says 'No' (wrong verdict) but body confirms both "
        "American. Gold=yes. Score 0.25."),
    "doc0_qa0__after7": (0.25,
        "Pred: 'No, Scott Derrickson is American, while Ed Wood was also American.' "
        "Same contradiction as after6 — wrong yes/no verdict, correct facts. "
        "Score 0.25."),
    "doc0_qa0__after8": (0.25,
        "Pred: 'No, Scott Derrickson is American, while Ed Wood was also American.' "
        "Same contradiction — says No but confirms both American. Gold=yes. "
        "Score 0.25."),
    "doc0_qa0__after9": (1.0,
        "Pred: 'Yes, both Scott Derrickson and Ed Wood were American.' EXACT. "
        "Score 1.0."),
    # ── doc1_qa0: government position? gold=Chief of Protocol ────────────────
    "doc1_qa0__after1": (1.0,
        "Pred: 'Shirley Temple served as United States ambassador to Ghana and "
        "to Czechoslovakia, and also as Chief of Protocol of the United States.' "
        "EXACT: gold=Chief of Protocol, prediction explicitly names this position. "
        "Score 1.0."),
    "doc1_qa0__after2": (1.0,
        "Pred: '...Shirley Temple...held...United States ambassador...and also as "
        "Chief of Protocol...' EXACT: Chief of Protocol present. Score 1.0."),
    "doc1_qa0__after3": (1.0,
        "Pred: '...Shirley Temple held...ambassador...and also served as Chief of "
        "Protocol...' EXACT: Chief of Protocol present. Score 1.0."),
    "doc1_qa0__after4": (1.0,
        "Pred: '...Shirley Temple served as...ambassador...and also as Chief of "
        "Protocol of the United States.' EXACT: Chief of Protocol present. "
        "Score 1.0."),
    "doc1_qa0__after5": (1.0,
        "Pred: '...served as...ambassador...and also as Chief of Protocol of the "
        "United States.' EXACT. Score 1.0."),
    "doc1_qa0__after6": (1.0,
        "Pred: '...Shirley Temple held...ambassador...and also served as Chief of "
        "Protocol...' EXACT. Score 1.0."),
    "doc1_qa0__after7": (1.0,
        "Pred: '...Shirley Temple held...ambassador...and also served as Chief of "
        "Protocol...' EXACT. Score 1.0."),
    "doc1_qa0__after8": (1.0,
        "Pred: '...Shirley Temple held...ambassador...and also served as Chief of "
        "Protocol...' EXACT. Score 1.0."),
    "doc1_qa0__after9": (1.0,
        "Pred: '...Shirley Temple held...ambassador...and also served as Chief of "
        "Protocol...' EXACT. Score 1.0."),
    # ── doc2_qa0: science fantasy YA series? gold=Animorphs ──────────────────
    "doc2_qa0__after2": (1.0,
        "Pred: 'Animorphs' EXACT match. Score 1.0."),
    "doc2_qa0__after3": (1.0,
        "Pred: 'Animorphs' EXACT match. Score 1.0."),
    "doc2_qa0__after4": (1.0,
        "Pred: 'Animorphs' EXACT match. Score 1.0."),
    "doc2_qa0__after5": (1.0,
        "Pred: 'Animorphs' EXACT match. Score 1.0."),
    "doc2_qa0__after6": (1.0,
        "Pred: 'Animorphs' EXACT match. Score 1.0."),
    "doc2_qa0__after7": (1.0,
        "Pred: 'Animorphs' EXACT match. Score 1.0."),
    "doc2_qa0__after8": (1.0,
        "Pred: 'Animorphs' EXACT match. Score 1.0."),
    "doc2_qa0__after9": (1.0,
        "Pred: 'Animorphs' EXACT match. Score 1.0."),
    # ── doc3_qa0: same neighborhood? gold=no ─────────────────────────────────
    "doc3_qa0__after3": (1.0,
        "Pred: 'No, the Laleli Mosque is located in Laleli, Fatih, Istanbul, "
        "while the Esma Sultan Mansion...' EXACT: gold=no, correct reasoning "
        "provided. Score 1.0."),
    "doc3_qa0__after4": (1.0,
        "Pred: 'No, the Laleli Mosque is located in Laleli, Fatih, while the "
        "Esma Sultan Mansion...' EXACT: gold=no. Score 1.0."),
    "doc3_qa0__after5": (1.0,
        "Pred: 'No, the Laleli Mosque is located in Laleli, Fatih, Istanbul, "
        "while the Esma Sultan Mansion...' EXACT. Score 1.0."),
    "doc3_qa0__after6": (1.0,
        "Pred: 'No, the Laleli Mosque is located in Laleli, Fatih, Istanbul, "
        "while the Esma Sultan Mansion...' EXACT. Score 1.0."),
    "doc3_qa0__after7": (1.0,
        "Pred: 'No, the Laleli Mosque is located in Laleli, Fatih, Istanbul, "
        "while the Esma Sultan Mansion...' EXACT. Score 1.0."),
    "doc3_qa0__after8": (1.0,
        "Pred: 'No, the Laleli Mosque is located in Laleli, Fatih, Istanbul, "
        "while the Esma Sultan Mansion...' EXACT. Score 1.0."),
    "doc3_qa0__after9": (1.0,
        "Pred: 'No, the Laleli Mosque is located in Laleli, Fatih, while the "
        "Esma Sultan Mansion...' EXACT: gold=no. Score 1.0."),
    # ── doc4_qa0: NYC neighborhood? gold=Greenwich Village, New York City ─────
    "doc4_qa0__after4": (1.0,
        "Pred: 'Greenwich Village' — core answer present, New York City context "
        "implied. Gold=Greenwich Village, New York City. Score 1.0."),
    "doc4_qa0__after5": (1.0,
        "Pred: 'Greenwich Village, New York City.' EXACT. Score 1.0."),
    "doc4_qa0__after6": (1.0,
        "Pred: 'Greenwich Village' — core answer present. Score 1.0."),
    "doc4_qa0__after7": (1.0,
        "Pred: 'Greenwich Village' — core answer present. Score 1.0."),
    "doc4_qa0__after8": (1.0,
        "Pred: 'Greenwich Village' — core answer present. Score 1.0."),
    "doc4_qa0__after9": (1.0,
        "Pred: 'Greenwich Village' — core answer present. Score 1.0."),
    # ── doc5_qa0: formed by? gold=YG Entertainment ───────────────────────────
    "doc5_qa0__after5": (1.0,
        "Pred: '...WINNER, which was formed by YG Entertainment.' EXACT: "
        "gold=YG Entertainment present. Score 1.0."),
    "doc5_qa0__after6": (1.0,
        "Pred: '...WINNER...formed by YG Entertainment.' EXACT. Score 1.0."),
    "doc5_qa0__after7": (1.0,
        "Pred: '...WINNER...formed by YG Entertainment.' EXACT. Score 1.0."),
    "doc5_qa0__after8": (1.0,
        "Pred: '...WINNER...formed by YG Entertainment.' EXACT. Score 1.0."),
    "doc5_qa0__after9": (1.0,
        "Pred: '...WINNER...formed by YG Entertainment.' EXACT. Score 1.0."),
    # ── doc6_qa0: stage name Aladin? gold=Eenasul Fateh ──────────────────────
    "doc6_qa0__after6": (1.0,
        "Pred: 'Eenasul Fateh' EXACT match. Score 1.0."),
    "doc6_qa0__after7": (1.0,
        "Pred: 'Eenasul Fateh' EXACT match. Score 1.0."),
    "doc6_qa0__after8": (1.0,
        "Pred: 'Eenasul Fateh' EXACT match. Score 1.0."),
    "doc6_qa0__after9": (1.0,
        "Pred: 'Eenasul Fateh' EXACT match. Score 1.0."),
    # ── doc7_qa0: arena capacity? gold=3,677 seated ──────────────────────────
    "doc7_qa0__after7": (1.0,
        "Pred: 'The Androscoggin Bank Colisée...has a capacity of 4,000 "
        "(3,677 seated).' EXACT: gold=3,677 seated, prediction contains '3,677 "
        "seated'. The 4,000 total is the standing capacity context. Score 1.0."),
    "doc7_qa0__after8": (1.0,
        "Pred: '...has a capacity of 4,000 (3,677 seated).' EXACT: 3,677 seated "
        "present. Score 1.0."),
    "doc7_qa0__after9": (1.0,
        "Pred: '...has a capacity of 4,000 (3,677 seated).' EXACT: 3,677 seated "
        "present. Score 1.0."),
    # ── doc8_qa0: who is older? gold=Terry Richardson ────────────────────────
    "doc8_qa0__after8": (1.0,
        "Pred: 'Terry Richardson is older.' EXACT match. Score 1.0."),
    "doc8_qa0__after9": (1.0,
        "Pred: 'Terry Richardson is older.' EXACT match. Score 1.0."),
    # ── doc9_qa0: both from US? gold=yes ─────────────────────────────────────
    "doc9_qa0__after9": (1.0,
        "Pred: 'Yes, both Local H and For Against are from the United States.' "
        "EXACT: gold=yes. Score 1.0."),
}


# ---------------------------------------------------------------------------
# BATCH_CALIB JUDGMENTS: suffix → (score, rationale)
# All 10 entries correct → perfect batch recall (10/10)
# ---------------------------------------------------------------------------
BATCH_JUDGMENTS: dict[str, tuple[float, str]] = {
    "doc0_qa0": (1.0,
        "Pred: 'Yes, both Scott Derrickson and Ed Wood were American.' "
        "EXACT: gold=yes, same nationality confirmed. Score 1.0."),
    "doc1_qa0": (1.0,
        "Pred: 'Shirley Temple held the position of United States ambassador "
        "to Ghana and to Czechoslovakia, and also served as Chief of Protocol "
        "of the United States.' EXACT: gold=Chief of Protocol, explicitly "
        "named. Score 1.0."),
    "doc2_qa0": (1.0,
        "Pred: 'Animorphs' EXACT match. Score 1.0."),
    "doc3_qa0": (1.0,
        "Pred: 'No, the Laleli Mosque is located in Laleli, Fatih, Istanbul, "
        "while the Esma Sultan Mansion...' EXACT: gold=no. Score 1.0."),
    "doc4_qa0": (1.0,
        "Pred: 'Greenwich Village' — core answer, gold=Greenwich Village, "
        "New York City. Score 1.0."),
    "doc5_qa0": (1.0,
        "Pred: '...WINNER, which was formed by YG Entertainment.' "
        "EXACT: gold=YG Entertainment. Score 1.0."),
    "doc6_qa0": (1.0,
        "Pred: 'Eenasul Fateh' EXACT match. Score 1.0."),
    "doc7_qa0": (1.0,
        "Pred: '...can seat 3,677...' EXACT: gold=3,677 seated. Score 1.0."),
    "doc8_qa0": (1.0,
        "Pred: 'Terry Richardson is older.' EXACT match. Score 1.0."),
    "doc9_qa0": (1.0,
        "Pred: 'Yes, both Local H and For Against are from the United States.' "
        "EXACT: gold=yes. Score 1.0."),
}


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def score_calib(r: dict) -> tuple[float, str]:
    pred = r.get("predicted", "")
    behavior = r.get("expected_behavior", "answer")
    cfg = "attention-corpus-tuned"
    suffix = r["qid"].replace(f"hotpotqa__{cfg}__calibration__", "").replace("__seed42", "")

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
    suffix = r["qid"].replace(f"hotpotqa__{cfg}__batch__", "").replace("__seed42", "")

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
    print("Judging hotpotqa__attention-corpus-tuned — calibration + batch_calib ...")
    write_judgments(CALIB_DIR / "queue.jsonl", CALIB_RESULTS, score_calib, "calib")
    write_judgments(BATCH_DIR  / "queue.jsonl", BATCH_RESULTS,  score_batch, "batch")
    print("Done.")
