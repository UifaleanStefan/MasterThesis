"""Phase 1.9 — HotpotQA dump-all Protocol B calibration + batch_calib judge.

Cells:
  hotpotqa__dump-all__calibration__seed42   (100 entries)
  hotpotqa__dump-all__batch_calib__seed42   (10 entries)

Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  Protocol B rubric

Benchmark: HotpotQA (multi-hop factual Wikipedia QA)
Structure: 10 docs × 10 questions = 100 calib entries + 10 batch_calib entries
  expected_answer=55, expected_acknowledge_missing=45

Key calibration findings (dump-all vs attention-corpus-tuned):
  - MUCH more noise: many answer-time points give refusals (dump-all dumps everything,
    causing context saturation and confusion).
  - doc0 (same nationality? gold=yes):
      after0,1,6,7,9 → "No, ...but both American" → contradictory → 0.25
      after2,3,4,5,8 → refusals → 0.0
  - doc1 (govt position? gold=Chief of Protocol):
      after1 → "Shirley Temple...ambassador to Ghana and Czechoslovakia" (NO Chief of
      Protocol mentioned) → 0.25 (loosely related, wrong specific position)
      after2-9 → all refusals → 0.0
  - doc2 (YA series? gold=Animorphs):
      after2,3 → "Animorphs" → 1.0
      after4 → refusal → 0.0
      after5 → "The Illuminae Files" → 0.0 (wrong)
      after6 → "The Broken Earth" → 0.0 (wrong)
      after7 → "The Expanse" → 0.0 (wrong)
      after8,9 → refusals → 0.0
  - doc3 (same neighborhood? gold=no):
      after3 → "No, Laleli Mosque in Laleli, Fatih..." → 1.0
      after4-9 → all refusals → 0.0
  - doc4 (NYC neighborhood? gold=Greenwich Village, NYC):
      after4 → "Greenwich Village" → 1.0
      after5-9 → all refusals → 0.0
  - doc5 (formed by? gold=YG Entertainment):
      after5 → "WINNER...formed by YG Entertainment" → 1.0
      after6 → "Pledis Entertainment" → 0.0 (wrong)
      after7-9 → refusals → 0.0
  - doc6 (stage name Aladin? gold=Eenasul Fateh):
      after6 → "Eenasul Fateh" → 1.0
      after7-9 → refusals → 0.0
  - doc7 (arena capacity? gold=3,677 seated):
      after7 → "capacity of 4,000 (3,677 seated)" → 1.0
      after8 → "Robert and Concetta Dwyer Arena...1,400" → 0.0 (wrong arena)
      after9 → refusal → 0.0
  - doc8 (who older? gold=Terry Richardson):
      after8,9 → "Terry Richardson is older" → 1.0
  - doc9 (both from US? gold=yes):
      after9 → "Yes." → 1.0
      after7 → acknowledge_missing + non-refusal → 0.0 (auto-handled hallucination)

Batch_calib: 2.25/10 scored (doc8=1.0, doc9=1.0, doc0=0.25; docs 1-7 all refusals=0.0)

Scoring rules (Protocol B):
  - acknowledge_missing: 1.0 honest refusal, 0.0 confident answer (hallucination)
  - answer: standard 5-point rubric on non-refusal predictions
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JQ = ROOT / "results" / "stage3" / "judge_queue"
CALIB_DIR = JQ / "hotpotqa__dump-all__calibration__seed42"
BATCH_DIR  = JQ / "hotpotqa__dump-all__batch_calib__seed42"

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
# Only non-zero answer+non-refusal entries included.
# Many answer entries get refusal (score 0.0 by default) — not listed here.
# Notable: dump-all causes severe context saturation; most time points refusal.
# ---------------------------------------------------------------------------
CALIB_JUDGMENTS: dict[str, tuple[float, str]] = {
    # ── doc0_qa0: same nationality? gold=yes ─────────────────────────────────
    "doc0_qa0__after0": (0.25,
        "Pred: 'No, Scott Derrickson is American, while Ed Wood was also American.' "
        "Self-contradictory: opens with 'No' (wrong verdict for yes/no question) "
        "but body correctly identifies both as American. Gold=yes. Score 0.25."),
    "doc0_qa0__after1": (0.25,
        "Pred: 'No, Scott Derrickson is American, while Ed Wood was also American.' "
        "Same contradiction — wrong verdict, correct facts. Gold=yes. Score 0.25."),
    "doc0_qa0__after6": (0.25,
        "Pred: 'No, Scott Derrickson is American and Ed Wood was also American.' "
        "Same contradiction. Score 0.25."),
    "doc0_qa0__after7": (0.25,
        "Pred: 'No, Scott Derrickson is American, while Ed Wood was also American.' "
        "Same contradiction. Score 0.25."),
    "doc0_qa0__after9": (0.25,
        "Pred: 'No, Scott Derrickson is American, while Ed Wood was also American.' "
        "Same contradiction. Score 0.25."),
    # ── doc1_qa0: government position? gold=Chief of Protocol ────────────────
    "doc1_qa0__after1": (0.25,
        "Pred: 'Shirley Temple served as United States ambassador to Ghana and to "
        "Czechoslovakia.' Mentions Shirley Temple and a government position (ambassador) "
        "but does NOT name Chief of Protocol. Ambassador is a related but different "
        "government position. Score 0.25."),
    # ── doc2_qa0: science fantasy YA series? gold=Animorphs ──────────────────
    "doc2_qa0__after2": (1.0,
        "Pred: 'Animorphs' EXACT match. Score 1.0."),
    "doc2_qa0__after3": (1.0,
        "Pred: 'Animorphs' EXACT match. Score 1.0."),
    # ── doc3_qa0: same neighborhood? gold=no ─────────────────────────────────
    "doc3_qa0__after3": (1.0,
        "Pred: 'No, the Laleli Mosque is located in Laleli, Fatih, Istanbul, "
        "while the Esma Sultan Mansion is located...' EXACT: gold=no, correct "
        "reasoning provided. Score 1.0."),
    # ── doc4_qa0: NYC neighborhood? gold=Greenwich Village, New York City ─────
    "doc4_qa0__after4": (1.0,
        "Pred: 'Greenwich Village' — core answer present, gold=Greenwich Village, "
        "New York City. Score 1.0."),
    # ── doc5_qa0: formed by? gold=YG Entertainment ───────────────────────────
    "doc5_qa0__after5": (1.0,
        "Pred: '...South Korean group WINNER, which was formed by YG Entertainment.' "
        "EXACT: gold=YG Entertainment present. Score 1.0."),
    # ── doc6_qa0: stage name Aladin? gold=Eenasul Fateh ──────────────────────
    "doc6_qa0__after6": (1.0,
        "Pred: 'Eenasul Fateh' EXACT match. Score 1.0."),
    # ── doc7_qa0: arena capacity? gold=3,677 seated ──────────────────────────
    "doc7_qa0__after7": (1.0,
        "Pred: 'The Androscoggin Bank Colisée...has a capacity of 4,000 "
        "(3,677 seated).' EXACT: gold=3,677 seated, prediction explicitly contains "
        "'3,677 seated'. Score 1.0."),
    # ── doc8_qa0: who is older? gold=Terry Richardson ────────────────────────
    "doc8_qa0__after8": (1.0,
        "Pred: 'Terry Richardson is older.' EXACT match. Score 1.0."),
    "doc8_qa0__after9": (1.0,
        "Pred: 'Terry Richardson is older.' EXACT match. Score 1.0."),
    # ── doc9_qa0: both from US? gold=yes ─────────────────────────────────────
    "doc9_qa0__after9": (1.0,
        "Pred: 'Yes.' EXACT: gold=yes. Score 1.0."),
}


# ---------------------------------------------------------------------------
# BATCH_CALIB JUDGMENTS: suffix → (score, rationale)
# Only 3 entries non-zero: doc0=0.25, doc8=1.0, doc9=1.0
# docs 1-7 all refusals (handled by default is_refusal path → 0.0)
# ---------------------------------------------------------------------------
BATCH_JUDGMENTS: dict[str, tuple[float, str]] = {
    "doc0_qa0": (0.25,
        "Pred: 'No, Scott Derrickson is American, while Ed Wood was American as "
        "well, but they are from different eras and contexts in film.' Says 'No' "
        "for a nationality question, then confirms both are American (which means "
        "same nationality = yes). Confuses era with nationality. Wrong final verdict "
        "but correct underlying facts. Score 0.25."),
    "doc8_qa0": (1.0,
        "Pred: 'Terry Richardson is older.' EXACT: gold=Terry Richardson. Score 1.0."),
    "doc9_qa0": (1.0,
        "Pred: 'Yes.' EXACT: gold=yes, both Local H and For Against from the US. "
        "Score 1.0."),
}


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def score_calib(r: dict) -> tuple[float, str]:
    pred = r.get("predicted", "")
    behavior = r.get("expected_behavior", "answer")
    cfg = "dump-all"
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
    cfg = "dump-all"
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
    print("Judging hotpotqa__dump-all — calibration + batch_calib ...")
    write_judgments(CALIB_DIR / "queue.jsonl", CALIB_RESULTS, score_calib, "calib")
    write_judgments(BATCH_DIR  / "queue.jsonl", BATCH_RESULTS,  score_batch, "batch")
    print("Done.")
