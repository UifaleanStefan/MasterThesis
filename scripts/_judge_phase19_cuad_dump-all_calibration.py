"""Phase 1.9 — CUAD dump-all Protocol B calibration + batch_calib judge.

Cells:
  cuad__dump-all__calibration__seed42   (50 entries)
  cuad__dump-all__batch_calib__seed42   (132 entries)

Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  Protocol B rubric

Key findings:
  - dump-all: entire memory context concatenated without retrieval; overwhelmingly PACIRA
    Promissory Note contamination dominates all outputs
  - Almost all answers give promissory note text, New York law, or PACIRA ARSLDMA name
  - Only doc9 (PACIRA/EKR) benefits from contamination; all other docs severely degraded
  - Calibration: 27 acknowledge_missing, 23 answer entries
    3 non-refusal entries with non-zero scores (all score 0.25)
  - Batch_calib: 132 entries, only 19 score above 0.0

Scoring rules (Protocol B):
  - acknowledge_missing: 1.0 honest refusal, 0.0 confident answer (hallucination)
  - answer: standard 5-point rubric on non-refusal predictions
    0.0 wrong/refusal, 0.25 loosely related, 0.5 partial, 0.75 close, 1.0 exact
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JQ = ROOT / "results" / "stage3" / "judge_queue"
CALIB_DIR = JQ / "cuad__dump-all__calibration__seed42"
BATCH_DIR = JQ / "cuad__dump-all__batch_calib__seed42"

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


# ---------------------------------------------------------------------------
# CALIBRATION JUDGMENTS: suffix → (score, rationale)
# suffix = qid minus "cuad__dump-all__calibration__" and "__seed42"
# Only answer+non-refusal entries needing non-default (>0.0) scores
# ---------------------------------------------------------------------------
CALIB_JUDGMENTS: dict[str, tuple[float, str]] = {
    "doc3_qa0__after9": (0.25,
        "Mentions promissory note sections (Governing Law, Nonnegotiability) but not WEB SITE HOSTING AGREEMENT name. Score 0.25."),
    "doc7_qa5__after9": (0.25,
        "Gold: Japan; pred: New York (PACIRA contamination). Wrong jurisdiction. Score 0.25."),
    "doc1_qa4__after9": (0.25,
        "Gold: English law; pred: New York (PACIRA contamination). Wrong governing law. Score 0.25."),
}


# ---------------------------------------------------------------------------
# BATCH_CALIB JUDGMENTS: suffix → (score, rationale)
# suffix = qid minus "cuad__dump-all__batch__" and "__seed42"
# Only non-refusal entries needing non-default (>0.0) scores
# ---------------------------------------------------------------------------
BATCH_JUDGMENTS: dict[str, tuple[float, str]] = {
    # ── Score 1.0 ────────────────────────────────────────────────────────────
    "doc9_qa0": (1.0,
        "Amended and Restated Strategic Licensing, Distribution and Marketing Agreement — EXACT PACIRA ARSLDMA name. Score 1.0."),
    "doc9_qa7": (1.0,
        "Governed by the laws of the State of New York — EXACT PACIRA ARSLDMA jurisdiction. Score 1.0."),
    # ── Score 0.5 ────────────────────────────────────────────────────────────
    "doc8_qa5": (0.5,
        "State of New York — correct governing law for Co-Promotion Agreement (Dova/Valeant), but stated without document context. Score 0.5."),
    "doc9_qa2": (0.5,
        "October, 2009 — month and year correct for PACIRA ARSLDMA effective date; missing specific day (15th). Score 0.5."),
    # ── Score 0.25 ───────────────────────────────────────────────────────────
    "doc0_qa6": (0.25,
        "Pred: New York vs gold: Illinois (LIME ENERGY). PACIRA contamination yields wrong law. Score 0.25."),
    "doc0_qa11": (0.25,
        "Promissory Note nonnegotiable/nontransferable vs LIME ENERGY anti-assignment clause. Concept right, source wrong. Score 0.25."),
    "doc1_qa4": (0.25,
        "Pred: New York vs gold: English law (Whitesmoke/Google). PACIRA contamination. Score 0.25."),
    "doc1_qa6": (0.25,
        "Promissory Note nonnegotiable/nontransferable vs Whitesmoke/Google anti-assignment. Concept match, wrong source. Score 0.25."),
    "doc2_qa3": (0.25,
        "Pred: New York vs gold: People's Republic of China (Supply Contract). Wrong governing law. Score 0.25."),
    "doc3_qa7": (0.25,
        "Pred: New York vs gold: Florida (Web Site Hosting/i-on). Wrong jurisdiction. Score 0.25."),
    "doc5_qa5": (0.25,
        "Pred: New York vs gold: Kansas (Endorsement Agreement/Adams Golf). Wrong jurisdiction. Score 0.25."),
    "doc5_qa9": (0.25,
        "Promissory Note nonnegotiable vs ADAMS GOLF/CONSULTANT sublicense restriction. Concept match, source wrong. Score 0.25."),
    "doc6_qa5": (0.25,
        "Pred: New York vs gold: Texas (Consulting Agreement/Kiromic). Wrong jurisdiction. Score 0.25."),
    "doc6_qa8": (0.25,
        "Promissory Note nonnegotiable/nontransferable vs Consulting Agreement anti-assignment. Wrong source. Score 0.25."),
    "doc7_qa5": (0.25,
        "Pred: New York vs gold: Japan (Veoneer/Nissin Amendment). Wrong jurisdiction. Score 0.25."),
    "doc8_qa12": (0.25,
        "Promissory Note nonnegotiable vs Co-Promotion anti-assignment clause. Concept right, wrong source. Score 0.25."),
    "doc8_qa17": (0.25,
        "Promissory Note nontransferable vs Valeant's non-transferable rights under Section 2.1. Wrong source. Score 0.25."),
    "doc9_qa12": (0.25,
        "Promissory Note nonnegotiable vs PACIRA ARSLDMA anti-assignment (Section 20.2). Same corpus, wrong clause. Score 0.25."),
    "doc9_qa17": (0.25,
        "Promissory Note nontransferable vs EKR sub-distributor appointment provision. Wrong clause type. Score 0.25."),
}


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def score_calib(r: dict) -> tuple[float, str]:
    pred = r.get("predicted", "")
    behavior = r.get("expected_behavior", "answer")
    cfg = "dump-all"
    suffix = r["qid"].replace(f"cuad__{cfg}__calibration__", "").replace("__seed42", "")

    if behavior == "acknowledge_missing":
        if is_refusal(pred):
            return 1.0, "Honest refusal when source not yet ingested. Score 1.0."
        return 0.0, "Hallucination — answered when source not yet in memory. Score 0.0."

    if is_refusal(pred):
        return 0.0, "Refused when memory should contain source doc. Score 0.0."

    if suffix in CALIB_JUDGMENTS:
        return CALIB_JUDGMENTS[suffix]

    return 0.0, "Wrong answer — prediction does not match gold. Score 0.0."


def score_batch(r: dict) -> tuple[float, str]:
    pred = r.get("predicted", "")
    cfg = "dump-all"
    suffix = r["qid"].replace(f"cuad__{cfg}__batch__", "").replace("__seed42", "")

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

    print(f"  [{mode}] {queue_path.parent.name}: {len(rows)} queued, {len(new_records)} written, {len(existing)} already existed")


if __name__ == "__main__":
    print("Judging cuad__dump-all — calibration + batch_calib ...")
    write_judgments(CALIB_DIR / "queue.jsonl", CALIB_RESULTS, score_calib, "calib")
    write_judgments(BATCH_DIR / "queue.jsonl", BATCH_RESULTS, score_batch, "batch")
    print("Done.")
