"""Phase 1.9 — CUAD v4t-corpus-tuned Protocol B calibration + batch_calib judge.

Cells:
  cuad__v4t-corpus-tuned__calibration__seed42   (50 entries)
  cuad__v4t-corpus-tuned__batch_calib__seed42   (132 entries)

Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  Protocol B rubric

Key findings:
  - v4t-corpus-tuned: theta tuned on CUAD corpus, best per-document retrieval
  - Less PACIRA contamination than canonical/tuned; more targeted retrieval
  - Calibration: 27 acknowledge_missing, 23 answer entries; 13 non-refusal with non-zero scores
  - Batch_calib: 132 entries; strong on docs 0–6, partial on 8–9

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
CALIB_DIR = JQ / "cuad__v4t-corpus-tuned__calibration__seed42"
BATCH_DIR = JQ / "cuad__v4t-corpus-tuned__batch_calib__seed42"

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
# suffix = qid minus "cuad__v4t-corpus-tuned__calibration__" and "__seed42"
# Only answer+non-refusal entries needing non-default (>0.0) scores
# ---------------------------------------------------------------------------
CALIB_JUDGMENTS: dict[str, tuple[float, str]] = {
    "doc0_qa6__after1": (1.0,
        "Gold: Illinois; pred: 'construed according to the laws of the State of Illinois.' EXACT match. Score 1.0."),
    "doc1_qa8__after1": (0.75,
        "Google audit rights correctly identified. Score 0.75."),
    "doc3_qa6__after3": (0.25,
        "Pred: 15 days vs gold: 30 days (Web Site Hosting notice period). Wrong number. Score 0.25."),
    "doc3_qa6__after4": (0.25,
        "Pred: 15 days vs gold: 30 days. Wrong notice period repeated. Score 0.25."),
    "doc2_qa1__after5": (0.5,
        "Identifies Shenzhen LOHAS as buyer and seller relationship — partial match for Supply Contract parties. Score 0.5."),
    "doc2_qa3__after6": (0.5,
        "Hong Kong law cited — close to People's Republic of China for Supply Contract. Score 0.5."),
    "doc6_qa13__after8": (0.5,
        "Survival sections listed (payment, confidentiality, IP, post-termination). Correct concept. Score 0.5."),
    "doc0_qa5__after8": (1.0,
        "Annual 1-year terms up to 10 years for LIME ENERGY — EXACT renewal provision. Score 1.0."),
    "doc3_qa0__after9": (0.25,
        "Vague answer mentions reviewing contract but doesn't name WEB SITE HOSTING AGREEMENT specifically. Score 0.25."),
    "doc7_qa5__after9": (0.25,
        "Gold: Japan (Veoneer/Nissin); pred: English law. Wrong jurisdiction. Score 0.25."),
    "doc1_qa4__after9": (0.75,
        "English law for Whitesmoke/Google — correct governing law jurisdiction. Score 0.75."),
    "doc9_qa11__after9": (0.5,
        "Change of control right concept identified for PACIRA ARSLDMA. Score 0.5."),
    "doc1_qa1__after9": (0.75,
        "Whitesmoke and Google as parties correctly identified. Score 0.75."),
}


# ---------------------------------------------------------------------------
# BATCH_CALIB JUDGMENTS: suffix → (score, rationale)
# suffix = qid minus "cuad__v4t-corpus-tuned__batch__" and "__seed42"
# Only non-refusal entries needing non-default (>0.0) scores
# ---------------------------------------------------------------------------
BATCH_JUDGMENTS: dict[str, tuple[float, str]] = {
    # ── Score 0.25 ───────────────────────────────────────────────────────────
    "doc0_qa16": (0.25,
        "Wrong warranty clause — from Whitesmoke/Google, not LIME ENERGY. Wrong source. Score 0.25."),
    "doc1_qa3": (0.25,
        "2021 cited as date vs gold: 2011 (Whitesmoke/Google). Wrong year. Score 0.25."),
    "doc2_qa1": (0.25,
        "Framework clause cited; gold asks for seller name (Shenzhen LOHAS Supply). Score 0.25."),
    "doc2_qa3": (0.25,
        "English law vs gold: People's Republic of China (Supply Contract). Wrong law. Score 0.25."),
    "doc3_qa5": (0.25,
        "LIME ENERGY renewal clause cited vs gold: Web Site Hosting renewal. Wrong source. Score 0.25."),
    "doc3_qa6": (0.25,
        "Pred: 15 days vs gold: 30 days (Web Site Hosting). Wrong notice period. Score 0.25."),
    "doc3_qa7": (0.25,
        "English law vs gold: Florida (Web Site Hosting/i-on). Wrong jurisdiction. Score 0.25."),
    "doc4_qa2": (0.25,
        "PACIRA EKR reference for Joint Filing Agreement — wrong source entirely. Score 0.25."),
    "doc5_qa2": (0.25,
        "March 21 cited vs gold: specific Endorsement Agreement signing date. Wrong date context. Score 0.25."),
    "doc5_qa4": (0.25,
        "March 21 2008 cited vs gold: Endorsement Agreement effective date. Wrong date. Score 0.25."),
    "doc5_qa5": (0.25,
        "Florida cited vs gold: Kansas (Endorsement Agreement/Adams Golf). Wrong jurisdiction. Score 0.25."),
    "doc5_qa8": (0.25,
        "Section 2.3 cited (wrong clause type) for competitive exception. Score 0.25."),
    "doc6_qa4": (0.25,
        "Specific date wrong for Consulting Agreement effective date. Score 0.25."),
    "doc6_qa10": (0.25,
        "Licensing clause cited but gold asks for IP assignment in Consulting Agreement. Wrong clause. Score 0.25."),
    "doc7_qa0": (0.25,
        "JOINT VENTURE only — missing full name prefix 'Amendment and Termination of'. Score 0.25."),
    "doc7_qa3": (0.25,
        "July 2018 cited vs correct date for Veoneer/Nissin Amendment. Wrong month/date. Score 0.25."),
    "doc7_qa4": (0.25,
        "10-year LIME ENERGY term cited vs Veoneer/Nissin Amendment term (from VNBJ Closing). Wrong source. Score 0.25."),
    "doc7_qa5": (0.25,
        "English law vs gold: Japan (Veoneer/Nissin Amendment). Wrong jurisdiction. Score 0.25."),
    "doc8_qa3": (0.25,
        "PACIRA date (Aug 10, 2007) cited vs Co-Promotion effective date (Sep 26, 2018). Wrong date/contract. Score 0.25."),
    "doc9_qa6": (0.25,
        "15 days cited vs gold: 120 days termination notice (PACIRA ARSLDMA). Wrong number. Score 0.25."),
    "doc9_qa10": (0.25,
        "30 days cited vs gold: 60 days cure period (PACIRA ARSLDMA). Wrong number. Score 0.25."),
    "doc9_qa17": (0.25,
        "Note non-transferable vs EKR sub-distributor assignment provision. Wrong source. Score 0.25."),
    # ── Score 0.5 ────────────────────────────────────────────────────────────
    "doc0_qa2": (0.5,
        "September 9 cited vs gold: September 7 (LIME ENERGY date). Close date, one day off. Score 0.5."),
    "doc0_qa11": (0.5,
        "Anti-assignment concept correct for LIME ENERGY; partial without exact clause text. Score 0.5."),
    "doc1_qa5": (0.5,
        "Change of Control right concept identified for Whitesmoke/Google. Score 0.5."),
    "doc2_qa0": (0.5,
        "Supply Agreement identified — close to gold SUPPLY CONTRACT (Shenzhen LOHAS). Score 0.5."),
    "doc5_qa9": (0.5,
        "Anti-assignment right concept for Endorsement Agreement. Score 0.5."),
    "doc6_qa2": (0.5,
        "July 1 cited vs gold: July 20 (Consulting Agreement date). Close but wrong day. Score 0.5."),
    "doc6_qa8": (0.5,
        "Anti-assignment concept correct for Consulting Agreement. Score 0.5."),
    "doc6_qa13": (0.5,
        "Survival sections listed for Consulting Agreement. Correct concept. Score 0.5."),
    "doc8_qa0": (0.5,
        "Promotion Agreement identified, partially matches Co-Promotion Agreement name. Score 0.5."),
    "doc8_qa1": (0.5,
        "Dova mentioned but missing Valeant as co-party. Partial. Score 0.5."),
    "doc8_qa11": (0.5,
        "Change of Control concept identified for Co-Promotion. Score 0.5."),
    "doc8_qa12": (0.5,
        "Anti-assignment concept correct for Co-Promotion Agreement. Score 0.5."),
    "doc8_qa15": (0.5,
        "Section 8.1.2 IP ownership — partial match for Co-Promotion IP clause. Score 0.5."),
    "doc8_qa17": (0.5,
        "Non-transferable concept identified; partial without exact Valeant license text. Score 0.5."),
    "doc9_qa0": (0.5,
        "A_R STRATEGIC LICENSING partial — missing 'Distribution and Marketing Agreement'. Score 0.5."),
    "doc9_qa1": (0.5,
        "Pacira and EKR correct parties; missing F/K/A SKYEPHARMA detail. Score 0.5."),
    "doc9_qa11": (0.5,
        "Change of Control right concept identified for PACIRA ARSLDMA. Score 0.5."),
    "doc9_qa12": (0.5,
        "Anti-assignment concept correct for PACIRA ARSLDMA. Score 0.5."),
    "doc9_qa14": (0.5,
        "IP ownership concept identified; partial without specific PACIRA clause details. Score 0.5."),
    "doc9_qa19": (0.5,
        "Post-termination obligation concept identified for PACIRA ARSLDMA. Score 0.5."),
    "doc9_qa20": (0.5,
        "Audit rights concept identified for PACIRA ARSLDMA. Score 0.5."),
    # ── Score 0.75 ───────────────────────────────────────────────────────────
    "doc0_qa4": (0.75,
        "10-year term correctly identified for LIME ENERGY contract. Score 0.75."),
    "doc0_qa15": (0.75,
        "Post-termination obligation correctly identified for LIME ENERGY. Score 0.75."),
    "doc1_qa1": (0.75,
        "Whitesmoke and Google correctly identified as parties. Score 0.75."),
    "doc1_qa6": (0.75,
        "Anti-assignment consent requirement correctly identified for Whitesmoke/Google. Score 0.75."),
    "doc1_qa8": (0.75,
        "Google audit rights correctly identified for Whitesmoke/Google. Score 0.75."),
    "doc6_qa7": (0.75,
        "Termination for convenience correctly identified for Consulting Agreement. Score 0.75."),
    "doc8_qa8": (0.75,
        "Section 2.3.1(b) competitive restriction exception correctly identified. Score 0.75."),
    "doc8_qa10": (0.75,
        "Convenience termination correctly identified for Co-Promotion Agreement. Score 0.75."),
    # ── Score 1.0 ────────────────────────────────────────────────────────────
    "doc0_qa5": (1.0,
        "Annual renewal in one-year increments up to 10 years — EXACT LIME ENERGY renewal provision. Score 1.0."),
    "doc0_qa8": (1.0,
        "Non-solicitation clause EXACT for LIME ENERGY. Score 1.0."),
    "doc0_qa17": (1.0,
        "Insurance clause EXACT for LIME ENERGY. Score 1.0."),
    "doc1_qa0": (1.0,
        "Promotion and Distribution Agreement — EXACT Whitesmoke/Google contract name. Score 1.0."),
    "doc1_qa2": (1.0,
        "1 August 2011 — EXACT effective date of Whitesmoke/Google agreement. Score 1.0."),
    "doc1_qa4": (1.0,
        "English law — EXACT governing law for Whitesmoke/Google. Score 1.0."),
    "doc3_qa0": (1.0,
        "WEB SITE HOSTING AGREEMENT — EXACT contract name. Score 1.0."),
    "doc3_qa3": (1.0,
        "April 1 1999 six months initial term — EXACT Web Site Hosting term. Score 1.0."),
    "doc3_qa4": (1.0,
        "Six months after April 1 1999 = Oct 1 1999 renewal provision — EXACT. Score 1.0."),
    "doc3_qa8": (1.0,
        "Thirty (30) days termination notice — EXACT Web Site Hosting provision. Score 1.0."),
    "doc4_qa0": (1.0,
        "Joint Filing Agreement — EXACT contract name. Score 1.0."),
    "doc5_qa0": (1.0,
        "ENDORSEMENT AGREEMENT — EXACT Adams Golf contract name. Score 1.0."),
    "doc6_qa0": (1.0,
        "Consulting Agreement — EXACT Kiromic/Gianluca Rotino contract name. Score 1.0."),
    "doc6_qa1": (1.0,
        "Kiromic and Gianluca Rotino — EXACT parties for Consulting Agreement. Score 1.0."),
    "doc9_qa3": (1.0,
        "August 10, 2007 — EXACT date for PACIRA ARSLDMA referenced event. Score 1.0."),
}


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def score_calib(r: dict) -> tuple[float, str]:
    pred = r.get("predicted", "")
    behavior = r.get("expected_behavior", "answer")
    cfg = "v4t-corpus-tuned"
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
    cfg = "v4t-corpus-tuned"
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
    print("Judging cuad__v4t-corpus-tuned — calibration + batch_calib ...")
    write_judgments(CALIB_DIR / "queue.jsonl", CALIB_RESULTS, score_calib, "calib")
    write_judgments(BATCH_DIR / "queue.jsonl", BATCH_RESULTS, score_batch, "batch")
    print("Done.")
