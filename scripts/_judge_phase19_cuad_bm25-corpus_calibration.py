"""Phase 1.9 — CUAD bm25-corpus Protocol B calibration + batch_calib judge.

Cells:
  cuad__bm25-corpus__calibration__seed42   (50 entries)
  cuad__bm25-corpus__batch_calib__seed42   (132 entries)

Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  Protocol B rubric

Key findings:
  - bm25-corpus: keyword-based retrieval, no recency decay, no theta parameters
  - Good on exact-name and governing-law questions for doc0/doc1/doc3/doc5/doc6/doc9
  - Calibration: 27 acknowledge_missing, 23 answer; 20 non-refusals with non-zero scores
  - Batch_calib: 132 entries; strong on doc0/doc1/doc3/doc5/doc6/doc9
    Fails on doc2/doc4/doc7/doc8 with contamination from PACIRA/other sources

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
CALIB_DIR = JQ / "cuad__bm25-corpus__calibration__seed42"
BATCH_DIR = JQ / "cuad__bm25-corpus__batch_calib__seed42"

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
# suffix = qid minus "cuad__bm25-corpus__calibration__" and "__seed42"
# Only answer+non-refusal entries needing non-default (>0.0) scores
# ---------------------------------------------------------------------------
CALIB_JUDGMENTS: dict[str, tuple[float, str]] = {
    "doc0_qa6__after1": (0.25,
        "Gold: Illinois; pred gives English law (BM25 retrieves wrong governing law clause). Score 0.25."),
    "doc1_qa8__after1": (0.75,
        "Google audit rights correctly identified. Score 0.75."),
    "doc0_qa1__after2": (0.25,
        "Vague parties answer, doesn't name Distributor specifically. Score 0.25."),
    "doc3_qa6__after3": (0.25,
        "Pred: 15 days vs gold: 30 days (Web Site Hosting notice period). Score 0.25."),
    "doc1_qa6__after4": (0.75,
        "Consent required for assignment — correctly identifies anti-assignment for Whitesmoke/Google. Score 0.75."),
    "doc3_qa6__after4": (0.25,
        "Pred: 15 days vs gold: 30 days. Wrong notice period. Score 0.25."),
    "doc2_qa1__after5": (0.25,
        "Framework clause cited; gold asks for seller name. Score 0.25."),
    "doc2_qa3__after6": (0.5,
        "Hong Kong law cited — close to People's Republic of China for Supply Contract governing law. Score 0.5."),
    "doc5_qa12__after6": (1.0,
        "CONSULTANT grants ADAMS GOLF exclusive ENDORSEMENT license — EXACT. Score 1.0."),
    "doc1_qa6__after6": (0.75,
        "Anti-assignment consent requirement correct for Whitesmoke/Google. Score 0.75."),
    "doc7_qa4__after7": (0.25,
        "Five years from October 2019 cited vs VNBJ Closing term — wrong provision. Score 0.25."),
    "doc5_qa8__after7": (0.25,
        "Vague competitive restriction exception, not specific enough. Score 0.25."),
    "doc0_qa6__after7": (1.0,
        "Illinois EXACT — correct governing law for LIME ENERGY. Score 1.0."),
    "doc0_qa5__after8": (1.0,
        "Annual 1-year terms up to 10 years — LIME ENERGY renewal provision EXACT. Score 1.0."),
    "doc6_qa13__after8": (0.5,
        "Post-termination obligations concept correct for Consulting Agreement. Score 0.5."),
    "doc3_qa0__after9": (1.0,
        "WEB SITE HOSTING AGREEMENT — EXACT contract name. Score 1.0."),
    "doc7_qa5__after9": (0.25,
        "Gold: Japan; pred: New York. Wrong jurisdiction for Veoneer/Nissin Amendment. Score 0.25."),
    "doc1_qa4__after9": (0.25,
        "Gold: English law; pred: Illinois. Wrong governing law for Whitesmoke/Google. Score 0.25."),
    "doc9_qa11__after9": (0.5,
        "Change of Control right concept identified for PACIRA ARSLDMA. Score 0.5."),
    "doc1_qa1__after9": (0.75,
        "Whitesmoke and Google correctly identified as parties. Score 0.75."),
}


# ---------------------------------------------------------------------------
# BATCH_CALIB JUDGMENTS: suffix → (score, rationale)
# suffix = qid minus "cuad__bm25-corpus__batch__" and "__seed42"
# Only non-refusal entries needing non-default (>0.0) scores
# ---------------------------------------------------------------------------
BATCH_JUDGMENTS: dict[str, tuple[float, str]] = {
    # ── Score 1.0 ────────────────────────────────────────────────────────────
    "doc0_qa0": (1.0,
        "DISTRIBUTOR AGREEMENT — EXACT contract name. Score 1.0."),
    "doc0_qa4": (1.0,
        "Ten (10) years from date of first Sample delivery — EXACT LIME ENERGY term provision. Score 1.0."),
    "doc0_qa6": (1.0,
        "Construed according to the laws of the State of Illinois — EXACT governing law. Score 1.0."),
    "doc0_qa11": (1.0,
        "No assignment of this Agreement...shall be made by the Distributor in whole or in part, without... — EXACT. Score 1.0."),
    "doc0_qa13": (1.0,
        "Minimum purchase order of $250,000.00 by the first of each month — EXACT. Score 1.0."),
    "doc1_qa0": (1.0,
        "PROMOTION AND DISTRIBUTION AGREEMENT — EXACT contract name. Score 1.0."),
    "doc1_qa2": (1.0,
        "1 August 2011 — EXACT effective date of Whitesmoke/Google agreement. Score 1.0."),
    "doc3_qa0": (1.0,
        "WEB SITE HOSTING AGREEMENT — EXACT contract name. Score 1.0."),
    "doc3_qa1": (1.0,
        "Centrack International — EXACT party in Web Site Hosting Agreement. Score 1.0."),
    "doc3_qa2": (1.0,
        "6th day of April, 1999 — EXACT agreement date. Score 1.0."),
    "doc3_qa7": (1.0,
        "Governed by the laws of the State of Florida — EXACT jurisdiction. Score 1.0."),
    "doc5_qa0": (1.0,
        "ENDORSEMENT AGREEMENT — EXACT Adams Golf contract name. Score 1.0."),
    "doc5_qa12": (1.0,
        "CONSULTANT grants ADAMS GOLF exclusive right to use CONSULTANT'S ENDORSEMENT — EXACT. Score 1.0."),
    "doc6_qa0": (1.0,
        "Consulting Agreement — EXACT Kiromic/Gianluca Rotino contract name. Score 1.0."),
    "doc8_qa23": (1.0,
        "Each Party shall maintain comprehensive product liability insurance — EXACT Co-Promotion insurance clause. Score 1.0."),
    "doc9_qa0": (1.0,
        "Amended and Restated Strategic Licensing, Distribution and Marketing Agreement — EXACT PACIRA ARSLDMA name. Score 1.0."),
    "doc9_qa2": (1.0,
        "October 15, 2009 — EXACT effective date of PACIRA ARSLDMA. Score 1.0."),
    "doc9_qa4": (1.0,
        "Fifteen (15) years from the first Commercial Launch of the Product in the Territory — EXACT PACIRA term. Score 1.0."),
    "doc9_qa5": (1.0,
        "Two (2) year consecutive renewal periods — EXACT PACIRA ARSLDMA renewal provision. Score 1.0."),
    "doc9_qa7": (1.0,
        "Governed by the laws of the State of New York — EXACT PACIRA ARSLDMA jurisdiction. Score 1.0."),
    "doc9_qa23": (1.0,
        "Each Party shall maintain comprehensive product liability insurance — EXACT PACIRA insurance clause. Score 1.0."),
    # ── Score 0.75 ───────────────────────────────────────────────────────────
    "doc0_qa12": (0.75,
        "Price restriction concept correct (annual adjustments); specific mechanism slightly different from gold. Score 0.75."),
    "doc0_qa17": (0.75,
        "Insurance requirement correctly identified for LIME ENERGY; correct concept, slightly broader than gold. Score 0.75."),
    "doc1_qa1": (0.75,
        "Whitesmoke and Google parties correctly identified; slightly broader than gold (Distributor only). Score 0.75."),
    "doc1_qa7": (0.75,
        "Google grants to Whitesmoke Inc. for distribution — correct license direction and parties. Score 0.75."),
    "doc1_qa8": (0.75,
        "Google may audit Distributor's relevant records — correct audit direction and parties. Score 0.75."),
    "doc5_qa7": (0.75,
        "CONSULTANT shall exclusively play/use ADAMS GOLF product during term — correct non-compete. Score 0.75."),
    "doc6_qa7": (0.75,
        "Either party may terminate without cause upon 30 days notice — correct termination for convenience. Score 0.75."),
    "doc6_qa8": (0.75,
        "Anti-assignment clause correctly identified for Consulting Agreement. Score 0.75."),
    "doc7_qa3": (0.75,
        "October 30, 2019 correct date for Amendment Effective Date (VNBJ Closing); concept right. Score 0.75."),
    "doc8_qa5": (0.75,
        "Governed by the laws of the State of New York — correct jurisdiction for Co-Promotion Agreement. Score 0.75."),
    "doc8_qa12": (0.75,
        "Anti-assignment consent requirement correctly identified for Co-Promotion Agreement. Score 0.75."),
    "doc8_qa19": (0.75,
        "Both Dova and Valeant have audit rights — slightly broader than gold (Dova only), but mostly correct. Score 0.75."),
    "doc9_qa16": (0.75,
        "PPI grants EKR exclusive right and license — correct license direction and parties. Score 0.75."),
    "doc9_qa22": (0.75,
        "EKR/PPI mutual liability cap correctly identified with correct party names. Score 0.75."),
    # ── Score 0.5 ────────────────────────────────────────────────────────────
    "doc0_qa14": (0.5,
        "License concept identified (Section 2.1); gold asks for exclusive distributor appointment. Partial. Score 0.5."),
    "doc0_qa15": (0.5,
        "Post-termination obligation concept correct for LIME ENERGY. Score 0.5."),
    "doc0_qa16": (0.5,
        "Warranty concept identified (24 months); specific provision may differ from gold. Score 0.5."),
    "doc1_qa5": (0.5,
        "Change of Control right concept identified for Whitesmoke/Google. Score 0.5."),
    "doc1_qa9": (0.5,
        "Limitation of liability concept correct; specific cap clause not fully specified. Score 0.5."),
    "doc1_qa11": (0.5,
        "Warranty duration (24 months) cited; gold value is redacted [*], possible match. Score 0.5."),
    "doc5_qa9": (0.5,
        "Anti-assignment/sublicense restriction concept correct for Endorsement Agreement. Score 0.5."),
    "doc6_qa13": (0.5,
        "Post-termination obligations concept correct for Consulting Agreement. Score 0.5."),
    "doc7_qa1": (0.5,
        "Parties listed; likely mentions Nissin and Veoneer but vague response format. Score 0.5."),
    "doc8_qa0": (0.5,
        "'Promotion Agreement' partial match for Co-Promotion Agreement name. Score 0.5."),
    "doc8_qa11": (0.5,
        "Change of Control provision concept identified for Co-Promotion Agreement. Score 0.5."),
    "doc8_qa14": (0.5,
        "Minimum obligation/detail requirement concept correct for Co-Promotion. Score 0.5."),
    "doc9_qa1": (0.5,
        "PACIRA PHARMACEUTICALS and EKR THERAPEUTICS correct parties; missing F/K/A SKYEPHARMA. Score 0.5."),
    "doc9_qa11": (0.5,
        "Change of Control right concept identified for PACIRA ARSLDMA. Score 0.5."),
    "doc9_qa19": (0.5,
        "Post-termination obligations concept (transfer of Materials) correct for PACIRA. Score 0.5."),
    "doc9_qa20": (0.5,
        "Audit rights concept correct; both parties right to audit counterparty. Score 0.5."),
    # ── Score 0.25 ───────────────────────────────────────────────────────────
    "doc0_qa5": (0.25,
        "Pred: two (2) year consecutive renewal vs gold: one (1) year terms for LIME ENERGY. Wrong period. Score 0.25."),
    "doc1_qa4": (0.25,
        "Pred: Illinois vs gold: English law for Whitesmoke/Google. Wrong governing law. Score 0.25."),
    "doc3_qa6": (0.25,
        "Pred: 15 days vs gold: 30 days notice period for Web Site Hosting. Wrong number. Score 0.25."),
    "doc3_qa8": (0.25,
        "Pred: 60 days vs gold: 30 days termination notice for Web Site Hosting. Wrong number. Score 0.25."),
    "doc3_qa9": (0.25,
        "Sections 10.9/10.8 cited are from EKR/PACIRA context, not i-on (Web Site Hosting). Wrong source. Score 0.25."),
    "doc5_qa1": (0.25,
        "Vague parties response; gold: ADAMS GOLF. Score 0.25."),
    "doc5_qa5": (0.25,
        "Pred: New York vs gold: Kansas for Endorsement Agreement. Wrong jurisdiction. Score 0.25."),
    "doc5_qa8": (0.25,
        "Competitive restriction exception cited from wrong context (Dova/Co-Promotion). Score 0.25."),
    "doc6_qa4": (0.25,
        "Five years from effective date cited; gold: Consulting Agreement continues until termination (no fixed term). Score 0.25."),
    "doc6_qa5": (0.25,
        "Pred: New York vs gold: Texas for Consulting Agreement. Wrong jurisdiction. Score 0.25."),
    "doc6_qa10": (0.25,
        "Other Party shall not acquire ownership (reversed/wrong direction vs gold: Consultant assigns to Company). Score 0.25."),
    "doc6_qa12": (0.25,
        "Non-exclusive license grant cited; gold asks about Consultant incorporating work into assignments. Different clause. Score 0.25."),
    "doc7_qa0": (0.25,
        "Joint Venture Agreement — missing 'AMENDMENT AND TERMINATION OF' prefix. Score 0.25."),
    "doc7_qa5": (0.25,
        "Pred: New York vs gold: Japan for Veoneer/Nissin Amendment. Wrong jurisdiction. Score 0.25."),
    "doc8_qa15": (0.25,
        "Other Party shall not acquire ownership rights (reversed vs gold: Dova owns Product Materials). Score 0.25."),
    "doc8_qa17": (0.25,
        "Distributor shall not assign cited vs gold: Valeant's rights non-transferable under Section 2.1. Wrong source. Score 0.25."),
    "doc8_qa20": (0.25,
        "Death/personal injury exception cited; gold asks about specific liability cap exclusions. Different provision. Score 0.25."),
    "doc9_qa10": (0.25,
        "Pred: 30 days vs gold: 60 days prior written notice for PPI termination right. Wrong number. Score 0.25."),
    "doc9_qa12": (0.25,
        "Section 13.2 cited vs gold: Section 20.2 for anti-assignment. Wrong section. Score 0.25."),
    "doc9_qa14": (0.25,
        "Other Party shall not acquire ownership rights; gold asks about EKR IP acquisition upon termination. Reversed. Score 0.25."),
    "doc9_qa17": (0.25,
        "Anti-assignment language cited vs gold: EKR may appoint sub-distributors. Wrong clause type. Score 0.25."),
    "doc9_qa18": (0.25,
        "Non-exclusive, royalty-free license cited; gold asks about PPI/EKR Improvements ownership structure. Score 0.25."),
}


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def score_calib(r: dict) -> tuple[float, str]:
    pred = r.get("predicted", "")
    behavior = r.get("expected_behavior", "answer")
    cfg = "bm25-corpus"
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
    cfg = "bm25-corpus"
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
    print("Judging cuad__bm25-corpus — calibration + batch_calib ...")
    write_judgments(CALIB_DIR / "queue.jsonl", CALIB_RESULTS, score_calib, "calib")
    write_judgments(BATCH_DIR / "queue.jsonl", BATCH_RESULTS, score_batch, "batch")
    print("Done.")
