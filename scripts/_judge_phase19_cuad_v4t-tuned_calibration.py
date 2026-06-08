"""Phase 1.9 — CUAD v4t-tuned Protocol B calibration + batch_calib judge.

Cells:
  cuad__v4t-tuned__calibration__seed42   (50 entries)
  cuad__v4t-tuned__batch_calib__seed42   (132 entries)

Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  Protocol B rubric

Key findings:
  - v4t-tuned: theta tuned on CUAD corpus, better retrieval than canonical
  - Still suffers from PACIRA Promissory Note contamination but less severely
  - Calibration: 27 acknowledge_missing, 23 answer entries
    Of 23 answer: ~7 refuse, 16 non-refusal entries with various scores
  - Batch_calib: 132 entries, many PACIRA-contaminated (0.0), but more successes
    doc8/doc9 (Co-Promotion + PACIRA) fare better due to CUAD-tuned retrieval

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
CALIB_DIR = JQ / "cuad__v4t-tuned__calibration__seed42"
BATCH_DIR = JQ / "cuad__v4t-tuned__batch_calib__seed42"

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
# suffix = qid minus "cuad__v4t-tuned__calibration__" and "__seed42"
# Only answer+non-refusal entries needing non-default (>0.0) scores
# ---------------------------------------------------------------------------
CALIB_JUDGMENTS: dict[str, tuple[float, str]] = {
    "doc0_qa6__after1": (0.25,
        "Gold: Illinois; pred gives English law (PACIRA contamination). Wrong governing law. Score 0.25."),
    "doc1_qa8__after1": (0.75,
        "Gold: Google audit rights; pred correctly identifies Google's right to audit Distributor records. Score 0.75."),
    "doc0_qa1__after2": (0.25,
        "Vague answer about parties, doesn't name Distributor specifically. Score 0.25."),
    "doc3_qa6__after3": (0.25,
        "Pred: 15 days notice period vs gold: 30 days (Web Site Hosting). Wrong number. Score 0.25."),
    "doc1_qa6__after4": (0.75,
        "Consent required for assignment — correctly identifies anti-assignment clause. Score 0.75."),
    "doc3_qa6__after4": (0.25,
        "Pred: 15 days vs gold: 30 days notice. Wrong number repeated. Score 0.25."),
    "doc3_qa9__after6": (0.25,
        "Clause 9.4 cited is from wrong source (not Web Site Hosting Agreement). Score 0.25."),
    "doc5_qa12__after6": (1.0,
        "ADAMS GOLF exclusive license to CONSULTANT's ENDORSEMENT — EXACT match. Score 1.0."),
    "doc1_qa6__after6": (0.75,
        "Consent required for Distributor assignment — correct anti-assignment identification. Score 0.75."),
    "doc5_qa8__after7": (0.25,
        "Vague competitive restriction exception reference, not specific enough. Score 0.25."),
    "doc6_qa6__after8": (0.25,
        "Conflicting Obligations mentioned but gold asks for Non-Compete clause. Wrong clause type. Score 0.25."),
    "doc0_qa5__after8": (0.25,
        "Monthly renewal (one month periods) vs gold: annual (one year terms up to 10 years). Wrong renewal period. Score 0.25."),
    "doc6_qa13__after8": (0.5,
        "Survival sections listed (payment, confidentiality, post-termination). Correct concept, partial specificity. Score 0.5."),
    "doc7_qa5__after9": (0.25,
        "Gold: Japan (Nissin/Veoneer Amendment); pred gives New York. Wrong jurisdiction. Score 0.25."),
    "doc1_qa4__after9": (0.25,
        "Gold: English law (Whitesmoke/Google); pred gives New York. Wrong governing law. Score 0.25."),
    "doc9_qa11__after9": (0.5,
        "Change of control right concept identified, relevant clause present. Score 0.5."),
}


# ---------------------------------------------------------------------------
# BATCH_CALIB JUDGMENTS: suffix → (score, rationale)
# suffix = qid minus "cuad__v4t-tuned__batch__" and "__seed42"
# Only non-refusal entries needing non-default (>0.0) scores
# ---------------------------------------------------------------------------
BATCH_JUDGMENTS: dict[str, tuple[float, str]] = {
    # ── Score 0.25 ───────────────────────────────────────────────────────────
    "doc0_qa6": (0.25,
        "Pred: New York vs gold: Illinois (LIME ENERGY). Wrong governing law. Score 0.25."),
    "doc0_qa9": (0.25,
        "Valeant/Dova non-solicitation vs LIME ENERGY non-solicitation. Wrong source contract. Score 0.25."),
    "doc0_qa12": (0.25,
        "Additional Royalty clause cited but gold asks for price restriction on LIME ENERGY. Wrong clause. Score 0.25."),
    "doc1_qa4": (0.25,
        "Pred: New York vs gold: English law (Whitesmoke/Google). Wrong jurisdiction. Score 0.25."),
    "doc1_qa8": (0.25,
        "Valeant audits Dova vs gold: Google audits Distributor — wrong parties. Score 0.25."),
    "doc1_qa10": (0.25,
        "PPI/EKR liability cap cited vs LIME ENERGY/Google limitation of liability. Wrong source. Score 0.25."),
    "doc2_qa3": (0.25,
        "Pred: New York vs gold: People's Republic of China (Supply Contract). Wrong law. Score 0.25."),
    "doc2_qa5": (0.25,
        "Products liability clause vs gold: All Risks shipping insurance. Wrong clause type. Score 0.25."),
    "doc3_qa7": (0.25,
        "Pred: New York vs gold: Florida (Web Site Hosting/i-on). Wrong jurisdiction. Score 0.25."),
    "doc3_qa9": (0.25,
        "DOVA/VALEANT liability vs i-on (Web Site Hosting) limitation of liability. Wrong source. Score 0.25."),
    "doc5_qa5": (0.25,
        "Pred: New York vs gold: Kansas (Endorsement Agreement/Adams Golf). Wrong jurisdiction. Score 0.25."),
    "doc5_qa6": (0.25,
        "Section 2.3.1 PACIRA cited vs gold: CONSULTANT wearing Adams Golf gear. Wrong clause. Score 0.25."),
    "doc5_qa8": (0.25,
        "Dova termination exception vs CONSULTANT competitive exception (Endorsement). Wrong source. Score 0.25."),
    "doc5_qa11": (0.25,
        "Dova speaker programs vs CONSULTANT days available (Endorsement Agreement). Wrong source. Score 0.25."),
    "doc6_qa5": (0.25,
        "Pred: New York vs gold: Texas (Consulting Agreement/Kiromic). Wrong jurisdiction. Score 0.25."),
    "doc7_qa5": (0.25,
        "Pred: New York vs gold: Japan (Veoneer/Nissin Amendment). Wrong jurisdiction. Score 0.25."),
    "doc8_qa3": (0.25,
        "PACIRA Aug 10 2007 date vs gold: Co-Promotion effective date (Sep 26 2018). Wrong date/contract. Score 0.25."),
    "doc8_qa16": (0.25,
        "License direction reversed — pred: Valeant grants to Dova vs gold: Dova grants to Valeant. Score 0.25."),
    "doc8_qa17": (0.25,
        "Promissory Note nontransferable vs Valeant non-transferable license. Wrong source. Score 0.25."),
    "doc8_qa18": (0.25,
        "License grant restrictions described vaguely; missing specific sublicense/transfer terms. Score 0.25."),
    "doc8_qa19": (0.25,
        "Pred: Valeant audits Dova (reversed) vs gold: Dova and Valeant mutual audit rights. Score 0.25."),
    "doc8_qa20": (0.25,
        "Section 11.4 liability cap context wrong; missing specific Co-Promotion cap terms. Score 0.25."),
    "doc9_qa8": (0.25,
        "EKR restriction cited but gold asks about PPI (Pacira) competitive restriction. Wrong party. Score 0.25."),
    "doc9_qa10": (0.25,
        "EKR terminates vs gold: PPI terminates; no specific date context. Wrong party/source. Score 0.25."),
    "doc9_qa12": (0.25,
        "Promissory Note nonnegotiable vs PACIRA anti-assignment clause. Source confusion. Score 0.25."),
    "doc9_qa17": (0.25,
        "Promissory Note nontransferable vs EKR sub-distributor assignment. Wrong source. Score 0.25."),
    "doc9_qa20": (0.25,
        "ARTICLE 7 AUDIT RIGHTS title only; no specific content about who audits whom. Score 0.25."),
    # ── Score 0.5 ────────────────────────────────────────────────────────────
    "doc0_qa11": (0.5,
        "Anti-assignment right concept identified for LIME ENERGY; partial without exact clause text. Score 0.5."),
    "doc0_qa17": (0.5,
        "Insurance right concept identified; partial without specific LIME ENERGY policy terms. Score 0.5."),
    "doc1_qa5": (0.5,
        "Change of Control right concept identified for Whitesmoke/Google. Score 0.5."),
    "doc2_qa0": (0.5,
        "Supply Agreement identified — close to gold SUPPLY CONTRACT (Shenzhen LOHAS). Score 0.5."),
    "doc3_qa8": (0.5,
        "Convenience termination right concept correct for Web Site Hosting. Score 0.5."),
    "doc5_qa9": (0.5,
        "Anti-assignment right concept for Endorsement Agreement. Score 0.5."),
    "doc7_qa1": (0.5,
        "Signature Page identifies parties; mentions Nissin — partial correct identification. Score 0.5."),
    "doc8_qa1": (0.5,
        "Co-Promotion Agreement identified; September 26 2018 date partially correct. Score 0.5."),
    "doc8_qa11": (0.5,
        "Change of Control Section 20.4 right concept identified. Score 0.5."),
    "doc8_qa23": (0.5,
        "Insurance right concept identified for Co-Promotion. Score 0.5."),
    "doc9_qa1": (0.5,
        "EKR and Pacira correct parties; missing F/K/A SKYEPHARMA. Score 0.5."),
    "doc9_qa11": (0.5,
        "Change of Control right concept identified for PACIRA ARSLDMA. Score 0.5."),
    "doc9_qa19": (0.5,
        "Post-termination right concept identified for PACIRA ARSLDMA. Score 0.5."),
    # ── Score 0.75 ───────────────────────────────────────────────────────────
    "doc1_qa6": (0.75,
        "Consent required for assignment — correct anti-assignment for Whitesmoke/Google. Score 0.75."),
    "doc6_qa7": (0.75,
        "Termination for convenience right concept correct for Consulting Agreement. Score 0.75."),
    "doc8_qa5": (0.75,
        "New York governing law for Co-Promotion Agreement — correct jurisdiction. Score 0.75."),
    "doc8_qa6": (0.75,
        "Valeant competitive restriction correctly identified in Co-Promotion Agreement. Score 0.75."),
    "doc8_qa8": (0.75,
        "Section 2.3.1(a) competitive restriction exception identified correctly. Score 0.75."),
    "doc8_qa10": (0.75,
        "Convenience termination clause correctly identified for Co-Promotion. Score 0.75."),
    "doc8_qa12": (0.75,
        "Anti-assignment correctly identified for Co-Promotion Agreement. Score 0.75."),
    "doc8_qa15": (0.75,
        "Valeant assigns IP ownership to Dova — correct direction. Score 0.75."),
    "doc9_qa23": (0.75,
        "Insurance clause correctly identified for PACIRA ARSLDMA. Score 0.75."),
    # ── Score 1.0 ────────────────────────────────────────────────────────────
    "doc8_qa2": (1.0,
        "September 26, 2018 — EXACT effective date of Co-Promotion Agreement. Score 1.0."),
    "doc8_qa9": (1.0,
        "Neither Valeant nor Dova shall solicit employees — EXACT non-solicitation clause. Score 1.0."),
    "doc9_qa0": (1.0,
        "Amended and Restated Strategic Licensing, Distribution and Marketing Agreement — EXACT PACIRA ARSLDMA name. Score 1.0."),
    "doc9_qa2": (1.0,
        "October 15, 2009 — EXACT effective date of PACIRA ARSLDMA. Score 1.0."),
    "doc9_qa5": (1.0,
        "Two-year consecutive renewal term — EXACT PACIRA ARSLDMA renewal provision. Score 1.0."),
    "doc9_qa7": (1.0,
        "New York governing law for PACIRA ARSLDMA — EXACT jurisdiction. Score 1.0."),
}


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def score_calib(r: dict) -> tuple[float, str]:
    pred = r.get("predicted", "")
    behavior = r.get("expected_behavior", "answer")
    cfg = "v4t-tuned"
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
    cfg = "v4t-tuned"
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
    print("Judging cuad__v4t-tuned — calibration + batch_calib ...")
    write_judgments(CALIB_DIR / "queue.jsonl", CALIB_RESULTS, score_calib, "calib")
    write_judgments(BATCH_DIR / "queue.jsonl", BATCH_RESULTS, score_batch, "batch")
    print("Done.")
