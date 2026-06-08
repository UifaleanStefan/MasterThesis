"""Phase 1.9 — CUAD attention-corpus-tuned Protocol B calibration + batch_calib judge.

Cells:
  cuad__attention-corpus-tuned__calibration__seed42   (50 entries)
  cuad__attention-corpus-tuned__batch_calib__seed42   (132 entries)

Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  Protocol B rubric

Key findings:
  - attention-corpus-tuned: attention-weighted retrieval + corpus tuning
  - Best calibration performance among 6 configs: lowest contamination, best per-doc accuracy
  - Calibration: 27 acknowledge_missing, 23 answer entries; 22 non-refusal with non-zero scores
  - Batch_calib: 132 entries; strong on most docs except doc2 (Supply Contract) and partial doc9

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
CALIB_DIR = JQ / "cuad__attention-corpus-tuned__calibration__seed42"
BATCH_DIR = JQ / "cuad__attention-corpus-tuned__batch_calib__seed42"

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
# suffix = qid minus "cuad__attention-corpus-tuned__calibration__" and "__seed42"
# Only answer+non-refusal entries needing non-default (>0.0) scores
# ---------------------------------------------------------------------------
CALIB_JUDGMENTS: dict[str, tuple[float, str]] = {
    "doc0_qa6__after1": (0.25,
        "Gold: Illinois; pred gives English law. Wrong governing law. Score 0.25."),
    "doc1_qa8__after1": (0.75,
        "Google audit rights correctly identified. Score 0.75."),
    "doc0_qa1__after2": (0.25,
        "Mentions DISTRIBUTOR AGREEMENT title but vague, doesn't name Distributor party. Score 0.25."),
    "doc3_qa6__after3": (0.25,
        "Pred: 15 days vs gold: 30 days (Web Site Hosting notice period). Score 0.25."),
    "doc1_qa6__after4": (0.75,
        "Consent required for assignment — correct anti-assignment for Whitesmoke/Google. Score 0.75."),
    "doc3_qa6__after4": (0.25,
        "Pred: 15 days vs gold: 30 days. Wrong notice period. Score 0.25."),
    "doc2_qa1__after5": (0.5,
        "Shenzhen LOHAS identified as buyer/seller — partial match for Supply Contract parties. Score 0.5."),
    "doc3_qa9__after6": (0.25,
        "Clause 9.4 cited is from Whitesmoke/Google context, not Web Site Hosting (i-on). Wrong source. Score 0.25."),
    "doc2_qa3__after6": (0.25,
        "Hong Kong cited with no PRC mention — close but not the correct governing law. Score 0.25."),
    "doc5_qa12__after6": (1.0,
        "ADAMS GOLF exclusive ENDORSEMENT license from CONSULTANT — EXACT. Score 1.0."),
    "doc1_qa6__after6": (0.75,
        "Anti-assignment consent requirement correctly identified for Whitesmoke/Google. Score 0.75."),
    "doc7_qa4__after7": (0.5,
        "October 30, 2019 cited — correct date for VNBJ Closing which governs Amendment effective date. Score 0.5."),
    "doc5_qa8__after7": (0.25,
        "Vague competitive restriction exception reference, not specific enough. Score 0.25."),
    "doc0_qa6__after7": (0.25,
        "Gold: Illinois; pred gives English law again at later calibration point. Score 0.25."),
    "doc5_qa11__after8": (0.25,
        "Speaker Program Threshold cited (Co-Promotion context) vs CONSULTANT days available (Endorsement). Wrong source. Score 0.25."),
    "doc6_qa6__after8": (0.75,
        "Consulting Agreement non-compete correctly identified — Consultant agrees not to participate in competing business. Score 0.75."),
    "doc0_qa5__after8": (1.0,
        "One (1) year terms for up to another ten (10) years — LIME ENERGY renewal provision EXACT. Score 1.0."),
    "doc6_qa13__after8": (1.0,
        "Upon termination, Consultant delivers Confidential Information to Company — EXACT post-termination obligation. Score 1.0."),
    "doc3_qa0__after9": (0.75,
        "Mentions 'WEB SITE HOSTING AGREEMENT' by name in lawyer-review context. Score 0.75."),
    "doc7_qa5__after9": (0.25,
        "Gold: Japan; pred: English law. Wrong jurisdiction for Veoneer/Nissin Amendment. Score 0.25."),
    "doc9_qa11__after9": (0.5,
        "Change of Control right concept identified for PACIRA ARSLDMA. Score 0.5."),
    "doc1_qa1__after9": (0.75,
        "Whitesmoke and Google correctly identified as parties. Score 0.75."),
}


# ---------------------------------------------------------------------------
# BATCH_CALIB JUDGMENTS: suffix → (score, rationale)
# suffix = qid minus "cuad__attention-corpus-tuned__batch__" and "__seed42"
# Only non-refusal entries needing non-default (>0.0) scores
# ---------------------------------------------------------------------------
BATCH_JUDGMENTS: dict[str, tuple[float, str]] = {
    # ── Score 1.0 ────────────────────────────────────────────────────────────
    "doc0_qa4": (1.0,
        "Ten (10) years from the date upon which the Company delivers the last Sample — EXACT LIME ENERGY term. Score 1.0."),
    "doc0_qa5": (1.0,
        "Annual basis for one (1) year terms for up to another ten (10) years — EXACT LIME ENERGY renewal. Score 1.0."),
    "doc1_qa0": (1.0,
        "Promotion and Distribution Agreement — EXACT contract name. Score 1.0."),
    "doc1_qa2": (1.0,
        "1 August 2011 — EXACT effective date. Score 1.0."),
    "doc1_qa8": (1.0,
        "Google right to audit Distributor's relevant records — EXACT audit direction. Score 1.0."),
    "doc3_qa1": (1.0,
        "Centrack International — EXACT party in Web Site Hosting Agreement. Score 1.0."),
    "doc3_qa2": (1.0,
        "April 6, 1999 — EXACT agreement date (same as 6th day of April, 1999). Score 1.0."),
    "doc3_qa4": (1.0,
        "October 1, 1999 — EXACT initial term expiration (April 1 + 6 months). Score 1.0."),
    "doc3_qa6": (1.0,
        "Thirty (30) days' written notice — EXACT notice period for Web Site Hosting. Score 1.0."),
    "doc3_qa8": (1.0,
        "Either party may terminate without cause upon thirty (30) days' written notice — EXACT. Score 1.0."),
    "doc4_qa0": (1.0,
        "JOINT FILING AGREEMENT — EXACT contract name. Score 1.0."),
    "doc5_qa0": (1.0,
        "ENDORSEMENT AGREEMENT — EXACT Adams Golf contract name. Score 1.0."),
    "doc5_qa5": (1.0,
        "Governed by the laws of the State of Kansas — EXACT Endorsement Agreement jurisdiction. Score 1.0."),
    "doc5_qa12": (1.0,
        "CONSULTANT grants ADAMS GOLF exclusive right and license to use ENDORSEMENT — EXACT. Score 1.0."),
    "doc6_qa0": (1.0,
        "Consulting Agreement — EXACT Kiromic/Gianluca Rotino contract name. Score 1.0."),
    "doc6_qa1": (1.0,
        "Kiromic as Company and Gianluca Rotino as Consultant — EXACT parties. Score 1.0."),
    "doc6_qa3": (1.0,
        "July 1, 2018 — EXACT Effective Date of Consulting Agreement. Score 1.0."),
    "doc8_qa2": (1.0,
        "September 26, 2018 — EXACT Co-Promotion Agreement effective date. Score 1.0."),
    "doc8_qa9": (1.0,
        "Neither Valeant nor Dova (nor their Affiliates) shall solicit employees — EXACT. Score 1.0."),
    "doc9_qa1": (1.0,
        "PACIRA PHARMACEUTICALS (F/K/A SKYEPHARMA) and EKR THERAPEUTICS — EXACT parties including F/K/A. Score 1.0."),
    "doc9_qa2": (1.0,
        "October 15, 2009 — EXACT effective date of PACIRA ARSLDMA. Score 1.0."),
    "doc9_qa5": (1.0,
        "Two (2) year consecutive renewal periods — EXACT PACIRA ARSLDMA renewal. Score 1.0."),
    "doc9_qa23": (1.0,
        "Each Party shall maintain comprehensive product liability insurance — EXACT PACIRA insurance. Score 1.0."),
    # ── Score 0.75 ───────────────────────────────────────────────────────────
    "doc0_qa9": (0.75,
        "Non-solicitation restriction correctly identified for LIME ENERGY; correct concept. Score 0.75."),
    "doc0_qa11": (0.75,
        "Consent required for Distributor to assign Agreement — correct anti-assignment concept. Score 0.75."),
    "doc0_qa15": (0.75,
        "Post-termination obligations correctly identified for LIME ENERGY. Score 0.75."),
    "doc0_qa17": (0.75,
        "Insurance requirement correctly identified; mentions product liability coverage. Score 0.75."),
    "doc1_qa1": (0.75,
        "Whitesmoke and Google correctly identified as parties; slightly broader than gold (Distributor). Score 0.75."),
    "doc1_qa6": (0.75,
        "Consent required for assignment — correct anti-assignment for Whitesmoke/Google. Score 0.75."),
    "doc3_qa0": (0.75,
        "WEB SITE HOSTING AGREEMENT mentioned by name in review context. Score 0.75."),
    "doc5_qa6": (0.75,
        "CONSULTANT non-compete clause correctly identified (Consultant agrees not to compete). Score 0.75."),
    "doc6_qa6": (0.75,
        "Consulting Agreement non-compete correctly identified. Score 0.75."),
    "doc6_qa7": (0.75,
        "Either party may terminate for convenience upon written notice — correct concept. Score 0.75."),
    "doc6_qa13": (0.75,
        "Survival sections cited (post-termination obligations concept correct). Score 0.75."),
    "doc7_qa3": (0.75,
        "October 30, 2019 cited as Effective Date of Amendment — correct VNBJ Closing date. Score 0.75."),
    "doc7_qa4": (0.75,
        "October 30, 2019 cited as Amendment term — correct VNBJ Closing date. Score 0.75."),
    "doc8_qa1": (0.75,
        "Dova and Valeant both named as parties — correct, slightly broader than gold (Valeant). Score 0.75."),
    "doc8_qa6": (0.75,
        "Valeant competitive restriction correctly identified for Co-Promotion Territory. Score 0.75."),
    "doc8_qa10": (0.75,
        "Either party may terminate for convenience upon 30 days notice — correct concept. Score 0.75."),
    "doc8_qa12": (0.75,
        "Consent required for assignment — correct anti-assignment for Co-Promotion. Score 0.75."),
    "doc8_qa15": (0.75,
        "Dova shall own all inventions and intellectual property — correct IP ownership direction. Score 0.75."),
    "doc8_qa23": (0.75,
        "Each Party maintain adequate insurance including product liability — correct concept. Score 0.75."),
    "doc9_qa12": (0.75,
        "Consent required for assignment or transfer — correct anti-assignment for PACIRA ARSLDMA. Score 0.75."),
    # ── Score 0.5 ────────────────────────────────────────────────────────────
    "doc0_qa12": (0.5,
        "Price restriction concept (step 117 internal reference); gold asks about Company right to adjust prices. Score 0.5."),
    "doc0_qa16": (0.5,
        "Warranty duration 24 months; gold describes Company warranty obligations. Partial. Score 0.5."),
    "doc1_qa5": (0.5,
        "Change of Control concept identified for Whitesmoke/Google. Score 0.5."),
    "doc1_qa10": (0.5,
        "Limitation of liability Section 11.4 cited; correct concept, different section from gold. Score 0.5."),
    "doc1_qa11": (0.5,
        "Warranty duration 24 months; gold value redacted [*], possible correct match. Score 0.5."),
    "doc2_qa0": (0.5,
        "'Supply Agreement' — partial match for gold SUPPLY CONTRACT. Score 0.5."),
    "doc3_qa3": (0.5,
        "April 6 date then mentions term; gold asks about commencement (April 1, 1999). Partial. Score 0.5."),
    "doc5_qa8": (0.5,
        "Section 2.3.1(a) competitive restriction exception concept correct for Endorsement. Score 0.5."),
    "doc5_qa9": (0.5,
        "Consent required for assignment/transfer — anti-assignment concept; gold asks about sublicense. Score 0.5."),
    "doc7_qa1": (0.5,
        "Parties identification; likely names Veoneer/Nissin. Score 0.5."),
    "doc8_qa4": (0.5,
        "September 26, 2023 — plausible 5-year initial term from Sep 26, 2018. Score 0.5."),
    "doc8_qa8": (0.5,
        "Competitive restriction exception concept identified for Co-Promotion. Score 0.5."),
    "doc8_qa11": (0.5,
        "Change of Control provision concept identified for Co-Promotion. Score 0.5."),
    "doc8_qa17": (0.5,
        "Transfer restriction concept identified; Valeant's non-transferable rights. Score 0.5."),
    "doc8_qa19": (0.5,
        "Audit rights concept correct for Co-Promotion. Score 0.5."),
    "doc9_qa0": (0.5,
        "A_R STRATEGIC LICENSING DISTRIBUTION AND MARKETING AGREEMENT with company prefix — partial name match. Score 0.5."),
    "doc9_qa11": (0.5,
        "Change of Control right concept identified for PACIRA ARSLDMA. Score 0.5."),
    "doc9_qa19": (0.5,
        "Post-Termination Services concept identified for PACIRA ARSLDMA. Score 0.5."),
    "doc9_qa20": (0.5,
        "Audit rights concept correct for PACIRA ARSLDMA. Score 0.5."),
    # ── Score 0.25 ───────────────────────────────────────────────────────────
    "doc0_qa2": (0.25,
        "September 9, 1999 vs gold: 7th day of September, 1999. Close but wrong day. Score 0.25."),
    "doc0_qa13": (0.25,
        "Minimum commitment cited in wrong direction (Company fails to deliver vs Distributor must order $250k). Score 0.25."),
    "doc2_qa5": (0.25,
        "General product liability insurance vs gold: 110% invoice value All Risks and War Risk. Wrong type. Score 0.25."),
    "doc3_qa5": (0.25,
        "Annual one-year renewal cited vs gold: one-month renewal periods for Web Site Hosting. Wrong period. Score 0.25."),
    "doc3_qa7": (0.25,
        "Pred: English law vs gold: Florida (Web Site Hosting). Wrong jurisdiction. Score 0.25."),
    "doc3_qa9": (0.25,
        "Clause 10.8 PPI/EKR cited vs gold: i-on limitation of liability (Web Site Hosting). Wrong source. Score 0.25."),
    "doc4_qa2": (0.25,
        "April 8, 2020 vs gold: March 27, 2020 for Joint Filing Agreement date. Wrong date. Score 0.25."),
    "doc5_qa11": (0.25,
        "Dova Section 3.5.2 (Co-Promotion) cited vs CONSULTANT days available (Endorsement). Wrong source. Score 0.25."),
    "doc6_qa2": (0.25,
        "Pred: July 1, 2018 vs gold: July 20, 2018 (Consulting Agreement Date). Wrong day. Score 0.25."),
    "doc6_qa4": (0.25,
        "July 1, 2019 cited as expiration; gold: continues until termination (no fixed end date). Score 0.25."),
    "doc7_qa0": (0.25,
        "JOINT VENTURE AGREEMENT — missing 'AMENDMENT AND TERMINATION OF' prefix. Score 0.25."),
    "doc7_qa5": (0.25,
        "Pred: English law vs gold: Japan for Veoneer/Nissin Amendment. Wrong jurisdiction. Score 0.25."),
    "doc8_qa16": (0.25,
        "Reversed license direction — pred: Valeant grants to Dova vs gold: Dova grants to Valeant. Score 0.25."),
    "doc8_qa18": (0.25,
        "Vague license grant restrictions; gold specifies Valeant grants non-transferable, non-exclusive license to Dova. Score 0.25."),
    "doc8_qa20": (0.25,
        "Section 11.4 general limit cited; gold asks about specific liability cap exclusions (FOREGOING SENTENCE SHALL NOT LIMIT). Score 0.25."),
    "doc9_qa6": (0.25,
        "Fifteen (15) days notice vs gold: 120 days notice for EKR termination right. Wrong number. Score 0.25."),
    "doc9_qa8": (0.25,
        "EKR restriction cited vs gold: PPI (Pacira) competitive restriction. Wrong party. Score 0.25."),
    "doc9_qa14": (0.25,
        "Section 8.1 IP ownership; gold asks about EKR's IP acquisition rights upon termination. Wrong section/context. Score 0.25."),
    "doc9_qa17": (0.25,
        "Clause 2.3 anti-assignment cited vs gold: EKR may appoint sub-distributors. Wrong clause type. Score 0.25."),
    "doc9_qa21": (0.25,
        "IP infringement remedy cited; gold asks about liability cap exception (EKR indemnified party). Wrong clause. Score 0.25."),
}


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def score_calib(r: dict) -> tuple[float, str]:
    pred = r.get("predicted", "")
    behavior = r.get("expected_behavior", "answer")
    cfg = "attention-corpus-tuned"
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
    cfg = "attention-corpus-tuned"
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
    print("Judging cuad__attention-corpus-tuned — calibration + batch_calib ...")
    write_judgments(CALIB_DIR / "queue.jsonl", CALIB_RESULTS, score_calib, "calib")
    write_judgments(BATCH_DIR / "queue.jsonl", BATCH_RESULTS, score_batch, "batch")
    print("Done.")
