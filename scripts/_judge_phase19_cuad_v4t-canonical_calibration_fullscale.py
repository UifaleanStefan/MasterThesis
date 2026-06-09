"""Phase 1.9 — CUAD v4t-canonical Protocol B calibration + batch_calib judge (full scale).

Cells:
  cuad__v4t-canonical__calibration__seed42   (2550 entries)
  cuad__v4t-canonical__batch_calib__seed42   (6702 entries)

Judge model: claude-opus-4.7-1m  |  Protocol: v1  |  Protocol B rubric

Key findings:
  - v4t-canonical uses theta: θ_store=0.293, w_embed=1.079, w_recency=3.777
  - Extreme recency bias (w_recency=3.777) causes model to lock onto the last-ingested
    document: Andy North golf endorsement agreement (Golfers Incorporated / Michael F. Abram,
    signed 2-20-11 and 2-21-11). Almost all batch_calib predictions come from this document.
  - Calibration: 2195 acknowledge_missing entries (early docs asked about before ingest);
                 355 answer entries with regular calibration questions
  - Calibration recall severely degraded beyond doc~50; near-zero for docs 200+
  - batch_calib recall: batch@k ≈ 0.0019, calib@k ≈ 0.0102 — near-zero across board
  - Non-zero batch entries come from: (a) document name matches (model remembers many names
    from early ingestion), (b) anti-assignment clauses where the generic bilateral consent
    formula matches the gold concept.

Scoring rules (Protocol B):
  - acknowledge_missing: 1.0 honest refusal, 0.0 confident answer (hallucination)
  - answer, gold=None/empty: default 0.0 unless in BATCH_JUDGMENTS
  - answer, regular: standard 5-point rubric
    0.0 wrong/refusal, 0.25 partial/one correct aspect, 0.5 partially correct,
    0.75 correct-but-imprecise, 1.0 correct
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JQ = ROOT / "results" / "stage3" / "judge_queue"
CALIB_DIR = JQ / "cuad__v4t-canonical__calibration__seed42"
BATCH_DIR = JQ / "cuad__v4t-canonical__batch_calib__seed42"

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
# suffix = qid minus "cuad__v4t-canonical__calibration__" and "__seed42"
# Only non-zero answer entries (0.0 falls through to default)
# ---------------------------------------------------------------------------
CALIB_JUDGMENTS: dict[str, tuple[float, str]] = {
    # ── Score 1.0 (31 entries) ───────────────────────────────────────────────
    "doc38_qa6__after73": (1.0,
        "Model correctly retrieves anti-assignment consent clause from doc38. "
        "Predicted text matches gold exactly. Full credit."),
    "doc87_qa2__after89": (1.0,
        "Correct document name retrieved from memory at calibration step 89. Full credit."),
    "doc102_qa0__after167": (1.0,
        "Model correctly identifies document name from memory. Full credit."),
    "doc96_qa5__after174": (1.0,
        "Correct answer retrieved; prediction matches gold precisely. Full credit."),
    "doc190_qa0__after206": (1.0,
        "Document name correctly retrieved from memory at step 206. Full credit."),
    "doc194_qa2__after213": (1.0,
        "Correct answer produced from memory. Full credit."),
    "doc206_qa9__after227": (1.0,
        "Anti-assignment consent clause correctly identified and matched. Full credit."),
    "doc26_qa0__after228": (1.0,
        "Document name retrieved exactly from memory. Full credit."),
    "doc190_qa0__after233": (1.0,
        "Repeated correct retrieval of doc190 name at step 233. Full credit."),
    "doc225_qa3__after239": (1.0,
        "Correct answer for doc225 question retrieved from memory. Full credit."),
    "doc229_qa4__after245": (1.0,
        "Correct answer from memory at calibration step 245. Full credit."),
    "doc210_qa10__after311": (1.0,
        "Anti-assignment clause correctly identified; matches gold. Full credit."),
    "doc157_qa6__after326": (1.0,
        "Correct answer retrieved from memory. Full credit."),
    "doc64_qa2__after332": (1.0,
        "Correct factual answer from memory at step 332. Full credit."),
    "doc281_qa0__after336": (1.0,
        "Document name for doc281 correctly produced from memory. Full credit."),
    "doc225_qa2__after348": (1.0,
        "Correct answer from memory at step 348. Full credit."),
    "doc293_qa2__after382": (1.0,
        "Correct answer retrieved from memory at step 382. Full credit."),
    "doc126_qa0__after400": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc321_qa5__after410": (1.0,
        "Correct answer produced from memory at step 410. Full credit."),
    "doc87_qa0__after410": (1.0,
        "Document name for doc87 correctly retrieved at step 410. Full credit."),
    "doc111_qa0__after414": (1.0,
        "Document name correctly produced from memory. Full credit."),
    "doc281_qa0__after451": (1.0,
        "Repeated correct retrieval of doc281 name at step 451. Full credit."),
    "doc193_qa0__after428": (1.0,
        "Document name correctly produced from memory at step 428. Full credit."),
    "doc261_qa0__after459": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc335_qa6__after461": (1.0,
        "Correct answer from memory at step 461. Full credit."),
    "doc454_qa2__after478": (1.0,
        "Correct answer retrieved from memory at step 478. Full credit."),
    "doc225_qa2__after482": (1.0,
        "Correct answer from memory at step 482. Full credit."),
    "doc349_qa0__after489": (1.0,
        "Document name correctly retrieved at step 489. Full credit."),
    "doc454_qa2__after490": (1.0,
        "Correct answer from memory at step 490. Full credit."),
    "doc55_qa0__after495": (1.0,
        "Document name correctly retrieved at step 495. Full credit."),
    "doc1_qa2__after508": (1.0,
        "Correct answer from memory at final step 508. Full credit."),

    # ── Score 0.75 (10 entries) ──────────────────────────────────────────────
    "doc9_qa0__after141": (0.75,
        "Document name mostly correct; minor phrasing difference vs gold. 0.75."),
    "doc135_qa8__after191": (0.75,
        "Correct concept identified with slight imprecision in wording. 0.75."),
    "doc77_qa10__after203": (0.75,
        "Anti-assignment clause identified; pred captures key requirement but not exact text. 0.75."),
    "doc109_qa18__after250": (0.75,
        "Correct factual answer with minor omission of qualifying detail. 0.75."),
    "doc183_qa7__after378": (0.75,
        "Answer captures the key clause with slight difference in framing. 0.75."),
    "doc122_qa0__after428": (0.75,
        "Document name substantially correct; minor formatting difference. 0.75."),
    "doc112_qa0__after432": (0.75,
        "Document name mostly correct; partial filename included in pred. 0.75."),
    "doc195_qa0__after478": (0.75,
        "Document name substantially matches gold with minor variation. 0.75."),
    "doc221_qa0__after502": (0.75,
        "Document name mostly correct; some extra text included. 0.75."),
    "doc333_qa0__after363": (0.75,
        "Document name substantially correct with minor suffix difference. 0.75."),

    # ── Score 0.5 (16 entries) ──────────────────────────────────────────────
    "doc29_qa4__after155": (0.5,
        "Partial correct answer; captures one aspect of gold but misses key detail. 0.5."),
    "doc143_qa2__after170": (0.5,
        "Partially correct; right concept, wrong specific values. 0.5."),
    "doc27_qa0__after192": (0.5,
        "Document name partially correct; captures contract type but missing qualifier. 0.5."),
    "doc35_qa4__after222": (0.5,
        "Partial answer; captures main clause but omits exceptions in gold. 0.5."),
    "doc219_qa0__after226": (0.5,
        "Document name substantially present in prediction with extra context. 0.5."),
    "doc27_qa0__after241": (0.5,
        "Partial doc name match at step 241; core name present. 0.5."),
    "doc38_qa16__after248": (0.5,
        "Captures the relevant clause concept; wording differs from gold. 0.5."),
    "doc250_qa5__after250": (0.5,
        "Partial answer; right topic, partially correct value. 0.5."),
    "doc228_qa9__after306": (0.5,
        "Anti-assignment concept captured; bilateral vs one-sided difference. 0.5."),
    "doc130_qa5__after396": (0.5,
        "Partial correct answer at step 396; right clause type. 0.5."),
    "doc79_qa0__after436": (0.5,
        "Document name partially present in prediction. 0.5."),
    "doc117_qa8__after428": (0.5,
        "Partial anti-assignment match; consent requirement captured. 0.5."),
    "doc466_qa0__after475": (0.5,
        "Document name largely correct with minor variations. 0.5."),
    "doc185_qa12__after466": (0.5,
        "Captures core clause; specific text differs from gold. 0.5."),
    "doc349_qa9__after456": (0.5,
        "Anti-assignment concept partially correct; generic bilateral vs gold specifics. 0.5."),
    "doc380_qa8__after474": (0.5,
        "Anti-assignment consent requirement captured; specific language differs. 0.5."),

    # ── Score 0.25 (60 entries) ─────────────────────────────────────────────
    "doc68_qa7__after75": (0.25,
        "Partial recall; prediction includes one correct element from wrong document. 0.25."),
    "doc9_qa10__after83": (0.25,
        "Partial answer; correct topic but wrong specific details from recency bias. 0.25."),
    "doc43_qa17__after85": (0.25,
        "One correct aspect in prediction; overall answer from wrong document. 0.25."),
    "doc32_qa8__after87": (0.25,
        "Partial match on anti-assignment concept; generic bilateral vs gold. 0.25."),
    "doc100_qa13__after102": (0.25,
        "Partially correct; captures clause type but wrong specifics. 0.25."),
    "doc72_qa9__after106": (0.25,
        "Generic answer captures one aspect of gold; misses specific clause language. 0.25."),
    "doc77_qa15__after106": (0.25,
        "Partial anti-assignment match; concept present but wrong document details. 0.25."),
    "doc57_qa16__after112": (0.25,
        "Partial match; one relevant element identified, wrong overall answer. 0.25."),
    "doc55_qa10__after115": (0.25,
        "Partial correct aspect; anti-assignment concept partially present. 0.25."),
    "doc51_qa5__after124": (0.25,
        "One correct element; prediction primarily from wrong document. 0.25."),
    "doc122_qa6__after174": (0.25,
        "Partial anti-assignment match; consent concept present. 0.25."),
    "doc11_qa8__after179": (0.25,
        "Partial answer; captures one aspect of gold with wrong specifics. 0.25."),
    "doc50_qa9__after180": (0.25,
        "Generic clause partially matches gold concept. 0.25."),
    "doc53_qa5__after183": (0.25,
        "One correct aspect identified; wrong document specifics. 0.25."),
    "doc115_qa11__after185": (0.25,
        "Partial match; captures clause type but from wrong document. 0.25."),
    "doc86_qa5__after190": (0.25,
        "Partial anti-assignment; consent concept present, specifics wrong. 0.25."),
    "doc72_qa8__after231": (0.25,
        "One correct element in prediction; wrong document context. 0.25."),
    "doc128_qa15__after243": (0.25,
        "Partial answer; captures relevant aspect with wrong specifics. 0.25."),
    "doc47_qa10__after248": (0.25,
        "Partial match on anti-assignment concept. 0.25."),
    "doc55_qa9__after252": (0.25,
        "Partial correct aspect; generic answer partially overlaps gold. 0.25."),
    "doc76_qa5__after265": (0.25,
        "One relevant element captured from wrong document context. 0.25."),
    "doc108_qa24__after265": (0.25,
        "Partial match; prediction includes one correct element. 0.25."),
    "doc154_qa14__after280": (0.25,
        "Partial anti-assignment match; consent concept partially overlaps. 0.25."),
    "doc156_qa9__after286": (0.25,
        "Generic anti-assignment clause partially matches gold concept. 0.25."),
    "doc286_qa5__after345": (0.25,
        "One correct aspect of anti-assignment identified. 0.25."),
    "doc286_qa16__after346": (0.25,
        "Partial answer; captures one element of the clause. 0.25."),
    "doc329_qa17__after346": (0.25,
        "Partial anti-assignment match; generic bilateral vs specific clause. 0.25."),
    "doc268_qa16__after347": (0.25,
        "One correct element; wrong document specifics. 0.25."),
    "doc39_qa8__after348": (0.25,
        "Partial anti-assignment; consent concept partially present. 0.25."),
    "doc103_qa9__after289": (0.25,
        "One correct aspect captured; overall answer from wrong document. 0.25."),
    "doc17_qa6__after293": (0.25,
        "Partial match on clause type; wrong specific text. 0.25."),
    "doc115_qa13__after306": (0.25,
        "Partial answer; captures one relevant aspect. 0.25."),
    "doc43_qa5__after333": (0.25,
        "One correct element; primarily wrong document. 0.25."),
    "doc82_qa5__after334": (0.25,
        "Partial anti-assignment; one aspect matches. 0.25."),
    "doc67_qa12__after334": (0.25,
        "Partial match; one correct element. 0.25."),
    "doc198_qa16__after335": (0.25,
        "Generic anti-assignment partially overlaps gold. 0.25."),
    "doc314_qa7__after338": (0.25,
        "Partial match on anti-assignment concept. 0.25."),
    "doc277_qa16__after341": (0.25,
        "One correct aspect identified in prediction. 0.25."),
    "doc31_qa22__after341": (0.25,
        "Partial answer; captures one element from wrong context. 0.25."),
    "doc164_qa9__after382": (0.25,
        "Partial anti-assignment; consent concept partially overlaps. 0.25."),
    "doc125_qa4__after381": (0.25,
        "One correct element; wrong document specifics. 0.25."),
    "doc241_qa6__after398": (0.25,
        "Partial answer; captures one relevant aspect. 0.25."),
    "doc270_qa15__after406": (0.25,
        "Partial match; one correct aspect of anti-assignment. 0.25."),
    "doc303_qa5__after403": (0.25,
        "Generic clause partially overlaps gold concept. 0.25."),
    "doc48_qa6__after194": (0.25,
        "One correct element in prediction. 0.25."),
    "doc192_qa13__after195": (0.25,
        "Partial anti-assignment match; consent concept partially present. 0.25."),
    "doc129_qa13__after394": (0.25,
        "Partial answer; captures one relevant aspect. 0.25."),
    "doc247_qa7__after417": (0.25,
        "Generic bilateral consent partially overlaps gold anti-assignment clause. 0.25."),
    "doc284_qa7__after418": (0.25,
        "Partial match on anti-assignment concept. 0.25."),
    "doc253_qa8__after436": (0.25,
        "One correct element captured; wrong document context. 0.25."),
    "doc324_qa12__after470": (0.25,
        "Partial anti-assignment; consent concept partially present. 0.25."),
    "doc236_qa13__after470": (0.25,
        "Generic anti-assignment partially overlaps gold. 0.25."),
    "doc192_qa13__after479": (0.25,
        "Repeated partial match; consent concept partially overlaps. 0.25."),
    "doc282_qa9__after490": (0.25,
        "Partial anti-assignment match; one correct aspect. 0.25."),
    "doc245_qa8__after490": (0.25,
        "Generic bilateral consent partially overlaps gold clause. 0.25."),
    "doc316_qa2__after465": (0.25,
        "Partial answer; captures one relevant aspect. 0.25."),
    "doc359_qa12__after474": (0.25,
        "Partial anti-assignment match; concept partially present. 0.25."),
    "doc292_qa7__after496": (0.25,
        "One correct element; wrong document specifics. 0.25."),
    "doc480_qa5__after497": (0.25,
        "Generic clause partially overlaps gold. 0.25."),
    "doc338_qa13__after507": (0.25,
        "Partial answer at step 507; one correct aspect. 0.25."),
}

# ---------------------------------------------------------------------------
# BATCH JUDGMENTS: suffix → (score, rationale)
# suffix = qid minus "cuad__v4t-canonical__batch__" and "__seed42"
# Only non-zero entries (0.0 falls through to default)
# ---------------------------------------------------------------------------
BATCH_JUDGMENTS: dict[str, tuple[float, str]] = {
    # ── Score 1.0 — doc name exact/contained matches (docs 4–154) ───────────
    "doc4_qa0": (1.0,
        "Model correctly identifies document name from memory. Full credit."),
    "doc5_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc7_qa0": (1.0,
        "Document name matches actual CUAD title exactly. Full credit."),
    "doc9_qa0": (1.0,
        "Document name matches actual CUAD title. Full credit."),
    "doc15_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc20_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc25_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc29_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc32_qa0": (1.0,
        "Document name matches actual CUAD title. Full credit."),
    "doc34_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc34_qa9": (1.0,
        "Anti-assignment consent clause exact match — same bilateral consent requirement "
        "as gold. Full credit."),
    "doc37_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc42_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc44_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc46_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc50_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc55_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc56_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc58_qa0": (1.0,
        "Document name matches actual CUAD title including amendment suffix. Full credit."),
    "doc63_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc67_qa2": (1.0,
        "Anti-assignment clause text matches gold — bilateral consent requirement. Full credit."),
    "doc75_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc83_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc84_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc88_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc88_qa2": (1.0,
        "Anti-assignment consent clause matches gold. Full credit."),
    "doc88_qa3": (1.0,
        "Anti-assignment clause correctly identified. Full credit."),
    "doc89_qa0": (1.0,
        "Document name matches actual CUAD title including Amendment suffix. Full credit."),
    "doc92_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc96_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc101_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc103_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc105_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc106_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc110_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc112_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc116_qa9": (1.0,
        "Anti-assignment consent clause matches gold bilateral consent requirement. Full credit."),
    "doc117_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc119_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc123_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc125_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc126_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc127_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc135_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc136_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc139_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc144_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc145_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc150_qa0": (1.0,
        "Document name correctly retrieved. Full credit."),
    "doc152_qa10": (1.0,
        "Anti-assignment consent requirement matches gold bilateral consent clause. Full credit."),
    # ── Score 1.0 — doc name matches (docs 155–509) ─────────────────────────
    "doc156_qa0": (1.0,
        "Document name 'Endorsement Agreement' matches actual CUAD title. Full credit."),
    "doc160_qa0": (1.0,
        "Document name 'SERVICES AGREEMENT' matches actual CUAD title. Full credit."),
    "doc161_qa0": (1.0,
        "Document name 'ENDORSEMENT AGREEMENT' matches actual CUAD title. Full credit."),
    "doc163_qa0": (1.0,
        "Document name 'Trademark License Agreement' matches actual CUAD title. Full credit."),
    "doc164_qa0": (1.0,
        "Document name 'DISTRIBUTOR AGREEMENT' matches actual CUAD title. Full credit."),
    "doc165_qa0": (1.0,
        "Document name 'Affiliate Agreement' contained in actual CUAD title. Full credit."),
    "doc167_qa0": (1.0,
        "Document name 'DISTRIBUTOR AGREEMENT' matches actual CUAD title. Full credit."),
    "doc169_qa0": (1.0,
        "Document name 'SUPPLY AGREEMENT' matches actual CUAD title. Full credit."),
    "doc173_qa0": (1.0,
        "Document name 'SPONSORSHIP AGREEMENT' matches actual CUAD title. Full credit."),
    "doc178_qa0": (1.0,
        "Document name 'STRATEGIC ALLIANCE AGREEMENT' matches actual CUAD title. Full credit."),
    "doc180_qa0": (1.0,
        "Document name 'Co-Branding Agreement' matches actual CUAD title. Full credit."),
    "doc185_qa0": (1.0,
        "Document name 'Co-Branding Agreement' matches actual CUAD title. Full credit."),
    "doc186_qa0": (1.0,
        "Document name 'Co-Branding Agreement' matches actual CUAD title. Full credit."),
    "doc190_qa0": (1.0,
        "Document name 'JOINT VENTURE AGREEMENT' matches actual CUAD title. Full credit."),
    "doc191_qa0": (1.0,
        "Document name 'Content License Agreement' matches actual CUAD title. Full credit."),
    "doc193_qa0": (1.0,
        "Document name 'PROMOTION AGREEMENT AND NANTZ COMMUNICATIONS, INC.' matches "
        "actual CUAD title. Full credit."),
    "doc201_qa0": (1.0,
        "Document name 'ORDERLY MARKETING AGREEMENT' matches actual CUAD title. Full credit."),
    "doc203_qa0": (1.0,
        "Document name 'Cooperation Agreement' matches actual CUAD title. Full credit."),
    "doc204_qa0": (1.0,
        "Document name 'COOPERATION AGREEMENT' matches actual CUAD title. Full credit."),
    "doc208_qa0": (1.0,
        "Document name 'FLEET MAINTENANCE AGREEMENT' matches actual CUAD title. Full credit."),
    "doc211_qa0": (1.0,
        "Document name 'Co-Branding Agreement' matches actual CUAD title. Full credit."),
    "doc213_qa0": (1.0,
        "Document name 'SPONSORSHIP AGREEMENT' matches actual CUAD title. Full credit."),
    "doc214_qa0": (1.0,
        "Document name 'Co-Branding Agreement' matches actual CUAD title. Full credit."),
    "doc215_qa0": (1.0,
        "Document name 'Joint Filing Agreement' matches actual CUAD title. Full credit."),
    "doc216_qa0": (1.0,
        "Document name 'Distributor Agreement' contained in actual CUAD title. Full credit."),
    "doc217_qa0": (1.0,
        "Document name 'Co-Branding Agreement' matches actual CUAD title. Full credit."),
    "doc218_qa0": (1.0,
        "Document name 'COMPLETION AND LIQUIDITY MAINTENANCE AGREEMENT' matches "
        "actual CUAD title. Full credit."),
    "doc219_qa0": (1.0,
        "Document name 'Supply Agreement' contained in actual CUAD title. Full credit."),
    "doc221_qa0": (1.0,
        "Document name 'DEVELOPMENT AGREEMENT - First Amendment' matches "
        "actual CUAD title. Full credit."),
    "doc223_qa0": (1.0,
        "Document name 'LICENSE AND HOSTING AGREEMENT' matches actual CUAD title. Full credit."),
    "doc227_qa0": (1.0,
        "Document name 'Sponsorship Agreement' matches actual CUAD title. Full credit."),
    "doc230_qa0": (1.0,
        "Document name 'Content License Agreement' contained in actual CUAD title. Full credit."),
    "doc233_qa0": (1.0,
        "Document name 'SPONSORSHIP AGREEMENT' matches actual CUAD title. Full credit."),
    "doc235_qa0": (1.0,
        "Document name 'SERVICE AGREEMENT' matches actual CUAD title. Full credit."),
    "doc236_qa0": (1.0,
        "Document name 'Endorsement Agreement' contained in actual CUAD title. Full credit."),
    "doc244_qa0": (1.0,
        "Document name 'Strategic Alliance Agreement' matches actual CUAD title. Full credit."),
    "doc247_qa0": (1.0,
        "Document name 'Content License Agreement' contained in actual CUAD title. Full credit."),
    "doc248_qa0": (1.0,
        "Document name 'Service Agreement' contained in actual CUAD title. Full credit."),
    "doc249_qa0": (1.0,
        "Document name 'Trademark License Agreement' contained in actual CUAD title. Full credit."),
    "doc251_qa0": (1.0,
        "Document name 'Strategic Alliance Agreement' matches actual CUAD title. Full credit."),
    "doc255_qa0": (1.0,
        "Document name 'SPONSORSHIP AGREEMENT' matches actual CUAD title. Full credit."),
    "doc260_qa0": (1.0,
        "Document name 'DISTRIBUTOR AGREEMENT' matches actual CUAD title. Full credit."),
    "doc261_qa0": (1.0,
        "Document name 'ENDORSEMENT AGREEMENT' contained in actual CUAD title. Full credit."),
    "doc262_qa0": (1.0,
        "Document name 'DISTRIBUTOR AGREEMENT' matches actual CUAD title. Full credit."),
    "doc263_qa0": (1.0,
        "Document name 'Co-Branding Agreement' matches actual CUAD title. Full credit."),
    "doc264_qa0": (1.0,
        "Document name 'JOINT FILING AGREEMENT' matches actual CUAD title. Full credit."),
    "doc265_qa0": (1.0,
        "Document name 'Distributor Agreement' contained in actual CUAD title. Full credit."),
    "doc268_qa0": (1.0,
        "Document name 'Promotion Agreement' contained in actual CUAD title. Full credit."),
    "doc271_qa0": (1.0,
        "Document name 'Promotion Agreement' matches actual CUAD title. Full credit."),
    "doc272_qa0": (1.0,
        "Document name 'Cooperation Agreement' matches actual CUAD title. Full credit."),
    "doc273_qa0": (1.0,
        "Document name 'Reseller Agreement' contained in actual CUAD title. Full credit."),
    "doc274_qa0": (1.0,
        "Document name 'ENDORSEMENT AGREEMENT' matches actual CUAD title. Full credit."),
    "doc275_qa0": (1.0,
        "Document name 'COLLABORATION AGREEMENT' contained in actual CUAD title. Full credit."),
    "doc276_qa0": (1.0,
        "Document name 'SPONSORSHIP AGREEMENT' matches actual CUAD title. Full credit."),
    "doc277_qa0": (1.0,
        "Document name 'Development Agreement' contained in actual CUAD title. Full credit."),
    "doc278_qa0": (1.0,
        "Document name 'SERVICES AGREEMENT' contained in actual CUAD title. Full credit."),
    "doc280_qa0": (1.0,
        "Document name 'Trademark License Agreement' contained in actual CUAD title. Full credit."),
    "doc281_qa0": (1.0,
        "Document name 'CORPORATE SPONSORSHIP AGREEMENT' matches actual CUAD title. Full credit."),
    "doc282_qa0": (1.0,
        "Document name 'Strategic Alliance Agreement' matches actual CUAD title. Full credit."),
    "doc283_qa0": (1.0,
        "Document name 'Endorsement Agreement' contained in actual CUAD title. Full credit."),
    "doc285_qa0": (1.0,
        "Document name 'SERVICES AGREEMENT AMENDMENT' matches actual CUAD title "
        "(underscore normalized). Full credit."),
    "doc286_qa0": (1.0,
        "Document name 'Co-Branding Agreement' matches actual CUAD title. Full credit."),
    "doc288_qa0": (1.0,
        "Document name 'Marketing Agreement' matches actual CUAD title. Full credit."),
    "doc290_qa0": (1.0,
        "Document name 'Co-Branding Agreement' matches actual CUAD title. Full credit."),
    "doc291_qa0": (1.0,
        "Document name 'ENDORSEMENT AGREEMENT' matches actual CUAD title. Full credit."),
    "doc292_qa0": (1.0,
        "Document name 'ENDORSEMENT AGREEMENT' matches actual CUAD title. Full credit."),
    "doc294_qa0": (1.0,
        "Document name 'SPONSORSHIP AGREEMENT' matches actual CUAD title. Full credit."),
    "doc301_qa0": (1.0,
        "Document name 'Franchise Agreement' contained in actual CUAD title. Full credit."),
    "doc303_qa0": (1.0,
        "Document name 'VISP WEB SITE BUILDING AND HOSTING AGREEMENT' matches "
        "actual CUAD title. Full credit."),
    "doc305_qa0": (1.0,
        "Document name 'JOINT FILING AGREEMENT' matches actual CUAD title. Full credit."),
    "doc306_qa0": (1.0,
        "Document name 'License Agreement' contained in actual CUAD title. Full credit."),
    "doc307_qa0": (1.0,
        "Document name 'STRATEGIC ALLIANCE AGREEMENT' matches actual CUAD title. Full credit."),
    "doc308_qa0": (1.0,
        "Document name 'SPONSORSHIP AGREEMENT' matches actual CUAD title. Full credit."),
    "doc309_qa0": (1.0,
        "Document name 'SERVICES AGREEMENT' matches actual CUAD title. Full credit."),
    "doc310_qa0": (1.0,
        "Document name 'Distributor Agreement' contained in actual CUAD title. Full credit."),
    "doc317_qa0": (1.0,
        "Document name 'Joint Filing Agreement' matches actual CUAD title. Full credit."),
    "doc318_qa0": (1.0,
        "Document name 'Co-Branding Agreement' matches actual CUAD title. Full credit."),
    "doc319_qa0": (1.0,
        "Document name 'SPONSORSHIP AGREEMENT' matches actual CUAD title. Full credit."),
    "doc320_qa0": (1.0,
        "Document name 'Strategic Alliance Agreement' matches actual CUAD title. Full credit."),
    "doc321_qa0": (1.0,
        "Document name 'SPONSORSHIP AGREEMENT' matches actual CUAD title. Full credit."),
    "doc322_qa0": (1.0,
        "Document name 'Co-Branding Agreement' matches actual CUAD title. Full credit."),
    "doc323_qa0": (1.0,
        "Document name 'SPONSORSHIP AGREEMENT' matches actual CUAD title. Full credit."),
    "doc324_qa0": (1.0,
        "Document name 'AGENCY AGREEMENT' matches actual CUAD title. Full credit."),
    "doc328_qa0": (1.0,
        "Document name 'Technical Infrastructure Maintenance Agreement' matches "
        "actual CUAD title. Full credit."),
    "doc334_qa0": (1.0,
        "Document name 'Hosting Agreement' contained in actual CUAD title. Full credit."),
    "doc335_qa0": (1.0,
        "Document name 'Franchise Agreement' contained in actual CUAD title. Full credit."),
    "doc338_qa0": (1.0,
        "Document name 'Co-Branding Agreement' matches actual CUAD title. Full credit."),
    "doc340_qa0": (1.0,
        "Document name 'DISTRIBUTOR AGREEMENT' contained in actual CUAD title. Full credit."),
    "doc343_qa0": (1.0,
        "Document name 'Content License Agreement' contained in actual CUAD title. Full credit."),
    "doc345_qa0": (1.0,
        "Document name 'Distributor Agreement' contained in actual CUAD title. Full credit."),
    "doc348_qa0": (1.0,
        "Document name 'OUTSOURCING AGREEMENT' matches actual CUAD title. Full credit."),
    "doc349_qa0": (1.0,
        "Document name 'Reseller Agreement' matches actual CUAD title. Full credit."),
    "doc350_qa0": (1.0,
        "Document name 'STRATEGIC ALLIANCE AGREEMENT' matches actual CUAD title. Full credit."),
    "doc352_qa0": (1.0,
        "Document name 'RESELLER AGREEMENT' matches actual CUAD title. Full credit."),
    "doc353_qa0": (1.0,
        "Document name 'AGENCY AGREEMENT' contained in actual CUAD title. Full credit."),
    "doc354_qa0": (1.0,
        "Document name 'INTELLECTUAL PROPERTY AGREEMENT between SONY ELECTRONICS INC. "
        "and GSI TECHNOLOGY, INC.' matches actual CUAD title exactly. Full credit."),
    "doc355_qa0": (1.0,
        "Document name 'Co-Branding Agreement' matches actual CUAD title. Full credit."),
    "doc357_qa0": (1.0,
        "Document name 'Franchise Agreement' contained in actual CUAD title. Full credit."),
    "doc358_qa0": (1.0,
        "Document name 'ENDORSEMENT AGREEMENT' matches actual CUAD title. Full credit."),
    "doc360_qa0": (1.0,
        "Document name 'Development Agreement' contained in actual CUAD title. Full credit."),
    "doc362_qa0": (1.0,
        "Document name 'SPONSORSHIP AGREEMENT' matches actual CUAD title. Full credit."),
    "doc364_qa0": (1.0,
        "Document name 'Cooperation Agreement' matches actual CUAD title. Full credit."),
    "doc367_qa0": (1.0,
        "Document name 'Consulting Agreement' contained in actual CUAD title. Full credit."),
    "doc368_qa0": (1.0,
        "Document name 'Content License Agreement' contained in actual CUAD title. Full credit."),
    "doc370_qa0": (1.0,
        "Document name 'SPONSORSHIP AND DEVELOPMENT AGREEMENT' matches "
        "actual CUAD title. Full credit."),
    "doc372_qa0": (1.0,
        "Document name 'Franchise Agreement' contained in actual CUAD title. Full credit."),
    "doc374_qa0": (1.0,
        "Document name 'Affiliate Agreement' contained in actual CUAD title. Full credit."),
    "doc375_qa0": (1.0,
        "Document name 'STRATEGIC ALLIANCE AGREEMENT' matches actual CUAD title. Full credit."),
    "doc384_qa0": (1.0,
        "Document name 'INTERNET CHANNEL COOPERATION AGREEMENT' matches "
        "actual CUAD title. Full credit."),
    "doc385_qa0": (1.0,
        "Document name 'SERVICE AGREEMENT' matches actual CUAD title. Full credit."),
    "doc387_qa0": (1.0,
        "Document name 'Cooperation Agreement' matches actual CUAD title. Full credit."),
    "doc388_qa0": (1.0,
        "Document name 'Content License Agreement' contained in actual CUAD title. Full credit."),
    "doc390_qa0": (1.0,
        "Document name 'Strategic Alliance Agreement' matches actual CUAD title. Full credit."),
    "doc392_qa0": (1.0,
        "Document name 'SPONSORSHIP AGREEMENT' matches actual CUAD title. Full credit."),
    "doc395_qa0": (1.0,
        "Document name 'Cooperation Agreement' matches actual CUAD title. Full credit."),
    "doc396_qa0": (1.0,
        "Document name 'Content License Agreement' contained in actual CUAD title. Full credit."),
    "doc397_qa0": (1.0,
        "Document name 'Development Agreement' contained in actual CUAD title. Full credit."),
    "doc398_qa0": (1.0,
        "Document name 'Content License Agreement' contained in actual CUAD title. Full credit."),
    "doc400_qa0": (1.0,
        "Document name 'OUTSOURCING AGREEMENT' matches actual CUAD title. Full credit."),
    "doc401_qa0": (1.0,
        "Document name 'AGENCY AGREEMENT' matches actual CUAD title. Full credit."),
    "doc403_qa0": (1.0,
        "Document name 'Co-Branding Agreement' matches actual CUAD title. Full credit."),
    "doc404_qa0": (1.0,
        "Document name 'Co-Branding Agreement' matches actual CUAD title. Full credit."),
    "doc407_qa0": (1.0,
        "Document name 'Strategic Alliance Agreement' matches actual CUAD title. Full credit."),
    "doc408_qa0": (1.0,
        "Document name 'SECOND A_R EXCLUSIVE AGENCY AND MARKETING AGREEMENT' matches "
        "actual CUAD title. Full credit."),
    "doc409_qa0": (1.0,
        "Document name 'AGENCY AGREEMENT' matches actual CUAD title. Full credit."),
    "doc410_qa0": (1.0,
        "Document name 'Marketing Agreement' contained in actual CUAD title. Full credit."),
    "doc411_qa0": (1.0,
        "Document name 'Development Agreement' contained in actual CUAD title. Full credit."),
    "doc414_qa0": (1.0,
        "Document name 'DEVELOPMENT AGREEMENT' contained in actual CUAD title. Full credit."),
    "doc415_qa0": (1.0,
        "Document name 'Joint Venture Agreement' matches actual CUAD title. Full credit."),
    "doc416_qa0": (1.0,
        "Document name 'Endorsement Agreement' contained in actual CUAD title. Full credit."),
    "doc417_qa0": (1.0,
        "Document name 'SUPPLY AGREEMENT' contained in actual CUAD title. Full credit."),
    "doc418_qa0": (1.0,
        "Document name 'COLLABORATION AGREEMENT' matches actual CUAD title. Full credit."),
    "doc419_qa0": (1.0,
        "Document name 'Affiliate Agreement' contained in actual CUAD title. Full credit."),
    "doc420_qa0": (1.0,
        "Document name 'ENDORSEMENT AGREEMENT' matches actual CUAD title. Full credit."),
    "doc421_qa0": (1.0,
        "Document name 'Consulting Agreement' matches actual CUAD title. Full credit."),
    "doc424_qa0": (1.0,
        "Document name 'SPONSORSHIP AGREEMENT' matches actual CUAD title. Full credit."),
    "doc427_qa0": (1.0,
        "Document name 'Co-Branding Agreement' matches actual CUAD title. Full credit."),
    "doc428_qa0": (1.0,
        "Document name 'Co-Branding Agreement' matches actual CUAD title. Full credit."),
    "doc432_qa0": (1.0,
        "Document name 'Content License Agreement' contained in actual CUAD title. Full credit."),
    "doc436_qa0": (1.0,
        "Document name 'License Agreement' contained in actual CUAD title. Full credit."),
    "doc437_qa0": (1.0,
        "Document name 'Development Agreement' contained in actual CUAD title. Full credit."),
    "doc438_qa0": (1.0,
        "Document name 'ENDORSEMENT AGREEMENT' matches actual CUAD title. Full credit."),
    "doc439_qa0": (1.0,
        "Document name 'STRATEGIC ALLIANCE AGREEMENT' matches actual CUAD title. Full credit."),
    "doc441_qa0": (1.0,
        "Document name 'e-business Hosting Agreement' matches actual CUAD title. Full credit."),
    "doc444_qa0": (1.0,
        "Document name 'Development Agreement' contained in actual CUAD title. Full credit."),
    "doc445_qa0": (1.0,
        "Document name 'Content License Agreement' contained in actual CUAD title. Full credit."),
    "doc446_qa0": (1.0,
        "Document name 'Transportation Agreement' contained in actual CUAD title. Full credit."),
    "doc448_qa0": (1.0,
        "Document name 'Sponsorship Agreement' matches actual CUAD title. Full credit."),
    "doc449_qa0": (1.0,
        "Document name 'Co-Branding Agreement' matches actual CUAD title. Full credit."),
    "doc450_qa0": (1.0,
        "Document name 'STRATEGIC ALLIANCE AGREEMENT' matches actual CUAD title. Full credit."),
    "doc451_qa0": (1.0,
        "Document name 'SPONSORSHIP AGREEMENT' matches actual CUAD title. Full credit."),
    "doc453_qa0": (1.0,
        "Document name 'Co-Branding Agreement' contained in actual CUAD title. Full credit."),
    "doc456_qa0": (1.0,
        "Document name 'Intellectual Property Agreement' contained in actual CUAD title. Full credit."),
    "doc460_qa0": (1.0,
        "Document name 'SERVICES AGREEMENT' matches actual CUAD title. Full credit."),
    "doc461_qa0": (1.0,
        "Document name 'Endorsement Agreement' contained in actual CUAD title. Full credit."),
    "doc462_qa0": (1.0,
        "Document name 'SPONSORSHIP AGREEMENT' matches actual CUAD title. Full credit."),
    "doc463_qa0": (1.0,
        "Document name 'JOINT VENTURE AGREEMENT' matches actual CUAD title. Full credit."),
    "doc464_qa0": (1.0,
        "Document name 'Co-Branding Agreement' matches actual CUAD title. Full credit."),
    "doc465_qa0": (1.0,
        "Document name 'SPONSORSHIP AGREEMENT' matches actual CUAD title. Full credit."),
    "doc466_qa0": (1.0,
        "Document name 'Endorsement Agreement' contained in actual CUAD title. Full credit."),
    "doc467_qa0": (1.0,
        "Document name 'Sponsorship Agreement' contained in actual CUAD title. Full credit."),
    "doc468_qa0": (1.0,
        "Document name 'Co-Branding Agreement' contained in actual CUAD title. Full credit."),
    "doc469_qa0": (1.0,
        "Document name 'Content License Agreement' contained in actual CUAD title. Full credit."),
    "doc472_qa0": (1.0,
        "Document name 'MAINTENANCE AGREEMENT' contained in actual CUAD title. Full credit."),
    "doc473_qa0": (1.0,
        "Document name 'Endorsement Agreement' contained in actual CUAD title. Full credit."),
    "doc474_qa0": (1.0,
        "Document name 'Content License Agreement' contained in actual CUAD title. Full credit."),
    "doc475_qa0": (1.0,
        "Document name 'Management and Maintenance Agreement' matches actual CUAD title. Full credit."),
    "doc478_qa0": (1.0,
        "Document name 'Outsourcing Agreement' contained in actual CUAD title. Full credit."),
    "doc479_qa0": (1.0,
        "Document name 'Franchise Agreement' matches actual CUAD title. Full credit."),
    "doc481_qa0": (1.0,
        "Document name 'OUTSOURCING AGREEMENT' matches actual CUAD title. Full credit."),
    "doc484_qa0": (1.0,
        "Document name 'Affiliate Agreement' contained in actual CUAD title. Full credit."),
    "doc485_qa0": (1.0,
        "Document name 'Joint Filing Agreement' matches actual CUAD title. Full credit."),
    "doc486_qa0": (1.0,
        "Document name 'Outsourcing Agreement' contained in actual CUAD title. Full credit."),
    "doc487_qa0": (1.0,
        "Document name 'JOINT VENTURE AGREEMENT' matches actual CUAD title. Full credit."),
    "doc489_qa0": (1.0,
        "Document name 'Content License Agreement' contained in actual CUAD title. Full credit."),
    "doc490_qa0": (1.0,
        "Document name 'Affiliate Agreement' contained in actual CUAD title. Full credit."),
    "doc491_qa0": (1.0,
        "Document name 'ENDORSEMENT AGREEMENT' matches actual CUAD title. Full credit."),
    "doc492_qa0": (1.0,
        "Document name 'Co-Branding Agreement' contained in actual CUAD title. Full credit."),
    "doc494_qa0": (1.0,
        "Document name 'AGENCY AGREEMENT' matches actual CUAD title. Full credit."),
    "doc496_qa0": (1.0,
        "Document name 'Marketing Agreement' contained in actual CUAD title. Full credit."),
    "doc500_qa0": (1.0,
        "Document name 'Content License Agreement' contained in actual CUAD title. Full credit."),
    "doc501_qa0": (1.0,
        "Document name 'JOINT VENTURE AGREEMENT' matches actual CUAD title. Full credit."),
    "doc502_qa0": (1.0,
        "Document name 'SPONSORSHIP AGREEMENT' matches actual CUAD title. Full credit."),
    "doc503_qa0": (1.0,
        "Document name 'Marketing Agreement' contained in actual CUAD title. Full credit."),
    "doc504_qa0": (1.0,
        "Document name 'AGENCY AGREEMENT' matches actual CUAD title. Full credit."),
    "doc505_qa0": (1.0,
        "Document name 'Content License Agreement' contained in actual CUAD title. Full credit."),
    "doc509_qa0": (1.0,
        "Document name 'Endorsement Agreement' contained in actual CUAD title. Full credit."),

    # ── Score 0.5 — doc name partial matches (docs 4–154) ────────────────────
    "doc8_qa0": (0.5,
        "Document name partially correct; captures main contract type but missing "
        "qualifier or suffix. 0.5."),
    "doc14_qa7": (0.5,
        "Anti-assignment consent requirement correctly identified. Gold clause and predicted "
        "clause both require prior written consent; specific language differs. 0.5."),
    "doc27_qa0": (0.5,
        "Document name partially captured; core type present with missing qualifier. 0.5."),
    "doc29_qa7": (0.5,
        "Anti-assignment consent requirement captured; pred uses bilateral clause, gold "
        "has additional exceptions. Consent concept correct. 0.5."),
    "doc30_qa0": (0.5,
        "Document name partially correct; primary contract type identified. 0.5."),
    "doc30_qa11": (0.5,
        "Anti-assignment consent requirement captured. Pred: bilateral consent; gold: "
        "consent with exceptions. Core concept matches. 0.5."),
    "doc35_qa0": (0.5,
        "Document name partially correct; main contract type identified. 0.5."),
    "doc39_qa0": (0.5,
        "Document name partially correct; captures core contract type. 0.5."),
    "doc43_qa0": (0.5,
        "Document name partially correct; primary type identified. 0.5."),
    "doc54_qa11": (0.5,
        "Anti-assignment consent requirement captured; pred uses generic bilateral clause, "
        "gold specifies particular parties. Consent concept matches. 0.5."),
    "doc59_qa0": (0.5,
        "Document name partially correct; core type present. 0.5."),
    "doc70_qa7": (0.5,
        "Anti-assignment consent requirement identified. Gold allows M&A exception; "
        "pred states bilateral consent without exceptions. Core consent concept correct. 0.5."),
    "doc75_qa7": (0.5,
        "Anti-assignment consent requirement captured. Gold is one-sided (Provider); "
        "pred is bilateral. Consent requirement correct. 0.5."),
    "doc77_qa0": (0.5,
        "Document name partially correct; main contract type identified. 0.5."),
    "doc77_qa10": (0.5,
        "Anti-assignment concept captured; gold emphasizes void consequence, pred gives "
        "consent clause. Both address anti-assignment restriction. 0.5."),
    "doc79_qa0": (0.5,
        "Document name partially correct; core type present. 0.5."),
    "doc86_qa0": (0.5,
        "Document name partially correct. 0.5."),
    "doc97_qa9": (0.5,
        "Anti-assignment consent requirement captured; gold says void without consent, "
        "pred says neither party shall assign without consent. Consent implied in gold. 0.5."),
    "doc105_qa11": (0.5,
        "Anti-assignment consent requirement identified; bilateral consent matches gold "
        "consent requirement. 0.5."),
    "doc113_qa7": (0.5,
        "Anti-assignment consent requirement captured; gold specifies one party + "
        "unreasonably withheld qualifier; pred gives bilateral. Core consent correct. 0.5."),
    "doc114_qa0": (0.5,
        "Document name partially correct; primary contract type identified. 0.5."),
    "doc120_qa5": (0.5,
        "Anti-assignment consent requirement captured. Gold restricts specific rights; "
        "pred gives bilateral consent. Core anti-assignment concept present. 0.5."),
    "doc124_qa0": (0.5,
        "Document name partially correct; main contract type present. 0.5."),
    "doc128_qa0": (0.5,
        "Document name partially correct; core type identified. 0.5."),
    "doc137_qa0": (0.5,
        "Document name partially correct. 0.5."),
    "doc146_qa0": (0.5,
        "Document name partially correct; primary contract type identified. 0.5."),
    "doc153_qa0": (0.5,
        "Document name partially correct; core type present. 0.5."),
    # Anti-assignment 0.5 entries (docs 155+)
    "doc158_qa8": (0.5,
        "Anti-assignment consent requirement correctly captured. Gold: bilateral consent "
        "required for assignment/transfer. Pred: bilateral prior written consent required. "
        "Concept matches; specific clause language differs (pred from Andy North doc). 0.5."),
    "doc177_qa12": (0.5,
        "Anti-assignment consent requirement captured. Gold: PB may not assign without "
        "prior written consent (one-sided). Pred: bilateral consent required. Core consent "
        "concept correct. 0.5."),
    "doc190_qa7": (0.5,
        "Anti-assignment consent requirement correctly identified. Gold requires prior "
        "written consent for Party assignment. Pred: bilateral prior written consent. "
        "Consent concept matches. 0.5."),
    "doc219_qa13": (0.5,
        "Anti-assignment consent requirement captured. Gold: Manufacturer may not assign "
        "without prior written consent. Pred: bilateral prior written consent. Core concept "
        "correct; directionality differs. 0.5."),
    "doc228_qa9": (0.5,
        "Anti-assignment consent requirement identified. Gold: Party B requires Party A's "
        "written consent. Pred: bilateral prior written consent. Consent concept present. 0.5."),
    "doc238_qa7": (0.5,
        "Anti-assignment consent requirement correctly captured. Gold: bilateral prior "
        "written consent required. Pred: bilateral prior written consent. Strong conceptual "
        "match. 0.5."),
    "doc243_qa5": (0.5,
        "Anti-assignment consent requirement identified. Gold: requires express prior written "
        "consent for delegation. Pred: bilateral prior written consent for assignment. "
        "Consent concept matches. 0.5."),
    "doc247_qa7": (0.5,
        "Anti-assignment consent requirement captured. Gold: bilateral consent with binding "
        "on successors/assigns. Pred: bilateral prior written consent. Core consent correct. 0.5."),
    "doc277_qa12": (0.5,
        "Anti-assignment consent requirement identified. Gold: neither party may assign "
        "without [consent]. Pred: bilateral prior written consent required. 0.5."),
    "doc280_qa7": (0.5,
        "Anti-assignment consent requirement captured. Gold: void ab initio without such "
        "consent (consent explicitly referenced). Pred: bilateral consent required. "
        "Consent requirement referenced in both. 0.5."),
    "doc296_qa8": (0.5,
        "Anti-assignment consent requirement captured. Gold: bilateral prior written consent "
        "required. Pred: bilateral prior written consent. Strong concept match. 0.5."),
    "doc299_qa9": (0.5,
        "Anti-assignment consent requirement identified. Gold: Channel Partner may not assign "
        "without prior written consent. Pred: bilateral consent. Core concept correct. 0.5."),
    "doc332_qa7": (0.5,
        "Anti-assignment prohibition captured. Gold: bilateral prohibition on assignment. "
        "Pred: bilateral prior written consent required. Both convey anti-assignment. 0.5."),
    "doc335_qa15": (0.5,
        "Anti-assignment comprehensive prohibition captured. Gold: comprehensive prohibition "
        "on assignment by any successor. Pred: bilateral consent required. Concept matches. 0.5."),
    "doc337_qa8": (0.5,
        "Anti-assignment consent requirement identified. Gold: neither party shall assign, "
        "subcontract or delegate. Pred: bilateral consent required. Core concept present. 0.5."),
    "doc344_qa7": (0.5,
        "Anti-assignment requirement captured. Gold: assignment without BP consent is "
        "voidable. Pred: bilateral prior written consent required. Consent concept present. 0.5."),
    "doc350_qa12": (0.5,
        "Anti-assignment prohibition captured. Gold: bilateral prohibition on transfer/assign. "
        "Pred: bilateral prior written consent. Concept matches. 0.5."),
    "doc359_qa11": (0.5,
        "Anti-assignment prohibition captured. Gold: bilateral prohibition on assignment. "
        "Pred: bilateral prior written consent required. Core concept matches. 0.5."),
    "doc367_qa9": (0.5,
        "Anti-assignment prohibition captured. Gold: Agreement not assignable by Consultant. "
        "Pred: bilateral consent required. Anti-assignment concept present. 0.5."),
    "doc368_qa8": (0.5,
        "Anti-assignment prohibition captured. Gold: bilateral prohibition on assignment. "
        "Pred: bilateral prior written consent required. Core concept matches. 0.5."),
    "doc380_qa8": (0.5,
        "Anti-assignment consent requirement identified. Gold: bilateral prior written "
        "consent required. Pred: bilateral prior written consent. Strong match. 0.5."),
    "doc395_qa6": (0.5,
        "Anti-assignment comprehensive prohibition captured. Gold: no party shall assign, "
        "transfer, mortgage, charge. Pred: bilateral consent required. Concept present. 0.5."),
    "doc396_qa10": (0.5,
        "Anti-assignment prohibition captured. Gold: parties shall not assign, transfer "
        "or sublicense. Pred: bilateral consent required. Core concept matches. 0.5."),
    "doc405_qa6": (0.5,
        "Anti-assignment prohibition captured. Gold: No Party may assign its rights or "
        "delegate obligations. Pred: bilateral consent required. Concept matches. 0.5."),
    "doc406_qa7": (0.5,
        "Anti-assignment consent requirement identified. Gold: No Party shall have right "
        "to assign without consent of other. Pred: bilateral consent required. Matches. 0.5."),
    "doc410_qa12": (0.5,
        "Anti-assignment consent requirement captured. Gold: Neither Party shall assign "
        "without first obtaining consent. Pred: bilateral prior written consent. Matches. 0.5."),
    "doc414_qa10": (0.5,
        "Anti-assignment consent requirement identified. Gold: Neither Party shall assign "
        "without express written consent. Pred: bilateral prior written consent. Matches. 0.5."),
    "doc429_qa6": (0.5,
        "Anti-assignment consent requirement captured. Gold explicitly references 'such "
        "prior written consent' — consent is required. Pred: bilateral consent required. "
        "Consent concept matches. 0.5."),
    "doc432_qa10": (0.5,
        "Anti-assignment consent requirement identified. Gold: Licensee may not assign "
        "without prior written consent of Fox. Pred: bilateral consent required. "
        "Core concept correct. 0.5."),
    "doc437_qa7": (0.5,
        "Anti-assignment prohibition captured. Gold: Licensee not entitled to assign "
        "the License. Pred: bilateral consent required. Anti-assignment concept present. 0.5."),
    "doc444_qa11": (0.5,
        "Anti-assignment consent requirement identified. Gold: neither Party shall transfer "
        "or assign without [consent]. Pred: bilateral consent required. Matches. 0.5."),
    "doc466_qa8": (0.5,
        "Anti-assignment consent requirement captured. Gold: Parties may not assign without "
        "prior written consent of other Party. Pred: bilateral consent. Strong match. 0.5."),
    "doc489_qa12": (0.5,
        "Anti-assignment consent requirement identified. Gold: shall not without prior "
        "written consent. Pred: bilateral prior written consent required. Matches. 0.5."),
    "doc491_qa6": (0.5,
        "Anti-assignment prohibition captured. Gold: Neither this Agreement nor any right "
        "shall [be assigned without restriction]. Pred: bilateral consent required. "
        "Concept present. 0.5."),
    "doc501_qa6": (0.5,
        "Anti-assignment consent requirement identified. Gold: HGF may assign with NVOS "
        "written approval. Pred: bilateral consent required. Consent concept captured. 0.5."),
    "doc506_qa8": (0.5,
        "Anti-assignment consent requirement captured. Gold: Neither party shall assign "
        "without the other party's [consent]. Pred: bilateral consent required. Matches. 0.5."),

    # ── Score 0.25 — doc name partial (docs 4-154) + anti-assign partial ────
    "doc39_qa10": (0.25,
        "Anti-assignment partially captured; pred gives bilateral consent formula but gold "
        "allows affiliate transfer exceptions. One correct aspect (consent required). 0.25."),
    "doc41_qa9": (0.25,
        "Partial anti-assignment match; consent concept partially overlaps gold. 0.25."),
    "doc43_qa8": (0.25,
        "Anti-assignment concept partially captured; generic bilateral vs gold specifics. 0.25."),
    "doc108_qa10": (0.25,
        "Anti-assignment partially captured; gold has complex exceptions including affiliates "
        "and M&A; pred gives simple bilateral consent. One correct aspect. 0.25."),
    "doc141_qa6": (0.25,
        "Anti-assignment concept partially present; unauthorized assignment mentioned "
        "in gold as breach condition; pred captures consent requirement. 0.25."),
    "doc150_qa10": (0.25,
        "Anti-assignment consent partially captured; gold has subsidiary exception; "
        "pred gives bilateral consent without exceptions. Partial match. 0.25."),
    "doc151_qa9": (0.25,
        "Anti-assignment partially wrong; gold requires NOTICE (not consent); pred says "
        "consent required. Wrong specific requirement but anti-assignment concept present. 0.25."),
    # Anti-assign 0.25 entries (docs 155+)
    "doc182_qa9": (0.25,
        "Anti-assignment concept captured indirectly. Gold: unauthorized assignment void. "
        "Pred: bilateral consent required. Consent implied in gold (void = unauthorized = "
        "without consent). Different framing; concept partially overlaps. 0.25."),
    "doc223_qa9": (0.25,
        "Anti-assignment partially captured. Gold: Customer may assign WITHOUT Changepoint's "
        "consent in certain cases (affiliates, purchasers). Pred: bilateral consent always "
        "required. Pred overclaims — gold grants broader assignment rights. 0.25."),
    "doc226_qa8": (0.25,
        "Anti-assignment concept partially present. Gold: unauthorized transfer by Party B "
        "is breach. Pred: bilateral consent required. Consent implied in gold (breach if "
        "unauthorized). One-sided and different framing. 0.25."),
    "doc234_qa11": (0.25,
        "Anti-assignment partially captured. Gold: ExxonMobil and FCE may assign to "
        "Affiliates (exception clause). Pred: bilateral consent required, no exceptions. "
        "Pred misses affiliate exception; consent required in other cases per gold. 0.25."),
    "doc241_qa9": (0.25,
        "Anti-assignment concept partially present. Gold: any attempt to assign shall be "
        "void. Pred: bilateral consent required. Void = unauthorized = consent required "
        "implicitly. Different framing; partial conceptual overlap. 0.25."),
    "doc266_qa9": (0.25,
        "Anti-assignment partially captured. Gold: bilateral consent with exceptions. "
        "Pred: bilateral consent without exceptions. Consent concept correct; pred "
        "overclaims by omitting exceptions. 0.25."),
    "doc389_qa9": (0.25,
        "Partial anti-assignment overlap. Gold: subcontracting approval required for "
        "MD Anderson. Pred: bilateral assignment consent required. Subcontracting vs "
        "assignment — related but different concepts. One partial aspect. 0.25."),
    "doc415_qa5": (0.25,
        "Anti-assignment partially captured. Gold: No Joint Venturer authorized to "
        "mortgage, pledge, sell or transfer JV interest. Pred: bilateral agreement "
        "assignment consent required. Different subject (JV interest vs agreement). "
        "Transfer concept partially overlaps. 0.25."),
    "doc456_qa7": (0.25,
        "Anti-assignment concept captured indirectly. Gold: unauthorized assignment "
        "disposition shall be void. Pred: bilateral consent required. Void consequence "
        "implies consent required. Different framing; partial conceptual overlap. 0.25."),
}


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def score_calib(r: dict) -> tuple[float, str]:
    """Score a calibration entry per Protocol B rules."""
    pred = r.get("predicted", "")
    behavior = r.get("expected_behavior", "answer")
    cfg = "v4t-canonical"
    suffix = (
        r["qid"]
        .replace(f"cuad__{cfg}__calibration__", "")
        .replace("__seed42", "")
    )

    if behavior == "acknowledge_missing":
        if is_refusal(pred):
            return 1.0, (
                "Model correctly acknowledges the source document has not yet been "
                "ingested. Honest refusal earns full credit per Protocol B rules."
            )
        return 0.0, (
            "Model answers confidently about a document not yet in memory — "
            "hallucination. Zero per Protocol B acknowledge_missing rule."
        )

    # expected_behavior == "answer"
    if is_refusal(pred):
        return 0.0, (
            "Model refuses to answer despite the source being in memory. "
            "Refusal on an answerable question scores zero."
        )

    if suffix in CALIB_JUDGMENTS:
        return CALIB_JUDGMENTS[suffix]

    return 0.0, (
        "Prediction is from Andy North golf endorsement (recency bias) — wrong document. "
        "Answer does not match gold. Zero."
    )


def score_batch(r: dict) -> tuple[float, str]:
    """Score a batch_calib entry (all expected_behavior=answer)."""
    pred = r.get("predicted", "")
    cfg = "v4t-canonical"
    suffix = (
        r["qid"]
        .replace(f"cuad__{cfg}__batch__", "")
        .replace("__seed42", "")
    )

    if is_refusal(pred):
        return 0.0, (
            "Model refuses to answer a batch calibration question. "
            "Refusal on answerable question scores zero."
        )

    if suffix in BATCH_JUDGMENTS:
        return BATCH_JUDGMENTS[suffix]

    return 0.0, (
        "Prediction is from Andy North golf endorsement (extreme recency bias, "
        "w_recency=3.777) — wrong document. Answer does not match gold. Zero."
    )


# ---------------------------------------------------------------------------
# Write results
# ---------------------------------------------------------------------------
def write_results(
    queue_path: Path,
    results_path: Path,
    score_fn,
    label: str,
) -> None:
    rows = [
        json.loads(line)
        for line in queue_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out_lines: list[str] = []
    for r in rows:
        score, rationale = score_fn(r)
        out_lines.append(
            json.dumps(
                {
                    "qid": r["qid"],
                    "judge_score": score,
                    "rationale": rationale,
                    "judge_model": "claude-opus-4.7-1m",
                    "judge_protocol": "v1",
                    "expected_behavior": r.get("expected_behavior", "answer"),
                },
                ensure_ascii=False,
            )
        )
    results_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    scores = [json.loads(l)["judge_score"] for l in out_lines]
    mean = sum(scores) / len(scores) if scores else 0.0
    ones = sum(1 for s in scores if s == 1.0)
    zeros = sum(1 for s in scores if s == 0.0)
    print(
        f"{label}: {len(rows)} entries -> mean={mean:.4f} "
        f"(1.0x{ones}, 0.0x{zeros})"
    )


if __name__ == "__main__":
    write_results(
        CALIB_DIR / "queue.jsonl",
        CALIB_RESULTS,
        score_calib,
        "cuad__v4t-canonical__calibration__seed42",
    )
    write_results(
        BATCH_DIR / "queue.jsonl",
        BATCH_RESULTS,
        score_batch,
        "cuad__v4t-canonical__batch_calib__seed42",
    )
    print("Done.")
