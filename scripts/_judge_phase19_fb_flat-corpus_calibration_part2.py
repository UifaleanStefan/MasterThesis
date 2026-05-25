"""Phase 1.9 — FB flat-corpus calibration part2 (entries 150-299)."""
from __future__ import annotations

import json
from pathlib import Path

QID_PREFIX = "financebench__flat-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__flat-corpus__calibration__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

SPECIAL: dict[int, tuple[float, str]] = {
    25: (0.0, "[ANS] doc2 seen=18; gold=NO 3M not cap-intensive; pred=refuses on definitive ANS."),
    26: (0.5, "[ACK] doc64 not seen; PRED gives correct world-knowledge (Boeing cyclical) — confident-correct from outside corpus."),
    35: (0.0, "[ANS] doc4 seen=19; gold=consumer segment 0.9%; pred=refuses on definitive ANS."),
    37: (0.0, "[ANS] doc3 seen=19; pred=refuses."),
    38: (0.0, "[ANS] doc14 seen=19; pred=refuses."),
    55: (0.25, "[ANS] doc19 seen=21; gold=30.8%; pred=26.5% (14% off — confident-wrong)."),
    59: (0.0, "[ANS] doc18 seen=21; gold=93.86; pred=36.12 (61% off — confident-wrong)."),
    60: (0.25, "[ACK] doc122 not seen; PRED='0' confident-wrong specific."),
    63: (0.0, "[ANS] doc11 seen=22; gold=65.4%; pred=refuses on definitive ANS."),
    66: (0.25, "[ACK] doc63 not seen; PRED gives partial world-knowledge (airlines + govt + defense) — confident specific partially correct."),
    78: (0.25, "[ACK] doc114 not seen; PRED=61% Nike COGS margin — confident-wrong specific (gold=55.1%)."),
    82: (0.25, "[ACK] doc63 same partial world-knowledge."),
    86: (0.0, "[ANS] doc11 seen=24; pred=refuses."),
    89: (1.0, "[ANS] doc15 seen=24; gold=0; pred=0 exact."),
    90: (0.5, "[ACK] doc125 not seen; PRED='not approved' = defeated (correct world-knowledge of PepsiCo AGM)."),
    91: (0.25, "[ACK] doc26 not seen; PRED dismisses metric as not useful (wrong framing — gold gives declining direction)."),
    92: (0.0, "[ANS] doc1 seen=25; pred=refuses on definitive ANS."),
    101: (0.25, "[ACK] doc139 not seen; PRED gives confident-wrong specific (sales growth investment vs gold's 47 new stores)."),
    105: (0.0, "[ANS] doc11 seen=26; pred=refuses."),
    106: (0.25, "[ACK] doc26 not seen; same partial mis-framing."),
    108: (0.0, "[ANS] doc2 seen=26; pred=refuses on definitive ANS."),
    78: (0.25, "[ACK] doc114 not seen; PRED gives confident-wrong specific (61% vs gold 55.1%)."),
    117: (0.25, "[ACK] doc63 same partial WK."),
    125: (1.0, "[ANS] doc20 seen=28; gold=$11588; pred=11,588 exact."),
    128: (0.0, "[ANS] doc0 seen=28; pred=refuses on definitive ANS."),
    133: (0.25, "[ACK] doc29 not seen; PRED=-1% confident-wrong specific (gold=flat)."),
    149: (0.0, "[ANS] doc18 seen=30; pred=refuses on definitive ANS."),
}

# Additional fix: 0228 doc43 confident-wrong, 0240 doc125 (already at 90), 0241 doc26 (already at 91), etc.
# Need to also add: 78 doc114 wrong, 116 doc114 — wait let me re-check the entry numbers carefully

# Recompute index for some entries — part2 starts at entry 150 (offset 0 in this part):
#   0150→idx 0, 0151→idx 1, ..., 0299→idx 149
# So entries listed in SPECIAL above use idx (position within part2).
# 0175 doc2 [ANS] refuses → idx 25 ✓
# 0176 doc64 [ACK] WK → idx 26 ✓
# 0185 doc4 [ANS] refuses → idx 35 ✓
# 0187 doc3 [ANS] refuses → idx 37 ✓
# 0188 doc14 [ANS] refuses → idx 38 ✓
# 0205 doc19 [ANS] 26.5% → idx 55 ✓
# 0209 doc18 [ANS] 36.12 → idx 59 ✓
# 0210 doc122 [ACK] '0' → idx 60 ✓
# 0213 doc11 [ANS] refuses → idx 63 ✓
# 0216 doc63 [ACK] WK → idx 66 ✓
# 0221 doc114 [ACK] 61% wrong → idx 71 (NOT 78!)
# 0228 doc43 [ACK] long-term debt wrong → idx 78
# 0232 doc63 [ACK] WK → idx 82 ✓
# 0236 doc11 [ANS] refuses → idx 86 ✓
# 0239 doc15 [ANS] 0=0 → idx 89 ✓
# 0240 doc125 [ACK] not approved → idx 90 ✓
# 0241 doc26 [ACK] mis-framing → idx 91 ✓
# 0242 doc1 [ANS] refuses → idx 92 ✓
# 0251 doc139 [ACK] wrong specific → idx 101 ✓
# 0255 doc11 [ANS] refuses → idx 105 ✓
# 0256 doc26 [ACK] same mis-framing → idx 106 ✓
# 0258 doc2 [ANS] refuses → idx 108 ✓
# 0267 doc63 [ACK] WK → idx 117 ✓
# 0275 doc20 [ANS] exact → idx 125 ✓
# 0278 doc0 [ANS] refuses → idx 128 ✓
# 0283 doc29 [ACK] -1% wrong → idx 133 ✓
# 0299 doc18 [ANS] refuses → idx 149 ✓

# Fix SPECIAL: change idx 78 mismatch
SPECIAL = {
    25: (0.0, "[ANS] doc2 seen=18; gold=NO 3M not cap-intensive; pred=refuses on definitive ANS."),
    26: (0.5, "[ACK] doc64 not seen; PRED gives correct world-knowledge (Boeing cyclical) — confident-correct from outside corpus."),
    35: (0.0, "[ANS] doc4 seen=19; gold=consumer segment 0.9%; pred=refuses on definitive ANS."),
    37: (0.0, "[ANS] doc3 seen=19; pred=refuses."),
    38: (0.0, "[ANS] doc14 seen=19; pred=refuses."),
    55: (0.25, "[ANS] doc19 seen=21; gold=30.8%; pred=26.5% (14% off — confident-wrong)."),
    59: (0.0, "[ANS] doc18 seen=21; gold=93.86; pred=36.12 (61% off — confident-wrong)."),
    60: (0.25, "[ACK] doc122 not seen; PRED='0' confident-wrong specific."),
    63: (0.0, "[ANS] doc11 seen=22; gold=65.4%; pred=refuses on definitive ANS."),
    66: (0.25, "[ACK] doc63 not seen; PRED gives partial world-knowledge (airlines + govt + defense) — confident specific partially correct."),
    71: (0.25, "[ACK] doc114 not seen; PRED=61% Nike COGS margin (gold=55.1%) — confident-wrong specific."),
    78: (0.25, "[ACK] doc43 not seen; PRED='long-term debt' (gold=Customer deposits) — confident-wrong specific."),
    82: (0.25, "[ACK] doc63 same partial world-knowledge."),
    86: (0.0, "[ANS] doc11 seen=24; pred=refuses."),
    89: (1.0, "[ANS] doc15 seen=24; gold=0; pred=0 exact."),
    90: (0.5, "[ACK] doc125 not seen; PRED='not approved' = defeated (correct world-knowledge of PepsiCo AGM)."),
    91: (0.25, "[ACK] doc26 not seen; PRED dismisses metric as not useful (wrong framing — gold gives declining direction)."),
    92: (0.0, "[ANS] doc1 seen=25; pred=refuses on definitive ANS."),
    101: (0.25, "[ACK] doc139 not seen; PRED gives confident-wrong specific."),
    105: (0.0, "[ANS] doc11 seen=26; pred=refuses."),
    106: (0.25, "[ACK] doc26 not seen; same partial mis-framing."),
    108: (0.0, "[ANS] doc2 seen=26; pred=refuses on definitive ANS."),
    117: (0.25, "[ACK] doc63 same partial WK."),
    125: (1.0, "[ANS] doc20 seen=28; gold=$11588; pred=11,588 exact."),
    128: (0.0, "[ANS] doc0 seen=28; pred=refuses on definitive ANS."),
    133: (0.25, "[ACK] doc29 not seen; PRED=-1% confident-wrong specific (gold=flat)."),
    149: (0.0, "[ANS] doc18 seen=30; pred=refuses on definitive ANS."),
}

DEFAULT_RATIONALE = "[ACK] source doc not yet seen in FlatMemory window. PRED honest refusal — correct."

ENTRY_SUFFIXES: list[str] = [
    "doc80_qa0__after15", "doc81_qa0__after15", "doc26_qa0__after15", "doc46_qa0__after15",
    "doc127_qa0__after15", "doc23_qa0__after15", "doc36_qa0__after15", "doc130_qa0__after15",
    "doc48_qa0__after15", "doc34_qa0__after15",
    "doc71_qa0__after16", "doc115_qa0__after16", "doc138_qa0__after16", "doc86_qa0__after16",
    "doc136_qa0__after16", "doc145_qa0__after16", "doc89_qa0__after16", "doc105_qa0__after16",
    "doc116_qa0__after16", "doc23_qa0__after16",
    "doc103_qa0__after17", "doc73_qa0__after17", "doc124_qa0__after17", "doc18_qa0__after17",
    "doc115_qa0__after17", "doc2_qa0__after17", "doc64_qa0__after17", "doc85_qa0__after17",
    "doc74_qa0__after17", "doc33_qa0__after17",
    "doc37_qa0__after18", "doc39_qa0__after18", "doc139_qa0__after18", "doc34_qa0__after18",
    "doc109_qa0__after18", "doc4_qa0__after18", "doc49_qa0__after18", "doc3_qa0__after18",
    "doc14_qa0__after18", "doc97_qa0__after18",
    "doc136_qa0__after19", "doc113_qa0__after19", "doc57_qa0__after19", "doc59_qa0__after19",
    "doc75_qa0__after19", "doc36_qa0__after19", "doc110_qa0__after19", "doc51_qa0__after19",
    "doc119_qa0__after19", "doc138_qa0__after19",
    "doc105_qa0__after20", "doc74_qa0__after20", "doc84_qa0__after20", "doc36_qa0__after20",
    "doc83_qa0__after20", "doc19_qa0__after20", "doc140_qa0__after20", "doc61_qa0__after20",
    "doc111_qa0__after20", "doc18_qa0__after20",
    "doc122_qa0__after21", "doc113_qa0__after21", "doc91_qa0__after21", "doc11_qa0__after21",
    "doc110_qa0__after21", "doc140_qa0__after21", "doc63_qa0__after21", "doc48_qa0__after21",
    "doc87_qa0__after21", "doc68_qa0__after21",
    "doc120_qa0__after22", "doc114_qa0__after22", "doc99_qa0__after22", "doc80_qa0__after22",
    "doc45_qa0__after22", "doc68_qa0__after22", "doc53_qa0__after22", "doc84_qa0__after22",
    "doc43_qa0__after22", "doc61_qa0__after22",
    "doc48_qa0__after23", "doc66_qa0__after23", "doc63_qa0__after23", "doc113_qa0__after23",
    "doc117_qa0__after23", "doc41_qa0__after23", "doc11_qa0__after23", "doc128_qa0__after23",
    "doc119_qa0__after23", "doc15_qa0__after23",
    "doc125_qa0__after24", "doc26_qa0__after24", "doc1_qa0__after24", "doc32_qa0__after24",
    "doc61_qa0__after24", "doc126_qa0__after24", "doc134_qa0__after24", "doc53_qa0__after24",
    "doc120_qa0__after24", "doc135_qa0__after24",
    "doc59_qa0__after25", "doc139_qa0__after25", "doc134_qa0__after25", "doc83_qa0__after25",
    "doc31_qa0__after25", "doc11_qa0__after25", "doc26_qa0__after25", "doc94_qa0__after25",
    "doc2_qa0__after25", "doc49_qa0__after25",
    "doc36_qa0__after26", "doc131_qa0__after26", "doc115_qa0__after26", "doc85_qa0__after26",
    "doc118_qa0__after26", "doc77_qa0__after26", "doc110_qa0__after26", "doc63_qa0__after26",
    "doc40_qa0__after26", "doc74_qa0__after26",
    "doc102_qa0__after27", "doc124_qa0__after27", "doc39_qa0__after27", "doc105_qa0__after27",
    "doc132_qa0__after27", "doc20_qa0__after27", "doc106_qa0__after27", "doc80_qa0__after27",
    "doc0_qa0__after27", "doc104_qa0__after27",
    "doc89_qa0__after28", "doc63_qa0__after28", "doc41_qa0__after28", "doc29_qa0__after28",
    "doc124_qa0__after28", "doc109_qa0__after28", "doc106_qa0__after28", "doc39_qa0__after28",
    "doc56_qa0__after28", "doc70_qa0__after28",
    "doc147_qa0__after29", "doc135_qa0__after29", "doc124_qa0__after29", "doc97_qa0__after29",
    "doc58_qa0__after29", "doc91_qa0__after29", "doc138_qa0__after29", "doc108_qa0__after29",
    "doc71_qa0__after29", "doc18_qa0__after29",
]


def main() -> None:
    assert len(ENTRY_SUFFIXES) == 150, f"expected 150 entries, got {len(ENTRY_SUFFIXES)}"

    existing_qids: set[str] = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                existing_qids.add(obj["qid"])
            except (json.JSONDecodeError, KeyError):
                pass

    added = 0
    skipped = 0
    total_score = 0.0
    with RESULTS.open("a", encoding="utf-8") as f:
        for i, suffix in enumerate(ENTRY_SUFFIXES):
            qid = QID_PREFIX + suffix + QID_SUFFIX
            if qid in existing_qids:
                skipped += 1
                continue
            if i in SPECIAL:
                score, rationale = SPECIAL[i]
            else:
                score, rationale = 1.0, DEFAULT_RATIONALE
            row = {
                "qid": qid,
                "judge_score": score,
                "rationale": rationale,
                "judge_model": JUDGE_MODEL,
                "judge_protocol": JUDGE_PROTOCOL,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            added += 1
            total_score += score
            existing_qids.add(qid)

    mean = total_score / added if added else 0.0
    print(f"part2 added={added} skipped={skipped} mean_score={mean:.4f}")


if __name__ == "__main__":
    main()
