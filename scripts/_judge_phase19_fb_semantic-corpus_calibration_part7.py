"""Phase 1.9 — FB semantic-corpus calibration part7 (entries 900-1049)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__semantic-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__semantic-corpus__calibration__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

SPECIAL: dict[int, tuple[float, str]] = {
    0: (0.0, "[ANS] doc66 seen=91; pred=refuses on definitive ANS."),
    2: (0.0, "[ANS] doc30 seen=91; pred=refuses."),
    4: (0.0, "[ANS] doc41 seen=91; pred=refuses."),
    5: (0.0, "[ANS] doc45 seen=91; pred=refuses."),
    6: (0.0, "[ANS] doc5 seen=91; pred=refuses."),
    8: (0.5, "[ACK] doc125 not seen; PRED='not approved' correct WK."),
    10: (0.5, "[ACK] doc96 not seen; PRED='gross margin not relevant for JPM' correct WK."),
    11: (0.0, "[ANS] doc88 seen=92; pred=refuses."),
    12: (0.0, "[ANS] doc79 seen=92; gold=Yes Mary Dillon; pred=refuses."),
    13: (0.0, "[ANS] doc33 seen=92; pred=refuses."),
    14: (0.0, "[ANS] doc20 seen=92; pred=refuses."),
    15: (0.0, "[ANS] doc40 seen=92; pred=refuses."),
    16: (0.0, "[ANS] doc86 seen=92; pred=refuses."),
    19: (0.0, "[ANS] doc18 seen=92; pred=refuses."),
    21: (0.0, "[ANS] doc45 seen=93; pred=refuses."),
    23: (0.0, "[ANS] doc78 seen=93; gold=Yes $0.55 dividend; pred=refuses."),
    24: (0.0, "[ANS] doc91 seen=93; pred=refuses."),
    25: (0.0, "[ANS] doc10 seen=93; pred=refuses."),
    26: (0.0, "[ANS] doc12 seen=93; pred=refuses."),
    28: (0.0, "[ANS] doc86 seen=93; pred=refuses."),
    29: (0.25, "[ACK] doc122 not seen; pred='0' confident-wrong."),
    30: (0.0, "[ANS] doc26 seen=94; pred=refuses."),
    31: (0.0, "[ANS] doc64 seen=94; pred=refuses."),
    34: (0.0, "[ANS] doc54 seen=94; pred=refuses."),
    39: (0.0, "[ANS] doc82 seen=94; pred=refuses."),
    40: (0.0, "[ANS] doc18 seen=95; pred=refuses."),
    42: (0.0, "[ANS] doc52 seen=95; pred=refuses."),
    43: (0.0, "[ANS] doc9 seen=95; pred=refuses."),
    44: (0.0, "[ANS] doc64 seen=95; pred=refuses."),
    47: (0.0, "[ANS] doc83 seen=95; pred=refuses."),
    50: (0.0, "[ANS] doc18 seen=96; pred=refuses."),
    51: (0.0, "[ANS] doc80 seen=96; pred=refuses."),
    52: (0.0, "[ANS] doc52 seen=96; pred=refuses."),
    55: (0.0, "[ANS] doc51 seen=96; pred=refuses."),
    57: (0.25, "[ACK] doc122 not seen; pred='0' confident-wrong."),
    58: (0.0, "[ANS] doc8 seen=96; pred=refuses."),
    59: (0.0, "[ANS] doc17 seen=96; pred=refuses."),
    60: (0.0, "[ANS] doc86 seen=97; pred=refuses."),
    61: (0.0, "[ANS] doc80 seen=97; pred=refuses."),
    62: (0.0, "[ANS] doc94 seen=97; pred=refuses."),
    64: (0.0, "[ANS] doc95 seen=97; pred=refuses."),
    66: (0.0, "[ANS] doc53 seen=97; pred=refuses."),
    67: (0.0, "[ANS] doc52 seen=97; pred=refuses."),
    68: (0.0, "[ANS] doc50 seen=97; pred=refuses."),
    69: (0.0, "[ANS] doc39 seen=97; pred=refuses."),
    71: (0.0, "[ANS] doc63 seen=98; pred=refuses."),
    73: (0.0, "[ANS] doc8 seen=98; pred=refuses."),
    74: (0.0, "[ANS] doc47 seen=98; pred=refuses."),
    75: (0.5, "[ACK] doc125 not seen; PRED='not approved' correct WK."),
    76: (0.0, "[ANS] doc95 seen=98; pred=refuses."),
    77: (0.0, "[ANS] doc37 seen=98; pred=refuses."),
    78: (0.0, "[ANS] doc6 seen=98; pred=refuses."),
    79: (0.0, "[ANS] doc50 seen=98; pred=refuses."),
    80: (0.0, "[ANS] doc42 seen=99; pred=refuses."),
    82: (0.0, "[ANS] doc80 seen=99; pred=refuses."),
    83: (0.0, "[ANS] doc91 seen=99; pred=refuses."),
    84: (0.0, "[ANS] doc60 seen=99; pred=refuses."),
    87: (0.0, "[ANS] doc97 seen=99; pred=refuses."),
    89: (0.0, "[ANS] doc16 seen=99; pred=refuses."),
    91: (0.0, "[ANS] doc11 seen=100; pred=refuses."),
    92: (0.0, "[ANS] doc40 seen=100; pred=refuses."),
    96: (0.0, "[ANS] doc43 seen=100; pred=refuses."),
    97: (0.0, "[ANS] doc71 seen=100; pred=refuses."),
    100: (0.0, "[ANS] doc5 seen=101; pred=refuses."),
    102: (0.0, "[ANS] doc10 seen=101; pred=refuses."),
    106: (0.0, "[ANS] doc67 seen=101; pred=refuses."),
    108: (0.0, "[ANS] doc65 seen=101; pred=refuses."),
    109: (0.0, "[ANS] doc63 seen=101; pred=refuses."),
    110: (0.0, "[ANS] doc81 seen=102; pred=refuses."),
    112: (0.0, "[ANS] doc35 seen=102; pred=refuses."),
    113: (0.0, "[ANS] doc41 seen=102; pred=refuses."),
    114: (0.0, "[ANS] doc100 seen=102; pred=refuses."),
    115: (0.0, "[ANS] doc98 seen=102; pred=refuses."),
    116: (0.0, "[ANS] doc78 seen=102; pred=refuses."),
    117: (0.0, "[ANS] doc75 seen=102; pred=refuses."),
    119: (0.5, "[ACK] doc125 not seen; PRED='not approved' correct WK."),
    120: (0.0, "[ANS] doc31 seen=103; pred=refuses."),
    121: (0.0, "[ANS] doc39 seen=103; pred=refuses."),
    122: (0.0, "[ANS] doc24 seen=103; pred=refuses."),
    123: (0.0, "[ANS] doc68 seen=103; pred=refuses."),
    125: (0.0, "[ANS] doc44 seen=103; pred=refuses."),
    126: (0.0, "[ANS] doc36 seen=103; pred=refuses."),
    127: (0.0, "[ANS] doc59 seen=103; pred=refuses."),
    128: (0.0, "[ANS] doc46 seen=103; pred=refuses."),
    131: (0.0, "[ANS] doc61 seen=104; pred=refuses."),
    133: (0.0, "[ANS] doc60 seen=104; pred=refuses."),
    134: (0.0, "[ANS] doc36 seen=104; pred=refuses."),
    135: (0.0, "[ANS] doc51 seen=104; pred=refuses."),
    136: (0.0, "[ANS] doc85 seen=104; pred=refuses."),
    138: (0.0, "[ANS] doc71 seen=104; pred=refuses."),
    140: (0.0, "[ANS] doc46 seen=105; pred=refuses."),
    144: (0.0, "[ANS] doc16 seen=105; pred=refuses."),
    145: (0.0, "[ANS] doc80 seen=105; pred=refuses."),
    146: (0.0, "[ANS] doc31 seen=105; pred=refuses."),
    147: (0.0, "[ANS] doc14 seen=105; pred=refuses."),
    148: (0.0, "[ANS] doc101 seen=105; pred=refuses."),
    149: (0.0, "[ANS] doc103 seen=105; pred=refuses."),
}

DEFAULT_RATIONALE = "[ACK] source doc not yet seen in semantic corpus. PRED honest refusal — correct."

ENTRY_SUFFIXES: list[str] = [
    "doc66_qa0__after90", "doc113_qa0__after90", "doc30_qa0__after90", "doc116_qa0__after90",
    "doc41_qa0__after90", "doc45_qa0__after90", "doc5_qa0__after90", "doc91_qa0__after90",
    "doc125_qa0__after90", "doc126_qa0__after90",
    "doc96_qa0__after91", "doc88_qa0__after91", "doc79_qa0__after91", "doc33_qa0__after91",
    "doc20_qa0__after91", "doc40_qa0__after91", "doc86_qa0__after91", "doc15_qa0__after91",
    "doc99_qa0__after91", "doc18_qa0__after91",
    "doc101_qa0__after92", "doc45_qa0__after92", "doc114_qa0__after92", "doc78_qa0__after92",
    "doc91_qa0__after92", "doc10_qa0__after92", "doc12_qa0__after92", "doc94_qa0__after92",
    "doc86_qa0__after92", "doc122_qa0__after92",
    "doc26_qa0__after93", "doc64_qa0__after93", "doc146_qa0__after93", "doc136_qa0__after93",
    "doc54_qa0__after93", "doc106_qa0__after93", "doc149_qa0__after93", "doc144_qa0__after93",
    "doc143_qa0__after93", "doc82_qa0__after93",
    "doc18_qa0__after94", "doc126_qa0__after94", "doc52_qa0__after94", "doc9_qa0__after94",
    "doc64_qa0__after94", "doc117_qa0__after94", "doc129_qa0__after94", "doc83_qa0__after94",
    "doc112_qa0__after94", "doc104_qa0__after94",
    "doc18_qa0__after95", "doc80_qa0__after95", "doc52_qa0__after95", "doc100_qa0__after95",
    "doc106_qa0__after95", "doc51_qa0__after95", "doc142_qa0__after95", "doc122_qa0__after95",
    "doc8_qa0__after95", "doc17_qa0__after95",
    "doc86_qa0__after96", "doc80_qa0__after96", "doc94_qa0__after96", "doc15_qa0__after96",
    "doc95_qa0__after96", "doc127_qa0__after96", "doc53_qa0__after96", "doc52_qa0__after96",
    "doc50_qa0__after96", "doc39_qa0__after96",
    "doc133_qa0__after97", "doc63_qa0__after97", "doc118_qa0__after97", "doc8_qa0__after97",
    "doc47_qa0__after97", "doc125_qa0__after97", "doc95_qa0__after97", "doc37_qa0__after97",
    "doc6_qa0__after97", "doc50_qa0__after97",
    "doc42_qa0__after98", "doc141_qa0__after98", "doc80_qa0__after98", "doc91_qa0__after98",
    "doc60_qa0__after98", "doc149_qa0__after98", "doc108_qa0__after98", "doc97_qa0__after98",
    "doc138_qa0__after98", "doc16_qa0__after98",
    "doc113_qa0__after99", "doc11_qa0__after99", "doc40_qa0__after99", "doc127_qa0__after99",
    "doc108_qa0__after99", "doc145_qa0__after99", "doc43_qa0__after99", "doc71_qa0__after99",
    "doc124_qa0__after99", "doc116_qa0__after99",
    "doc5_qa0__after100", "doc129_qa0__after100", "doc10_qa0__after100", "doc90_qa0__after100",
    "doc148_qa0__after100", "doc15_qa0__after100", "doc67_qa0__after100", "doc127_qa0__after100",
    "doc65_qa0__after100", "doc63_qa0__after100",
    "doc81_qa0__after101", "doc114_qa0__after101", "doc35_qa0__after101", "doc41_qa0__after101",
    "doc100_qa0__after101", "doc98_qa0__after101", "doc78_qa0__after101", "doc75_qa0__after101",
    "doc96_qa0__after101", "doc125_qa0__after101",
    "doc31_qa0__after102", "doc39_qa0__after102", "doc24_qa0__after102", "doc68_qa0__after102",
    "doc119_qa0__after102", "doc44_qa0__after102", "doc36_qa0__after102", "doc59_qa0__after102",
    "doc46_qa0__after102", "doc108_qa0__after102",
    "doc108_qa0__after103", "doc61_qa0__after103", "doc135_qa0__after103", "doc60_qa0__after103",
    "doc36_qa0__after103", "doc51_qa0__after103", "doc85_qa0__after103", "doc105_qa0__after103",
    "doc71_qa0__after103", "doc137_qa0__after103",
    "doc46_qa0__after104", "doc136_qa0__after104", "doc121_qa0__after104", "doc96_qa0__after104",
    "doc16_qa0__after104", "doc80_qa0__after104", "doc31_qa0__after104", "doc14_qa0__after104",
    "doc101_qa0__after104", "doc103_qa0__after104",
]


def main() -> None:
    assert len(ENTRY_SUFFIXES) == 150
    existing: set[str] = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing.add(json.loads(line)["qid"])
                except (json.JSONDecodeError, KeyError):
                    pass
    added = skipped = 0
    total = 0.0
    with RESULTS.open("a", encoding="utf-8") as f:
        for i, suffix in enumerate(ENTRY_SUFFIXES):
            qid = QID_PREFIX + suffix + QID_SUFFIX
            if qid in existing:
                skipped += 1
                continue
            score, rationale = SPECIAL.get(i, (1.0, DEFAULT_RATIONALE))
            f.write(json.dumps({"qid": qid, "judge_score": score, "rationale": rationale,
                                "judge_model": JUDGE_MODEL, "judge_protocol": JUDGE_PROTOCOL},
                               ensure_ascii=False) + "\n")
            added += 1
            total += score
            existing.add(qid)
    print(f"part7 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
