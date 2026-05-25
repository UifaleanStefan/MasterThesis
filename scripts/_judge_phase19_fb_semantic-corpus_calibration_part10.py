"""Phase 1.9 — FB semantic-corpus calibration part10 (entries 1350-1499, FINAL)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__semantic-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__semantic-corpus__calibration__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

SPECIAL: dict[int, tuple[float, str]] = {
    0: (0.0, "[ANS] doc55 seen=136; pred=refuses."),
    1: (0.0, "[ANS] doc60 seen=136; pred=refuses."),
    2: (0.0, "[ANS] doc102 seen=136; pred=refuses."),
    3: (0.0, "[ANS] doc88 seen=136; pred=refuses."),
    4: (0.0, "[ANS] doc86 seen=136; pred=refuses."),
    5: (0.0, "[ANS] doc81 seen=136; pred=refuses."),
    6: (0.0, "[ANS] doc118 seen=136; pred=refuses."),
    8: (0.25, "[ANS] doc127 seen=136; pred='$4.5B' wrong (gold $8.4B)."),
    9: (0.0, "[ANS] doc10 seen=136; pred=refuses."),
    10: (0.0, "[ANS] doc115 seen=137; pred=refuses."),
    11: (0.5, "[ANS] doc120 seen=137; partial geography list."),
    12: (0.0, "[ANS] doc27 seen=137; pred=refuses."),
    14: (0.0, "[ANS] doc108 seen=137; pred=refuses."),
    15: (0.0, "[ANS] doc2 seen=137; Y/N flip."),
    16: (0.0, "[ANS] doc58 seen=137; pred=refuses."),
    17: (0.0, "[ANS] doc80 seen=137; pred=refuses."),
    18: (0.0, "[ANS] doc63 seen=137; pred=refuses."),
    19: (0.0, "[ANS] doc103 seen=137; pred=refuses."),
    20: (0.0, "[ANS] doc128 seen=138; pred=refuses."),
    21: (0.0, "[ANS] doc39 seen=138; pred=refuses."),
    22: (0.0, "[ANS] doc60 seen=138; pred=refuses."),
    23: (0.0, "[ANS] doc88 seen=138; pred=refuses."),
    24: (0.0, "[ANS] doc134 seen=138; pred=refuses."),
    25: (0.0, "[ANS] doc135 seen=138; pred=refuses."),
    26: (0.0, "[ANS] doc113 seen=138; pred=refuses."),
    27: (0.25, "[ANS] doc126 seen=138; pred='$1.5B' wrong (gold $400M)."),
    28: (0.0, "[ANS] doc18 seen=138; pred=refuses."),
    29: (0.0, "[ANS] doc13 seen=138; pred=refuses."),
    30: (0.0, "[ANS] doc60 seen=139; pred=refuses."),
    31: (0.0, "[ANS] doc39 seen=139; pred=refuses."),
    32: (0.0, "[ANS] doc119 seen=139; pred=refuses."),
    34: (0.0, "[ANS] doc35 seen=139; pred=refuses."),
    35: (0.0, "[ANS] doc8 seen=139; pred=refuses."),
    36: (0.0, "[ANS] doc131 seen=139; pred=refuses."),
    37: (0.0, "[ANS] doc67 seen=139; pred=refuses."),
    38: (0.0, "[ANS] doc47 seen=139; pred=refuses."),
    41: (0.0, "[ANS] doc70 seen=140; pred=refuses."),
    42: (0.0, "[ANS] doc118 seen=140; pred=refuses."),
    43: (0.0, "[ANS] doc39 seen=140; pred=refuses."),
    44: (0.25, "[ANS] doc74 seen=140; pred=$52,694 vs $59,268 (11% off)."),
    45: (0.0, "[ANS] doc12 seen=140; pred=refuses."),
    46: (0.0, "[ANS] doc24 seen=140; pred=refuses."),
    47: (0.0, "[ANS] doc25 seen=140; pred=refuses."),
    48: (0.75, "[ANS] doc0 seen=140; pred=$1,501 close to gold $1577."),
    49: (0.0, "[ANS] doc92 seen=140; pred=refuses."),
    50: (0.0, "[ANS] doc5 seen=141; pred=refuses."),
    51: (0.0, "[ANS] doc135 seen=141; pred=refuses."),
    52: (0.0, "[ANS] doc76 seen=141; pred=refuses."),
    53: (0.0, "[ANS] doc26 seen=141; pred=refuses."),
    54: (0.0, "[ANS] doc55 seen=141; pred=refuses."),
    55: (0.0, "[ANS] doc58 seen=141; pred=refuses."),
    56: (0.0, "[ANS] doc105 seen=141; pred=refuses."),
    57: (0.0, "[ANS] doc31 seen=141; pred=refuses."),
    58: (0.0, "[ANS] doc123 seen=141; pred=refuses."),
    60: (0.0, "[ANS] doc62 seen=142; pred=refuses."),
    62: (0.0, "[ANS] doc38 seen=142; pred=refuses."),
    65: (0.0, "[ANS] doc87 seen=142; pred=refuses."),
    66: (0.0, "[ANS] doc63 seen=142; pred=refuses."),
    67: (0.0, "[ANS] doc69 seen=142; pred=refuses."),
    68: (0.0, "[ANS] doc124 seen=142; pred=refuses."),
    69: (0.0, "[ANS] doc17 seen=142; pred=refuses."),
    70: (0.0, "[ANS] doc34 seen=143; pred=refuses."),
    71: (0.0, "[ANS] doc102 seen=143; pred=refuses."),
    72: (0.25, "[ANS] doc127 seen=143; pred='$4.0B' wrong."),
    74: (0.0, "[ANS] doc2 seen=143; Y/N flip."),
    75: (0.0, "[ANS] doc113 seen=143; pred=refuses."),
    76: (0.0, "[ANS] doc139 seen=143; pred=refuses."),
    77: (0.0, "[ANS] doc74 seen=143; pred=refuses."),
    78: (0.0, "[ANS] doc132 seen=143; pred=refuses."),
    79: (0.0, "[ANS] doc107 seen=143; pred=refuses."),
    80: (0.0, "[ANS] doc63 seen=144; pred=refuses."),
    81: (0.0, "[ANS] doc45 seen=144; pred=refuses."),
    82: (0.5, "[ANS] doc4 seen=144; pred='Consumer segment' partial."),
    83: (0.0, "[ANS] doc141 seen=144; pred=refuses."),
    84: (0.0, "[ANS] doc93 seen=144; pred=refuses."),
    85: (0.0, "[ANS] doc134 seen=144; pred=refuses."),
    86: (0.0, "[ANS] doc79 seen=144; pred=refuses."),
    87: (0.0, "[ANS] doc138 seen=144; pred=refuses."),
    88: (0.0, "[ANS] doc11 seen=144; pred=refuses."),
    89: (0.0, "[ANS] doc7 seen=144; pred=refuses."),
    90: (0.0, "[ANS] doc86 seen=145; pred=refuses."),
    91: (0.0, "[ANS] doc31 seen=145; pred=refuses."),
    92: (0.0, "[ANS] doc139 seen=145; pred=refuses."),
    93: (0.0, "[ANS] doc44 seen=145; pred=refuses."),
    94: (0.0, "[ANS] doc24 seen=145; pred=refuses."),
    95: (0.0, "[ANS] doc97 seen=145; pred=refuses."),
    96: (0.0, "[ANS] doc63 seen=145; pred=refuses."),
    97: (0.0, "[ANS] doc110 seen=145; pred=refuses."),
    98: (0.0, "[ANS] doc23 seen=145; pred=refuses."),
    99: (0.0, "[ANS] doc78 seen=145; pred=refuses."),
    100: (0.0, "[ANS] doc23 seen=146; pred=refuses."),
    101: (0.0, "[ANS] doc110 seen=146; pred=refuses."),
    102: (0.0, "[ANS] doc19 seen=146; pred=refuses."),
    103: (0.0, "[ANS] doc20 seen=146; pred=refuses."),
    104: (0.0, "[ANS] doc136 seen=146; pred=refuses."),
    105: (0.0, "[ANS] doc95 seen=146; pred=refuses."),
    106: (0.25, "[ANS] doc119 seen=146; gold=$4.6B; pred=$4.2B (8.7% off)."),
    107: (0.0, "[ANS] doc109 seen=146; pred=refuses."),
    108: (0.0, "[ANS] doc62 seen=146; pred=refuses."),
    109: (0.0, "[ANS] doc12 seen=146; pred=refuses."),
    110: (0.0, "[ANS] doc111 seen=147; pred=refuses."),
    111: (0.0, "[ANS] doc51 seen=147; pred=refuses."),
    112: (0.0, "[ANS] doc10 seen=147; pred=refuses."),
    113: (0.0, "[ANS] doc64 seen=147; pred=refuses."),
    114: (0.0, "[ANS] doc139 seen=147; pred=refuses."),
    115: (0.0, "[ANS] doc24 seen=147; pred=refuses."),
    116: (0.0, "[ANS] doc98 seen=147; pred=refuses."),
    117: (0.0, "[ANS] doc5 seen=147; pred=refuses."),
    118: (0.0, "[ANS] doc13 seen=147; pred=refuses."),
    119: (0.0, "[ANS] doc53 seen=147; pred=refuses."),
    120: (0.0, "[ANS] doc25 seen=148; pred=refuses."),
    121: (0.0, "[ANS] doc24 seen=148; pred=refuses."),
    122: (0.0, "[ANS] doc35 seen=148; pred=refuses."),
    123: (0.0, "[ANS] doc22 seen=148; pred=refuses."),
    124: (0.0, "[ANS] doc117 seen=148; pred=refuses."),
    125: (0.0, "[ANS] doc26 seen=148; pred=refuses."),
    126: (0.0, "[ANS] doc141 seen=148; pred=refuses."),
    127: (0.0, "[ANS] doc83 seen=148; pred=refuses."),
    128: (0.0, "[ANS] doc102 seen=148; pred=refuses."),
    129: (0.0, "[ANS] doc111 seen=148; pred=refuses."),
    130: (0.0, "[ANS] doc140 seen=149; pred=refuses."),
    131: (0.0, "[ANS] doc107 seen=149; pred=refuses."),
    132: (0.0, "[ANS] doc38 seen=149; pred=refuses."),
    133: (0.0, "[ANS] doc59 seen=149; pred=refuses."),
    134: (0.5, "[ANS] doc120 seen=149; partial."),
    135: (0.25, "[ANS] doc127 seen=149; pred='$4.5B' wrong."),
    136: (0.0, "[ANS] doc77 seen=149; pred=refuses."),
    137: (0.0, "[ANS] doc118 seen=149; pred=refuses."),
    138: (0.0, "[ANS] doc85 seen=149; pred=refuses."),
    139: (0.0, "[ANS] doc137 seen=149; gold=no acquisitions; pred=lacks information (not equivalent)."),
    141: (0.0, "[ANS] doc82 seen=150; pred=refuses."),
    142: (0.0, "[ANS] doc63 seen=150; pred=refuses."),
    143: (0.0, "[ANS] doc109 seen=150; pred=refuses."),
    144: (0.0, "[ANS] doc61 seen=150; pred=refuses."),
    145: (0.0, "[ANS] doc55 seen=150; pred=refuses."),
    146: (0.0, "[ANS] doc80 seen=150; pred=refuses."),
    147: (0.0, "[ANS] doc105 seen=150; pred=refuses."),
    148: (0.0, "[ANS] doc108 seen=150; pred=refuses."),
    149: (0.0, "[ANS] doc128 seen=150; pred=refuses."),
}

DEFAULT_RATIONALE = "[ACK] source doc not yet seen in semantic corpus. PRED honest refusal — correct."

ENTRY_SUFFIXES: list[str] = [
    "doc55_qa0__after135", "doc60_qa0__after135", "doc102_qa0__after135", "doc88_qa0__after135",
    "doc86_qa0__after135", "doc81_qa0__after135", "doc118_qa0__after135", "doc139_qa0__after135",
    "doc127_qa0__after135", "doc10_qa0__after135",
    "doc115_qa0__after136", "doc120_qa0__after136", "doc27_qa0__after136", "doc148_qa0__after136",
    "doc108_qa0__after136", "doc2_qa0__after136", "doc58_qa0__after136", "doc80_qa0__after136",
    "doc63_qa0__after136", "doc103_qa0__after136",
    "doc128_qa0__after137", "doc39_qa0__after137", "doc60_qa0__after137", "doc88_qa0__after137",
    "doc134_qa0__after137", "doc135_qa0__after137", "doc113_qa0__after137", "doc126_qa0__after137",
    "doc18_qa0__after137", "doc13_qa0__after137",
    "doc60_qa0__after138", "doc39_qa0__after138", "doc119_qa0__after138", "doc142_qa0__after138",
    "doc35_qa0__after138", "doc8_qa0__after138", "doc131_qa0__after138", "doc67_qa0__after138",
    "doc47_qa0__after138", "doc3_qa0__after138",
    "doc148_qa0__after139", "doc70_qa0__after139", "doc118_qa0__after139", "doc39_qa0__after139",
    "doc74_qa0__after139", "doc12_qa0__after139", "doc24_qa0__after139", "doc25_qa0__after139",
    "doc0_qa0__after139", "doc92_qa0__after139",
    "doc5_qa0__after140", "doc135_qa0__after140", "doc76_qa0__after140", "doc26_qa0__after140",
    "doc55_qa0__after140", "doc58_qa0__after140", "doc105_qa0__after140", "doc31_qa0__after140",
    "doc123_qa0__after140", "doc3_qa0__after140",
    "doc62_qa0__after141", "doc3_qa0__after141", "doc38_qa0__after141", "doc143_qa0__after141",
    "doc125_qa0__after141", "doc87_qa0__after141", "doc63_qa0__after141", "doc69_qa0__after141",
    "doc124_qa0__after141", "doc17_qa0__after141",
    "doc34_qa0__after142", "doc102_qa0__after142", "doc127_qa0__after142", "doc146_qa0__after142",
    "doc2_qa0__after142", "doc113_qa0__after142", "doc139_qa0__after142", "doc74_qa0__after142",
    "doc132_qa0__after142", "doc107_qa0__after142",
    "doc63_qa0__after143", "doc45_qa0__after143", "doc4_qa0__after143", "doc141_qa0__after143",
    "doc93_qa0__after143", "doc134_qa0__after143", "doc79_qa0__after143", "doc138_qa0__after143",
    "doc11_qa0__after143", "doc7_qa0__after143",
    "doc86_qa0__after144", "doc31_qa0__after144", "doc139_qa0__after144", "doc44_qa0__after144",
    "doc24_qa0__after144", "doc97_qa0__after144", "doc63_qa0__after144", "doc110_qa0__after144",
    "doc23_qa0__after144", "doc78_qa0__after144",
    "doc23_qa0__after145", "doc110_qa0__after145", "doc19_qa0__after145", "doc20_qa0__after145",
    "doc136_qa0__after145", "doc95_qa0__after145", "doc119_qa0__after145", "doc109_qa0__after145",
    "doc62_qa0__after145", "doc12_qa0__after145",
    "doc111_qa0__after146", "doc51_qa0__after146", "doc10_qa0__after146", "doc64_qa0__after146",
    "doc139_qa0__after146", "doc24_qa0__after146", "doc98_qa0__after146", "doc5_qa0__after146",
    "doc13_qa0__after146", "doc53_qa0__after146",
    "doc25_qa0__after147", "doc24_qa0__after147", "doc35_qa0__after147", "doc22_qa0__after147",
    "doc117_qa0__after147", "doc26_qa0__after147", "doc141_qa0__after147", "doc83_qa0__after147",
    "doc102_qa0__after147", "doc111_qa0__after147",
    "doc140_qa0__after148", "doc107_qa0__after148", "doc38_qa0__after148", "doc59_qa0__after148",
    "doc120_qa0__after148", "doc127_qa0__after148", "doc77_qa0__after148", "doc118_qa0__after148",
    "doc85_qa0__after148", "doc137_qa0__after148",
    "doc90_qa0__after149", "doc82_qa0__after149", "doc63_qa0__after149", "doc109_qa0__after149",
    "doc61_qa0__after149", "doc55_qa0__after149", "doc80_qa0__after149", "doc105_qa0__after149",
    "doc108_qa0__after149", "doc128_qa0__after149",
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
    print(f"part10 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
