"""Phase 1.9 — FB flat-corpus calibration part8 (entries 1050-1199)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__flat-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__flat-corpus__calibration__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

SPECIAL: dict[int, tuple[float, str]] = {
    0: (0.0, "[ANS] doc22 seen=106; pred=refuses on definitive ANS."),
    2: (0.0, "[ANS] doc25 seen=106; pred=refuses."),
    4: (0.0, "[ANS] doc62 seen=106; pred=refuses on definitive ANS."),
    5: (0.0, "[ANS] doc98 seen=106; pred=refuses on definitive ANS."),
    6: (0.0, "[ANS] doc1 seen=106; pred=refuses."),
    7: (0.25, "[ACK] doc138 not seen; PRED vague WK (operating efficiencies and cost management) — confident specific partially related."),
    9: (0.0, "[ANS] doc76 seen=106; pred=refuses on definitive ANS."),
    11: (0.0, "[ANS] doc70 seen=107; pred=refuses."),
    12: (0.0, "[ANS] doc28 seen=107; pred=refuses on definitive ANS."),
    13: (0.0, "[ANS] doc30 seen=107; pred=refuses."),
    14: (0.0, "[ANS] doc85 seen=107; pred=refuses on definitive ANS."),
    16: (0.0, "[ANS] doc87 seen=107; pred=refuses."),
    19: (0.0, "[ANS] doc49 seen=107; pred=refuses."),
    20: (0.0, "[ANS] doc9 seen=108; pred=refuses."),
    22: (0.0, "[ANS] doc13 seen=108; pred=refuses on definitive ANS."),
    25: (0.25, "[ACK] doc122 not seen; pred='0' confident-wrong."),
    27: (1.0, "[ANS] doc103 seen=108; gold=$303; pred=$302.578M (0.14% off — exact)."),
    29: (0.0, "[ANS] doc87 seen=108; pred=refuses."),
    30: (0.0, "[ANS] doc75 seen=109; pred=refuses."),
    31: (1.0, "[ANS] doc90 seen=109; gold=Consumer Health Aug 30; pred=exact."),
    32: (0.0, "[ANS] doc98 seen=109; pred=refuses on definitive ANS."),
    34: (0.0, "[ANS] doc42 seen=109; pred=refuses."),
    35: (0.0, "[ANS] doc43 seen=109; pred=refuses on definitive ANS."),
    36: (0.0, "[ANS] doc51 seen=109; pred=refuses on definitive ANS."),
    37: (0.0, "[ANS] doc68 seen=109; pred=refuses."),
    38: (0.0, "[ANS] doc45 seen=109; pred=refuses."),
    39: (1.0, "[ANS] doc108 seen=109; gold=MGM China worst 44% decline; pred=MGM China worst $674M = 44% decrease. Exact + figures."),
    40: (0.0, "[ANS] doc26 seen=110; pred=refuses on definitive ANS."),
    41: (0.0, "[ANS] doc7 seen=110; pred=refuses."),
    43: (0.0, "[ANS] doc14 seen=110; pred=refuses."),
    44: (0.0, "[ANS] doc44 seen=110; pred=refuses on definitive ANS."),
    45: (0.0, "[ANS] doc102 seen=110; pred=refuses."),
    46: (0.0, "[ANS] doc65 seen=110; pred=refuses."),
    48: (0.0, "[ANS] doc18 seen=110; pred=refuses."),
    51: (0.0, "[ANS] doc7 seen=111; pred=refuses."),
    52: (0.0, "[ANS] doc72 seen=111; pred=refuses on definitive ANS."),
    53: (0.0, "[ANS] doc35 seen=111; pred=refuses."),
    54: (0.0, "[ANS] doc99 seen=111; pred=refuses."),
    55: (0.0, "[ANS] doc33 seen=111; pred=refuses."),
    57: (0.0, "[ANS] doc90 seen=111; pred=refuses (gold = Consumer Health Aug 30)."),
    58: (0.0, "[ANS] doc97 seen=111; pred=refuses."),
    59: (0.25, "[ACK] doc122 not seen; pred='0' confident-wrong."),
    62: (0.0, "[ANS] doc26 seen=112; pred=refuses on definitive ANS."),
    64: (0.0, "[ANS] doc30 seen=112; pred=refuses."),
    65: (0.0, "[ANS] doc82 seen=112; pred=refuses."),
    66: (0.0, "[ANS] doc36 seen=112; pred=refuses."),
    67: (0.0, "[ANS] doc72 seen=112; pred=refuses on definitive ANS."),
    68: (0.0, "[ANS] doc37 seen=112; pred=refuses."),
    69: (0.0, "[ANS] doc101 seen=112; pred=refuses."),
    70: (0.0, "[ANS] doc35 seen=113; pred=refuses."),
    71: (0.0, "[ANS] doc52 seen=113; pred=refuses."),
    72: (0.0, "[ANS] doc23 seen=113; pred=refuses on definitive ANS."),
    74: (0.0, "[ANS] doc21 seen=113; pred=refuses."),
    75: (0.0, "[ANS] doc59 seen=113; pred=refuses."),
    77: (0.0, "[ANS] doc92 seen=113; pred=refuses."),
    78: (0.0, "[ANS] doc89 seen=113; pred=refuses."),
    79: (0.25, "[ACK] doc122 not seen; pred='0' confident-wrong."),
    81: (0.0, "[ANS] doc104 seen=114; pred=refuses."),
    83: (0.0, "[ANS] doc82 seen=114; pred=refuses."),
    84: (0.0, "[ANS] doc60 seen=114; pred=refuses on definitive ANS."),
    85: (0.0, "[ANS] doc89 seen=114; pred=refuses."),
    86: (0.0, "[ANS] doc47 seen=114; pred=refuses on definitive ANS."),
    88: (0.0, "[ANS] doc56 seen=114; pred=refuses."),
    89: (0.0, "[ANS] doc98 seen=114; pred=refuses on definitive ANS."),
    90: (0.0, "[ANS] doc24 seen=115; pred=refuses on definitive ANS."),
    91: (1.0, "[ANS] doc113 seen=115; gold=$5466; pred=5,466.312 exact."),
    92: (0.0, "[ANS] doc27 seen=115; pred=refuses."),
    94: (0.0, "[ANS] doc97 seen=115; pred=refuses."),
    95: (0.0, "[ANS] doc99 seen=115; pred=refuses."),
    97: (0.0, "[ANS] doc19 seen=115; pred=refuses."),
    98: (0.0, "[ANS] doc98 seen=115; pred=refuses on definitive ANS."),
    99: (0.0, "[ANS] doc12 seen=115; pred=refuses."),
    100: (0.0, "[ANS] doc80 seen=116; pred=refuses on definitive ANS."),
    101: (0.0, "[ANS] doc23 seen=116; pred=refuses on definitive ANS."),
    102: (0.0, "[ANS] doc111 seen=116; gold=No decreased $2.5B; pred='Yes increased $47032→$41990' (text shows decrease, label says Yes increased) — Y/N flip and self-contradictory."),
    104: (0.0, "[ANS] doc33 seen=116; pred=refuses."),
    105: (0.0, "[ANS] doc87 seen=116; pred=refuses."),
    107: (0.0, "[ANS] doc81 seen=116; pred=refuses."),
    109: (0.0, "[ANS] doc68 seen=116; pred=refuses."),
    110: (0.0, "[ANS] doc4 seen=117; pred=refuses on definitive ANS."),
    111: (0.0, "[ANS] doc66 seen=117; pred=refuses."),
    114: (0.0, "[ANS] doc88 seen=117; pred=refuses on definitive ANS."),
    115: (0.0, "[ANS] doc93 seen=117; pred=refuses on definitive ANS."),
    116: (0.0, "[ANS] doc105 seen=117; pred=refuses on definitive ANS."),
    117: (0.0, "[ANS] doc44 seen=117; pred=refuses on definitive ANS."),
    118: (0.0, "[ANS] doc104 seen=117; pred=refuses."),
    119: (0.0, "[ANS] doc21 seen=117; pred=refuses."),
    122: (0.0, "[ANS] doc6 seen=118; pred=refuses on definitive ANS."),
    123: (0.0, "[ANS] doc44 seen=118; pred=refuses on definitive ANS."),
    124: (0.0, "[ANS] doc42 seen=118; pred=refuses."),
    125: (0.0, "[ANS] doc54 seen=118; pred=refuses on definitive ANS."),
    126: (0.0, "[ANS] doc2 seen=118; pred=refuses on definitive ANS."),
    129: (0.0, "[ANS] doc3 seen=118; pred=refuses on definitive ANS."),
    130: (0.0, "[ANS] doc107 seen=119; pred=refuses on definitive ANS."),
    131: (0.0, "[ANS] doc93 seen=119; pred=refuses on definitive ANS."),
    132: (0.0, "[ANS] doc4 seen=119; pred=refuses on definitive ANS."),
    134: (0.0, "[ANS] doc22 seen=119; pred=refuses on definitive ANS."),
    135: (0.0, "[ANS] doc37 seen=119; pred=refuses."),
    136: (0.0, "[ANS] doc73 seen=119; pred=refuses on definitive ANS."),
    137: (0.0, "[ANS] doc45 seen=119; pred=refuses."),
    138: (0.0, "[ANS] doc41 seen=119; pred=refuses on definitive ANS."),
    139: (0.0, "[ANS] doc34 seen=119; pred=refuses."),
    140: (1.0, "[ANS] doc15 seen=120; gold=0; pred=0 exact."),
    142: (0.0, "[ANS] doc45 seen=120; pred=refuses."),
    143: (0.0, "[ANS] doc49 seen=120; pred=refuses."),
    144: (0.0, "[ANS] doc68 seen=120; pred=refuses."),
    145: (0.0, "[ANS] doc48 seen=120; pred=refuses."),
    146: (0.0, "[ANS] doc25 seen=120; pred=refuses."),
    148: (0.0, "[ANS] doc59 seen=120; pred=refuses."),
    149: (0.75, "[ANS] doc52 seen=120; gold=operations highest CF; pred='operations brought most cash flow' (no $). Match."),
}

DEFAULT_RATIONALE = "[ACK] source doc not yet seen in FlatMemory window. PRED honest refusal — correct."

ENTRY_SUFFIXES: list[str] = [
    "doc22_qa0__after105", "doc119_qa0__after105", "doc25_qa0__after105", "doc146_qa0__after105",
    "doc62_qa0__after105", "doc98_qa0__after105", "doc1_qa0__after105", "doc138_qa0__after105",
    "doc123_qa0__after105", "doc76_qa0__after105",
    "doc124_qa0__after106", "doc70_qa0__after106", "doc28_qa0__after106", "doc30_qa0__after106",
    "doc85_qa0__after106", "doc130_qa0__after106", "doc87_qa0__after106", "doc135_qa0__after106",
    "doc148_qa0__after106", "doc49_qa0__after106",
    "doc9_qa0__after107", "doc134_qa0__after107", "doc13_qa0__after107", "doc142_qa0__after107",
    "doc127_qa0__after107", "doc122_qa0__after107", "doc133_qa0__after107", "doc103_qa0__after107",
    "doc139_qa0__after107", "doc87_qa0__after107",
    "doc75_qa0__after108", "doc90_qa0__after108", "doc98_qa0__after108", "doc140_qa0__after108",
    "doc42_qa0__after108", "doc43_qa0__after108", "doc51_qa0__after108", "doc68_qa0__after108",
    "doc45_qa0__after108", "doc108_qa0__after108",
    "doc26_qa0__after109", "doc7_qa0__after109", "doc119_qa0__after109", "doc14_qa0__after109",
    "doc44_qa0__after109", "doc102_qa0__after109", "doc65_qa0__after109", "doc133_qa0__after109",
    "doc18_qa0__after109", "doc134_qa0__after109",
    "doc120_qa0__after110", "doc7_qa0__after110", "doc72_qa0__after110", "doc35_qa0__after110",
    "doc99_qa0__after110", "doc33_qa0__after110", "doc145_qa0__after110", "doc90_qa0__after110",
    "doc97_qa0__after110", "doc122_qa0__after110",
    "doc117_qa0__after111", "doc120_qa0__after111", "doc26_qa0__after111", "doc113_qa0__after111",
    "doc30_qa0__after111", "doc82_qa0__after111", "doc36_qa0__after111", "doc72_qa0__after111",
    "doc37_qa0__after111", "doc101_qa0__after111",
    "doc35_qa0__after112", "doc52_qa0__after112", "doc23_qa0__after112", "doc120_qa0__after112",
    "doc21_qa0__after112", "doc59_qa0__after112", "doc114_qa0__after112", "doc92_qa0__after112",
    "doc89_qa0__after112", "doc122_qa0__after112",
    "doc139_qa0__after113", "doc104_qa0__after113", "doc136_qa0__after113", "doc82_qa0__after113",
    "doc60_qa0__after113", "doc89_qa0__after113", "doc47_qa0__after113", "doc137_qa0__after113",
    "doc56_qa0__after113", "doc98_qa0__after113",
    "doc24_qa0__after114", "doc113_qa0__after114", "doc27_qa0__after114", "doc124_qa0__after114",
    "doc97_qa0__after114", "doc99_qa0__after114", "doc131_qa0__after114", "doc19_qa0__after114",
    "doc98_qa0__after114", "doc12_qa0__after114",
    "doc80_qa0__after115", "doc23_qa0__after115", "doc111_qa0__after115", "doc131_qa0__after115",
    "doc33_qa0__after115", "doc87_qa0__after115", "doc140_qa0__after115", "doc81_qa0__after115",
    "doc121_qa0__after115", "doc68_qa0__after115",
    "doc4_qa0__after116", "doc66_qa0__after116", "doc120_qa0__after116", "doc138_qa0__after116",
    "doc88_qa0__after116", "doc93_qa0__after116", "doc105_qa0__after116", "doc44_qa0__after116",
    "doc104_qa0__after116", "doc21_qa0__after116",
    "doc146_qa0__after117", "doc131_qa0__after117", "doc6_qa0__after117", "doc44_qa0__after117",
    "doc42_qa0__after117", "doc54_qa0__after117", "doc2_qa0__after117", "doc148_qa0__after117",
    "doc121_qa0__after117", "doc3_qa0__after117",
    "doc107_qa0__after118", "doc93_qa0__after118", "doc4_qa0__after118", "doc133_qa0__after118",
    "doc22_qa0__after118", "doc37_qa0__after118", "doc73_qa0__after118", "doc45_qa0__after118",
    "doc41_qa0__after118", "doc34_qa0__after118",
    "doc15_qa0__after119", "doc142_qa0__after119", "doc45_qa0__after119", "doc49_qa0__after119",
    "doc68_qa0__after119", "doc48_qa0__after119", "doc25_qa0__after119", "doc146_qa0__after119",
    "doc59_qa0__after119", "doc52_qa0__after119",
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
    print(f"part8 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
