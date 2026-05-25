"""Phase 1.9 — FB semantic-corpus calibration part9 (entries 1200-1349)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__semantic-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__semantic-corpus__calibration__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

SPECIAL: dict[int, tuple[float, str]] = {
    0: (0.0, "[ANS] doc54 seen=121; pred=refuses."),
    1: (0.0, "[ACK] doc121 not seen; pred='Yes Combat Arms litigation' confidently wrong (Combat Arms is 3M, not Pepsi; gold=No legal battles) — fabrication."),
    2: (0.0, "[ANS] doc112 seen=121; pred=refuses."),
    3: (0.0, "[ANS] doc117 seen=121; pred=refuses."),
    5: (0.75, "[ANS] doc0 seen=121; gold=$1577; pred=$1,501 (4.8% off — within 5% tolerance, close)."),
    6: (0.0, "[ANS] doc99 seen=121; pred=refuses."),
    7: (0.0, "[ANS] doc88 seen=121; pred=refuses."),
    8: (0.0, "[ANS] doc34 seen=121; pred=refuses."),
    9: (0.0, "[ANS] doc72 seen=121; pred=refuses."),
    10: (0.0, "[ANS] doc114 seen=122; pred=refuses."),
    11: (0.25, "[ACK] doc127 not seen; pred='$4.0B' confident-wrong (gold $8.4B)."),
    12: (0.0, "[ANS] doc11 seen=122; pred=refuses."),
    14: (0.0, "[ANS] doc46 seen=122; pred=refuses."),
    15: (0.0, "[ANS] doc92 seen=122; pred=refuses."),
    16: (0.0, "[ANS] doc115 seen=122; pred=refuses."),
    17: (0.0, "[ANS] doc49 seen=122; pred=refuses."),
    18: (0.0, "[ANS] doc72 seen=122; pred=refuses."),
    19: (0.0, "[ANS] doc106 seen=122; pred=refuses."),
    20: (0.0, "[ANS] doc12 seen=123; pred=refuses."),
    21: (0.5, "[ACK] doc125 not seen; PRED='not approved' WK."),
    23: (0.0, "[ANS] doc67 seen=123; pred=refuses."),
    24: (0.0, "[ANS] doc31 seen=123; pred=refuses."),
    27: (0.0, "[ANS] doc85 seen=123; pred=refuses."),
    28: (0.0, "[ANS] doc27 seen=123; pred=refuses."),
    29: (0.0, "[ANS] doc63 seen=123; pred=refuses."),
    30: (0.0, "[ANS] doc83 seen=124; pred=refuses."),
    31: (0.75, "[ANS] doc0 seen=124; pred=$1,501 close to gold $1577."),
    33: (0.0, "[ANS] doc47 seen=124; pred=refuses."),
    34: (0.0, "[ANS] doc67 seen=124; pred=refuses."),
    35: (0.0, "[ANS] doc2 seen=124; Y/N flip."),
    36: (0.0, "[ANS] doc100 seen=124; pred=refuses."),
    37: (0.0, "[ANS] doc45 seen=124; pred=refuses."),
    38: (0.0, "[ANS] doc30 seen=124; pred=refuses."),
    39: (0.0, "[ANS] doc117 seen=124; pred=refuses."),
    40: (0.0, "[ANS] doc20 seen=125; pred=refuses."),
    42: (0.0, "[ANS] doc55 seen=125; pred=refuses."),
    43: (0.0, "[ANS] doc24 seen=125; pred=refuses."),
    46: (0.0, "[ANS] doc85 seen=125; pred=refuses."),
    47: (0.0, "[ANS] doc58 seen=125; pred=refuses."),
    48: (0.0, "[ANS] doc71 seen=125; pred=refuses."),
    49: (0.0, "[ANS] doc81 seen=125; pred=refuses."),
    51: (0.0, "[ANS] doc59 seen=126; pred=refuses."),
    52: (0.0, "[ANS] doc97 seen=126; pred=refuses."),
    54: (0.0, "[ANS] doc101 seen=126; pred=refuses."),
    55: (0.0, "[ANS] doc47 seen=126; pred=refuses."),
    56: (0.0, "[ANS] doc19 seen=126; pred=refuses."),
    57: (0.0, "[ANS] doc77 seen=126; pred=refuses."),
    58: (0.0, "[ANS] doc34 seen=126; pred=refuses."),
    59: (0.0, "[ANS] doc32 seen=126; pred=refuses."),
    62: (0.0, "[ANS] doc41 seen=127; pred=refuses."),
    63: (0.0, "[ANS] doc40 seen=127; pred=refuses."),
    64: (0.0, "[ANS] doc66 seen=127; pred=refuses."),
    65: (0.0, "[ANS] doc99 seen=127; pred=refuses."),
    66: (0.0, "[ANS] doc7 seen=127; pred=refuses."),
    68: (0.0, "[ANS] doc98 seen=127; pred=refuses."),
    69: (0.0, "[ANS] doc103 seen=127; pred=refuses."),
    70: (0.0, "[ANS] doc28 seen=128; pred=refuses."),
    72: (0.0, "[ANS] doc62 seen=128; pred=refuses."),
    73: (0.0, "[ANS] doc25 seen=128; pred=refuses."),
    74: (0.0, "[ANS] doc26 seen=128; pred=refuses."),
    75: (0.0, "[ANS] doc80 seen=128; pred=refuses."),
    77: (0.0, "[ANS] doc100 seen=128; pred=refuses."),
    78: (0.0, "[ANS] doc123 seen=128; pred=refuses."),
    79: (0.0, "[ANS] doc14 seen=128; pred=refuses."),
    80: (0.0, "[ANS] doc72 seen=129; pred=refuses."),
    82: (0.0, "[ANS] doc106 seen=129; pred=refuses."),
    83: (0.0, "[ANS] doc39 seen=129; pred=refuses."),
    84: (0.0, "[ANS] doc117 seen=129; pred=refuses."),
    86: (0.0, "[ANS] doc32 seen=129; pred=refuses."),
    87: (0.0, "[ANS] doc98 seen=129; pred=refuses."),
    88: (0.0, "[ANS] doc41 seen=129; pred=refuses."),
    89: (0.0, "[ANS] doc79 seen=129; pred=refuses."),
    91: (0.0, "[ANS] doc42 seen=130; pred=refuses."),
    92: (0.0, "[ANS] doc85 seen=130; pred=refuses."),
    93: (0.25, "[ANS] doc124 seen=130; pred=19.4% vs gold 16.5% (17.6% off — outside tolerance)."),
    94: (0.0, "[ANS] doc59 seen=130; pred=refuses."),
    95: (0.0, "[ANS] doc123 seen=130; pred=refuses."),
    96: (0.75, "[ANS] doc0 seen=130; pred=$1,501 close."),
    97: (0.0, "[ANS] doc38 seen=130; pred=refuses."),
    98: (0.0, "[ANS] doc100 seen=130; pred=refuses."),
    100: (0.0, "[ANS] doc122 seen=131; pred='0' wrong (gold $411M)."),
    101: (0.0, "[ANS] doc17 seen=131; pred=refuses."),
    102: (0.0, "[ANS] doc78 seen=131; pred=refuses."),
    103: (0.0, "[ANS] doc38 seen=131; pred=refuses."),
    104: (0.25, "[ANS] doc74 seen=131; pred=$52,694 vs gold $59,268 (11% off — outside tolerance)."),
    105: (0.0, "[ANS] doc86 seen=131; pred=refuses."),
    106: (0.0, "[ANS] doc37 seen=131; pred=refuses."),
    107: (0.0, "[ANS] doc42 seen=131; pred=refuses."),
    108: (0.0, "[ANS] doc10 seen=131; pred=refuses."),
    109: (0.0, "[ANS] doc101 seen=131; pred=refuses."),
    110: (0.0, "[ANS] doc26 seen=132; pred=refuses."),
    111: (0.0, "[ANS] doc89 seen=132; pred=refuses."),
    113: (0.0, "[ANS] doc58 seen=132; pred=refuses."),
    114: (0.0, "[ANS] doc71 seen=132; pred=refuses."),
    115: (0.0, "[ANS] doc94 seen=132; pred=refuses."),
    116: (0.0, "[ANS] doc9 seen=132; pred=refuses."),
    117: (0.0, "[ANS] doc18 seen=132; pred=refuses."),
    118: (0.0, "[ANS] doc97 seen=132; pred=refuses."),
    119: (0.0, "[ANS] doc61 seen=132; pred=refuses."),
    120: (0.75, "[ANS] doc0 seen=133; pred=$1,501 close."),
    121: (0.5, "[ANS] doc120 seen=133; pred='US/Canada/LatAm/Europe' partial (missing Africa/ME/Asia/Australia)."),
    122: (0.0, "[ANS] doc32 seen=133; pred=refuses."),
    124: (0.0, "[ANS] doc112 seen=133; pred=refuses."),
    125: (0.0, "[ANS] doc43 seen=133; gold='Customer deposits'; pred='long-term debt' wrong."),
    126: (0.5, "[ANS] doc4 seen=133; pred='Consumer segment' partial."),
    127: (0.25, "[ANS] doc126 seen=133; pred='$1.5B' wrong (gold $400M)."),
    128: (0.0, "[ANS] doc93 seen=133; pred=refuses."),
    129: (0.0, "[ANS] doc9 seen=133; pred=refuses."),
    130: (0.0, "[ANS] doc75 seen=134; pred=refuses."),
    131: (0.0, "[ANS] doc84 seen=134; pred=refuses."),
    132: (0.0, "[ANS] doc19 seen=134; pred=refuses."),
    133: (0.5, "[ANS] doc120 seen=134; pred partial."),
    134: (0.0, "[ANS] doc76 seen=134; pred=refuses."),
    135: (0.0, "[ANS] doc11 seen=134; pred=refuses."),
    136: (0.0, "[ANS] doc86 seen=134; pred=refuses."),
    137: (0.0, "[ANS] doc131 seen=134; pred=refuses."),
    139: (0.0, "[ANS] doc117 seen=134; pred=refuses."),
    140: (0.0, "[ANS] doc80 seen=135; pred=refuses."),
    142: (0.0, "[ANS] doc20 seen=135; pred=refuses."),
    143: (0.0, "[ANS] doc107 seen=135; pred=refuses."),
    145: (0.0, "[ANS] doc134 seen=135; pred=refuses."),
    146: (0.0, "[ANS] doc108 seen=135; pred=refuses."),
    147: (0.0, "[ANS] doc114 seen=135; pred=refuses."),
    148: (0.0, "[ANS] doc109 seen=135; pred=refuses."),
    149: (0.0, "[ANS] doc25 seen=135; pred=refuses."),
}

DEFAULT_RATIONALE = "[ACK] source doc not yet seen in semantic corpus. PRED honest refusal — correct."

ENTRY_SUFFIXES: list[str] = [
    "doc54_qa0__after120", "doc121_qa0__after120", "doc112_qa0__after120", "doc117_qa0__after120",
    "doc3_qa0__after120", "doc0_qa0__after120", "doc99_qa0__after120", "doc88_qa0__after120",
    "doc34_qa0__after120", "doc72_qa0__after120",
    "doc114_qa0__after121", "doc127_qa0__after121", "doc11_qa0__after121", "doc136_qa0__after121",
    "doc46_qa0__after121", "doc92_qa0__after121", "doc115_qa0__after121", "doc49_qa0__after121",
    "doc72_qa0__after121", "doc106_qa0__after121",
    "doc12_qa0__after122", "doc125_qa0__after122", "doc128_qa0__after122", "doc67_qa0__after122",
    "doc31_qa0__after122", "doc134_qa0__after122", "doc90_qa0__after122", "doc85_qa0__after122",
    "doc27_qa0__after122", "doc63_qa0__after122",
    "doc83_qa0__after123", "doc0_qa0__after123", "doc96_qa0__after123", "doc47_qa0__after123",
    "doc67_qa0__after123", "doc2_qa0__after123", "doc100_qa0__after123", "doc45_qa0__after123",
    "doc30_qa0__after123", "doc117_qa0__after123",
    "doc20_qa0__after124", "doc128_qa0__after124", "doc55_qa0__after124", "doc24_qa0__after124",
    "doc3_qa0__after124", "doc139_qa0__after124", "doc85_qa0__after124", "doc58_qa0__after124",
    "doc71_qa0__after124", "doc81_qa0__after124",
    "doc1_qa0__after125", "doc59_qa0__after125", "doc97_qa0__after125", "doc143_qa0__after125",
    "doc101_qa0__after125", "doc47_qa0__after125", "doc19_qa0__after125", "doc77_qa0__after125",
    "doc34_qa0__after125", "doc32_qa0__after125",
    "doc132_qa0__after126", "doc130_qa0__after126", "doc41_qa0__after126", "doc40_qa0__after126",
    "doc66_qa0__after126", "doc99_qa0__after126", "doc7_qa0__after126", "doc142_qa0__after126",
    "doc98_qa0__after126", "doc103_qa0__after126",
    "doc28_qa0__after127", "doc130_qa0__after127", "doc62_qa0__after127", "doc25_qa0__after127",
    "doc26_qa0__after127", "doc80_qa0__after127", "doc135_qa0__after127", "doc100_qa0__after127",
    "doc123_qa0__after127", "doc14_qa0__after127",
    "doc72_qa0__after128", "doc131_qa0__after128", "doc106_qa0__after128", "doc39_qa0__after128",
    "doc117_qa0__after128", "doc141_qa0__after128", "doc32_qa0__after128", "doc98_qa0__after128",
    "doc41_qa0__after128", "doc79_qa0__after128",
    "doc134_qa0__after129", "doc42_qa0__after129", "doc85_qa0__after129", "doc124_qa0__after129",
    "doc59_qa0__after129", "doc123_qa0__after129", "doc0_qa0__after129", "doc38_qa0__after129",
    "doc100_qa0__after129", "doc146_qa0__after129",
    "doc122_qa0__after130", "doc17_qa0__after130", "doc78_qa0__after130", "doc38_qa0__after130",
    "doc74_qa0__after130", "doc86_qa0__after130", "doc37_qa0__after130", "doc42_qa0__after130",
    "doc10_qa0__after130", "doc101_qa0__after130",
    "doc26_qa0__after131", "doc89_qa0__after131", "doc3_qa0__after131", "doc58_qa0__after131",
    "doc71_qa0__after131", "doc94_qa0__after131", "doc9_qa0__after131", "doc18_qa0__after131",
    "doc97_qa0__after131", "doc61_qa0__after131",
    "doc0_qa0__after132", "doc120_qa0__after132", "doc32_qa0__after132", "doc141_qa0__after132",
    "doc112_qa0__after132", "doc43_qa0__after132", "doc4_qa0__after132", "doc126_qa0__after132",
    "doc93_qa0__after132", "doc9_qa0__after132",
    "doc75_qa0__after133", "doc84_qa0__after133", "doc19_qa0__after133", "doc120_qa0__after133",
    "doc76_qa0__after133", "doc11_qa0__after133", "doc86_qa0__after133", "doc131_qa0__after133",
    "doc148_qa0__after133", "doc117_qa0__after133",
    "doc80_qa0__after134", "doc143_qa0__after134", "doc20_qa0__after134", "doc107_qa0__after134",
    "doc15_qa0__after134", "doc134_qa0__after134", "doc108_qa0__after134", "doc114_qa0__after134",
    "doc109_qa0__after134", "doc25_qa0__after134",
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
    print(f"part9 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
