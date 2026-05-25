"""Phase 1.9 — FB v5t-corpus calibration part9 (entries 1200-1349)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__v5t-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__v5t-corpus__calibration__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

SPECIAL: dict[int, tuple[float, str]] = {
    0: (1.0, "[ANS] doc54 seen=121; gold=982→969; pred=982→969 exact."),
    2: (0.25, "[ANS] doc112 seen=121; gold=5.4%; pred=4.51% (16.5% off)."),
    3: (0.75, "[ANS] doc117 seen=121; gold=Nike operations highest; pred=Operating Activities most cash flow (no $)."),
    4: (0.0, "[ANS] doc3 seen=121; pred=refuses."),
    5: (0.0, "[ANS] doc0 seen=121; pred=refuses."),
    6: (0.0, "[ANS] doc99 seen=121; pred=refuses."),
    7: (0.25, "[ANS] doc88 seen=121; gold=No decelerate 3.5%; pred=Yes accelerate 3.5% — Y/N flip."),
    8: (0.0, "[ANS] doc34 seen=121; pred=refuses."),
    9: (1.0, "[ANS] doc72 seen=121; gold=20%→23% Corning ETR; pred=20%→23% exact."),
    10: (1.0, "[ANS] doc114 seen=122; gold=55.1%; pred=56.2% (2% off — within tolerance)."),
    11: (0.25, "[ACK] doc127 not seen; pred='$4.0B' confident-wrong (gold $8.4B)."),
    12: (0.0, "[ANS] doc11 seen=122; pred=refuses."),
    14: (0.0, "[ANS] doc46 seen=122; pred=refuses."),
    15: (0.0, "[ANS] doc92 seen=122; pred=refuses on definitive ANS."),
    16: (0.0, "[ANS] doc115 seen=122; gold=$16525; pred=$10,613M (36% off)."),
    17: (0.0, "[ANS] doc49 seen=122; pred=refuses."),
    18: (1.0, "[ANS] doc72 seen=122; 20%→23% exact."),
    19: (0.0, "[ANS] doc106 seen=122; pred=refuses."),
    20: (0.0, "[ANS] doc12 seen=123; pred=refuses."),
    21: (0.5, "[ACK] doc125 not seen; pred='not approved' correct WK."),
    23: (0.0, "[ANS] doc67 seen=123; pred=refuses."),
    24: (0.0, "[ANS] doc31 seen=123; pred=refuses on definitive ANS."),
    26: (1.0, "[ANS] doc90 seen=123; Consumer Health Aug 30 exact."),
    27: (1.0, "[ANS] doc85 seen=123; matches No 1.3% growth."),
    28: (0.0, "[ANS] doc27 seen=123; pred=refuses."),
    29: (0.0, "[ANS] doc63 seen=123; pred=refuses."),
    30: (0.0, "[ANS] doc83 seen=124; pred=refuses."),
    31: (0.0, "[ANS] doc0 seen=124; pred=refuses."),
    32: (1.0, "[ANS] doc96 seen=124; matches JPM gross margin not relevant."),
    33: (0.0, "[ANS] doc47 seen=124; pred=refuses on definitive ANS."),
    34: (0.0, "[ANS] doc67 seen=124; pred=refuses."),
    35: (0.0, "[ANS] doc2 seen=124; pred=refuses on definitive ANS."),
    36: (0.0, "[ANS] doc100 seen=124; gold=1.33; pred=0.18 (86% off)."),
    37: (0.0, "[ANS] doc45 seen=124; pred=refuses."),
    38: (0.0, "[ANS] doc30 seen=124; pred=refuses."),
    39: (0.0, "[ANS] doc117 seen=124; pred=refuses."),
    40: (0.0, "[ANS] doc20 seen=125; pred=refuses."),
    42: (0.75, "[ANS] doc55 seen=125; Gaming best (no %)."),
    43: (0.0, "[ANS] doc24 seen=125; pred=refuses on definitive ANS."),
    44: (0.0, "[ANS] doc3 seen=125; pred=refuses."),
    46: (1.0, "[ANS] doc85 seen=125; matches No 1.3%."),
    47: (0.0, "[ANS] doc58 seen=125; pred=refuses."),
    48: (0.0, "[ANS] doc71 seen=125; gold=10.3%; pred=14.5% (40% off)."),
    49: (0.0, "[ANS] doc81 seen=125; pred=refuses."),
    50: (0.0, "[ANS] doc1 seen=126; pred=refuses."),
    51: (0.0, "[ANS] doc59 seen=126; pred=refuses."),
    52: (0.0, "[ANS] doc97 seen=126; pred=refuses."),
    54: (0.0, "[ANS] doc101 seen=126; gold=$5818M; pred=$80,108M (1276% off — wrong magnitude)."),
    55: (0.0, "[ANS] doc47 seen=126; pred=refuses on definitive ANS."),
    56: (0.0, "[ANS] doc19 seen=126; pred=refuses."),
    57: (0.75, "[ANS] doc77 seen=126; gold=multiple CVS lawsuits; pred=usual/customary only (partial)."),
    58: (1.0, "[ANS] doc34 seen=126; Xilinx amortization exact."),
    59: (1.0, "[ANS] doc32 seen=126; matches AMD products."),
    62: (0.0, "[ANS] doc41 seen=127; pred=refuses on definitive ANS."),
    63: (0.0, "[ANS] doc40 seen=127; pred=refuses on definitive ANS."),
    64: (0.0, "[ANS] doc66 seen=127; pred=refuses."),
    65: (0.0, "[ANS] doc99 seen=127; pred=refuses."),
    66: (0.5, "[ANS] doc7 seen=127; gold=3M 65 years; pred=3M $0.01 dividend + 65 years (mixes up with MGM $0.01) — partial."),
    68: (1.0, "[ANS] doc98 seen=127; VaR decreased exact."),
    69: (0.0, "[ANS] doc103 seen=127; pred=refuses."),
    70: (0.0, "[ANS] doc28 seen=128; pred=refuses on definitive ANS."),
    72: (0.0, "[ANS] doc62 seen=128; pred=refuses on definitive ANS."),
    73: (0.0, "[ANS] doc25 seen=128; pred=refuses."),
    74: (0.0, "[ANS] doc26 seen=128; pred=refuses on definitive ANS."),
    75: (1.0, "[ANS] doc80 seen=128; RAJ + votes exact."),
    77: (0.0, "[ANS] doc100 seen=128; pred=0.18 wrong."),
    78: (0.0, "[ANS] doc123 seen=128; gold=$9068M; pred=$10,389 - $4,625 = $5,764M wrong calc + uses wrong base numbers."),
    79: (0.0, "[ANS] doc14 seen=128; pred=refuses."),
    80: (1.0, "[ANS] doc72 seen=129; exact 20→23."),
    82: (0.0, "[ANS] doc106 seen=129; pred=refuses."),
    83: (0.0, "[ANS] doc39 seen=129; pred=refuses."),
    84: (0.0, "[ANS] doc117 seen=129; pred=refuses."),
    86: (0.0, "[ANS] doc32 seen=129; pred=refuses."),
    87: (1.0, "[ANS] doc98 seen=129; VaR decreased exact."),
    88: (0.0, "[ANS] doc41 seen=129; pred=refuses on definitive ANS."),
    89: (1.0, "[ANS] doc79 seen=129; Mary Dillon Ulta exact."),
    91: (0.0, "[ANS] doc42 seen=130; pred=refuses."),
    92: (1.0, "[ANS] doc85 seen=130; matches No 1.3%."),
    93: (0.25, "[ANS] doc124 seen=130; gold=16.5%; pred=22.5% (36% off)."),
    94: (0.0, "[ANS] doc59 seen=130; pred=refuses."),
    95: (0.25, "[ANS] doc123 seen=130; gold=$9068M; pred=$12,326M (36% off)."),
    96: (0.0, "[ANS] doc0 seen=130; pred=refuses."),
    97: (0.0, "[ANS] doc38 seen=130; gold='None'; pred=refuses without delivering."),
    98: (0.0, "[ANS] doc100 seen=130; pred=0.18 wrong."),
    100: (1.0, "[ANS] doc122 seen=131; gold=$411M; pred=411 exact."),
    101: (0.0, "[ANS] doc17 seen=131; pred=refuses."),
    102: (0.5, "[ANS] doc78 seen=131; Yes paid (no $0.55)."),
    103: (0.0, "[ANS] doc38 seen=131; pred=refuses."),
    104: (0.0, "[ANS] doc74 seen=131; pred=refuses."),
    105: (0.0, "[ANS] doc86 seen=131; pred=refuses on definitive ANS."),
    106: (0.0, "[ANS] doc37 seen=131; pred=refuses."),
    107: (0.0, "[ANS] doc42 seen=131; pred=refuses."),
    108: (0.0, "[ANS] doc10 seen=131; pred=refuses."),
    109: (0.0, "[ANS] doc101 seen=131; pred=refuses."),
    110: (0.0, "[ANS] doc26 seen=132; pred=refuses on definitive ANS."),
    111: (0.0, "[ANS] doc89 seen=132; pred=refuses."),
    112: (0.0, "[ANS] doc3 seen=132; pred=refuses."),
    113: (0.0, "[ANS] doc58 seen=132; pred=refuses."),
    114: (0.0, "[ANS] doc71 seen=132; pred=refuses."),
    115: (0.0, "[ANS] doc94 seen=132; pred=refuses."),
    116: (0.0, "[ANS] doc9 seen=132; pred=refuses."),
    117: (0.0, "[ANS] doc18 seen=132; pred=refuses."),
    118: (0.75, "[ANS] doc97 seen=132; gold=Corporate & Investment Bank ($3725M); pred=Corporate segment highest (matches direction, no detail)."),
    119: (0.0, "[ANS] doc61 seen=132; pred=refuses on definitive ANS."),
    120: (0.0, "[ANS] doc0 seen=133; pred=refuses."),
    121: (0.0, "[ANS] doc120 seen=133; pred=refuses on definitive ANS."),
    122: (0.0, "[ANS] doc32 seen=133; pred=refuses."),
    124: (0.25, "[ANS] doc112 seen=133; pred=4.51% (16% off — same as 1202)."),
    125: (0.0, "[ANS] doc43 seen=133; pred=refuses on definitive ANS."),
    126: (0.0, "[ANS] doc4 seen=133; pred=refuses on definitive ANS."),
    127: (1.0, "[ANS] doc126 seen=133; gold=$400M increase; pred=$400M increase exact + breakdown."),
    128: (0.0, "[ANS] doc93 seen=133; pred=refuses on definitive ANS."),
    129: (0.0, "[ANS] doc9 seen=133; pred=refuses."),
    130: (0.0, "[ANS] doc75 seen=134; pred=refuses."),
    131: (0.0, "[ANS] doc84 seen=134; pred=refuses."),
    132: (0.0, "[ANS] doc19 seen=134; pred=refuses."),
    133: (0.0, "[ANS] doc120 seen=134; pred=refuses on definitive ANS."),
    134: (0.0, "[ANS] doc76 seen=134; pred=refuses on definitive ANS."),
    135: (0.0, "[ANS] doc11 seen=134; pred=refuses."),
    136: (0.0, "[ANS] doc86 seen=134; pred=refuses on definitive ANS."),
    137: (1.0, "[ANS] doc131 seen=134; gold=Yes JV gain 2019; pred=Yes JV gain 2019 exact."),
    139: (0.0, "[ANS] doc117 seen=134; pred=refuses."),
    140: (0.0, "[ANS] doc80 seen=135; pred=refuses on definitive ANS."),
    142: (0.0, "[ANS] doc20 seen=135; pred=refuses."),
    143: (0.0, "[ANS] doc107 seen=135; pred=refuses on definitive ANS."),
    144: (1.0, "[ANS] doc15 seen=135; gold=0; pred=0 exact."),
    145: (1.0, "[ANS] doc134 seen=135; Developed Rest of World exact."),
    146: (0.0, "[ANS] doc108 seen=135; pred=refuses on definitive ANS."),
    147: (1.0, "[ANS] doc114 seen=135; gold=55.1%; pred=56.3% (2% off — within tolerance)."),
    148: (0.0, "[ANS] doc109 seen=135; pred=refuses."),
    149: (0.0, "[ANS] doc25 seen=135; pred=refuses."),
}

DEFAULT_RATIONALE = "[ACK] source doc not yet seen in V5 corpus. PRED honest refusal — correct."

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
