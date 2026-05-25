"""Phase 1.9 — FB v5t-corpus calibration part7 (entries 900-1049)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__v5t-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__v5t-corpus__calibration__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

SPECIAL: dict[int, tuple[float, str]] = {
    0: (0.0, "[ANS] doc66 seen=91; pred=refuses."),
    2: (0.0, "[ANS] doc30 seen=91; pred=refuses."),
    4: (0.0, "[ANS] doc41 seen=91; pred=refuses on definitive ANS."),
    5: (0.0, "[ANS] doc45 seen=91; pred=refuses."),
    6: (0.0, "[ANS] doc5 seen=91; pred=refuses on definitive ANS."),
    11: (0.25, "[ANS] doc88 seen=92; gold=No decelerate 3.5%; pred=Yes 3.5% — same number but Y/N flip."),
    12: (1.0, "[ANS] doc79 seen=92; gold=Yes Mary Dillon Ulta CEO; pred=Yes Mary Dillon Ulta CEO exact."),
    13: (1.0, "[ANS] doc33 seen=92; gold=AMD drivers; pred=Data Center 64% + Gaming 21% + Embedded match."),
    14: (1.0, "[ANS] doc20 seen=92; gold=$11588; pred=$11,588M exact."),
    15: (0.0, "[ANS] doc40 seen=92; pred=refuses on definitive ANS."),
    16: (1.0, "[ANS] doc86 seen=92; gold=JnJ gross margin drivers (COVID, currency, commodity); pred=matches."),
    17: (1.0, "[ANS] doc15 seen=92; gold=0; pred=0 exact."),
    19: (0.0, "[ANS] doc18 seen=92; pred=refuses."),
    21: (0.0, "[ANS] doc45 seen=93; pred=refuses."),
    23: (0.5, "[ANS] doc78 seen=93; gold=Yes $0.55/qtr; pred=Yes paid dividends (no $)."),
    24: (0.0, "[ANS] doc91 seen=93; pred=refuses on definitive ANS."),
    25: (0.0, "[ANS] doc10 seen=93; gold=0.66; pred=1.31 (99% off)."),
    26: (0.0, "[ANS] doc12 seen=93; pred=refuses."),
    28: (1.0, "[ANS] doc86 seen=93; matches drivers."),
    29: (0.25, "[ACK] doc122 not seen; pred='0' confident-wrong."),
    30: (0.0, "[ANS] doc26 seen=94; pred=refuses on definitive ANS."),
    31: (1.0, "[ANS] doc64 seen=94; gold=Yes cyclical; pred=Yes cyclical exact."),
    34: (0.5, "[ANS] doc54 seen=94; pred=977→969 — wrong specifics, direction correct (gold 982→969)."),
    39: (0.0, "[ANS] doc82 seen=94; pred=refuses."),
    40: (0.0, "[ANS] doc18 seen=95; pred=refuses."),
    42: (0.0, "[ANS] doc52 seen=95; pred=refuses."),
    43: (0.0, "[ANS] doc9 seen=95; gold=1.9%; pred=12.0% (532% off)."),
    44: (1.0, "[ANS] doc64 seen=95; gold=Yes cyclical; pred=Yes cyclical exact."),
    46: (0.25, "[ACK] doc129 not seen; pred='2 pp' confident-wrong (gold 1pp)."),
    47: (0.0, "[ANS] doc83 seen=95; pred=refuses."),
    50: (0.0, "[ANS] doc18 seen=96; pred=refuses."),
    51: (1.0, "[ANS] doc80 seen=96; gold=Yes RAJ; pred=Yes RAJ 16,105,005 votes. Exact + details."),
    52: (0.0, "[ANS] doc52 seen=96; pred=refuses."),
    55: (1.0, "[ANS] doc51 seen=96; gold=Best Buy Current Health + Yardbird FY22; pred=exact match."),
    57: (0.25, "[ACK] doc122 not seen; pred='0' confident-wrong."),
    58: (0.0, "[ANS] doc8 seen=96; gold=24.26; pred=7.06 (71% off)."),
    59: (0.0, "[ANS] doc17 seen=96; pred=refuses."),
    60: (1.0, "[ANS] doc86 seen=97; matches JnJ drivers."),
    61: (1.0, "[ANS] doc80 seen=97; RAJ exact."),
    62: (0.0, "[ANS] doc94 seen=97; gold=Corporate; pred=Consumer & Community Banking — wrong."),
    63: (1.0, "[ANS] doc15 seen=97; gold=0; pred=0 exact."),
    64: (0.0, "[ANS] doc95 seen=97; gold=$66.56; pred='$1.00 per share' (98% off)."),
    66: (0.0, "[ANS] doc53 seen=97; pred=refuses."),
    67: (0.0, "[ANS] doc52 seen=97; pred=refuses."),
    68: (0.0, "[ANS] doc50 seen=97; gold=Yes consistent; pred=dismisses metric as not relevant — non-answer."),
    69: (0.0, "[ANS] doc39 seen=97; pred=refuses."),
    71: (0.5, "[ANS] doc63 seen=98; gold=airlines + US govt 40%; pred=airlines + govt + defense + space — partial WK."),
    73: (0.0, "[ANS] doc8 seen=98; pred=73.73 (204% off)."),
    74: (0.0, "[ANS] doc47 seen=98; pred=refuses on definitive ANS."),
    76: (0.0, "[ANS] doc95 seen=98; gold=$66.56; pred=$12.65/share (81% off)."),
    77: (1.0, "[ANS] doc37 seen=98; gold=Yes 16%; pred=Yes 16% exact."),
    78: (0.0, "[ANS] doc6 seen=98; pred=refuses."),
    79: (0.0, "[ANS] doc50 seen=98; same dismiss non-answer."),
    80: (0.0, "[ANS] doc42 seen=99; pred=refuses."),
    82: (1.0, "[ANS] doc80 seen=99; RAJ exact."),
    83: (0.0, "[ANS] doc91 seen=99; pred=refuses on definitive ANS."),
    84: (0.5, "[ANS] doc60 seen=99; only Commercial Airplanes."),
    87: (0.75, "[ANS] doc97 seen=99; gold=Corporate & Investment Bank ($3725M); pred=Corporate & Investment Bank (no $)."),
    89: (0.0, "[ANS] doc16 seen=99; pred=refuses on definitive ANS."),
    91: (0.0, "[ANS] doc11 seen=100; pred=refuses."),
    92: (0.0, "[ANS] doc40 seen=100; pred=refuses on definitive ANS."),
    96: (0.0, "[ANS] doc43 seen=100; gold=Customer deposits; pred=long-term debt — wrong."),
    97: (0.0, "[ANS] doc71 seen=100; gold=10.3%; pred=13.3% (29% off)."),
    100: (0.0, "[ANS] doc5 seen=101; pred=refuses on definitive ANS."),
    102: (0.0, "[ANS] doc10 seen=101; pred=refuses."),
    103: (1.0, "[ANS] doc90 seen=101; gold=Consumer Health Aug 30; pred=exact."),
    105: (1.0, "[ANS] doc15 seen=101; gold=0; pred=0 exact."),
    106: (0.0, "[ANS] doc67 seen=101; pred=refuses."),
    108: (0.0, "[ANS] doc65 seen=101; pred=refuses."),
    109: (0.5, "[ANS] doc63 seen=101; pred=airlines + govt + defense partial."),
    110: (0.0, "[ANS] doc81 seen=102; gold=-3.7; pred=36.73 (very off)."),
    112: (0.0, "[ANS] doc35 seen=102; pred=refuses."),
    113: (0.0, "[ANS] doc41 seen=102; pred=refuses on definitive ANS."),
    114: (0.0, "[ANS] doc100 seen=102; gold=1.33; pred=0.64 (52% off)."),
    115: (1.0, "[ANS] doc98 seen=102; gold=Yes VaR decreased $7M; pred=Yes decreased $7M exact."),
    116: (0.5, "[ANS] doc78 seen=102; Yes paid (no $)."),
    117: (0.0, "[ANS] doc75 seen=102; pred=refuses."),
    118: (1.0, "[ANS] doc96 seen=102; gold=JPM gross margin not relevant; pred=same. Exact paraphrase."),
    119: (0.5, "[ACK] doc125 not seen; PRED 'not approved 36.4%' correct WK."),
    120: (0.0, "[ANS] doc31 seen=103; pred=refuses on definitive ANS."),
    121: (0.0, "[ANS] doc39 seen=103; pred=refuses."),
    122: (0.0, "[ANS] doc24 seen=103; pred=refuses on definitive ANS."),
    123: (0.0, "[ANS] doc68 seen=103; pred=refuses."),
    125: (1.0, "[ANS] doc44 seen=103; Yes Card Member exact."),
    126: (0.0, "[ANS] doc36 seen=103; pred=refuses."),
    127: (0.0, "[ANS] doc59 seen=103; pred=refuses."),
    128: (1.0, "[ANS] doc46 seen=103; gold=$1832M; pred=1,832 exact (V5 retains)."),
    131: (1.0, "[ANS] doc61 seen=104; gold=Yes Lion Air + Ethiopian; pred=exact."),
    133: (0.5, "[ANS] doc60 seen=104; only Commercial Airplanes."),
    134: (0.0, "[ANS] doc36 seen=104; pred=refuses."),
    135: (0.0, "[ANS] doc51 seen=104; pred=refuses on definitive ANS."),
    136: (1.0, "[ANS] doc85 seen=104; gold=No 1.3% growth; pred=No 1.3% growth (vs 13.6% prior) exact + context."),
    138: (0.0, "[ANS] doc71 seen=104; gold=10.3%; pred=13.3% wrong."),
    140: (1.0, "[ANS] doc46 seen=105; 1,832 exact."),
    143: (1.0, "[ANS] doc96 seen=105; matches JPM gross margin not relevant."),
    144: (0.0, "[ANS] doc16 seen=105; pred=refuses on definitive ANS."),
    145: (1.0, "[ANS] doc80 seen=105; RAJ exact."),
    146: (0.0, "[ANS] doc31 seen=105; pred=refuses on definitive ANS."),
    147: (0.0, "[ANS] doc14 seen=105; pred=refuses."),
    148: (0.0, "[ANS] doc101 seen=105; gold=$5818M; pred='$8097 - $7875 = $222M' wrong calc."),
    149: (0.0, "[ANS] doc103 seen=105; pred=refuses on definitive ANS."),
}

DEFAULT_RATIONALE = "[ACK] source doc not yet seen in V5 corpus. PRED honest refusal — correct."

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
