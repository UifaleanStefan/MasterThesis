"""Phase 1.9 — FB v5t-corpus calibration part5 (entries 600-749)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__v5t-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__v5t-corpus__calibration__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

SPECIAL: dict[int, tuple[float, str]] = {
    0: (0.0, "[ANS] doc13 seen=61; pred=refuses on definitive ANS."),
    1: (0.0, "[ANS] doc59 seen=61; pred=refuses."),
    2: (0.0, "[ANS] doc47 seen=61; pred=refuses on definitive ANS."),
    5: (0.0, "[ANS] doc18 seen=61; pred=refuses."),
    7: (1.0, "[ANS] doc7 seen=61; gold=Yes 65 years; pred=Yes 65th year exact."),
    10: (0.0, "[ANS] doc50 seen=62; pred=refuses on definitive ANS."),
    11: (0.0, "[ANS] doc20 seen=62; pred=refuses."),
    12: (0.5, "[ACK] doc96 not seen; PRED 'JPM gross margin not relevant' correct WK."),
    14: (0.0, "[ANS] doc12 seen=62; pred=refuses."),
    15: (0.5, "[ANS] doc54 seen=62; gold=982→969 decline; pred=977→966 decline — wrong specifics, direction correct."),
    20: (0.0, "[ANS] doc47 seen=63; pred=refuses on definitive ANS."),
    21: (1.0, "[ANS] doc40 seen=63; gold='not measured through op margin'; pred='not useful for AMEX' exact paraphrase."),
    31: (0.5, "[ACK] doc64 not seen; PRED 'Yes cyclical' correct WK."),
    36: (1.0, "[ANS] doc61 seen=64; gold=Yes Lion Air + Ethiopian; pred=exact."),
    37: (1.0, "[ANS] doc33 seen=64; matches gold AMD drivers (Data Center 64% etc)."),
    38: (0.0, "[ANS] doc39 seen=64; pred=refuses."),
    39: (0.0, "[ANS] doc25 seen=64; pred=refuses on definitive ANS."),
    41: (0.5, "[ANS] doc60 seen=65; only Commercial Airplanes (gold has 3)."),
    45: (0.0, "[ANS] doc24 seen=65; pred=refuses on definitive ANS."),
    46: (0.0, "[ANS] doc36 seen=65; pred=refuses."),
    48: (0.0, "[ANS] doc27 seen=65; pred=refuses."),
    49: (0.0, "[ANS] doc41 seen=65; pred=refuses on definitive ANS."),
    52: (0.0, "[ANS] doc26 seen=66; pred=refuses on definitive ANS."),
    53: (0.0, "[ANS] doc18 seen=66; pred=refuses."),
    57: (0.0, "[ANS] doc38 seen=66; gold='None'; pred=refuses without delivering."),
    60: (0.75, "[ANS] doc55 seen=67; gold=entertainment/gaming 9%; pred=Gaming best (no %)."),
    61: (0.0, "[ANS] doc51 seen=67; pred=refuses on definitive ANS."),
    62: (0.0, "[ANS] doc62 seen=67; gold=Yes improved 4.8→5.3%; pred=dismisses metric as not useful — non-answer."),
    68: (0.0, "[ANS] doc66 seen=67; pred=refuses."),
    69: (0.0, "[ANS] doc17 seen=67; pred=refuses."),
    72: (0.0, "[ANS] doc25 seen=68; pred=refuses."),
    75: (0.0, "[ANS] doc2 seen=68; pred=refuses on definitive ANS."),
    76: (0.0, "[ANS] doc3 seen=68; pred=refuses."),
    78: (0.0, "[ANS] doc35 seen=68; gold=AMD operations; pred='investing activities' — wrong category."),
    79: (0.0, "[ANS] doc39 seen=68; pred=refuses."),
    80: (0.0, "[ANS] doc66 seen=69; pred=refuses."),
    81: (0.0, "[ANS] doc25 seen=69; pred=refuses."),
    84: (0.0, "[ANS] doc24 seen=69; pred=refuses on definitive ANS."),
    86: (1.0, "[ANS] doc32 seen=69; matches gold AMD products list."),
    87: (1.0, "[ANS] doc15 seen=69; gold=0; pred=0 exact."),
    88: (0.25, "[ACK] doc82 not seen; PRED=1.74 confident-wrong specific (gold 0.68)."),
    93: (0.0, "[ANS] doc30 seen=70; pred=refuses."),
    94: (0.25, "[ACK] doc108 not seen; PRED='Las Vegas Strip worst' confident-wrong specific (gold MGM China)."),
    95: (1.0, "[ANS] doc32 seen=70; matches AMD products."),
    98: (0.5, "[ANS] doc65 seen=70; 787 + 737 increase (no 777X)."),
    99: (0.0, "[ANS] doc16 seen=70; pred=refuses."),
    100: (0.0, "[ANS] doc26 seen=71; pred=refuses on definitive ANS."),
    101: (0.0, "[ANS] doc66 seen=71; pred=refuses."),
    103: (0.25, "[ACK] doc138 not seen; PRED vague WK 'improved operating efficiencies' — partial."),
    104: (0.25, "[ACK] doc129 not seen; PRED='2 pp' confident-wrong (gold 1pp)."),
    107: (0.5, "[ANS] doc65 seen=71; 787 + 737 (no 777X)."),
    108: (0.25, "[ACK] doc104 not seen; PRED='10.0%' confident-wrong (gold 7.9%)."),
    110: (0.0, "[ANS] doc10 seen=72; pred=refuses."),
    111: (1.0, "[ANS] doc46 seen=72; gold=$1832M; pred=1,832 exact (V5 graph retains)."),
    112: (0.0, "[ANS] doc59 seen=72; pred=refuses."),
    114: (0.75, "[ANS] doc55 seen=72; same Gaming best (no %)."),
    116: (0.0, "[ANS] doc42 seen=72; pred=refuses."),
    118: (0.0, "[ANS] doc58 seen=72; pred=refuses."),
    119: (0.0, "[ANS] doc14 seen=72; pred=refuses."),
    120: (0.0, "[ANS] doc3 seen=73; pred=refuses."),
    123: (0.0, "[ANS] doc12 seen=73; pred=refuses."),
    124: (0.0, "[ANS] doc71 seen=73; pred=refuses."),
    125: (0.0, "[ANS] doc52 seen=73; pred=refuses."),
    126: (1.0, "[ANS] doc64 seen=73; gold=Yes cyclical; pred=Yes cyclical exact."),
    127: (0.0, "[ANS] doc26 seen=73; pred=refuses on definitive ANS."),
    130: (0.0, "[ANS] doc14 seen=74; pred=refuses."),
    132: (0.0, "[ANS] doc12 seen=74; pred=refuses."),
    137: (0.0, "[ANS] doc69 seen=74; pred=refuses."),
    138: (0.0, "[ANS] doc4 seen=74; pred=refuses."),
    139: (0.0, "[ANS] doc26 seen=74; pred=refuses on definitive ANS."),
    142: (0.0, "[ANS] doc69 seen=75; gold=0.8; pred=0.19 (76% off)."),
    144: (0.5, "[ACK] doc90 not seen; correct WK."),
    147: (0.0, "[ANS] doc50 seen=75; gold=Yes consistent (1.1% decline); pred='not consistent' — Y/N flip."),
    148: (0.0, "[ANS] doc22 seen=75; pred=refuses on definitive ANS."),
    149: (0.0, "[ANS] doc6 seen=75; pred=refuses."),
}

DEFAULT_RATIONALE = "[ACK] source doc not yet seen in V5 corpus. PRED honest refusal — correct."

ENTRY_SUFFIXES: list[str] = [
    "doc13_qa0__after60", "doc59_qa0__after60", "doc47_qa0__after60", "doc67_qa0__after60",
    "doc130_qa0__after60", "doc18_qa0__after60", "doc133_qa0__after60", "doc7_qa0__after60",
    "doc137_qa0__after60", "doc134_qa0__after60",
    "doc50_qa0__after61", "doc20_qa0__after61", "doc96_qa0__after61", "doc69_qa0__after61",
    "doc12_qa0__after61", "doc54_qa0__after61", "doc126_qa0__after61", "doc106_qa0__after61",
    "doc142_qa0__after61", "doc75_qa0__after61",
    "doc47_qa0__after62", "doc40_qa0__after62", "doc101_qa0__after62", "doc140_qa0__after62",
    "doc87_qa0__after62", "doc121_qa0__after62", "doc83_qa0__after62", "doc72_qa0__after62",
    "doc147_qa0__after62", "doc126_qa0__after62",
    "doc126_qa0__after63", "doc64_qa0__after63", "doc115_qa0__after63", "doc77_qa0__after63",
    "doc143_qa0__after63", "doc123_qa0__after63", "doc61_qa0__after63", "doc33_qa0__after63",
    "doc39_qa0__after63", "doc25_qa0__after63",
    "doc132_qa0__after64", "doc60_qa0__after64", "doc134_qa0__after64", "doc107_qa0__after64",
    "doc68_qa0__after64", "doc24_qa0__after64", "doc36_qa0__after64", "doc117_qa0__after64",
    "doc27_qa0__after64", "doc41_qa0__after64",
    "doc105_qa0__after65", "doc146_qa0__after65", "doc26_qa0__after65", "doc18_qa0__after65",
    "doc89_qa0__after65", "doc114_qa0__after65", "doc102_qa0__after65", "doc38_qa0__after65",
    "doc94_qa0__after65", "doc145_qa0__after65",
    "doc55_qa0__after66", "doc51_qa0__after66", "doc62_qa0__after66", "doc139_qa0__after66",
    "doc142_qa0__after66", "doc149_qa0__after66", "doc116_qa0__after66", "doc103_qa0__after66",
    "doc66_qa0__after66", "doc17_qa0__after66",
    "doc74_qa0__after67", "doc76_qa0__after67", "doc25_qa0__after67", "doc71_qa0__after67",
    "doc113_qa0__after67", "doc2_qa0__after67", "doc3_qa0__after67", "doc141_qa0__after67",
    "doc35_qa0__after67", "doc39_qa0__after67",
    "doc66_qa0__after68", "doc25_qa0__after68", "doc99_qa0__after68", "doc85_qa0__after68",
    "doc24_qa0__after68", "doc126_qa0__after68", "doc32_qa0__after68", "doc15_qa0__after68",
    "doc82_qa0__after68", "doc121_qa0__after68",
    "doc105_qa0__after69", "doc85_qa0__after69", "doc139_qa0__after69", "doc30_qa0__after69",
    "doc108_qa0__after69", "doc32_qa0__after69", "doc87_qa0__after69", "doc93_qa0__after69",
    "doc65_qa0__after69", "doc16_qa0__after69",
    "doc26_qa0__after70", "doc66_qa0__after70", "doc93_qa0__after70", "doc138_qa0__after70",
    "doc129_qa0__after70", "doc71_qa0__after70", "doc135_qa0__after70", "doc65_qa0__after70",
    "doc104_qa0__after70", "doc91_qa0__after70",
    "doc10_qa0__after71", "doc46_qa0__after71", "doc59_qa0__after71", "doc95_qa0__after71",
    "doc55_qa0__after71", "doc139_qa0__after71", "doc42_qa0__after71", "doc94_qa0__after71",
    "doc58_qa0__after71", "doc14_qa0__after71",
    "doc3_qa0__after72", "doc110_qa0__after72", "doc134_qa0__after72", "doc12_qa0__after72",
    "doc71_qa0__after72", "doc52_qa0__after72", "doc64_qa0__after72", "doc26_qa0__after72",
    "doc117_qa0__after72", "doc119_qa0__after72",
    "doc14_qa0__after73", "doc106_qa0__after73", "doc12_qa0__after73", "doc114_qa0__after73",
    "doc92_qa0__after73", "doc140_qa0__after73", "doc115_qa0__after73", "doc69_qa0__after73",
    "doc4_qa0__after73", "doc26_qa0__after73",
    "doc119_qa0__after74", "doc117_qa0__after74", "doc69_qa0__after74", "doc123_qa0__after74",
    "doc90_qa0__after74", "doc83_qa0__after74", "doc126_qa0__after74", "doc50_qa0__after74",
    "doc22_qa0__after74", "doc6_qa0__after74",
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
    print(f"part5 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
