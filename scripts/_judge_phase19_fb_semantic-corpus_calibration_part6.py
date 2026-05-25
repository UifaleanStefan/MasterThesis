"""Phase 1.9 — FB semantic-corpus calibration part6 (entries 750-899)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__semantic-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__semantic-corpus__calibration__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

SPECIAL: dict[int, tuple[float, str]] = {
    0: (0.0, "[ANS] doc37 seen=76; pred=refuses on definitive ANS."),
    2: (0.25, "[ACK] doc122 not seen; pred='0' confident-wrong."),
    3: (0.0, "[ANS] doc26 seen=76; pred=refuses."),
    6: (0.0, "[ANS] doc53 seen=76; pred=refuses."),
    7: (0.0, "[ANS] doc25 seen=76; pred=refuses."),
    11: (0.0, "[ANS] doc41 seen=77; pred=refuses."),
    14: (0.0, "[ANS] doc37 seen=77; pred=refuses."),
    15: (0.0, "[ANS] doc55 seen=77; pred=refuses."),
    16: (0.0, "[ANS] doc18 seen=77; pred=refuses."),
    18: (0.25, "[ANS] doc74 seen=77; gold=$59,268; pred=$49,164 — 17% off, outside tolerance."),
    19: (0.0, "[ANS] doc11 seen=77; pred=refuses."),
    20: (0.0, "[ANS] doc64 seen=78; pred=refuses."),
    21: (0.0, "[ANS] doc60 seen=78; pred=refuses."),
    23: (0.0, "[ANS] doc44 seen=78; gold=Yes Card Member; pred=refuses."),
    26: (0.0, "[ANS] doc52 seen=78; pred=refuses."),
    29: (0.0, "[ANS] doc11 seen=78; pred=refuses."),
    32: (0.0, "[ANS] doc19 seen=79; pred=refuses."),
    33: (0.0, "[ANS] doc44 seen=79; pred=refuses."),
    34: (0.0, "[ANS] doc63 seen=79; pred=refuses."),
    36: (0.0, "[ANS] doc67 seen=79; pred=refuses."),
    37: (0.0, "[ANS] doc40 seen=79; pred=refuses."),
    38: (0.0, "[ANS] doc52 seen=79; pred=refuses."),
    39: (0.0, "[ANS] doc65 seen=79; pred=refuses."),
    41: (0.0, "[ANS] doc23 seen=80; pred=refuses."),
    43: (0.0, "[ANS] doc56 seen=80; pred=refuses on definitive ANS."),
    45: (0.0, "[ANS] doc55 seen=80; pred=refuses."),
    46: (0.0, "[ANS] doc28 seen=80; pred=refuses."),
    48: (0.0, "[ANS] doc2 seen=80; Y/N flip — pred='Yes capital-intensive'."),
    49: (0.0, "[ANS] doc14 seen=80; pred=refuses."),
    51: (0.0, "[ANS] doc44 seen=81; pred=refuses."),
    53: (0.0, "[ANS] doc25 seen=81; pred=refuses."),
    54: (0.0, "[ANS] doc60 seen=81; pred=refuses."),
    56: (0.0, "[ANS] doc35 seen=81; pred=refuses."),
    57: (0.0, "[ANS] doc12 seen=81; pred=refuses."),
    59: (0.0, "[ANS] doc43 seen=81; pred=refuses."),
    60: (0.0, "[ANS] doc30 seen=82; pred=refuses."),
    61: (0.0, "[ANS] doc75 seen=82; pred=refuses on definitive ANS."),
    62: (0.0, "[ANS] doc79 seen=82; gold=Yes Mary Dillon Ulta; pred=refuses."),
    63: (0.0, "[ANS] doc2 seen=82; Y/N flip."),
    65: (0.0, "[ANS] doc60 seen=82; pred=refuses."),
    66: (0.0, "[ANS] doc23 seen=82; pred=refuses."),
    67: (0.0, "[ANS] doc59 seen=82; pred=refuses."),
    70: (0.0, "[ANS] doc79 seen=83; pred=refuses."),
    71: (0.0, "[ANS] doc12 seen=83; pred=refuses."),
    72: (0.5, "[ACK] doc125 not seen; PRED='not approved' correct WK."),
    73: (0.0, "[ANS] doc28 seen=83; pred=refuses."),
    74: (0.0, "[ANS] doc35 seen=83; pred=refuses."),
    75: (0.0, "[ANS] doc27 seen=83; pred=refuses."),
    76: (0.0, "[ANS] doc43 seen=83; pred=refuses."),
    78: (0.0, "[ANS] doc71 seen=83; pred=refuses."),
    80: (0.0, "[ANS] doc39 seen=84; pred=refuses."),
    82: (0.0, "[ANS] doc54 seen=84; pred=refuses."),
    83: (0.0, "[ANS] doc42 seen=84; pred=refuses."),
    86: (0.5, "[ACK] doc90 not seen; PRED='Consumer Health Aug 30' WK."),
    87: (0.0, "[ANS] doc17 seen=84; pred=refuses."),
    88: (0.0, "[ANS] doc46 seen=84; pred=refuses."),
    89: (0.0, "[ANS] doc57 seen=84; pred=refuses."),
    91: (0.0, "[ANS] doc46 seen=85; pred=refuses."),
    92: (0.0, "[ANS] doc84 seen=85; pred=refuses."),
    93: (0.0, "[ANS] doc12 seen=85; pred=refuses."),
    94: (0.0, "[ANS] doc77 seen=85; pred=refuses on definitive ANS."),
    95: (0.0, "[ANS] doc58 seen=85; pred=refuses."),
    96: (0.0, "[ANS] doc29 seen=85; pred=refuses."),
    98: (0.0, "[ANS] doc13 seen=85; pred=refuses."),
    99: (0.0, "[ANS] doc8 seen=85; pred=refuses."),
    100: (0.0, "[ANS] doc18 seen=86; pred=refuses."),
    102: (0.0, "[ANS] doc67 seen=86; pred=refuses."),
    103: (0.0, "[ANS] doc11 seen=86; pred=refuses."),
    105: (0.0, "[ANS] doc48 seen=86; pred=refuses."),
    110: (0.0, "[ANS] doc48 seen=87; pred=refuses."),
    111: (0.0, "[ANS] doc46 seen=87; pred=refuses."),
    112: (0.0, "[ANS] doc84 seen=87; pred=refuses."),
    113: (0.5, "[ANS] doc4 seen=87; pred='Consumer segment' (no %)."),
    114: (0.0, "[ANS] doc40 seen=87; pred=refuses."),
    115: (0.0, "[ANS] doc26 seen=87; pred=refuses."),
    119: (0.0, "[ANS] doc76 seen=87; pred=refuses."),
    120: (0.0, "[ANS] doc12 seen=88; pred=refuses."),
    122: (0.0, "[ANS] doc43 seen=88; pred=refuses."),
    124: (0.0, "[ANS] doc59 seen=88; pred=refuses."),
    125: (0.5, "[ANS] doc4 seen=88; pred='Consumer segment' (no %)."),
    127: (0.0, "[ANS] doc16 seen=88; pred=refuses."),
    130: (0.0, "[ANS] doc22 seen=89; pred=refuses."),
    131: (0.0, "[ANS] doc27 seen=89; pred=refuses."),
    132: (0.0, "[ANS] doc25 seen=89; pred=refuses."),
    135: (0.0, "[ANS] doc66 seen=89; pred=refuses."),
    136: (0.0, "[ANS] doc60 seen=89; pred=refuses."),
    138: (0.0, "[ANS] doc21 seen=89; pred=refuses."),
    140: (0.0, "[ANS] doc34 seen=90; pred=refuses."),
    142: (0.0, "[ANS] doc89 seen=90; pred=refuses."),
    143: (0.0, "[ANS] doc43 seen=90; pred=refuses."),
    145: (0.0, "[ANS] doc75 seen=90; pred=refuses."),
    146: (0.0, "[ANS] doc58 seen=90; pred=refuses."),
    148: (0.0, "[ANS] doc83 seen=90; pred=refuses."),
    149: (0.0, "[ANS] doc2 seen=90; Y/N flip."),
}

DEFAULT_RATIONALE = "[ACK] source doc not yet seen in semantic corpus. PRED honest refusal — correct."

ENTRY_SUFFIXES: list[str] = [
    "doc37_qa0__after75", "doc0_qa0__after75", "doc122_qa0__after75", "doc26_qa0__after75",
    "doc126_qa0__after75", "doc111_qa0__after75", "doc53_qa0__after75", "doc25_qa0__after75",
    "doc121_qa0__after75", "doc133_qa0__after75",
    "doc3_qa0__after76", "doc41_qa0__after76", "doc112_qa0__after76", "doc100_qa0__after76",
    "doc37_qa0__after76", "doc55_qa0__after76", "doc18_qa0__after76", "doc86_qa0__after76",
    "doc74_qa0__after76", "doc11_qa0__after76",
    "doc64_qa0__after77", "doc60_qa0__after77", "doc113_qa0__after77", "doc44_qa0__after77",
    "doc87_qa0__after77", "doc82_qa0__after77", "doc52_qa0__after77", "doc97_qa0__after77",
    "doc130_qa0__after77", "doc11_qa0__after77",
    "doc149_qa0__after78", "doc120_qa0__after78", "doc19_qa0__after78", "doc44_qa0__after78",
    "doc63_qa0__after78", "doc102_qa0__after78", "doc67_qa0__after78", "doc40_qa0__after78",
    "doc52_qa0__after78", "doc65_qa0__after78",
    "doc146_qa0__after79", "doc23_qa0__after79", "doc109_qa0__after79", "doc56_qa0__after79",
    "doc92_qa0__after79", "doc55_qa0__after79", "doc28_qa0__after79", "doc83_qa0__after79",
    "doc2_qa0__after79", "doc14_qa0__after79",
    "doc106_qa0__after80", "doc44_qa0__after80", "doc82_qa0__after80", "doc25_qa0__after80",
    "doc60_qa0__after80", "doc103_qa0__after80", "doc35_qa0__after80", "doc12_qa0__after80",
    "doc141_qa0__after80", "doc43_qa0__after80",
    "doc30_qa0__after81", "doc75_qa0__after81", "doc79_qa0__after81", "doc2_qa0__after81",
    "doc138_qa0__after81", "doc60_qa0__after81", "doc23_qa0__after81", "doc59_qa0__after81",
    "doc98_qa0__after81", "doc106_qa0__after81",
    "doc79_qa0__after82", "doc12_qa0__after82", "doc125_qa0__after82", "doc28_qa0__after82",
    "doc35_qa0__after82", "doc27_qa0__after82", "doc43_qa0__after82", "doc101_qa0__after82",
    "doc71_qa0__after82", "doc144_qa0__after82",
    "doc39_qa0__after83", "doc3_qa0__after83", "doc54_qa0__after83", "doc42_qa0__after83",
    "doc144_qa0__after83", "doc126_qa0__after83", "doc90_qa0__after83", "doc17_qa0__after83",
    "doc46_qa0__after83", "doc57_qa0__after83",
    "doc148_qa0__after84", "doc46_qa0__after84", "doc84_qa0__after84", "doc12_qa0__after84",
    "doc77_qa0__after84", "doc58_qa0__after84", "doc29_qa0__after84", "doc124_qa0__after84",
    "doc13_qa0__after84", "doc8_qa0__after84",
    "doc18_qa0__after85", "doc131_qa0__after85", "doc67_qa0__after85", "doc11_qa0__after85",
    "doc118_qa0__after85", "doc48_qa0__after85", "doc139_qa0__after85", "doc116_qa0__after85",
    "doc135_qa0__after85", "doc119_qa0__after85",
    "doc48_qa0__after86", "doc46_qa0__after86", "doc84_qa0__after86", "doc4_qa0__after86",
    "doc40_qa0__after86", "doc26_qa0__after86", "doc109_qa0__after86", "doc116_qa0__after86",
    "doc138_qa0__after86", "doc76_qa0__after86",
    "doc12_qa0__after87", "doc138_qa0__after87", "doc43_qa0__after87", "doc108_qa0__after87",
    "doc59_qa0__after87", "doc4_qa0__after87", "doc92_qa0__after87", "doc16_qa0__after87",
    "doc91_qa0__after87", "doc124_qa0__after87",
    "doc22_qa0__after88", "doc27_qa0__after88", "doc25_qa0__after88", "doc149_qa0__after88",
    "doc146_qa0__after88", "doc66_qa0__after88", "doc60_qa0__after88", "doc117_qa0__after88",
    "doc21_qa0__after88", "doc113_qa0__after88",
    "doc34_qa0__after89", "doc129_qa0__after89", "doc89_qa0__after89", "doc43_qa0__after89",
    "doc101_qa0__after89", "doc75_qa0__after89", "doc58_qa0__after89", "doc111_qa0__after89",
    "doc83_qa0__after89", "doc2_qa0__after89",
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
    print(f"part6 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
