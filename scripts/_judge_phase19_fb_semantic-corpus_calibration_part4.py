"""Phase 1.9 — FB semantic-corpus calibration part4 (entries 450-599)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__semantic-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__semantic-corpus__calibration__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

SPECIAL: dict[int, tuple[float, str]] = {
    3: (0.0, "[ANS] doc11 seen=46; pred=refuses on definitive ANS."),
    7: (0.0, "[ANS] doc30 seen=46; pred=refuses."),
    8: (0.0, "[ANS] doc32 seen=46; gold=AMD products CPU/GPU/DPU/FPGA; pred=refuses."),
    9: (0.0, "[ANS] doc31 seen=46; pred=refuses."),
    11: (0.0, "[ANS] doc37 seen=47; gold=Yes 16% concentration; pred=refuses."),
    15: (0.0, "[ANS] doc24 seen=47; pred=refuses."),
    16: (0.0, "[ANS] doc30 seen=47; pred=refuses."),
    29: (0.0, "[ANS] doc31 seen=48; pred=refuses."),
    30: (0.5, "[ACK] doc125 not seen; PRED='not approved' correct WK."),
    31: (0.5, "[ANS] doc4 seen=49; gold=consumer segment shrunk 0.9%; pred='Consumer segment' (no %)."),
    34: (0.0, "[ANS] doc40 seen=49; pred=refuses."),
    36: (0.0, "[ANS] doc30 seen=49; pred=refuses."),
    40: (0.0, "[ANS] doc41 seen=50; pred=refuses."),
    41: (0.0, "[ANS] doc27 seen=50; pred=refuses."),
    42: (0.0, "[ANS] doc16 seen=50; pred=refuses."),
    49: (0.5, "[ANS] doc4 seen=50; pred='Consumer segment' (no %)."),
    52: (0.0, "[ANS] doc9 seen=51; gold=1.9% Activision capex/rev; pred=refuses."),
    54: (0.0, "[ANS] doc24 seen=51; pred=refuses."),
    56: (0.0, "[ANS] doc11 seen=51; pred=refuses."),
    57: (0.0, "[ANS] doc35 seen=51; pred=refuses."),
    58: (0.0, "[ANS] doc29 seen=51; pred=refuses."),
    61: (0.25, "[ACK] doc122 not seen; pred='0' confident-wrong."),
    66: (0.0, "[ANS] doc17 seen=52; pred=refuses."),
    71: (0.0, "[ANS] doc30 seen=53; pred=refuses."),
    75: (0.0, "[ANS] doc36 seen=53; pred=refuses."),
    77: (0.5, "[ACK] doc125 not seen; PRED='not approved' correct WK."),
    79: (0.0, "[ANS] doc35 seen=53; pred=refuses."),
    81: (0.0, "[ANS] doc36 seen=54; pred=refuses."),
    83: (0.0, "[ANS] doc29 seen=54; pred=refuses."),
    88: (0.0, "[ANS] doc50 seen=54; pred=refuses on definitive ANS."),
    95: (0.0, "[ANS] doc29 seen=55; pred=refuses."),
    96: (0.0, "[ANS] doc42 seen=55; pred=refuses."),
    103: (0.0, "[ANS] doc37 seen=56; pred=refuses."),
    104: (0.0, "[ANS] doc50 seen=56; pred=refuses."),
    106: (0.0, "[ANS] doc53 seen=56; pred=refuses."),
    107: (0.0, "[ANS] doc29 seen=56; pred=refuses."),
    111: (0.0, "[ANS] doc22 seen=57; gold=Amcor Finance supplemental indentures; pred=refuses."),
    114: (0.0, "[ANS] doc14 seen=57; gold=Yes Adobe FCF/NI improved 13%; pred=refuses."),
    122: (0.0, "[ANS] doc27 seen=58; pred=refuses."),
    123: (0.0, "[ANS] doc28 seen=58; pred=refuses."),
    124: (0.0, "[ANS] doc31 seen=58; pred=refuses."),
    126: (0.25, "[ACK] doc74 not seen; pred='$49,205M Costco' confident-wrong specific (gold $59,268)."),
    129: (0.0, "[ANS] doc57 seen=58; pred=refuses."),
    130: (0.0, "[ANS] doc55 seen=59; pred=refuses."),
    134: (0.0, "[ANS] doc17 seen=59; pred=refuses."),
    135: (0.0, "[ANS] doc14 seen=59; pred=refuses."),
    136: (0.0, "[ANS] doc16 seen=59; pred=refuses."),
    140: (0.0, "[ANS] doc29 seen=60; pred=refuses."),
    146: (0.0, "[ANS] doc30 seen=60; pred=refuses."),
}

DEFAULT_RATIONALE = "[ACK] source doc not yet seen in semantic corpus. PRED honest refusal — correct."

ENTRY_SUFFIXES: list[str] = [
    "doc124_qa0__after45", "doc141_qa0__after45", "doc56_qa0__after45", "doc11_qa0__after45",
    "doc109_qa0__after45", "doc59_qa0__after45", "doc57_qa0__after45", "doc30_qa0__after45",
    "doc32_qa0__after45", "doc31_qa0__after45",
    "doc99_qa0__after46", "doc37_qa0__after46", "doc54_qa0__after46", "doc118_qa0__after46",
    "doc58_qa0__after46", "doc24_qa0__after46", "doc30_qa0__after46", "doc50_qa0__after46",
    "doc148_qa0__after46", "doc95_qa0__after46",
    "doc1_qa0__after47", "doc75_qa0__after47", "doc92_qa0__after47", "doc87_qa0__after47",
    "doc93_qa0__after47", "doc78_qa0__after47", "doc97_qa0__after47", "doc49_qa0__after47",
    "doc136_qa0__after47", "doc31_qa0__after47",
    "doc125_qa0__after48", "doc4_qa0__after48", "doc58_qa0__after48", "doc133_qa0__after48",
    "doc40_qa0__after48", "doc148_qa0__after48", "doc30_qa0__after48", "doc76_qa0__after48",
    "doc121_qa0__after48", "doc75_qa0__after48",
    "doc41_qa0__after49", "doc27_qa0__after49", "doc16_qa0__after49", "doc145_qa0__after49",
    "doc117_qa0__after49", "doc65_qa0__after49", "doc66_qa0__after49", "doc58_qa0__after49",
    "doc138_qa0__after49", "doc4_qa0__after49",
    "doc76_qa0__after50", "doc113_qa0__after50", "doc9_qa0__after50", "doc136_qa0__after50",
    "doc24_qa0__after50", "doc130_qa0__after50", "doc11_qa0__after50", "doc35_qa0__after50",
    "doc29_qa0__after50", "doc53_qa0__after50",
    "doc52_qa0__after51", "doc122_qa0__after51", "doc128_qa0__after51", "doc53_qa0__after51",
    "doc104_qa0__after51", "doc98_qa0__after51", "doc17_qa0__after51", "doc77_qa0__after51",
    "doc136_qa0__after51", "doc61_qa0__after51",
    "doc137_qa0__after52", "doc30_qa0__after52", "doc54_qa0__after52", "doc53_qa0__after52",
    "doc80_qa0__after52", "doc36_qa0__after52", "doc121_qa0__after52", "doc125_qa0__after52",
    "doc136_qa0__after52", "doc35_qa0__after52",
    "doc94_qa0__after53", "doc36_qa0__after53", "doc56_qa0__after53", "doc29_qa0__after53",
    "doc139_qa0__after53", "doc15_qa0__after53", "doc0_qa0__after53", "doc78_qa0__after53",
    "doc50_qa0__after53", "doc145_qa0__after53",
    "doc63_qa0__after54", "doc0_qa0__after54", "doc134_qa0__after54", "doc80_qa0__after54",
    "doc133_qa0__after54", "doc29_qa0__after54", "doc42_qa0__after54", "doc83_qa0__after54",
    "doc137_qa0__after54", "doc92_qa0__after54",
    "doc147_qa0__after55", "doc108_qa0__after55", "doc100_qa0__after55", "doc37_qa0__after55",
    "doc50_qa0__after55", "doc92_qa0__after55", "doc53_qa0__after55", "doc29_qa0__after55",
    "doc120_qa0__after55", "doc128_qa0__after55",
    "doc3_qa0__after56", "doc22_qa0__after56", "doc116_qa0__after56", "doc141_qa0__after56",
    "doc14_qa0__after56", "doc88_qa0__after56", "doc148_qa0__after56", "doc60_qa0__after56",
    "doc67_qa0__after56", "doc109_qa0__after56",
    "doc120_qa0__after57", "doc63_qa0__after57", "doc27_qa0__after57", "doc28_qa0__after57",
    "doc31_qa0__after57", "doc107_qa0__after57", "doc74_qa0__after57", "doc121_qa0__after57",
    "doc69_qa0__after57", "doc57_qa0__after57",
    "doc55_qa0__after58", "doc118_qa0__after58", "doc59_qa0__after58", "doc64_qa0__after58",
    "doc17_qa0__after58", "doc14_qa0__after58", "doc16_qa0__after58", "doc66_qa0__after58",
    "doc78_qa0__after58", "doc95_qa0__after58",
    "doc29_qa0__after59", "doc65_qa0__after59", "doc87_qa0__after59", "doc116_qa0__after59",
    "doc66_qa0__after59", "doc110_qa0__after59", "doc30_qa0__after59", "doc134_qa0__after59",
    "doc119_qa0__after59", "doc147_qa0__after59",
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
    print(f"part4 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
