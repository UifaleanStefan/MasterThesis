"""Phase 1.9 — FB v5t-corpus calibration part4 (entries 450-599)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__v5t-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__v5t-corpus__calibration__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

SPECIAL: dict[int, tuple[float, str]] = {
    3: (0.0, "[ANS] doc11 seen=46; pred=refuses."),
    7: (0.0, "[ANS] doc30 seen=46; pred=refuses."),
    8: (1.0, "[ANS] doc32 seen=46; gold=AMD products list; pred=matches AMD products. V5 graph retains."),
    9: (0.0, "[ANS] doc31 seen=46; pred=refuses on definitive ANS."),
    11: (1.0, "[ANS] doc37 seen=47; gold=Yes 16% customer concentration; pred=Yes 16% exact."),
    15: (0.0, "[ANS] doc24 seen=47; pred=refuses on definitive ANS."),
    16: (0.0, "[ANS] doc30 seen=47; pred=refuses."),
    20: (0.0, "[ANS] doc1 seen=48; pred=refuses."),
    29: (0.0, "[ANS] doc31 seen=48; pred=refuses on definitive ANS."),
    30: (0.5, "[ACK] doc125 not seen; PRED 'not approved' correct WK."),
    31: (0.0, "[ANS] doc4 seen=49; pred=refuses on definitive ANS."),
    34: (1.0, "[ANS] doc40 seen=49; gold='not measured through operating margin'; pred='operating margin not useful for AMEX' exact paraphrase."),
    36: (0.0, "[ANS] doc30 seen=49; pred=refuses."),
    40: (1.0, "[ANS] doc41 seen=50; gold='not measured through gross margin'; pred='gross margin not useful for AMEX' exact paraphrase."),
    41: (0.5, "[ANS] doc27 seen=50; same partial (employee+fixed asset+other but no %)."),
    42: (1.0, "[ANS] doc16 seen=50; gold=9.5 times; pred=9.5 times with calculation. Exact."),
    49: (0.0, "[ANS] doc4 seen=50; pred=refuses."),
    52: (0.0, "[ANS] doc9 seen=51; pred=refuses."),
    54: (0.0, "[ANS] doc24 seen=51; pred=refuses on definitive ANS."),
    56: (0.0, "[ANS] doc11 seen=51; pred=refuses."),
    57: (0.75, "[ANS] doc35 seen=51; gold=AMD operations highest; pred='operating activities most' (no $)."),
    58: (0.0, "[ANS] doc29 seen=51; gold=flat; pred=-5% wrong direction."),
    66: (0.0, "[ANS] doc17 seen=52; pred=refuses."),
    71: (0.0, "[ANS] doc30 seen=53; pred=refuses."),
    75: (0.0, "[ANS] doc36 seen=53; gold=Data Center; pred=Gaming segment — wrong."),
    79: (0.75, "[ANS] doc35 seen=53; same operations match."),
    81: (0.0, "[ANS] doc36 seen=54; same wrong."),
    83: (0.0, "[ANS] doc29 seen=54; -5% wrong."),
    85: (1.0, "[ANS] doc15 seen=54; gold=0; pred=0 exact."),
    86: (0.0, "[ANS] doc0 seen=54; pred=refuses."),
    88: (0.0, "[ANS] doc50 seen=54; gold=Yes consistent (1.1% decline); pred=dismisses as not relevant. Non-answer."),
    91: (0.0, "[ANS] doc0 seen=55; pred=refuses."),
    95: (0.0, "[ANS] doc29 seen=55; -5% wrong."),
    96: (1.0, "[ANS] doc42 seen=55; gold=24.6%→21.6%; pred=24.6%→21.6% exact."),
    103: (1.0, "[ANS] doc37 seen=56; gold=Yes 16%; pred=Yes 16% exact."),
    104: (0.0, "[ANS] doc50 seen=56; same dismiss-metric non-answer."),
    106: (0.0, "[ANS] doc53 seen=56; pred=refuses on definitive ANS."),
    107: (0.0, "[ANS] doc29 seen=56; -5% wrong."),
    110: (0.0, "[ANS] doc3 seen=57; pred=refuses."),
    111: (0.0, "[ANS] doc22 seen=57; pred=refuses on definitive ANS."),
    114: (0.0, "[ANS] doc14 seen=57; pred=refuses."),
    121: (0.25, "[ACK] doc63 not seen; partial WK."),
    122: (0.5, "[ANS] doc27 seen=58; partial (no %)."),
    123: (1.0, "[ANS] doc28 seen=58; gold=$2,018M Amcor Adj EBITDA; pred=2,018 exact."),
    124: (0.0, "[ANS] doc31 seen=58; pred=refuses on definitive ANS."),
    126: (0.5, "[ACK] doc74 not seen; PRED=$59,206M (0.1% off gold $59,268M) — correct world-knowledge from outside corpus."),
    129: (1.0, "[ANS] doc57 seen=58; gold=101.5%; pred=101.7% (0.2% off — exact)."),
    130: (0.75, "[ANS] doc55 seen=59; gold=entertainment 9%; pred=Gaming best (no %)."),
    134: (0.0, "[ANS] doc17 seen=59; pred=refuses."),
    135: (0.0, "[ANS] doc14 seen=59; pred=refuses."),
    136: (0.25, "[ANS] doc16 seen=59; PRED starts calc but refuses on missing COGS — partial."),
    140: (0.0, "[ANS] doc29 seen=60; -5% wrong."),
    146: (0.0, "[ANS] doc30 seen=60; pred=refuses."),
}

DEFAULT_RATIONALE = "[ACK] source doc not yet seen in V5 corpus. PRED honest refusal — correct."

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
