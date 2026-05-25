"""Claude manual judging — Phase 1.9 FB calibration v4t-canonical (entries 486-649).

Manually judged 1-by-1 per HARD RULE.
"""

from __future__ import annotations

import json
from pathlib import Path

JUDGE_DIR = Path(
    "results/stage3/judge_queue/financebench__v4t-canonical__calibration__seed42"
)
QID_PREFIX = "financebench__v4t-canonical__calibration__"
QID_SUFFIX = "__seed42"

JUDGMENTS: list[tuple[str, float, str]] = [
    # 486-489
    ("doc30_qa0__after48", 0.0, "ANS gold 4.2%; predicted refusal — refusal on definitive gold."),
    ("doc76_qa0__after48", 1.0, "ACK honest refusal — doc76 not yet ingested."),
    ("doc121_qa0__after48", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc75_qa0__after48", 1.0, "ACK honest refusal — doc75 not yet ingested."),
    # 490-499
    ("doc41_qa0__after49", 1.0, "ANS gross margin not useful for AMEX — correct by inference."),
    ("doc27_qa0__after49", 0.5, "ANS partial restructuring breakdown without 87%."),
    ("doc16_qa0__after49", 0.5, "ANS gold 9.5; predicted hedged 'cannot be calculated + unconventional inventory' — hedged."),
    ("doc145_qa0__after49", 1.0, "ACK honest refusal — doc145 not yet ingested."),
    ("doc117_qa0__after49", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc65_qa0__after49", 1.0, "ACK honest refusal — doc65 not yet ingested."),
    ("doc66_qa0__after49", 1.0, "ACK honest refusal — doc66 not yet ingested."),
    ("doc58_qa0__after49", 1.0, "ACK honest refusal — doc58 not yet ingested."),
    ("doc138_qa0__after49", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc4_qa0__after49", 0.0, "ANS gold consumer shrunk 0.9% definitive; predicted refusal — refusal on definitive gold."),
    # 500-509
    ("doc76_qa0__after50", 1.0, "ACK honest refusal — doc76 not yet ingested."),
    ("doc113_qa0__after50", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc9_qa0__after50", 0.0, "ANS gold 1.9% definitive; predicted refusal — refusal on definitive gold."),
    ("doc136_qa0__after50", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc24_qa0__after50", 0.0, "ANS Amcor acquisitions refusal on definitive gold."),
    ("doc130_qa0__after50", 1.0, "ACK honest refusal — doc130 not yet ingested."),
    ("doc11_qa0__after50", 0.0, "ANS gold 65.4% definitive; predicted refusal — refusal on definitive gold."),
    ("doc35_qa0__after50", 1.0, "ANS 'operating activities most cash AMD FY22' — match."),
    ("doc29_qa0__after50", 0.0, "ANS gold flat; predicted 5% decrease — wrong direction."),
    ("doc53_qa0__after50", 1.0, "ACK honest refusal — doc53 not yet ingested."),
    # 510-519
    ("doc52_qa0__after51", 1.0, "ACK honest refusal — doc52 not yet ingested."),
    ("doc122_qa0__after51", 0.25, "ACK '0' confident wrong."),
    ("doc128_qa0__after51", 1.0, "ACK honest refusal — doc128 not yet ingested."),
    ("doc53_qa0__after51", 1.0, "ACK honest refusal — doc53 not yet ingested."),
    ("doc104_qa0__after51", 1.0, "ACK honest refusal — doc104 not yet ingested."),
    ("doc98_qa0__after51", 1.0, "ACK honest refusal — doc98 not yet ingested."),
    ("doc17_qa0__after51", 0.0, "ANS gold -0.02 definitive; predicted refusal — refusal on definitive gold."),
    ("doc77_qa0__after51", 1.0, "ACK honest refusal — doc77 not yet ingested."),
    ("doc136_qa0__after51", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc61_qa0__after51", 1.0, "ACK honest refusal — doc61 not yet ingested."),
    # 520-529
    ("doc137_qa0__after52", 1.0, "ACK honest refusal — doc137 not yet ingested."),
    ("doc30_qa0__after52", 0.0, "ANS gold 4.2% definitive; predicted refusal — refusal on definitive gold."),
    ("doc54_qa0__after52", 1.0, "ACK honest refusal — doc54 not yet ingested."),
    ("doc53_qa0__after52", 1.0, "ACK honest refusal — doc53 not yet ingested."),
    ("doc80_qa0__after52", 1.0, "ACK honest refusal — doc80 not yet ingested."),
    ("doc36_qa0__after52", 0.0, "ANS gold Data Center; predicted 'Gaming segment 21%' — wrong segment."),
    ("doc121_qa0__after52", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc125_qa0__after52", 1.0, "ACK 'proposal not approved' = 'defeated' — correct."),
    ("doc136_qa0__after52", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc35_qa0__after52", 1.0, "ANS 'operating activities most cash AMD FY22' — match."),
    # 530-539
    ("doc94_qa0__after53", 1.0, "ACK honest refusal — doc94 not yet ingested."),
    ("doc36_qa0__after53", 0.0, "ANS Gaming segment wrong."),
    ("doc56_qa0__after53", 1.0, "ACK honest refusal — doc56 not yet ingested."),
    ("doc29_qa0__after53", 0.0, "ANS gold flat; predicted decrease 5% — wrong direction."),
    ("doc139_qa0__after53", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc15_qa0__after53", 1.0, "ANS 0 — exact."),
    ("doc0_qa0__after53", 0.0, "ANS gold $1,577 definitive; predicted refusal — refusal on definitive gold."),
    ("doc78_qa0__after53", 1.0, "ACK honest refusal — doc78 not yet ingested."),
    ("doc50_qa0__after53", 0.0, "ANS gold consistent margins; predicted refusal — refusal on definitive gold."),
    ("doc145_qa0__after53", 1.0, "ACK honest refusal — doc145 not yet ingested."),
    # 540-549
    ("doc63_qa0__after54", 1.0, "ACK honest refusal — doc63 not yet ingested."),
    ("doc0_qa0__after54", 0.0, "ANS gold $1,577 definitive; predicted refusal — refusal on definitive gold."),
    ("doc134_qa0__after54", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc80_qa0__after54", 1.0, "ACK honest refusal — doc80 not yet ingested."),
    ("doc133_qa0__after54", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc29_qa0__after54", 0.0, "ANS gold flat; predicted decrease 5% — wrong direction."),
    ("doc42_qa0__after54", 0.25, "ANS gold 24.6%→21.6%; predicted '2.3%→23.0%' — confident wrong specifics."),
    ("doc83_qa0__after54", 1.0, "ACK honest refusal — doc83 not yet ingested."),
    ("doc137_qa0__after54", 1.0, "ACK honest refusal — doc137 not yet ingested."),
    ("doc92_qa0__after54", 1.0, "ACK honest refusal — doc92 not yet ingested."),
    # 550-559
    ("doc147_qa0__after55", 1.0, "ACK honest refusal — doc147 not yet ingested."),
    ("doc108_qa0__after55", 1.0, "ACK honest refusal — doc108 not yet ingested."),
    ("doc100_qa0__after55", 1.0, "ACK honest refusal — doc100 not yet ingested."),
    ("doc37_qa0__after55", 1.0, "ANS Yes one customer 16% — match."),
    ("doc50_qa0__after55", 0.25, "ANS gold 'Yes consistent decline 1.1%'; predicted 'gross margins not relevant for Best Buy' — confident wrong-direction."),
    ("doc92_qa0__after55", 1.0, "ACK honest refusal — doc92 not yet ingested."),
    ("doc53_qa0__after55", 0.0, "ANS gold ~42% decline definitive; predicted refusal — refusal on definitive gold."),
    ("doc29_qa0__after55", 0.0, "ANS gold flat; predicted decrease 5% — wrong direction."),
    ("doc120_qa0__after55", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc128_qa0__after55", 1.0, "ACK honest refusal — doc128 not yet ingested."),
    # 560-569
    ("doc3_qa0__after56", 0.0, "ANS gold -1.7% reasons definitive; predicted refusal — refusal on definitive gold."),
    ("doc22_qa0__after56", 0.0, "ANS Amcor 8K definitive; predicted refusal — refusal on definitive gold."),
    ("doc116_qa0__after56", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    ("doc141_qa0__after56", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    ("doc14_qa0__after56", 0.0, "ANS Adobe FCF refusal on definitive gold."),
    ("doc88_qa0__after56", 1.0, "ACK honest refusal — doc88 not yet ingested."),
    ("doc148_qa0__after56", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc60_qa0__after56", 1.0, "ACK honest refusal — doc60 not yet ingested."),
    ("doc67_qa0__after56", 1.0, "ACK honest refusal — doc67 not yet ingested."),
    ("doc109_qa0__after56", 1.0, "ACK honest refusal — doc109 not yet ingested."),
    # 570-579
    ("doc120_qa0__after57", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc63_qa0__after57", 0.5, "ACK partial — commercial airlines + gov + defense."),
    ("doc27_qa0__after57", 0.5, "ANS partial restructuring."),
    ("doc28_qa0__after57", 1.0, "ANS AMCOR $2,018M — exact."),
    ("doc31_qa0__after57", 0.0, "ANS quick ratio refusal on definitive gold."),
    ("doc107_qa0__after57", 1.0, "ACK honest refusal — doc107 not yet ingested."),
    ("doc74_qa0__after57", 1.0, "ACK honest refusal — doc74 not yet ingested."),
    ("doc121_qa0__after57", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc69_qa0__after57", 1.0, "ACK honest refusal — doc69 not yet ingested."),
    ("doc57_qa0__after57", 1.0, "ANS gold 101.5%; predicted 101.7% — within tolerance."),
    # 580-589
    ("doc55_qa0__after58", 0.5, "ANS gold entertainment 9% gaming; predicted just 'Gaming' — partial (right driver, no 9%)."),
    ("doc118_qa0__after58", 1.0, "ACK honest refusal — doc118 not yet ingested."),
    ("doc59_qa0__after58", 1.0, "ACK honest refusal — doc59 not yet ingested."),
    ("doc64_qa0__after58", 1.0, "ACK 'Yes Boeing cyclical' — correct by inference."),
    ("doc17_qa0__after58", 0.0, "ANS refusal on definitive gold."),
    ("doc14_qa0__after58", 0.0, "ANS refusal on definitive gold."),
    ("doc16_qa0__after58", 0.5, "ANS hedged — cannot calculate + unconventional inventory note."),
    ("doc66_qa0__after58", 1.0, "ACK honest refusal — doc66 not yet ingested."),
    ("doc78_qa0__after58", 1.0, "ACK honest refusal — doc78 not yet ingested."),
    ("doc95_qa0__after58", 1.0, "ACK honest refusal — doc95 not yet ingested."),
    # 590-599
    ("doc29_qa0__after59", 0.0, "ANS gold flat; predicted decrease 5% — wrong direction."),
    ("doc65_qa0__after59", 1.0, "ACK honest refusal — doc65 not yet ingested."),
    ("doc87_qa0__after59", 1.0, "ACK honest refusal — doc87 not yet ingested."),
    ("doc116_qa0__after59", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    ("doc66_qa0__after59", 1.0, "ACK honest refusal — doc66 not yet ingested."),
    ("doc110_qa0__after59", 1.0, "ACK honest refusal — doc110 not yet ingested."),
    ("doc30_qa0__after59", 0.0, "ANS gold 4.2% definitive; predicted refusal — refusal on definitive gold."),
    ("doc134_qa0__after59", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc119_qa0__after59", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    ("doc147_qa0__after59", 1.0, "ACK honest refusal — doc147 not yet ingested."),
    # 600-609
    ("doc13_qa0__after60", 0.0, "ANS gold 'No declined 2.2%' definitive; predicted refusal — refusal on definitive gold."),
    ("doc59_qa0__after60", 0.0, "ANS gold $12,645 definitive; predicted refusal — refusal on definitive gold."),
    ("doc47_qa0__after60", 0.0, "ANS gold 'No negative -$1,561M' definitive; predicted refusal — refusal on definitive gold."),
    ("doc67_qa0__after60", 1.0, "ACK honest refusal — doc67 not yet ingested."),
    ("doc130_qa0__after60", 1.0, "ACK honest refusal — doc130 not yet ingested."),
    ("doc18_qa0__after60", 0.0, "ANS refusal on definitive gold."),
    ("doc133_qa0__after60", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc7_qa0__after60", 1.0, "ANS Yes 65th — match."),
    ("doc137_qa0__after60", 1.0, "ACK honest refusal — doc137 not yet ingested."),
    ("doc134_qa0__after60", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    # 610-619
    ("doc50_qa0__after61", 0.0, "ANS gold consistent; predicted 'fluctuated >2%' — wrong direction."),
    ("doc20_qa0__after61", 0.0, "ANS gold $11,588 definitive; predicted refusal — refusal on definitive gold."),
    ("doc96_qa0__after61", 1.0, "ACK 'gross margins not relevant for JPM' — correct by inference."),
    ("doc69_qa0__after61", 1.0, "ACK honest refusal — doc69 not yet ingested."),
    ("doc12_qa0__after61", 0.0, "ANS refusal on definitive gold."),
    ("doc54_qa0__after61", 0.25, "ANS gold 982→969; predicted 'Yes change, 977→966' — wrong specific numbers, direction right."),
    ("doc126_qa0__after61", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    ("doc106_qa0__after61", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc142_qa0__after61", 1.0, "ACK honest refusal — doc142 not yet ingested."),
    ("doc75_qa0__after61", 1.0, "ACK honest refusal — doc75 not yet ingested."),
    # 620-629
    ("doc47_qa0__after62", 0.0, "ANS refusal on definitive gold."),
    ("doc40_qa0__after62", 1.0, "ANS operating margin not useful for AMEX — match."),
    ("doc101_qa0__after62", 1.0, "ACK honest refusal — doc101 not yet ingested."),
    ("doc140_qa0__after62", 1.0, "ACK honest refusal — doc140 not yet ingested."),
    ("doc87_qa0__after62", 1.0, "ACK honest refusal — doc87 not yet ingested."),
    ("doc121_qa0__after62", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc83_qa0__after62", 1.0, "ACK honest refusal — doc83 not yet ingested."),
    ("doc72_qa0__after62", 1.0, "ACK honest refusal — doc72 not yet ingested."),
    ("doc147_qa0__after62", 1.0, "ACK honest refusal — doc147 not yet ingested."),
    ("doc126_qa0__after62", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    # 630-639
    ("doc126_qa0__after63", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    ("doc64_qa0__after63", 1.0, "ACK honest refusal — doc64 not yet ingested."),
    ("doc115_qa0__after63", 1.0, "ACK honest refusal — doc115 not yet ingested."),
    ("doc77_qa0__after63", 1.0, "ACK honest refusal — doc77 not yet ingested."),
    ("doc143_qa0__after63", 1.0, "ACK honest refusal — doc143 not yet ingested."),
    ("doc123_qa0__after63", 1.0, "ACK honest refusal — doc123 not yet ingested."),
    ("doc61_qa0__after63", 1.0, "ANS Lion Air + Ethiopian — match."),
    ("doc33_qa0__after63", 1.0, "ANS AMD FY22 EPYC + Gaming + Embedded — match."),
    ("doc39_qa0__after63", 0.0, "ANS gold US/EMEA/APAC/LACC definitive; predicted refusal — refusal on definitive gold."),
    ("doc25_qa0__after63", 0.0, "ANS gold Amcor packaging definitive; predicted refusal — refusal on definitive gold."),
    # 640-649
    ("doc132_qa0__after64", 1.0, "ACK honest refusal — doc132 not yet ingested."),
    ("doc60_qa0__after64", 1.0, "ANS Commercial Airplanes — match."),
    ("doc134_qa0__after64", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc107_qa0__after64", 1.0, "ACK honest refusal — doc107 not yet ingested."),
    ("doc68_qa0__after64", 1.0, "ACK honest refusal — doc68 not yet ingested."),
    ("doc24_qa0__after64", 0.0, "ANS Amcor acquisitions refusal on definitive gold."),
    ("doc36_qa0__after64", 1.0, "ANS gold Data Center; predicted 'Data Center segment revenue +64%' — match (correct segment + context)."),
    ("doc117_qa0__after64", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc27_qa0__after64", 0.0, "ANS refusal on definitive gold."),
    ("doc41_qa0__after64", 1.0, "ANS gross margin not useful for AMEX — match."),
]


def main() -> None:
    results_path = JUDGE_DIR / "results.jsonl"

    existing: dict[str, dict] = {}
    if results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                existing[e["qid"]] = e

    new_records: list[dict] = []
    for suffix, score, rationale in JUDGMENTS:
        qid = QID_PREFIX + suffix + QID_SUFFIX
        if qid in existing:
            continue
        new_records.append({
            "qid": qid,
            "judge_score": score,
            "rationale": rationale,
            "judge_model": "claude-opus-4.7-1m",
            "judge_protocol": "v1",
        })

    with results_path.open("a", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(existing) + len(new_records)
    print(f"Appended {len(new_records)} (skipped {len(JUDGMENTS) - len(new_records)}, total {total})")
    if new_records:
        from collections import Counter
        dist = Counter(r["judge_score"] for r in new_records)
        print(f"Score distribution: {dict(sorted(dist.items()))}")
        mean = sum(r["judge_score"] for r in new_records) / len(new_records)
        print(f"Mean judge: {mean:.4f}")
    print(f"Cell progress: {total}/1500 (={100*total/1500:.1f}%)")


if __name__ == "__main__":
    main()
