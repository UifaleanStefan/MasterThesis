"""Claude manual judging — Phase 1.9 FB calibration dump-all (entries 568-714)."""

from __future__ import annotations

import json
from pathlib import Path

JUDGE_DIR = Path(
    "results/stage3/judge_queue/financebench__dump-all__calibration__seed42"
)
QID_PREFIX = "financebench__dump-all__calibration__"
QID_SUFFIX = "__seed42"

JUDGMENTS: list[tuple[str, float, str]] = [
    # 568-569
    ("doc67_qa0__after56", 1.0, "ACK honest refusal — doc67 not yet ingested."),
    ("doc109_qa0__after56", 1.0, "ACK honest refusal — doc109 not yet ingested."),
    # 570-579
    ("doc120_qa0__after57", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc63_qa0__after57", 0.5, "ACK partial 'commercial airlines + gov + defense' — no US govt 40%."),
    ("doc27_qa0__after57", 0.0, "ANS refusal on definitive gold (87% restructuring)."),
    ("doc28_qa0__after57", 0.0, "ANS refusal on definitive gold (AMCOR EBITDA)."),
    ("doc31_qa0__after57", 0.0, "ANS refusal on definitive gold (quick ratio 1.57)."),
    ("doc107_qa0__after57", 1.0, "ACK honest refusal — doc107 not yet ingested."),
    ("doc74_qa0__after57", 1.0, "ACK honest refusal — doc74 not yet ingested."),
    ("doc121_qa0__after57", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc69_qa0__after57", 1.0, "ACK honest refusal — doc69 not yet ingested."),
    ("doc57_qa0__after57", 1.0, "ANS gold 101.5%; predicted 101.7% — within tolerance."),
    # 580-589
    ("doc55_qa0__after58", 0.75, "ANS gold entertainment 9% from gaming; predicted 'Gaming performed best in domestic market Q2 FY24' — partial (right driver, no 9%)."),
    ("doc118_qa0__after58", 1.0, "ACK honest refusal — doc118 not yet ingested."),
    ("doc59_qa0__after58", 1.0, "ACK honest refusal — doc59 not yet ingested."),
    ("doc64_qa0__after58", 1.0, "ACK calibration: 'Yes Boeing cyclical' — correct by inference."),
    ("doc17_qa0__after58", 0.0, "ANS refusal on definitive gold."),
    ("doc14_qa0__after58", 0.0, "ANS refusal on definitive gold."),
    ("doc16_qa0__after58", 0.0, "ANS refusal on definitive gold (9.5 turnover)."),
    ("doc66_qa0__after58", 1.0, "ACK honest refusal — doc66 not yet ingested."),
    ("doc78_qa0__after58", 1.0, "ACK honest refusal — doc78 not yet ingested."),
    ("doc95_qa0__after58", 1.0, "ACK honest refusal — doc95 not yet ingested."),
    # 590-599
    ("doc29_qa0__after59", 0.0, "ANS refusal on definitive gold."),
    ("doc65_qa0__after59", 1.0, "ACK honest refusal — doc65 not yet ingested."),
    ("doc87_qa0__after59", 1.0, "ACK honest refusal — doc87 not yet ingested."),
    ("doc116_qa0__after59", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    ("doc66_qa0__after59", 1.0, "ACK honest refusal — doc66 not yet ingested."),
    ("doc110_qa0__after59", 1.0, "ACK honest refusal — doc110 not yet ingested."),
    ("doc30_qa0__after59", 0.0, "ANS refusal on definitive gold (4.2%)."),
    ("doc134_qa0__after59", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc119_qa0__after59", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    ("doc147_qa0__after59", 1.0, "ACK honest refusal — doc147 not yet ingested."),
    # 600-609
    ("doc13_qa0__after60", 0.0, "ANS gold 'No declined 2.2%'; predicted refusal — refusal on definitive gold."),
    ("doc59_qa0__after60", 1.0, "ANS gold $12,645; predicted $12,645 — exact match!"),
    ("doc47_qa0__after60", 0.0, "ANS refusal on definitive gold."),
    ("doc67_qa0__after60", 1.0, "ACK honest refusal — doc67 not yet ingested."),
    ("doc130_qa0__after60", 1.0, "ACK honest refusal — doc130 not yet ingested."),
    ("doc18_qa0__after60", 0.0, "ANS refusal on definitive gold."),
    ("doc133_qa0__after60", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc7_qa0__after60", 0.0, "ANS gold Yes 65th 3M dividend; predicted refusal — refusal on definitive gold."),
    ("doc137_qa0__after60", 1.0, "ACK honest refusal — doc137 not yet ingested."),
    ("doc134_qa0__after60", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    # 610-619
    ("doc50_qa0__after61", 0.0, "ANS gold consistent margins; predicted 'fluctuated >2%' — wrong direction."),
    ("doc20_qa0__after61", 0.0, "ANS refusal on definitive gold ($11,588)."),
    ("doc96_qa0__after61", 1.0, "ACK 'JPM gross margins not relevant' — correct by inference."),
    ("doc69_qa0__after61", 1.0, "ACK honest refusal — doc69 not yet ingested."),
    ("doc12_qa0__after61", 0.0, "ANS refusal on definitive gold."),
    ("doc54_qa0__after61", 0.25, "ANS gold 982→969; predicted '977→966' — wrong specific numbers."),
    ("doc126_qa0__after61", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    ("doc106_qa0__after61", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc142_qa0__after61", 1.0, "ACK honest refusal — doc142 not yet ingested."),
    ("doc75_qa0__after61", 1.0, "ACK honest refusal — doc75 not yet ingested."),
    # 620-629
    ("doc47_qa0__after62", 0.0, "ANS refusal on definitive gold."),
    ("doc40_qa0__after62", 0.0, "ANS refusal on definitive gold."),
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
    ("doc64_qa0__after63", 1.0, "ACK 'Yes Boeing cyclical' — correct by inference."),
    ("doc115_qa0__after63", 1.0, "ACK honest refusal — doc115 not yet ingested."),
    ("doc77_qa0__after63", 1.0, "ACK honest refusal — doc77 not yet ingested."),
    ("doc143_qa0__after63", 1.0, "ACK honest refusal — doc143 not yet ingested."),
    ("doc123_qa0__after63", 1.0, "ACK honest refusal — doc123 not yet ingested."),
    ("doc61_qa0__after63", 1.0, "ANS gold Lion Air + Ethiopian crashes; predicted same flights with dates — match."),
    ("doc33_qa0__after63", 0.0, "ANS refusal on definitive gold."),
    ("doc39_qa0__after63", 0.0, "ANS refusal on definitive gold."),
    ("doc25_qa0__after63", 0.0, "ANS refusal on definitive gold."),
    # 640-649
    ("doc132_qa0__after64", 1.0, "ACK honest refusal — doc132 not yet ingested."),
    ("doc60_qa0__after64", 1.0, "ANS Yes Commercial Airplanes $25,867M — match."),
    ("doc134_qa0__after64", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc107_qa0__after64", 1.0, "ACK honest refusal — doc107 not yet ingested."),
    ("doc68_qa0__after64", 1.0, "ACK honest refusal — doc68 not yet ingested."),
    ("doc24_qa0__after64", 0.0, "ANS refusal on definitive gold."),
    ("doc36_qa0__after64", 0.0, "ANS refusal on definitive gold (Data Center)."),
    ("doc117_qa0__after64", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc27_qa0__after64", 0.0, "ANS refusal on definitive gold."),
    ("doc41_qa0__after64", 0.0, "ANS refusal on definitive gold."),
    # 650-659
    ("doc105_qa0__after65", 1.0, "ACK honest refusal — doc105 not yet ingested."),
    ("doc146_qa0__after65", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc26_qa0__after65", 0.0, "ANS refusal on definitive gold."),
    ("doc18_qa0__after65", 0.0, "ANS refusal on definitive gold."),
    ("doc89_qa0__after65", 1.0, "ACK honest refusal — doc89 not yet ingested."),
    ("doc114_qa0__after65", 1.0, "ACK honest refusal — doc114 not yet ingested."),
    ("doc102_qa0__after65", 1.0, "ACK honest refusal — doc102 not yet ingested."),
    ("doc38_qa0__after65", 1.0, "ANS gold 'There are none' AMEX debt securities; predicted refusal — effectively equivalent."),
    ("doc94_qa0__after65", 1.0, "ACK honest refusal — doc94 not yet ingested."),
    ("doc145_qa0__after65", 1.0, "ACK honest refusal — doc145 not yet ingested."),
    # 660-669
    ("doc55_qa0__after66", 0.0, "ANS refusal on definitive gold (entertainment 9% gaming)."),
    ("doc51_qa0__after66", 0.0, "ANS refusal on definitive gold (Best Buy acquisitions)."),
    ("doc62_qa0__after66", 0.0, "ANS gold Yes Boeing improving gross margin; predicted 'gross margin not useful' — wrong direction."),
    ("doc139_qa0__after66", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc142_qa0__after66", 1.0, "ACK honest refusal — doc142 not yet ingested."),
    ("doc149_qa0__after66", 1.0, "ACK honest refusal — doc149 not yet ingested."),
    ("doc116_qa0__after66", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    ("doc103_qa0__after66", 1.0, "ACK honest refusal — doc103 not yet ingested."),
    ("doc66_qa0__after66", 0.5, "ANS gold 0.62% vs -14.76%; predicted Boeing tax lower in FY22 with $(31)M expense vs $743M benefit — direction right, no specific rates."),
    ("doc17_qa0__after66", 0.0, "ANS refusal on definitive gold."),
    # 670-679
    ("doc74_qa0__after67", 1.0, "ACK honest refusal — doc74 not yet ingested."),
    ("doc76_qa0__after67", 1.0, "ACK honest refusal — doc76 not yet ingested."),
    ("doc25_qa0__after67", 0.0, "ANS refusal on definitive gold."),
    ("doc71_qa0__after67", 1.0, "ACK honest refusal — doc71 not yet ingested."),
    ("doc113_qa0__after67", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc2_qa0__after67", 0.0, "ANS refusal on definitive gold (No efficient CAPEX)."),
    ("doc3_qa0__after67", 0.0, "ANS refusal on definitive gold."),
    ("doc141_qa0__after67", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    ("doc35_qa0__after67", 0.0, "ANS refusal on definitive gold."),
    ("doc39_qa0__after67", 0.0, "ANS refusal on definitive gold."),
    # 680-689
    ("doc66_qa0__after68", 0.5, "ANS Boeing tax FY22 lower than FY21 with $(31)M/$743M — direction right, no specific rates."),
    ("doc25_qa0__after68", 0.0, "ANS refusal on definitive gold."),
    ("doc99_qa0__after68", 1.0, "ACK honest refusal — doc99 not yet ingested."),
    ("doc85_qa0__after68", 1.0, "ACK honest refusal — doc85 not yet ingested."),
    ("doc24_qa0__after68", 0.0, "ANS refusal on definitive gold."),
    ("doc126_qa0__after68", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    ("doc32_qa0__after68", 0.5, "ANS gold AMD CPUs/GPUs/DPUs/FPGAs/SoC; predicted 'microprocessors, graphics processors, SoC' — partial (covers core products less detail)."),
    ("doc15_qa0__after68", 1.0, "ANS 0 — exact."),
    ("doc82_qa0__after68", 1.0, "ACK honest refusal — doc82 not yet ingested."),
    ("doc121_qa0__after68", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    # 690-699
    ("doc105_qa0__after69", 1.0, "ACK honest refusal — doc105 not yet ingested."),
    ("doc85_qa0__after69", 1.0, "ACK honest refusal — doc85 not yet ingested."),
    ("doc139_qa0__after69", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc30_qa0__after69", 0.0, "ANS refusal on definitive gold."),
    ("doc108_qa0__after69", 1.0, "ACK honest refusal — doc108 not yet ingested."),
    ("doc32_qa0__after69", 0.0, "ANS refusal on definitive gold (AMD products)."),
    ("doc87_qa0__after69", 1.0, "ACK honest refusal — doc87 not yet ingested."),
    ("doc93_qa0__after69", 1.0, "ACK honest refusal — doc93 not yet ingested."),
    ("doc65_qa0__after69", 1.0, "ANS gold Boeing 737/777X/787 production increase 2023; predicted 787→5/month, 737 gradual, 777X resume 2023 — match all 3 aircraft."),
    ("doc16_qa0__after69", 0.0, "ANS refusal on definitive gold."),
    # 700-709
    ("doc26_qa0__after70", 0.0, "ANS refusal on definitive gold."),
    ("doc66_qa0__after70", 0.5, "ANS 'Boeing tax FY22 lower than FY21' — direction right, no specific rates."),
    ("doc93_qa0__after70", 1.0, "ACK honest refusal — doc93 not yet ingested."),
    ("doc138_qa0__after70", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc129_qa0__after70", 0.25, "ACK '2 percentage points' confident wrong (gold 1 pp PepsiCo EPS)."),
    ("doc71_qa0__after70", 0.25, "ACK calibration: confident wrong '4.4%' (gold 10.3%)."),
    ("doc135_qa0__after70", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc65_qa0__after70", 1.0, "ANS Boeing production rates — match."),
    ("doc104_qa0__after70", 1.0, "ACK honest refusal — doc104 not yet ingested."),
    ("doc91_qa0__after70", 1.0, "ACK honest refusal — doc91 not yet ingested."),
    # 710-714
    ("doc10_qa0__after71", 0.0, "ANS refusal on definitive gold (0.66)."),
    ("doc46_qa0__after71", 0.0, "ANS refusal on definitive gold ($1,832)."),
    ("doc59_qa0__after71", 0.0, "ANS refusal on definitive gold."),
    ("doc95_qa0__after71", 1.0, "ACK honest refusal — doc95 not yet ingested."),
    ("doc55_qa0__after71", 0.0, "ANS refusal on definitive gold (entertainment 9% gaming)."),
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
