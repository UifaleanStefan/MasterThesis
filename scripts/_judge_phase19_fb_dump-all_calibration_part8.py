"""Claude manual judging — Phase 1.9 FB calibration dump-all (entries 1008-1155)."""

from __future__ import annotations

import json
from pathlib import Path

JUDGE_DIR = Path(
    "results/stage3/judge_queue/financebench__dump-all__calibration__seed42"
)
QID_PREFIX = "financebench__dump-all__calibration__"
QID_SUFFIX = "__seed42"

JUDGMENTS: list[tuple[str, float, str]] = [
    # 1008-1009
    ("doc65_qa0__after100", 0.0, "ANS refusal on definitive gold (Boeing 737/777X/787)."),
    ("doc63_qa0__after100", 0.0, "ANS refusal on definitive gold (Boeing customers)."),
    # 1010-1019
    ("doc81_qa0__after101", 0.0, "ANS refusal on definitive gold (-3.7)."),
    ("doc114_qa0__after101", 1.0, "ACK honest refusal — doc114 not yet ingested."),
    ("doc35_qa0__after101", 0.0, "ANS refusal on definitive gold."),
    ("doc41_qa0__after101", 0.0, "ANS refusal on definitive gold."),
    ("doc100_qa0__after101", 1.0, "ANS gold 1.33; predicted 1.30 — within tolerance."),
    ("doc98_qa0__after101", 1.0, "ANS Yes decreased $7M VaR — match."),
    ("doc78_qa0__after101", 0.0, "ANS refusal on definitive gold (Yes $0.55/quarter)."),
    ("doc75_qa0__after101", 0.0, "ANS refusal on definitive gold."),
    ("doc96_qa0__after101", 1.0, "ANS JPM gross margins not relevant — match."),
    ("doc125_qa0__after101", 1.0, "ACK 'proposal not approved 66%' — correct."),
    # 1020-1029
    ("doc31_qa0__after102", 0.0, "ANS refusal on definitive gold."),
    ("doc39_qa0__after102", 0.0, "ANS refusal on definitive gold."),
    ("doc24_qa0__after102", 0.0, "ANS refusal on definitive gold (Amcor acquisitions)."),
    ("doc68_qa0__after102", 0.0, "ANS gold 39.7%; predicted 32.2% — wrong specific."),
    ("doc119_qa0__after102", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    ("doc44_qa0__after102", 0.0, "ANS refusal on definitive gold (Yes Amex retention)."),
    ("doc36_qa0__after102", 0.0, "ANS refusal on definitive gold (Data Center)."),
    ("doc59_qa0__after102", 0.0, "ANS refusal on definitive gold."),
    ("doc46_qa0__after102", 0.0, "ANS refusal on definitive gold."),
    ("doc108_qa0__after102", 1.0, "ACK honest refusal — doc108 not yet ingested."),
    # 1030-1039
    ("doc108_qa0__after103", 1.0, "ACK honest refusal — doc108 not yet ingested."),
    ("doc61_qa0__after103", 0.0, "ANS refusal on definitive gold."),
    ("doc135_qa0__after103", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc60_qa0__after103", 0.0, "ANS refusal on definitive gold."),
    ("doc36_qa0__after103", 0.0, "ANS refusal on definitive gold."),
    ("doc51_qa0__after103", 0.0, "ANS refusal on definitive gold."),
    ("doc85_qa0__after103", 0.0, "ANS refusal on definitive gold (No JnJ 1.3%)."),
    ("doc105_qa0__after103", 1.0, "ACK honest refusal — doc105 not yet ingested."),
    ("doc71_qa0__after103", 0.0, "ANS refusal on definitive gold (10.3%)."),
    ("doc137_qa0__after103", 1.0, "ACK honest refusal — doc137 not yet ingested."),
    # 1040-1049
    ("doc46_qa0__after104", 0.0, "ANS refusal on definitive gold."),
    ("doc136_qa0__after104", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc121_qa0__after104", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc96_qa0__after104", 1.0, "ANS JPM gross margins not relevant — match."),
    ("doc16_qa0__after104", 0.0, "ANS refusal on definitive gold."),
    ("doc80_qa0__after104", 0.0, "ANS refusal on definitive gold (Yes Richard A. Johnson)."),
    ("doc31_qa0__after104", 0.0, "ANS refusal on definitive gold."),
    ("doc14_qa0__after104", 0.0, "ANS refusal on definitive gold."),
    ("doc101_qa0__after104", 1.0, "ANS Lockheed $5,818M — exact."),
    ("doc103_qa0__after104", 1.0, "ANS gold $303; predicted $302.578M — within tolerance."),
    # 1050-1059
    ("doc22_qa0__after105", 0.0, "ANS refusal on definitive gold (Amcor 8K)."),
    ("doc119_qa0__after105", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    ("doc25_qa0__after105", 1.0, "ANS Amcor packaging — match."),
    ("doc146_qa0__after105", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc62_qa0__after105", 0.0, "ANS refusal on definitive gold (Yes Boeing improving)."),
    ("doc98_qa0__after105", 1.0, "ANS Yes decreased $7M VaR — match."),
    ("doc1_qa0__after105", 0.25, "ANS gold $8.70 (likely $8.70B); predicted '$20.73B' — confident wrong specific (or different interpretation, far off either way)."),
    ("doc138_qa0__after105", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc123_qa0__after105", 1.0, "ACK honest refusal — doc123 not yet ingested."),
    ("doc76_qa0__after105", 0.0, "ANS refusal on definitive gold."),
    # 1060-1069
    ("doc124_qa0__after106", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc70_qa0__after106", 0.0, "ANS refusal on definitive gold (63.86 DPO)."),
    ("doc28_qa0__after106", 0.0, "ANS refusal on definitive gold (AMCOR EBITDA)."),
    ("doc30_qa0__after106", 0.0, "ANS refusal on definitive gold."),
    ("doc85_qa0__after106", 0.0, "ANS refusal on definitive gold."),
    ("doc130_qa0__after106", 1.0, "ACK honest refusal — doc130 not yet ingested."),
    ("doc87_qa0__after106", 0.0, "ANS refusal on definitive gold."),
    ("doc135_qa0__after106", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc148_qa0__after106", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc49_qa0__after106", 0.0, "ANS refusal on definitive gold ($5,409)."),
    # 1070-1079
    ("doc9_qa0__after107", 0.0, "ANS refusal on definitive gold (1.9%)."),
    ("doc134_qa0__after107", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc13_qa0__after107", 0.0, "ANS refusal on definitive gold."),
    ("doc142_qa0__after107", 1.0, "ACK honest refusal — doc142 not yet ingested."),
    ("doc127_qa0__after107", 1.0, "ACK honest refusal — doc127 not yet ingested."),
    ("doc122_qa0__after107", 0.25, "ACK '0' confident wrong."),
    ("doc133_qa0__after107", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc103_qa0__after107", 1.0, "ANS $302.578M ≈ $303 — within tolerance."),
    ("doc139_qa0__after107", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc87_qa0__after107", 0.0, "ANS refusal on definitive gold."),
    # 1080-1089
    ("doc75_qa0__after108", 0.0, "ANS refusal on definitive gold."),
    ("doc90_qa0__after108", 1.0, "ANS Consumer Health discontinued — exact."),
    ("doc98_qa0__after108", 0.0, "ANS refusal on definitive gold."),
    ("doc140_qa0__after108", 1.0, "ACK honest refusal — doc140 not yet ingested."),
    ("doc42_qa0__after108", 0.0, "ANS refusal on definitive gold."),
    ("doc43_qa0__after108", 0.25, "ANS 'long-term debt' confident wrong (gold Customer deposits)."),
    ("doc51_qa0__after108", 0.0, "ANS refusal on definitive gold."),
    ("doc68_qa0__after108", 0.0, "ANS refusal on definitive gold."),
    ("doc45_qa0__after108", 0.0, "ANS refusal on definitive gold."),
    ("doc108_qa0__after108", 1.0, "ANS gold MGM China 44%; predicted MGM China $674M (44% decline) — match (correct region + 44%)."),
    # 1090-1099
    ("doc26_qa0__after109", 0.0, "ANS refusal on definitive gold."),
    ("doc7_qa0__after109", 0.0, "ANS refusal on definitive gold (Yes 65th)."),
    ("doc119_qa0__after109", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    ("doc14_qa0__after109", 0.0, "ANS refusal on definitive gold."),
    ("doc44_qa0__after109", 0.0, "ANS refusal on definitive gold."),
    ("doc102_qa0__after109", 0.0, "ANS gold 0.4%; predicted 1.4% — wrong specific."),
    ("doc65_qa0__after109", 0.0, "ANS refusal on definitive gold."),
    ("doc133_qa0__after109", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc18_qa0__after109", 0.0, "ANS refusal on definitive gold."),
    ("doc134_qa0__after109", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    # 1100-1109
    ("doc120_qa0__after110", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc7_qa0__after110", 0.0, "ANS refusal on definitive gold."),
    ("doc72_qa0__after110", 0.0, "ANS refusal on definitive gold (Corning tax)."),
    ("doc35_qa0__after110", 0.0, "ANS refusal on definitive gold."),
    ("doc99_qa0__after110", 1.0, "ACK honest refusal — doc99 not yet ingested."),
    ("doc33_qa0__after110", 0.0, "ANS refusal on definitive gold."),
    ("doc145_qa0__after110", 1.0, "ACK honest refusal — doc145 not yet ingested."),
    ("doc90_qa0__after110", 1.0, "ANS Consumer Health discontinued — exact."),
    ("doc97_qa0__after110", 0.0, "ANS refusal on definitive gold (Corporate & Investment Bank)."),
    ("doc122_qa0__after110", 0.25, "ACK '0' confident wrong."),
    # 1110-1119
    ("doc117_qa0__after111", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc120_qa0__after111", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc26_qa0__after111", 0.0, "ANS refusal on definitive gold."),
    ("doc113_qa0__after111", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc30_qa0__after111", 0.0, "ANS refusal on definitive gold."),
    ("doc82_qa0__after111", 0.0, "ANS refusal on definitive gold."),
    ("doc36_qa0__after111", 0.0, "ANS refusal on definitive gold."),
    ("doc72_qa0__after111", 0.0, "ANS refusal on definitive gold."),
    ("doc37_qa0__after111", 0.0, "ANS refusal on definitive gold (16% customer)."),
    ("doc101_qa0__after111", 1.0, "ACK honest refusal — doc101 not yet ingested."),
    # 1120-1129
    ("doc35_qa0__after112", 1.0, "ANS gold AMD cashflow from Operations; predicted 'cash flow from operating activities brought in the most cash for AMD in FY22' — match."),
    ("doc52_qa0__after112", 0.0, "ANS refusal on definitive gold."),
    ("doc23_qa0__after112", 0.0, "ANS refusal on definitive gold (quick ratio)."),
    ("doc120_qa0__after112", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc21_qa0__after112", 0.0, "ANS refusal on definitive gold ($1,616)."),
    ("doc59_qa0__after112", 0.0, "ANS refusal on definitive gold."),
    ("doc114_qa0__after112", 1.0, "ACK honest refusal — doc114 not yet ingested."),
    ("doc92_qa0__after112", 1.0, "ACK honest refusal — doc92 not yet ingested."),
    ("doc89_qa0__after112", 0.0, "ANS refusal on definitive gold."),
    ("doc122_qa0__after112", 0.25, "ACK '0' confident wrong."),
    # 1130-1139
    ("doc139_qa0__after113", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc104_qa0__after113", 0.0, "ANS gold 7.9%; predicted 12.0% — wrong specific."),
    ("doc136_qa0__after113", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc82_qa0__after113", 0.0, "ANS refusal on definitive gold."),
    ("doc60_qa0__after113", 0.0, "ANS refusal on definitive gold."),
    ("doc89_qa0__after113", 0.0, "ANS refusal on definitive gold."),
    ("doc47_qa0__after113", 0.0, "ANS refusal on definitive gold."),
    ("doc137_qa0__after113", 1.0, "ACK honest refusal — doc137 not yet ingested."),
    ("doc56_qa0__after113", 1.0, "ANS gold 1.73; predicted 1.67 — 3.5% off, within tolerance."),
    ("doc98_qa0__after113", 0.0, "ANS refusal on definitive gold."),
    # 1140-1149
    ("doc24_qa0__after114", 0.0, "ANS refusal on definitive gold."),
    ("doc113_qa0__after114", 1.0, "ANS gold $5,466; predicted $5,466.312M — exact match."),
    ("doc27_qa0__after114", 0.0, "ANS refusal on definitive gold."),
    ("doc124_qa0__after114", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc97_qa0__after114", 0.0, "ANS refusal on definitive gold."),
    ("doc99_qa0__after114", 0.0, "ANS refusal on definitive gold."),
    ("doc131_qa0__after114", 1.0, "ACK honest refusal — doc131 not yet ingested."),
    ("doc19_qa0__after114", 0.0, "ANS refusal on definitive gold."),
    ("doc98_qa0__after114", 0.0, "ANS refusal on definitive gold."),
    ("doc12_qa0__after114", 0.0, "ANS refusal on definitive gold."),
    # 1150-1155
    ("doc80_qa0__after115", 0.5, "ANS gold Yes Richard A. Johnson; predicted 'Yes there were nominees with more votes against' — partial (Yes direction, no name)."),
    ("doc23_qa0__after115", 0.0, "ANS refusal on definitive gold."),
    ("doc111_qa0__after115", 0.0, "ANS gold No (-$2.5bn); predicted 'Yes increased' — wrong direction."),
    ("doc131_qa0__after115", 1.0, "ACK honest refusal — doc131 not yet ingested."),
    ("doc33_qa0__after115", 0.0, "ANS refusal on definitive gold."),
    ("doc87_qa0__after115", 0.0, "ANS refusal on definitive gold."),
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
