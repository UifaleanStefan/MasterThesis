"""Claude manual judging — Phase 1.9 FB calibration v4t-corpus-tuned (entries 1000-1199).

Idempotent append. All scores by Claude per HARD RULE.
"""

from __future__ import annotations

import json
from pathlib import Path

JUDGE_DIR = Path(
    "results/stage3/judge_queue/financebench__v4t-corpus-tuned__calibration__seed42"
)

QID_PREFIX = "financebench__v4t-corpus-tuned__calibration__seed42::"

JUDGMENTS: list[tuple[str, float, str]] = [
    # 1000-1009
    ("doc5_qa0__after100", 0.0, "ANS gold No quick ratio 0.96; predicted 'not provided' — refusal on definitive gold."),
    ("doc129_qa0__after100", 1.0, "ACK honest refusal — doc129 not yet ingested."),
    ("doc10_qa0__after100", 0.0, "ANS gold 0.66; predicted 1.24 — wrong specific."),
    ("doc90_qa0__after100", 1.0, "ANS Consumer Health discontinued Aug 30, 2023 — exact match."),
    ("doc148_qa0__after100", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc15_qa0__after100", 1.0, "ANS 0 — exact."),
    ("doc67_qa0__after100", 0.0, "ANS gold 0.01 (1%); predicted 1.43% — outside 5% tolerance."),
    ("doc127_qa0__after100", 1.0, "ACK honest refusal — doc127 not yet ingested."),
    ("doc65_qa0__after100", 1.0, "ANS Boeing 737/777X/787 production rates — match."),
    ("doc63_qa0__after100", 0.5, "ANS partial — commercial airlines without US govt 40%."),
    # 1010-1019
    ("doc81_qa0__after101", 0.0, "ANS gold -3.7 cash conversion cycle; predicted 66.73 days — wrong specific (different sign and magnitude)."),
    ("doc114_qa0__after101", 1.0, "ACK honest refusal — doc114 not yet ingested."),
    ("doc35_qa0__after101", 1.0, "ANS cashflow $3,565M — match."),
    ("doc41_qa0__after101", 1.0, "ANS gross margin not useful AMEX — match."),
    ("doc100_qa0__after101", 1.0, "ANS gold 1.33 asset turnover; predicted 1.30 — within 2.3% tolerance."),
    ("doc98_qa0__after101", 1.0, "ANS gold Yes decreased; predicted Yes decreased $7M — match with detail."),
    ("doc78_qa0__after101", 0.75, "ANS gold Yes $0.55/quarter; predicted Yes paid dividends Q2 — partial (Yes correct, no specific)."),
    ("doc75_qa0__after101", 0.0, "ANS gold 17.98; predicted 8.99 — wrong specific."),
    ("doc96_qa0__after101", 1.0, "ANS gross margins not relevant for JPM — match."),
    ("doc125_qa0__after101", 1.0, "ACK 'proposal not approved' — correct."),
    # 1020-1029
    ("doc31_qa0__after102", 0.0, "ANS gold Yes quick ratio 1.57 definitive; predicted 'cannot be determined' — refusal on definitive gold."),
    ("doc39_qa0__after102", 1.0, "ANS US/EMEA/APAC/LACC + Other — match."),
    ("doc24_qa0__after102", 0.75, "ANS Amcor acquisitions — mostly correct."),
    ("doc68_qa0__after102", 1.0, "ANS gold 39.7%; predicted 39.7% with calc — exact match."),
    ("doc119_qa0__after102", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    ("doc44_qa0__after102", 1.0, "ANS Yes Card Member retention — match."),
    ("doc36_qa0__after102", 1.0, "ANS Data Center — match."),
    ("doc59_qa0__after102", 1.0, "ANS $12,645 — exact."),
    ("doc46_qa0__after102", 1.0, "ANS 1,832 — exact."),
    ("doc108_qa0__after102", 0.25, "ACK 'International 11.5% decline' — confident wrong (gold MGM China 44%)."),
    # 1030-1039
    ("doc108_qa0__after103", 1.0, "ACK honest refusal — doc108 not yet ingested."),
    ("doc61_qa0__after103", 1.0, "ANS Lion Air + Ethiopian crashes detailed — match."),
    ("doc135_qa0__after103", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc60_qa0__after103", 1.0, "ANS Commercial Airplanes — match."),
    ("doc36_qa0__after103", 1.0, "ANS Data Center — match."),
    ("doc51_qa0__after103", 1.0, "ANS Best Buy acquisitions — match."),
    ("doc85_qa0__after103", 1.0, "ANS gold No JnJ 1.3% growth; predicted same — exact match."),
    ("doc105_qa0__after103", 1.0, "ACK honest refusal — doc105 not yet ingested."),
    ("doc71_qa0__after103", 1.0, "ANS gold 10.3%; predicted 10.5% — within tolerance."),
    ("doc137_qa0__after103", 1.0, "ACK honest refusal — doc137 not yet ingested."),
    # 1040-1049
    ("doc46_qa0__after104", 1.0, "ANS 1,832 — exact."),
    ("doc136_qa0__after104", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc121_qa0__after104", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc96_qa0__after104", 1.0, "ANS JPM gross margins not relevant — match."),
    ("doc16_qa0__after104", 0.0, "ANS gold 9.5; predicted 12.00 — wrong specific."),
    ("doc80_qa0__after104", 1.0, "ANS Richard A. Johnson — match."),
    ("doc31_qa0__after104", 0.0, "ANS quick ratio refusal on definitive gold."),
    ("doc14_qa0__after104", 0.0, "ANS Adobe FCF refusal on definitive gold."),
    ("doc101_qa0__after104", 1.0, "ANS $5,818M — exact."),
    ("doc103_qa0__after104", 1.0, "ANS gold $303; predicted $302.6M — within tolerance."),
    # 1050-1059
    ("doc22_qa0__after105", 1.0, "ANS Amcor 8K — match."),
    ("doc119_qa0__after105", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    ("doc25_qa0__after105", 1.0, "ANS Amcor packaging — match."),
    ("doc146_qa0__after105", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc62_qa0__after105", 0.0, "ANS gold Yes Boeing improving gross margin; predicted 'not useful metric' — wrong."),
    ("doc98_qa0__after105", 1.0, "ANS Yes decreased $7M — match."),
    ("doc1_qa0__after105", 1.0, "ANS $8.70 vs 8.738B — within 0.5% tolerance."),
    ("doc138_qa0__after105", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc123_qa0__after105", 0.75, "ACK calibration: hedged with uncertainty — model sets up EBITDA framework but admits missing values."),
    ("doc76_qa0__after105", 1.0, "ANS Yes CVS capital-intensive — direction match."),
    # 1060-1069
    ("doc124_qa0__after106", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc70_qa0__after106", 1.0, "ANS gold 63.86 DPO Corning FY2020; predicted 66.67 days — within 5% tolerance (4.4% off)."),
    ("doc28_qa0__after106", 1.0, "ANS $2,018M — exact."),
    ("doc30_qa0__after106", 1.0, "ANS 4.18% — within tolerance."),
    ("doc85_qa0__after106", 1.0, "ANS No 1.3% — exact."),
    ("doc130_qa0__after106", 1.0, "ACK honest refusal — doc130 not yet ingested."),
    ("doc87_qa0__after106", 0.0, "ANS gold 2.7 turnover JnJ FY2022 definitive; predicted 'not provided' — refusal on definitive gold."),
    ("doc135_qa0__after106", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc148_qa0__after106", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc49_qa0__after106", 1.0, "ANS gold $5,409; predicted 5,409 — exact."),
    # 1070-1079
    ("doc9_qa0__after107", 0.0, "ANS gold 1.9%; predicted 3.5% — wrong specific."),
    ("doc134_qa0__after107", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc13_qa0__after107", 0.0, "ANS gold 'No declined 2.2%'; predicted 'Yes improving' — wrong direction."),
    ("doc142_qa0__after107", 1.0, "ACK honest refusal — doc142 not yet ingested."),
    ("doc127_qa0__after107", 1.0, "ACK honest refusal — doc127 not yet ingested."),
    ("doc122_qa0__after107", 0.25, "ACK '0' confident wrong (gold $411M)."),
    ("doc133_qa0__after107", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc103_qa0__after107", 1.0, "ANS gold $303; predicted $302.578M — within tolerance."),
    ("doc139_qa0__after107", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc87_qa0__after107", 0.0, "ANS refusal on definitive gold."),
    # 1080-1089
    ("doc75_qa0__after108", 0.0, "ANS gold 17.98; predicted 8.99 — wrong specific."),
    ("doc90_qa0__after108", 1.0, "ANS Consumer Health discontinued — exact."),
    ("doc98_qa0__after108", 1.0, "ANS Yes decreased $7M — match."),
    ("doc140_qa0__after108", 1.0, "ACK honest refusal — doc140 not yet ingested."),
    ("doc42_qa0__after108", 1.0, "ANS AMEX tax 24.6%→21.6% — match."),
    ("doc43_qa0__after108", 0.0, "ANS Long-term debt wrong (gold Customer deposits)."),
    ("doc51_qa0__after108", 1.0, "ANS Best Buy acquisitions — match."),
    ("doc68_qa0__after108", 1.0, "ANS 39.7% with calc — match."),
    ("doc45_qa0__after108", 1.0, "ANS gold $0.40 (≈$0.40B = $400M); predicted $0.389B = $389M — 2.75% off, within 5% tolerance."),
    ("doc108_qa0__after108", 0.75, "ANS gold MGM China worst 44% decline; predicted 'MGM China worst' — gets region right, no 44%."),
    # 1090-1099
    ("doc26_qa0__after109", 0.75, "ANS Amcor gross margin declining — direction right."),
    ("doc7_qa0__after109", 1.0, "ANS Yes 65th consecutive — match."),
    ("doc119_qa0__after109", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    ("doc14_qa0__after109", 0.0, "ANS refusal on definitive gold."),
    ("doc44_qa0__after109", 1.0, "ANS Yes — match."),
    ("doc102_qa0__after109", 1.0, "ANS gold 0.4% CAGR Lockheed FY20→FY22; predicted 0.4% — exact."),
    ("doc65_qa0__after109", 1.0, "ANS Boeing production rates — match."),
    ("doc133_qa0__after109", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc18_qa0__after109", 0.0, "ANS gold 93.86; predicted 30.77 — wrong specific."),
    ("doc134_qa0__after109", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    # 1100-1109
    ("doc120_qa0__after110", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc7_qa0__after110", 1.0, "ANS Yes 65th — match."),
    ("doc72_qa0__after110", 1.0, "ANS gold 20%→23% Corning; predicted 20%→23% — exact."),
    ("doc35_qa0__after110", 1.0, "ANS cashflow $3,565M — match."),
    ("doc99_qa0__after110", 1.0, "ANS gold 6.25; predicted 6.19 — within 1% tolerance."),
    ("doc33_qa0__after110", 1.0, "ANS AMD FY22 EPYC etc. — match."),
    ("doc145_qa0__after110", 1.0, "ACK honest refusal — doc145 not yet ingested."),
    ("doc90_qa0__after110", 1.0, "ANS Consumer Health discontinued — exact."),
    ("doc97_qa0__after110", 0.0, "ANS gold Corporate & Investment Bank; predicted 'Consumer & Community Banking' — wrong segment."),
    ("doc122_qa0__after110", 0.25, "ACK '0' confident wrong."),
    # 1110-1119
    ("doc117_qa0__after111", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc120_qa0__after111", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc26_qa0__after111", 0.75, "ANS Amcor declining — direction right."),
    ("doc113_qa0__after111", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc30_qa0__after111", 1.0, "ANS 4.18% — within tolerance."),
    ("doc82_qa0__after111", 1.0, "ANS gold 0.68; predicted 0.69 — within 1.5% tolerance."),
    ("doc36_qa0__after111", 1.0, "ANS Data Center — match."),
    ("doc72_qa0__after111", 1.0, "ANS 20%→23% — match."),
    ("doc37_qa0__after111", 1.0, "ANS 16% one customer — match."),
    ("doc101_qa0__after111", 1.0, "ANS $5,818M — exact."),
    # 1120-1129
    ("doc35_qa0__after112", 1.0, "ANS cashflow $3,565M — match."),
    ("doc52_qa0__after112", 1.0, "ANS Best Buy $1,824M — within tolerance."),
    ("doc23_qa0__after112", 0.0, "ANS refusal on definitive gold (quick ratio)."),
    ("doc120_qa0__after112", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc21_qa0__after112", 1.0, "ANS $1,615.9M — within tolerance."),
    ("doc59_qa0__after112", 1.0, "ANS $12,645 — exact."),
    ("doc114_qa0__after112", 1.0, "ACK honest refusal — doc114 not yet ingested."),
    ("doc92_qa0__after112", 1.0, "ANS gold JnJ Kenvue $13.2B; predicted $13.2B — exact."),
    ("doc89_qa0__after112", 1.0, "ANS JnJ US 3.0% intl -0.6% — match."),
    ("doc122_qa0__after112", 0.25, "ACK '0' confident wrong."),
    # 1130-1139
    ("doc139_qa0__after113", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc104_qa0__after113", 0.0, "ANS gold 7.9% MGM capex/revenue; predicted -3.5% — wrong specific."),
    ("doc136_qa0__after113", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc82_qa0__after113", 1.0, "ANS 0.68 vs 0.69 — within tolerance."),
    ("doc60_qa0__after113", 1.0, "ANS Commercial Airplanes — match."),
    ("doc89_qa0__after113", 1.0, "ANS US 3.0% intl -0.6% — match."),
    ("doc47_qa0__after113", 0.5, "ANS confused — Yes positive but describes -$1,561M and acknowledges negative."),
    ("doc137_qa0__after113", 1.0, "ACK honest refusal — doc137 not yet ingested."),
    ("doc56_qa0__after113", 1.0, "ANS 1.74 vs 1.73 — within tolerance."),
    ("doc98_qa0__after113", 1.0, "ANS Yes decreased $7M — match."),
    # 1140-1149
    ("doc24_qa0__after114", 0.0, "ANS gold Amcor acquisitions definitive; predicted refusal — refusal on definitive gold."),
    ("doc113_qa0__after114", 1.0, "ANS gold $5,466; predicted 5,466.3M — within tolerance."),
    ("doc27_qa0__after114", 0.5, "ANS restructuring partial — no 87%."),
    ("doc124_qa0__after114", 0.25, "ACK calibration: confident truncated EBITDA calculation attempt (gold 16.5%)."),
    ("doc97_qa0__after114", 0.0, "ANS Consumer & Community Banking wrong."),
    ("doc99_qa0__after114", 1.0, "ANS 6.20 vs 6.25 — within tolerance."),
    ("doc131_qa0__after114", 1.0, "ACK honest refusal — doc131 not yet ingested."),
    ("doc19_qa0__after114", 1.0, "ANS 30.7% vs 30.8% — within tolerance."),
    ("doc98_qa0__after114", 1.0, "ANS Yes decreased $7M — match."),
    ("doc12_qa0__after114", 0.0, "ANS 0.83 vs 1.25 — wrong specific."),
    # 1150-1159
    ("doc80_qa0__after115", 1.0, "ANS Richard A. Johnson — match."),
    ("doc23_qa0__after115", 0.0, "ANS refusal on definitive gold."),
    ("doc111_qa0__after115", 0.25, "ANS gold 'No Microsoft -$2.5bn'; predicted 'Yes increased' with detailed numbers showing net -$2.5bn — Yes/No flip but underlying numbers support gold direction."),
    ("doc131_qa0__after115", 1.0, "ACK honest refusal — doc131 not yet ingested."),
    ("doc33_qa0__after115", 1.0, "ANS AMD FY22 — match."),
    ("doc87_qa0__after115", 0.0, "ANS refusal on definitive gold."),
    ("doc140_qa0__after115", 1.0, "ACK honest refusal — doc140 not yet ingested."),
    ("doc81_qa0__after115", 0.0, "ANS gold -3.7; predicted 66.73 days — wrong specific."),
    ("doc121_qa0__after115", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc68_qa0__after115", 1.0, "ANS 39.7% — exact."),
    # 1160-1169
    ("doc4_qa0__after116", 0.5, "ANS partial — Consumer segment without 0.9% figure."),
    ("doc66_qa0__after116", 0.5, "ANS effective tax direction right, no specific rates."),
    ("doc120_qa0__after116", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc138_qa0__after116", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc88_qa0__after116", 0.0, "ANS gold No (decelerate); predicted 'Yes 12.5% increase' — wrong direction."),
    ("doc93_qa0__after116", 1.0, "ANS gold 20%→20.1%; predicted 20.0%→20.1% — exact."),
    ("doc105_qa0__after116", 1.0, "ANS gold Yes $0.01/share MGM; predicted Yes $0.01/share — exact."),
    ("doc44_qa0__after116", 1.0, "ANS Yes — match."),
    ("doc104_qa0__after116", 0.0, "ANS -3.5% vs 7.9% — wrong specific."),
    ("doc21_qa0__after116", 1.0, "ANS $1,615.9M — within tolerance."),
    # 1170-1179
    ("doc146_qa0__after117", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc131_qa0__after117", 1.0, "ACK honest refusal — doc131 not yet ingested."),
    ("doc6_qa0__after117", 1.0, "ANS 3M notes — match."),
    ("doc44_qa0__after117", 1.0, "ANS Yes — match."),
    ("doc42_qa0__after117", 1.0, "ANS AMEX tax — match."),
    ("doc54_qa0__after117", 1.0, "ANS Best Buy 982→969 — exact."),
    ("doc2_qa0__after117", 0.0, "ANS 'Yes capital-intensive' wrong direction."),
    ("doc148_qa0__after117", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc121_qa0__after117", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc3_qa0__after117", 0.5, "ANS partial — mentions same items (special items, litigation, impairment, restructuring) but no -1.7%."),
    # 1180-1189
    ("doc107_qa0__after118", 0.0, "ANS gold zero (negative EBIT MGM); predicted 1.61 — wrong specific."),
    ("doc93_qa0__after118", 1.0, "ANS 20.0%→20.1% — exact."),
    ("doc4_qa0__after118", 0.5, "ANS partial."),
    ("doc133_qa0__after118", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc22_qa0__after118", 1.0, "ANS Amcor 8K — match."),
    ("doc37_qa0__after118", 1.0, "ANS 16% — match."),
    ("doc73_qa0__after118", 0.5, "ANS gold Yes Corning positive WC $831M (operating only); predicted Yes positive $2,278M (total) — direction right, specific number differs by methodology."),
    ("doc45_qa0__after118", 1.0, "ANS $0.389B vs $0.40 — within tolerance."),
    ("doc41_qa0__after118", 1.0, "ANS gross margin not useful AMEX — match."),
    ("doc34_qa0__after118", 1.0, "ANS Xilinx amortization AMD — match."),
    # 1190-1199
    ("doc15_qa0__after119", 1.0, "ANS 0 — exact."),
    ("doc142_qa0__after119", 1.0, "ACK honest refusal — doc142 not yet ingested."),
    ("doc45_qa0__after119", 1.0, "ANS $0.389B vs $0.40 — within tolerance."),
    ("doc49_qa0__after119", 1.0, "ANS 5,409 — exact."),
    ("doc68_qa0__after119", 1.0, "ANS 39.7% — exact."),
    ("doc48_qa0__after119", 0.0, "ANS 3.9% vs 2.8% — wrong specific."),
    ("doc25_qa0__after119", 1.0, "ANS Amcor packaging — match."),
    ("doc146_qa0__after119", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc59_qa0__after119", 1.0, "ANS $12,645 — exact."),
    ("doc52_qa0__after119", 1.0, "ANS Best Buy $1,824M — within tolerance."),
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
        qid = QID_PREFIX + suffix
        if qid in existing:
            continue
        new_records.append(
            {
                "qid": qid,
                "judge_score": score,
                "rationale": rationale,
                "judge_model": "claude-opus-4.7-1m",
                "judge_protocol": "v1",
            }
        )

    with results_path.open("a", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_after = len(existing) + len(new_records)
    print(
        f"Appended {len(new_records)} (skipped {len(JUDGMENTS) - len(new_records)}, "
        f"total {total_after})"
    )
    if new_records:
        from collections import Counter
        dist = Counter(r["judge_score"] for r in new_records)
        print(f"Score distribution: {dict(sorted(dist.items()))}")
        mean = sum(r["judge_score"] for r in new_records) / len(new_records)
        print(f"Mean judge: {mean:.4f}")
    print(f"Cell progress: {total_after}/1500 (={100*total_after/1500:.1f}%)")


if __name__ == "__main__":
    main()
