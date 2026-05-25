"""Claude manual judging — Phase 1.9 FB calibration v4t-corpus-tuned (entries 800-999).

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
    # 800-809
    ("doc106_qa0__after80", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc44_qa0__after80", 1.0, "ANS Yes Card Member retention high — match."),
    ("doc82_qa0__after80", 1.0, "ACK honest refusal — doc82 not yet ingested."),
    ("doc25_qa0__after80", 1.0, "ANS Amcor packaging — match."),
    ("doc60_qa0__after80", 1.0, "ANS Yes Commercial Airplanes 39%/$25,867M — match."),
    ("doc103_qa0__after80", 1.0, "ACK honest refusal — doc103 not yet ingested."),
    ("doc35_qa0__after80", 1.0, "ANS AMD operations $3,565M — match."),
    ("doc12_qa0__after80", 0.0, "ANS gold 0.83; predicted 1.25 — wrong specific."),
    ("doc141_qa0__after80", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    ("doc43_qa0__after80", 1.0, "ANS gold Customer deposits AMEX largest liability; predicted Customer deposits $110,239M — match exact."),
    # 810-819
    ("doc30_qa0__after81", 1.0, "ANS 4.18% vs 4.2% — within tolerance."),
    ("doc75_qa0__after81", 0.0, "ANS gold 17.98 fixed asset turnover CVS FY2018; predicted 8.73 — wrong specific."),
    ("doc79_qa0__after81", 1.0, "ANS gold Yes Mary Dillon former Ulta CEO; predicted Yes Mary N. Dillon former Executive Chair Ulta — match."),
    ("doc2_qa0__after81", 0.0, "ANS gold 'No efficient CAPEX'; predicted 'Yes capital-intensive $9,178M PP&E' — wrong direction."),
    ("doc138_qa0__after81", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc60_qa0__after81", 1.0, "ANS Commercial Airplanes — match."),
    ("doc23_qa0__after81", 0.0, "ANS gold quick ratio 0.67→0.69; predicted 'not explicitly provided' — refusal on definitive gold."),
    ("doc59_qa0__after81", 1.0, "ANS $12,645 — exact."),
    ("doc98_qa0__after81", 1.0, "ACK honest refusal — doc98 not yet ingested."),
    ("doc106_qa0__after81", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    # 820-829
    ("doc79_qa0__after82", 1.0, "ANS Mary Dillon — match."),
    ("doc12_qa0__after82", 0.0, "ANS gold 0.83; predicted 1.25 — wrong specific."),
    ("doc125_qa0__after82", 1.0, "ACK 'proposal not approved' = 'defeated' — correct by interpretation."),
    ("doc28_qa0__after82", 1.0, "ANS Adjusted EBITDA $2,018M — match."),
    ("doc35_qa0__after82", 1.0, "ANS cashflow $3,565M — match."),
    ("doc27_qa0__after82", 0.5, "ANS restructuring partial — no 87%."),
    ("doc43_qa0__after82", 0.0, "ANS gold Customer deposits AMEX largest liability; predicted 'Long-term debt $42,573M' — wrong."),
    ("doc101_qa0__after82", 1.0, "ACK honest refusal — doc101 not yet ingested."),
    ("doc71_qa0__after82", 0.0, "ANS gold 10.3%; predicted 15.5% — wrong specific."),
    ("doc144_qa0__after82", 1.0, "ACK honest refusal — doc144 not yet ingested."),
    # 830-839
    ("doc39_qa0__after83", 1.0, "ANS US/EMEA/APAC/LACC — match."),
    ("doc3_qa0__after83", 0.0, "ANS gold -1.7% reasons; predicted refusal — refusal on definitive gold."),
    ("doc54_qa0__after83", 0.25, "ANS Best Buy stores Q2 FY23→FY24 — wrong specific numbers (907/930 vs gold 982/969)."),
    ("doc42_qa0__after83", 1.0, "ANS AMEX tax 24.6%→21.6% — match."),
    ("doc144_qa0__after83", 1.0, "ACK honest refusal — doc144 not yet ingested."),
    ("doc126_qa0__after83", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    ("doc90_qa0__after83", 1.0, "ACK 'Consumer Health discontinued Aug 30, 2023' — correct by inference."),
    ("doc17_qa0__after83", 0.0, "ANS -0.02 vs -1.32 — wrong specific."),
    ("doc46_qa0__after83", 1.0, "ANS 1,832 — exact."),
    ("doc57_qa0__after83", 1.0, "ANS gold 101.5%; predicted 101.7% — within tolerance."),
    # 840-849
    ("doc148_qa0__after84", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc46_qa0__after84", 1.0, "ANS 1,832 — exact."),
    ("doc84_qa0__after84", 1.0, "ANS gold 0.54; predicted 0.54 — exact."),
    ("doc12_qa0__after84", 0.0, "ANS gold 0.83; predicted 1.25 — wrong specific."),
    ("doc77_qa0__after84", 0.75, "ANS gold Yes multiple legal battles + usual and customary pricing; predicted Yes lawsuits about usual and customary pricing — partial (subset of items)."),
    ("doc58_qa0__after84", 1.0, "ANS $381.6M ≈ $382 — within tolerance."),
    ("doc29_qa0__after84", 0.0, "ANS gold flat real growth; predicted 5% decrease — wrong direction."),
    ("doc124_qa0__after84", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc13_qa0__after84", 0.0, "ANS gold 'No declined 2.2%'; predicted 'improving operating margin profile, increased from $5.8B' — wrong direction."),
    ("doc8_qa0__after84", 0.0, "ANS gold 24.26; predicted 2.57 — wrong specific (factor of 10 off)."),
    # 850-859
    ("doc18_qa0__after85", 0.0, "ANS gold 93.86 DPO; predicted 29.12 — wrong specific."),
    ("doc131_qa0__after85", 1.0, "ACK honest refusal — doc131 not yet ingested."),
    ("doc67_qa0__after85", 0.0, "ANS gold 0.01 (1%); predicted 1.43% — outside 5% tolerance."),
    ("doc11_qa0__after85", 0.0, "ANS garbled — wrong."),
    ("doc118_qa0__after85", 1.0, "ACK honest refusal — doc118 not yet ingested."),
    ("doc48_qa0__after85", 0.0, "ANS gold 2.8%; predicted 3.9% — wrong specific."),
    ("doc139_qa0__after85", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc116_qa0__after85", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    ("doc135_qa0__after85", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc119_qa0__after85", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    # 860-869
    ("doc48_qa0__after86", 0.0, "ANS 3.9% vs 2.8% — wrong specific."),
    ("doc46_qa0__after86", 1.0, "ANS 1,832 — exact."),
    ("doc84_qa0__after86", 1.0, "ANS 0.54 — exact."),
    ("doc4_qa0__after86", 0.5, "ANS gold consumer shrunk 0.9%; predicted just 'Consumer segment' — partial."),
    ("doc40_qa0__after86", 1.0, "ANS operating margin not useful — match."),
    ("doc26_qa0__after86", 0.75, "ANS Amcor gross margin declining — direction right, no 0.8% figure."),
    ("doc109_qa0__after86", 1.0, "ACK honest refusal — doc109 not yet ingested."),
    ("doc116_qa0__after86", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    ("doc138_qa0__after86", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc76_qa0__after86", 1.0, "ANS gold Yes CVS capital-intensive 1.82%/3.39% ROA; predicted Yes capital-intensive based on significant PP&E — direction match."),
    # 870-879
    ("doc12_qa0__after87", 0.0, "ANS gold 0.83; predicted 1.25 — wrong specific."),
    ("doc138_qa0__after87", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc43_qa0__after87", 0.0, "ANS gold Customer deposits; predicted 'Long-term debt' — wrong."),
    ("doc108_qa0__after87", 1.0, "ACK honest refusal — doc108 not yet ingested."),
    ("doc59_qa0__after87", 1.0, "ANS $12,645 — exact."),
    ("doc4_qa0__after87", 0.5, "ANS gold consumer shrunk 0.9%; predicted 'Consumer segment dragged down growth in 2022' — partial (qualitative right, no figure)."),
    ("doc92_qa0__after87", 0.25, "ACK calibration: confident wrong $3.7B Kenvue cash (gold $13.2B)."),
    ("doc16_qa0__after87", 0.0, "ANS gold 9.5; predicted 11.97 — wrong specific."),
    ("doc91_qa0__after87", 1.0, "ACK honest refusal — doc91 not yet ingested."),
    ("doc124_qa0__after87", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    # 880-889
    ("doc22_qa0__after88", 1.0, "ANS Amcor 8K substitution — match."),
    ("doc27_qa0__after88", 0.5, "ANS restructuring partial — no 87%."),
    ("doc25_qa0__after88", 1.0, "ANS Amcor packaging — match."),
    ("doc149_qa0__after88", 1.0, "ACK honest refusal — doc149 not yet ingested."),
    ("doc146_qa0__after88", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc66_qa0__after88", 0.5, "ANS effective tax direction right, no specific rates."),
    ("doc60_qa0__after88", 1.0, "ANS Commercial Airplanes — match."),
    ("doc117_qa0__after88", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc21_qa0__after88", 1.0, "ANS $1,615.9M ≈ $1616 — within tolerance."),
    ("doc113_qa0__after88", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    # 890-899
    ("doc34_qa0__after89", 1.0, "ANS gold Xilinx amortization drove AMD operating income decrease; predicted same — match."),
    ("doc129_qa0__after89", 1.0, "ACK honest refusal — doc129 not yet ingested."),
    ("doc89_qa0__after89", 1.0, "ANS gold US 3.0% vs intl -0.6%; predicted same — exact match."),
    ("doc43_qa0__after89", 0.0, "ANS gold Customer deposits; predicted 'Long-term debt' — wrong."),
    ("doc101_qa0__after89", 1.0, "ACK honest refusal — doc101 not yet ingested."),
    ("doc75_qa0__after89", 0.0, "ANS gold 17.98; predicted 8.99 — wrong specific."),
    ("doc58_qa0__after89", 1.0, "ANS $381.6M ≈ $382 — within tolerance."),
    ("doc111_qa0__after89", 1.0, "ACK honest refusal — doc111 not yet ingested."),
    ("doc83_qa0__after89", 1.0, "ANS gold $3,215; predicted $3,215.4M — within tolerance."),
    ("doc2_qa0__after89", 0.0, "ANS gold 'No efficient'; predicted 'Yes capital-intensive $25,998M PP&E' — wrong direction."),
    # 900-909
    ("doc66_qa0__after90", 0.5, "ANS effective tax direction right, no specific rates."),
    ("doc113_qa0__after90", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc30_qa0__after90", 1.0, "ANS 4.18% vs 4.2% — within tolerance."),
    ("doc116_qa0__after90", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    ("doc41_qa0__after90", 1.0, "ANS gross margin not useful AMEX — match."),
    ("doc45_qa0__after90", 0.0, "ANS gold $0.40 (likely $0.40B = $400M); predicted $0.353B — 12% off, outside 5% tolerance."),
    ("doc5_qa0__after90", 0.0, "ANS gold No quick ratio 0.96; predicted 'not provided' — refusal on definitive gold."),
    ("doc91_qa0__after90", 1.0, "ACK calibration: '$20 billion JnJ Consumer Health gain' — correct by inference (matches gold)."),
    ("doc125_qa0__after90", 1.0, "ACK 'proposal not approved' = 'defeated' — correct."),
    ("doc126_qa0__after90", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    # 910-919
    ("doc96_qa0__after91", 1.0, "ACK 'gross margins not relevant for JPM' — correct by inference."),
    ("doc88_qa0__after91", 0.0, "ANS gold No (3.6%→3.5% decelerate); predicted 'Yes 12.5% increase' — wrong direction."),
    ("doc79_qa0__after91", 1.0, "ANS Foot Locker CEO Mary Dillon from Ulta — match with similar-company context."),
    ("doc33_qa0__after91", 1.0, "ANS AMD FY22 EPYC + Xilinx — match."),
    ("doc20_qa0__after91", 1.0, "ANS 11,588 — exact."),
    ("doc40_qa0__after91", 1.0, "ANS operating margin not useful — match."),
    ("doc86_qa0__after91", 1.0, "ANS gold COVID-19 vaccine + currency + commodity inflation; predicted same items — match."),
    ("doc15_qa0__after91", 1.0, "ANS 0 — exact."),
    ("doc99_qa0__after91", 1.0, "ACK honest refusal — doc99 not yet ingested."),
    ("doc18_qa0__after91", 0.0, "ANS gold 93.86; predicted 30.77 — wrong specific."),
    # 920-929
    ("doc101_qa0__after92", 1.0, "ACK honest refusal — doc101 not yet ingested."),
    ("doc45_qa0__after92", 0.0, "ANS $0.353B vs $0.40 — 12% off."),
    ("doc114_qa0__after92", 1.0, "ACK honest refusal — doc114 not yet ingested."),
    ("doc78_qa0__after92", 0.75, "ANS gold Yes $0.55/quarter; predicted Yes paid dividends Q2 FY22 — partial (Yes correct, no $0.55)."),
    ("doc91_qa0__after92", 1.0, "ANS $20 billion — exact match."),
    ("doc10_qa0__after92", 0.0, "ANS gold 0.66; predicted 1.29 — wrong specific."),
    ("doc12_qa0__after92", 0.0, "ANS gold 0.83; predicted 1.25 — wrong specific."),
    ("doc94_qa0__after92", 1.0, "ACK honest refusal — doc94 not yet ingested."),
    ("doc86_qa0__after92", 0.0, "ANS gold COVID-19 vaccine reasons; predicted refusal — refusal on definitive gold."),
    ("doc122_qa0__after92", 0.25, "ACK '0' confident wrong (gold $411M)."),
    # 930-939
    ("doc26_qa0__after93", 0.75, "ANS Amcor gross margin declining — direction right."),
    ("doc64_qa0__after93", 1.0, "ANS Yes Boeing cyclical — match."),
    ("doc146_qa0__after93", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc136_qa0__after93", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc54_qa0__after93", 1.0, "ANS gold 982→969 Best Buy; predicted same 982→969 — exact match."),
    ("doc106_qa0__after93", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc149_qa0__after93", 1.0, "ACK honest refusal — doc149 not yet ingested."),
    ("doc144_qa0__after93", 1.0, "ACK honest refusal — doc144 not yet ingested."),
    ("doc143_qa0__after93", 1.0, "ACK honest refusal — doc143 not yet ingested."),
    ("doc82_qa0__after93", 0.25, "ANS gold 0.68; predicted 0.73 — 7.4% off, outside 5% tolerance."),
    # 940-949
    ("doc18_qa0__after94", 0.0, "ANS gold 93.86; predicted 29.12 — wrong specific."),
    ("doc126_qa0__after94", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    ("doc52_qa0__after94", 1.0, "ANS Best Buy operating $1,824M — within tolerance."),
    ("doc9_qa0__after94", 0.0, "ANS gold 1.9%; predicted 3.5% — wrong specific."),
    ("doc64_qa0__after94", 1.0, "ANS Yes cyclical — match."),
    ("doc117_qa0__after94", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc129_qa0__after94", 1.0, "ACK honest refusal — doc129 not yet ingested."),
    ("doc83_qa0__after94", 1.0, "ANS gold $3,215; predicted $3,189.9M — within tolerance."),
    ("doc112_qa0__after94", 1.0, "ACK honest refusal — doc112 not yet ingested."),
    ("doc104_qa0__after94", 1.0, "ACK honest refusal — doc104 not yet ingested."),
    # 950-959
    ("doc18_qa0__after95", 0.0, "ANS gold 93.86; predicted 30.77 — wrong specific."),
    ("doc80_qa0__after95", 1.0, "ANS gold Yes Richard A. Johnson; predicted Yes Richard A. Johnson 16,105,005 votes — match."),
    ("doc52_qa0__after95", 1.0, "ANS Best Buy operating $1,824M — within tolerance."),
    ("doc100_qa0__after95", 1.0, "ACK honest refusal — doc100 not yet ingested."),
    ("doc106_qa0__after95", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc51_qa0__after95", 1.0, "ANS Best Buy acquisitions — match."),
    ("doc142_qa0__after95", 1.0, "ACK honest refusal — doc142 not yet ingested."),
    ("doc122_qa0__after95", 0.25, "ACK '0' confident wrong (gold $411M)."),
    ("doc8_qa0__after95", 0.0, "ANS gold 24.26; predicted 2.58 — wrong specific (factor of 10 off)."),
    ("doc17_qa0__after95", 0.0, "ANS -0.02 vs -1.32 — wrong specific."),
    # 960-969
    ("doc86_qa0__after96", 0.0, "ANS gold COVID-19 vaccine reasons; predicted 'Gross margin not useful metric' — wrong (says not useful when gold gives specific reasons)."),
    ("doc80_qa0__after96", 1.0, "ANS Richard A. Johnson — match."),
    ("doc94_qa0__after96", 0.0, "ANS gold Corporate -$473M; predicted 'Consumer & Community Banking' — wrong segment."),
    ("doc15_qa0__after96", 1.0, "ANS 0 — exact."),
    ("doc95_qa0__after96", 0.5, "ANS gold $66.56/share; predicted '$292.3B equity divided by shares' — partial (provides components for calc, doesn't compute)."),
    ("doc127_qa0__after96", 1.0, "ACK honest refusal — doc127 not yet ingested."),
    ("doc53_qa0__after96", 1.0, "ANS gold ~42% decline; predicted $1,874M→$1,093M — within tolerance."),
    ("doc52_qa0__after96", 1.0, "ANS Best Buy operating $1,824M — within tolerance."),
    ("doc50_qa0__after96", 0.0, "ANS gold consistent margins minor decline; predicted fluctuated >2% — wrong direction."),
    ("doc39_qa0__after96", 1.0, "ANS US/EMEA/APAC/LACC + Other — match (Other Unallocated valid)."),
    # 970-979
    ("doc133_qa0__after97", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc63_qa0__after97", 0.5, "ANS gold limited commercial airlines + US govt 40%; predicted limited commercial airlines (no US govt 40%) — partial."),
    ("doc118_qa0__after97", 1.0, "ACK honest refusal — doc118 not yet ingested."),
    ("doc8_qa0__after97", 0.0, "ANS gold 24.26; predicted 2.57 — wrong specific."),
    ("doc47_qa0__after97", 0.5, "ANS confused — Yes positive but describes -$1,561M."),
    ("doc125_qa0__after97", 1.0, "ACK 'proposal not approved' — correct."),
    ("doc95_qa0__after97", 0.25, "ANS gold $66.56/share; predicted just '$292.3B equity' (no per-share division) — confident partial information."),
    ("doc37_qa0__after97", 1.0, "ANS one customer 16% — match."),
    ("doc6_qa0__after97", 1.0, "ANS 3M notes 1.500% 2026, 1.750% 2030, 1.500% 2031 — match."),
    ("doc50_qa0__after97", 0.0, "ANS fluctuated >2% wrong direction."),
    # 980-989
    ("doc42_qa0__after98", 1.0, "ANS AMEX tax 24.6%→21.6% — match."),
    ("doc141_qa0__after98", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    ("doc80_qa0__after98", 1.0, "ANS Richard A. Johnson — match."),
    ("doc91_qa0__after98", 1.0, "ANS $20 billion JnJ Consumer Health gain — match."),
    ("doc60_qa0__after98", 1.0, "ANS Commercial Airplanes — match."),
    ("doc149_qa0__after98", 1.0, "ACK honest refusal — doc149 not yet ingested."),
    ("doc108_qa0__after98", 0.25, "ACK calibration: confident wrong 'International region 11.5% decline' (gold MGM China 44%)."),
    ("doc97_qa0__after98", 0.0, "ANS gold Corporate & Investment Bank; predicted 'Consumer & Community Banking' — wrong segment."),
    ("doc138_qa0__after98", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc16_qa0__after98", 0.0, "ANS gold 9.5; predicted 11.97 — wrong specific."),
    # 990-999
    ("doc113_qa0__after99", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc11_qa0__after99", 0.0, "ANS garbled — wrong."),
    ("doc40_qa0__after99", 1.0, "ANS operating margin not useful — match."),
    ("doc127_qa0__after99", 1.0, "ACK honest refusal — doc127 not yet ingested."),
    ("doc108_qa0__after99", 0.5, "ACK 'International' — vague, gold is MGM China specifically (subset of international)."),
    ("doc145_qa0__after99", 1.0, "ACK honest refusal — doc145 not yet ingested."),
    ("doc43_qa0__after99", 0.0, "ANS gold Customer deposits; predicted 'Long-term debt' — wrong."),
    ("doc71_qa0__after99", 1.0, "ANS gold 10.3%; predicted 10.5% — within tolerance."),
    ("doc124_qa0__after99", 0.25, "ACK calibration: confident wrong EBITDA calculation attempt (gold 16.5%)."),
    ("doc116_qa0__after99", 1.0, "ACK honest refusal — doc116 not yet ingested."),
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
