"""Claude manual judging — Phase 1.9 FB calibration v4t-corpus-tuned (entries 600-799).

Idempotent append to results.jsonl per evaluation/claude_judge_protocol.md.
All scores assigned by Claude in-session per HARD RULE in AGENTS.md §0.
"""

from __future__ import annotations

import json
from pathlib import Path

JUDGE_DIR = Path(
    "results/stage3/judge_queue/financebench__v4t-corpus-tuned__calibration__seed42"
)

QID_PREFIX = "financebench__v4t-corpus-tuned__calibration__seed42::"
QID_SUFFIX = ""

JUDGMENTS: list[tuple[str, float, str]] = [
    # 600-609
    ("doc13_qa0__after60", 0.0, "ANS gold 'No, operating margins declined 36.8%→34.6% (-2.2%)'; predicted 'Yes improving' — wrong direction (Yes vs No)."),
    ("doc59_qa0__after60", 1.0, "ANS gold $12,645; predicted $12,645 — exact match."),
    ("doc47_qa0__after60", 0.5, "ANS gold 'No, negative working capital -$1,561M'; predicted 'Yes positive' but then describes -$1,561M and acknowledges negative — confused, ends correct on direction."),
    ("doc67_qa0__after60", 1.0, "ACK honest refusal — doc67 not yet ingested."),
    ("doc130_qa0__after60", 1.0, "ACK honest refusal — doc130 not yet ingested."),
    ("doc18_qa0__after60", 0.0, "ANS gold 93.86 DPO Amazon; predicted 30.12 — wrong specific."),
    ("doc133_qa0__after60", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc7_qa0__after60", 1.0, "ANS gold Yes 3M increased dividend 65 years; predicted Yes 65th consecutive — match."),
    ("doc137_qa0__after60", 1.0, "ACK honest refusal — doc137 not yet ingested."),
    ("doc134_qa0__after60", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    # 610-619
    ("doc50_qa0__after61", 0.0, "ANS gold consistent margins minor decline 1.1%; predicted 'fluctuated >2%' — wrong direction."),
    ("doc20_qa0__after61", 1.0, "ANS gold $11,588; predicted $11,588 — exact match."),
    ("doc96_qa0__after61", 1.0, "ACK calibration: 'gross margins not relevant for JPM financial services' — correct by inference (matches gold)."),
    ("doc69_qa0__after61", 1.0, "ACK honest refusal — doc69 not yet ingested."),
    ("doc12_qa0__after61", 0.0, "ANS gold 0.83 operating cash flow ratio FY2017 Adobe; predicted 1.23 — wrong specific."),
    ("doc54_qa0__after61", 0.25, "ANS gold Yes decline 1.32% (982→969 stores); predicted 'Yes change, 907 down from 930' — wrong specific numbers, direction right."),
    ("doc126_qa0__after61", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    ("doc106_qa0__after61", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc142_qa0__after61", 1.0, "ACK honest refusal — doc142 not yet ingested."),
    ("doc75_qa0__after61", 1.0, "ACK honest refusal — doc75 not yet ingested."),
    # 620-629
    ("doc47_qa0__after62", 0.5, "ANS confused — Yes positive but describes -$1,561M — same pattern as 602."),
    ("doc40_qa0__after62", 1.0, "ANS gold 'not measured through operating margin'; predicted equivalent — match."),
    ("doc101_qa0__after62", 1.0, "ACK honest refusal — doc101 not yet ingested."),
    ("doc140_qa0__after62", 1.0, "ACK honest refusal — doc140 not yet ingested."),
    ("doc87_qa0__after62", 1.0, "ACK honest refusal — doc87 not yet ingested."),
    ("doc121_qa0__after62", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc83_qa0__after62", 1.0, "ACK honest refusal — doc83 not yet ingested."),
    ("doc72_qa0__after62", 0.25, "ACK calibration: confident wrong specific tax rates (15.6%→24.8% vs gold 20%→23%); direction correct but numbers wrong."),
    ("doc147_qa0__after62", 1.0, "ACK honest refusal — doc147 not yet ingested."),
    ("doc126_qa0__after62", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    # 630-639
    ("doc126_qa0__after63", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    ("doc64_qa0__after63", 1.0, "ACK calibration: 'Yes Boeing cyclical due to commercial airlines' — correct by inference."),
    ("doc115_qa0__after63", 1.0, "ACK honest refusal — doc115 not yet ingested."),
    ("doc77_qa0__after63", 1.0, "ACK honest refusal — doc77 not yet ingested."),
    ("doc143_qa0__after63", 1.0, "ACK honest refusal — doc143 not yet ingested."),
    ("doc123_qa0__after63", 1.0, "ACK honest refusal — doc123 not yet ingested."),
    ("doc61_qa0__after63", 1.0, "ANS gold Lion Air + Ethiopian crashes; predicted Lion Air Flight 610 + Ethiopian Flight 302 — match with detail."),
    ("doc33_qa0__after63", 1.0, "ANS gold AMD FY22 EPYC + semi-custom + Xilinx; predicted 64% Data Center EPYC, 21% Gaming semi-custom, Xilinx — match."),
    ("doc39_qa0__after63", 1.0, "ANS gold US/EMEA/APAC/LACC; predicted same — match."),
    ("doc25_qa0__after63", 1.0, "ANS gold Amcor packaging; predicted Amcor packaging various uses — match."),
    # 640-649
    ("doc132_qa0__after64", 1.0, "ACK honest refusal — doc132 not yet ingested."),
    ("doc60_qa0__after64", 1.0, "ANS gold Yes Commercial Airplanes 39% revenue; predicted Yes Commercial Airplanes $25,867M (≈39% of Boeing FY22) — match."),
    ("doc134_qa0__after64", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc107_qa0__after64", 1.0, "ACK honest refusal — doc107 not yet ingested."),
    ("doc68_qa0__after64", 1.0, "ACK honest refusal — doc68 not yet ingested."),
    ("doc24_qa0__after64", 0.75, "ANS Amcor acquisitions — same as earlier, mostly correct."),
    ("doc36_qa0__after64", 0.0, "ANS gold Data Center; predicted 'Gaming segment' — wrong."),
    ("doc117_qa0__after64", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc27_qa0__after64", 0.5, "ANS partial — restructuring breakdown without 87% figure."),
    ("doc41_qa0__after64", 1.0, "ANS gross margin not useful for AMEX — match."),
    # 650-659
    ("doc105_qa0__after65", 1.0, "ACK honest refusal — doc105 not yet ingested."),
    ("doc146_qa0__after65", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc26_qa0__after65", 0.75, "ANS gold 'No slight decline 0.8%'; predicted 'gross profit $2,820→$2,725 declining' — direction right (declining/no improvement), no 0.8% figure."),
    ("doc18_qa0__after65", 0.0, "ANS gold 93.86; predicted 30.12 — wrong specific."),
    ("doc89_qa0__after65", 1.0, "ACK honest refusal — doc89 not yet ingested."),
    ("doc114_qa0__after65", 1.0, "ACK honest refusal — doc114 not yet ingested."),
    ("doc102_qa0__after65", 1.0, "ACK honest refusal — doc102 not yet ingested."),
    ("doc38_qa0__after65", 0.0, "ANS gold 'There are none' (no debt securities); predicted 'Common Shares par value $0.20' — wrong specific."),
    ("doc94_qa0__after65", 1.0, "ACK honest refusal — doc94 not yet ingested."),
    ("doc145_qa0__after65", 1.0, "ACK honest refusal — doc145 not yet ingested."),
    # 660-669
    ("doc55_qa0__after66", 1.0, "ANS gold entertainment 9% gaming; predicted same — match."),
    ("doc51_qa0__after66", 1.0, "ANS gold Best Buy two acquisitions (Current Health + Yardbird) FY22; predicted Current Health $389M + Yardbird $79M FY22, no FY23/21 — match."),
    ("doc62_qa0__after66", 0.0, "ANS gold Yes Boeing improving gross margin; predicted 'gross margin not useful metric' — wrong (says not useful when gold says Yes improving)."),
    ("doc139_qa0__after66", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc142_qa0__after66", 1.0, "ACK honest refusal — doc142 not yet ingested."),
    ("doc149_qa0__after66", 1.0, "ACK honest refusal — doc149 not yet ingested."),
    ("doc116_qa0__after66", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    ("doc103_qa0__after66", 1.0, "ACK honest refusal — doc103 not yet ingested."),
    ("doc66_qa0__after66", 0.5, "ANS gold 0.62% vs -14.76% effective tax; predicted lower tax in FY22 with $(31)M expense vs $743M benefit — direction right, no specific rates."),
    ("doc17_qa0__after66", 0.0, "ANS gold -0.02; predicted -1.32 — wrong specific."),
    # 670-679
    ("doc74_qa0__after67", 0.25, "ACK calibration: confident wrong specific ($52,694M vs gold $59,268, ~11% off)."),
    ("doc76_qa0__after67", 1.0, "ACK honest refusal — doc76 not yet ingested."),
    ("doc25_qa0__after67", 1.0, "ANS gold Amcor packaging; predicted same — match."),
    ("doc71_qa0__after67", 1.0, "ACK honest refusal — doc71 not yet ingested."),
    ("doc113_qa0__after67", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc2_qa0__after67", 0.0, "ANS gold 'No, efficient CAPEX 5.1%'; predicted 'Yes capital-intensive based on $25,998M PP&E' — wrong direction."),
    ("doc3_qa0__after67", 0.5, "ANS gold -1.7% due to gross margin + one-off charges; predicted 'increased special items, litigation, impairment, restructuring' — partial (mentions items, no -1.7%)."),
    ("doc141_qa0__after67", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    ("doc35_qa0__after67", 1.0, "ANS cashflow from operations $3,565M — match."),
    ("doc39_qa0__after67", 1.0, "ANS US/EMEA/APAC/LACC — match."),
    # 680-689
    ("doc66_qa0__after68", 0.5, "ANS effective tax direction right, no specific rates — same as 668."),
    ("doc25_qa0__after68", 1.0, "ANS Amcor packaging — match."),
    ("doc99_qa0__after68", 1.0, "ACK honest refusal — doc99 not yet ingested."),
    ("doc85_qa0__after68", 1.0, "ACK honest refusal — doc85 not yet ingested."),
    ("doc24_qa0__after68", 0.75, "ANS Amcor acquisitions — mostly correct."),
    ("doc126_qa0__after68", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    ("doc32_qa0__after68", 1.0, "ANS AMD products — match."),
    ("doc15_qa0__after68", 1.0, "ANS 0 — exact."),
    ("doc82_qa0__after68", 0.25, "ACK calibration: confident wrong working capital ratio 1.14 (gold 0.68)."),
    ("doc121_qa0__after68", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    # 690-699
    ("doc105_qa0__after69", 1.0, "ACK honest refusal — doc105 not yet ingested."),
    ("doc85_qa0__after69", 1.0, "ACK honest refusal — doc85 not yet ingested."),
    ("doc139_qa0__after69", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc30_qa0__after69", 1.0, "ANS 4.18% vs 4.2% — within tolerance."),
    ("doc108_qa0__after69", 1.0, "ACK honest refusal — doc108 not yet ingested."),
    ("doc32_qa0__after69", 1.0, "ANS AMD products — match."),
    ("doc87_qa0__after69", 1.0, "ACK honest refusal — doc87 not yet ingested."),
    ("doc93_qa0__after69", 1.0, "ACK honest refusal — doc93 not yet ingested."),
    ("doc65_qa0__after69", 1.0, "ANS Boeing 737/777X/787 production rate increases — match."),
    ("doc16_qa0__after69", 0.0, "ANS gold 9.5 turnover; predicted calculation giving ~12 — wrong specific."),
    # 700-709
    ("doc26_qa0__after70", 0.75, "ANS Amcor gross margin declining — direction right, no 0.8% figure."),
    ("doc66_qa0__after70", 0.5, "ANS effective tax direction right, no specific rates."),
    ("doc93_qa0__after70", 1.0, "ACK honest refusal — doc93 not yet ingested."),
    ("doc138_qa0__after70", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc129_qa0__after70", 1.0, "ACK honest refusal — doc129 not yet ingested."),
    ("doc71_qa0__after70", 0.25, "ACK calibration: confident wrong 4.5% (gold 10.3%)."),
    ("doc135_qa0__after70", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc65_qa0__after70", 1.0, "ANS Boeing production rates — match."),
    ("doc104_qa0__after70", 1.0, "ACK honest refusal — doc104 not yet ingested."),
    ("doc91_qa0__after70", 1.0, "ACK honest refusal — doc91 not yet ingested."),
    # 710-719
    ("doc10_qa0__after71", 0.0, "ANS gold 0.66; predicted 1.24 — wrong specific."),
    ("doc46_qa0__after71", 1.0, "ANS gold $1,832; predicted 1,832 — exact."),
    ("doc59_qa0__after71", 1.0, "ANS $12,645 — exact."),
    ("doc95_qa0__after71", 1.0, "ACK honest refusal — doc95 not yet ingested."),
    ("doc55_qa0__after71", 1.0, "ANS entertainment 9% — match."),
    ("doc139_qa0__after71", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc42_qa0__after71", 1.0, "ANS AMEX tax rate 24.6%→21.6% — exact match."),
    ("doc94_qa0__after71", 1.0, "ACK honest refusal — doc94 not yet ingested."),
    ("doc58_qa0__after71", 1.0, "ANS gold $382; predicted $381.6M — within tolerance."),
    ("doc14_qa0__after71", 0.0, "ANS gold Yes Adobe FCF improved ~13%; predicted refusal — refusal on definitive gold."),
    # 720-729
    ("doc3_qa0__after72", 0.0, "ANS gold -1.7% operating margin due to gross margin + one-off; predicted 'do not provide specific information' — refusal on definitive gold."),
    ("doc110_qa0__after72", 1.0, "ACK honest refusal — doc110 not yet ingested."),
    ("doc134_qa0__after72", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc12_qa0__after72", 0.0, "ANS gold 0.83 OCF ratio; predicted 1.23 — wrong specific."),
    ("doc71_qa0__after72", 0.0, "ANS gold 10.3%; predicted 15.5% — wrong specific."),
    ("doc52_qa0__after72", 1.0, "ANS gold Best Buy operating $1.8bn; predicted operating $1,824M — within tolerance."),
    ("doc64_qa0__after72", 1.0, "ANS Yes Boeing cyclical — match."),
    ("doc26_qa0__after72", 0.75, "ANS Amcor gross margin declining — direction right, no 0.8% figure."),
    ("doc117_qa0__after72", 1.0, "ACK calibration: 'operating activities most cash flow Nike FY23' — correct by inference (matches gold)."),
    ("doc119_qa0__after72", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    # 730-739
    ("doc14_qa0__after73", 0.0, "ANS Adobe FCF improved ~13% gold; predicted refusal — refusal on definitive gold."),
    ("doc106_qa0__after73", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc12_qa0__after73", 0.0, "ANS gold 0.83; predicted 1.25 — wrong specific."),
    ("doc114_qa0__after73", 1.0, "ACK honest refusal — doc114 not yet ingested."),
    ("doc92_qa0__after73", 0.25, "ACK calibration: confident wrong $3.5B Kenvue cash (gold $13.2B)."),
    ("doc140_qa0__after73", 1.0, "ACK honest refusal — doc140 not yet ingested."),
    ("doc115_qa0__after73", 1.0, "ACK honest refusal — doc115 not yet ingested."),
    ("doc69_qa0__after73", 1.0, "ANS gold 0.8; predicted 0.80 — exact."),
    ("doc4_qa0__after73", 0.5, "ANS gold consumer shrunk 0.9%; predicted just 'Consumer segment' — partial."),
    ("doc26_qa0__after73", 0.75, "ANS Amcor gross margin declining — direction right."),
    # 740-749
    ("doc119_qa0__after74", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    ("doc117_qa0__after74", 1.0, "ACK calibration: 'operating most cash Nike FY23' — correct by inference."),
    ("doc69_qa0__after74", 1.0, "ANS 0.80 — exact."),
    ("doc123_qa0__after74", 1.0, "ACK honest refusal — doc123 not yet ingested."),
    ("doc90_qa0__after74", 1.0, "ACK calibration: 'Consumer Health discontinued from Aug 30, 2023' — correct by inference (matches gold exactly; JnJ docs ingested)."),
    ("doc83_qa0__after74", 0.25, "ACK calibration: confident wrong $1,000M FCF (gold $3,215M General Mills FY2020)."),
    ("doc126_qa0__after74", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    ("doc50_qa0__after74", 0.0, "ANS gold consistent margins minor decline; predicted fluctuated >2% — wrong direction."),
    ("doc22_qa0__after74", 1.0, "ANS Amcor 8K substitution — match."),
    ("doc6_qa0__after74", 1.0, "ANS gold 3M notes (1.500% 2026, 1.750% 2030, 1.500% 2031); predicted same — match."),
    # 750-759
    ("doc37_qa0__after75", 1.0, "ANS Yes one customer 16% — match."),
    ("doc0_qa0__after75", 1.0, "ANS gold $1,577; predicted $1,501M — 4.8% off, within 5% tolerance."),
    ("doc122_qa0__after75", 0.25, "ACK calibration: confident wrong '0' (gold $411M Pepsico restructuring)."),
    ("doc26_qa0__after75", 0.75, "ANS Amcor gross margin declining — direction right."),
    ("doc126_qa0__after75", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    ("doc111_qa0__after75", 1.0, "ACK honest refusal — doc111 not yet ingested."),
    ("doc53_qa0__after75", 1.0, "ANS gold ~42% decline; predicted $1,874M→$1,093M (41.7% decline) — within tolerance."),
    ("doc25_qa0__after75", 1.0, "ANS Amcor packaging — match."),
    ("doc121_qa0__after75", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc133_qa0__after75", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    # 760-769
    ("doc3_qa0__after76", 0.0, "ANS gold -1.7% operating margin reasons; predicted refusal — refusal on definitive gold."),
    ("doc41_qa0__after76", 1.0, "ANS gross margin not useful for AMEX — match."),
    ("doc112_qa0__after76", 1.0, "ACK honest refusal — doc112 not yet ingested."),
    ("doc100_qa0__after76", 1.0, "ACK honest refusal — doc100 not yet ingested."),
    ("doc37_qa0__after76", 1.0, "ANS one customer 16% — match."),
    ("doc55_qa0__after76", 1.0, "ANS entertainment 9% — match."),
    ("doc18_qa0__after76", 0.0, "ANS gold 93.86; predicted 29.73 — wrong specific."),
    ("doc86_qa0__after76", 1.0, "ACK honest refusal — doc86 not yet ingested."),
    ("doc74_qa0__after76", 1.0, "ANS gold $59,268; predicted 59,268 — exact."),
    ("doc11_qa0__after76", 0.0, "ANS garbled — wrong."),
    # 770-779
    ("doc64_qa0__after77", 1.0, "ANS Yes Boeing cyclical — match."),
    ("doc60_qa0__after77", 1.0, "ANS Yes Commercial Airplanes 39%/$25,867M — match."),
    ("doc113_qa0__after77", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc44_qa0__after77", 1.0, "ANS gold Yes; predicted Yes Card Member retention high — match."),
    ("doc87_qa0__after77", 1.0, "ACK honest refusal — doc87 not yet ingested."),
    ("doc82_qa0__after77", 1.0, "ACK honest refusal — doc82 not yet ingested."),
    ("doc52_qa0__after77", 1.0, "ANS Best Buy operating $1,824M — within tolerance."),
    ("doc97_qa0__after77", 1.0, "ACK honest refusal — doc97 not yet ingested."),
    ("doc130_qa0__after77", 1.0, "ACK honest refusal — doc130 not yet ingested."),
    ("doc11_qa0__after77", 0.0, "ANS garbled — wrong."),
    # 780-789
    ("doc149_qa0__after78", 1.0, "ACK honest refusal — doc149 not yet ingested."),
    ("doc120_qa0__after78", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc19_qa0__after78", 1.0, "ANS gold 30.8% revenue YoY; predicted 30.7% — within tolerance."),
    ("doc44_qa0__after78", 1.0, "ANS Yes retention high — match."),
    ("doc63_qa0__after78", 0.5, "ANS gold limited commercial airlines + US govt 40%; predicted limited commercial airlines (no US govt 40%) — partial."),
    ("doc102_qa0__after78", 1.0, "ACK honest refusal — doc102 not yet ingested."),
    ("doc67_qa0__after78", 0.0, "ANS gold 0.01 (1%) Coca-Cola ROA FY2017; predicted 0.0146 (1.46%) — 46% relative error, outside tolerance."),
    ("doc40_qa0__after78", 1.0, "ANS not measured through operating margin — match."),
    ("doc52_qa0__after78", 1.0, "ANS Best Buy operating $1,824M — within tolerance."),
    ("doc65_qa0__after78", 1.0, "ANS Boeing production rates — match."),
    # 790-799
    ("doc146_qa0__after79", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc23_qa0__after79", 0.0, "ANS gold quick ratio 0.67→0.69; predicted 'not explicitly provided' — refusal on definitive answerable gold."),
    ("doc109_qa0__after79", 1.0, "ACK honest refusal — doc109 not yet ingested."),
    ("doc56_qa0__after79", 1.0, "ANS gold 1.73; predicted 1.74 — within tolerance."),
    ("doc92_qa0__after79", 1.0, "ACK honest refusal — doc92 not yet ingested."),
    ("doc55_qa0__after79", 1.0, "ANS entertainment 9% — match."),
    ("doc28_qa0__after79", 1.0, "ANS AMCOR $2,018mn Adj EBITDA — exact."),
    ("doc83_qa0__after79", 0.25, "ACK calibration: confident wrong $1,000M FCF (gold $3,215M)."),
    ("doc2_qa0__after79", 0.0, "ANS gold 'No efficient CAPEX'; predicted 'Yes capital-intensive' — wrong direction."),
    ("doc14_qa0__after79", 0.0, "ANS gold Yes Adobe FCF improved; predicted refusal — refusal on definitive gold."),
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
