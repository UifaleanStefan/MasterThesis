"""Claude manual judging — Phase 1.9 FB calibration v4t-corpus-tuned (entries 400-599).

Idempotent append to results.jsonl per evaluation/claude_judge_protocol.md.
All scores assigned by Claude in-session per HARD RULE in AGENTS.md §0.
"""

from __future__ import annotations

import json
from pathlib import Path

JUDGE_DIR = Path(
    "results/stage3/judge_queue/financebench__v4t-corpus-tuned__calibration__seed42"
)

# qid format: financebench__v4t-corpus-tuned__calibration__seed42::<doc>_qa0__after<N>
QID_PREFIX = "financebench__v4t-corpus-tuned__calibration__seed42::"
QID_SUFFIX = ""

# (qid_suffix, score, rationale)
JUDGMENTS: list[tuple[str, float, str]] = [
    # 400-409
    ("doc16_qa0__after40", 0.0, "ANS gold 9.5 inventory turnover; predicted 11.98 — wrong specific number outside 5% tolerance."),
    ("doc93_qa0__after40", 1.0, "ACK honest refusal — doc93 not yet ingested."),
    ("doc128_qa0__after40", 1.0, "ACK honest refusal — doc128 not yet ingested."),
    ("doc110_qa0__after40", 1.0, "ACK honest refusal — doc110 not yet ingested."),
    ("doc59_qa0__after40", 1.0, "ACK honest refusal — doc59 not yet ingested."),
    ("doc54_qa0__after40", 1.0, "ACK honest refusal — doc54 not yet ingested."),
    ("doc135_qa0__after40", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc11_qa0__after40", 0.0, "ANS gold 65.4% operating income YoY; predicted garbled calc with no final figure — incoherent wrong."),
    ("doc53_qa0__after40", 1.0, "ACK honest refusal — doc53 not yet ingested."),
    ("doc57_qa0__after40", 1.0, "ACK honest refusal — doc57 not yet ingested."),
    # 410-419
    ("doc85_qa0__after41", 1.0, "ACK honest refusal — doc85 not yet ingested."),
    ("doc88_qa0__after41", 1.0, "ACK honest refusal — doc88 not yet ingested."),
    ("doc53_qa0__after41", 1.0, "ACK honest refusal — doc53 not yet ingested."),
    ("doc61_qa0__after41", 1.0, "ACK honest refusal — doc61 not yet ingested."),
    ("doc46_qa0__after41", 1.0, "ACK honest refusal — doc46 not yet ingested."),
    ("doc124_qa0__after41", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc84_qa0__after41", 1.0, "ACK honest refusal — doc84 not yet ingested."),
    ("doc134_qa0__after41", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc21_qa0__after41", 1.0, "ANS gold $1616 AR; predicted $1,615.9M — within rounding tolerance."),
    ("doc87_qa0__after41", 1.0, "ACK honest refusal — doc87 not yet ingested."),
    # 420-429
    ("doc106_qa0__after42", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc124_qa0__after42", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc98_qa0__after42", 1.0, "ACK honest refusal — doc98 not yet ingested."),
    ("doc56_qa0__after42", 1.0, "ACK honest refusal — doc56 not yet ingested."),
    ("doc36_qa0__after42", 1.0, "ANS gold 'Data Center'; predicted 'Data Center segment' — equivalent."),
    ("doc51_qa0__after42", 1.0, "ACK honest refusal — doc51 not yet ingested."),
    ("doc111_qa0__after42", 1.0, "ACK honest refusal — doc111 not yet ingested."),
    ("doc60_qa0__after42", 1.0, "ACK honest refusal — doc60 not yet ingested."),
    ("doc148_qa0__after42", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc50_qa0__after42", 1.0, "ACK honest refusal — doc50 not yet ingested."),
    # 430-439
    ("doc25_qa0__after43", 1.0, "ANS gold Amcor packaging; predicted same scope (food/beverage/pharma packaging) — semantically equivalent."),
    ("doc114_qa0__after43", 1.0, "ACK honest refusal — doc114 not yet ingested."),
    ("doc133_qa0__after43", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc141_qa0__after43", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    ("doc55_qa0__after43", 1.0, "ACK honest refusal — doc55 not yet ingested."),
    ("doc85_qa0__after43", 1.0, "ACK honest refusal — doc85 not yet ingested."),
    ("doc27_qa0__after43", 0.5, "ANS gold '87% restructuring liability is employee'; predicted describes restructuring breakdown but doesn't quantify 87% — partial."),
    ("doc94_qa0__after43", 1.0, "ACK honest refusal — doc94 not yet ingested."),
    ("doc122_qa0__after43", 0.25, "ACK calibration: confident wrong specific '0' (gold $411M Pepsico restructuring)."),
    ("doc24_qa0__after43", 0.75, "ANS gold lists FY2023 acquisitions; predicted lists all 3 (Czech + Shanghai medical + NZ protein) but slightly muddles FY2022 vs FY2023 framing — mostly correct."),
    # 440-449
    ("doc76_qa0__after44", 1.0, "ACK honest refusal — doc76 not yet ingested."),
    ("doc35_qa0__after44", 1.0, "ANS gold 'cashflow from Operations'; predicted '$3,565M from operating activities' — matches."),
    ("doc17_qa0__after44", 0.0, "ANS gold -0.02; predicted -1.32 — wrong specific number outside tolerance."),
    ("doc30_qa0__after44", 1.0, "ANS gold 4.2% D&A margin; predicted 4.18% — within tolerance."),
    ("doc66_qa0__after44", 1.0, "ACK honest refusal — doc66 not yet ingested."),
    ("doc101_qa0__after44", 1.0, "ACK honest refusal — doc101 not yet ingested."),
    ("doc95_qa0__after44", 1.0, "ACK honest refusal — doc95 not yet ingested."),
    ("doc67_qa0__after44", 1.0, "ACK honest refusal — doc67 not yet ingested."),
    ("doc53_qa0__after44", 1.0, "ACK honest refusal — doc53 not yet ingested."),
    ("doc141_qa0__after44", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    # 450-459
    ("doc124_qa0__after45", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc141_qa0__after45", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    ("doc56_qa0__after45", 1.0, "ACK honest refusal — doc56 not yet ingested."),
    ("doc11_qa0__after45", 1.0, "ANS gold 65.4% operating income YoY; predicted 65.3% — within tolerance."),
    ("doc109_qa0__after45", 1.0, "ACK honest refusal — doc109 not yet ingested."),
    ("doc59_qa0__after45", 1.0, "ACK honest refusal — doc59 not yet ingested."),
    ("doc57_qa0__after45", 1.0, "ACK honest refusal — doc57 not yet ingested."),
    ("doc30_qa0__after45", 1.0, "ANS 4.2% gold vs 4.18% pred — within tolerance."),
    ("doc32_qa0__after45", 1.0, "ANS gold lists AMD products (CPUs/GPUs/DPUs/FPGAs/SoC); predicted same list — matches."),
    ("doc31_qa0__after45", 0.0, "ANS gold Yes quick ratio 1.57 definitive; predicted 'cannot be determined' — refusal on definitive answerable gold."),
    # 460-469
    ("doc99_qa0__after46", 1.0, "ACK honest refusal — doc99 not yet ingested."),
    ("doc37_qa0__after46", 1.0, "ANS gold Yes 16% one customer; predicted Yes 16% — exact match."),
    ("doc54_qa0__after46", 1.0, "ACK honest refusal — doc54 not yet ingested."),
    ("doc118_qa0__after46", 1.0, "ACK honest refusal — doc118 not yet ingested."),
    ("doc58_qa0__after46", 0.25, "ACK calibration: confident wrong specific '$1,426M FY2020 operating cash flow' (gold $382M Block FY2020)."),
    ("doc24_qa0__after46", 0.75, "ANS Amcor acquisitions — same as 439, mostly correct with minor framing issue."),
    ("doc30_qa0__after46", 1.0, "ANS 4.2% vs 4.18% — within tolerance."),
    ("doc50_qa0__after46", 1.0, "ACK honest refusal — doc50 not yet ingested."),
    ("doc148_qa0__after46", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc95_qa0__after46", 1.0, "ACK honest refusal — doc95 not yet ingested."),
    # 470-479
    ("doc1_qa0__after47", 1.0, "ANS gold $8.70; predicted $8.738B — within 0.5% tolerance."),
    ("doc75_qa0__after47", 1.0, "ACK honest refusal — doc75 not yet ingested."),
    ("doc92_qa0__after47", 1.0, "ACK honest refusal — doc92 not yet ingested."),
    ("doc87_qa0__after47", 1.0, "ACK honest refusal — doc87 not yet ingested."),
    ("doc93_qa0__after47", 1.0, "ACK honest refusal — doc93 not yet ingested."),
    ("doc78_qa0__after47", 1.0, "ACK honest refusal — doc78 not yet ingested."),
    ("doc97_qa0__after47", 1.0, "ACK honest refusal — doc97 not yet ingested."),
    ("doc49_qa0__after47", 1.0, "ACK honest refusal — doc49 not yet ingested."),
    ("doc136_qa0__after47", 1.0, "ACK honest refusal — doc136 not yet ingested (gold is 'There are none' but refusal is honestly equivalent here)."),
    ("doc31_qa0__after47", 0.0, "ANS gold Yes quick ratio 1.57; predicted 'cannot be determined' — refusal on definitive gold."),
    # 480-489
    ("doc125_qa0__after48", 1.0, "ACK calibration: 'proposal was not approved' = correct by interpretation (gold says 'defeated')."),
    ("doc4_qa0__after48", 0.5, "ANS gold consumer segment shrunk 0.9% organically; predicted just 'Consumer segment' — partial (right segment, no figure)."),
    ("doc58_qa0__after48", 0.25, "ACK calibration: confident wrong $1,426M (gold $382M Block FY2020 OCF)."),
    ("doc133_qa0__after48", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc40_qa0__after48", 1.0, "ANS gold 'performance not measured through operating margin'; predicted explains why operating margin not useful for AMEX — equivalent."),
    ("doc148_qa0__after48", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc30_qa0__after48", 1.0, "ANS 4.2% vs 4.18% — within tolerance."),
    ("doc76_qa0__after48", 1.0, "ACK honest refusal — doc76 not yet ingested."),
    ("doc121_qa0__after48", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc75_qa0__after48", 1.0, "ACK honest refusal — doc75 not yet ingested."),
    # 490-499
    ("doc41_qa0__after49", 1.0, "ANS gold 'performance not measured through gross margin'; predicted equivalent — matches."),
    ("doc27_qa0__after49", 0.5, "ANS gold 87% employee restructuring; predicted $93M total with breakdown but no 87% figure — partial."),
    ("doc16_qa0__after49", 0.0, "ANS gold 9.5 turnover; predicted 11.98 — wrong specific outside tolerance."),
    ("doc145_qa0__after49", 1.0, "ACK honest refusal — doc145 not yet ingested."),
    ("doc117_qa0__after49", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc65_qa0__after49", 1.0, "ACK honest refusal — doc65 not yet ingested."),
    ("doc66_qa0__after49", 1.0, "ACK honest refusal — doc66 not yet ingested."),
    ("doc58_qa0__after49", 1.0, "ACK honest refusal — doc58 not yet ingested."),
    ("doc138_qa0__after49", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc4_qa0__after49", 0.5, "ANS gold consumer shrunk 0.9%; predicted just 'Consumer segment' — partial."),
    # 500-509
    ("doc76_qa0__after50", 1.0, "ACK honest refusal — doc76 not yet ingested."),
    ("doc113_qa0__after50", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc9_qa0__after50", 0.0, "ANS gold 1.9%; predicted 5.0% — wrong specific outside tolerance."),
    ("doc136_qa0__after50", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc24_qa0__after50", 0.75, "ANS Amcor acquisitions — mostly correct, FY framing slightly muddled."),
    ("doc130_qa0__after50", 1.0, "ACK honest refusal — doc130 not yet ingested."),
    ("doc11_qa0__after50", 0.0, "ANS gold 65.4%; predicted garbled calculation, no clear final figure."),
    ("doc35_qa0__after50", 1.0, "ANS gold 'cashflow from Operations'; predicted '$3,565M operating cash flow' — matches."),
    ("doc29_qa0__after50", 0.0, "ANS gold 'Real Growth was flat'; predicted 'decrease of 5%' — wrong direction."),
    ("doc53_qa0__after50", 1.0, "ACK honest refusal — doc53 not yet ingested."),
    # 510-519
    ("doc52_qa0__after51", 1.0, "ACK calibration: 'cash flows from operating activities for Best Buy in FY2023' — correct answer by inference, matches gold 'operating activities ($1.8bn)'."),
    ("doc122_qa0__after51", 0.25, "ACK calibration: confident wrong '0' (gold $411M Pepsico restructuring)."),
    ("doc128_qa0__after51", 1.0, "ACK honest refusal — doc128 not yet ingested."),
    ("doc53_qa0__after51", 1.0, "ACK honest refusal — doc53 not yet ingested."),
    ("doc104_qa0__after51", 1.0, "ACK honest refusal — doc104 not yet ingested."),
    ("doc98_qa0__after51", 1.0, "ACK honest refusal — doc98 not yet ingested."),
    ("doc17_qa0__after51", 0.0, "ANS gold -0.02; predicted -1.32 — wrong specific."),
    ("doc77_qa0__after51", 1.0, "ACK honest refusal — doc77 not yet ingested."),
    ("doc136_qa0__after51", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc61_qa0__after51", 1.0, "ACK honest refusal — doc61 not yet ingested."),
    # 520-529
    ("doc137_qa0__after52", 1.0, "ACK honest refusal — doc137 not yet ingested (gold 'no acquisitions FY2023/22' but refusal is honestly equivalent)."),
    ("doc30_qa0__after52", 1.0, "ANS 4.2% vs 4.18% — within tolerance."),
    ("doc54_qa0__after52", 1.0, "ACK honest refusal — doc54 not yet ingested."),
    ("doc53_qa0__after52", 1.0, "ACK honest refusal — doc53 not yet ingested."),
    ("doc80_qa0__after52", 1.0, "ACK honest refusal — doc80 not yet ingested."),
    ("doc36_qa0__after52", 1.0, "ANS gold Data Center; predicted 'Data Center segment' — match."),
    ("doc121_qa0__after52", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc125_qa0__after52", 1.0, "ACK calibration: 'proposal was not approved' = correct by interpretation (gold 'defeated')."),
    ("doc136_qa0__after52", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc35_qa0__after52", 1.0, "ANS cashflow from Operations $3,565M — matches gold."),
    # 530-539
    ("doc94_qa0__after53", 1.0, "ACK honest refusal — doc94 not yet ingested."),
    ("doc36_qa0__after53", 1.0, "ANS Data Center — match."),
    ("doc56_qa0__after53", 1.0, "ACK honest refusal — doc56 not yet ingested."),
    ("doc29_qa0__after53", 0.0, "ANS gold 'flat real growth'; predicted '5% decrease' — wrong direction."),
    ("doc139_qa0__after53", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc15_qa0__after53", 1.0, "ANS gold 0; predicted 0 — exact."),
    ("doc0_qa0__after53", 0.0, "ANS gold $1577 capex; predicted 'not explicitly provided' — refusal on definitive gold."),
    ("doc78_qa0__after53", 1.0, "ACK honest refusal — doc78 not yet ingested."),
    ("doc50_qa0__after53", 0.0, "ANS gold 'consistent margins minor 1.1% decline'; predicted 'fluctuated more than 2%' — wrong direction."),
    ("doc145_qa0__after53", 1.0, "ACK honest refusal — doc145 not yet ingested."),
    # 540-549
    ("doc63_qa0__after54", 1.0, "ACK honest refusal — doc63 not yet ingested."),
    ("doc0_qa0__after54", 1.0, "ANS gold $1577 capex; predicted $1,501M — 4.8% off, within 5% tolerance."),
    ("doc134_qa0__after54", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc80_qa0__after54", 1.0, "ACK honest refusal — doc80 not yet ingested."),
    ("doc133_qa0__after54", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc29_qa0__after54", 0.0, "ANS gold flat; predicted decrease 5% — wrong direction."),
    ("doc42_qa0__after54", 1.0, "ANS gold 24.6% → 21.6%; predicted same exact figures — match."),
    ("doc83_qa0__after54", 1.0, "ACK honest refusal — doc83 not yet ingested."),
    ("doc137_qa0__after54", 1.0, "ACK honest refusal — doc137 not yet ingested."),
    ("doc92_qa0__after54", 1.0, "ACK honest refusal — doc92 not yet ingested."),
    # 550-559
    ("doc147_qa0__after55", 1.0, "ACK honest refusal — doc147 not yet ingested."),
    ("doc108_qa0__after55", 1.0, "ACK honest refusal — doc108 not yet ingested."),
    ("doc100_qa0__after55", 1.0, "ACK honest refusal — doc100 not yet ingested."),
    ("doc37_qa0__after55", 1.0, "ANS Yes 16% one customer — match."),
    ("doc50_qa0__after55", 0.0, "ANS gold consistent margins minor decline; predicted fluctuated >2% — wrong direction."),
    ("doc92_qa0__after55", 0.25, "ACK calibration: confident wrong '$3.7B Kenvue cash proceeds' (gold $13.2B JnJ Kenvue)."),
    ("doc53_qa0__after55", 1.0, "ANS gold ~42% decline FY23→Q2 FY24; predicted Yes drop $1,874M→$1,093M (41.7% decline) — within tolerance."),
    ("doc29_qa0__after55", 0.0, "ANS gold flat; predicted 5% decrease — wrong direction."),
    ("doc120_qa0__after55", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc128_qa0__after55", 1.0, "ACK honest refusal — doc128 not yet ingested."),
    # 560-569
    ("doc3_qa0__after56", 0.0, "ANS gold operating margin -1.7% due to gross margin and one-off charges; predicted 'do not provide specific information' — refusal on definitive answerable gold."),
    ("doc22_qa0__after56", 1.0, "ANS gold Amcor 8K substitution of issuer for Senior Notes 2026/2028; predicted same with details — match."),
    ("doc116_qa0__after56", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    ("doc141_qa0__after56", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    ("doc14_qa0__after56", 0.0, "ANS gold Yes Adobe FCF improved ~13% (143%→156%); predicted 'do not contain specific information' — refusal on definitive answerable gold."),
    ("doc88_qa0__after56", 1.0, "ACK honest refusal — doc88 not yet ingested."),
    ("doc148_qa0__after56", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc60_qa0__after56", 1.0, "ACK honest refusal — doc60 not yet ingested."),
    ("doc67_qa0__after56", 1.0, "ACK honest refusal — doc67 not yet ingested."),
    ("doc109_qa0__after56", 1.0, "ACK honest refusal — doc109 not yet ingested."),
    # 570-579
    ("doc120_qa0__after57", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc63_qa0__after57", 1.0, "ACK honest refusal — doc63 not yet ingested."),
    ("doc27_qa0__after57", 0.5, "ANS gold 87% employee; predicted $93M total without 87% — partial."),
    ("doc28_qa0__after57", 1.0, "ANS gold $2,018mn AMCOR Adj EBITDA; predicted '2,018 million' — exact."),
    ("doc31_qa0__after57", 0.0, "ANS gold Yes quick ratio 1.57 definitive; predicted 'cannot be determined' — refusal on definitive gold."),
    ("doc107_qa0__after57", 1.0, "ACK honest refusal — doc107 not yet ingested."),
    ("doc74_qa0__after57", 1.0, "ACK calibration: predicted $59,364M Costco FY2021 total assets — within tolerance of gold $59,268 (0.16% off). Correct by inference."),
    ("doc121_qa0__after57", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc69_qa0__after57", 1.0, "ACK honest refusal — doc69 not yet ingested."),
    ("doc57_qa0__after57", 0.0, "ANS gold 101.5% Block revenue growth FY19→FY20; predicted 16.5% — wrong specific."),
    # 580-589
    ("doc55_qa0__after58", 1.0, "ANS gold entertainment 9% growth from gaming; predicted same — match."),
    ("doc118_qa0__after58", 1.0, "ACK honest refusal — doc118 not yet ingested."),
    ("doc59_qa0__after58", 1.0, "ACK honest refusal — doc59 not yet ingested."),
    ("doc64_qa0__after58", 1.0, "ACK calibration: 'Yes Boeing business is cyclical' — correct by inference; gold says yes due to airlines."),
    ("doc17_qa0__after58", 0.0, "ANS gold -0.02; predicted -1.32 — wrong specific."),
    ("doc14_qa0__after58", 0.0, "ANS gold Yes Adobe FCF improved ~13%; predicted refusal — refusal on definitive gold."),
    ("doc16_qa0__after58", 0.0, "ANS gold 9.5 turnover; predicted 11.98 — wrong specific outside tolerance."),
    ("doc66_qa0__after58", 1.0, "ACK honest refusal — doc66 not yet ingested."),
    ("doc78_qa0__after58", 1.0, "ACK honest refusal — doc78 not yet ingested."),
    ("doc95_qa0__after58", 1.0, "ACK honest refusal — doc95 not yet ingested."),
    # 590-599
    ("doc29_qa0__after59", 0.0, "ANS gold flat real growth; predicted 5% decrease — wrong direction."),
    ("doc65_qa0__after59", 1.0, "ACK honest refusal — doc65 not yet ingested."),
    ("doc87_qa0__after59", 1.0, "ACK honest refusal — doc87 not yet ingested."),
    ("doc116_qa0__after59", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    ("doc66_qa0__after59", 1.0, "ACK honest refusal — doc66 not yet ingested."),
    ("doc110_qa0__after59", 1.0, "ACK honest refusal — doc110 not yet ingested."),
    ("doc30_qa0__after59", 1.0, "ANS gold 4.2% D&A margin; predicted 4.18% — within tolerance."),
    ("doc134_qa0__after59", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc119_qa0__after59", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    ("doc147_qa0__after59", 1.0, "ACK honest refusal — doc147 not yet ingested."),
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
