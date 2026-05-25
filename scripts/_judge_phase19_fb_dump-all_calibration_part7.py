"""Claude manual judging — Phase 1.9 FB calibration dump-all (entries 860-1007)."""

from __future__ import annotations

import json
from pathlib import Path

JUDGE_DIR = Path(
    "results/stage3/judge_queue/financebench__dump-all__calibration__seed42"
)
QID_PREFIX = "financebench__dump-all__calibration__"
QID_SUFFIX = "__seed42"

JUDGMENTS: list[tuple[str, float, str]] = [
    # 860-869
    ("doc48_qa0__after86", 0.0, "ANS refusal on definitive gold (2.8%)."),
    ("doc46_qa0__after86", 0.0, "ANS refusal on definitive gold."),
    ("doc84_qa0__after86", 0.0, "ANS gold 0.54; predicted 0.46 — 15% off, outside tolerance."),
    ("doc4_qa0__after86", 0.0, "ANS refusal on definitive gold."),
    ("doc40_qa0__after86", 0.0, "ANS refusal on definitive gold."),
    ("doc26_qa0__after86", 0.0, "ANS refusal on definitive gold."),
    ("doc109_qa0__after86", 1.0, "ACK honest refusal — doc109 not yet ingested."),
    ("doc116_qa0__after86", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    ("doc138_qa0__after86", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc76_qa0__after86", 0.0, "ANS refusal on definitive gold (CVS capital-intensive)."),
    # 870-879
    ("doc12_qa0__after87", 0.0, "ANS refusal on definitive gold (0.83)."),
    ("doc138_qa0__after87", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc43_qa0__after87", 0.0, "ANS refusal on definitive gold (Customer deposits)."),
    ("doc108_qa0__after87", 1.0, "ACK honest refusal — doc108 not yet ingested."),
    ("doc59_qa0__after87", 0.0, "ANS refusal on definitive gold."),
    ("doc4_qa0__after87", 0.0, "ANS refusal on definitive gold."),
    ("doc92_qa0__after87", 0.25, "ACK calibration: '$36.5B Kenvue cash proceeds' confident wrong (gold $13.2B, ~3× over)."),
    ("doc16_qa0__after87", 0.0, "ANS refusal on definitive gold."),
    ("doc91_qa0__after87", 1.0, "ACK honest refusal — doc91 not yet ingested."),
    ("doc124_qa0__after87", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    # 880-889
    ("doc22_qa0__after88", 0.0, "ANS refusal on definitive gold (Amcor 8K)."),
    ("doc27_qa0__after88", 0.0, "ANS refusal on definitive gold."),
    ("doc25_qa0__after88", 0.0, "ANS refusal on definitive gold."),
    ("doc149_qa0__after88", 1.0, "ACK honest refusal — doc149 not yet ingested."),
    ("doc146_qa0__after88", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc66_qa0__after88", 0.0, "ANS refusal on definitive gold."),
    ("doc60_qa0__after88", 0.0, "ANS refusal on definitive gold."),
    ("doc117_qa0__after88", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc21_qa0__after88", 0.0, "ANS refusal on definitive gold ($1,616)."),
    ("doc113_qa0__after88", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    # 890-899
    ("doc34_qa0__after89", 0.0, "ANS refusal on definitive gold (Xilinx)."),
    ("doc129_qa0__after89", 1.0, "ACK honest refusal — doc129 not yet ingested."),
    ("doc89_qa0__after89", 1.0, "ANS gold US 3.0% intl -0.6%; predicted same exact figures — match."),
    ("doc43_qa0__after89", 0.25, "ANS 'long-term debt' confident wrong (gold Customer deposits)."),
    ("doc101_qa0__after89", 1.0, "ACK honest refusal — doc101 not yet ingested."),
    ("doc75_qa0__after89", 0.0, "ANS refusal on definitive gold."),
    ("doc58_qa0__after89", 0.0, "ANS refusal on definitive gold."),
    ("doc111_qa0__after89", 1.0, "ACK honest refusal — doc111 not yet ingested."),
    ("doc83_qa0__after89", 1.0, "ANS gold $3,215; predicted 3,189 — 0.8% off, within tolerance."),
    ("doc2_qa0__after89", 0.0, "ANS refusal on definitive gold."),
    # 900-909
    ("doc66_qa0__after90", 0.0, "ANS refusal on definitive gold."),
    ("doc113_qa0__after90", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc30_qa0__after90", 0.0, "ANS refusal on definitive gold."),
    ("doc116_qa0__after90", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    ("doc41_qa0__after90", 0.0, "ANS refusal on definitive gold."),
    ("doc45_qa0__after90", 0.0, "ANS refusal on definitive gold."),
    ("doc5_qa0__after90", 0.0, "ANS refusal on definitive gold."),
    ("doc91_qa0__after90", 1.0, "ACK calibration: 'Approximately $20 billion' — correct by inference (matches gold)."),
    ("doc125_qa0__after90", 1.0, "ACK honest refusal — doc125 not yet ingested."),
    ("doc126_qa0__after90", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    # 910-919
    ("doc96_qa0__after91", 1.0, "ACK 'JPM gross margins not relevant' — correct by inference."),
    ("doc88_qa0__after91", 0.0, "ANS gold No (3.6%→3.5%); predicted Yes 12.5% — wrong direction."),
    ("doc79_qa0__after91", 0.0, "ANS gold Yes Mary Dillon definitive; predicted refusal — refusal on definitive gold."),
    ("doc33_qa0__after91", 0.25, "ANS 'increased sales volume contributed' — vague, no EPYC/Xilinx specifics."),
    ("doc20_qa0__after91", 0.0, "ANS refusal on definitive gold ($11,588)."),
    ("doc40_qa0__after91", 0.0, "ANS refusal on definitive gold."),
    ("doc86_qa0__after91", 1.0, "ANS gold COVID-19 + currency + commodity inflation; predicted same + supply chain Consumer detail — match."),
    ("doc15_qa0__after91", 1.0, "ANS 0 — exact."),
    ("doc99_qa0__after91", 1.0, "ACK honest refusal — doc99 not yet ingested."),
    ("doc18_qa0__after91", 0.0, "ANS refusal on definitive gold."),
    # 920-929
    ("doc101_qa0__after92", 1.0, "ACK honest refusal — doc101 not yet ingested."),
    ("doc45_qa0__after92", 0.0, "ANS refusal on definitive gold."),
    ("doc114_qa0__after92", 1.0, "ACK honest refusal — doc114 not yet ingested."),
    ("doc78_qa0__after92", 0.0, "ANS refusal on definitive gold (Yes $0.55/quarter)."),
    ("doc91_qa0__after92", 1.0, "ANS '$20 billion' — exact match."),
    ("doc10_qa0__after92", 0.0, "ANS refusal on definitive gold."),
    ("doc12_qa0__after92", 0.0, "ANS refusal on definitive gold."),
    ("doc94_qa0__after92", 1.0, "ACK honest refusal — doc94 not yet ingested."),
    ("doc86_qa0__after92", 1.0, "ANS gold COVID-19 + currency + commodity inflation; predicted same — match."),
    ("doc122_qa0__after92", 0.25, "ACK '0' confident wrong."),
    # 930-939
    ("doc26_qa0__after93", 0.0, "ANS refusal on definitive gold."),
    ("doc64_qa0__after93", 0.0, "ANS refusal on definitive gold (Yes Boeing cyclical)."),
    ("doc146_qa0__after93", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc136_qa0__after93", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc54_qa0__after93", 0.0, "ANS refusal on definitive gold."),
    ("doc106_qa0__after93", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc149_qa0__after93", 1.0, "ACK honest refusal — doc149 not yet ingested."),
    ("doc144_qa0__after93", 1.0, "ACK honest refusal — doc144 not yet ingested."),
    ("doc143_qa0__after93", 1.0, "ACK honest refusal — doc143 not yet ingested."),
    ("doc82_qa0__after93", 0.0, "ANS gold 0.68; predicted 1.29 — wrong specific."),
    # 940-949
    ("doc18_qa0__after94", 0.0, "ANS refusal on definitive gold."),
    ("doc126_qa0__after94", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    ("doc52_qa0__after94", 1.0, "ANS gold Best Buy operating $1.8bn; predicted 'cash flow from operating activities most cash flow Best Buy FY23' — match direction (no $1.8bn)."),
    ("doc9_qa0__after94", 0.0, "ANS refusal on definitive gold (1.9%)."),
    ("doc64_qa0__after94", 0.0, "ANS refusal on definitive gold."),
    ("doc117_qa0__after94", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc129_qa0__after94", 0.25, "ACK '2 percentage points' confident wrong (gold 1 pp)."),
    ("doc83_qa0__after94", 0.0, "ANS refusal on definitive gold."),
    ("doc112_qa0__after94", 1.0, "ACK honest refusal — doc112 not yet ingested."),
    ("doc104_qa0__after94", 1.0, "ACK honest refusal — doc104 not yet ingested."),
    # 950-959
    ("doc18_qa0__after95", 0.0, "ANS refusal on definitive gold."),
    ("doc80_qa0__after95", 0.0, "ANS refusal on definitive gold (Yes Richard A. Johnson)."),
    ("doc52_qa0__after95", 0.0, "ANS refusal on definitive gold."),
    ("doc100_qa0__after95", 1.0, "ACK honest refusal — doc100 not yet ingested."),
    ("doc106_qa0__after95", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc51_qa0__after95", 0.0, "ANS refusal on definitive gold."),
    ("doc142_qa0__after95", 1.0, "ACK honest refusal — doc142 not yet ingested."),
    ("doc122_qa0__after95", 0.25, "ACK '0' confident wrong."),
    ("doc8_qa0__after95", 0.0, "ANS refusal on definitive gold (24.26)."),
    ("doc17_qa0__after95", 0.0, "ANS refusal on definitive gold."),
    # 960-969
    ("doc86_qa0__after96", 1.0, "ANS COVID-19 + currency + commodity — match."),
    ("doc80_qa0__after96", 0.0, "ANS refusal on definitive gold."),
    ("doc94_qa0__after96", 0.0, "ANS gold 'Corporate -$473M'; predicted 'Consumer & Community Banking' — wrong segment."),
    ("doc15_qa0__after96", 1.0, "ANS 0 — exact."),
    ("doc95_qa0__after96", 0.0, "ANS gold $66.56/share; predicted '$1,000' — wildly wrong specific."),
    ("doc127_qa0__after96", 1.0, "ACK honest refusal — doc127 not yet ingested."),
    ("doc53_qa0__after96", 0.0, "ANS refusal on definitive gold."),
    ("doc52_qa0__after96", 0.0, "ANS refusal on definitive gold."),
    ("doc50_qa0__after96", 0.0, "ANS refusal on definitive gold."),
    ("doc39_qa0__after96", 0.0, "ANS refusal on definitive gold."),
    # 970-979
    ("doc133_qa0__after97", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc63_qa0__after97", 0.0, "ANS refusal on definitive gold."),
    ("doc118_qa0__after97", 1.0, "ACK honest refusal — doc118 not yet ingested."),
    ("doc8_qa0__after97", 0.0, "ANS refusal on definitive gold."),
    ("doc47_qa0__after97", 0.0, "ANS refusal on definitive gold."),
    ("doc125_qa0__after97", 1.0, "ACK 'proposal not approved' = 'defeated' — correct."),
    ("doc95_qa0__after97", 0.5, "ANS gold $66.56/share; predicted '$292.3B equity, proportional amount, exact amount cannot be determined' — hedged, provides framework no specific."),
    ("doc37_qa0__after97", 0.0, "ANS refusal on definitive gold."),
    ("doc6_qa0__after97", 0.0, "ANS refusal on definitive gold (3M debt securities)."),
    ("doc50_qa0__after97", 0.0, "ANS refusal on definitive gold."),
    # 980-989
    ("doc42_qa0__after98", 0.0, "ANS refusal on definitive gold."),
    ("doc141_qa0__after98", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    ("doc80_qa0__after98", 0.0, "ANS refusal on definitive gold."),
    ("doc91_qa0__after98", 1.0, "ANS '$20 billion' JnJ Consumer Health — exact match."),
    ("doc60_qa0__after98", 0.0, "ANS refusal on definitive gold."),
    ("doc149_qa0__after98", 1.0, "ACK honest refusal — doc149 not yet ingested."),
    ("doc108_qa0__after98", 1.0, "ACK honest refusal — doc108 not yet ingested."),
    ("doc97_qa0__after98", 0.0, "ANS gold Corporate & Investment Bank $3725M; predicted 'Consumer & Community Banking $3,100M' — wrong segment."),
    ("doc138_qa0__after98", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc16_qa0__after98", 0.0, "ANS refusal on definitive gold."),
    # 990-999
    ("doc113_qa0__after99", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc11_qa0__after99", 0.0, "ANS refusal on definitive gold."),
    ("doc40_qa0__after99", 0.0, "ANS refusal on definitive gold."),
    ("doc127_qa0__after99", 1.0, "ACK honest refusal — doc127 not yet ingested."),
    ("doc108_qa0__after99", 0.25, "ACK 'International 11.5% decline' confident wrong (gold MGM China 44%)."),
    ("doc145_qa0__after99", 1.0, "ACK honest refusal — doc145 not yet ingested."),
    ("doc43_qa0__after99", 0.25, "ANS 'Total liabilities... AP, accrued, long-term debt' confident vague-wrong (gold Customer deposits)."),
    ("doc71_qa0__after99", 0.0, "ANS refusal on definitive gold."),
    ("doc124_qa0__after99", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc116_qa0__after99", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    # 1000-1007
    ("doc5_qa0__after100", 0.0, "ANS refusal on definitive gold."),
    ("doc129_qa0__after100", 0.25, "ACK '2 percentage points' confident wrong."),
    ("doc10_qa0__after100", 0.0, "ANS refusal on definitive gold."),
    ("doc90_qa0__after100", 1.0, "ANS Consumer Health discontinued — exact."),
    ("doc148_qa0__after100", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc15_qa0__after100", 1.0, "ANS 0 — exact."),
    ("doc67_qa0__after100", 0.0, "ANS refusal on definitive gold."),
    ("doc127_qa0__after100", 1.0, "ACK honest refusal — doc127 not yet ingested."),
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
