"""Claude manual judging — Phase 1.9 FB calibration dump-all (entries 422-567)."""

from __future__ import annotations

import json
from pathlib import Path

JUDGE_DIR = Path(
    "results/stage3/judge_queue/financebench__dump-all__calibration__seed42"
)
QID_PREFIX = "financebench__dump-all__calibration__"
QID_SUFFIX = "__seed42"

JUDGMENTS: list[tuple[str, float, str]] = [
    # 422-429
    ("doc98_qa0__after42", 1.0, "ACK honest refusal — doc98 not yet ingested."),
    ("doc56_qa0__after42", 1.0, "ACK honest refusal — doc56 not yet ingested."),
    ("doc36_qa0__after42", 0.0, "ANS gold Data Center; predicted 'Gaming segment' — wrong segment."),
    ("doc51_qa0__after42", 1.0, "ACK honest refusal — doc51 not yet ingested."),
    ("doc111_qa0__after42", 1.0, "ACK honest refusal — doc111 not yet ingested."),
    ("doc60_qa0__after42", 1.0, "ACK honest refusal — doc60 not yet ingested."),
    ("doc148_qa0__after42", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc50_qa0__after42", 1.0, "ACK honest refusal — doc50 not yet ingested."),
    # 430-439
    ("doc25_qa0__after43", 0.0, "ANS gold Amcor packaging definitive; predicted refusal — refusal on definitive gold."),
    ("doc114_qa0__after43", 1.0, "ACK honest refusal — doc114 not yet ingested."),
    ("doc133_qa0__after43", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc141_qa0__after43", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    ("doc55_qa0__after43", 1.0, "ACK honest refusal — doc55 not yet ingested."),
    ("doc85_qa0__after43", 1.0, "ACK honest refusal — doc85 not yet ingested."),
    ("doc27_qa0__after43", 0.0, "ANS refusal on definitive gold (87% restructuring)."),
    ("doc94_qa0__after43", 1.0, "ACK honest refusal — doc94 not yet ingested."),
    ("doc122_qa0__after43", 0.25, "ACK '0' confident wrong (gold $411M)."),
    ("doc24_qa0__after43", 0.0, "ANS refusal on definitive gold (Amcor acquisitions)."),
    # 440-449
    ("doc76_qa0__after44", 1.0, "ACK honest refusal — doc76 not yet ingested."),
    ("doc35_qa0__after44", 1.0, "ANS gold 'cashflow from Operations AMD FY22'; predicted '$3,565M' — match."),
    ("doc17_qa0__after44", 0.0, "ANS refusal on definitive gold (-0.02)."),
    ("doc30_qa0__after44", 0.0, "ANS refusal on definitive gold (4.2%)."),
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
    ("doc11_qa0__after45", 0.0, "ANS refusal on definitive gold (65.4%)."),
    ("doc109_qa0__after45", 1.0, "ACK honest refusal — doc109 not yet ingested."),
    ("doc59_qa0__after45", 1.0, "ACK honest refusal — doc59 not yet ingested."),
    ("doc57_qa0__after45", 1.0, "ACK honest refusal — doc57 not yet ingested."),
    ("doc30_qa0__after45", 0.0, "ANS refusal on definitive gold (4.2%)."),
    ("doc32_qa0__after45", 0.75, "ANS gold AMD CPUs/GPUs/DPUs/FPGAs/SoC; predicted 'Data Center, Client, Gaming, Embedded' — partial (gives segments not product list)."),
    ("doc31_qa0__after45", 0.0, "ANS refusal on definitive gold (quick ratio 1.57)."),
    # 460-469
    ("doc99_qa0__after46", 1.0, "ACK honest refusal — doc99 not yet ingested."),
    ("doc37_qa0__after46", 1.0, "ANS Yes one customer 16% — match."),
    ("doc54_qa0__after46", 1.0, "ACK honest refusal — doc54 not yet ingested."),
    ("doc118_qa0__after46", 1.0, "ACK honest refusal — doc118 not yet ingested."),
    ("doc58_qa0__after46", 0.25, "ACK calibration: '1,194' confident wrong specific (gold $382 Block FY20 OCF)."),
    ("doc24_qa0__after46", 0.0, "ANS refusal on definitive gold."),
    ("doc30_qa0__after46", 0.0, "ANS refusal on definitive gold."),
    ("doc50_qa0__after46", 1.0, "ACK honest refusal — doc50 not yet ingested."),
    ("doc148_qa0__after46", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc95_qa0__after46", 1.0, "ACK honest refusal — doc95 not yet ingested."),
    # 470-479
    ("doc1_qa0__after47", 0.0, "ANS refusal on definitive gold ($8.70)."),
    ("doc75_qa0__after47", 1.0, "ACK honest refusal — doc75 not yet ingested."),
    ("doc92_qa0__after47", 1.0, "ACK honest refusal — doc92 not yet ingested."),
    ("doc87_qa0__after47", 1.0, "ACK honest refusal — doc87 not yet ingested."),
    ("doc93_qa0__after47", 1.0, "ACK honest refusal — doc93 not yet ingested."),
    ("doc78_qa0__after47", 1.0, "ACK honest refusal — doc78 not yet ingested."),
    ("doc97_qa0__after47", 1.0, "ACK honest refusal — doc97 not yet ingested."),
    ("doc49_qa0__after47", 1.0, "ACK honest refusal — doc49 not yet ingested."),
    ("doc136_qa0__after47", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc31_qa0__after47", 0.0, "ANS refusal on definitive gold."),
    # 480-489
    ("doc125_qa0__after48", 1.0, "ACK 'proposal not approved 62%' — correct + detail."),
    ("doc4_qa0__after48", 0.0, "ANS gold consumer shrunk 0.9% definitive; predicted refusal — refusal on definitive gold."),
    ("doc58_qa0__after48", 1.0, "ACK honest refusal — doc58 not yet ingested."),
    ("doc133_qa0__after48", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc40_qa0__after48", 1.0, "ANS gold 'not measured through op margin'; predicted 'operating margin not useful for AMEX' — correct by inference."),
    ("doc148_qa0__after48", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc30_qa0__after48", 0.0, "ANS refusal on definitive gold."),
    ("doc76_qa0__after48", 1.0, "ACK honest refusal — doc76 not yet ingested."),
    ("doc121_qa0__after48", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc75_qa0__after48", 1.0, "ACK honest refusal — doc75 not yet ingested."),
    # 490-499
    ("doc41_qa0__after49", 1.0, "ANS gold gross margin not useful AMEX; predicted same — match."),
    ("doc27_qa0__after49", 0.0, "ANS refusal on definitive gold (87%)."),
    ("doc16_qa0__after49", 0.0, "ANS refusal on definitive gold (9.5 AES turnover)."),
    ("doc145_qa0__after49", 1.0, "ACK honest refusal — doc145 not yet ingested."),
    ("doc117_qa0__after49", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc65_qa0__after49", 1.0, "ACK honest refusal — doc65 not yet ingested."),
    ("doc66_qa0__after49", 1.0, "ACK honest refusal — doc66 not yet ingested."),
    ("doc58_qa0__after49", 1.0, "ACK honest refusal — doc58 not yet ingested."),
    ("doc138_qa0__after49", 0.25, "ACK calibration: 'operating efficiencies' vague (gold specific lower marketing + leverage)."),
    ("doc4_qa0__after49", 0.0, "ANS refusal on definitive gold."),
    # 500-509
    ("doc76_qa0__after50", 1.0, "ACK honest refusal — doc76 not yet ingested."),
    ("doc113_qa0__after50", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc9_qa0__after50", 0.0, "ANS refusal on definitive gold (1.9%)."),
    ("doc136_qa0__after50", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc24_qa0__after50", 0.0, "ANS refusal on definitive gold."),
    ("doc130_qa0__after50", 1.0, "ACK honest refusal — doc130 not yet ingested."),
    ("doc11_qa0__after50", 0.0, "ANS refusal on definitive gold."),
    ("doc35_qa0__after50", 0.0, "ANS gold AMD cashflow from Operations definitive; predicted refusal — refusal on definitive gold."),
    ("doc29_qa0__after50", 0.0, "ANS refusal on definitive gold (flat real growth)."),
    ("doc53_qa0__after50", 1.0, "ACK honest refusal — doc53 not yet ingested."),
    # 510-519
    ("doc52_qa0__after51", 1.0, "ACK calibration: 'Operating activities most cash Best Buy FY23' — correct by inference."),
    ("doc122_qa0__after51", 0.25, "ACK '0' confident wrong."),
    ("doc128_qa0__after51", 1.0, "ACK honest refusal — doc128 not yet ingested."),
    ("doc53_qa0__after51", 1.0, "ACK honest refusal — doc53 not yet ingested."),
    ("doc104_qa0__after51", 1.0, "ACK honest refusal — doc104 not yet ingested."),
    ("doc98_qa0__after51", 1.0, "ACK honest refusal — doc98 not yet ingested."),
    ("doc17_qa0__after51", 0.0, "ANS refusal on definitive gold."),
    ("doc77_qa0__after51", 1.0, "ACK honest refusal — doc77 not yet ingested."),
    ("doc136_qa0__after51", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc61_qa0__after51", 1.0, "ACK honest refusal — doc61 not yet ingested."),
    # 520-529
    ("doc137_qa0__after52", 1.0, "ACK honest refusal — doc137 not yet ingested."),
    ("doc30_qa0__after52", 0.0, "ANS refusal on definitive gold."),
    ("doc54_qa0__after52", 1.0, "ACK honest refusal — doc54 not yet ingested."),
    ("doc53_qa0__after52", 1.0, "ACK honest refusal — doc53 not yet ingested."),
    ("doc80_qa0__after52", 1.0, "ACK honest refusal — doc80 not yet ingested."),
    ("doc36_qa0__after52", 0.0, "ANS refusal on definitive gold (Data Center)."),
    ("doc121_qa0__after52", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc125_qa0__after52", 1.0, "ACK 'proposal not approved' = 'defeated' — correct."),
    ("doc136_qa0__after52", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc35_qa0__after52", 0.0, "ANS gold AMD cashflow from Operations; predicted same direction — match. Actually 'cash flow from operating activities brought in the most cash for AMD in FY22' matches gold."),
    # Actually re-check 529 — that predicted matches gold direction → 1.0
    # Let me correct:
    # 530-539
    ("doc94_qa0__after53", 1.0, "ACK honest refusal — doc94 not yet ingested."),
    ("doc36_qa0__after53", 0.0, "ANS refusal on definitive gold."),
    ("doc56_qa0__after53", 1.0, "ACK honest refusal — doc56 not yet ingested."),
    ("doc29_qa0__after53", 0.0, "ANS refusal on definitive gold."),
    ("doc139_qa0__after53", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc15_qa0__after53", 1.0, "ANS 0 — exact."),
    ("doc0_qa0__after53", 0.0, "ANS refusal on definitive gold."),
    ("doc78_qa0__after53", 1.0, "ACK honest refusal — doc78 not yet ingested."),
    ("doc50_qa0__after53", 0.0, "ANS gold consistent margins; predicted 'fluctuated >2%' — wrong direction."),
    ("doc145_qa0__after53", 1.0, "ACK honest refusal — doc145 not yet ingested."),
    # 540-549
    ("doc63_qa0__after54", 1.0, "ACK honest refusal — doc63 not yet ingested."),
    ("doc0_qa0__after54", 0.0, "ANS refusal on definitive gold."),
    ("doc134_qa0__after54", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc80_qa0__after54", 1.0, "ACK honest refusal — doc80 not yet ingested."),
    ("doc133_qa0__after54", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc29_qa0__after54", 0.0, "ANS refusal on definitive gold."),
    ("doc42_qa0__after54", 0.25, "ANS gold 24.6%→21.6% AMEX; predicted '24.8%→26.1%' — confident wrong specifics (also wrong direction)."),
    ("doc83_qa0__after54", 1.0, "ACK honest refusal — doc83 not yet ingested."),
    ("doc137_qa0__after54", 1.0, "ACK honest refusal — doc137 not yet ingested."),
    ("doc92_qa0__after54", 1.0, "ACK honest refusal — doc92 not yet ingested."),
    # 550-559
    ("doc147_qa0__after55", 1.0, "ACK honest refusal — doc147 not yet ingested."),
    ("doc108_qa0__after55", 1.0, "ACK honest refusal — doc108 not yet ingested."),
    ("doc100_qa0__after55", 1.0, "ACK honest refusal — doc100 not yet ingested."),
    ("doc37_qa0__after55", 0.0, "ANS refusal on definitive gold (16% customer)."),
    ("doc50_qa0__after55", 0.0, "ANS gold consistent margins; predicted 'fluctuated >2%' — wrong direction."),
    ("doc92_qa0__after55", 1.0, "ACK honest refusal — doc92 not yet ingested."),
    ("doc53_qa0__after55", 1.0, "ANS gold ~42% decline; predicted $1,874M→$1,093M (41.7%) — within tolerance."),
    ("doc29_qa0__after55", 0.0, "ANS refusal on definitive gold."),
    ("doc120_qa0__after55", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc128_qa0__after55", 1.0, "ACK honest refusal — doc128 not yet ingested."),
    # 560-567
    ("doc3_qa0__after56", 0.0, "ANS refusal on definitive gold."),
    ("doc22_qa0__after56", 0.0, "ANS refusal on definitive gold (Amcor 8K)."),
    ("doc116_qa0__after56", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    ("doc141_qa0__after56", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    ("doc14_qa0__after56", 0.0, "ANS refusal on definitive gold."),
    ("doc88_qa0__after56", 1.0, "ACK honest refusal — doc88 not yet ingested."),
    ("doc148_qa0__after56", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc60_qa0__after56", 1.0, "ACK honest refusal — doc60 not yet ingested."),
]

# Note: entry 529 (doc35_qa0__after52) — I judged it 0.0 by mistake but PRED matches gold.
# However the PRED is "cash flow from operating activities brought in the most cash for AMD in FY22"
# which matches gold "AMD brought in the most cashflow from Operations". This should be 1.0.
# Fix below — replace with corrected list
for i, (sfx, sc, rat) in enumerate(JUDGMENTS):
    if sfx == "doc35_qa0__after52":
        JUDGMENTS[i] = (sfx, 1.0, "ANS gold AMD cashflow from Operations; predicted 'operating activities brought in most cash AMD FY22' — match.")
        break


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
