"""Claude manual judging — Phase 1.9 FB calibration v4t-canonical (entries 340-485).

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
    # 340-349
    ("doc130_qa0__after34", 1.0, "ACK honest refusal — doc130 not yet ingested."),
    ("doc26_qa0__after34", 0.25, "ANS gold 'No slight decline 0.8%'; predicted 'gross margin not useful' — wrong-direction calibration."),
    ("doc68_qa0__after34", 1.0, "ACK honest refusal — doc68 not yet ingested."),
    ("doc40_qa0__after34", 1.0, "ACK honest refusal — doc40 not yet ingested."),
    ("doc129_qa0__after34", 1.0, "ACK honest refusal — doc129 not yet ingested."),
    ("doc144_qa0__after34", 1.0, "ACK honest refusal — doc144 not yet ingested."),
    ("doc25_qa0__after34", 1.0, "ANS Amcor packaging — match."),
    ("doc34_qa0__after34", 1.0, "ANS Xilinx amortization — match."),
    ("doc131_qa0__after34", 1.0, "ACK honest refusal — doc131 not yet ingested."),
    ("doc29_qa0__after34", 0.0, "ANS gold 'flat real growth'; predicted 'decrease of 5%' — wrong direction."),
    # 350-359
    ("doc136_qa0__after35", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc93_qa0__after35", 1.0, "ACK honest refusal — doc93 not yet ingested."),
    ("doc146_qa0__after35", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc149_qa0__after35", 1.0, "ACK honest refusal — doc149 not yet ingested."),
    ("doc42_qa0__after35", 1.0, "ACK honest refusal — doc42 not yet ingested."),
    ("doc85_qa0__after35", 1.0, "ACK honest refusal — doc85 not yet ingested."),
    ("doc98_qa0__after35", 1.0, "ACK honest refusal — doc98 not yet ingested."),
    ("doc92_qa0__after35", 1.0, "ACK honest refusal — doc92 not yet ingested."),
    ("doc78_qa0__after35", 1.0, "ACK honest refusal — doc78 not yet ingested."),
    ("doc100_qa0__after35", 1.0, "ACK honest refusal — doc100 not yet ingested."),
    # 360-369
    ("doc88_qa0__after36", 1.0, "ACK honest refusal — doc88 not yet ingested."),
    ("doc69_qa0__after36", 1.0, "ACK honest refusal — doc69 not yet ingested."),
    ("doc120_qa0__after36", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc112_qa0__after36", 1.0, "ACK honest refusal — doc112 not yet ingested."),
    ("doc133_qa0__after36", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc136_qa0__after36", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc145_qa0__after36", 1.0, "ACK honest refusal — doc145 not yet ingested."),
    ("doc131_qa0__after36", 1.0, "ACK honest refusal — doc131 not yet ingested."),
    ("doc31_qa0__after36", 0.0, "ANS gold quick ratio 1.57 definitive; predicted 'not mentioned' — refusal on definitive gold."),
    ("doc3_qa0__after36", 0.0, "ANS gold -1.7% reasons definitive; predicted refusal — refusal on definitive gold."),
    # 370-379
    ("doc52_qa0__after37", 1.0, "ACK honest refusal — doc52 not yet ingested."),
    ("doc70_qa0__after37", 1.0, "ACK honest refusal — doc70 not yet ingested."),
    ("doc11_qa0__after37", 1.0, "ANS gold 65.4%; predicted 65.3% — within tolerance."),
    ("doc10_qa0__after37", 0.0, "ANS gold 0.66 Adobe FY15 OCF ratio definitive; predicted refusal — refusal on definitive gold."),
    ("doc90_qa0__after37", 1.0, "ACK Consumer Health discontinued — correct by inference."),
    ("doc54_qa0__after37", 1.0, "ACK honest refusal — doc54 not yet ingested."),
    ("doc50_qa0__after37", 1.0, "ACK honest refusal — doc50 not yet ingested."),
    ("doc107_qa0__after37", 1.0, "ACK honest refusal — doc107 not yet ingested."),
    ("doc129_qa0__after37", 1.0, "ACK honest refusal — doc129 not yet ingested."),
    ("doc108_qa0__after37", 1.0, "ACK honest refusal — doc108 not yet ingested."),
    # 380-389
    ("doc90_qa0__after38", 1.0, "ACK Consumer Health discontinued — correct by inference."),
    ("doc138_qa0__after38", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc43_qa0__after38", 1.0, "ACK honest refusal — doc43 not yet ingested."),
    ("doc71_qa0__after38", 1.0, "ACK honest refusal — doc71 not yet ingested."),
    ("doc1_qa0__after38", 0.0, "ANS gold $8.70 3M FY2018 net PPNE definitive; predicted refusal — refusal on definitive gold."),
    ("doc27_qa0__after38", 0.5, "ANS partial — restructuring employee/fixed-asset breakdown without 87%."),
    ("doc140_qa0__after38", 1.0, "ACK honest refusal — doc140 not yet ingested."),
    ("doc24_qa0__after38", 0.0, "ANS Amcor acquisitions definitive; predicted refusal — refusal on definitive gold."),
    ("doc135_qa0__after38", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc88_qa0__after38", 1.0, "ACK honest refusal — doc88 not yet ingested."),
    # 390-399
    ("doc115_qa0__after39", 1.0, "ACK honest refusal — doc115 not yet ingested."),
    ("doc92_qa0__after39", 1.0, "ACK honest refusal — doc92 not yet ingested."),
    ("doc146_qa0__after39", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc76_qa0__after39", 1.0, "ACK honest refusal — doc76 not yet ingested."),
    ("doc80_qa0__after39", 1.0, "ACK honest refusal — doc80 not yet ingested."),
    ("doc8_qa0__after39", 0.0, "ANS gold 24.26 Activision Blizzard FY19 fixed asset turnover; predicted refusal — refusal on definitive gold."),
    ("doc33_qa0__after39", 1.0, "ANS AMD FY22 64% Data Center EPYC + 21% Gaming + Embedded — match gold."),
    ("doc95_qa0__after39", 1.0, "ACK honest refusal — doc95 not yet ingested."),
    ("doc46_qa0__after39", 1.0, "ACK honest refusal — doc46 not yet ingested."),
    ("doc2_qa0__after39", 0.0, "ANS gold 'No efficient' definitive; predicted refusal — refusal on definitive gold."),
    # 400-409
    ("doc16_qa0__after40", 0.5, "ANS gold 9.5 turnover; predicted hedged 'cannot be calculated' with additional note about unconventional inventory — refusal but with hedging."),
    ("doc93_qa0__after40", 1.0, "ACK honest refusal — doc93 not yet ingested."),
    ("doc128_qa0__after40", 1.0, "ACK honest refusal — doc128 not yet ingested."),
    ("doc110_qa0__after40", 1.0, "ACK honest refusal — doc110 not yet ingested."),
    ("doc59_qa0__after40", 1.0, "ACK honest refusal — doc59 not yet ingested."),
    ("doc54_qa0__after40", 1.0, "ACK honest refusal — doc54 not yet ingested."),
    ("doc135_qa0__after40", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc11_qa0__after40", 0.0, "ANS gold 65.4% definitive; predicted refusal — refusal on definitive gold."),
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
    ("doc21_qa0__after41", 0.0, "ANS gold $1,616 Amcor FY20 AR definitive; predicted refusal — refusal on definitive gold."),
    ("doc87_qa0__after41", 1.0, "ACK honest refusal — doc87 not yet ingested."),
    # 420-429
    ("doc106_qa0__after42", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc124_qa0__after42", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc98_qa0__after42", 1.0, "ACK honest refusal — doc98 not yet ingested."),
    ("doc56_qa0__after42", 1.0, "ACK honest refusal — doc56 not yet ingested."),
    ("doc36_qa0__after42", 0.0, "ANS gold Data Center; predicted 'Gaming segment 21% growth' — wrong segment."),
    ("doc51_qa0__after42", 1.0, "ACK honest refusal — doc51 not yet ingested."),
    ("doc111_qa0__after42", 1.0, "ACK honest refusal — doc111 not yet ingested."),
    ("doc60_qa0__after42", 1.0, "ACK honest refusal — doc60 not yet ingested."),
    ("doc148_qa0__after42", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc50_qa0__after42", 1.0, "ACK honest refusal — doc50 not yet ingested."),
    # 430-439
    ("doc25_qa0__after43", 1.0, "ANS Amcor packaging — match."),
    ("doc114_qa0__after43", 1.0, "ACK honest refusal — doc114 not yet ingested."),
    ("doc133_qa0__after43", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc141_qa0__after43", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    ("doc55_qa0__after43", 1.0, "ACK honest refusal — doc55 not yet ingested."),
    ("doc85_qa0__after43", 1.0, "ACK honest refusal — doc85 not yet ingested."),
    ("doc27_qa0__after43", 0.5, "ANS partial — restructuring breakdown without 87%."),
    ("doc94_qa0__after43", 1.0, "ACK honest refusal — doc94 not yet ingested."),
    ("doc122_qa0__after43", 0.25, "ACK '0' confident wrong."),
    ("doc24_qa0__after43", 0.0, "ANS Amcor acquisitions definitive; predicted refusal — refusal on definitive gold."),
    # 440-449
    ("doc76_qa0__after44", 1.0, "ACK honest refusal — doc76 not yet ingested."),
    ("doc35_qa0__after44", 1.0, "ANS 'operating activities most cash' — match."),
    ("doc17_qa0__after44", 0.0, "ANS gold -0.02 AES ROA definitive; predicted refusal — refusal on definitive gold."),
    ("doc30_qa0__after44", 0.0, "ANS gold 4.2% definitive; predicted refusal — refusal on definitive gold."),
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
    ("doc11_qa0__after45", 1.0, "ANS gold 65.4%; predicted 65.3% — within tolerance."),
    ("doc109_qa0__after45", 1.0, "ACK honest refusal — doc109 not yet ingested."),
    ("doc59_qa0__after45", 1.0, "ACK honest refusal — doc59 not yet ingested."),
    ("doc57_qa0__after45", 1.0, "ACK honest refusal — doc57 not yet ingested."),
    ("doc30_qa0__after45", 0.0, "ANS gold 4.2% definitive; predicted refusal — refusal on definitive gold."),
    ("doc32_qa0__after45", 1.0, "ANS AMD products — match."),
    ("doc31_qa0__after45", 0.0, "ANS gold quick ratio 1.57 definitive; predicted refusal — refusal on definitive gold."),
    # 460-469
    ("doc99_qa0__after46", 1.0, "ACK honest refusal — doc99 not yet ingested."),
    ("doc37_qa0__after46", 1.0, "ANS Yes one customer 16% — match."),
    ("doc54_qa0__after46", 1.0, "ACK honest refusal — doc54 not yet ingested."),
    ("doc118_qa0__after46", 1.0, "ACK honest refusal — doc118 not yet ingested."),
    ("doc58_qa0__after46", 1.0, "ACK honest refusal — doc58 not yet ingested."),
    ("doc24_qa0__after46", 0.0, "ANS Amcor acquisitions refusal on definitive gold."),
    ("doc30_qa0__after46", 0.0, "ANS gold 4.2% definitive; predicted refusal."),
    ("doc50_qa0__after46", 1.0, "ACK honest refusal — doc50 not yet ingested."),
    ("doc148_qa0__after46", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc95_qa0__after46", 1.0, "ACK honest refusal — doc95 not yet ingested."),
    # 470-479
    ("doc1_qa0__after47", 0.0, "ANS gold $8.70 definitive; predicted refusal — refusal on definitive gold."),
    ("doc75_qa0__after47", 1.0, "ACK honest refusal — doc75 not yet ingested."),
    ("doc92_qa0__after47", 1.0, "ACK honest refusal — doc92 not yet ingested."),
    ("doc87_qa0__after47", 1.0, "ACK honest refusal — doc87 not yet ingested."),
    ("doc93_qa0__after47", 1.0, "ACK honest refusal — doc93 not yet ingested."),
    ("doc78_qa0__after47", 1.0, "ACK honest refusal — doc78 not yet ingested."),
    ("doc97_qa0__after47", 1.0, "ACK honest refusal — doc97 not yet ingested."),
    ("doc49_qa0__after47", 1.0, "ACK honest refusal — doc49 not yet ingested."),
    ("doc136_qa0__after47", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc31_qa0__after47", 0.0, "ANS quick ratio refusal on definitive gold."),
    # 480-485
    ("doc125_qa0__after48", 1.0, "ACK 'proposal not approved' = 'defeated' — correct."),
    ("doc4_qa0__after48", 0.0, "ANS gold consumer shrunk 0.9% definitive; predicted refusal — refusal on definitive gold."),
    ("doc58_qa0__after48", 1.0, "ACK honest refusal — doc58 not yet ingested."),
    ("doc133_qa0__after48", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc40_qa0__after48", 1.0, "ANS gold 'not measured through operating margin'; predicted 'operating margin not useful for AMEX' — correct by inference (matches gold's meaning)."),
    ("doc148_qa0__after48", 1.0, "ACK honest refusal — doc148 not yet ingested."),
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
