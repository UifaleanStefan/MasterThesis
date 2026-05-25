"""Claude manual judging — Phase 1.9 FB calibration v4t-canonical (entries 198-339).

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
    # 198-199 (missed from part1)
    ("doc119_qa0__after19", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    ("doc138_qa0__after19", 0.25, "ACK calibration: confident wrong-detail 'improved operating efficiencies' (gold says specific 'lower marketing + leverage of incentive comp due to higher sales')."),
    # 200-209
    ("doc105_qa0__after20", 1.0, "ACK honest refusal — doc105 not yet ingested."),
    ("doc74_qa0__after20", 0.25, "ACK calibration: confident wrong $52,694M Costco FY21 total assets (gold $59,268)."),
    ("doc84_qa0__after20", 1.0, "ACK honest refusal — doc84 not yet ingested."),
    ("doc36_qa0__after20", 1.0, "ACK honest refusal — doc36 not yet ingested."),
    ("doc83_qa0__after20", 1.0, "ACK honest refusal — doc83 not yet ingested."),
    ("doc19_qa0__after20", 1.0, "ANS gold 30.8%; predicted 30.8% — exact."),
    ("doc140_qa0__after20", 1.0, "ACK honest refusal — doc140 not yet ingested."),
    ("doc61_qa0__after20", 1.0, "ACK honest refusal — doc61 not yet ingested."),
    ("doc111_qa0__after20", 1.0, "ACK honest refusal — doc111 not yet ingested."),
    ("doc18_qa0__after20", 0.5, "ANS gold 93.86 DPO Amazon; predicted starts DPO calc framework with COGS $111,934M but truncated — hedged, no final number."),
    # 210-219
    ("doc122_qa0__after21", 0.25, "ACK '0' confident wrong (gold $411M)."),
    ("doc113_qa0__after21", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc91_qa0__after21", 1.0, "ACK honest refusal — doc91 not yet ingested."),
    ("doc11_qa0__after21", 1.0, "ANS gold 65.4%; predicted 65.2% — within tolerance."),
    ("doc110_qa0__after21", 1.0, "ACK honest refusal — doc110 not yet ingested."),
    ("doc140_qa0__after21", 1.0, "ACK honest refusal — doc140 not yet ingested."),
    ("doc63_qa0__after21", 0.5, "ACK calibration: 'commercial airlines, government agencies, defense contractors' — partial (commercial airlines correct, gov agencies generic, no US govt 40%)."),
    ("doc48_qa0__after21", 1.0, "ACK honest refusal — doc48 not yet ingested."),
    ("doc87_qa0__after21", 1.0, "ACK honest refusal — doc87 not yet ingested."),
    ("doc68_qa0__after21", 1.0, "ACK honest refusal — doc68 not yet ingested."),
    # 220-229
    ("doc120_qa0__after22", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc114_qa0__after22", 1.0, "ACK honest refusal — doc114 not yet ingested."),
    ("doc99_qa0__after22", 1.0, "ACK honest refusal — doc99 not yet ingested."),
    ("doc80_qa0__after22", 1.0, "ACK honest refusal — doc80 not yet ingested."),
    ("doc45_qa0__after22", 1.0, "ACK honest refusal — doc45 not yet ingested."),
    ("doc68_qa0__after22", 1.0, "ACK honest refusal — doc68 not yet ingested."),
    ("doc53_qa0__after22", 1.0, "ACK honest refusal — doc53 not yet ingested."),
    ("doc84_qa0__after22", 1.0, "ACK honest refusal — doc84 not yet ingested."),
    ("doc43_qa0__after22", 1.0, "ACK honest refusal — doc43 not yet ingested."),
    ("doc61_qa0__after22", 1.0, "ACK honest refusal — doc61 not yet ingested."),
    # 230-239
    ("doc48_qa0__after23", 1.0, "ACK honest refusal — doc48 not yet ingested."),
    ("doc66_qa0__after23", 1.0, "ACK honest refusal — doc66 not yet ingested."),
    ("doc63_qa0__after23", 0.5, "ACK partial — 'gov customers, DoD, commercial airline customers' (some specifics, no 40% US govt figure)."),
    ("doc113_qa0__after23", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc117_qa0__after23", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc41_qa0__after23", 1.0, "ACK honest refusal — doc41 not yet ingested."),
    ("doc11_qa0__after23", 1.0, "ANS gold 65.4%; predicted 65.3% — within tolerance."),
    ("doc128_qa0__after23", 1.0, "ACK honest refusal — doc128 not yet ingested."),
    ("doc119_qa0__after23", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    ("doc15_qa0__after23", 1.0, "ANS 0 — exact."),
    # 240-249
    ("doc125_qa0__after24", 1.0, "ACK 'proposal not approved' = 'defeated' — correct."),
    ("doc26_qa0__after24", 0.25, "ACK calibration: confident wrong-direction 'gross margin not useful for Amcor' (gold says No slight decline 0.8%, i.e. there IS a meaningful direction)."),
    ("doc1_qa0__after24", 0.0, "ANS gold $8.70; predicted 9.178B — wrong specific."),
    ("doc32_qa0__after24", 1.0, "ACK honest refusal — doc32 not yet ingested."),
    ("doc61_qa0__after24", 1.0, "ACK honest refusal — doc61 not yet ingested."),
    ("doc126_qa0__after24", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    ("doc134_qa0__after24", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc53_qa0__after24", 1.0, "ACK honest refusal — doc53 not yet ingested."),
    ("doc120_qa0__after24", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc135_qa0__after24", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    # 250-259
    ("doc59_qa0__after25", 1.0, "ACK honest refusal — doc59 not yet ingested."),
    ("doc139_qa0__after25", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc134_qa0__after25", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc83_qa0__after25", 1.0, "ACK honest refusal — doc83 not yet ingested."),
    ("doc31_qa0__after25", 1.0, "ACK honest refusal — doc31 not yet ingested."),
    ("doc11_qa0__after25", 1.0, "ANS gold 65.4%; predicted 65.3% — within tolerance."),
    ("doc26_qa0__after25", 0.25, "ACK calibration: confident wrong-direction 'gross margin not useful for Amcor'."),
    ("doc94_qa0__after25", 1.0, "ACK honest refusal — doc94 not yet ingested."),
    ("doc2_qa0__after25", 0.0, "ANS gold 'No efficient'; predicted 'Yes capital-intensive' — wrong direction."),
    ("doc49_qa0__after25", 1.0, "ACK honest refusal — doc49 not yet ingested."),
    # 260-269
    ("doc36_qa0__after26", 1.0, "ACK honest refusal — doc36 not yet ingested."),
    ("doc131_qa0__after26", 1.0, "ACK honest refusal — doc131 not yet ingested."),
    ("doc115_qa0__after26", 1.0, "ACK honest refusal — doc115 not yet ingested."),
    ("doc85_qa0__after26", 1.0, "ACK honest refusal — doc85 not yet ingested."),
    ("doc118_qa0__after26", 1.0, "ACK honest refusal — doc118 not yet ingested."),
    ("doc77_qa0__after26", 1.0, "ACK honest refusal — doc77 not yet ingested."),
    ("doc110_qa0__after26", 1.0, "ACK honest refusal — doc110 not yet ingested."),
    ("doc63_qa0__after26", 0.5, "ACK partial — commercial airlines + gov + defense (some specifics, no 40%)."),
    ("doc40_qa0__after26", 1.0, "ACK honest refusal — doc40 not yet ingested."),
    ("doc74_qa0__after26", 1.0, "ACK honest refusal — doc74 not yet ingested."),
    # 270-279
    ("doc102_qa0__after27", 1.0, "ACK honest refusal — doc102 not yet ingested."),
    ("doc124_qa0__after27", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc39_qa0__after27", 1.0, "ACK honest refusal — doc39 not yet ingested."),
    ("doc105_qa0__after27", 1.0, "ACK honest refusal — doc105 not yet ingested."),
    ("doc132_qa0__after27", 1.0, "ACK honest refusal — doc132 not yet ingested."),
    ("doc20_qa0__after27", 0.0, "ANS gold $11,588 Amazon FY2019 net income definitive; predicted 'do not contain' — refusal on definitive gold."),
    ("doc106_qa0__after27", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc80_qa0__after27", 1.0, "ACK honest refusal — doc80 not yet ingested."),
    ("doc0_qa0__after27", 0.0, "ANS gold $1,577 3M FY2018 capex definitive; predicted 'do not provide' — refusal on definitive gold."),
    ("doc104_qa0__after27", 1.0, "ACK honest refusal — doc104 not yet ingested."),
    # 280-289
    ("doc89_qa0__after28", 1.0, "ACK honest refusal — doc89 not yet ingested."),
    ("doc63_qa0__after28", 0.5, "ACK partial — commercial airlines + gov + defense."),
    ("doc41_qa0__after28", 1.0, "ACK calibration: 'gross margin not useful for AMEX' — correct by inference."),
    ("doc29_qa0__after28", 1.0, "ACK honest refusal — doc29 not yet ingested."),
    ("doc124_qa0__after28", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc109_qa0__after28", 1.0, "ACK honest refusal — doc109 not yet ingested."),
    ("doc106_qa0__after28", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc39_qa0__after28", 1.0, "ACK honest refusal — doc39 not yet ingested."),
    ("doc56_qa0__after28", 1.0, "ACK honest refusal — doc56 not yet ingested."),
    ("doc70_qa0__after28", 1.0, "ACK honest refusal — doc70 not yet ingested."),
    # 290-299
    ("doc147_qa0__after29", 1.0, "ACK honest refusal — doc147 not yet ingested."),
    ("doc135_qa0__after29", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc124_qa0__after29", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc97_qa0__after29", 1.0, "ACK honest refusal — doc97 not yet ingested."),
    ("doc58_qa0__after29", 1.0, "ACK honest refusal — doc58 not yet ingested."),
    ("doc91_qa0__after29", 1.0, "ACK honest refusal — doc91 not yet ingested."),
    ("doc138_qa0__after29", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc108_qa0__after29", 1.0, "ACK honest refusal — doc108 not yet ingested."),
    ("doc71_qa0__after29", 1.0, "ACK honest refusal — doc71 not yet ingested."),
    ("doc18_qa0__after29", 0.0, "ANS gold 93.86 DPO Amazon definitive; predicted 'cannot be determined' — refusal on definitive gold."),
    # 300-309
    ("doc12_qa0__after30", 0.0, "ANS gold 0.83 OCF ratio definitive; predicted refusal — refusal on definitive gold."),
    ("doc98_qa0__after30", 1.0, "ACK honest refusal — doc98 not yet ingested."),
    ("doc47_qa0__after30", 1.0, "ACK honest refusal — doc47 not yet ingested."),
    ("doc97_qa0__after30", 1.0, "ACK honest refusal — doc97 not yet ingested."),
    ("doc52_qa0__after30", 1.0, "ACK honest refusal — doc52 not yet ingested."),
    ("doc0_qa0__after30", 0.0, "ANS gold $1,577 definitive; predicted refusal — refusal on definitive gold."),
    ("doc60_qa0__after30", 1.0, "ACK honest refusal — doc60 not yet ingested."),
    ("doc5_qa0__after30", 0.0, "ANS gold No 3M quick ratio 0.96 definitive; predicted refusal — refusal on definitive gold."),
    ("doc42_qa0__after30", 1.0, "ACK honest refusal — doc42 not yet ingested."),
    ("doc90_qa0__after30", 1.0, "ACK 'Consumer Health discontinued Aug 30, 2023' — correct by inference."),
    # 310-319
    ("doc124_qa0__after31", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc91_qa0__after31", 1.0, "ACK honest refusal — doc91 not yet ingested."),
    ("doc21_qa0__after31", 0.0, "ANS gold $1,616 Amcor FY20 AR definitive; predicted 'do not contain' — refusal on definitive gold."),
    ("doc63_qa0__after31", 0.5, "ACK partial — commercial airlines + gov + defense."),
    ("doc120_qa0__after31", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc67_qa0__after31", 1.0, "ACK honest refusal — doc67 not yet ingested."),
    ("doc139_qa0__after31", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc18_qa0__after31", 0.0, "ANS refusal on definitive gold (DPO Amazon)."),
    ("doc135_qa0__after31", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc141_qa0__after31", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    # 320-329
    ("doc117_qa0__after32", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc18_qa0__after32", 0.0, "ANS refusal on definitive gold."),
    ("doc7_qa0__after32", 1.0, "ANS Yes 65th — match."),
    ("doc115_qa0__after32", 1.0, "ACK honest refusal — doc115 not yet ingested."),
    ("doc47_qa0__after32", 1.0, "ACK honest refusal — doc47 not yet ingested."),
    ("doc106_qa0__after32", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc87_qa0__after32", 1.0, "ACK honest refusal — doc87 not yet ingested."),
    ("doc56_qa0__after32", 1.0, "ACK honest refusal — doc56 not yet ingested."),
    ("doc77_qa0__after32", 1.0, "ACK honest refusal — doc77 not yet ingested."),
    ("doc112_qa0__after32", 1.0, "ACK honest refusal — doc112 not yet ingested."),
    # 330-339
    ("doc135_qa0__after33", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc144_qa0__after33", 1.0, "ACK honest refusal — doc144 not yet ingested."),
    ("doc18_qa0__after33", 0.0, "ANS refusal on definitive gold (DPO Amazon)."),
    ("doc34_qa0__after33", 0.25, "ACK calibration: confident wrong-direction 'operating margin not useful for AMD' (gold gives specific Xilinx amortization reason)."),
    ("doc72_qa0__after33", 1.0, "ACK honest refusal — doc72 not yet ingested."),
    ("doc15_qa0__after33", 1.0, "ANS 0 — exact."),
    ("doc90_qa0__after33", 1.0, "ACK Consumer Health discontinued — correct by inference."),
    ("doc89_qa0__after33", 1.0, "ACK honest refusal — doc89 not yet ingested."),
    ("doc64_qa0__after33", 1.0, "ACK honest refusal — doc64 not yet ingested."),
    ("doc125_qa0__after33", 1.0, "ACK honest refusal — doc125 not yet ingested."),
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
