"""Claude manual judging — Phase 1.9 FB calibration dump-all (entries 277-422)."""

from __future__ import annotations

import json
from pathlib import Path

JUDGE_DIR = Path(
    "results/stage3/judge_queue/financebench__dump-all__calibration__seed42"
)
QID_PREFIX = "financebench__dump-all__calibration__"
QID_SUFFIX = "__seed42"

JUDGMENTS: list[tuple[str, float, str]] = [
    # 277-279
    ("doc80_qa0__after27", 1.0, "ACK honest refusal — doc80 not yet ingested."),
    ("doc0_qa0__after27", 0.0, "ANS gold $1,577 definitive; predicted refusal — refusal on definitive gold."),
    ("doc104_qa0__after27", 1.0, "ACK honest refusal — doc104 not yet ingested."),
    # 280-289
    ("doc89_qa0__after28", 1.0, "ACK honest refusal — doc89 not yet ingested."),
    ("doc63_qa0__after28", 0.5, "ACK partial 'commercial airlines + gov + defense' — no US govt 40%."),
    ("doc41_qa0__after28", 1.0, "ACK honest refusal — doc41 not yet ingested."),
    ("doc29_qa0__after28", 0.25, "ACK calibration: 'decrease of 1%' — gold says flat (0%), confident wrong specific."),
    ("doc124_qa0__after28", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc109_qa0__after28", 1.0, "ACK honest refusal — doc109 not yet ingested."),
    ("doc106_qa0__after28", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc39_qa0__after28", 1.0, "ACK honest refusal — doc39 not yet ingested."),
    ("doc56_qa0__after28", 0.25, "ACK calibration: confident wrong '1.04 Block FY2016 WC ratio' (gold 1.73)."),
    ("doc70_qa0__after28", 1.0, "ACK honest refusal — doc70 not yet ingested."),
    # 290-299
    ("doc147_qa0__after29", 1.0, "ACK honest refusal — doc147 not yet ingested."),
    ("doc135_qa0__after29", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc124_qa0__after29", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc97_qa0__after29", 1.0, "ACK honest refusal — doc97 not yet ingested."),
    ("doc58_qa0__after29", 1.0, "ACK honest refusal — doc58 not yet ingested."),
    ("doc91_qa0__after29", 0.25, "ACK calibration: '$8.9B Kenvue gain' confident wrong (gold $20B)."),
    ("doc138_qa0__after29", 0.25, "ACK calibration: vague 'operating efficiencies' (gold specific lower marketing + leverage)."),
    ("doc108_qa0__after29", 1.0, "ACK honest refusal — doc108 not yet ingested."),
    ("doc71_qa0__after29", 1.0, "ACK honest refusal — doc71 not yet ingested."),
    ("doc18_qa0__after29", 0.5, "ANS gold 93.86 Amazon DPO; predicted truncated calc framework with AP $25,309/$34,616 — hedged setup, no final number."),
    # 300-309
    ("doc12_qa0__after30", 0.0, "ANS gold 0.83 definitive; predicted refusal — refusal on definitive gold."),
    ("doc98_qa0__after30", 1.0, "ACK honest refusal — doc98 not yet ingested."),
    ("doc47_qa0__after30", 1.0, "ACK honest refusal — doc47 not yet ingested."),
    ("doc97_qa0__after30", 1.0, "ACK honest refusal — doc97 not yet ingested."),
    ("doc52_qa0__after30", 1.0, "ACK honest refusal — doc52 not yet ingested."),
    ("doc0_qa0__after30", 0.0, "ANS refusal on definitive gold."),
    ("doc60_qa0__after30", 1.0, "ACK honest refusal — doc60 not yet ingested."),
    ("doc5_qa0__after30", 0.0, "ANS gold No 3M quick ratio 0.96 definitive; predicted refusal — refusal on definitive gold."),
    ("doc42_qa0__after30", 1.0, "ACK honest refusal — doc42 not yet ingested."),
    ("doc90_qa0__after30", 1.0, "ACK Consumer Health discontinued — correct by inference."),
    # 310-319
    ("doc124_qa0__after31", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc91_qa0__after31", 1.0, "ACK honest refusal — doc91 not yet ingested."),
    ("doc21_qa0__after31", 1.0, "ANS gold $1,616 Amcor FY20 AR; predicted $1,615.9M — within tolerance."),
    ("doc63_qa0__after31", 0.5, "ACK partial 'commercial airlines + gov + defense'."),
    ("doc120_qa0__after31", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc67_qa0__after31", 1.0, "ACK honest refusal — doc67 not yet ingested."),
    ("doc139_qa0__after31", 0.25, "ACK calibration: vague 'strategic decision to invest in inventory' (gold specifies 47 new stores)."),
    ("doc18_qa0__after31", 0.0, "ANS refusal on definitive gold."),
    ("doc135_qa0__after31", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc141_qa0__after31", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    # 320-329
    ("doc117_qa0__after32", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc18_qa0__after32", 0.0, "ANS refusal on definitive gold."),
    ("doc7_qa0__after32", 0.0, "ANS gold Yes 3M 65th dividend definitive; predicted refusal — refusal on definitive gold."),
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
    ("doc18_qa0__after33", 0.0, "ANS refusal on definitive gold."),
    ("doc34_qa0__after33", 0.25, "ACK calibration: 'operating margin not useful for AMD' confident wrong-direction (gold gives Xilinx amortization reason)."),
    ("doc72_qa0__after33", 1.0, "ACK honest refusal — doc72 not yet ingested."),
    ("doc15_qa0__after33", 1.0, "ANS 0 — exact."),
    ("doc90_qa0__after33", 1.0, "ACK Consumer Health discontinued — correct by inference."),
    ("doc89_qa0__after33", 1.0, "ACK honest refusal — doc89 not yet ingested."),
    ("doc64_qa0__after33", 1.0, "ACK honest refusal — doc64 not yet ingested."),
    ("doc125_qa0__after33", 1.0, "ACK 'proposal not approved with 66% against' — correct."),
    # 340-349
    ("doc130_qa0__after34", 1.0, "ACK honest refusal — doc130 not yet ingested."),
    ("doc26_qa0__after34", 0.75, "ANS gold 'No slight decline 0.8%'; predicted 'gross profit $2,820→$2,725 declining' — direction right, no 0.8% figure."),
    ("doc68_qa0__after34", 1.0, "ACK honest refusal — doc68 not yet ingested."),
    ("doc40_qa0__after34", 1.0, "ACK honest refusal — doc40 not yet ingested."),
    ("doc129_qa0__after34", 1.0, "ACK honest refusal — doc129 not yet ingested."),
    ("doc144_qa0__after34", 1.0, "ACK honest refusal — doc144 not yet ingested."),
    ("doc25_qa0__after34", 1.0, "ANS gold Amcor packaging; predicted same with detail — match."),
    ("doc34_qa0__after34", 1.0, "ANS Xilinx amortization AMD operating income — exact match."),
    ("doc131_qa0__after34", 1.0, "ACK honest refusal — doc131 not yet ingested."),
    ("doc29_qa0__after34", 0.0, "ANS gold flat real growth; predicted decrease 5% — wrong direction."),
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
    ("doc31_qa0__after36", 0.0, "ANS refusal on definitive gold (quick ratio 1.57)."),
    ("doc3_qa0__after36", 0.0, "ANS refusal on definitive gold (3M operating margin reasons)."),
    # 370-379
    ("doc52_qa0__after37", 1.0, "ACK calibration: 'cash flow from operating activities Best Buy FY23' — correct by inference."),
    ("doc70_qa0__after37", 1.0, "ACK honest refusal — doc70 not yet ingested."),
    ("doc11_qa0__after37", 0.0, "ANS refusal on definitive gold (65.4%)."),
    ("doc10_qa0__after37", 0.0, "ANS refusal on definitive gold."),
    ("doc90_qa0__after37", 1.0, "ACK Consumer Health discontinued — correct by inference."),
    ("doc54_qa0__after37", 1.0, "ACK honest refusal — doc54 not yet ingested."),
    ("doc50_qa0__after37", 1.0, "ACK honest refusal — doc50 not yet ingested."),
    ("doc107_qa0__after37", 1.0, "ACK honest refusal — doc107 not yet ingested."),
    ("doc129_qa0__after37", 1.0, "ACK honest refusal — doc129 not yet ingested."),
    ("doc108_qa0__after37", 1.0, "ACK honest refusal — doc108 not yet ingested."),
    # 380-389
    ("doc90_qa0__after38", 1.0, "ACK Consumer Health discontinued — correct by inference."),
    ("doc138_qa0__after38", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc43_qa0__after38", 0.25, "ACK calibration: '$6,369M current liabilities' confident wrong (gold Customer deposits)."),
    ("doc71_qa0__after38", 1.0, "ACK honest refusal — doc71 not yet ingested."),
    ("doc1_qa0__after38", 0.0, "ANS refusal on definitive gold ($8.70)."),
    ("doc27_qa0__after38", 0.0, "ANS refusal on definitive gold (87% restructuring)."),
    ("doc140_qa0__after38", 1.0, "ACK honest refusal — doc140 not yet ingested."),
    ("doc24_qa0__after38", 0.0, "ANS refusal on definitive gold (Amcor acquisitions)."),
    ("doc135_qa0__after38", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc88_qa0__after38", 1.0, "ACK honest refusal — doc88 not yet ingested."),
    # 390-399
    ("doc115_qa0__after39", 1.0, "ACK honest refusal — doc115 not yet ingested."),
    ("doc92_qa0__after39", 1.0, "ACK honest refusal — doc92 not yet ingested."),
    ("doc146_qa0__after39", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc76_qa0__after39", 1.0, "ACK honest refusal — doc76 not yet ingested."),
    ("doc80_qa0__after39", 1.0, "ACK honest refusal — doc80 not yet ingested."),
    ("doc8_qa0__after39", 0.0, "ANS refusal on definitive gold (24.26)."),
    ("doc33_qa0__after39", 1.0, "ANS gold AMD FY22 EPYC + Gaming + Embedded; predicted 64% Data Center + 21% Gaming + Embedded — match."),
    ("doc95_qa0__after39", 1.0, "ACK honest refusal — doc95 not yet ingested."),
    ("doc46_qa0__after39", 1.0, "ACK honest refusal — doc46 not yet ingested."),
    ("doc2_qa0__after39", 0.0, "ANS refusal on definitive gold."),
    # 400-409
    ("doc16_qa0__after40", 0.0, "ANS refusal on definitive gold (9.5 AES inventory turnover)."),
    ("doc93_qa0__after40", 1.0, "ACK honest refusal — doc93 not yet ingested."),
    ("doc128_qa0__after40", 1.0, "ACK honest refusal — doc128 not yet ingested."),
    ("doc110_qa0__after40", 1.0, "ACK honest refusal — doc110 not yet ingested."),
    ("doc59_qa0__after40", 1.0, "ACK honest refusal — doc59 not yet ingested."),
    ("doc54_qa0__after40", 1.0, "ACK honest refusal — doc54 not yet ingested."),
    ("doc135_qa0__after40", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc11_qa0__after40", 0.0, "ANS refusal on definitive gold (65.4%)."),
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
    ("doc21_qa0__after41", 0.0, "ANS refusal on definitive gold ($1,616)."),
    ("doc87_qa0__after41", 1.0, "ACK honest refusal — doc87 not yet ingested."),
    # 420-422
    ("doc106_qa0__after42", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc124_qa0__after42", 1.0, "ACK honest refusal — doc124 not yet ingested."),
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
