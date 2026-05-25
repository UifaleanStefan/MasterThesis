"""Claude manual judging — Phase 1.9 FB calibration dump-all (entries 146-275)."""

from __future__ import annotations

import json
from pathlib import Path

JUDGE_DIR = Path(
    "results/stage3/judge_queue/financebench__dump-all__calibration__seed42"
)
QID_PREFIX = "financebench__dump-all__calibration__"
QID_SUFFIX = "__seed42"

JUDGMENTS: list[tuple[str, float, str]] = [
    # 146-149
    ("doc3_qa0__after14", 0.0, "ANS gold -1.7% reasons definitive; predicted refusal — refusal on definitive gold."),
    ("doc90_qa0__after14", 1.0, "ACK Consumer Health discontinued — correct by inference."),
    ("doc65_qa0__after14", 1.0, "ACK honest refusal — doc65 not yet ingested."),
    ("doc140_qa0__after14", 1.0, "ACK honest refusal — doc140 not yet ingested."),
    # 150-159
    ("doc80_qa0__after15", 1.0, "ACK honest refusal — doc80 not yet ingested."),
    ("doc81_qa0__after15", 1.0, "ACK honest refusal — doc81 not yet ingested."),
    ("doc26_qa0__after15", 1.0, "ACK honest refusal — doc26 not yet ingested."),
    ("doc46_qa0__after15", 1.0, "ACK honest refusal — doc46 not yet ingested."),
    ("doc127_qa0__after15", 1.0, "ACK honest refusal — doc127 not yet ingested."),
    ("doc23_qa0__after15", 1.0, "ACK honest refusal — doc23 not yet ingested."),
    ("doc36_qa0__after15", 1.0, "ACK honest refusal — doc36 not yet ingested."),
    ("doc130_qa0__after15", 1.0, "ACK honest refusal — doc130 not yet ingested."),
    ("doc48_qa0__after15", 1.0, "ACK honest refusal — doc48 not yet ingested."),
    ("doc34_qa0__after15", 1.0, "ACK honest refusal — doc34 not yet ingested."),
    # 160-169
    ("doc71_qa0__after16", 1.0, "ACK honest refusal — doc71 not yet ingested."),
    ("doc115_qa0__after16", 1.0, "ACK honest refusal — doc115 not yet ingested."),
    ("doc138_qa0__after16", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc86_qa0__after16", 1.0, "ACK honest refusal — doc86 not yet ingested."),
    ("doc136_qa0__after16", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc145_qa0__after16", 1.0, "ACK honest refusal — doc145 not yet ingested."),
    ("doc89_qa0__after16", 1.0, "ACK honest refusal — doc89 not yet ingested."),
    ("doc105_qa0__after16", 1.0, "ACK honest refusal — doc105 not yet ingested."),
    ("doc116_qa0__after16", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    ("doc23_qa0__after16", 1.0, "ACK honest refusal — doc23 not yet ingested."),
    # 170-179
    ("doc103_qa0__after17", 1.0, "ACK honest refusal — doc103 not yet ingested."),
    ("doc73_qa0__after17", 1.0, "ACK honest refusal — doc73 not yet ingested."),
    ("doc124_qa0__after17", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc18_qa0__after17", 1.0, "ACK honest refusal — doc18 not yet ingested."),
    ("doc115_qa0__after17", 1.0, "ACK honest refusal — doc115 not yet ingested."),
    ("doc2_qa0__after17", 0.0, "ANS gold 'No efficient CAPEX'; predicted refusal — refusal on definitive gold."),
    ("doc64_qa0__after17", 1.0, "ACK honest refusal — doc64 not yet ingested."),
    ("doc85_qa0__after17", 1.0, "ACK honest refusal — doc85 not yet ingested."),
    ("doc74_qa0__after17", 1.0, "ACK honest refusal — doc74 not yet ingested."),
    ("doc33_qa0__after17", 1.0, "ACK honest refusal — doc33 not yet ingested."),
    # 180-189
    ("doc37_qa0__after18", 1.0, "ACK honest refusal — doc37 not yet ingested."),
    ("doc39_qa0__after18", 1.0, "ACK honest refusal — doc39 not yet ingested."),
    ("doc139_qa0__after18", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc34_qa0__after18", 1.0, "ACK honest refusal — doc34 not yet ingested."),
    ("doc109_qa0__after18", 1.0, "ACK honest refusal — doc109 not yet ingested."),
    ("doc4_qa0__after18", 0.0, "ANS gold consumer shrunk 0.9% definitive; predicted refusal — refusal on definitive gold."),
    ("doc49_qa0__after18", 1.0, "ACK honest refusal — doc49 not yet ingested."),
    ("doc3_qa0__after18", 0.0, "ANS gold -1.7% reasons definitive; predicted refusal — refusal on definitive gold."),
    ("doc14_qa0__after18", 0.0, "ANS gold Yes Adobe FCF improved ~13%; predicted refusal — refusal on definitive gold."),
    ("doc97_qa0__after18", 1.0, "ACK honest refusal — doc97 not yet ingested."),
    # 190-199
    ("doc136_qa0__after19", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc113_qa0__after19", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc57_qa0__after19", 1.0, "ACK honest refusal — doc57 not yet ingested."),
    ("doc59_qa0__after19", 1.0, "ACK honest refusal — doc59 not yet ingested."),
    ("doc75_qa0__after19", 1.0, "ACK honest refusal — doc75 not yet ingested."),
    ("doc36_qa0__after19", 1.0, "ACK honest refusal — doc36 not yet ingested."),
    ("doc110_qa0__after19", 1.0, "ACK honest refusal — doc110 not yet ingested."),
    ("doc51_qa0__after19", 1.0, "ACK honest refusal — doc51 not yet ingested."),
    ("doc119_qa0__after19", 0.25, "ACK calibration: confident wrong-ish '$4.2B PepsiCo FY2021 capex' (gold $4.60B). ~8.7% off."),
    ("doc138_qa0__after19", 0.25, "ACK calibration: vague 'improved operating efficiencies' (gold 'lower marketing + leverage of incentive comp')."),
    # 200-209
    ("doc105_qa0__after20", 1.0, "ACK honest refusal — doc105 not yet ingested."),
    ("doc74_qa0__after20", 1.0, "ACK calibration: '$58,942M Costco FY2021 total assets' — 0.55% off gold $59,268, within tolerance."),
    ("doc84_qa0__after20", 1.0, "ACK honest refusal — doc84 not yet ingested."),
    ("doc36_qa0__after20", 1.0, "ACK honest refusal — doc36 not yet ingested."),
    ("doc83_qa0__after20", 1.0, "ACK honest refusal — doc83 not yet ingested."),
    ("doc19_qa0__after20", 1.0, "ANS gold 30.8%; predicted 30.8% — exact!"),
    ("doc140_qa0__after20", 1.0, "ACK honest refusal — doc140 not yet ingested."),
    ("doc61_qa0__after20", 1.0, "ACK honest refusal — doc61 not yet ingested."),
    ("doc111_qa0__after20", 1.0, "ACK honest refusal — doc111 not yet ingested."),
    ("doc18_qa0__after20", 0.0, "ANS gold 93.86 Amazon DPO; predicted 36.45 — wrong specific."),
    # 210-219
    ("doc122_qa0__after21", 0.25, "ACK '0' confident wrong."),
    ("doc113_qa0__after21", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc91_qa0__after21", 0.25, "ACK calibration: confident wrong '$9.6B JnJ Consumer Health gain' (gold $20B)."),
    ("doc11_qa0__after21", 0.0, "ANS gold 65.4% definitive; predicted refusal — refusal on definitive gold."),
    ("doc110_qa0__after21", 1.0, "ACK honest refusal — doc110 not yet ingested."),
    ("doc140_qa0__after21", 1.0, "ACK honest refusal — doc140 not yet ingested."),
    ("doc63_qa0__after21", 0.5, "ACK 'commercial airlines, gov agencies, defense space security' — partial (some specifics, no US govt 40%)."),
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
    ("doc43_qa0__after22", 0.25, "ACK calibration: 'Long-term debt' confident wrong (gold Customer deposits)."),
    ("doc61_qa0__after22", 1.0, "ACK honest refusal — doc61 not yet ingested."),
    # 230-239
    ("doc48_qa0__after23", 1.0, "ACK honest refusal — doc48 not yet ingested."),
    ("doc66_qa0__after23", 1.0, "ACK honest refusal — doc66 not yet ingested."),
    ("doc63_qa0__after23", 0.5, "ACK partial 'commercial airlines, gov, defense' — no US govt 40%."),
    ("doc113_qa0__after23", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc117_qa0__after23", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc41_qa0__after23", 1.0, "ACK honest refusal — doc41 not yet ingested."),
    ("doc11_qa0__after23", 0.0, "ANS refusal on definitive gold."),
    ("doc128_qa0__after23", 1.0, "ACK honest refusal — doc128 not yet ingested."),
    ("doc119_qa0__after23", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    ("doc15_qa0__after23", 1.0, "ANS 0 — exact."),
    # 240-249
    ("doc125_qa0__after24", 1.0, "ACK 'proposal not approved with 66% against' — correct + extra detail."),
    ("doc26_qa0__after24", 0.25, "ACK 'gross margin not useful for Amcor' confident wrong direction (gold says No slight decline 0.8%, has meaningful direction)."),
    ("doc1_qa0__after24", 0.0, "ANS gold $8.70; predicted refusal — refusal on definitive gold."),
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
    ("doc11_qa0__after25", 0.0, "ANS refusal on definitive gold."),
    ("doc26_qa0__after25", 0.25, "ACK 'gross margin not useful for Amcor' confident wrong direction."),
    ("doc94_qa0__after25", 1.0, "ACK honest refusal — doc94 not yet ingested."),
    ("doc2_qa0__after25", 0.0, "ANS refusal on definitive gold."),
    ("doc49_qa0__after25", 1.0, "ACK honest refusal — doc49 not yet ingested."),
    # 260-269
    ("doc36_qa0__after26", 1.0, "ACK honest refusal — doc36 not yet ingested."),
    ("doc131_qa0__after26", 1.0, "ACK honest refusal — doc131 not yet ingested."),
    ("doc115_qa0__after26", 1.0, "ACK honest refusal — doc115 not yet ingested."),
    ("doc85_qa0__after26", 1.0, "ACK honest refusal — doc85 not yet ingested."),
    ("doc118_qa0__after26", 1.0, "ACK honest refusal — doc118 not yet ingested."),
    ("doc77_qa0__after26", 1.0, "ACK honest refusal — doc77 not yet ingested."),
    ("doc110_qa0__after26", 1.0, "ACK honest refusal — doc110 not yet ingested."),
    ("doc63_qa0__after26", 0.5, "ACK partial commercial airlines + gov + defense."),
    ("doc40_qa0__after26", 1.0, "ACK honest refusal — doc40 not yet ingested."),
    ("doc74_qa0__after26", 1.0, "ACK honest refusal — doc74 not yet ingested."),
    # 270-275
    ("doc102_qa0__after27", 1.0, "ACK honest refusal — doc102 not yet ingested."),
    ("doc124_qa0__after27", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc39_qa0__after27", 1.0, "ACK honest refusal — doc39 not yet ingested."),
    ("doc105_qa0__after27", 1.0, "ACK honest refusal — doc105 not yet ingested."),
    ("doc132_qa0__after27", 1.0, "ACK honest refusal — doc132 not yet ingested."),
    ("doc20_qa0__after27", 1.0, "ANS gold $11,588; predicted 11,588 — exact!"),
    ("doc106_qa0__after27", 1.0, "ACK honest refusal — doc106 not yet ingested."),
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
