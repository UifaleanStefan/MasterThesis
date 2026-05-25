"""Claude manual judging — Phase 1.9 FB calibration v4t-canonical (entries 0-199).

Manually judged 1-by-1 per HARD RULE in evaluation/claude_judge_protocol.md.
"""

from __future__ import annotations

import json
from pathlib import Path

JUDGE_DIR = Path(
    "results/stage3/judge_queue/financebench__v4t-canonical__calibration__seed42"
)

# qids in canonical format: financebench__v4t-canonical__calibration__doc{N}_qa0__after{M}__seed42
QID_PREFIX = "financebench__v4t-canonical__calibration__"
QID_SUFFIX = "__seed42"

JUDGMENTS: list[tuple[str, float, str]] = [
    # 0-9
    ("doc123_qa0__after0", 1.0, "ACK honest refusal — doc123 not yet ingested."),
    ("doc31_qa0__after0", 1.0, "ACK honest refusal — doc31 not yet ingested."),
    ("doc147_qa0__after0", 1.0, "ACK honest refusal — doc147 not yet ingested."),
    ("doc130_qa0__after0", 1.0, "ACK honest refusal — doc130 not yet ingested."),
    ("doc115_qa0__after0", 1.0, "ACK honest refusal — doc115 not yet ingested."),
    ("doc119_qa0__after0", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    ("doc133_qa0__after0", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc137_qa0__after0", 1.0, "ACK honest refusal — doc137 not yet ingested."),
    ("doc59_qa0__after0", 1.0, "ACK honest refusal — doc59 not yet ingested."),
    ("doc27_qa0__after0", 1.0, "ACK honest refusal — doc27 not yet ingested."),
    # 10-19
    ("doc93_qa0__after1", 1.0, "ACK honest refusal — doc93 not yet ingested."),
    ("doc72_qa0__after1", 1.0, "ACK honest refusal — doc72 not yet ingested."),
    ("doc64_qa0__after1", 1.0, "ACK calibration: 'Yes Boeing cyclical' — correct by inference."),
    ("doc6_qa0__after1", 1.0, "ACK honest refusal — doc6 not yet ingested."),
    ("doc27_qa0__after1", 1.0, "ACK honest refusal — doc27 not yet ingested."),
    ("doc35_qa0__after1", 1.0, "ACK calibration: 'cash flow from operating activities most cash AMD FY22' — correct by inference."),
    ("doc5_qa0__after1", 1.0, "ACK honest refusal — doc5 not yet ingested."),
    ("doc60_qa0__after1", 1.0, "ACK honest refusal — doc60 not yet ingested."),
    ("doc106_qa0__after1", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc87_qa0__after1", 1.0, "ACK honest refusal — doc87 not yet ingested."),
    # 20-29
    ("doc101_qa0__after2", 1.0, "ACK honest refusal — doc101 not yet ingested."),
    ("doc71_qa0__after2", 1.0, "ACK honest refusal — doc71 not yet ingested."),
    ("doc1_qa0__after2", 0.0, "ANS gold $8.70; predicted 9.178B — wrong specific."),
    ("doc118_qa0__after2", 1.0, "ACK honest refusal — doc118 not yet ingested."),
    ("doc75_qa0__after2", 1.0, "ACK honest refusal — doc75 not yet ingested."),
    ("doc67_qa0__after2", 1.0, "ACK honest refusal — doc67 not yet ingested."),
    ("doc13_qa0__after2", 1.0, "ACK honest refusal — doc13 not yet ingested."),
    ("doc78_qa0__after2", 1.0, "ACK honest refusal — doc78 not yet ingested."),
    ("doc116_qa0__after2", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    ("doc91_qa0__after2", 1.0, "ACK honest refusal — doc91 not yet ingested."),
    # 30-39
    ("doc43_qa0__after3", 1.0, "ACK honest refusal — doc43 not yet ingested."),
    ("doc120_qa0__after3", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc101_qa0__after3", 1.0, "ACK honest refusal — doc101 not yet ingested."),
    ("doc64_qa0__after3", 1.0, "ACK honest refusal — doc64 not yet ingested."),
    ("doc107_qa0__after3", 1.0, "ACK honest refusal — doc107 not yet ingested."),
    ("doc121_qa0__after3", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc102_qa0__after3", 1.0, "ACK honest refusal — doc102 not yet ingested."),
    ("doc90_qa0__after3", 1.0, "ACK calibration: 'Consumer Health discontinued Aug 30, 2023' — correct by inference (JnJ data ingested elsewhere)."),
    ("doc26_qa0__after3", 1.0, "ACK honest refusal — doc26 not yet ingested."),
    ("doc22_qa0__after3", 1.0, "ACK honest refusal — doc22 not yet ingested."),
    # 40-49
    ("doc122_qa0__after4", 0.25, "ACK calibration: confident wrong '0' (gold $411M Pepsico restructuring)."),
    ("doc141_qa0__after4", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    ("doc25_qa0__after4", 1.0, "ACK honest refusal — doc25 not yet ingested."),
    ("doc43_qa0__after4", 1.0, "ACK honest refusal — doc43 not yet ingested."),
    ("doc76_qa0__after4", 1.0, "ACK honest refusal — doc76 not yet ingested."),
    ("doc120_qa0__after4", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc138_qa0__after4", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc42_qa0__after4", 1.0, "ACK honest refusal — doc42 not yet ingested."),
    ("doc83_qa0__after4", 1.0, "ACK honest refusal — doc83 not yet ingested."),
    ("doc95_qa0__after4", 1.0, "ACK honest refusal — doc95 not yet ingested."),
    # 50-59
    ("doc147_qa0__after5", 1.0, "ACK honest refusal — doc147 not yet ingested."),
    ("doc32_qa0__after5", 1.0, "ACK honest refusal — doc32 not yet ingested."),
    ("doc131_qa0__after5", 1.0, "ACK honest refusal — doc131 not yet ingested."),
    ("doc97_qa0__after5", 1.0, "ACK honest refusal — doc97 not yet ingested."),
    ("doc93_qa0__after5", 1.0, "ACK honest refusal — doc93 not yet ingested."),
    ("doc80_qa0__after5", 1.0, "ACK honest refusal — doc80 not yet ingested."),
    ("doc109_qa0__after5", 1.0, "ACK honest refusal — doc109 not yet ingested."),
    ("doc113_qa0__after5", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc13_qa0__after5", 1.0, "ACK honest refusal — doc13 not yet ingested."),
    ("doc110_qa0__after5", 1.0, "ACK honest refusal — doc110 not yet ingested."),
    # 60-69
    ("doc127_qa0__after6", 1.0, "ACK honest refusal — doc127 not yet ingested."),
    ("doc149_qa0__after6", 1.0, "ACK honest refusal — doc149 not yet ingested."),
    ("doc46_qa0__after6", 1.0, "ACK honest refusal — doc46 not yet ingested."),
    ("doc34_qa0__after6", 1.0, "ACK honest refusal — doc34 not yet ingested."),
    ("doc62_qa0__after6", 1.0, "ACK honest refusal — doc62 not yet ingested."),
    ("doc25_qa0__after6", 1.0, "ACK honest refusal — doc25 not yet ingested."),
    ("doc126_qa0__after6", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    ("doc43_qa0__after6", 1.0, "ACK honest refusal — doc43 not yet ingested."),
    ("doc83_qa0__after6", 1.0, "ACK honest refusal — doc83 not yet ingested."),
    ("doc146_qa0__after6", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    # 70-79
    ("doc127_qa0__after7", 1.0, "ACK honest refusal — doc127 not yet ingested."),
    ("doc125_qa0__after7", 1.0, "ACK honest refusal — doc125 not yet ingested."),
    ("doc81_qa0__after7", 1.0, "ACK honest refusal — doc81 not yet ingested."),
    ("doc58_qa0__after7", 1.0, "ACK honest refusal — doc58 not yet ingested."),
    ("doc133_qa0__after7", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc6_qa0__after7", 1.0, "ANS gold 3M debt securities 1.500% 2026, 1.750% 2030; predicted same notes — match."),
    ("doc136_qa0__after7", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc141_qa0__after7", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    ("doc47_qa0__after7", 1.0, "ACK honest refusal — doc47 not yet ingested."),
    ("doc91_qa0__after7", 1.0, "ACK honest refusal — doc91 not yet ingested."),
    # 80-89
    ("doc61_qa0__after8", 1.0, "ACK honest refusal — doc61 not yet ingested."),
    ("doc147_qa0__after8", 1.0, "ACK honest refusal — doc147 not yet ingested."),
    ("doc143_qa0__after8", 1.0, "ACK honest refusal — doc143 not yet ingested."),
    ("doc69_qa0__after8", 1.0, "ACK honest refusal — doc69 not yet ingested."),
    ("doc5_qa0__after8", 0.0, "ANS gold No quick ratio 0.96 definitive; predicted refusal — refusal on definitive gold."),
    ("doc138_qa0__after8", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc108_qa0__after8", 0.25, "ACK calibration: confident wrong 'Las Vegas Strip' (gold MGM China worst)."),
    ("doc76_qa0__after8", 1.0, "ACK honest refusal — doc76 not yet ingested."),
    ("doc131_qa0__after8", 1.0, "ACK honest refusal — doc131 not yet ingested."),
    ("doc145_qa0__after8", 1.0, "ACK honest refusal — doc145 not yet ingested."),
    # 90-99
    ("doc37_qa0__after9", 1.0, "ACK honest refusal — doc37 not yet ingested."),
    ("doc82_qa0__after9", 1.0, "ACK honest refusal — doc82 not yet ingested."),
    ("doc23_qa0__after9", 1.0, "ACK honest refusal — doc23 not yet ingested."),
    ("doc119_qa0__after9", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    ("doc20_qa0__after9", 1.0, "ACK honest refusal — doc20 not yet ingested."),
    ("doc50_qa0__after9", 1.0, "ACK honest refusal — doc50 not yet ingested."),
    ("doc7_qa0__after9", 1.0, "ANS gold Yes 3M 65th dividend year; predicted Yes 65th — match."),
    ("doc73_qa0__after9", 1.0, "ACK honest refusal — doc73 not yet ingested."),
    ("doc33_qa0__after9", 1.0, "ACK honest refusal — doc33 not yet ingested."),
    ("doc96_qa0__after9", 1.0, "ACK calibration: 'gross margins not relevant for JPM' — correct by inference."),
    # 100-109
    ("doc142_qa0__after10", 1.0, "ACK honest refusal — doc142 not yet ingested."),
    ("doc129_qa0__after10", 0.25, "ACK calibration: confident wrong '2 percentage points' (gold 1 pp PepsiCo EPS guidance raise)."),
    ("doc138_qa0__after10", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc70_qa0__after10", 1.0, "ACK honest refusal — doc70 not yet ingested."),
    ("doc58_qa0__after10", 1.0, "ACK honest refusal — doc58 not yet ingested."),
    ("doc130_qa0__after10", 1.0, "ACK honest refusal — doc130 not yet ingested."),
    ("doc46_qa0__after10", 1.0, "ACK honest refusal — doc46 not yet ingested."),
    ("doc1_qa0__after10", 0.0, "ANS gold $8.70; predicted 9.178B — wrong specific."),
    ("doc122_qa0__after10", 0.25, "ACK calibration: '0' confident wrong (gold $411M)."),
    ("doc87_qa0__after10", 1.0, "ACK honest refusal — doc87 not yet ingested."),
    # 110-119
    ("doc108_qa0__after11", 1.0, "ACK honest refusal — doc108 not yet ingested."),
    ("doc53_qa0__after11", 1.0, "ACK honest refusal — doc53 not yet ingested."),
    ("doc94_qa0__after11", 1.0, "ACK honest refusal — doc94 not yet ingested."),
    ("doc67_qa0__after11", 1.0, "ACK honest refusal — doc67 not yet ingested."),
    ("doc75_qa0__after11", 1.0, "ACK honest refusal — doc75 not yet ingested."),
    ("doc132_qa0__after11", 1.0, "ACK honest refusal — doc132 not yet ingested."),
    ("doc143_qa0__after11", 1.0, "ACK honest refusal — doc143 not yet ingested."),
    ("doc95_qa0__after11", 1.0, "ACK honest refusal — doc95 not yet ingested."),
    ("doc86_qa0__after11", 1.0, "ACK honest refusal — doc86 not yet ingested."),
    ("doc40_qa0__after11", 1.0, "ACK honest refusal — doc40 not yet ingested."),
    # 120-129
    ("doc100_qa0__after12", 1.0, "ACK honest refusal — doc100 not yet ingested."),
    ("doc12_qa0__after12", 0.0, "ANS gold 0.83 operating cash flow ratio; predicted 2.90 — wrong specific."),
    ("doc107_qa0__after12", 1.0, "ACK honest refusal — doc107 not yet ingested."),
    ("doc142_qa0__after12", 1.0, "ACK honest refusal — doc142 not yet ingested."),
    ("doc105_qa0__after12", 1.0, "ACK honest refusal — doc105 not yet ingested."),
    ("doc48_qa0__after12", 1.0, "ACK honest refusal — doc48 not yet ingested."),
    ("doc21_qa0__after12", 1.0, "ACK honest refusal — doc21 not yet ingested."),
    ("doc58_qa0__after12", 1.0, "ACK honest refusal — doc58 not yet ingested."),
    ("doc143_qa0__after12", 1.0, "ACK honest refusal — doc143 not yet ingested."),
    ("doc17_qa0__after12", 1.0, "ACK honest refusal — doc17 not yet ingested."),
    # 130-139
    ("doc99_qa0__after13", 1.0, "ACK honest refusal — doc99 not yet ingested."),
    ("doc64_qa0__after13", 1.0, "ACK 'Yes Boeing cyclical' — correct by inference."),
    ("doc98_qa0__after13", 1.0, "ACK honest refusal — doc98 not yet ingested."),
    ("doc54_qa0__after13", 1.0, "ACK honest refusal — doc54 not yet ingested."),
    ("doc68_qa0__after13", 1.0, "ACK honest refusal — doc68 not yet ingested."),
    ("doc110_qa0__after13", 1.0, "ACK honest refusal — doc110 not yet ingested."),
    ("doc3_qa0__after13", 0.75, "ANS gold -1.7% reasons (PFAS, Russia, restructuring); predicted same items — partial, no -1.7%."),
    ("doc12_qa0__after13", 0.0, "ANS gold 0.83; predicted 2.90 — wrong specific."),
    ("doc124_qa0__after13", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc137_qa0__after13", 1.0, "ACK honest refusal — doc137 not yet ingested."),
    # 140-149
    ("doc53_qa0__after14", 1.0, "ACK honest refusal — doc53 not yet ingested."),
    ("doc63_qa0__after14", 1.0, "ACK honest refusal — doc63 not yet ingested."),
    ("doc30_qa0__after14", 1.0, "ACK honest refusal — doc30 not yet ingested."),
    ("doc19_qa0__after14", 1.0, "ACK honest refusal — doc19 not yet ingested."),
    ("doc7_qa0__after14", 1.0, "ANS Yes 65th — match."),
    ("doc111_qa0__after14", 1.0, "ACK honest refusal — doc111 not yet ingested."),
    ("doc3_qa0__after14", 0.75, "ANS gold -1.7% reasons; predicted PFAS Russia restructuring — partial."),
    ("doc90_qa0__after14", 1.0, "ACK calibration: Consumer Health discontinued — correct by inference."),
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
    ("doc2_qa0__after17", 0.0, "ANS gold 'No efficient CAPEX 5.1%'; predicted 'Yes capital-intensive' — wrong direction."),
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
    ("doc4_qa0__after18", 0.5, "ANS gold consumer shrunk 0.9% organically; predicted just 'Consumer segment' — partial."),
    ("doc49_qa0__after18", 1.0, "ACK honest refusal — doc49 not yet ingested."),
    ("doc3_qa0__after18", 0.75, "ANS PFAS Russia — partial."),
    ("doc14_qa0__after18", 0.0, "ANS gold Yes Adobe FCF improved; predicted refusal — refusal on definitive gold."),
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
    # Need to peek at next 2 entries (198, 199). Will add in next batch.
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
