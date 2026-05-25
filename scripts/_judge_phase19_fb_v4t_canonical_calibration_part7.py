"""Claude manual judging — Phase 1.9 FB calibration v4t-canonical (entries 956-1103)."""

from __future__ import annotations

import json
from pathlib import Path

JUDGE_DIR = Path(
    "results/stage3/judge_queue/financebench__v4t-canonical__calibration__seed42"
)
QID_PREFIX = "financebench__v4t-canonical__calibration__"
QID_SUFFIX = "__seed42"

JUDGMENTS: list[tuple[str, float, str]] = [
    # 956-959
    ("doc142_qa0__after95", 1.0, "ACK honest refusal — doc142 not yet ingested."),
    ("doc122_qa0__after95", 0.25, "ACK '0' confident wrong."),
    ("doc8_qa0__after95", 0.0, "ANS refusal on definitive gold."),
    ("doc17_qa0__after95", 0.0, "ANS refusal on definitive gold."),
    # 960-969
    ("doc86_qa0__after96", 1.0, "ANS gold COVID-19 + currency + commodity inflation; predicted same — match."),
    ("doc80_qa0__after96", 1.0, "ANS Richard A. Johnson — match."),
    ("doc94_qa0__after96", 0.0, "ANS Consumer & Community wrong."),
    ("doc15_qa0__after96", 1.0, "ANS 0 — exact."),
    ("doc95_qa0__after96", 0.0, "ANS gold $66.56/share; predicted '$1.26 trillion' — wildly wrong specific."),
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
    ("doc125_qa0__after97", 1.0, "ACK honest refusal — doc125 not yet ingested."),
    ("doc95_qa0__after97", 0.25, "ANS gold $66.56/share; predicted '$292.3B equity' (no per-share division) — confident partial."),
    ("doc37_qa0__after97", 0.0, "ANS refusal on definitive gold."),
    ("doc6_qa0__after97", 0.0, "ANS 3M debt securities refusal on definitive gold."),
    ("doc50_qa0__after97", 0.0, "ANS refusal on definitive gold."),
    # 980-989
    ("doc42_qa0__after98", 0.0, "ANS refusal on definitive gold."),
    ("doc141_qa0__after98", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    ("doc80_qa0__after98", 1.0, "ANS Richard A. Johnson — match."),
    ("doc91_qa0__after98", 0.0, "ANS gold $20B definitive; predicted refusal — refusal on definitive gold."),
    ("doc60_qa0__after98", 0.0, "ANS refusal on definitive gold."),
    ("doc149_qa0__after98", 1.0, "ACK honest refusal — doc149 not yet ingested."),
    ("doc108_qa0__after98", 1.0, "ACK honest refusal — doc108 not yet ingested."),
    ("doc97_qa0__after98", 1.0, "ANS gold Corporate & Investment Bank; predicted 'Corporate & Investment Bank' — match (no $3725M but right segment)."),
    ("doc138_qa0__after98", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc16_qa0__after98", 0.0, "ANS refusal on definitive gold."),
    # 990-999
    ("doc113_qa0__after99", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc11_qa0__after99", 0.0, "ANS refusal on definitive gold."),
    ("doc40_qa0__after99", 0.0, "ANS refusal on definitive gold."),
    ("doc127_qa0__after99", 1.0, "ACK honest refusal — doc127 not yet ingested."),
    ("doc108_qa0__after99", 1.0, "ACK honest refusal — doc108 not yet ingested."),
    ("doc145_qa0__after99", 1.0, "ACK honest refusal — doc145 not yet ingested."),
    ("doc43_qa0__after99", 0.0, "ANS refusal on definitive gold."),
    ("doc71_qa0__after99", 0.0, "ANS gold 10.3%; predicted 13.3% — outside tolerance."),
    ("doc124_qa0__after99", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc116_qa0__after99", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    # 1000-1009
    ("doc5_qa0__after100", 0.0, "ANS refusal on definitive gold."),
    ("doc129_qa0__after100", 0.25, "ACK '2 percentage points' confident wrong (gold 1 pp)."),
    ("doc10_qa0__after100", 0.0, "ANS refusal on definitive gold."),
    ("doc90_qa0__after100", 1.0, "ANS Consumer Health discontinued — exact."),
    ("doc148_qa0__after100", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc15_qa0__after100", 1.0, "ANS 0 — exact."),
    ("doc67_qa0__after100", 0.0, "ANS refusal on definitive gold."),
    ("doc127_qa0__after100", 1.0, "ACK honest refusal — doc127 not yet ingested."),
    ("doc65_qa0__after100", 0.0, "ANS refusal on definitive gold."),
    ("doc63_qa0__after100", 0.5, "ANS partial — commercial airlines + gov + defense."),
    # 1010-1019
    ("doc81_qa0__after101", 0.0, "ANS refusal on definitive gold."),
    ("doc114_qa0__after101", 1.0, "ACK honest refusal — doc114 not yet ingested."),
    ("doc35_qa0__after101", 0.0, "ANS refusal on definitive gold."),
    ("doc41_qa0__after101", 0.0, "ANS refusal on definitive gold."),
    ("doc100_qa0__after101", 0.0, "ANS gold 1.33; predicted 0.39 — wrong specific."),
    ("doc98_qa0__after101", 1.0, "ANS Yes decreased $7M — match."),
    ("doc78_qa0__after101", 0.5, "ANS partial Yes dividends Q2 — no $0.55."),
    ("doc75_qa0__after101", 0.0, "ANS refusal on definitive gold."),
    ("doc96_qa0__after101", 1.0, "ANS JPM gross margins not relevant — match."),
    ("doc125_qa0__after101", 1.0, "ACK 'proposal not approved' = 'defeated' — correct."),
    # 1020-1029
    ("doc31_qa0__after102", 0.0, "ANS refusal on definitive gold."),
    ("doc39_qa0__after102", 0.0, "ANS refusal on definitive gold."),
    ("doc24_qa0__after102", 0.0, "ANS Amcor acquisitions refusal on definitive gold."),
    ("doc68_qa0__after102", 0.0, "ANS gold 39.7% definitive; predicted refusal — refusal on definitive gold."),
    ("doc119_qa0__after102", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    ("doc44_qa0__after102", 1.0, "ANS Yes — match."),
    ("doc36_qa0__after102", 0.0, "ANS refusal on definitive gold."),
    ("doc59_qa0__after102", 0.0, "ANS refusal on definitive gold."),
    ("doc46_qa0__after102", 1.0, "ANS 1,832 — exact."),
    ("doc108_qa0__after102", 1.0, "ACK honest refusal — doc108 not yet ingested."),
    # 1030-1039
    ("doc108_qa0__after103", 1.0, "ACK honest refusal — doc108 not yet ingested."),
    ("doc61_qa0__after103", 1.0, "ANS Lion Air + Ethiopian crashes — match."),
    ("doc135_qa0__after103", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc60_qa0__after103", 1.0, "ANS Commercial Airplanes — match."),
    ("doc36_qa0__after103", 0.0, "ANS refusal on definitive gold."),
    ("doc51_qa0__after103", 0.0, "ANS refusal on definitive gold."),
    ("doc85_qa0__after103", 1.0, "ANS No 1.3% growth — match."),
    ("doc105_qa0__after103", 1.0, "ACK honest refusal — doc105 not yet ingested."),
    ("doc71_qa0__after103", 0.0, "ANS gold 10.3%; predicted 13.3% — outside tolerance."),
    ("doc137_qa0__after103", 1.0, "ACK honest refusal — doc137 not yet ingested."),
    # 1040-1049
    ("doc46_qa0__after104", 1.0, "ANS 1,832 — exact."),
    ("doc136_qa0__after104", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc121_qa0__after104", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc96_qa0__after104", 1.0, "ANS JPM gross margins not relevant — match."),
    ("doc16_qa0__after104", 0.0, "ANS refusal on definitive gold."),
    ("doc80_qa0__after104", 1.0, "ANS Richard A. Johnson — match."),
    ("doc31_qa0__after104", 0.0, "ANS refusal on definitive gold."),
    ("doc14_qa0__after104", 0.0, "ANS refusal on definitive gold."),
    ("doc101_qa0__after104", 0.25, "ANS gold $5,818; predicted '818 million USD' — wrong specific (missed leading $5,)."),
    ("doc103_qa0__after104", 0.0, "ANS refusal on definitive gold."),
    # 1050-1059
    ("doc22_qa0__after105", 0.0, "ANS Amcor 8K refusal on definitive gold."),
    ("doc119_qa0__after105", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    ("doc25_qa0__after105", 0.0, "ANS Amcor packaging refusal on definitive gold."),
    ("doc146_qa0__after105", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc62_qa0__after105", 0.0, "ANS refusal on definitive gold."),
    ("doc98_qa0__after105", 1.0, "ANS Yes decreased $7M — match."),
    ("doc1_qa0__after105", 0.0, "ANS refusal on definitive gold."),
    ("doc138_qa0__after105", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc123_qa0__after105", 1.0, "ACK honest refusal — doc123 not yet ingested."),
    ("doc76_qa0__after105", 0.0, "ANS refusal on definitive gold."),
    # 1060-1069
    ("doc124_qa0__after106", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc70_qa0__after106", 0.0, "ANS gold 63.86 DPO Corning definitive; predicted refusal — refusal on definitive gold."),
    ("doc28_qa0__after106", 0.0, "ANS AMCOR EBITDA refusal on definitive gold."),
    ("doc30_qa0__after106", 0.0, "ANS refusal on definitive gold."),
    ("doc85_qa0__after106", 1.0, "ANS No 1.3% — match."),
    ("doc130_qa0__after106", 1.0, "ACK honest refusal — doc130 not yet ingested."),
    ("doc87_qa0__after106", 0.25, "ANS gold 2.7 turnover JnJ; predicted partial calc with COGS $8,212M + inventory $12,483M (turnover ~0.66 if completed) — confident wrong setup."),
    ("doc135_qa0__after106", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc148_qa0__after106", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc49_qa0__after106", 0.0, "ANS gold $5,409 definitive; predicted refusal — refusal on definitive gold."),
    # 1070-1079
    ("doc9_qa0__after107", 0.0, "ANS refusal on definitive gold."),
    ("doc134_qa0__after107", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc13_qa0__after107", 0.0, "ANS refusal on definitive gold."),
    ("doc142_qa0__after107", 1.0, "ACK honest refusal — doc142 not yet ingested."),
    ("doc127_qa0__after107", 1.0, "ACK honest refusal — doc127 not yet ingested."),
    ("doc122_qa0__after107", 0.25, "ACK '0' confident wrong."),
    ("doc133_qa0__after107", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc103_qa0__after107", 0.0, "ANS refusal on definitive gold."),
    ("doc139_qa0__after107", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc87_qa0__after107", 0.0, "ANS refusal on definitive gold."),
    # 1080-1089
    ("doc75_qa0__after108", 0.0, "ANS refusal on definitive gold."),
    ("doc90_qa0__after108", 1.0, "ANS Consumer Health discontinued — exact."),
    ("doc98_qa0__after108", 0.75, "ANS gold Yes decreased; predicted Yes decreased (general) — match Yes direction."),
    ("doc140_qa0__after108", 1.0, "ACK honest refusal — doc140 not yet ingested."),
    ("doc42_qa0__after108", 0.0, "ANS refusal on definitive gold."),
    ("doc43_qa0__after108", 0.25, "ANS 'long-term debt' wrong specific (gold Customer deposits)."),
    ("doc51_qa0__after108", 0.0, "ANS refusal on definitive gold."),
    ("doc68_qa0__after108", 0.0, "ANS refusal on definitive gold."),
    ("doc45_qa0__after108", 0.0, "ANS refusal on definitive gold."),
    ("doc108_qa0__after108", 1.0, "ANS gold MGM China 44% decline; predicted 'MGM China worst, $674M revenues -44%' — match (gets region + 44%)."),
    # 1090-1099
    ("doc26_qa0__after109", 0.0, "ANS refusal on definitive gold."),
    ("doc7_qa0__after109", 0.0, "ANS gold Yes 3M 65th dividend definitive; predicted refusal — refusal on definitive gold."),
    ("doc119_qa0__after109", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    ("doc14_qa0__after109", 0.0, "ANS refusal on definitive gold."),
    ("doc44_qa0__after109", 1.0, "ANS Yes — match."),
    ("doc102_qa0__after109", 0.0, "ANS gold 0.4%; predicted 5.0% — wrong specific."),
    ("doc65_qa0__after109", 0.0, "ANS refusal on definitive gold."),
    ("doc133_qa0__after109", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc18_qa0__after109", 0.0, "ANS refusal on definitive gold."),
    ("doc134_qa0__after109", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    # 1100-1103
    ("doc120_qa0__after110", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc7_qa0__after110", 1.0, "ANS Yes 65th — match."),
    ("doc72_qa0__after110", 0.0, "ANS refusal on definitive gold."),
    ("doc35_qa0__after110", 0.0, "ANS refusal on definitive gold."),
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
