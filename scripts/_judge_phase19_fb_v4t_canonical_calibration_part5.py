"""Claude manual judging — Phase 1.9 FB calibration v4t-canonical (entries 650-808).

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
    # 650-659
    ("doc105_qa0__after65", 1.0, "ACK honest refusal — doc105 not yet ingested."),
    ("doc146_qa0__after65", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc26_qa0__after65", 0.0, "ANS Amcor gross margin refusal on definitive gold."),
    ("doc18_qa0__after65", 0.0, "ANS refusal on definitive gold."),
    ("doc89_qa0__after65", 1.0, "ACK honest refusal — doc89 not yet ingested."),
    ("doc114_qa0__after65", 1.0, "ACK honest refusal — doc114 not yet ingested."),
    ("doc102_qa0__after65", 1.0, "ACK honest refusal — doc102 not yet ingested."),
    ("doc38_qa0__after65", 1.0, "ANS gold 'There are none' AMEX debt securities; predicted refusal — refusal effectively matches 'none/no info'."),
    ("doc94_qa0__after65", 1.0, "ACK honest refusal — doc94 not yet ingested."),
    ("doc145_qa0__after65", 1.0, "ACK honest refusal — doc145 not yet ingested."),
    # 660-669
    ("doc55_qa0__after66", 0.5, "ANS partial 'Gaming'."),
    ("doc51_qa0__after66", 1.0, "ANS Best Buy Current Health + Yardbird match — exact."),
    ("doc62_qa0__after66", 0.0, "ANS gold Yes improving gross margin; predicted 'gross margin not useful' — wrong direction."),
    ("doc139_qa0__after66", 0.5, "ACK calibration: 'strategic decision to invest in inventory to support sales growth' — close but no 47 stores or brand launches specifics."),
    ("doc142_qa0__after66", 1.0, "ACK honest refusal — doc142 not yet ingested."),
    ("doc149_qa0__after66", 1.0, "ACK honest refusal — doc149 not yet ingested."),
    ("doc116_qa0__after66", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    ("doc103_qa0__after66", 1.0, "ACK honest refusal — doc103 not yet ingested."),
    ("doc66_qa0__after66", 0.0, "ANS gold 0.62%/-14.76% effective tax definitive; predicted refusal — refusal on definitive gold."),
    ("doc17_qa0__after66", 0.0, "ANS refusal on definitive gold."),
    # 670-679
    ("doc74_qa0__after67", 0.0, "ACK calibration: confident wildly wrong '$1,211,362M Costco FY21 total assets' (gold $59,268M — 20x off)."),
    ("doc76_qa0__after67", 1.0, "ACK honest refusal — doc76 not yet ingested."),
    ("doc25_qa0__after67", 0.0, "ANS gold Amcor packaging definitive; predicted refusal — refusal on definitive gold."),
    ("doc71_qa0__after67", 1.0, "ACK honest refusal — doc71 not yet ingested."),
    ("doc113_qa0__after67", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc2_qa0__after67", 0.0, "ANS gold 'No efficient' definitive; predicted refusal — refusal on definitive gold."),
    ("doc3_qa0__after67", 0.0, "ANS refusal on definitive gold."),
    ("doc141_qa0__after67", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    ("doc35_qa0__after67", 0.0, "ANS gold 'AMD cashflow from Operations' definitive; predicted refusal — refusal on definitive gold."),
    ("doc39_qa0__after67", 0.0, "ANS refusal on definitive gold."),
    # 680-689
    ("doc66_qa0__after68", 0.0, "ANS refusal on definitive gold."),
    ("doc25_qa0__after68", 1.0, "ANS Amcor packaging — match."),
    ("doc99_qa0__after68", 1.0, "ACK honest refusal — doc99 not yet ingested."),
    ("doc85_qa0__after68", 1.0, "ACK honest refusal — doc85 not yet ingested."),
    ("doc24_qa0__after68", 0.0, "ANS Amcor acquisitions refusal on definitive gold."),
    ("doc126_qa0__after68", 0.25, "ACK calibration: confident wrong '$1.5 billion PepsiCo credit increase' (gold $400M)."),
    ("doc32_qa0__after68", 1.0, "ANS AMD products — match."),
    ("doc15_qa0__after68", 1.0, "ANS 0 — exact."),
    ("doc82_qa0__after68", 0.25, "ACK calibration: confident wrong '1.74' (gold 0.68)."),
    ("doc121_qa0__after68", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    # 690-699
    ("doc105_qa0__after69", 1.0, "ACK honest refusal — doc105 not yet ingested."),
    ("doc85_qa0__after69", 1.0, "ACK honest refusal — doc85 not yet ingested."),
    ("doc139_qa0__after69", 0.5, "ACK 'invest in inventory to support sales growth' — partial."),
    ("doc30_qa0__after69", 0.0, "ANS refusal on definitive gold."),
    ("doc108_qa0__after69", 0.25, "ACK calibration: 'Las Vegas Strip' wrong (gold MGM China)."),
    ("doc32_qa0__after69", 1.0, "ANS AMD products — match."),
    ("doc87_qa0__after69", 1.0, "ACK honest refusal — doc87 not yet ingested."),
    ("doc93_qa0__after69", 1.0, "ACK honest refusal — doc93 not yet ingested."),
    ("doc65_qa0__after69", 1.0, "ANS Boeing production rates — match."),
    ("doc16_qa0__after69", 1.0, "ACK honest refusal — doc16 not yet ingested."),
    # 700-709
    ("doc26_qa0__after70", 0.0, "ANS refusal on definitive gold."),
    ("doc66_qa0__after70", 0.0, "ANS refusal on definitive gold."),
    ("doc93_qa0__after70", 1.0, "ACK honest refusal — doc93 not yet ingested."),
    ("doc138_qa0__after70", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc129_qa0__after70", 0.25, "ACK '2 percentage points' confident wrong (gold 1 pp)."),
    ("doc71_qa0__after70", 1.0, "ACK honest refusal — doc71 not yet ingested."),
    ("doc135_qa0__after70", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc65_qa0__after70", 1.0, "ANS Boeing production rates — match."),
    ("doc104_qa0__after70", 1.0, "ACK honest refusal — doc104 not yet ingested."),
    ("doc91_qa0__after70", 1.0, "ACK honest refusal — doc91 not yet ingested."),
    # 710-719
    ("doc10_qa0__after71", 0.0, "ANS refusal on definitive gold."),
    ("doc46_qa0__after71", 1.0, "ANS 1,832 — exact."),
    ("doc59_qa0__after71", 0.0, "ANS gold $12,645 definitive; predicted refusal — refusal on definitive gold."),
    ("doc95_qa0__after71", 1.0, "ACK honest refusal — doc95 not yet ingested."),
    ("doc55_qa0__after71", 0.5, "ANS partial 'Gaming' — no 9% or entertainment."),
    ("doc139_qa0__after71", 0.5, "ACK 'invest in inventory' partial."),
    ("doc42_qa0__after71", 0.0, "ANS gold 24.6%→21.6% definitive; predicted refusal — refusal on definitive gold."),
    ("doc94_qa0__after71", 1.0, "ACK honest refusal — doc94 not yet ingested."),
    ("doc58_qa0__after71", 0.0, "ANS gold $382 Block FY20 OCF; predicted 213.1M — wrong specific."),
    ("doc14_qa0__after71", 0.0, "ANS refusal on definitive gold."),
    # 720-729
    ("doc3_qa0__after72", 0.0, "ANS refusal on definitive gold."),
    ("doc110_qa0__after72", 1.0, "ACK honest refusal — doc110 not yet ingested."),
    ("doc134_qa0__after72", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc12_qa0__after72", 0.0, "ANS refusal on definitive gold."),
    ("doc71_qa0__after72", 0.0, "ANS gold 10.3% definitive; predicted refusal — refusal on definitive gold."),
    ("doc52_qa0__after72", 0.0, "ANS refusal on definitive gold."),
    ("doc64_qa0__after72", 1.0, "ANS Yes Boeing cyclical — match."),
    ("doc26_qa0__after72", 0.0, "ANS refusal on definitive gold."),
    ("doc117_qa0__after72", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc119_qa0__after72", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    # 730-739
    ("doc14_qa0__after73", 0.0, "ANS refusal on definitive gold."),
    ("doc106_qa0__after73", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc12_qa0__after73", 0.0, "ANS refusal on definitive gold."),
    ("doc114_qa0__after73", 0.25, "ACK calibration: confident wrong '43.0% Nike CoGS' (gold 55.1%)."),
    ("doc92_qa0__after73", 1.0, "ACK honest refusal — doc92 not yet ingested."),
    ("doc140_qa0__after73", 1.0, "ACK honest refusal — doc140 not yet ingested."),
    ("doc115_qa0__after73", 1.0, "ACK honest refusal — doc115 not yet ingested."),
    ("doc69_qa0__after73", 0.0, "ANS gold 0.8 Coca-Cola FY22 dividend payout; predicted 0.19 — wrong specific."),
    ("doc4_qa0__after73", 0.0, "ANS gold consumer shrunk definitive; predicted refusal — refusal on definitive gold."),
    ("doc26_qa0__after73", 0.0, "ANS refusal on definitive gold."),
    # 740-749
    ("doc119_qa0__after74", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    ("doc117_qa0__after74", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc69_qa0__after74", 0.0, "ANS Coca-Cola dividend payout refusal on definitive gold."),
    ("doc123_qa0__after74", 1.0, "ACK honest refusal — doc123 not yet ingested."),
    ("doc90_qa0__after74", 1.0, "ACK Consumer Health discontinued — correct by inference."),
    ("doc83_qa0__after74", 1.0, "ACK honest refusal — doc83 not yet ingested."),
    ("doc126_qa0__after74", 0.25, "ACK '$1.5 billion' confident wrong (gold $400M)."),
    ("doc50_qa0__after74", 0.0, "ANS refusal on definitive gold."),
    ("doc22_qa0__after74", 0.0, "ANS Amcor 8K refusal on definitive gold."),
    ("doc6_qa0__after74", 0.0, "ANS 3M debt securities refusal on definitive gold."),
    # 750-759
    ("doc37_qa0__after75", 1.0, "ANS Yes one customer 16% — match."),
    ("doc0_qa0__after75", 0.0, "ANS refusal on definitive gold ($1,577)."),
    ("doc122_qa0__after75", 0.25, "ACK '0' confident wrong."),
    ("doc26_qa0__after75", 0.0, "ANS refusal on definitive gold."),
    ("doc126_qa0__after75", 0.25, "ACK '$1.5 billion' confident wrong."),
    ("doc111_qa0__after75", 1.0, "ACK honest refusal — doc111 not yet ingested."),
    ("doc53_qa0__after75", 0.0, "ANS refusal on definitive gold."),
    ("doc25_qa0__after75", 1.0, "ANS Amcor packaging — match."),
    ("doc121_qa0__after75", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc133_qa0__after75", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    # 760-769
    ("doc3_qa0__after76", 0.0, "ANS refusal on definitive gold."),
    ("doc41_qa0__after76", 1.0, "ANS gross margin not useful AMEX — match."),
    ("doc112_qa0__after76", 1.0, "ACK honest refusal — doc112 not yet ingested."),
    ("doc100_qa0__after76", 1.0, "ACK honest refusal — doc100 not yet ingested."),
    ("doc37_qa0__after76", 1.0, "ANS one customer 16% — match."),
    ("doc55_qa0__after76", 0.5, "ANS partial 'Gaming'."),
    ("doc18_qa0__after76", 0.0, "ANS refusal on definitive gold."),
    ("doc86_qa0__after76", 1.0, "ACK honest refusal — doc86 not yet ingested."),
    ("doc74_qa0__after76", 1.0, "ANS $59,268 — exact."),
    ("doc11_qa0__after76", 1.0, "ANS gold 65.4%; predicted truncated calc computing 590,507/903,095 ≈ 65.4% — within tolerance."),
    # 770-779
    ("doc64_qa0__after77", 1.0, "ANS Yes cyclical — match."),
    ("doc60_qa0__after77", 1.0, "ANS Commercial Airplanes — match."),
    ("doc113_qa0__after77", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc44_qa0__after77", 1.0, "ANS Yes — match."),
    ("doc87_qa0__after77", 1.0, "ACK honest refusal — doc87 not yet ingested."),
    ("doc82_qa0__after77", 1.0, "ACK honest refusal — doc82 not yet ingested."),
    ("doc52_qa0__after77", 0.0, "ANS Best Buy operating cash flow refusal on definitive gold."),
    ("doc97_qa0__after77", 1.0, "ACK honest refusal — doc97 not yet ingested."),
    ("doc130_qa0__after77", 1.0, "ACK honest refusal — doc130 not yet ingested."),
    ("doc11_qa0__after77", 0.0, "ANS refusal on definitive gold."),
    # 780-789
    ("doc149_qa0__after78", 1.0, "ACK honest refusal — doc149 not yet ingested."),
    ("doc120_qa0__after78", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc19_qa0__after78", 0.0, "ANS gold 30.8% definitive; predicted refusal — refusal on definitive gold."),
    ("doc44_qa0__after78", 1.0, "ANS Yes — match."),
    ("doc63_qa0__after78", 0.5, "ANS partial 'airlines and government entities including defense and space' — gov/defense ~ US govt, no 40%."),
    ("doc102_qa0__after78", 1.0, "ACK honest refusal — doc102 not yet ingested."),
    ("doc67_qa0__after78", 0.0, "ANS refusal on definitive gold."),
    ("doc40_qa0__after78", 0.0, "ANS gold 'not measured through op margin' definitive; predicted refusal — refusal on definitive gold."),
    ("doc52_qa0__after78", 0.0, "ANS refusal on definitive gold."),
    ("doc65_qa0__after78", 1.0, "ANS Boeing 737/777X/787 production — match."),
    # 790-799
    ("doc146_qa0__after79", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc23_qa0__after79", 0.0, "ANS quick ratio refusal on definitive gold."),
    ("doc109_qa0__after79", 1.0, "ACK honest refusal — doc109 not yet ingested."),
    ("doc56_qa0__after79", 0.0, "ANS gold 1.73 definitive; predicted refusal — refusal on definitive gold."),
    ("doc92_qa0__after79", 1.0, "ACK honest refusal — doc92 not yet ingested."),
    ("doc55_qa0__after79", 0.5, "ANS partial 'Gaming'."),
    ("doc28_qa0__after79", 0.0, "ANS gold $2,018M definitive; predicted refusal — refusal on definitive gold."),
    ("doc83_qa0__after79", 1.0, "ACK honest refusal — doc83 not yet ingested."),
    ("doc2_qa0__after79", 0.0, "ANS refusal on definitive gold."),
    ("doc14_qa0__after79", 0.0, "ANS refusal on definitive gold."),
    # 800-808
    ("doc106_qa0__after80", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc44_qa0__after80", 1.0, "ANS Yes — match."),
    ("doc82_qa0__after80", 1.0, "ACK honest refusal — doc82 not yet ingested."),
    ("doc25_qa0__after80", 0.0, "ANS Amcor packaging refusal on definitive gold."),
    ("doc60_qa0__after80", 1.0, "ANS Commercial Airplanes — match."),
    ("doc103_qa0__after80", 1.0, "ACK honest refusal — doc103 not yet ingested."),
    ("doc35_qa0__after80", 0.0, "ANS AMD operations refusal on definitive gold."),
    ("doc12_qa0__after80", 0.0, "ANS refusal on definitive gold."),
    ("doc141_qa0__after80", 1.0, "ACK honest refusal — doc141 not yet ingested."),
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
