"""Claude manual judging — Phase 1.9 FB calibration v4t-canonical (entries 1104-1264)."""

from __future__ import annotations

import json
from pathlib import Path

JUDGE_DIR = Path(
    "results/stage3/judge_queue/financebench__v4t-canonical__calibration__seed42"
)
QID_PREFIX = "financebench__v4t-canonical__calibration__"
QID_SUFFIX = "__seed42"

JUDGMENTS: list[tuple[str, float, str]] = [
    # 1104-1109
    ("doc99_qa0__after110", 0.0, "ANS refusal on definitive gold (Kraft Heinz inventory turnover)."),
    ("doc33_qa0__after110", 1.0, "ANS AMD FY22 EPYC + Gaming + Embedded — match."),
    ("doc145_qa0__after110", 1.0, "ACK honest refusal — doc145 not yet ingested."),
    ("doc90_qa0__after110", 1.0, "ANS Consumer Health discontinued — exact."),
    ("doc97_qa0__after110", 0.0, "ANS refusal on definitive gold."),
    ("doc122_qa0__after110", 0.25, "ACK '0' confident wrong (gold $411M Pepsico restructuring)."),
    # 1110-1119
    ("doc117_qa0__after111", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc120_qa0__after111", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc26_qa0__after111", 0.0, "ANS refusal on definitive gold (Amcor gross margin)."),
    ("doc113_qa0__after111", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc30_qa0__after111", 0.0, "ANS refusal on definitive gold (4.2% D&A)."),
    ("doc82_qa0__after111", 0.0, "ANS refusal on definitive gold (0.68 GIS WC ratio)."),
    ("doc36_qa0__after111", 0.0, "ANS refusal on definitive gold (Data Center)."),
    ("doc72_qa0__after111", 0.0, "ANS refusal on definitive gold (Corning tax)."),
    ("doc37_qa0__after111", 0.0, "ANS refusal on definitive gold (16% customer)."),
    ("doc101_qa0__after111", 1.0, "ANS gold $5,818; predicted $5,818M — exact."),
    # 1120-1129
    ("doc35_qa0__after112", 0.0, "ANS refusal on definitive gold."),
    ("doc52_qa0__after112", 0.0, "ANS refusal on definitive gold."),
    ("doc23_qa0__after112", 0.0, "ANS refusal on definitive gold (Amcor quick ratio)."),
    ("doc120_qa0__after112", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc21_qa0__after112", 0.0, "ANS refusal on definitive gold ($1,616 Amcor AR)."),
    ("doc59_qa0__after112", 0.0, "ANS refusal on definitive gold ($12,645 Boeing PPNE)."),
    ("doc114_qa0__after112", 1.0, "ACK honest refusal — doc114 not yet ingested."),
    ("doc92_qa0__after112", 0.0, "ANS refusal on definitive gold ($13.2B Kenvue)."),
    ("doc89_qa0__after112", 0.0, "ANS refusal on definitive gold (US 3.0% intl -0.6%)."),
    ("doc122_qa0__after112", 0.25, "ACK '0' confident wrong."),
    # 1130-1139
    ("doc139_qa0__after113", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc104_qa0__after113", 0.0, "ANS refusal on definitive gold (7.9% MGM capex)."),
    ("doc136_qa0__after113", 1.0, "ACK honest refusal — doc136 not yet ingested (gold 'There are none' essentially equivalent)."),
    ("doc82_qa0__after113", 0.0, "ANS refusal on definitive gold."),
    ("doc60_qa0__after113", 0.0, "ANS refusal on definitive gold (Boeing Commercial Airplanes)."),
    ("doc89_qa0__after113", 0.0, "ANS refusal on definitive gold."),
    ("doc47_qa0__after113", 0.0, "ANS refusal on definitive gold."),
    ("doc137_qa0__after113", 1.0, "ACK honest refusal — doc137 not yet ingested."),
    ("doc56_qa0__after113", 0.0, "ANS refusal on definitive gold (1.73 Block WC ratio)."),
    ("doc98_qa0__after113", 1.0, "ANS Yes decreased $7M VaR — match."),
    # 1140-1149
    ("doc24_qa0__after114", 0.0, "ANS refusal on definitive gold (Amcor acquisitions)."),
    ("doc113_qa0__after114", 0.0, "ANS refusal on definitive gold ($5,466 Netflix CL)."),
    ("doc27_qa0__after114", 0.0, "ANS refusal on definitive gold (87% restructuring)."),
    ("doc124_qa0__after114", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc97_qa0__after114", 0.0, "ANS refusal on definitive gold (Corporate & Investment Bank)."),
    ("doc99_qa0__after114", 0.0, "ANS refusal on definitive gold (6.25 Kraft Heinz)."),
    ("doc131_qa0__after114", 1.0, "ACK honest refusal — doc131 not yet ingested."),
    ("doc19_qa0__after114", 0.0, "ANS refusal on definitive gold (30.8% Amazon)."),
    ("doc98_qa0__after114", 1.0, "ANS Yes decreased $7M — match."),
    ("doc12_qa0__after114", 0.0, "ANS refusal on definitive gold (0.83 Adobe OCF)."),
    # 1150-1159
    ("doc80_qa0__after115", 1.0, "ANS Richard A. Johnson 16,105,005 votes — match."),
    ("doc23_qa0__after115", 0.0, "ANS refusal on definitive gold."),
    ("doc111_qa0__after115", 0.25, "ANS gold 'No Microsoft -$2.5bn'; predicted 'Yes increased long-term debt $47,032→$41,990' — contradictory wording (says 'increased' but numbers show -$5,042M decrease). Yes/No flip with the underlying numbers actually supporting gold."),
    ("doc131_qa0__after115", 1.0, "ACK honest refusal — doc131 not yet ingested."),
    ("doc33_qa0__after115", 0.0, "ANS refusal on definitive gold (AMD revenue change)."),
    ("doc87_qa0__after115", 0.0, "ANS refusal on definitive gold (JnJ inventory turnover)."),
    ("doc140_qa0__after115", 1.0, "ACK honest refusal — doc140 not yet ingested."),
    ("doc81_qa0__after115", 0.0, "ANS refusal on definitive gold (-3.7 CCC)."),
    ("doc121_qa0__after115", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc68_qa0__after115", 0.0, "ANS refusal on definitive gold (39.7% Coca-Cola)."),
    # 1160-1169
    ("doc4_qa0__after116", 0.0, "ANS refusal on definitive gold (consumer shrunk 0.9%)."),
    ("doc66_qa0__after116", 0.0, "ANS refusal on definitive gold (Boeing tax rate)."),
    ("doc120_qa0__after116", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc138_qa0__after116", 0.25, "ACK calibration: confident vague 'improved operating efficiencies' (gold specific 'lower marketing + leverage of incentive comp')."),
    ("doc88_qa0__after116", 0.25, "ANS gold 'No 3.6%→3.5%'; predicted 'Yes 3.5%' — gets 3.5% but Yes direction wrong."),
    ("doc93_qa0__after116", 0.0, "ANS refusal on definitive gold (20%→20.1%)."),
    ("doc105_qa0__after116", 1.0, "ANS gold Yes MGM $0.01/share; predicted Yes $0.01/share — exact."),
    ("doc44_qa0__after116", 1.0, "ANS Yes retained card members — match."),
    ("doc104_qa0__after116", 0.0, "ANS gold 7.9%; predicted 12.0% — wrong specific."),
    ("doc21_qa0__after116", 0.0, "ANS refusal on definitive gold."),
    # 1170-1179
    ("doc146_qa0__after117", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc131_qa0__after117", 1.0, "ACK honest refusal — doc131 not yet ingested."),
    ("doc6_qa0__after117", 0.0, "ANS refusal on definitive gold (3M debt securities)."),
    ("doc44_qa0__after117", 1.0, "ANS Yes — match."),
    ("doc42_qa0__after117", 0.25, "ANS gold 24.6%→21.6%; predicted '19.4%→19.5%' — confident wrong specifics."),
    ("doc54_qa0__after117", 1.0, "ANS gold 982→969 Best Buy; predicted same — exact match."),
    ("doc2_qa0__after117", 0.0, "ANS refusal on definitive gold (3M efficient CAPEX)."),
    ("doc148_qa0__after117", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc121_qa0__after117", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc3_qa0__after117", 0.0, "ANS refusal on definitive gold (3M operating margin reasons)."),
    # 1180-1189
    ("doc107_qa0__after118", 0.0, "ANS gold zero (negative EBIT MGM); predicted 1.79 — wrong."),
    ("doc93_qa0__after118", 0.0, "ANS refusal on definitive gold."),
    ("doc4_qa0__after118", 0.0, "ANS refusal on definitive gold."),
    ("doc133_qa0__after118", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    ("doc22_qa0__after118", 0.0, "ANS refusal on definitive gold (Amcor 8K)."),
    ("doc37_qa0__after118", 1.0, "ANS Yes 16% one customer — match."),
    ("doc73_qa0__after118", 0.0, "ANS refusal on definitive gold (Corning WC $831M)."),
    ("doc45_qa0__after118", 0.0, "ANS refusal on definitive gold ($0.40 AWK)."),
    ("doc41_qa0__after118", 0.0, "ANS refusal on definitive gold."),
    ("doc34_qa0__after118", 0.0, "ANS refusal on definitive gold (Xilinx)."),
    # 1190-1199
    ("doc15_qa0__after119", 1.0, "ANS 0 — exact."),
    ("doc142_qa0__after119", 1.0, "ACK honest refusal — doc142 not yet ingested."),
    ("doc45_qa0__after119", 0.0, "ANS refusal on definitive gold."),
    ("doc49_qa0__after119", 0.0, "ANS refusal on definitive gold ($5,409)."),
    ("doc68_qa0__after119", 0.0, "ANS refusal on definitive gold (39.7%)."),
    ("doc48_qa0__after119", 0.0, "ANS refusal on definitive gold (2.8%)."),
    ("doc25_qa0__after119", 0.0, "ANS refusal on definitive gold (Amcor packaging)."),
    ("doc146_qa0__after119", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc59_qa0__after119", 0.0, "ANS refusal on definitive gold."),
    ("doc52_qa0__after119", 1.0, "ANS gold 'Best Buy operating $1.8bn'; predicted 'operating activities most cash flow' — match direction (no $1.8bn)."),
    # 1200-1209
    ("doc54_qa0__after120", 1.0, "ANS Best Buy 982→969 — exact match."),
    ("doc121_qa0__after120", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc112_qa0__after120", 0.0, "ANS gold 5.4%; predicted 4.51% — outside 5% tolerance."),
    ("doc117_qa0__after120", 1.0, "ANS Nike operating activities most cash flow — match."),
    ("doc3_qa0__after120", 0.0, "ANS refusal on definitive gold."),
    ("doc0_qa0__after120", 0.0, "ANS refusal on definitive gold."),
    ("doc99_qa0__after120", 0.0, "ANS refusal on definitive gold."),
    ("doc88_qa0__after120", 0.25, "ANS Yes 3.5% — gets 3.5% but Yes wrong direction."),
    ("doc34_qa0__after120", 0.0, "ANS refusal on definitive gold."),
    ("doc72_qa0__after120", 1.0, "ANS Corning 20%→23% — exact."),
    # 1210-1219
    ("doc114_qa0__after121", 1.0, "ANS gold 55.1%; predicted 56.2% — within tolerance."),
    ("doc127_qa0__after121", 0.25, "ACK '$4.0 billion' confident wrong (gold $8.4B)."),
    ("doc11_qa0__after121", 0.0, "ANS refusal on definitive gold (65.4%)."),
    ("doc136_qa0__after121", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc46_qa0__after121", 0.0, "ANS refusal on definitive gold ($1,832)."),
    ("doc92_qa0__after121", 0.25, "ANS gold $13.2B; predicted $3.7B — confident wrong specific."),
    ("doc115_qa0__after121", 0.25, "ANS gold $16,525; predicted $11,511M — wrong specific."),
    ("doc49_qa0__after121", 0.0, "ANS refusal on definitive gold."),
    ("doc72_qa0__after121", 1.0, "ANS Corning 20%→23% — exact."),
    ("doc106_qa0__after121", 0.0, "ANS refusal on definitive gold (Las Vegas EBITDAR)."),
    # 1220-1229
    ("doc12_qa0__after122", 0.0, "ANS refusal on definitive gold."),
    ("doc125_qa0__after122", 1.0, "ACK 'not approved' = 'defeated' — correct."),
    ("doc128_qa0__after122", 1.0, "ACK honest refusal — doc128 not yet ingested."),
    ("doc67_qa0__after122", 0.0, "ANS refusal on definitive gold."),
    ("doc31_qa0__after122", 0.0, "ANS refusal on definitive gold (quick ratio 1.57)."),
    ("doc134_qa0__after122", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc90_qa0__after122", 1.0, "ANS Consumer Health discontinued — exact."),
    ("doc85_qa0__after122", 1.0, "ANS No JnJ 1.3% — match."),
    ("doc27_qa0__after122", 0.0, "ANS refusal on definitive gold."),
    ("doc63_qa0__after122", 0.0, "ANS refusal on definitive gold (Boeing customers)."),
    # 1230-1239
    ("doc83_qa0__after123", 0.0, "ANS refusal on definitive gold."),
    ("doc0_qa0__after123", 0.0, "ANS refusal on definitive gold."),
    ("doc96_qa0__after123", 1.0, "ANS JPM gross margins not relevant — match."),
    ("doc47_qa0__after123", 0.0, "ANS refusal on definitive gold."),
    ("doc67_qa0__after123", 0.0, "ANS refusal on definitive gold."),
    ("doc2_qa0__after123", 0.0, "ANS refusal on definitive gold."),
    ("doc100_qa0__after123", 0.0, "ANS gold 1.33; predicted 0.73 — wrong specific."),
    ("doc45_qa0__after123", 0.0, "ANS refusal on definitive gold."),
    ("doc30_qa0__after123", 0.0, "ANS refusal on definitive gold."),
    ("doc117_qa0__after123", 0.0, "ANS refusal on definitive gold."),
    # 1240-1249
    ("doc20_qa0__after124", 0.0, "ANS refusal on definitive gold ($11,588)."),
    ("doc128_qa0__after124", 1.0, "ACK honest refusal — doc128 not yet ingested."),
    ("doc55_qa0__after124", 0.5, "ANS partial 'Gaming' — gets segment, no 9% growth."),
    ("doc24_qa0__after124", 0.0, "ANS refusal on definitive gold."),
    ("doc3_qa0__after124", 0.0, "ANS refusal on definitive gold."),
    ("doc139_qa0__after124", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc85_qa0__after124", 1.0, "ANS No JnJ 1.3% — match."),
    ("doc58_qa0__after124", 0.0, "ANS refusal on definitive gold."),
    ("doc71_qa0__after124", 0.0, "ANS gold 10.3%; predicted 15.0% — wrong specific."),
    ("doc81_qa0__after124", 0.0, "ANS refusal on definitive gold."),
    # 1250-1259
    ("doc1_qa0__after125", 0.0, "ANS refusal on definitive gold."),
    ("doc59_qa0__after125", 0.0, "ANS refusal on definitive gold."),
    ("doc97_qa0__after125", 0.0, "ANS refusal on definitive gold."),
    ("doc143_qa0__after125", 1.0, "ACK honest refusal — doc143 not yet ingested."),
    ("doc101_qa0__after125", 1.0, "ANS gold $5,818M Lockheed net working capital; predicted $5,818M — exact."),
    ("doc47_qa0__after125", 0.0, "ANS refusal on definitive gold."),
    ("doc19_qa0__after125", 0.0, "ANS refusal on definitive gold."),
    ("doc77_qa0__after125", 0.75, "ANS partial — CVS lawsuits about overcharging directors/officers — relates to legal battles, but specific gold mentions usual and customary pricing."),
    ("doc34_qa0__after125", 1.0, "ANS Xilinx amortization — match."),
    ("doc32_qa0__after125", 1.0, "ANS AMD products — match."),
    # 1260-1264
    ("doc132_qa0__after126", 1.0, "ACK honest refusal — doc132 not yet ingested."),
    ("doc130_qa0__after126", 1.0, "ACK honest refusal — doc130 not yet ingested."),
    ("doc41_qa0__after126", 0.0, "ANS refusal on definitive gold."),
    ("doc40_qa0__after126", 0.0, "ANS refusal on definitive gold."),
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
