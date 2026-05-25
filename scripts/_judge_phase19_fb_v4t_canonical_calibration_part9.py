"""Claude manual judging — Phase 1.9 FB calibration v4t-canonical (entries 1264-1499). FINAL BATCH."""

from __future__ import annotations

import json
from pathlib import Path

JUDGE_DIR = Path(
    "results/stage3/judge_queue/financebench__v4t-canonical__calibration__seed42"
)
QID_PREFIX = "financebench__v4t-canonical__calibration__"
QID_SUFFIX = "__seed42"

JUDGMENTS: list[tuple[str, float, str]] = [
    # 1264-1269
    ("doc66_qa0__after126", 0.0, "ANS refusal on definitive gold (Boeing tax rate)."),
    ("doc99_qa0__after126", 0.0, "ANS refusal on definitive gold."),
    ("doc7_qa0__after126", 1.0, "ANS Yes 3M 65th consecutive dividend — match (extra '$0.01 annual dividend' is wrong but Yes 65th is the answer)."),
    ("doc142_qa0__after126", 1.0, "ACK honest refusal — doc142 not yet ingested."),
    ("doc98_qa0__after126", 1.0, "ANS Yes decreased $7M VaR — match."),
    ("doc103_qa0__after126", 0.0, "ANS refusal on definitive gold."),
    # 1270-1279
    ("doc28_qa0__after127", 0.0, "ANS refusal on definitive gold (AMCOR EBITDA)."),
    ("doc130_qa0__after127", 1.0, "ACK honest refusal — doc130 not yet ingested."),
    ("doc62_qa0__after127", 0.0, "ANS refusal on definitive gold (Boeing gross margin)."),
    ("doc25_qa0__after127", 0.0, "ANS refusal on definitive gold (Amcor packaging)."),
    ("doc26_qa0__after127", 0.0, "ANS refusal on definitive gold."),
    ("doc80_qa0__after127", 1.0, "ANS Richard A. Johnson 16,105,005 votes — match."),
    ("doc135_qa0__after127", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc100_qa0__after127", 0.0, "ANS gold 1.33; predicted 1.00 — wrong specific (24% off)."),
    ("doc123_qa0__after127", 0.0, "ANS gold $9,068 PepsiCo capex; predicted $12,835M EBITDA (then truncated subtract capex calc) — wrong specific."),
    ("doc14_qa0__after127", 0.0, "ANS refusal on definitive gold (Adobe FCF)."),
    # 1280-1289
    ("doc72_qa0__after128", 1.0, "ANS Corning 20%→23% — exact."),
    ("doc131_qa0__after128", 1.0, "ACK honest refusal — doc131 not yet ingested."),
    ("doc106_qa0__after128", 0.0, "ANS refusal on definitive gold (Las Vegas EBITDAR)."),
    ("doc39_qa0__after128", 0.0, "ANS refusal on definitive gold (US/EMEA/APAC/LACC)."),
    ("doc117_qa0__after128", 0.0, "ANS refusal on definitive gold."),
    ("doc141_qa0__after128", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    ("doc32_qa0__after128", 0.0, "ANS refusal on definitive gold (AMD products)."),
    ("doc98_qa0__after128", 1.0, "ANS Yes decreased $7M — match."),
    ("doc41_qa0__after128", 0.0, "ANS refusal on definitive gold."),
    ("doc79_qa0__after128", 1.0, "ANS Mary Dillon Foot Locker CEO from Ulta — match."),
    # 1290-1299
    ("doc134_qa0__after129", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc42_qa0__after129", 0.0, "ANS refusal on definitive gold."),
    ("doc85_qa0__after129", 1.0, "ANS No JnJ 1.3% — match."),
    ("doc124_qa0__after129", 0.25, "ANS gold 16.5%; predicted 22.5% — confident wrong specific."),
    ("doc59_qa0__after129", 0.0, "ANS refusal on definitive gold."),
    ("doc123_qa0__after129", 0.0, "ANS gold $9,068; predicted $13,985M — wrong (EBITDA vs capex confusion)."),
    ("doc0_qa0__after129", 0.0, "ANS refusal on definitive gold."),
    ("doc38_qa0__after129", 1.0, "ANS gold 'There are none'; predicted refusal — effectively equivalent (no securities found)."),
    ("doc100_qa0__after129", 0.0, "ANS gold 1.33; predicted 0.93 — wrong specific."),
    ("doc146_qa0__after129", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    # 1300-1309
    ("doc122_qa0__after130", 1.0, "ANS gold $411M Pepsico restructuring; predicted 411 — exact!"),
    ("doc17_qa0__after130", 0.0, "ANS refusal on definitive gold."),
    ("doc78_qa0__after130", 0.5, "ANS gold Yes $0.55/quarter; predicted Yes paid dividends Q2 — partial."),
    ("doc38_qa0__after130", 1.0, "ANS 'There are none' equivalent refusal — match."),
    ("doc74_qa0__after130", 0.0, "ANS refusal on definitive gold."),
    ("doc86_qa0__after130", 0.0, "ANS refusal on definitive gold."),
    ("doc37_qa0__after130", 0.0, "ANS refusal on definitive gold (16% customer)."),
    ("doc42_qa0__after130", 0.0, "ANS refusal on definitive gold."),
    ("doc10_qa0__after130", 0.0, "ANS refusal on definitive gold."),
    ("doc101_qa0__after130", 1.0, "ANS Lockheed $5,818M — exact."),
    # 1310-1319
    ("doc26_qa0__after131", 0.0, "ANS refusal on definitive gold."),
    ("doc89_qa0__after131", 0.0, "ANS refusal on definitive gold."),
    ("doc3_qa0__after131", 0.0, "ANS refusal on definitive gold."),
    ("doc58_qa0__after131", 0.0, "ANS refusal on definitive gold."),
    ("doc71_qa0__after131", 0.0, "ANS refusal on definitive gold."),
    ("doc94_qa0__after131", 1.0, "ACK honest refusal — doc94 not yet ingested."),
    ("doc9_qa0__after131", 0.0, "ANS refusal on definitive gold."),
    ("doc18_qa0__after131", 0.0, "ANS refusal on definitive gold."),
    ("doc97_qa0__after131", 0.25, "ANS gold Corporate & Investment Bank; predicted 'Corporate segment' — partial (close but Corporate ≠ Corporate & Investment Bank)."),
    ("doc61_qa0__after131", 0.0, "ANS refusal on definitive gold."),
    # 1320-1329
    ("doc0_qa0__after132", 0.0, "ANS refusal on definitive gold."),
    ("doc120_qa0__after132", 0.0, "ANS refusal on definitive gold."),
    ("doc32_qa0__after132", 0.0, "ANS refusal on definitive gold."),
    ("doc141_qa0__after132", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    ("doc112_qa0__after132", 0.0, "ANS gold 5.4%; predicted 4.51% — outside tolerance."),
    ("doc43_qa0__after132", 0.0, "ANS refusal on definitive gold (Customer deposits)."),
    ("doc4_qa0__after132", 0.0, "ANS refusal on definitive gold."),
    ("doc126_qa0__after132", 1.0, "ANS gold $400M; predicted $400M increase $3.8B→$4.2B — match."),
    ("doc93_qa0__after132", 0.0, "ANS refusal on definitive gold."),
    ("doc9_qa0__after132", 0.0, "ANS refusal on definitive gold."),
    # 1330-1339
    ("doc75_qa0__after133", 0.0, "ANS refusal on definitive gold."),
    ("doc84_qa0__after133", 0.0, "ANS refusal on definitive gold."),
    ("doc19_qa0__after133", 0.0, "ANS refusal on definitive gold."),
    ("doc120_qa0__after133", 0.0, "ANS refusal on definitive gold."),
    ("doc76_qa0__after133", 0.0, "ANS refusal on definitive gold."),
    ("doc11_qa0__after133", 0.0, "ANS refusal on definitive gold."),
    ("doc86_qa0__after133", 0.0, "ANS refusal on definitive gold."),
    ("doc131_qa0__after133", 0.75, "ANS gold Yes Consumer Healthcare JV gain; predicted Yes contributed to net income increase — match Yes direction."),
    ("doc148_qa0__after133", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc117_qa0__after133", 0.0, "ANS refusal on definitive gold."),
    # 1340-1349
    ("doc80_qa0__after134", 0.0, "ANS gold Yes Richard A. Johnson definitive; predicted refusal — refusal on definitive gold."),
    ("doc143_qa0__after134", 1.0, "ACK honest refusal — doc143 not yet ingested."),
    ("doc20_qa0__after134", 0.0, "ANS refusal on definitive gold."),
    ("doc107_qa0__after134", 0.0, "ANS refusal on definitive gold."),
    ("doc15_qa0__after134", 1.0, "ANS 0 — exact."),
    ("doc134_qa0__after134", 1.0, "ANS gold Developed Rest of the World; predicted Developed Rest of World — match (minor 'the' difference)."),
    ("doc108_qa0__after134", 0.0, "ANS refusal on definitive gold."),
    ("doc114_qa0__after134", 1.0, "ANS gold 55.1%; predicted 56.3% — within tolerance."),
    ("doc109_qa0__after134", 0.0, "ANS refusal on definitive gold."),
    ("doc25_qa0__after134", 0.0, "ANS refusal on definitive gold."),
    # 1350-1359
    ("doc55_qa0__after135", 0.0, "ANS refusal on definitive gold."),
    ("doc60_qa0__after135", 0.0, "ANS refusal on definitive gold."),
    ("doc102_qa0__after135", 0.0, "ANS refusal on definitive gold."),
    ("doc88_qa0__after135", 0.25, "ANS 'Yes 3.5%' — gets 3.5% but Yes wrong direction (gold says No decelerate)."),
    ("doc86_qa0__after135", 0.0, "ANS refusal on definitive gold."),
    ("doc81_qa0__after135", 0.0, "ANS refusal on definitive gold."),
    ("doc118_qa0__after135", 0.0, "ANS refusal on definitive gold."),
    ("doc139_qa0__after135", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc127_qa0__after135", 0.25, "ANS gold $8.4B total; predicted $4.2B Five Year + potential $4.95B — partial (gets one tranche but not total $8.4B)."),
    ("doc10_qa0__after135", 0.0, "ANS refusal on definitive gold."),
    # 1360-1369
    ("doc115_qa0__after136", 0.0, "ANS refusal on definitive gold."),
    ("doc120_qa0__after136", 0.0, "ANS refusal on definitive gold."),
    ("doc27_qa0__after136", 0.0, "ANS refusal on definitive gold."),
    ("doc148_qa0__after136", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc108_qa0__after136", 0.0, "ANS refusal on definitive gold."),
    ("doc2_qa0__after136", 0.0, "ANS refusal on definitive gold."),
    ("doc58_qa0__after136", 0.0, "ANS refusal on definitive gold."),
    ("doc80_qa0__after136", 0.0, "ANS refusal on definitive gold."),
    ("doc63_qa0__after136", 0.0, "ANS refusal on definitive gold."),
    ("doc103_qa0__after136", 0.0, "ANS refusal on definitive gold."),
    # 1370-1379
    ("doc128_qa0__after137", 1.0, "ANS gold PepsiCo strong start; predicted 'raised guidance due to strong start to year' — match."),
    ("doc39_qa0__after137", 0.0, "ANS refusal on definitive gold."),
    ("doc60_qa0__after137", 0.0, "ANS refusal on definitive gold."),
    ("doc88_qa0__after137", 0.0, "ANS refusal on definitive gold."),
    ("doc134_qa0__after137", 1.0, "ANS Developed Rest of World — match."),
    ("doc135_qa0__after137", 0.75, "ANS gold Yes spinning Upjohn; predicted Yes $700M separating Upjohn — match Yes with extra detail."),
    ("doc113_qa0__after137", 0.0, "ANS refusal on definitive gold."),
    ("doc126_qa0__after137", 1.0, "ANS gold $400M; predicted $400M increase $3.8B→$4.2B — match."),
    ("doc18_qa0__after137", 0.0, "ANS refusal on definitive gold."),
    ("doc13_qa0__after137", 0.0, "ANS refusal on definitive gold."),
    # 1380-1389
    ("doc60_qa0__after138", 0.0, "ANS refusal on definitive gold."),
    ("doc39_qa0__after138", 0.0, "ANS refusal on definitive gold."),
    ("doc119_qa0__after138", 0.0, "ANS refusal on definitive gold."),
    ("doc142_qa0__after138", 1.0, "ACK honest refusal — doc142 not yet ingested."),
    ("doc35_qa0__after138", 0.0, "ANS refusal on definitive gold."),
    ("doc8_qa0__after138", 0.0, "ANS refusal on definitive gold."),
    ("doc131_qa0__after138", 0.75, "ANS Yes Consumer Healthcare JV gain — match."),
    ("doc67_qa0__after138", 0.0, "ANS refusal on definitive gold."),
    ("doc47_qa0__after138", 0.0, "ANS refusal on definitive gold."),
    ("doc3_qa0__after138", 0.0, "ANS refusal on definitive gold."),
    # 1390-1399
    ("doc148_qa0__after139", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc70_qa0__after139", 0.0, "ANS refusal on definitive gold (63.86 DPO)."),
    ("doc118_qa0__after139", 0.0, "ANS refusal on definitive gold."),
    ("doc39_qa0__after139", 0.0, "ANS refusal on definitive gold."),
    ("doc74_qa0__after139", 0.0, "ANS refusal on definitive gold."),
    ("doc12_qa0__after139", 0.0, "ANS refusal on definitive gold."),
    ("doc24_qa0__after139", 0.0, "ANS refusal on definitive gold."),
    ("doc25_qa0__after139", 0.0, "ANS refusal on definitive gold."),
    ("doc0_qa0__after139", 0.0, "ANS refusal on definitive gold."),
    ("doc92_qa0__after139", 0.0, "ANS refusal on definitive gold."),
    # 1400-1409
    ("doc5_qa0__after140", 0.0, "ANS refusal on definitive gold."),
    ("doc135_qa0__after140", 1.0, "ANS Yes Pfizer separating Upjohn $700M — match."),
    ("doc76_qa0__after140", 0.0, "ANS refusal on definitive gold."),
    ("doc26_qa0__after140", 0.0, "ANS refusal on definitive gold."),
    ("doc55_qa0__after140", 0.0, "ANS refusal on definitive gold."),
    ("doc58_qa0__after140", 0.0, "ANS refusal on definitive gold."),
    ("doc105_qa0__after140", 1.0, "ANS MGM $0.01/share — exact."),
    ("doc31_qa0__after140", 0.0, "ANS refusal on definitive gold."),
    ("doc123_qa0__after140", 0.0, "ANS refusal on definitive gold (PepsiCo unadjusted EBITDA-capex)."),
    ("doc3_qa0__after140", 0.0, "ANS refusal on definitive gold."),
    # 1410-1419
    ("doc62_qa0__after141", 0.0, "ANS refusal on definitive gold."),
    ("doc3_qa0__after141", 0.0, "ANS refusal on definitive gold."),
    ("doc38_qa0__after141", 1.0, "ANS 'There are none' equivalent refusal — match."),
    ("doc143_qa0__after141", 1.0, "ACK honest refusal — doc143 not yet ingested."),
    ("doc125_qa0__after141", 1.0, "ANS gold defeated; predicted defeated with specific vote count — match with detail."),
    ("doc87_qa0__after141", 0.0, "ANS refusal on definitive gold."),
    ("doc63_qa0__after141", 0.0, "ANS refusal on definitive gold."),
    ("doc69_qa0__after141", 0.0, "ANS refusal on definitive gold."),
    ("doc124_qa0__after141", 0.0, "ANS refusal on definitive gold."),
    ("doc17_qa0__after141", 0.0, "ANS refusal on definitive gold."),
    # 1420-1429
    ("doc34_qa0__after142", 0.0, "ANS refusal on definitive gold (Xilinx)."),
    ("doc102_qa0__after142", 0.0, "ANS refusal on definitive gold."),
    ("doc127_qa0__after142", 0.25, "ANS $4.2B Five Year + $4.95B potential — partial (gold is $8.4B total)."),
    ("doc146_qa0__after142", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc2_qa0__after142", 0.0, "ANS refusal on definitive gold."),
    ("doc113_qa0__after142", 0.0, "ANS refusal on definitive gold."),
    ("doc139_qa0__after142", 0.25, "ANS gold '47 new stores'; predicted 'change in operating assets... decrease of $104,233' — confident wrong (wrong direction and rationale)."),
    ("doc74_qa0__after142", 0.0, "ANS refusal on definitive gold."),
    ("doc132_qa0__after142", 0.0, "ANS refusal on definitive gold."),
    ("doc107_qa0__after142", 0.0, "ANS refusal on definitive gold."),
    # 1430-1439
    ("doc63_qa0__after143", 0.0, "ANS refusal on definitive gold."),
    ("doc45_qa0__after143", 0.0, "ANS refusal on definitive gold."),
    ("doc4_qa0__after143", 0.0, "ANS refusal on definitive gold."),
    ("doc141_qa0__after143", 0.0, "ANS gold 'increased'; predicted 'Decrease' — wrong direction."),
    ("doc93_qa0__after143", 0.0, "ANS refusal on definitive gold."),
    ("doc134_qa0__after143", 1.0, "ANS Developed Rest of World — match."),
    ("doc79_qa0__after143", 1.0, "ANS Mary Dillon Ulta retail — match."),
    ("doc138_qa0__after143", 1.0, "ANS gold 'lower marketing + leverage of incentive comp'; predicted same with extra detail (corporate deleverage, store payroll) — match."),
    ("doc11_qa0__after143", 0.0, "ANS refusal on definitive gold."),
    ("doc7_qa0__after143", 1.0, "ANS Yes 65th — match."),
    # 1440-1449
    ("doc86_qa0__after144", 0.0, "ANS refusal on definitive gold."),
    ("doc31_qa0__after144", 0.0, "ANS refusal on definitive gold."),
    ("doc139_qa0__after144", 0.25, "ANS '$104,233 decrease' confident wrong direction."),
    ("doc44_qa0__after144", 0.0, "ANS gold Yes; predicted refusal — refusal on definitive gold."),
    ("doc24_qa0__after144", 0.0, "ANS refusal on definitive gold."),
    ("doc97_qa0__after144", 0.0, "ANS refusal on definitive gold."),
    ("doc63_qa0__after144", 0.0, "ANS refusal on definitive gold."),
    ("doc110_qa0__after144", 0.0, "ANS refusal on definitive gold."),
    ("doc23_qa0__after144", 0.0, "ANS refusal on definitive gold."),
    ("doc78_qa0__after144", 0.5, "ANS partial Yes dividends Q2 — no $0.55."),
    # 1450-1459
    ("doc23_qa0__after145", 0.0, "ANS refusal on definitive gold."),
    ("doc110_qa0__after145", 0.0, "ANS refusal on definitive gold."),
    ("doc19_qa0__after145", 0.0, "ANS refusal on definitive gold."),
    ("doc20_qa0__after145", 0.0, "ANS refusal on definitive gold."),
    ("doc136_qa0__after145", 0.25, "ANS gold 'There are none' (no debt securities); predicted 'common stock par $0.01 NASDAQ ULTA' — wrong category (common stock vs debt securities)."),
    ("doc95_qa0__after145", 0.0, "ANS refusal on definitive gold."),
    ("doc119_qa0__after145", 0.0, "ANS refusal on definitive gold."),
    ("doc109_qa0__after145", 0.0, "ANS refusal on definitive gold."),
    ("doc62_qa0__after145", 0.0, "ANS refusal on definitive gold."),
    ("doc12_qa0__after145", 0.0, "ANS refusal on definitive gold."),
    # 1460-1469
    ("doc111_qa0__after146", 0.0, "ANS refusal on definitive gold."),
    ("doc51_qa0__after146", 0.0, "ANS refusal on definitive gold."),
    ("doc10_qa0__after146", 0.0, "ANS refusal on definitive gold."),
    ("doc64_qa0__after146", 1.0, "ANS Yes Boeing cyclical — match."),
    ("doc139_qa0__after146", 0.25, "ANS '$104,233 decrease' confident wrong direction."),
    ("doc24_qa0__after146", 0.0, "ANS refusal on definitive gold."),
    ("doc98_qa0__after146", 1.0, "ANS Yes decreased $7M — match."),
    ("doc5_qa0__after146", 0.0, "ANS refusal on definitive gold."),
    ("doc13_qa0__after146", 0.0, "ANS refusal on definitive gold."),
    ("doc53_qa0__after146", 0.0, "ANS refusal on definitive gold."),
    # 1470-1479
    ("doc25_qa0__after147", 0.0, "ANS refusal on definitive gold."),
    ("doc24_qa0__after147", 0.0, "ANS refusal on definitive gold."),
    ("doc35_qa0__after147", 0.0, "ANS refusal on definitive gold."),
    ("doc22_qa0__after147", 0.0, "ANS refusal on definitive gold."),
    ("doc117_qa0__after147", 0.0, "ANS refusal on definitive gold."),
    ("doc26_qa0__after147", 0.0, "ANS refusal on definitive gold."),
    ("doc141_qa0__after147", 0.0, "ANS 'Decrease' wrong direction."),
    ("doc83_qa0__after147", 0.0, "ANS refusal on definitive gold."),
    ("doc102_qa0__after147", 0.0, "ANS refusal on definitive gold."),
    ("doc111_qa0__after147", 0.0, "ANS refusal on definitive gold."),
    # 1480-1489
    ("doc140_qa0__after148", 1.0, "ANS gold 36% Ulta stock repurchases; predicted ~36.5% — within tolerance."),
    ("doc107_qa0__after148", 0.0, "ANS refusal on definitive gold."),
    ("doc38_qa0__after148", 1.0, "ANS 'There are none' equivalent refusal — match."),
    ("doc59_qa0__after148", 0.0, "ANS refusal on definitive gold."),
    ("doc120_qa0__after148", 0.0, "ANS refusal on definitive gold."),
    ("doc127_qa0__after148", 0.25, "ANS partial — $4.2B + $4.95B potential, no $8.4B."),
    ("doc77_qa0__after148", 0.0, "ANS refusal on definitive gold."),
    ("doc118_qa0__after148", 0.0, "ANS refusal on definitive gold."),
    ("doc85_qa0__after148", 1.0, "ANS No JnJ 1.3% — match."),
    ("doc137_qa0__after148", 1.0, "ANS gold no Ulta acquisitions; predicted 'do not mention any major acquisitions' — equivalent."),
    # 1490-1499
    ("doc90_qa0__after149", 1.0, "ANS Consumer Health discontinued — exact."),
    ("doc82_qa0__after149", 0.0, "ANS refusal on definitive gold."),
    ("doc63_qa0__after149", 0.5, "ANS partial — commercial airlines + gov + defense (some specifics, no 40%)."),
    ("doc109_qa0__after149", 0.75, "ANS gold corporate bonds ~82%; predicted 'Corporate bonds' — partial (right answer no 82%)."),
    ("doc61_qa0__after149", 1.0, "ANS Lion Air + Ethiopian crashes — match."),
    ("doc55_qa0__after149", 0.0, "ANS refusal on definitive gold."),
    ("doc80_qa0__after149", 1.0, "ANS Richard A. Johnson — match."),
    ("doc105_qa0__after149", 1.0, "ANS MGM $0.01/share — exact."),
    ("doc108_qa0__after149", 0.25, "ANS gold MGM China; predicted 'Emerging Markets' — wrong region."),
    ("doc128_qa0__after149", 1.0, "ANS PepsiCo raised guidance strong start with 8% organic / 9% EPS — match."),
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
