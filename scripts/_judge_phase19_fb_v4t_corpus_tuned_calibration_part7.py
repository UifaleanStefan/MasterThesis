"""Claude manual judging — Phase 1.9 FB calibration v4t-corpus-tuned (entries 1200-1499).

Final part. Idempotent append. All scores by Claude per HARD RULE.
"""

from __future__ import annotations

import json
from pathlib import Path

JUDGE_DIR = Path(
    "results/stage3/judge_queue/financebench__v4t-corpus-tuned__calibration__seed42"
)

QID_PREFIX = "financebench__v4t-corpus-tuned__calibration__seed42::"

JUDGMENTS: list[tuple[str, float, str]] = [
    # 1200-1209
    ("doc54_qa0__after120", 1.0, "ANS Best Buy 982→969 — exact."),
    ("doc121_qa0__after120", 0.0, "ACK calibration: confident hallucinated 'Yes PepsiCo facing drug pricing lawsuits' (gold says No not involved in material legal battles)."),
    ("doc112_qa0__after120", 0.0, "ANS gold 5.4%; predicted 4.5% — outside 5% tolerance."),
    ("doc117_qa0__after120", 1.0, "ANS gold cash flow from operations highest Nike FY23; predicted $5,841M operating — match."),
    ("doc3_qa0__after120", 0.75, "ANS gold -1.7% reasons (PFAS, Russia, restructuring); predicted same items + PFAS + Russia — matches reasons, no -1.7%."),
    ("doc0_qa0__after120", 1.0, "ANS $1,501M vs $1,577 — within tolerance."),
    ("doc99_qa0__after120", 0.0, "ANS gold 6.25; predicted 3.09 — wrong specific."),
    ("doc88_qa0__after120", 0.0, "ANS gold No (decelerate); predicted Yes 12.5% — wrong direction."),
    ("doc34_qa0__after120", 1.0, "ANS Xilinx amortization — match."),
    ("doc72_qa0__after120", 1.0, "ANS Corning tax 20%→23% — exact."),
    # 1210-1219
    ("doc114_qa0__after121", 1.0, "ANS gold 55.1%; predicted 56.3% — within 2.2% tolerance."),
    ("doc127_qa0__after121", 0.25, "ACK calibration: confident wrong specific '$4.0B' (gold $8.4B PepsiCo unsecured credit)."),
    ("doc11_qa0__after121", 1.0, "ANS gold 65.4%; predicted 65.3% — within tolerance."),
    ("doc136_qa0__after121", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc46_qa0__after121", 1.0, "ANS 1,832 — exact."),
    ("doc92_qa0__after121", 1.0, "ANS JnJ Kenvue $13.2B — exact."),
    ("doc115_qa0__after121", 1.0, "ANS $16,525 — exact."),
    ("doc49_qa0__after121", 1.0, "ANS 5,409 — exact."),
    ("doc72_qa0__after121", 1.0, "ANS Corning tax — exact."),
    ("doc106_qa0__after121", 0.5, "ANS gold Las Vegas ~90% EBITDAR; predicted just 'Las Vegas Strip Resorts' — partial (right region, no 90%)."),
    # 1220-1229
    ("doc12_qa0__after122", 0.0, "ANS 0.83 vs 1.25 — wrong specific."),
    ("doc125_qa0__after122", 1.0, "ACK 'proposal not approved with 66.5% against' — correct + extra specifics."),
    ("doc128_qa0__after122", 1.0, "ACK honest refusal — doc128 not yet ingested."),
    ("doc67_qa0__after122", 0.0, "ANS 1.43% vs 0.01 — wrong specific."),
    ("doc31_qa0__after122", 0.0, "ANS refusal on definitive gold (quick ratio)."),
    ("doc134_qa0__after122", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc90_qa0__after122", 1.0, "ANS Consumer Health discontinued — exact."),
    ("doc85_qa0__after122", 1.0, "ANS No 1.3% — exact."),
    ("doc27_qa0__after122", 0.5, "ANS restructuring partial — no 87%."),
    ("doc63_qa0__after122", 0.5, "ANS partial — commercial airlines without US govt 40%."),
    # 1230-1239
    ("doc83_qa0__after123", 1.0, "ANS gold $3,215; predicted $3,115.4M — within 3.1% tolerance."),
    ("doc0_qa0__after123", 1.0, "ANS $1,501M — within tolerance."),
    ("doc96_qa0__after123", 1.0, "ANS JPM gross margins not relevant — match."),
    ("doc47_qa0__after123", 0.5, "ANS confused — Yes positive but describes -$1,561M."),
    ("doc67_qa0__after123", 0.0, "ANS 1.43% vs 0.01 — wrong specific."),
    ("doc2_qa0__after123", 0.0, "ANS 'Yes capital-intensive' wrong direction."),
    ("doc100_qa0__after123", 1.0, "ANS gold 1.33; predicted 1.36 — within 2.3% tolerance."),
    ("doc45_qa0__after123", 1.0, "ANS $0.389B vs $0.40 — within 2.75% tolerance."),
    ("doc30_qa0__after123", 1.0, "ANS 4.18% — within tolerance."),
    ("doc117_qa0__after123", 1.0, "ANS Nike cash from operations $5,841M — match."),
    # 1240-1249
    ("doc20_qa0__after124", 1.0, "ANS 11,588 — exact."),
    ("doc128_qa0__after124", 1.0, "ACK honest refusal — doc128 not yet ingested."),
    ("doc55_qa0__after124", 1.0, "ANS gold entertainment 9% gaming; predicted Gaming 9% — match (gaming = entertainment driver)."),
    ("doc24_qa0__after124", 0.0, "ANS Amcor acquisitions refusal on definitive gold."),
    ("doc3_qa0__after124", 0.75, "ANS 3M operating margin reasons + PFAS + Russia — partial, no -1.7%."),
    ("doc139_qa0__after124", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc85_qa0__after124", 1.0, "ANS No 1.3% — exact."),
    ("doc58_qa0__after124", 1.0, "ANS $381.6M — within tolerance."),
    ("doc71_qa0__after124", 1.0, "ANS 10.5% vs 10.3% — within tolerance."),
    ("doc81_qa0__after124", 0.5, "ANS gold -3.7 CCC General Mills; predicted truncated calculation setup, no final number — hedged."),
    # 1250-1259
    ("doc1_qa0__after125", 1.0, "ANS $8.738B vs $8.70 — within tolerance."),
    ("doc59_qa0__after125", 1.0, "ANS $12,645 — exact."),
    ("doc97_qa0__after125", 0.0, "ANS Consumer & Community Banking wrong."),
    ("doc143_qa0__after125", 1.0, "ACK honest refusal — doc143 not yet ingested."),
    ("doc101_qa0__after125", 1.0, "ANS $5,818M — exact."),
    ("doc47_qa0__after125", 0.5, "ANS confused — Yes positive but describes -$1,561M."),
    ("doc19_qa0__after125", 1.0, "ANS 30.7% vs 30.8% — within tolerance."),
    ("doc77_qa0__after125", 0.75, "ANS partial — subset of CVS legal items."),
    ("doc34_qa0__after125", 1.0, "ANS Xilinx — match."),
    ("doc32_qa0__after125", 1.0, "ANS AMD products — match."),
    # 1260-1269
    ("doc132_qa0__after126", 1.0, "ACK honest refusal — doc132 not yet ingested."),
    ("doc130_qa0__after126", 1.0, "ACK honest refusal — doc130 not yet ingested."),
    ("doc41_qa0__after126", 1.0, "ANS gross margin not useful AMEX — match."),
    ("doc40_qa0__after126", 1.0, "ANS operating margin not useful — match."),
    ("doc66_qa0__after126", 0.5, "ANS effective tax direction right, no specific rates."),
    ("doc99_qa0__after126", 1.0, "ANS 6.20 vs 6.25 — within tolerance."),
    ("doc7_qa0__after126", 1.0, "ANS Yes 65th — match."),
    ("doc142_qa0__after126", 1.0, "ACK honest refusal — doc142 not yet ingested."),
    ("doc98_qa0__after126", 1.0, "ANS Yes decreased $7M — match."),
    ("doc103_qa0__after126", 1.0, "ANS $302.6M vs $303 — within tolerance."),
    # 1270-1279
    ("doc28_qa0__after127", 1.0, "ANS $2,018M — exact."),
    ("doc130_qa0__after127", 1.0, "ACK honest refusal — doc130 not yet ingested."),
    ("doc62_qa0__after127", 0.0, "ANS 'gross margin not useful' wrong direction."),
    ("doc25_qa0__after127", 1.0, "ANS Amcor packaging — match."),
    ("doc26_qa0__after127", 0.75, "ANS Amcor declining — direction right."),
    ("doc80_qa0__after127", 1.0, "ANS Richard A. Johnson 16,105,005 votes — match."),
    ("doc135_qa0__after127", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc100_qa0__after127", 1.0, "ANS 1.31 vs 1.33 — within tolerance."),
    ("doc123_qa0__after127", 0.0, "ANS gold $9,068 PepsiCo capex FY2022; predicted $14,275M — wrong (this is EBITDA, not capex)."),
    ("doc14_qa0__after127", 0.0, "ANS Adobe FCF refusal on definitive gold."),
    # 1280-1289
    ("doc72_qa0__after128", 1.0, "ANS Corning 20%→23% — exact."),
    ("doc131_qa0__after128", 1.0, "ACK honest refusal — doc131 not yet ingested."),
    ("doc106_qa0__after128", 0.5, "ANS partial — 'Las Vegas Strip Resorts' without 90% EBITDAR."),
    ("doc39_qa0__after128", 1.0, "ANS US/EMEA/APAC/LACC + Other — match."),
    ("doc117_qa0__after128", 1.0, "ANS Nike $5,841M operations — match."),
    ("doc141_qa0__after128", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    ("doc32_qa0__after128", 1.0, "ANS AMD products — match."),
    ("doc98_qa0__after128", 1.0, "ANS Yes decreased $7M — match."),
    ("doc41_qa0__after128", 1.0, "ANS gross margin not useful — match."),
    ("doc79_qa0__after128", 1.0, "ANS Mary Dillon retail similar — match."),
    # 1290-1299
    ("doc134_qa0__after129", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc42_qa0__after129", 1.0, "ANS AMEX tax — match."),
    ("doc85_qa0__after129", 1.0, "ANS No 1.3% — match."),
    ("doc124_qa0__after129", 1.0, "ANS gold 16.5% PepsiCo unadjusted EBITDA % margin; predicted '$14,275M = 16.5%' — exact match."),
    ("doc59_qa0__after129", 1.0, "ANS $12,645 — exact."),
    ("doc123_qa0__after129", 0.0, "ANS gold $9,068 capex; predicted $14,275M EBITDA — wrong (different metric)."),
    ("doc0_qa0__after129", 1.0, "ANS $1,501M — within tolerance."),
    ("doc38_qa0__after129", 0.0, "ANS gold 'There are none'; predicted 'Common Shares' — wrong."),
    ("doc100_qa0__after129", 1.0, "ANS 1.36 vs 1.33 — within tolerance."),
    ("doc146_qa0__after129", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    # 1300-1309
    ("doc122_qa0__after130", 1.0, "ANS gold $411M Pepsico restructuring; predicted 411 — exact!"),
    ("doc17_qa0__after130", 0.0, "ANS -0.02 vs -1.32 — wrong specific."),
    ("doc78_qa0__after130", 1.0, "ANS gold Yes $0.55/quarter; predicted Yes $0.55/share quarterly — match."),
    ("doc38_qa0__after130", 0.0, "ANS 'Common Shares' wrong."),
    ("doc74_qa0__after130", 1.0, "ANS 59,268 — exact."),
    ("doc86_qa0__after130", 0.0, "ANS 'Gross margin not useful' wrong (gold gives specific COVID-19 reasons)."),
    ("doc37_qa0__after130", 1.0, "ANS one customer 16% — match."),
    ("doc42_qa0__after130", 1.0, "ANS AMEX tax — match."),
    ("doc10_qa0__after130", 0.0, "ANS 0.66 vs 1.73 — wrong specific."),
    ("doc101_qa0__after130", 1.0, "ANS $5,818M — exact."),
    # 1310-1319
    ("doc26_qa0__after131", 0.75, "ANS Amcor declining — direction right."),
    ("doc89_qa0__after131", 1.0, "ANS US 3.0% intl -0.6% — match."),
    ("doc3_qa0__after131", 0.75, "ANS 3M PFAS Russia — partial, no -1.7%."),
    ("doc58_qa0__after131", 1.0, "ANS $381.6M — within tolerance."),
    ("doc71_qa0__after131", 1.0, "ANS 10.5% vs 10.3% — within tolerance."),
    ("doc94_qa0__after131", 0.0, "ANS Consumer & Community wrong (gold Corporate -$473M)."),
    ("doc9_qa0__after131", 0.0, "ANS 3.5% vs 1.9% — wrong specific."),
    ("doc18_qa0__after131", 0.0, "ANS 36.12 vs 93.86 — wrong specific."),
    ("doc97_qa0__after131", 0.0, "ANS Consumer & Community wrong."),
    ("doc61_qa0__after131", 1.0, "ANS Lion Air + Ethiopian crashes — match."),
    # 1320-1329
    ("doc0_qa0__after132", 1.0, "ANS $1,501M — within tolerance."),
    ("doc120_qa0__after132", 1.0, "ANS gold NA/LA/Europe/AMESA/APAC; predicted same with full names — match."),
    ("doc32_qa0__after132", 1.0, "ANS AMD products — match."),
    ("doc141_qa0__after132", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    ("doc112_qa0__after132", 0.0, "ANS gold 5.4%; predicted 4.5% — outside tolerance."),
    ("doc43_qa0__after132", 0.0, "ANS Long-term debt wrong (gold Customer deposits)."),
    ("doc4_qa0__after132", 0.5, "ANS partial — Consumer segment without 0.9% figure."),
    ("doc126_qa0__after132", 1.0, "ANS gold $400M increase; predicted '$400,000,000 from $3.8B to $4.2B' — exact."),
    ("doc93_qa0__after132", 1.0, "ANS 20.0%→20.1% — exact."),
    ("doc9_qa0__after132", 0.0, "ANS 3.5% wrong."),
    # 1330-1339
    ("doc75_qa0__after133", 0.0, "ANS 8.99 vs 17.98 — wrong specific."),
    ("doc84_qa0__after133", 0.0, "ANS gold 0.54; predicted 0.46 — 15% off, outside tolerance."),
    ("doc19_qa0__after133", 1.0, "ANS 30.7% vs 30.8% — within tolerance."),
    ("doc120_qa0__after133", 1.0, "ANS PepsiCo regions NA/LA/Europe/AMESA/APAC — match."),
    ("doc76_qa0__after133", 1.0, "ANS Yes CVS capital-intensive — direction match."),
    ("doc11_qa0__after133", 1.0, "ANS gold 65.4%; predicted truncated calc but computes 590,507/903,095*100 ≈ 65.4% — within tolerance."),
    ("doc86_qa0__after133", 0.0, "ANS 'Gross margin not useful' wrong (gold gives specific reasons)."),
    ("doc131_qa0__after133", 0.75, "ANS gold Yes gain on Consumer Healthcare JV; predicted Yes with $(6)M specifier — match Yes with extra detail."),
    ("doc148_qa0__after133", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc117_qa0__after133", 1.0, "ANS Nike $5,841M operations — match."),
    # 1340-1349
    ("doc80_qa0__after134", 1.0, "ANS Richard A. Johnson — match."),
    ("doc143_qa0__after134", 1.0, "ACK honest refusal — doc143 not yet ingested."),
    ("doc20_qa0__after134", 1.0, "ANS 11,588 — exact."),
    ("doc107_qa0__after134", 0.0, "ANS 1.61 vs 0 (zero due to negative EBIT) — wrong specific."),
    ("doc15_qa0__after134", 1.0, "ANS 0 — exact."),
    ("doc134_qa0__after134", 1.0, "ANS gold Developed Rest of the World; predicted Developed Rest of World — match (minor 'the' difference)."),
    ("doc108_qa0__after134", 0.75, "ANS gold MGM China worst 44% decline; predicted MGM China with $(203,136)k — right region, different metric."),
    ("doc114_qa0__after134", 1.0, "ANS 56.3% vs 55.1% — within tolerance."),
    ("doc109_qa0__after134", 1.0, "ANS gold corporate bonds 82%; predicted corporate bonds $416,420,000 — right answer + specific."),
    ("doc25_qa0__after134", 1.0, "ANS Amcor packaging — match."),
    # 1350-1359
    ("doc55_qa0__after135", 1.0, "ANS Gaming 9% — match (gaming = entertainment driver)."),
    ("doc60_qa0__after135", 1.0, "ANS Commercial Airplanes — match."),
    ("doc102_qa0__after135", 1.0, "ANS 0.4% Lockheed CAGR — exact."),
    ("doc88_qa0__after135", 0.0, "ANS Yes 12.5% wrong direction."),
    ("doc86_qa0__after135", 0.0, "ANS 'Gross margin not useful' wrong."),
    ("doc81_qa0__after135", 0.5, "ANS gold -3.7; predicted truncated CCC calculation setup, no final — hedged."),
    ("doc118_qa0__after135", 0.5, "ANS gold Yes PayPal positive WC $1.6Bn; predicted Yes positive $12,416M — direction right, specific number ~7.7× off."),
    ("doc139_qa0__after135", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc127_qa0__after135", 1.0, "ANS PepsiCo $8.4B unsecured credit — exact."),
    ("doc10_qa0__after135", 0.0, "ANS 1.96 vs 0.66 — wrong specific."),
    # 1360-1369
    ("doc115_qa0__after136", 1.0, "ANS $16,525 — exact."),
    ("doc120_qa0__after136", 0.25, "ANS gold PepsiCo NA/LA/Europe/AMESA/APAC; predicted 'US, Developed Europe, Developed Rest of World, Emerging Markets' — wrong regional dimensions (uses developed/emerging framework)."),
    ("doc27_qa0__after136", 0.5, "ANS restructuring partial — no 87%."),
    ("doc148_qa0__after136", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc108_qa0__after136", 0.75, "ANS MGM China worst — direction right, different metric."),
    ("doc2_qa0__after136", 0.0, "ANS 'Yes capital-intensive' wrong."),
    ("doc58_qa0__after136", 1.0, "ANS $381.6M — within tolerance."),
    ("doc80_qa0__after136", 1.0, "ANS Richard A. Johnson — match."),
    ("doc63_qa0__after136", 0.5, "ANS partial — commercial airlines without US govt 40%."),
    ("doc103_qa0__after136", 1.0, "ANS $302.578M vs $303 — within tolerance."),
    # 1370-1379
    ("doc128_qa0__after137", 1.0, "ANS gold PepsiCo strong start FY2023; predicted 'PepsiCo raised guidance due to strong start' — match."),
    ("doc39_qa0__after137", 1.0, "ANS US/EMEA/APAC/LACC — match."),
    ("doc60_qa0__after137", 1.0, "ANS Commercial Airplanes — match."),
    ("doc88_qa0__after137", 0.0, "ANS Yes 12.5% wrong direction."),
    ("doc134_qa0__after137", 1.0, "ANS Developed Rest of World — match."),
    ("doc135_qa0__after137", 1.0, "ANS gold Yes Pfizer spinning Upjohn; predicted Yes Pfizer separating Upjohn — match."),
    ("doc113_qa0__after137", 1.0, "ANS 5,466.3M vs $5,466 — within tolerance."),
    ("doc126_qa0__after137", 1.0, "ANS PepsiCo $400M credit increase — exact."),
    ("doc18_qa0__after137", 0.0, "ANS 30.73 vs 93.86 — wrong specific."),
    ("doc13_qa0__after137", 0.0, "ANS 'Yes improving' wrong (gold No declined)."),
    # 1380-1389
    ("doc60_qa0__after138", 1.0, "ANS Commercial Airplanes — match."),
    ("doc39_qa0__after138", 1.0, "ANS US/EMEA/APAC/LACC — match."),
    ("doc119_qa0__after138", 1.0, "ANS gold $4.60; predicted $4.625B — within tolerance."),
    ("doc142_qa0__after138", 1.0, "ACK honest refusal — doc142 not yet ingested."),
    ("doc35_qa0__after138", 1.0, "ANS AMD $3,565M — match."),
    ("doc8_qa0__after138", 0.25, "ANS gold 24.26; predicted 25.73 — 6.06% off, marginally outside 5% tolerance."),
    ("doc131_qa0__after138", 0.75, "ANS Yes gain on Consumer Healthcare JV — match Yes."),
    ("doc67_qa0__after138", 0.0, "ANS 1.43% vs 0.01 — wrong specific."),
    ("doc47_qa0__after138", 0.5, "ANS confused — Yes positive but describes -$1,561M."),
    ("doc3_qa0__after138", 0.75, "ANS 3M operating margin items — partial, no -1.7%."),
    # 1390-1399
    ("doc148_qa0__after139", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc70_qa0__after139", 1.0, "ANS 66.67 vs 63.86 — within 4.4% tolerance."),
    ("doc118_qa0__after139", 0.5, "ANS PayPal WC $12,416M vs $1.6Bn — different methodology."),
    ("doc39_qa0__after139", 1.0, "ANS US/EMEA/APAC/LACC — match."),
    ("doc74_qa0__after139", 1.0, "ANS 59,268 — exact."),
    ("doc12_qa0__after139", 0.0, "ANS 0.83 vs 1.25 — wrong specific."),
    ("doc24_qa0__after139", 0.0, "ANS Amcor acquisitions refusal on definitive gold."),
    ("doc25_qa0__after139", 1.0, "ANS Amcor packaging — match."),
    ("doc0_qa0__after139", 1.0, "ANS $1,501M — within tolerance."),
    ("doc92_qa0__after139", 1.0, "ANS $13.2B — exact."),
    # 1400-1409
    ("doc5_qa0__after140", 0.0, "ANS 3M quick ratio 0.96 refusal on definitive gold."),
    ("doc135_qa0__after140", 1.0, "ANS Yes Pfizer separating Upjohn — match."),
    ("doc76_qa0__after140", 1.0, "ANS Yes CVS capital-intensive — direction match."),
    ("doc26_qa0__after140", 0.75, "ANS Amcor declining — direction right."),
    ("doc55_qa0__after140", 1.0, "ANS Gaming 9% — match."),
    ("doc58_qa0__after140", 1.0, "ANS $381.6M — within tolerance."),
    ("doc105_qa0__after140", 1.0, "ANS MGM $0.01/share — exact."),
    ("doc31_qa0__after140", 0.0, "ANS quick ratio refusal on definitive gold."),
    ("doc123_qa0__after140", 0.0, "ANS gold $9,068 capex; predicted $14,275M (EBITDA) — wrong metric."),
    ("doc3_qa0__after140", 0.75, "ANS 3M PFAS Russia — partial."),
    # 1410-1419
    ("doc62_qa0__after141", 0.0, "ANS 'gross margin not useful' wrong."),
    ("doc3_qa0__after141", 0.5, "ANS 3M generic items (litigation, impairment, restructuring) — partial, no PFAS or Russia specifics."),
    ("doc38_qa0__after141", 0.0, "ANS 'Common Shares' wrong (gold 'There are none')."),
    ("doc143_qa0__after141", 1.0, "ACK honest refusal — doc143 not yet ingested."),
    ("doc125_qa0__after141", 1.0, "ANS gold defeated; predicted 'defeated with 19,718,780 for and 977,228,788 against' — match with specifics."),
    ("doc87_qa0__after141", 0.0, "ANS JnJ inventory refusal on definitive gold."),
    ("doc63_qa0__after141", 0.5, "ANS partial — no US govt 40%."),
    ("doc69_qa0__after141", 1.0, "ANS 0.80 vs 0.8 — exact."),
    ("doc124_qa0__after141", 1.0, "ANS 16.5% — exact."),
    ("doc17_qa0__after141", 0.0, "ANS -1.42 vs -0.02 — wrong specific."),
    # 1420-1429
    ("doc34_qa0__after142", 1.0, "ANS Xilinx — match."),
    ("doc102_qa0__after142", 1.0, "ANS 0.4% — exact."),
    ("doc127_qa0__after142", 1.0, "ANS $8.4B PepsiCo credit — exact."),
    ("doc146_qa0__after142", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc2_qa0__after142", 0.0, "ANS 'Yes capital-intensive' wrong."),
    ("doc113_qa0__after142", 1.0, "ANS 5,466.3M — within tolerance."),
    ("doc139_qa0__after142", 1.0, "ANS Ulta 47 new stores + brand launches — match."),
    ("doc74_qa0__after142", 1.0, "ANS 59,268 — exact."),
    ("doc132_qa0__after142", 0.5, "ANS gold Trillium/Array/Therachon Pfizer acquisitions; predicted Trillium/Array/Upjohn — 2/3 right (Upjohn wrong)."),
    ("doc107_qa0__after142", 0.0, "ANS 1.61 vs 0 (negative EBIT) — wrong."),
    # 1430-1439
    ("doc63_qa0__after143", 0.5, "ANS partial — no US govt 40%."),
    ("doc45_qa0__after143", 1.0, "ANS $0.389B — within tolerance."),
    ("doc4_qa0__after143", 0.5, "ANS partial."),
    ("doc141_qa0__after143", 0.0, "ANS gold 'increased'; predicted 'Decrease' — wrong direction."),
    ("doc93_qa0__after143", 1.0, "ANS 20.0%→20.1% — exact."),
    ("doc134_qa0__after143", 1.0, "ANS Developed Rest of World — match."),
    ("doc79_qa0__after143", 1.0, "ANS Mary Dillon retail similar — match."),
    ("doc138_qa0__after143", 1.0, "ANS gold 'lower marketing + leverage of incentive comp'; predicted 'lower marketing expenses + leverage of incentive compensation due to higher sales' — exact match!"),
    ("doc11_qa0__after143", 1.0, "ANS gold 65.4%; predicted truncated calc computing 590,507/903,095 ≈ 65.4% — within tolerance."),
    ("doc7_qa0__after143", 1.0, "ANS Yes 65th — match."),
    # 1440-1449
    ("doc86_qa0__after144", 0.75, "ANS gold COVID-19 + currency + commodity inflation; predicted 'cost of products sold $29.9B→$31.1B + unfavorable FX + reduced COVID-19 V...' — mentions FX and COVID-19, no commodity inflation."),
    ("doc31_qa0__after144", 0.0, "ANS quick ratio refusal on definitive gold."),
    ("doc139_qa0__after144", 1.0, "ANS Ulta 47 new stores + brand launches — match."),
    ("doc44_qa0__after144", 1.0, "ANS Yes — match."),
    ("doc24_qa0__after144", 0.0, "ANS Amcor acquisitions refusal on definitive gold."),
    ("doc97_qa0__after144", 0.0, "ANS Consumer & Community wrong."),
    ("doc63_qa0__after144", 0.5, "ANS partial — no US govt 40%."),
    ("doc110_qa0__after144", 1.0, "ANS gold $32,780; predicted $32,780 — exact."),
    ("doc23_qa0__after144", 0.0, "ANS quick ratio refusal on definitive gold."),
    ("doc78_qa0__after144", 1.0, "ANS gold Yes $0.55; predicted Yes $0.55/share — match."),
    # 1450-1459
    ("doc23_qa0__after145", 0.0, "ANS quick ratio refusal on definitive gold."),
    ("doc110_qa0__after145", 1.0, "ANS 32,780 — exact."),
    ("doc19_qa0__after145", 1.0, "ANS gold 30.8%; predicted '30.8%' with full calc — exact."),
    ("doc20_qa0__after145", 1.0, "ANS $11,588 — exact."),
    ("doc136_qa0__after145", 1.0, "ANS gold 'There are none'; predicted 'no debt securities registered' — match."),
    ("doc95_qa0__after145", 0.0, "ANS gold $66.56/share; predicted $239.45/share — wrong specific."),
    ("doc119_qa0__after145", 1.0, "ANS $4.625B vs $4.60 — within tolerance."),
    ("doc109_qa0__after145", 1.0, "ANS corporate bonds $416,420,000 — match."),
    ("doc62_qa0__after145", 0.0, "ANS 'gross margin not useful' wrong."),
    ("doc12_qa0__after145", 0.0, "ANS 1.25 vs 0.83 — wrong specific."),
    # 1460-1469
    ("doc111_qa0__after146", 0.75, "ANS gold No Microsoft -$2.5bn; predicted 'Yes decreased long-term debt $47B→$42B' — direction correct (decreased), wording slightly mismatched."),
    ("doc51_qa0__after146", 1.0, "ANS Best Buy acquisitions — match."),
    ("doc10_qa0__after146", 0.0, "ANS 1.87 vs 0.66 — wrong specific."),
    ("doc64_qa0__after146", 1.0, "ANS Yes Boeing cyclical — match."),
    ("doc139_qa0__after146", 1.0, "ANS Ulta 47 stores + brand launches — match."),
    ("doc24_qa0__after146", 0.0, "ANS Amcor acquisitions refusal on definitive gold."),
    ("doc98_qa0__after146", 1.0, "ANS Yes decreased $7M — match."),
    ("doc5_qa0__after146", 0.0, "ANS 3M quick ratio refusal on definitive gold."),
    ("doc13_qa0__after146", 0.0, "ANS 'Yes improving' wrong."),
    ("doc53_qa0__after146", 1.0, "ANS gold ~42% decline; predicted $1,874M→$1,093M (41.7%) — within tolerance."),
    # 1470-1479
    ("doc25_qa0__after147", 1.0, "ANS Amcor packaging — match."),
    ("doc24_qa0__after147", 0.0, "ANS Amcor acquisitions refusal on definitive gold."),
    ("doc35_qa0__after147", 1.0, "ANS AMD $3,565M — match."),
    ("doc22_qa0__after147", 1.0, "ANS Amcor 8K — match."),
    ("doc117_qa0__after147", 1.0, "ANS Nike $5,841M — match."),
    ("doc26_qa0__after147", 0.75, "ANS Amcor declining — direction right."),
    ("doc141_qa0__after147", 0.0, "ANS 'Decrease' wrong direction (gold increased)."),
    ("doc83_qa0__after147", 1.0, "ANS $3,115.4M vs $3,215 — within tolerance."),
    ("doc102_qa0__after147", 1.0, "ANS 0.4% — exact."),
    ("doc111_qa0__after147", 0.75, "ANS Yes decreased long-term debt — direction right."),
    # 1480-1489
    ("doc140_qa0__after148", 1.0, "ANS gold 36% Ulta stock repurchases Q4; predicted ~36.5% — within tolerance."),
    ("doc107_qa0__after148", 0.0, "ANS 1.61 vs 0 — wrong."),
    ("doc38_qa0__after148", 0.0, "ANS 'Common Shares' wrong."),
    ("doc59_qa0__after148", 1.0, "ANS $12,645 — exact."),
    ("doc120_qa0__after148", 0.25, "ANS PepsiCo wrong regional dimensions (US/Developed Europe/Developed RoW/Emerging Markets)."),
    ("doc127_qa0__after148", 1.0, "ANS $8.4B PepsiCo credit — exact."),
    ("doc77_qa0__after148", 0.75, "ANS partial — subset of CVS legal items."),
    ("doc118_qa0__after148", 0.5, "ANS PayPal WC different methodology."),
    ("doc85_qa0__after148", 1.0, "ANS No 1.3% — match."),
    ("doc137_qa0__after148", 1.0, "ANS gold no Ulta acquisitions FY23/22; predicted 'do not mention any major acquisitions' — match (functionally equivalent)."),
    # 1490-1499
    ("doc90_qa0__after149", 1.0, "ANS Consumer Health discontinued — exact."),
    ("doc82_qa0__after149", 1.0, "ANS 0.69 vs 0.68 — within tolerance."),
    ("doc63_qa0__after149", 0.5, "ANS partial — no US govt 40%."),
    ("doc109_qa0__after149", 1.0, "ANS corporate bonds $416M — match."),
    ("doc61_qa0__after149", 1.0, "ANS Lion Air + Ethiopian crashes — match."),
    ("doc55_qa0__after149", 1.0, "ANS Gaming 9% — match."),
    ("doc80_qa0__after149", 1.0, "ANS Richard A. Johnson — match."),
    ("doc105_qa0__after149", 1.0, "ANS MGM $0.01/share — exact."),
    ("doc108_qa0__after149", 0.75, "ANS MGM China worst $(203,136)M — right region, different metric."),
    ("doc128_qa0__after149", 1.0, "ANS PepsiCo raised guidance strong start FY2023 — match."),
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
        qid = QID_PREFIX + suffix
        if qid in existing:
            continue
        new_records.append(
            {
                "qid": qid,
                "judge_score": score,
                "rationale": rationale,
                "judge_model": "claude-opus-4.7-1m",
                "judge_protocol": "v1",
            }
        )

    with results_path.open("a", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_after = len(existing) + len(new_records)
    print(
        f"Appended {len(new_records)} (skipped {len(JUDGMENTS) - len(new_records)}, "
        f"total {total_after})"
    )
    if new_records:
        from collections import Counter
        dist = Counter(r["judge_score"] for r in new_records)
        print(f"Score distribution: {dict(sorted(dist.items()))}")
        mean = sum(r["judge_score"] for r in new_records) / len(new_records)
        print(f"Mean judge: {mean:.4f}")
    print(f"Cell progress: {total_after}/1500 (={100*total_after/1500:.1f}%)")


if __name__ == "__main__":
    main()
