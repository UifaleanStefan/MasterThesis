"""Phase 1.9 — dump-all calibration cell — Claude-1-by-1 hand-judging.

Part 10 (final): entries 1297-1499 (203 entries).

HARD RULE: each judge_score from Claude reading (question, gold, predicted)
triple manually. NO heuristic / auto-judging.

§6.5.1 collapse continues at scale — almost-universal refusal on ANS-mode
late-corpus questions when all 188 paragraphs dumped to gpt-4o-mini context.
Surviving wins: doc15 "0" exact, doc134 "Developed Rest of World" robust
across multiple after-N, doc126 $400M PepsiCo revolving credit match,
doc131 Y JV gain match, doc132 acquisitions list match, doc135 Y Upjohn
match, doc139 47 new stores match, doc138 lower marketing+leverage match,
doc140 36.5% within tolerance of 36%, doc122 $411M match, doc128 strong-start
match.
"""

from __future__ import annotations

import json
from pathlib import Path

QID_PREFIX = "financebench__dump-all__calibration__"
QID_SUFFIX = "__seed42"

RESULTS = Path(
    "results/stage3/judge_queue/financebench__dump-all__calibration__seed42/results.jsonl"
)

JUDGMENTS: list[tuple[str, float, str]] = [
    # ── 1297-1325 ──────────────────────────────────────────────────
    ("doc38_qa0__after129", 0.0, "ANS: GOLD 'There are none' (AMEX debt securities); PRED refuses. Refusal on definitive → 0.0."),
    ("doc100_qa0__after129", 0.0, "ANS: GOLD 1.33; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc146_qa0__after129", 1.0, "ACK: src=doc146 not yet ingested; PRED honestly refuses on Verizon debt. Honest refusal."),
    ("doc122_qa0__after130", 1.0, "ANS: GOLD '$411 million' PRED '411' — numeric match → 1.0."),
    ("doc17_qa0__after130", 0.0, "ANS: GOLD -0.02 (AES ROA FY2022); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc78_qa0__after130", 0.0, "ANS: GOLD 'Yes, $0.55 CVS dividend FY2022'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc38_qa0__after130", 0.0, "ANS: GOLD 'None'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc74_qa0__after130", 0.0, "ANS: GOLD $59268 (Costco assets FY2021); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc86_qa0__after130", 0.0, "ANS: GOLD list of JnJ GM drivers; PRED refuses. Refusal on definitive list → 0.0."),
    ("doc37_qa0__after130", 0.0, "ANS: GOLD 'Yes, 16% one customer'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc42_qa0__after130", 0.0, "ANS: GOLD 'AMEX tax 24.6→21.6%'; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc10_qa0__after130", 0.0, "ANS: GOLD 0.66 (Adobe OCF ratio FY2015); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc101_qa0__after130", 0.0, "ANS: GOLD $5818; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc26_qa0__after131", 0.0, "ANS: GOLD 'No 0.8% decline' (Amcor GM); PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc89_qa0__after131", 0.0, "ANS: GOLD 'US +3.0% vs intl -0.6%' (JnJ FY22); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc3_qa0__after131", 0.0, "ANS: GOLD '3M OI -1.7%'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc58_qa0__after131", 0.0, "ANS: GOLD $382; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc71_qa0__after131", 0.0, "ANS: GOLD 10.3%; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc94_qa0__after131", 0.0, "ANS: GOLD 'Corporate -$473M' (JPM Q1 2021); PRED refuses. Refusal on definitive → 0.0."),
    ("doc9_qa0__after131", 0.0, "ANS: GOLD 1.9% (Activision capex/rev FY17-19); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc18_qa0__after131", 0.0, "ANS: GOLD 93.86 (Amazon DPO FY2017); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc97_qa0__after131", 0.0, "ANS: GOLD 'Corporate & Investment Bank $3725M'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc61_qa0__after131", 0.0, "ANS: GOLD 'Yes, Lion Air & Ethiopian crashes' (Boeing legal); PRED refuses. Refusal on Y/N+narrative → 0.0."),
    ("doc0_qa0__after132", 0.0, "ANS: GOLD $1577; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc120_qa0__after132", 0.0, "ANS: GOLD PepsiCo geographies list; PRED refuses. Refusal on definitive list → 0.0."),
    ("doc32_qa0__after132", 0.0, "ANS: GOLD AMD products list; PRED refuses. Refusal on definitive list → 0.0."),
    ("doc141_qa0__after132", 1.0, "ACK: src=doc141 not yet ingested; PRED honestly refuses on Ulta wages FY23. Honest refusal."),
    ("doc112_qa0__after132", 0.0, "ANS: GOLD 5.4%; PRED refuses citing Netflix. Refusal on definitive numeric → 0.0."),
    ("doc43_qa0__after132", 0.0, "ANS: GOLD 'Customer deposits' (AMEX 2022 liabilities); PRED refuses. Refusal on definitive → 0.0."),
    # ── 1326-1355 ──────────────────────────────────────────────────
    ("doc4_qa0__after132", 0.0, "ANS: GOLD '0.9% consumer segment shrunk'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc126_qa0__after132", 1.0, "ANS: GOLD '$400,000,000 increase'; PRED '$400,000,000 increase, from $3,800,000,000 to $4,200,000,000' — exact + bonus specifics → 1.0."),
    ("doc93_qa0__after132", 0.0, "ANS: GOLD 'Yes 20→20.1%'; PRED refuses. Refusal on Y/N → 0.0."),
    ("doc9_qa0__after132", 0.0, "ANS: GOLD 1.9%; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc75_qa0__after133", 0.0, "ANS: GOLD 17.98 (CVS fixed asset turnover FY2018); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc84_qa0__after133", 0.0, "ANS: GOLD 0.54 (General Mills retention FY22); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc19_qa0__after133", 0.0, "ANS: GOLD 30.8% (Amazon revenue YoY); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc120_qa0__after133", 0.0, "ANS: GOLD geographies; PRED refuses. Refusal on definitive list → 0.0."),
    ("doc76_qa0__after133", 0.0, "ANS: GOLD 'Yes CVS capital intensive ROA 1.82/3.39%'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc11_qa0__after133", 0.0, "ANS: GOLD 65.4%; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc86_qa0__after133", 0.0, "ANS: GOLD JnJ GM drivers; PRED refuses. Refusal on definitive → 0.0."),
    ("doc131_qa0__after133", 1.0, "ANS: GOLD 'Yes, gain on Consumer Healthcare JV Transaction'; PRED 'Yes, gain on completion of Consumer Healthcare JV transaction, $8,107M' — Y match + specific amount → 1.0."),
    ("doc148_qa0__after133", 1.0, "ACK: src=doc148 not yet ingested; PRED honestly refuses on Walmart op income margin FY18-19. Honest refusal."),
    ("doc117_qa0__after133", 0.0, "ANS: GOLD 'Nike cash flow ops highest FY23'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc80_qa0__after134", 0.0, "ANS: GOLD 'Yes Richard A Johnson'; PRED refuses. Refusal on Y/N+name → 0.0."),
    ("doc143_qa0__after134", 1.0, "ACK: src=doc143 not yet ingested; PRED honestly refuses on Verizon retiree pension 2024. Honest refusal."),
    ("doc20_qa0__after134", 0.0, "ANS: GOLD $11588; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc107_qa0__after134", 0.0, "ANS: GOLD 'coverage ratio is zero (EBIT negative)'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc15_qa0__after134", 1.0, "ANS: GOLD '0' PRED '0' → exact 1.0."),
    ("doc134_qa0__after134", 1.0, "ANS: GOLD 'Developed Rest of the World' PRED 'Developed Rest of World' — semantically identical → 1.0."),
    ("doc108_qa0__after134", 0.0, "ANS: GOLD 'MGM China revenue -44%'; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc114_qa0__after134", 0.0, "ANS: GOLD 55.1%; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc109_qa0__after134", 0.0, "ANS: GOLD 'corporate bonds 82% MGM short-term investments H1 FY23'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc25_qa0__after134", 0.0, "ANS: GOLD 'Amcor global packaging leader'; PRED refuses. Refusal on definitive qualitative → 0.0."),
    ("doc55_qa0__after135", 0.0, "ANS: GOLD 'entertainment 9% growth Q2 FY24'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc60_qa0__after135", 0.0, "ANS: GOLD 'Yes, Boeing 39% Commercial Airplanes etc.'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc102_qa0__after135", 0.0, "ANS: GOLD 0.4% (Lockheed 2-yr CAGR FY20-22); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc88_qa0__after135", 0.0, "ANS: GOLD 'No 3.6→3.5%'; PRED refuses. Refusal on Y/N → 0.0."),
    ("doc86_qa0__after135", 0.0, "ANS: GOLD JnJ GM drivers; PRED refuses. Refusal on definitive → 0.0."),
    ("doc81_qa0__after135", 0.0, "ANS: GOLD -3.7; PRED refuses. Refusal on definitive numeric → 0.0."),
    # ── 1356-1385 ──────────────────────────────────────────────────
    ("doc118_qa0__after135", 0.0, "ANS: GOLD 'Yes PayPal $1.6Bn FY2022 WC'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc139_qa0__after135", 1.0, "ACK: src=doc139 not yet ingested; PRED honestly refuses on Ulta merchandise inventories. Honest refusal."),
    ("doc127_qa0__after135", 0.25, "ANS: GOLD $8,400,000,000 PRED '$4,950,000,000' — confident wrong specific → 0.25."),
    ("doc10_qa0__after135", 0.0, "ANS: GOLD 0.66; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc115_qa0__after136", 0.0, "ANS: GOLD $16525; PRED refuses (different from earlier match). Refusal on definitive numeric → 0.0."),
    ("doc120_qa0__after136", 0.0, "ANS: GOLD PepsiCo geographies; PRED refuses. Refusal on definitive list → 0.0."),
    ("doc27_qa0__after136", 0.0, "ANS: GOLD '87% employee restructuring' (Amcor); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc148_qa0__after136", 1.0, "ACK: src=doc148 not yet ingested; PRED honestly refuses on Walmart 10-K FY18-19. Honest refusal."),
    ("doc108_qa0__after136", 0.0, "ANS: GOLD 'MGM China -44%'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc2_qa0__after136", 0.0, "ANS: GOLD 'No 3M efficiently managing capex'; PRED refuses. Refusal on definitive qualitative+numeric → 0.0."),
    ("doc58_qa0__after136", 0.0, "ANS: GOLD $382; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc80_qa0__after136", 0.0, "ANS: GOLD 'Yes Richard A Johnson'; PRED refuses. Refusal on Y/N+name → 0.0."),
    ("doc63_qa0__after136", 0.0, "ANS: GOLD 'Boeing customers airlines + US govt 40%'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc103_qa0__after136", 0.0, "ANS: GOLD $303 (MGM AP FY2018); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc128_qa0__after137", 1.0, "ANS: GOLD 'Pepsico strong start FY2023'; PRED 'PepsiCo raised full-year guidance for FY2023 due to strong start to the year, resilient performance and business momentum' — Y match → 1.0."),
    ("doc39_qa0__after137", 0.0, "ANS: GOLD 'US EMEA APAC LACC' (AMEX geos); PRED refuses. Refusal on definitive list → 0.0."),
    ("doc60_qa0__after137", 0.0, "ANS: GOLD 'Y Boeing 39% CA etc.'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc88_qa0__after137", 0.0, "ANS: GOLD 'No 3.6→3.5%'; PRED refuses. Refusal on Y/N → 0.0."),
    ("doc134_qa0__after137", 1.0, "ANS: GOLD 'Developed Rest of the World' PRED 'Developed Rest of World' → 1.0."),
    ("doc135_qa0__after137", 1.0, "ANS: GOLD 'Yes, spinning off Upjohn'; PRED 'Yes, $700M Upjohn separation costs' — Y match + bonus specifics → 1.0."),
    ("doc113_qa0__after137", 0.0, "ANS: GOLD $5466 (Netflix FY2017 TCL); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc126_qa0__after137", 1.0, "ANS: GOLD '$400M increase'; PRED '$400,000,000 increase from $3,800,000,000 to $4,200,000,000' — exact → 1.0."),
    ("doc18_qa0__after137", 0.0, "ANS: GOLD 93.86; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc13_qa0__after137", 0.0, "ANS: GOLD 'No Adobe OM 36.8→34.6%'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc60_qa0__after138", 0.0, "ANS: GOLD 'Y Boeing categories'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc39_qa0__after138", 0.0, "ANS: GOLD 'US EMEA APAC LACC'; PRED refuses. Refusal on definitive list → 0.0."),
    ("doc119_qa0__after138", 0.0, "ANS: GOLD $4.60 (PepsiCo FY2021 capex); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc142_qa0__after138", 1.0, "ACK: src=doc142 not yet ingested; PRED honestly refuses on Verizon derivatives FY 2021. Honest refusal."),
    ("doc35_qa0__after138", 0.0, "ANS: GOLD 'AMD cashflow ops highest 2022'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc8_qa0__after138", 0.0, "ANS: GOLD 24.26 (Activision FixAssetTurnover FY2019); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc131_qa0__after138", 1.0, "ANS: GOLD 'Yes, JV gain'; PRED 'Yes, gain on Consumer Healthcare JV transaction contributed to net income 2019' — Y match + bonus specifics → 1.0."),
    # ── 1387-1415 ──────────────────────────────────────────────────
    ("doc67_qa0__after138", 0.0, "ANS: GOLD 0.01 (Coca-Cola ROA FY2017); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc47_qa0__after138", 0.0, "ANS: GOLD 'No -$1561M'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc3_qa0__after138", 0.0, "ANS: GOLD '3M OI -1.7%'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc148_qa0__after139", 1.0, "ACK: src=doc148 not yet ingested; PRED honestly refuses on Walmart op income FY18-19. Honest refusal."),
    ("doc70_qa0__after139", 0.0, "ANS: GOLD 63.86 (Corning DPO FY2020); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc118_qa0__after139", 0.0, "ANS: GOLD 'Y PayPal $1.6Bn'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc39_qa0__after139", 0.0, "ANS: GOLD 'US EMEA APAC LACC'; PRED refuses. Refusal on definitive list → 0.0."),
    ("doc74_qa0__after139", 0.0, "ANS: GOLD $59268; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc12_qa0__after139", 0.0, "ANS: GOLD 0.83 (Adobe OCF ratio FY2017); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc24_qa0__after139", 0.0, "ANS: GOLD Amcor FY2023 acquisitions; PRED refuses. Refusal on definitive list → 0.0."),
    ("doc25_qa0__after139", 0.0, "ANS: GOLD 'Amcor global packaging'; PRED refuses. Refusal on definitive qualitative → 0.0."),
    ("doc0_qa0__after139", 0.0, "ANS: GOLD $1577; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc92_qa0__after139", 0.0, "ANS: GOLD '$13.2B Kenvue separation'; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc5_qa0__after140", 0.0, "ANS: GOLD 'No 3M quick ratio 0.96 Jun23'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc135_qa0__after140", 1.0, "ANS: GOLD 'Yes spinning off Upjohn'; PRED 'Yes, $700M Upjohn separation costs' — Y match → 1.0."),
    ("doc76_qa0__after140", 0.0, "ANS: GOLD 'Yes CVS capital intensive ROA 1.82/3.39%'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc26_qa0__after140", 0.0, "ANS: GOLD 'No Amcor GM -0.8%'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc55_qa0__after140", 0.0, "ANS: GOLD 'entertainment 9% Q2 FY24'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc58_qa0__after140", 0.0, "ANS: GOLD $382; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc105_qa0__after140", 0.0, "ANS: GOLD 'Y MGM $0.01 dividend FY22'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc31_qa0__after140", 0.0, "ANS: GOLD 'Y AMD quick ratio 1.57'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc123_qa0__after140", 0.0, "ANS: GOLD $9068 (PepsiCo EBITDA FY22); PRED refuses (different from earlier confident-wrong). Refusal on definitive → 0.0."),
    ("doc3_qa0__after140", 0.0, "ANS: GOLD '3M OI -1.7%'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc62_qa0__after141", 0.0, "ANS: GOLD 'Y Boeing GM 4.8→5.3%'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc3_qa0__after141", 0.0, "ANS: GOLD '3M OI -1.7%'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc38_qa0__after141", 0.0, "ANS: GOLD 'None'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc143_qa0__after141", 1.0, "ACK: src=doc143 not yet ingested; PRED honestly refuses on Verizon retiree 2024. Honest refusal."),
    ("doc125_qa0__after141", 0.0, "ANS: GOLD 'shareholder proposal defeated'; PRED refuses on outcome. Refusal on definitive → 0.0."),
    ("doc87_qa0__after141", 0.0, "ANS: GOLD '2.7 inventory turnover'; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc63_qa0__after141", 0.0, "ANS: GOLD 'Boeing customers airlines + US govt 40%'; PRED refuses. Refusal on definitive → 0.0."),
    # ── 1417-1445 ──────────────────────────────────────────────────
    ("doc69_qa0__after141", 0.0, "ANS: GOLD 0.8 (Coca-Cola payout FY2022); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc124_qa0__after141", 0.0, "ANS: GOLD 16.5%; PRED refuses (different from earlier confident-wrong). Refusal on definitive numeric → 0.0."),
    ("doc17_qa0__after141", 0.0, "ANS: GOLD -0.02; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc34_qa0__after142", 0.0, "ANS: GOLD 'AMD OI Xilinx'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc102_qa0__after142", 0.0, "ANS: GOLD 0.4%; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc127_qa0__after142", 0.0, "ANS: GOLD $8.4B; PRED refuses (different from earlier confident-wrong). Refusal on definitive → 0.0."),
    ("doc146_qa0__after142", 1.0, "ACK: src=doc146 not yet ingested; PRED honestly refuses on Verizon debt. Honest refusal."),
    ("doc2_qa0__after142", 0.0, "ANS: GOLD 'No 3M efficient capex'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc113_qa0__after142", 0.0, "ANS: GOLD $5466; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc139_qa0__after142", 1.0, "ANS: GOLD '47 new stores driver'; PRED 'opening of 47 new stores, brand launches, cost increases' — exact match + extras → 1.0."),
    ("doc74_qa0__after142", 0.0, "ANS: GOLD $59268; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc132_qa0__after142", 1.0, "ANS: GOLD 'Trillium, Array, and Therachon' PRED 'Trillium, Array, and Therachon.' — exact → 1.0."),
    ("doc107_qa0__after142", 0.0, "ANS: GOLD 'coverage zero'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc63_qa0__after143", 0.0, "ANS: GOLD 'Boeing customers airlines + US govt 40%'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc45_qa0__after143", 0.0, "ANS: GOLD $0.40; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc4_qa0__after143", 0.0, "ANS: GOLD '0.9% consumer segment'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc141_qa0__after143", 0.0, "ANS: GOLD 'Wages% increased FY2023' PRED 'Decrease' — Y/N FLIP wrong direction → 0.0."),
    ("doc93_qa0__after143", 0.0, "ANS: GOLD 'Y 20→20.1%'; PRED refuses. Refusal on Y/N → 0.0."),
    ("doc134_qa0__after143", 1.0, "ANS: GOLD 'Developed Rest of the World' PRED 'Developed Rest of World' → 1.0."),
    ("doc79_qa0__after143", 0.0, "ANS: GOLD 'Yes Ulta CEO experience'; PRED refuses. Refusal on Y/N+narrative → 0.0."),
    ("doc138_qa0__after143", 1.0, "ANS: GOLD 'Lower marketing + leverage incentive comp due to higher sales'; PRED 'lower marketing expenses and leverage of incentive compensation due to higher sales' — exact match → 1.0."),
    ("doc11_qa0__after143", 0.0, "ANS: GOLD 65.4%; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc7_qa0__after143", 0.0, "ANS: GOLD 'Y 3M 65 years dividend'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc86_qa0__after144", 0.0, "ANS: GOLD JnJ GM drivers; PRED refuses. Refusal on definitive → 0.0."),
    ("doc31_qa0__after144", 0.0, "ANS: GOLD 'Y AMD 1.57'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc139_qa0__after144", 1.0, "ANS: GOLD '47 new stores driver'; PRED matches with bonus specifics → 1.0."),
    ("doc44_qa0__after144", 0.0, "ANS: GOLD 'Yes' (AMEX); PRED refuses. Refusal on Y/N → 0.0."),
    ("doc24_qa0__after144", 0.0, "ANS: GOLD Amcor acquisitions; PRED refuses. Refusal on definitive list → 0.0."),
    ("doc97_qa0__after144", 0.0, "ANS: GOLD 'CIB $3725M'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc63_qa0__after144", 0.0, "ANS: GOLD 'Boeing customers'; PRED refuses. Refusal on definitive → 0.0."),
    # ── 1447-1475 ──────────────────────────────────────────────────
    ("doc110_qa0__after144", 0.0, "ANS: GOLD $32780 (MSFT COGS FY2016); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc23_qa0__after144", 0.0, "ANS: GOLD 'Amcor quick ratio 0.67→0.69'; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc78_qa0__after144", 0.0, "ANS: GOLD 'Y CVS $0.55'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc23_qa0__after145", 0.0, "ANS: GOLD 'Amcor 0.67→0.69'; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc110_qa0__after145", 0.0, "ANS: GOLD $32780; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc19_qa0__after145", 0.0, "ANS: GOLD 30.8%; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc20_qa0__after145", 0.0, "ANS: GOLD $11588; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc136_qa0__after145", 0.25, "ANS: GOLD 'There are none' (debt securities); PRED 'common stock $0.01 par ULTA' — confident wrong specific (talks about common stock, not debt securities) → 0.25."),
    ("doc95_qa0__after145", 0.0, "ANS: GOLD '$66.56/share' (JPM liquidation); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc119_qa0__after145", 0.0, "ANS: GOLD $4.60; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc109_qa0__after145", 0.0, "ANS: GOLD 'corporate bonds 82%'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc62_qa0__after145", 0.0, "ANS: GOLD 'Y Boeing GM 4.8→5.3%'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc12_qa0__after145", 0.0, "ANS: GOLD 0.83; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc111_qa0__after146", 0.0, "ANS: GOLD 'No MSFT debt -$2.5bn FY23 vs FY22'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc51_qa0__after146", 0.0, "ANS: GOLD 'Best Buy Current Health + others'; PRED refuses. Refusal on definitive list → 0.0."),
    ("doc10_qa0__after146", 0.0, "ANS: GOLD 0.66; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc64_qa0__after146", 0.0, "ANS: GOLD 'Y Boeing cyclical'; PRED refuses. Refusal on Y/N → 0.0."),
    ("doc139_qa0__after146", 1.0, "ANS: GOLD '47 new stores driver'; PRED matches with bonus specifics → 1.0."),
    ("doc24_qa0__after146", 0.0, "ANS: GOLD Amcor acquisitions; PRED refuses. Refusal on definitive → 0.0."),
    ("doc98_qa0__after146", 0.0, "ANS: GOLD 'Y JPM VaR decreased'; PRED refuses. Refusal on Y/N → 0.0."),
    ("doc5_qa0__after146", 0.0, "ANS: GOLD 'No 3M 0.96'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc13_qa0__after146", 0.0, "ANS: GOLD 'No Adobe OM 36.8→34.6%'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc53_qa0__after146", 0.0, "ANS: GOLD 'Y -42% Best Buy cash decline'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc25_qa0__after147", 0.0, "ANS: GOLD 'Amcor global packaging'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc24_qa0__after147", 0.0, "ANS: GOLD Amcor acquisitions; PRED refuses. Refusal on definitive → 0.0."),
    ("doc35_qa0__after147", 0.0, "ANS: GOLD 'AMD cashflow ops 2022'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc22_qa0__after147", 0.0, "ANS: GOLD 'Amcor 8k 1-Jul-2022 indentures'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc117_qa0__after147", 0.0, "ANS: GOLD 'Nike cash flow ops highest'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc26_qa0__after147", 0.0, "ANS: GOLD 'No Amcor GM -0.8%'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc141_qa0__after147", 0.0, "ANS: GOLD 'Wages% increased' PRED 'Decrease' → Y/N FLIP → 0.0."),
    # ── 1477-1499 ──────────────────────────────────────────────────
    ("doc83_qa0__after147", 0.0, "ANS: GOLD $3215; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc102_qa0__after147", 0.0, "ANS: GOLD 0.4%; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc111_qa0__after147", 0.0, "ANS: GOLD 'No MSFT -$2.5bn'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc140_qa0__after148", 1.0, "ANS: GOLD '36%' PRED '36.5%' — diff 0.5pp, ~1.4% relative, within 5% tolerance → 1.0."),
    ("doc107_qa0__after148", 0.0, "ANS: GOLD 'coverage zero'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc38_qa0__after148", 0.0, "ANS: GOLD 'None'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc59_qa0__after148", 0.0, "ANS: GOLD $12645; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc120_qa0__after148", 0.0, "ANS: GOLD PepsiCo geographies; PRED refuses. Refusal on definitive list → 0.0."),
    ("doc127_qa0__after148", 0.0, "ANS: GOLD $8.4B; PRED refuses. Refusal on definitive → 0.0."),
    ("doc77_qa0__after148", 0.0, "ANS: GOLD 'Y CVS legal'; PRED refuses. Refusal on Y/N → 0.0."),
    ("doc118_qa0__after148", 0.0, "ANS: GOLD 'Y PayPal $1.6Bn'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc85_qa0__after148", 0.0, "ANS: GOLD 'No 1.3% sales'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc137_qa0__after148", 0.0, "ANS: GOLD 'Ulta no acquisitions FY23 FY22'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc90_qa0__after149", 0.0, "ANS: GOLD 'Consumer Health discontinued Aug 30 2023'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc82_qa0__after149", 0.0, "ANS: GOLD 0.68 (General Mills WC FY2020); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc63_qa0__after149", 0.0, "ANS: GOLD 'Boeing customers'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc109_qa0__after149", 0.0, "ANS: GOLD 'corporate bonds 82%'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc61_qa0__after149", 0.0, "ANS: GOLD 'Y Lion Air'; PRED refuses. Refusal on Y/N → 0.0."),
    ("doc55_qa0__after149", 0.0, "ANS: GOLD 'entertainment 9%'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc80_qa0__after149", 0.0, "ANS: GOLD 'Y Richard A Johnson'; PRED refuses. Refusal on Y/N+name → 0.0."),
    ("doc105_qa0__after149", 0.0, "ANS: GOLD 'Y MGM $0.01 FY22 dividend'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc108_qa0__after149", 0.0, "ANS: GOLD 'MGM China -44%'; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc128_qa0__after149", 0.0, "ANS: GOLD 'Pepsico strong start FY2023'; PRED refuses (different from 1370). Refusal on definitive → 0.0."),
]


def main() -> None:
    existing = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
                existing.add(obj["qid"])
            except Exception:
                continue

    added = 0
    scores: list[float] = []
    with RESULTS.open("a", encoding="utf-8") as fh:
        for suffix, score, rationale in JUDGMENTS:
            qid = f"{QID_PREFIX}{suffix}{QID_SUFFIX}"
            if qid in existing:
                continue
            fh.write(
                json.dumps(
                    {
                        "qid": qid,
                        "judge_score": float(score),
                        "rationale": rationale,
                        "judge_model": "claude-opus-4.7-1m",
                        "judge_protocol": "v1",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            added += 1
            scores.append(score)

    print(f"Added {added} new judgments to {RESULTS.name}.")
    if scores:
        dist = {f"{s:.2f}": scores.count(s) for s in sorted(set(scores), reverse=True)}
        print(f"Score distribution: {dist}")
        print(f"Mean of this batch: {sum(scores)/len(scores):.4f}")

    total_lines = [_ for _ in RESULTS.read_text(encoding="utf-8").splitlines() if _.strip()]
    total = len(total_lines)
    print(f"Total entries in results.jsonl: {total} / 1500 ({100*total/1500:.1f}%)")
    if total >= 1500:
        all_scores = []
        for line in total_lines:
            try:
                all_scores.append(json.loads(line)["judge_score"])
            except Exception:
                continue
        if all_scores:
            print(f"CELL FINAL MEAN: {sum(all_scores)/len(all_scores):.4f}")


if __name__ == "__main__":
    main()
