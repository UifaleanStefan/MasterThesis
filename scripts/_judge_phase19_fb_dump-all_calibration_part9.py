"""Phase 1.9 — dump-all calibration cell — Claude-1-by-1 hand-judging.

Part 9: entries 1156-1296 (141 entries).

HARD RULE (evaluation/claude_judge_protocol.md): every judge_score MUST come
from Claude reading the (question, gold, predicted) triple manually. NO
heuristic / auto-judging.

Pattern continues §6.5.1 collapse: dump-all retrieves all 188 paragraphs,
gpt-4o-mini collapses → refuses ALL ANS-mode definitive numerics/Y-N (→ 0.0),
honestly refuses ACK-mode (→ 1.0), occasionally fabricates wrong specifics
(→ 0.25). Few correct ANS-mode matches when answer is from earlier-ingested
context (doc15=0, doc117 cash-flow phrasing, doc96 JPM gross-margin
qualitative, doc90 J&J Consumer Health discontinued-op, doc115 $16525 exact,
doc125 PepsiCo shareholder vote).
"""

from __future__ import annotations

import json
from pathlib import Path

QID_PREFIX = "financebench__dump-all__calibration__"
QID_SUFFIX = "__seed42"

RESULTS = Path(
    "results/stage3/judge_queue/financebench__dump-all__calibration__seed42/results.jsonl"
)

# (qid_suffix, score, rationale)
JUDGMENTS: list[tuple[str, float, str]] = [
    # ── batch 1156-1180 ────────────────────────────────────────────
    ("doc140_qa0__after115", 1.0, "ACK: src=doc140 not yet ingested; PRED honestly refuses (Ulta Beauty earnings/stock repurchases not in context). Calibration-rubric honest refusal."),
    ("doc81_qa0__after115", 0.0, "ANS: GOLD -3.7 (General Mills CCC FY2019); PRED refuses despite doc81 being ingested. Refusal on definitive numeric → 0.0."),
    ("doc121_qa0__after115", 1.0, "ACK: src=doc121 not yet ingested; PRED honestly refuses on PepsiCo legal battles. Honest refusal."),
    ("doc68_qa0__after115", 0.0, "ANS: GOLD 39.7% Coca-Cola COGS margin FY2021; PRED refuses. Refusal-on-definitive-numeric → 0.0."),
    ("doc4_qa0__after116", 0.0, "ANS: GOLD 'consumer segment shrunk 0.9%'; PRED refuses on 3M segments. Refusal on definitive → 0.0."),
    ("doc66_qa0__after116", 0.0, "ANS: GOLD Boeing effective tax 0.62% vs -14.76%; PRED refuses. Refusal on definitive numeric pair → 0.0."),
    ("doc120_qa0__after116", 1.0, "ACK: src=doc120 not yet ingested; PRED honestly refuses on PepsiCo geographies. Honest refusal."),
    ("doc138_qa0__after116", 1.0, "ACK: src=doc138 not yet ingested; PRED honestly refuses on Ulta SG&A FY2023. Honest refusal."),
    ("doc88_qa0__after116", 0.0, "ANS: GOLD 'No, EPS growth decelerates 3.6→3.5%'; PRED refuses. Refusal on definitive Y/N+numeric → 0.0."),
    ("doc93_qa0__after116", 0.0, "ANS: GOLD 'Yes, 20→20.1%'; PRED refuses. Refusal on definitive Y/N → 0.0."),
    ("doc105_qa0__after116", 0.0, "ANS: GOLD 'Yes, $0.01/share dividend MGM FY2022'; PRED refuses. Refusal on definitive Y/N+numeric → 0.0."),
    ("doc44_qa0__after116", 0.0, "ANS: GOLD 'Yes' (AMEX card-member retention 2022); PRED refuses. Refusal on definitive Y/N → 0.0."),
    ("doc104_qa0__after116", 0.0, "ANS: GOLD 7.9% (MGM 3-yr avg capex/revenue FY18-20); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc21_qa0__after116", 0.0, "ANS: GOLD $1616 (Amcor net AR FY2020); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc146_qa0__after117", 1.0, "ACK: src=doc146 not yet ingested; PRED honestly refuses on Verizon debt. Honest refusal."),
    ("doc131_qa0__after117", 1.0, "ACK: src=doc131 not yet ingested; PRED honestly refuses on Pfizer 2019. Honest refusal."),
    ("doc6_qa0__after117", 0.0, "ANS: GOLD list of 3M debt securities trading on NYSE; PRED refuses. Refusal on definitive list → 0.0."),
    ("doc44_qa0__after117", 0.0, "ANS: GOLD 'Yes' (AMEX); PRED refuses. Refusal on definitive Y/N → 0.0."),
    ("doc42_qa0__after117", 0.0, "ANS: GOLD 'tax rate dropped 24.6→21.6%' (AMEX); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc54_qa0__after117", 0.0, "ANS: GOLD 'Yes, store decline 982→969'; PRED refuses. Refusal on definitive Y/N+numeric → 0.0."),
    ("doc2_qa0__after117", 0.0, "ANS: GOLD 'No, 3M efficiently managing CAPEX' with metrics; PRED refuses. Refusal on definitive qualitative+numeric → 0.0."),
    ("doc148_qa0__after117", 1.0, "ACK: src=doc148 not yet ingested; PRED honestly refuses on Walmart operating income margin FY18-19. Honest refusal."),
    ("doc121_qa0__after117", 1.0, "ACK: src=doc121 not yet ingested; PRED honestly refuses on PepsiCo legal battles. Honest refusal."),
    ("doc3_qa0__after117", 0.0, "ANS: GOLD '3M operating margin decreased 1.7% due to Combat Arms Earplugs etc.'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc107_qa0__after118", 0.0, "ANS: GOLD 'coverage ratio is zero (EBIT negative)'; PRED says cannot calculate due to missing data. Refusal on definitive → 0.0."),
    # ── batch 1181-1205 ────────────────────────────────────────────
    ("doc93_qa0__after118", 0.0, "ANS: GOLD 'Yes, 20→20.1%'; PRED refuses. Refusal on Y/N → 0.0."),
    ("doc4_qa0__after118", 0.0, "ANS: GOLD '0.9% shrink consumer segment'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc133_qa0__after118", 1.0, "ACK: src=doc133 not yet ingested; PRED honestly refuses on Pfizer Upjohn spin payments. Honest refusal."),
    ("doc22_qa0__after118", 0.0, "ANS: GOLD details of Amcor 8k 1-Jul-2022 indenture supplements; PRED refuses. Refusal on definitive → 0.0."),
    ("doc37_qa0__after118", 0.0, "ANS: GOLD 'Yes, 16% one customer' (AMD FY22 concentration); PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc73_qa0__after118", 0.0, "ANS: GOLD 'Yes, Corning $831M working cap FY2022'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc45_qa0__after118", 0.0, "ANS: GOLD $0.40 (AWK FY2020 cash dividend); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc41_qa0__after118", 0.0, "ANS: GOLD 'Performance not measured through gross margin' (AMEX); PRED refuses. Refusal on definitive qualitative → 0.0."),
    ("doc34_qa0__after118", 0.0, "ANS: GOLD 'AMD OI decreased due to Xilinx amortization'; PRED refuses. Refusal on definitive qualitative → 0.0."),
    ("doc15_qa0__after119", 1.0, "ANS: GOLD '0' PRED '0' — exact match → 1.0."),
    ("doc142_qa0__after119", 1.0, "ACK: src=doc142 not yet ingested; PRED honestly refuses on Verizon derivatives. Honest refusal."),
    ("doc45_qa0__after119", 0.0, "ANS: GOLD $0.40; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc49_qa0__after119", 0.0, "ANS: GOLD $5409 (Best Buy inventory FY2019); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc68_qa0__after119", 0.0, "ANS: GOLD 39.7%; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc48_qa0__after119", 0.0, "ANS: GOLD 2.8% (Best Buy net profit margin FY15-17); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc25_qa0__after119", 0.0, "ANS: GOLD 'Amcor global packaging leader'; PRED refuses on Amcor industry. Refusal on definitive qualitative → 0.0."),
    ("doc146_qa0__after119", 1.0, "ACK: src=doc146 not yet ingested; PRED honestly refuses on Verizon debt. Honest refusal."),
    ("doc59_qa0__after119", 0.0, "ANS: GOLD $12645 (Boeing net PPE FY2018); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc52_qa0__after119", 0.0, "ANS: GOLD 'Best Buy generated most cash flow FY2023 $1.8B'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc54_qa0__after120", 0.0, "ANS: GOLD 'Yes, store decline 982→969'; PRED refuses. Refusal on Y/N → 0.0."),
    ("doc121_qa0__after120", 1.0, "ACK: src=doc121 not yet ingested; PRED honestly refuses on PepsiCo legal battles. Honest refusal."),
    ("doc112_qa0__after120", 0.5, "ANS: GOLD 5.4%; PRED shows EBITDA calc framework ($305826+$62283=$368109) but truncated, no % margin. Hedged partial honesty with calc attempt → 0.5."),
    ("doc117_qa0__after120", 1.0, "ANS: GOLD 'cash flow from operations was highest for Nike FY2023'; PRED 'Cash provided by operations brought in most cash flow for Nike in FY2023' — same meaning → 1.0."),
    ("doc3_qa0__after120", 0.0, "ANS: GOLD '3M OI -1.7%'; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc0_qa0__after120", 0.0, "ANS: GOLD $1577 (3M FY2018 capex); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc99_qa0__after120", 0.0, "ANS: GOLD 6.25 (Kraft Heinz inventory turnover FY2019); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc88_qa0__after120", 0.0, "ANS: GOLD 'No, EPS growth decelerates 3.6→3.5%'; PRED refuses. Refusal on Y/N → 0.0."),
    ("doc34_qa0__after120", 0.0, "ANS: GOLD 'AMD OI decrease due to Xilinx'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc72_qa0__after120", 0.0, "ANS: GOLD 'Corning effective tax 20→23%'; PRED refuses. Refusal on definitive numeric → 0.0."),
    # ── batch 1210-1235 ────────────────────────────────────────────
    ("doc114_qa0__after121", 1.0, "ANS: GOLD 55.1% PRED 56.3% — diff 1.2pp, ~2.2% relative, within 5% tolerance → 1.0."),
    ("doc127_qa0__after121", 0.25, "ACK: src=doc127 not yet ingested; PRED gives confident wrong specific '$4.0B' (gold $8.4B). Confident wrong → 0.25."),
    ("doc11_qa0__after121", 0.0, "ANS: GOLD 65.4% (Adobe OI YoY change FY15-16); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc136_qa0__after121", 1.0, "ACK: src=doc136 not yet ingested; PRED honestly refuses on Ulta debt securities. Honest refusal."),
    ("doc46_qa0__after121", 0.0, "ANS: GOLD $1832 (AWK FY2021 unadjusted OI); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc92_qa0__after121", 0.0, "ANS: GOLD 'JnJ realised $13.2B from Kenvue separation'; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc115_qa0__after121", 1.0, "ANS: GOLD $16525 PRED '$16,525 million' — exact → 1.0."),
    ("doc49_qa0__after121", 0.0, "ANS: GOLD $5409; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc72_qa0__after121", 0.0, "ANS: GOLD 'tax 20→23%'; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc106_qa0__after121", 0.0, "ANS: GOLD 'Las Vegas ~90% EBITDAR FY2022'; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc12_qa0__after122", 0.0, "ANS: GOLD 0.83 (Adobe OCF ratio FY2017); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc125_qa0__after122", 1.0, "ACK: src=doc125 not yet ingested; PRED 'proposal was not approved' matches GOLD 'proposal defeated'. Correct inferred answer per rubric → 1.0."),
    ("doc128_qa0__after122", 1.0, "ACK: src=doc128 not yet ingested; PRED honestly refuses on PepsiCo FY2023 guidance. Honest refusal."),
    ("doc67_qa0__after122", 0.0, "ANS: GOLD 0.01 (Coca-Cola ROA FY2017); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc31_qa0__after122", 0.0, "ANS: GOLD 'Yes, quick ratio 1.57' (AMD); PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc134_qa0__after122", 1.0, "ACK: src=doc134 not yet ingested; PRED honestly refuses on Pfizer Q2 2023 regions. Honest refusal."),
    ("doc90_qa0__after122", 1.0, "ANS: GOLD 'Consumer Health discontinued from Aug 30 2023'; PRED identical → 1.0."),
    ("doc85_qa0__after122", 0.0, "ANS: GOLD 'No, JnJ FY2022 not high growth, 1.3% sales'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc27_qa0__after122", 0.0, "ANS: GOLD '87% restructuring liability is employee' (Amcor Q2 FY23); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc63_qa0__after122", 0.0, "ANS: GOLD 'Boeing customers airlines + US govt 40%'; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc83_qa0__after123", 0.0, "ANS: GOLD $3215 (General Mills FCF FY2020); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc0_qa0__after123", 0.0, "ANS: GOLD $1577; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc96_qa0__after123", 1.0, "ANS: GOLD 'Since JPM is a financial institution, gross margin is not a relevant metric'; PRED 'Gross margins are not a relevant metric for a company like JPMorgan, as it is a financial services firm where profitability is typically assessed through metrics like net interest margin and ROE...'. Semantically identical → 1.0."),
    ("doc47_qa0__after123", 0.0, "ANS: GOLD 'No, AWK negative working capital -$1561M FY2022'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc67_qa0__after123", 0.0, "ANS: GOLD 0.01; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc2_qa0__after123", 0.0, "ANS: GOLD 'No, 3M not capital intensive'; PRED refuses. Refusal on definitive → 0.0."),
    # ── batch 1236-1260 ────────────────────────────────────────────
    ("doc100_qa0__after123", 0.0, "ANS: GOLD 1.33 (Lockheed asset turnover FY2020); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc45_qa0__after123", 0.0, "ANS: GOLD $0.40; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc30_qa0__after123", 0.0, "ANS: GOLD 4.2% (AMD D&A margin FY2015); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc117_qa0__after123", 1.0, "ANS: GOLD 'cash flow from operations was highest for Nike FY2023'; PRED matches semantically → 1.0."),
    ("doc20_qa0__after124", 0.0, "ANS: GOLD $11588 (Amazon net income FY2019); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc128_qa0__after124", 1.0, "ACK: src=doc128 not yet ingested; PRED honestly refuses on PepsiCo FY2023 guidance. Honest refusal."),
    ("doc55_qa0__after124", 0.0, "ANS: GOLD 'entertainment segment 9% growth Q2 FY24'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc24_qa0__after124", 0.0, "ANS: GOLD Amcor acquisitions list; PRED refuses. Refusal on definitive list → 0.0."),
    ("doc3_qa0__after124", 0.0, "ANS: GOLD '3M OI -1.7%'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc139_qa0__after124", 1.0, "ACK: src=doc139 not yet ingested; PRED honestly refuses on Ulta merchandise inventories. Honest refusal."),
    ("doc85_qa0__after124", 0.0, "ANS: GOLD 'No, JnJ FY22 1.3% sales'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc58_qa0__after124", 0.0, "ANS: GOLD $382 (Block OCF FY2020); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc71_qa0__after124", 0.0, "ANS: GOLD 10.3% (Corning OI margin FY19-21); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc81_qa0__after124", 0.0, "ANS: GOLD -3.7; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc1_qa0__after125", 0.0, "ANS: GOLD $8.70 (3M net PPE FY2018); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc59_qa0__after125", 0.0, "ANS: GOLD $12645; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc97_qa0__after125", 0.0, "ANS: GOLD 'Corporate & Investment Bank $3725M' (JPM Q2 2022); PRED refuses. Refusal on definitive → 0.0."),
    ("doc143_qa0__after125", 1.0, "ACK: src=doc143 not yet ingested; PRED honestly refuses on Verizon retiree expected payments 2024. Honest refusal."),
    ("doc101_qa0__after125", 0.0, "ANS: GOLD $5818 (Lockheed NWC FY2021); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc47_qa0__after125", 0.0, "ANS: GOLD 'No, AWK -$1561M FY22'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc19_qa0__after125", 0.0, "ANS: GOLD 30.8% (Amazon revenue YoY FY16-17); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc77_qa0__after125", 0.0, "ANS: GOLD 'Yes, CVS multiple legal battles' with examples; PRED refuses. Refusal on Y/N → 0.0."),
    ("doc34_qa0__after125", 0.0, "ANS: GOLD 'AMD OI Xilinx'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc32_qa0__after125", 0.0, "ANS: GOLD list of AMD products; PRED refuses. Refusal on definitive list → 0.0."),
    ("doc132_qa0__after126", 1.0, "ACK: src=doc132 not yet ingested; PRED honestly refuses on Pfizer acquisitions. Honest refusal."),
    # ── batch 1261-1296 ────────────────────────────────────────────
    ("doc130_qa0__after126", 1.0, "ACK: src=doc130 not yet ingested; PRED honestly refuses on Pfizer PPNE growth. Honest refusal."),
    ("doc41_qa0__after126", 0.0, "ANS: GOLD 'gross margin not measured' (AMEX); PRED refuses. Refusal on definitive qualitative → 0.0."),
    ("doc40_qa0__after126", 0.0, "ANS: GOLD 'operating margin not measured' (AMEX); PRED refuses. Refusal on definitive qualitative → 0.0."),
    ("doc66_qa0__after126", 0.0, "ANS: GOLD 'Boeing 0.62 vs -14.76%'; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc99_qa0__after126", 0.0, "ANS: GOLD 6.25; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc7_qa0__after126", 0.0, "ANS: GOLD 'Yes, 3M increasing dividend 65 consecutive years'; PRED refuses. Refusal on Y/N → 0.0."),
    ("doc142_qa0__after126", 1.0, "ACK: src=doc142 not yet ingested; PRED honestly refuses on Verizon derivatives. Honest refusal."),
    ("doc98_qa0__after126", 0.0, "ANS: GOLD 'Yes. It decreased' (JPM VaR Q2 2023); PRED refuses. Refusal on Y/N → 0.0."),
    ("doc103_qa0__after126", 0.0, "ANS: GOLD $303 (MGM AP FY2018); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc28_qa0__after127", 0.0, "ANS: GOLD $2018M (Amcor Adj EBITDA FY2023); PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc130_qa0__after127", 1.0, "ACK: src=doc130 not yet ingested; PRED honestly refuses on Pfizer PPNE. Honest refusal."),
    ("doc62_qa0__after127", 0.0, "ANS: GOLD 'Yes, Boeing GM improved 4.8→5.3%'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc25_qa0__after127", 0.0, "ANS: GOLD 'Amcor global packaging leader'; PRED refuses. Refusal on definitive qualitative → 0.0."),
    ("doc26_qa0__after127", 0.0, "ANS: GOLD 'No, Amcor GM slight decline 0.8%'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc80_qa0__after127", 0.0, "ANS: GOLD 'Yes, Richard A. Johnson' (Foot Locker board); PRED refuses. Refusal on definitive Y/N+name → 0.0."),
    ("doc135_qa0__after127", 1.0, "ACK: src=doc135 not yet ingested; PRED honestly refuses on Pfizer business segments. Honest refusal."),
    ("doc100_qa0__after127", 0.0, "ANS: GOLD 1.33; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc123_qa0__after127", 0.25, "ANS: GOLD $9068 PRED '$12,275 million' — confident wrong specific → 0.25."),
    ("doc14_qa0__after127", 0.0, "ANS: GOLD 'Yes, Adobe FCF conv 143→156%'; PRED refuses. Refusal on Y/N+numeric → 0.0."),
    ("doc72_qa0__after128", 0.0, "ANS: GOLD 'Corning tax 20→23%'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc131_qa0__after128", 1.0, "ACK: src=doc131 not yet ingested; PRED honestly refuses on Pfizer 2019 net income events. Honest refusal."),
    ("doc106_qa0__after128", 0.0, "ANS: GOLD 'Las Vegas ~90% EBITDAR'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc39_qa0__after128", 0.0, "ANS: GOLD 'US, EMEA, APAC, LACC' (AMEX geos); PRED refuses. Refusal on definitive list → 0.0."),
    ("doc117_qa0__after128", 0.0, "ANS: GOLD 'cash flow from ops highest Nike FY23'; PRED refuses this time (different from 1203/1239). Refusal on definitive → 0.0."),
    ("doc141_qa0__after128", 1.0, "ACK: src=doc141 not yet ingested; PRED honestly refuses on Ulta wages FY23. Honest refusal."),
    ("doc32_qa0__after128", 0.0, "ANS: GOLD AMD products list; PRED refuses. Refusal on definitive list → 0.0."),
    ("doc98_qa0__after128", 0.0, "ANS: GOLD 'Yes. It decreased'; PRED refuses. Refusal on Y/N → 0.0."),
    ("doc41_qa0__after128", 0.0, "ANS: GOLD 'gross margin not measured'; PRED refuses. Refusal on definitive qualitative → 0.0."),
    ("doc79_qa0__after128", 0.0, "ANS: GOLD 'Yes, she was previous Ulta CEO' (Foot Locker new CEO); PRED refuses. Refusal on Y/N+narrative → 0.0."),
    ("doc134_qa0__after129", 1.0, "ACK: src=doc134 not yet ingested; PRED honestly refuses on Pfizer Q2 2023. Honest refusal."),
    ("doc42_qa0__after129", 0.0, "ANS: GOLD 'AMEX tax 24.6→21.6%'; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc85_qa0__after129", 0.0, "ANS: GOLD 'No, JnJ FY22 1.3% sales'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc124_qa0__after129", 0.25, "ANS: GOLD 16.5% PRED '15.4%' — diff 1.1pp, ~6.7% relative, OUTSIDE 5% tolerance → 0.25 confident wrong specific."),
    ("doc59_qa0__after129", 0.0, "ANS: GOLD $12645; PRED refuses. Refusal on definitive numeric → 0.0."),
    ("doc123_qa0__after129", 0.25, "ANS: GOLD $9068 PRED '$13,275 million' — confident wrong specific → 0.25."),
    ("doc0_qa0__after129", 0.0, "ANS: GOLD $1577; PRED refuses. Refusal on definitive numeric → 0.0."),
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
        print(f"Mean: {sum(scores)/len(scores):.4f}")

    total = sum(1 for _ in RESULTS.read_text(encoding="utf-8").splitlines() if _.strip())
    print(f"Total entries in results.jsonl: {total} / 1500 ({100*total/1500:.1f}%)")


if __name__ == "__main__":
    main()
