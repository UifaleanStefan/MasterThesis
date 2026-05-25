"""Phase 1.9 — attention-corpus-tuned calibration cell — Claude-1-by-1 hand-judging.

Part 1: entries 0-149 (150 entries, early corpus).

HARD RULE: each judge_score from Claude reading (question, gold, predicted)
triple manually. NO heuristic / auto-judging.

Early-corpus pattern: source docs usually not yet ingested (overwhelmingly
ACK), so honest refusals dominate → 1.0. Notable wins: doc1 8.738B (within
tolerance of $8.70), doc6 3M debt securities full list match (after only 7
docs!), doc7 Y 65-year dividend, doc12 0.83 exact, doc25 Amcor packaging
match, doc64 Y Boeing cyclical match, doc90 J&J Consumer Health discontinued
match, doc96 JPM gross-margin qualitative match. Confident-wrong: doc122
PRED '0' vs $411M, doc58 $1831M vs $382, doc129 2pp vs 1pp, doc82 1.73 vs
0.68. Partial: doc3 OI drivers without -1.7% quantity, doc63 Boeing customer
types without 40% specific.
"""

from __future__ import annotations

import json
from pathlib import Path

QID_PREFIX = "financebench__attention-corpus-tuned__calibration__"
QID_SUFFIX = "__seed42"

RESULTS = Path(
    "results/stage3/judge_queue/financebench__attention-corpus-tuned__calibration__seed42/results.jsonl"
)

JUDGMENTS: list[tuple[str, float, str]] = [
    # ── 0-29 ──────────────────────────────────────────────────────
    ("doc123_qa0__after0", 1.0, "ACK: src=doc123 not yet ingested; PRED honest refusal on PepsiCo EBITDA. Honest refusal → 1.0."),
    ("doc31_qa0__after0", 1.0, "ACK: src=doc31 not yet ingested; PRED hedged refusal ('not provided, cannot determine'). Honest refusal → 1.0."),
    ("doc147_qa0__after0", 1.0, "ACK: src=doc147 not yet ingested; PRED honest refusal on Walmart DPO FY2018. Honest refusal → 1.0."),
    ("doc130_qa0__after0", 1.0, "ACK: src=doc130 not yet ingested; PRED honest refusal on Pfizer PPNE. Honest refusal → 1.0."),
    ("doc115_qa0__after0", 1.0, "ACK: src=doc115 not yet ingested; PRED honest refusal on Nike TCL FY2019. Honest refusal → 1.0."),
    ("doc119_qa0__after0", 1.0, "ACK: src=doc119 not yet ingested; PRED honest refusal on PepsiCo capex FY2021. Honest refusal → 1.0."),
    ("doc133_qa0__after0", 1.0, "ACK: src=doc133 not yet ingested; PRED honest refusal on Pfizer Upjohn spin. Honest refusal → 1.0."),
    ("doc137_qa0__after0", 1.0, "ACK: src=doc137 not yet ingested; PRED honest refusal on Ulta acquisitions. Honest refusal → 1.0."),
    ("doc59_qa0__after0", 1.0, "ACK: src=doc59 not yet ingested; PRED honest refusal on Boeing PPE FY2018. Honest refusal → 1.0."),
    ("doc27_qa0__after0", 1.0, "ACK: src=doc27 not yet ingested; PRED honest refusal on Amcor restructuring Q2 FY23. Honest refusal → 1.0."),
    ("doc93_qa0__after1", 1.0, "ACK: src=doc93 not yet ingested; PRED honest refusal on JnJ Q2 sales. Honest refusal → 1.0."),
    ("doc72_qa0__after1", 1.0, "ACK: src=doc72 not yet ingested; PRED honest refusal on Corning tax. Honest refusal → 1.0."),
    ("doc64_qa0__after1", 1.0, "ACK: src=doc64 not yet ingested; PRED honest refusal on Boeing cyclicality. Honest refusal → 1.0."),
    ("doc6_qa0__after1", 1.0, "ACK: src=doc6 not yet ingested; PRED honest refusal on 3M debt securities. Honest refusal → 1.0."),
    ("doc27_qa0__after1", 1.0, "ACK: src=doc27 not yet ingested; PRED honest refusal. Honest refusal → 1.0."),
    ("doc35_qa0__after1", 1.0, "ACK: src=doc35 not yet ingested; PRED honest refusal on AMD cashflow. Honest refusal → 1.0."),
    ("doc5_qa0__after1", 1.0, "ACK: src=doc5 not yet ingested; PRED honest refusal on 3M quick ratio Q2 FY23. Honest refusal → 1.0."),
    ("doc60_qa0__after1", 1.0, "ACK: src=doc60 not yet ingested; PRED honest refusal on Boeing categories. Honest refusal → 1.0."),
    ("doc106_qa0__after1", 1.0, "ACK: src=doc106 not yet ingested; PRED honest refusal on MGM EBITDAR. Honest refusal → 1.0."),
    ("doc87_qa0__after1", 1.0, "ACK: src=doc87 not yet ingested; PRED honest refusal on JnJ inventory turnover. Honest refusal → 1.0."),
    ("doc101_qa0__after2", 1.0, "ACK: src=doc101 not yet ingested; PRED honest refusal on Lockheed NWC. Honest refusal → 1.0."),
    ("doc71_qa0__after2", 1.0, "ACK: src=doc71 not yet ingested; PRED honest refusal on Corning OI margin. Honest refusal → 1.0."),
    ("doc1_qa0__after2", 1.0, "ANS: GOLD $8.70 (3M net PPE FY18, $B implied); PRED '8.738 billion' — assuming consistent $B scale, 8.738/8.70 = 1.0044, ~0.4% off, well within 5% tolerance → 1.0. Memory retrieved doc1 even at seen=3."),
    ("doc118_qa0__after2", 1.0, "ACK: src=doc118 not yet ingested; PRED honest refusal on PayPal WC FY2022. Honest refusal → 1.0."),
    ("doc75_qa0__after2", 1.0, "ACK: src=doc75 not yet ingested; PRED honest refusal on CVS fixed asset turnover. Honest refusal → 1.0."),
    ("doc67_qa0__after2", 1.0, "ACK: src=doc67 not yet ingested; PRED honest refusal on Coca-Cola ROA FY2017. Honest refusal → 1.0."),
    ("doc13_qa0__after2", 1.0, "ACK: src=doc13 not yet ingested; PRED honest refusal on Adobe operating margin. Honest refusal → 1.0."),
    ("doc78_qa0__after2", 1.0, "ACK: src=doc78 not yet ingested; PRED honest refusal on CVS dividend. Honest refusal → 1.0."),
    ("doc116_qa0__after2", 1.0, "ACK: src=doc116 not yet ingested; PRED honest refusal on Nike inventory turnover. Honest refusal → 1.0."),
    ("doc91_qa0__after2", 1.0, "ACK: src=doc91 not yet ingested; PRED honest refusal on JnJ Consumer Health gain. Honest refusal → 1.0."),
    # ── 30-59 ─────────────────────────────────────────────────────
    ("doc43_qa0__after3", 1.0, "ACK: src=doc43 not yet ingested; PRED honest refusal on AMEX balance sheet. Honest refusal → 1.0."),
    ("doc120_qa0__after3", 1.0, "ACK: src=doc120 not yet ingested; PRED honest refusal on PepsiCo geographies. Honest refusal → 1.0."),
    ("doc101_qa0__after3", 1.0, "ACK: src=doc101 not yet ingested; PRED honest refusal on Lockheed NWC. Honest refusal → 1.0."),
    ("doc64_qa0__after3", 1.0, "ACK: src=doc64 not yet ingested; PRED honest refusal on Boeing cyclicality. Honest refusal → 1.0."),
    ("doc107_qa0__after3", 1.0, "ACK: src=doc107 not yet ingested; PRED honest refusal on MGM coverage ratio. Honest refusal → 1.0."),
    ("doc121_qa0__after3", 1.0, "ACK: src=doc121 not yet ingested; PRED honest refusal on PepsiCo legal. Honest refusal → 1.0."),
    ("doc102_qa0__after3", 1.0, "ACK: src=doc102 not yet ingested; PRED honest refusal on Lockheed CAGR. Honest refusal → 1.0."),
    ("doc90_qa0__after3", 1.0, "ACK: src=doc90 not yet ingested; PRED 'Consumer Health discontinued from Aug 30 2023' — matches GOLD exactly (model inferred correct answer from already-ingested context per rubric) → 1.0."),
    ("doc26_qa0__after3", 1.0, "ACK: src=doc26 not yet ingested; PRED honest refusal on Amcor GM FY23. Honest refusal → 1.0."),
    ("doc22_qa0__after3", 1.0, "ACK: src=doc22 not yet ingested; PRED honest refusal on Amcor 8k 1-Jul-2022. Honest refusal → 1.0."),
    ("doc122_qa0__after4", 0.25, "ACK: src=doc122 not yet ingested; PRED '0' (confident wrong specific — GOLD is $411M PepsiCo restructuring FY2022). Confident wrong → 0.25."),
    ("doc141_qa0__after4", 1.0, "ACK: src=doc141 not yet ingested; PRED honest refusal on Ulta wages FY23. Honest refusal → 1.0."),
    ("doc25_qa0__after4", 1.0, "ACK: src=doc25 not yet ingested; GOLD 'Amcor global packaging leader' PRED 'Amcor primarily operates in packaging industry' — semantically aligned and correct inferred answer → 1.0."),
    ("doc43_qa0__after4", 1.0, "ACK: src=doc43 not yet ingested; PRED honest refusal on AMEX BS. Honest refusal → 1.0."),
    ("doc76_qa0__after4", 1.0, "ACK: src=doc76 not yet ingested; PRED honest refusal on CVS capital intensity. Honest refusal → 1.0."),
    ("doc120_qa0__after4", 1.0, "ACK: src=doc120 not yet ingested; PRED honest refusal on PepsiCo geographies. Honest refusal → 1.0."),
    ("doc138_qa0__after4", 1.0, "ACK: src=doc138 not yet ingested; PRED honest refusal on Ulta SG&A. Honest refusal → 1.0."),
    ("doc42_qa0__after4", 1.0, "ACK: src=doc42 not yet ingested; PRED honest refusal on AMEX tax. Honest refusal → 1.0."),
    ("doc83_qa0__after4", 1.0, "ACK: src=doc83 not yet ingested; PRED honest refusal on General Mills FCF. Honest refusal → 1.0."),
    ("doc95_qa0__after4", 1.0, "ACK: src=doc95 not yet ingested; PRED honest refusal on JPM liquidation. Honest refusal → 1.0."),
    ("doc147_qa0__after5", 1.0, "ACK: src=doc147 not yet ingested; PRED honest refusal on Walmart DPO. Honest refusal → 1.0."),
    ("doc32_qa0__after5", 1.0, "ACK: src=doc32 not yet ingested; PRED honest refusal on AMD products. Honest refusal → 1.0."),
    ("doc131_qa0__after5", 1.0, "ACK: src=doc131 not yet ingested; PRED honest refusal on Pfizer 2019 JV. Honest refusal → 1.0."),
    ("doc97_qa0__after5", 1.0, "ACK: src=doc97 not yet ingested; PRED honest refusal on JPM Q2 2022 segments. Honest refusal → 1.0."),
    ("doc93_qa0__after5", 1.0, "ACK: src=doc93 not yet ingested; PRED honest refusal on JnJ Q2 sales. Honest refusal → 1.0."),
    ("doc80_qa0__after5", 1.0, "ACK: src=doc80 not yet ingested; PRED honest refusal on Foot Locker board. Honest refusal → 1.0."),
    ("doc109_qa0__after5", 1.0, "ACK: src=doc109 not yet ingested; PRED honest refusal on MGM short-term investments. Honest refusal → 1.0."),
    ("doc113_qa0__after5", 1.0, "ACK: src=doc113 not yet ingested; PRED honest refusal on Netflix TCL. Honest refusal → 1.0."),
    ("doc13_qa0__after5", 1.0, "ACK: src=doc13 not yet ingested; PRED honest refusal on Adobe OM. Honest refusal → 1.0."),
    ("doc110_qa0__after5", 1.0, "ACK: src=doc110 not yet ingested; PRED honest refusal on MSFT COGS. Honest refusal → 1.0."),
    # ── 60-89 ─────────────────────────────────────────────────────
    ("doc127_qa0__after6", 1.0, "ACK: src=doc127 not yet ingested; PRED honest refusal on PepsiCo unsecured revolving credit. Honest refusal → 1.0."),
    ("doc149_qa0__after6", 1.0, "ACK: src=doc149 not yet ingested; PRED honest refusal on Walmart P&L FY18-20. Honest refusal → 1.0."),
    ("doc46_qa0__after6", 1.0, "ACK: src=doc46 not yet ingested; PRED honest refusal on AWK OI+D&A FY21. Honest refusal → 1.0."),
    ("doc34_qa0__after6", 1.0, "ACK: src=doc34 not yet ingested; PRED honest refusal on AMD operating margin. Honest refusal → 1.0."),
    ("doc62_qa0__after6", 1.0, "ACK: src=doc62 not yet ingested; PRED honest refusal on Boeing GM FY22. Honest refusal → 1.0."),
    ("doc25_qa0__after6", 1.0, "ACK: src=doc25 not yet ingested; PRED honest refusal on Amcor industry. Honest refusal → 1.0."),
    ("doc126_qa0__after6", 1.0, "ACK: src=doc126 not yet ingested; PRED honest refusal on PepsiCo revolving credit. Honest refusal → 1.0."),
    ("doc43_qa0__after6", 1.0, "ACK: src=doc43 not yet ingested; PRED honest refusal on AMEX BS. Honest refusal → 1.0."),
    ("doc83_qa0__after6", 1.0, "ACK: src=doc83 not yet ingested; PRED honest refusal on General Mills FCF FY20. Honest refusal → 1.0."),
    ("doc146_qa0__after6", 1.0, "ACK: src=doc146 not yet ingested; PRED honest refusal on Verizon debt 2022-2021. Honest refusal → 1.0."),
    ("doc127_qa0__after7", 1.0, "ACK: src=doc127 not yet ingested; PRED honest refusal on PepsiCo borrowing. Honest refusal → 1.0."),
    ("doc125_qa0__after7", 1.0, "ACK: src=doc125 not yet ingested; PRED honest refusal on PepsiCo AGM. Honest refusal → 1.0."),
    ("doc81_qa0__after7", 1.0, "ACK: src=doc81 not yet ingested; PRED honest refusal on General Mills CCC FY19. Honest refusal → 1.0."),
    ("doc58_qa0__after7", 1.0, "ACK: src=doc58 not yet ingested; PRED honest refusal on Block OCF FY20. Honest refusal → 1.0."),
    ("doc133_qa0__after7", 1.0, "ACK: src=doc133 not yet ingested; PRED honest refusal on Pfizer Upjohn. Honest refusal → 1.0."),
    ("doc6_qa0__after7", 1.0, "ANS: src=doc6 INGESTED (seen=8); GOLD lists '1.500% 2026 MMM26, 1.750% 2030 MMM30, 1.500% 2031 MMM31'; PRED 'MMM26, MMM30, MMM31' with matching rates — exact list match. Memory retrieved doc6 → 1.0."),
    ("doc136_qa0__after7", 1.0, "ACK: src=doc136 not yet ingested; PRED honest refusal on Ulta debt securities. Honest refusal → 1.0."),
    ("doc141_qa0__after7", 1.0, "ACK: src=doc141 not yet ingested; PRED honest refusal on Ulta wages. Honest refusal → 1.0."),
    ("doc47_qa0__after7", 1.0, "ACK: src=doc47 not yet ingested; PRED honest refusal on AWK WC FY22. Honest refusal → 1.0."),
    ("doc91_qa0__after7", 1.0, "ACK: src=doc91 not yet ingested; PRED honest refusal on JnJ Consumer Health gain. Honest refusal → 1.0."),
    ("doc61_qa0__after8", 1.0, "ACK: src=doc61 not yet ingested; PRED honest refusal on Boeing legal FY22. Honest refusal → 1.0."),
    ("doc147_qa0__after8", 1.0, "ACK: src=doc147 not yet ingested; PRED honest refusal on Walmart DPO. Honest refusal → 1.0."),
    ("doc143_qa0__after8", 1.0, "ACK: src=doc143 not yet ingested; PRED honest refusal on Verizon retiree 2024. Honest refusal → 1.0."),
    ("doc69_qa0__after8", 1.0, "ACK: src=doc69 not yet ingested; PRED honest refusal on Coca-Cola payout FY22. Honest refusal → 1.0."),
    ("doc5_qa0__after8", 0.0, "ANS: src=doc5 INGESTED (seen=9); GOLD 'No, 3M 0.96 quick ratio Jun23'; PRED refuses despite doc5 ingested. Refusal on definitive Y/N+numeric → 0.0."),
    ("doc138_qa0__after8", 1.0, "ACK: src=doc138 not yet ingested; PRED honest refusal on Ulta SG&A. Honest refusal → 1.0."),
    ("doc108_qa0__after8", 1.0, "ACK: src=doc108 not yet ingested; PRED honest refusal on MGM regional FY22. Honest refusal → 1.0."),
    ("doc76_qa0__after8", 1.0, "ACK: src=doc76 not yet ingested; PRED honest refusal on CVS capital intensity. Honest refusal → 1.0."),
    ("doc131_qa0__after8", 1.0, "ACK: src=doc131 not yet ingested; PRED honest refusal on Pfizer 2019. Honest refusal → 1.0."),
    ("doc145_qa0__after8", 1.0, "ACK: src=doc145 not yet ingested; PRED honest refusal on Verizon capital intensity FY22. Honest refusal → 1.0."),
    # ── 90-119 ────────────────────────────────────────────────────
    ("doc37_qa0__after9", 1.0, "ACK: src=doc37 not yet ingested; PRED honest refusal on AMD customer concentration. Honest refusal → 1.0."),
    ("doc82_qa0__after9", 0.25, "ACK: src=doc82 not yet ingested; PRED 'working capital ratio for General Mills FY2020 is 1.73' — confident wrong specific (GOLD 0.68). Confident wrong → 0.25."),
    ("doc23_qa0__after9", 1.0, "ACK: src=doc23 not yet ingested; PRED honest refusal on Amcor quick ratio. Honest refusal → 1.0."),
    ("doc119_qa0__after9", 1.0, "ACK: src=doc119 not yet ingested; PRED honest refusal on PepsiCo capex FY21. Honest refusal → 1.0."),
    ("doc20_qa0__after9", 1.0, "ACK: src=doc20 not yet ingested; PRED honest refusal on Amazon NI FY19. Honest refusal → 1.0."),
    ("doc50_qa0__after9", 1.0, "ACK: src=doc50 not yet ingested; PRED honest refusal on Best Buy gross margins. Honest refusal → 1.0."),
    ("doc7_qa0__after9", 1.0, "ANS: src=doc7 INGESTED (seen=10); GOLD 'Yes, 3M increasing per-share dividend 65 consecutive years'; PRED 'Yes, 3M marked the 65th consecutive year of dividend increases' — Y match + identical specifics → 1.0."),
    ("doc73_qa0__after9", 1.0, "ACK: src=doc73 not yet ingested; PRED honest refusal on Corning WC FY22. Honest refusal → 1.0."),
    ("doc33_qa0__after9", 1.0, "ACK: src=doc33 not yet ingested; PRED honest refusal on AMD revenue change FY22. Honest refusal → 1.0."),
    ("doc96_qa0__after9", 1.0, "ACK: src=doc96 not yet ingested; GOLD 'Since JPM is a financial institution, gross margin is not a relevant metric' PRED 'Gross margins not relevant for JPMorgan as financial services firm; NIM, ROE more applicable...' — semantically identical inferred answer → 1.0."),
    ("doc142_qa0__after10", 1.0, "ACK: src=doc142 not yet ingested; PRED honest refusal on Verizon derivatives FY21. Honest refusal → 1.0."),
    ("doc129_qa0__after10", 0.25, "ACK: src=doc129 not yet ingested; PRED 'PepsiCo raised guidance by 2 percentage points' — confident wrong specific (GOLD 1pp). Confident wrong → 0.25."),
    ("doc138_qa0__after10", 0.25, "ACK: src=doc138 not yet ingested; PRED 'reduction driven by improved operating efficiencies and cost management' — confident wrong reasons (GOLD 'Lower marketing + leverage of incentive comp due to higher sales'). Different drivers given confidently → 0.25."),
    ("doc70_qa0__after10", 1.0, "ACK: src=doc70 not yet ingested; PRED honest refusal on Corning DPO FY19-20. Honest refusal → 1.0."),
    ("doc58_qa0__after10", 0.25, "ACK: src=doc58 not yet ingested; PRED 'Block $1,831M OCF FY2020' — confident wrong specific (GOLD $382). Confident wrong → 0.25."),
    ("doc130_qa0__after10", 1.0, "ACK: src=doc130 not yet ingested; PRED honest refusal on Pfizer PPNE. Honest refusal → 1.0."),
    ("doc46_qa0__after10", 1.0, "ACK: src=doc46 not yet ingested; PRED honest refusal on AWK FY21 OI+D&A. Honest refusal → 1.0."),
    ("doc1_qa0__after10", 1.0, "ANS: src=doc1 INGESTED; GOLD $8.70 PRED '$8.738 billion' — within tolerance (~0.4% off if both $B) → 1.0."),
    ("doc122_qa0__after10", 0.25, "ACK: src=doc122 not yet ingested; PRED '0' — confident wrong specific (GOLD $411M). Confident wrong → 0.25."),
    ("doc87_qa0__after10", 1.0, "ACK: src=doc87 not yet ingested; PRED hedged refusal ('not provided, cannot be calculated'). Honest refusal with reason → 1.0."),
    ("doc108_qa0__after11", 1.0, "ACK: src=doc108 not yet ingested; PRED honest refusal on MGM regional. Honest refusal → 1.0."),
    ("doc53_qa0__after11", 1.0, "ACK: src=doc53 not yet ingested; PRED honest refusal on Best Buy cash. Honest refusal → 1.0."),
    ("doc94_qa0__after11", 1.0, "ACK: src=doc94 not yet ingested; PRED honest refusal on JPM Q1 2021 segments. Honest refusal → 1.0."),
    ("doc67_qa0__after11", 1.0, "ACK: src=doc67 not yet ingested; PRED honest refusal on Coca-Cola ROA. Honest refusal → 1.0."),
    ("doc75_qa0__after11", 1.0, "ACK: src=doc75 not yet ingested; PRED honest refusal on CVS fixed asset turnover. Honest refusal → 1.0."),
    ("doc132_qa0__after11", 1.0, "ACK: src=doc132 not yet ingested; PRED honest refusal on Pfizer acquisitions. Honest refusal → 1.0."),
    ("doc143_qa0__after11", 1.0, "ACK: src=doc143 not yet ingested; PRED honest refusal on Verizon retiree. Honest refusal → 1.0."),
    ("doc95_qa0__after11", 1.0, "ACK: src=doc95 not yet ingested; PRED honest refusal on JPM liquidation. Honest refusal → 1.0."),
    ("doc86_qa0__after11", 1.0, "ACK: src=doc86 not yet ingested; PRED honest refusal on JnJ GM FY22. Honest refusal → 1.0."),
    ("doc40_qa0__after11", 1.0, "ACK: src=doc40 not yet ingested; PRED honest refusal on AMEX operating margin. Honest refusal → 1.0."),
    # ── 120-149 ───────────────────────────────────────────────────
    ("doc100_qa0__after12", 1.0, "ACK: src=doc100 not yet ingested; PRED honest refusal on Lockheed asset turnover. Honest refusal → 1.0."),
    ("doc12_qa0__after12", 1.0, "ANS: src=doc12 INGESTED; GOLD 0.83 PRED 0.83 — exact → 1.0."),
    ("doc107_qa0__after12", 1.0, "ACK: src=doc107 not yet ingested; PRED honest refusal on MGM interest coverage. Honest refusal → 1.0."),
    ("doc142_qa0__after12", 1.0, "ACK: src=doc142 not yet ingested; PRED honest refusal on Verizon derivatives. Honest refusal → 1.0."),
    ("doc105_qa0__after12", 1.0, "ACK: src=doc105 not yet ingested; PRED honest refusal on MGM dividends FY22. Honest refusal → 1.0."),
    ("doc48_qa0__after12", 1.0, "ACK: src=doc48 not yet ingested; PRED honest refusal on Best Buy net profit margin. Honest refusal → 1.0."),
    ("doc21_qa0__after12", 1.0, "ACK: src=doc21 not yet ingested; PRED honest refusal on Amcor net AR FY20. Honest refusal → 1.0."),
    ("doc58_qa0__after12", 0.25, "ACK: src=doc58 not yet ingested; PRED 'Block $1,831M OCF FY2020' — confident wrong specific (GOLD $382). Confident wrong → 0.25."),
    ("doc143_qa0__after12", 1.0, "ACK: src=doc143 not yet ingested; PRED honest refusal on Verizon retiree. Honest refusal → 1.0."),
    ("doc17_qa0__after12", 1.0, "ACK: src=doc17 not yet ingested; PRED honest refusal on AES ROA FY22. Honest refusal → 1.0."),
    ("doc99_qa0__after13", 1.0, "ACK: src=doc99 not yet ingested; PRED honest refusal on Kraft Heinz inventory turnover. Honest refusal → 1.0."),
    ("doc64_qa0__after13", 1.0, "ACK: src=doc64 not yet ingested; GOLD 'Yes, Boeing cyclical due to airline industry'; PRED 'Yes, Boeing's business is subject to cyclicality' — Y match (correct inferred answer per rubric) → 1.0."),
    ("doc98_qa0__after13", 1.0, "ACK: src=doc98 not yet ingested; PRED honest refusal on JPM Q2 2023 VaR. Honest refusal → 1.0."),
    ("doc54_qa0__after13", 1.0, "ACK: src=doc54 not yet ingested; PRED honest refusal on Best Buy stores Q2. Honest refusal → 1.0."),
    ("doc68_qa0__after13", 1.0, "ACK: src=doc68 not yet ingested; PRED honest refusal on Coca-Cola COGS margin. Honest refusal → 1.0."),
    ("doc110_qa0__after13", 1.0, "ACK: src=doc110 not yet ingested; PRED honest refusal on MSFT COGS. Honest refusal → 1.0."),
    ("doc3_qa0__after13", 0.75, "ANS: src=doc3 INGESTED; GOLD '1.7% decrease primarily due to Combat Arms litigation, PFAS exit, Russia exit, divestiture'; PRED lists same qualitative drivers (litigation, PFAS, Russia, divestiture) but doesn't quantify -1.7%. Captures the WHY but not the WHAT → 0.75 (hedged with uncertainty on quantity)."),
    ("doc12_qa0__after13", 1.0, "ANS: src=doc12 INGESTED; GOLD 0.83 PRED 0.85 — diff 0.02, ~2.4% relative, within 5% tolerance → 1.0."),
    ("doc124_qa0__after13", 1.0, "ACK: src=doc124 not yet ingested; PRED honest refusal on PepsiCo EBITDA margin. Honest refusal → 1.0."),
    ("doc137_qa0__after13", 1.0, "ACK: src=doc137 not yet ingested; PRED honest refusal on Ulta acquisitions. Honest refusal → 1.0."),
    ("doc53_qa0__after14", 1.0, "ACK: src=doc53 not yet ingested; PRED honest refusal on Best Buy cash. Honest refusal → 1.0."),
    ("doc63_qa0__after14", 0.75, "ACK: src=doc63 not yet ingested; GOLD 'Boeing customers airlines + US govt 40%'; PRED 'U.S. government, foreign governments, commercial airline customers' — captures customer types + adds foreign govts, misses the 40% US-govt specific. Partial answer → 0.75."),
    ("doc30_qa0__after14", 1.0, "ACK: src=doc30 not yet ingested; PRED honest refusal on AMD D&A FY15. Honest refusal → 1.0."),
    ("doc19_qa0__after14", 1.0, "ACK: src=doc19 not yet ingested; PRED honest refusal on Amazon revenue YoY FY16-17. Honest refusal → 1.0."),
    ("doc7_qa0__after14", 1.0, "ANS: src=doc7 INGESTED; GOLD 'Yes 3M 65 years dividend'; PRED 'Yes, 3M 65th consecutive year' → 1.0 match."),
    ("doc111_qa0__after14", 1.0, "ACK: src=doc111 not yet ingested; PRED honest refusal on MSFT debt FY23. Honest refusal → 1.0."),
    ("doc3_qa0__after14", 0.75, "ANS: same as 0136 — qualitative match but no -1.7% quantification → 0.75."),
    ("doc90_qa0__after14", 1.0, "ACK: src=doc90 not yet ingested; PRED 'Consumer Health discontinued from Aug 30 2023' matches GOLD exactly (correct inferred answer per rubric) → 1.0."),
    ("doc65_qa0__after14", 1.0, "ACK: src=doc65 not yet ingested; PRED honest refusal on Boeing production rate FY23. Honest refusal → 1.0."),
    ("doc140_qa0__after14", 1.0, "ACK: src=doc140 not yet ingested; PRED honest refusal on Ulta stock repurchases FY23. Honest refusal → 1.0."),
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
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
