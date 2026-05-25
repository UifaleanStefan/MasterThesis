"""Phase 1.9 — attention-corpus-tuned calibration cell — Part 3.

Entries 0299-0451 (153 entries). Mid-corpus pattern.

Notable wins: doc25 'Amcor packaging' semantic match (twice), doc26 ANS
'gross profit decline' matches Y/N direction, doc34 ANS 'Xilinx amortization'
exact match, doc35 'AMD cashflow ops $3,565M' match, doc33 'AMD EPYC+semi-custom+Xilinx'
match, doc21 1615.9 exact, doc36 'Data Center' exact, doc7 65 years match,
doc15 0 exact, doc1 8.738B in tolerance, doc125/doc90 inferred matches,
doc30 4.18% in tolerance of 4.2%, doc52 cash flow from ops match (no $).
Confident-wrong: doc18 DPO 36.x vs 93.86 (3 times), doc12 0.52 vs 0.83,
doc149 7.5% vs 6.2%, doc16 11.98 vs 9.5 turnover, doc10 1.15 vs 0.66,
doc11 -99.6% calc vs 65.4%, doc43 'long-term debt' vs 'Customer deposits',
doc17 -1.32 vs -0.02, doc29 'decrease 5%' vs 'flat', doc122 '0' vs $411M.
Y/N flip: doc2 'Yes capital-intensive' vs 'No'. Refusal-on-definitive:
doc0 $1577, doc5 quick ratio, doc31 quick ratio.
"""

from __future__ import annotations

import json
from pathlib import Path

QID_PREFIX = "financebench__attention-corpus-tuned__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__attention-corpus-tuned__calibration__seed42/results.jsonl")

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc18_qa0__after29", 0.25, "ANS GOLD 93.86 PRED 36.12 → confident wrong DPO → 0.25."),
    ("doc12_qa0__after30", 0.25, "ANS GOLD 0.83 PRED 0.52 → confident wrong (37% off) → 0.25."),
    ("doc98_qa0__after30", 1.0, "ACK refuse → 1.0."),
    ("doc47_qa0__after30", 1.0, "ACK refuse → 1.0."),
    ("doc97_qa0__after30", 1.0, "ACK refuse → 1.0."),
    ("doc52_qa0__after30", 1.0, "ACK GOLD 'Best Buy cash flow from operations highest FY23 $1.8B'; PRED 'cash flow from operating activities brought in most cash flow Best Buy FY23' — qualitative match (most cash flow = operations), missing $1.8B but correct direction → 1.0."),
    ("doc0_qa0__after30", 0.0, "ANS src=doc0 INGESTED; GOLD $1577; PRED 'not explicitly provided' — refusal on definitive numeric → 0.0."),
    ("doc60_qa0__after30", 1.0, "ACK refuse → 1.0."),
    ("doc5_qa0__after30", 0.0, "ANS src=doc5 INGESTED; GOLD 'No, 0.96 quick ratio'; PRED 'not provided, cannot determine' — refusal on definitive Y/N+numeric → 0.0."),
    ("doc42_qa0__after30", 1.0, "ACK refuse → 1.0."),
    ("doc90_qa0__after30", 1.0, "ACK Consumer Health discontinued match → 1.0."),
    ("doc124_qa0__after31", 1.0, "ACK refuse → 1.0."),
    ("doc91_qa0__after31", 1.0, "ACK refuse → 1.0."),
    ("doc21_qa0__after31", 1.0, "ANS GOLD $1616 PRED '1,615.9' → exact (0.006% off) → 1.0."),
    ("doc63_qa0__after31", 0.5, "ACK same partial 'defense contractors' confusion → 0.5."),
    ("doc120_qa0__after31", 1.0, "ACK refuse → 1.0."),
    ("doc67_qa0__after31", 1.0, "ACK refuse → 1.0."),
    ("doc139_qa0__after31", 1.0, "ACK refuse → 1.0."),
    ("doc18_qa0__after31", 0.25, "ANS GOLD 93.86 PRED 36.45 → confident wrong → 0.25."),
    ("doc135_qa0__after31", 1.0, "ACK refuse → 1.0."),
    ("doc141_qa0__after31", 1.0, "ACK refuse → 1.0."),
    ("doc117_qa0__after32", 1.0, "ACK refuse → 1.0."),
    ("doc18_qa0__after32", 0.25, "ANS GOLD 93.86 PRED 36.12 → confident wrong → 0.25."),
    ("doc7_qa0__after32", 1.0, "ANS 65 years dividend match → 1.0."),
    ("doc115_qa0__after32", 1.0, "ACK refuse → 1.0."),
    ("doc47_qa0__after32", 1.0, "ACK refuse → 1.0."),
    ("doc106_qa0__after32", 1.0, "ACK refuse → 1.0."),
    ("doc87_qa0__after32", 1.0, "ACK refuse → 1.0."),
    ("doc56_qa0__after32", 1.0, "ACK refuse → 1.0."),
    ("doc77_qa0__after32", 1.0, "ACK refuse → 1.0."),
    ("doc112_qa0__after32", 0.5, "ACK GOLD 5.4% Netflix EBITDA margin FY2015; PRED shows EBITDA calc framework with $-481M operating income (wrong numbers — likely confused). Hedged partial with wrong inputs → 0.5."),
    ("doc135_qa0__after33", 1.0, "ACK refuse → 1.0."),
    ("doc144_qa0__after33", 1.0, "ACK refuse → 1.0."),
    ("doc18_qa0__after33", 0.25, "ANS GOLD 93.86 PRED 36.29 → confident wrong → 0.25."),
    ("doc34_qa0__after33", 0.25, "ACK GOLD 'AMD OI decrease driven by Xilinx amortization'; PRED talks about revenue increases from segments (Data Center 64%, Gaming 21%, Xilinx embedded) — different metric (revenue not OI), different drivers → 0.25 confident wrong reasons."),
    ("doc72_qa0__after33", 1.0, "ACK refuse → 1.0."),
    ("doc15_qa0__after33", 1.0, "ANS GOLD '0' PRED '0' → exact → 1.0."),
    ("doc90_qa0__after33", 1.0, "ACK Consumer Health match → 1.0."),
    ("doc89_qa0__after33", 1.0, "ACK refuse → 1.0."),
    ("doc64_qa0__after33", 1.0, "ACK 'Y Boeing cyclical' match → 1.0."),
    ("doc125_qa0__after33", 1.0, "ACK 'proposal not approved' matches 'defeated' → 1.0."),
    ("doc130_qa0__after34", 1.0, "ACK refuse → 1.0."),
    ("doc26_qa0__after34", 1.0, "ANS GOLD 'No, Amcor GM slight decline 0.8%'; PRED 'Amcor GP $2,725M from $2,820M, does not have improving GM' — Y/N matches direction (No improvement / decline) + quantifies decline ($95M = ~3.4%) → 1.0."),
    ("doc68_qa0__after34", 1.0, "ACK refuse → 1.0."),
    ("doc40_qa0__after34", 1.0, "ACK refuse → 1.0."),
    ("doc129_qa0__after34", 1.0, "ACK refuse → 1.0."),
    ("doc144_qa0__after34", 1.0, "ACK refuse → 1.0."),
    ("doc25_qa0__after34", 1.0, "ANS 'Amcor packaging' semantic match → 1.0."),
    ("doc34_qa0__after34", 1.0, "ANS GOLD 'AMD OI decrease driven by Xilinx amortization'; PRED 'Operating margin AMD FY22 impacted by decrease in OI primarily driven by amortization of Xilinx intangibles' — exact match → 1.0."),
    ("doc131_qa0__after34", 1.0, "ACK refuse → 1.0."),
    ("doc29_qa0__after34", 0.25, "ANS GOLD 'Real Growth flat FY23 vs FY22'; PRED 'decrease of 5%' — confident wrong direction → 0.25."),
    ("doc136_qa0__after35", 1.0, "ACK refuse → 1.0."),
    ("doc93_qa0__after35", 1.0, "ACK refuse → 1.0."),
    ("doc146_qa0__after35", 1.0, "ACK refuse → 1.0."),
    ("doc149_qa0__after35", 0.25, "ACK GOLD 6.2% PRED '7.5%' — confident wrong specific (~21% off) → 0.25."),
    ("doc42_qa0__after35", 1.0, "ACK refuse → 1.0."),
    ("doc85_qa0__after35", 1.0, "ACK refuse → 1.0."),
    ("doc98_qa0__after35", 1.0, "ACK refuse → 1.0."),
    ("doc92_qa0__after35", 1.0, "ACK refuse → 1.0."),
    ("doc78_qa0__after35", 1.0, "ACK refuse → 1.0."),
    ("doc100_qa0__after35", 1.0, "ACK refuse → 1.0."),
    ("doc88_qa0__after36", 1.0, "ACK refuse → 1.0."),
    ("doc69_qa0__after36", 1.0, "ACK refuse → 1.0."),
    ("doc120_qa0__after36", 1.0, "ACK refuse → 1.0."),
    ("doc112_qa0__after36", 1.0, "ACK hedged refusal ('not available, please provide') → 1.0."),
    ("doc133_qa0__after36", 1.0, "ACK refuse → 1.0."),
    ("doc136_qa0__after36", 1.0, "ACK refuse → 1.0."),
    ("doc145_qa0__after36", 1.0, "ACK refuse → 1.0."),
    ("doc131_qa0__after36", 1.0, "ACK refuse → 1.0."),
    ("doc31_qa0__after36", 0.0, "ANS src=doc31 INGESTED; GOLD 'Y, AMD quick ratio 1.57'; PRED 'not provided, cannot be determined' — refusal on definitive Y/N+numeric → 0.0."),
    ("doc3_qa0__after36", 0.75, "ANS doc3 OI — qualitative drivers (litigation, impairment, restructuring, cost mgmt) but no -1.7% → 0.75."),
    ("doc52_qa0__after37", 1.0, "ACK refuse → 1.0."),
    ("doc70_qa0__after37", 1.0, "ACK refuse → 1.0."),
    ("doc11_qa0__after37", 0.25, "ANS GOLD 65.4% PRED calc with FY16=$5,802 (wrong — should be ~$1.17B) yielding -99.6% — confident wrong → 0.25."),
    ("doc10_qa0__after37", 0.25, "ANS GOLD 0.66 PRED 1.15 → 74% off → 0.25."),
    ("doc90_qa0__after37", 1.0, "ACK Consumer Health match → 1.0."),
    ("doc54_qa0__after37", 1.0, "ACK refuse → 1.0."),
    ("doc50_qa0__after37", 1.0, "ACK refuse → 1.0."),
    ("doc107_qa0__after37", 1.0, "ACK refuse → 1.0."),
    ("doc129_qa0__after37", 1.0, "ACK refuse → 1.0."),
    ("doc108_qa0__after37", 1.0, "ACK refuse → 1.0."),
    ("doc90_qa0__after38", 1.0, "ACK Consumer Health match → 1.0."),
    ("doc138_qa0__after38", 1.0, "ACK refuse → 1.0."),
    ("doc43_qa0__after38", 0.25, "ACK GOLD 'Customer deposits' PRED 'long-term debt' — confident wrong specific → 0.25."),
    ("doc71_qa0__after38", 1.0, "ACK refuse → 1.0."),
    ("doc1_qa0__after38", 1.0, "ANS 8.738B in tolerance → 1.0."),
    ("doc27_qa0__after38", 0.5, "ANS GOLD '87% employee restructuring'; PRED 'employee + fixed asset + other costs $93M' — names employee as one component but no 87% specific and adds other categories → 0.5 partial."),
    ("doc140_qa0__after38", 1.0, "ACK refuse → 1.0."),
    ("doc24_qa0__after38", 0.5, "ANS GOLD lists Amcor FY23 acquisitions (Czech + Shanghai + New Zealand); PRED puts Czech in FY22 but Shanghai+NZ in FY23 with amounts. Mixed FY years; partial → 0.5."),
    ("doc135_qa0__after38", 1.0, "ACK refuse → 1.0."),
    ("doc88_qa0__after38", 1.0, "ACK refuse → 1.0."),
    ("doc115_qa0__after39", 1.0, "ACK refuse → 1.0."),
    ("doc92_qa0__after39", 1.0, "ACK refuse → 1.0."),
    ("doc146_qa0__after39", 1.0, "ACK refuse → 1.0."),
    ("doc76_qa0__after39", 1.0, "ACK refuse → 1.0."),
    ("doc80_qa0__after39", 1.0, "ACK refuse → 1.0."),
    ("doc8_qa0__after39", 0.5, "ANS GOLD 24.26 PRED 25.66 — diff 5.77% relative, just outside strict 5% tolerance but close ballpark → 0.5 partial."),
    ("doc33_qa0__after39", 1.0, "ANS GOLD 'AMD 2022 EPYC+semi-custom+Xilinx'; PRED '64% Data Center EPYC + 21% Gaming semi-custom + Embedded Xilinx products' — all three drivers match + growth %s → 1.0."),
    ("doc95_qa0__after39", 1.0, "ACK refuse → 1.0."),
    ("doc46_qa0__after39", 1.0, "ACK refuse → 1.0."),
    ("doc2_qa0__after39", 0.0, "ANS same Y/N flip 'Yes capital-intensive' vs gold 'No' → 0.0."),
    ("doc16_qa0__after40", 0.25, "ANS GOLD '9.5 inventory turnover' PRED '11.98 with calc' — confident wrong specific (26% off) → 0.25."),
    ("doc93_qa0__after40", 1.0, "ACK refuse → 1.0."),
    ("doc128_qa0__after40", 1.0, "ACK refuse → 1.0."),
    ("doc110_qa0__after40", 1.0, "ACK refuse → 1.0."),
    ("doc59_qa0__after40", 1.0, "ACK refuse → 1.0."),
    ("doc54_qa0__after40", 1.0, "ACK refuse → 1.0."),
    ("doc135_qa0__after40", 1.0, "ACK refuse → 1.0."),
    ("doc11_qa0__after40", 0.25, "ANS same -99.6% wrong calc → 0.25."),
    ("doc53_qa0__after40", 1.0, "ACK refuse → 1.0."),
    ("doc57_qa0__after40", 1.0, "ACK refuse → 1.0."),
    ("doc85_qa0__after41", 1.0, "ACK refuse → 1.0."),
    ("doc88_qa0__after41", 1.0, "ACK refuse → 1.0."),
    ("doc53_qa0__after41", 1.0, "ACK refuse → 1.0."),
    ("doc61_qa0__after41", 1.0, "ACK refuse → 1.0."),
    ("doc46_qa0__after41", 1.0, "ACK refuse → 1.0."),
    ("doc124_qa0__after41", 1.0, "ACK refuse → 1.0."),
    ("doc84_qa0__after41", 1.0, "ACK refuse → 1.0."),
    ("doc134_qa0__after41", 1.0, "ACK refuse → 1.0."),
    ("doc21_qa0__after41", 1.0, "ANS GOLD $1616 PRED '$1,615.9 million' → exact → 1.0."),
    ("doc87_qa0__after41", 1.0, "ACK refuse → 1.0."),
    ("doc106_qa0__after42", 1.0, "ACK refuse → 1.0."),
    ("doc124_qa0__after42", 1.0, "ACK refuse → 1.0."),
    ("doc98_qa0__after42", 1.0, "ACK refuse → 1.0."),
    ("doc56_qa0__after42", 1.0, "ACK refuse → 1.0."),
    ("doc36_qa0__after42", 1.0, "ANS GOLD 'Data Center' PRED 'Data Center segment' → exact → 1.0."),
    ("doc51_qa0__after42", 1.0, "ACK refuse → 1.0."),
    ("doc111_qa0__after42", 1.0, "ACK refuse → 1.0."),
    ("doc60_qa0__after42", 1.0, "ACK refuse → 1.0."),
    ("doc148_qa0__after42", 1.0, "ACK refuse → 1.0."),
    ("doc50_qa0__after42", 1.0, "ACK refuse → 1.0."),
    ("doc25_qa0__after43", 1.0, "ANS 'Amcor packaging' semantic match → 1.0."),
    ("doc114_qa0__after43", 1.0, "ACK refuse → 1.0."),
    ("doc133_qa0__after43", 1.0, "ACK refuse → 1.0."),
    ("doc141_qa0__after43", 1.0, "ACK refuse → 1.0."),
    ("doc55_qa0__after43", 1.0, "ACK refuse → 1.0."),
    ("doc85_qa0__after43", 1.0, "ACK refuse → 1.0."),
    ("doc27_qa0__after43", 0.5, "ANS same partial as 0385 → 0.5."),
    ("doc94_qa0__after43", 1.0, "ACK refuse → 1.0."),
    ("doc122_qa0__after43", 0.25, "ACK PRED '0' vs $411M → 0.25 confident wrong."),
    ("doc24_qa0__after43", 0.5, "ANS same partial as 0387 → 0.5."),
    ("doc76_qa0__after44", 1.0, "ACK refuse → 1.0."),
    ("doc35_qa0__after44", 1.0, "ANS GOLD 'AMD 2022 cashflow from ops highest'; PRED 'cash flows from operating activities brought in most cash flow AMD FY22 $3,565M' → 1.0 match + specific."),
    ("doc17_qa0__after44", 0.25, "ANS GOLD -0.02 PRED -1.32 — confident wrong (66x off) → 0.25."),
    ("doc30_qa0__after44", 1.0, "ANS GOLD 4.2% PRED '4.18%' (calc shown $167/$3991*100) — within tolerance (0.5% off) → 1.0."),
    ("doc66_qa0__after44", 1.0, "ACK refuse → 1.0."),
    ("doc101_qa0__after44", 1.0, "ACK refuse → 1.0."),
    ("doc95_qa0__after44", 1.0, "ACK refuse → 1.0."),
    ("doc67_qa0__after44", 1.0, "ACK refuse → 1.0."),
    ("doc53_qa0__after44", 1.0, "ACK refuse → 1.0."),
    ("doc141_qa0__after44", 1.0, "ACK refuse → 1.0."),
    ("doc124_qa0__after45", 1.0, "ACK refuse → 1.0."),
    ("doc141_qa0__after45", 1.0, "ACK refuse → 1.0."),
]


def main() -> None:
    existing = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line); existing.add(obj["qid"])
            except Exception: continue
    added, scores = 0, []
    with RESULTS.open("a", encoding="utf-8") as fh:
        for s, sc, r in JUDGMENTS:
            qid = f"{QID_PREFIX}{s}{QID_SUFFIX}"
            if qid in existing: continue
            fh.write(json.dumps({"qid": qid, "judge_score": float(sc), "rationale": r, "judge_model": "claude-opus-4.7-1m", "judge_protocol": "v1"}, ensure_ascii=False) + "\n")
            added += 1; scores.append(sc)
    print(f"Added {added}. Dist: {dict((f'{x:.2f}', scores.count(x)) for x in sorted(set(scores), reverse=True))}")
    if scores: print(f"Mean: {sum(scores)/len(scores):.4f}")
    total = sum(1 for _ in RESULTS.read_text(encoding="utf-8").splitlines() if _.strip())
    print(f"Total: {total}/1500 ({100*total/1500:.1f}%)")


if __name__ == "__main__":
    main()
