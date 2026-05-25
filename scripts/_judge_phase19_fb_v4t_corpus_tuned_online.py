"""Manual Claude 1-by-1 judgments for Phase 1.9 Protocol A
FinanceBench v4t-corpus-tuned online cell.

Applies the rubric in evaluation/claude_judge_protocol.md. Each tuple is
(qid_suffix, judge_score, rationale). Numeric tolerance 5%; refusals = 0.0;
exact yes/no required; multi-fact partials scored proportionally.

Run: python scripts/_judge_phase19_fb_v4t_corpus_tuned_online.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

JUDGMENTS = [
    # qid_short, judge_score, rationale
    ("doc0_qa0", 1.0, "Predicted $1,577M = gold $1577 exactly."),
    ("doc1_qa0", 1.0, "8.738B vs gold 8.70B; 0.4% off, within 5%."),
    ("doc2_qa0", 0.0, "Predicted Yes (capital-intensive) contradicts gold No (managing efficiently)."),
    ("doc3_qa0", 0.75, "Captures SG&A/litigation/PFAS/Russia/divestiture drivers; misses gross-margin + 1.7% headline figure."),
    ("doc4_qa0", 0.5, "Names Consumer segment but no -0.9% figure or growth direction."),
    ("doc5_qa0", 0.0, "Refusal: quick ratio 'not provided'; gold gives definitive 0.96."),
    ("doc6_qa0", 1.0, "Lists MMM26/MMM30/MMM31 = gold debt securities for 3M."),
    ("doc7_qa0", 1.0, "Yes + 65th consecutive year matches gold exactly."),
    ("doc8_qa0", 0.0, "25.73 vs gold 24.26: 6.1% off, exceeds 5% tolerance."),
    ("doc9_qa0", 0.0, "3.5% vs gold 1.9%: way off."),
    ("doc10_qa0", 1.0, "0.66 = 0.66 exactly."),
    ("doc11_qa0", 1.0, "65.4% = 65.4% exactly."),
    ("doc12_qa0", 1.0, "0.83 = 0.83 exactly."),
    ("doc13_qa0", 0.0, "Predicted Yes (improving op margin) contradicts gold No (declined 36.8% -> 34.6%)."),
    ("doc14_qa0", 0.0, "Refusal 'cannot be determined'; gold gives definitive Yes + ~10% improvement."),
    ("doc15_qa0", 1.0, "0 = 0 exactly."),
    ("doc16_qa0", 0.0, "Truncated calc cuts off before division; no final inventory turnover ratio given."),
    ("doc17_qa0", 0.0, "-1.42 vs gold -0.02: way off."),
    ("doc18_qa0", 0.0, "36.12 vs gold 93.86: way off."),
    ("doc19_qa0", 1.0, "30.8% = 30.8% exactly."),
    ("doc20_qa0", 1.0, "11,588 = $11,588 exactly."),
    ("doc21_qa0", 1.0, "$1,615.9M vs gold $1616: 0.006% off."),
    ("doc22_qa0", 1.0, "Captures Second Supplemental Indenture + Amcor Finance/Flexibles substitution story."),
    ("doc23_qa0", 0.0, "Refusal 'quick ratio not explicitly provided'; gold gives definitive 0.67 -> 0.69."),
    ("doc24_qa0", 0.5, "Lists 2 of 3 FY2023 acquisitions (Shanghai, Czech); misses NZ protein-packaging; tags Czech as FY2022 erroneously."),
    ("doc25_qa0", 1.0, "Packaging industry / food/beverage/pharma/medical use matches gold."),
    ("doc26_qa0", 0.0, "Refuses with 'gross margin not useful'; gold gives definitive No with -0.8% figure."),
    ("doc27_qa0", 0.25, "Mentions employee + fixed-asset + other cost categories but no 87% percentage."),
    ("doc28_qa0", 1.0, "2,018 million = gold $2,018mn FY2023."),
    ("doc29_qa0", 0.0, "Predicted -5% contradicts gold 'flat' for real growth FY23 vs FY22."),
    ("doc30_qa0", 1.0, "4.18% vs gold 4.2%: 0.5% off."),
    ("doc31_qa0", 0.0, "Refusal 'quick ratio not provided'; gold gives definitive 1.57 + Yes."),
    ("doc32_qa0", 1.0, "Lists CPUs/GPUs/DPUs/FPGAs/SoC = gold AMD product mix."),
    ("doc33_qa0", 0.75, "Captures Data Center 64% (EPYC) + Gaming 21% drivers; less complete than gold's full breakdown."),
    ("doc34_qa0", 1.0, "Names amortization of intangibles (Xilinx) as driver = gold."),
    ("doc35_qa0", 1.0, "Operating activities + $3,565M matches gold (most cashflow from Operations FY22)."),
    ("doc36_qa0", 1.0, "Data Center segment = gold."),
    ("doc37_qa0", 1.0, "Yes + one customer + 16% matches gold exactly."),
    ("doc38_qa0", 0.0, "Lists 3M's bonds (MMM26/30/31) — cross-doc memory bleed; gold says 'There are none' for AmEx."),
    ("doc39_qa0", 1.0, "Lists US/EMEA/APAC/LACC = gold AmEx geographies."),
    ("doc40_qa0", 1.0, "Operating margin not useful metric for AmEx matches gold."),
    ("doc41_qa0", 1.0, "Gross margin not useful for AmEx matches gold."),
    ("doc42_qa0", 0.75, "21.6% vs gold (likely 22.6%): ~4.4% off; direction (drop) right."),
    ("doc43_qa0", 1.0, "Customer deposits = Customer deposits exactly."),
    ("doc44_qa0", 1.0, "Yes + Card Member retention high = gold."),
    ("doc45_qa0", 1.0, "0.389B vs gold $0.40: 2.75% off, within tolerance."),
    ("doc46_qa0", 0.0, "1,199M vs gold $1832: way off."),
    ("doc47_qa0", 0.5, "Arithmetic correct (-$1,561M matches gold), and notes 'negative... does not have sufficient', but opens with 'Yes positive' (yes/no flipped)."),
    ("doc48_qa0", 0.0, "3.1% vs gold 2.8%: 10.7% off, exceeds tolerance."),
    ("doc49_qa0", 1.0, "5,409 = $5409 exactly."),
    ("doc50_qa0", 0.0, "Predicted 'fluctuated >2%, not consistent' contradicts gold 'Yes consistent, minor decline 1.1%'."),
    ("doc51_qa0", 0.5, "Lists 2 acquisitions (Current Health, Yardbird) but misses 'both partially owned' detail."),
    ("doc52_qa0", 1.0, "Operating activities + $1,824M = gold's most cashflow from operations FY23 $1.8bn."),
    ("doc53_qa0", 1.0, "Yes + $1,874M -> $1,093M = 42% decline matches gold's ~42%."),
    ("doc54_qa0", 0.5, "Yes (decline) correct; numbers 930->907 differ from gold's 982->969 (different reporting periods cited)."),
    ("doc55_qa0", 1.0, "Entertainment + 9% + gaming-driven matches gold exactly."),
    ("doc56_qa0", 1.0, "1.74 vs gold 1.73: 0.6% off."),
    ("doc57_qa0", 0.0, "17.5% vs gold 101.5%: way off."),
    ("doc58_qa0", 1.0, "381.6M vs gold $382: 0.1% off."),
    ("doc59_qa0", 1.0, "$12,645 = $12645 exactly."),
    ("doc60_qa0", 1.0, "Yes + Commercial Airplanes >20% matches gold's 'product/service categories >20% Boeing revenue'."),
    ("doc61_qa0", 1.0, "Yes + Lion Air 610 + Ethiopian 302 matches gold's multiple lawsuits from 2018 crash."),
    ("doc62_qa0", 0.0, "Refuses 'gross margin not useful for Boeing'; gold gives definitive Yes + improving profile."),
    ("doc63_qa0", 0.75, "Names limited commercial airlines but misses US Government as second primary customer."),
    ("doc64_qa0", 1.0, "Yes + cyclicality matches gold (with airline industry context)."),
    ("doc65_qa0", 1.0, "Names 737/777/787 production-rate increases matches gold's forecast."),
    ("doc66_qa0", 0.5, "Directional (lower in FY22, $(31)M vs $743M benefit) but no percentages (gold: 0.62% vs -14.76%)."),
    ("doc67_qa0", 0.0, "1.46% vs gold 0.01: scale way off (Coca-Cola ROA FY2017)."),
    ("doc68_qa0", 1.0, "Arrives at 39.7% via full calc; matches gold exactly."),
    ("doc69_qa0", 1.0, "0.80 vs gold 0.8: exact (trailing zero)."),
    ("doc70_qa0", 1.0, "Arrives at 63.49 (gold 63.86): 0.58% off, within tolerance."),
    ("doc71_qa0", 0.0, "14.0% vs gold 10.3%: 36% off."),
    ("doc72_qa0", 1.0, "20% FY21 -> 23% FY22 = gold Corning tax rate change."),
    ("doc73_qa0", 1.0, "Yes + working capital calc = gold positive working capital $831M FY22 (truncated)."),
    ("doc74_qa0", 1.0, "$59,268 = $59268 exactly."),
    ("doc75_qa0", 1.0, "17.19 vs gold 17.98: 4.4% off, within tolerance."),
    ("doc76_qa0", 1.0, "Yes capital-intensive with PP&E/goodwill/intangibles = gold (extensive asset base + ROA evidence)."),
    ("doc77_qa0", 1.0, "Yes + prescription pricing lawsuits matches gold (multiple ongoing legal battles)."),
    ("doc78_qa0", 0.75, "Yes (paid dividends) correct but missing $0.55/share/quarter specifics."),
    ("doc79_qa0", 1.0, "Yes + Mary Dillon + former Ulta CEO matches gold."),
    ("doc80_qa0", 1.0, "Yes + Richard A. Johnson (with vote count) = gold."),
    ("doc81_qa0", 0.0, "56.73 days vs gold -3.7: way off (CCC for General Mills FY19)."),
    ("doc82_qa0", 1.0, "0.68 = 0.68 exactly."),
    ("doc83_qa0", 1.0, "$3,215.4M vs gold $3215: 0.01% off."),
    ("doc84_qa0", 1.0, "0.54 = 0.54 exactly."),
    ("doc85_qa0", 1.0, "No + 1.3% sales vs 13.6% FY21 matches gold (JnJ not high-growth FY22)."),
    ("doc86_qa0", 0.75, "Captures COVID vaccine exit / PFAS / Russia drivers; less complete than gold's full list."),
    ("doc87_qa0", 0.0, "Calc truncated before division; no final 2.7x ratio given."),
    ("doc88_qa0", 0.0, "Predicted Yes (accelerate) contradicts gold No (decelerate from 3.6% to 3.5%)."),
    ("doc89_qa0", 1.0, "US 3.0% + international -0.6% matches gold exactly."),
    ("doc90_qa0", 1.0, "Consumer Health discontinued from Aug 30 2023 = gold."),
    ("doc91_qa0", 1.0, "$20 billion = gold approximately $20 billion."),
    ("doc92_qa0", 1.0, "$13.2 billion = gold exactly."),
    ("doc93_qa0", 1.0, "Yes + 20.0% -> 20.1% matches gold exactly."),
    ("doc94_qa0", 0.0, "Consumer & Community Banking; gold is Corporate (with -$473M net revenue)."),
    ("doc95_qa0", 1.0, "$66.56/share = gold exactly."),
    ("doc96_qa0", 1.0, "JPM gross margin not relevant (financial institution) matches gold."),
    ("doc97_qa0", 0.0, "Consumer & Community Banking; gold is Corporate & Investment Bank (net income $3725M)."),
    ("doc98_qa0", 1.0, "Yes + decreased + $7M figure matches gold (avg total VaR Q2 23 vs Q2 22)."),
    ("doc99_qa0", 1.0, "6.20 vs gold 6.25: 0.8% off."),
    ("doc100_qa0", 1.0, "1.30 vs gold 1.33: 2.3% off."),
    ("doc101_qa0", 1.0, "$5,818M = $5818 exactly."),
    ("doc102_qa0", 1.0, "0.4% = 0.4% exactly (Lockheed Martin revenue CAGR FY20-FY22)."),
    ("doc103_qa0", 1.0, "302.6M vs gold $303: 0.13% off."),
    ("doc104_qa0", 0.0, "10.0% vs gold 7.9%: 27% off."),
    ("doc105_qa0", 1.0, "Yes + $0.01/share annual dividend FY2022 matches gold (MGM)."),
    ("doc106_qa0", 0.5, "Names Las Vegas (Strip Resorts) but no ~90% EBITDAR figure."),
    ("doc107_qa0", 0.0, "1.61 vs gold 0 (gold: 'as EBIT is negative, coverage ratio is zero')."),
    ("doc108_qa0", 1.0, "MGM China + worst topline matches gold."),
    ("doc109_qa0", 0.75, "Corporate bonds correct; missing ~82% share detail."),
    ("doc110_qa0", 1.0, "$32,780M = $32780 exactly."),
    ("doc111_qa0", 0.0, "Predicted Yes (increased) contradicts gold No (decreased $2.5bn) — pred cherry-picks current-portion increase."),
    ("doc112_qa0", 0.0, "45.0% vs gold 5.4%: way off (Netflix EBITDA margin FY15)."),
    ("doc113_qa0", 1.0, "5,466.3M vs gold $5466: 0.005% off."),
    ("doc114_qa0", 1.0, "56.2% vs gold 55.1%: 2% off."),
    ("doc115_qa0", 1.0, "16,525 = $16525 exactly."),
    ("doc116_qa0", 1.0, "3.59 vs gold 3.46: 3.8% off, within tolerance."),
    ("doc117_qa0", 1.0, "Operating activities + $5,841M = gold (Nike FY23 highest cashflow)."),
    ("doc118_qa0", 0.75, "Yes + positive working capital correct, but doesn't reach gold's specific $1.6Bn figure."),
    ("doc119_qa0", 1.0, "4.625B vs gold $4.60: 0.5% off."),
    ("doc120_qa0", 0.25, "Lists Africa/ME/South Asia/AsiaPac but misses North America/Latin America/Europe (3 of 6 regions wrong/missing)."),
    ("doc121_qa0", 0.75, "Acknowledges 'no material impact' which aligns with gold 'not involved in material legal battles'; less direct."),
    ("doc122_qa0", 1.0, "411 = $411M restructuring FY22 matches gold."),
    ("doc123_qa0", 0.0, "$12,275M vs gold $9068: 35% off."),
    ("doc124_qa0", 1.0, "Arrives at 16.5% via full calc = gold."),
    ("doc125_qa0", 1.0, "Names shareholder net-zero proposal + vote counts (defeated) = gold."),
    ("doc126_qa0", 1.0, "$400M increase ($3.8B -> $4.2B) = gold."),
    ("doc127_qa0", 1.0, "$8.4B total ($4.2B + $4.2B) = gold."),
    ("doc128_qa0", 1.0, "Raised guidance due to strong start FY23 = gold."),
    ("doc129_qa0", 1.0, "Raised core CC EPS by 1pp (8% -> 9%) = gold exactly."),
    ("doc130_qa0", 1.0, "Yes + PP&E $13,745M -> $14,882M (positive YoY) = gold."),
    ("doc131_qa0", 1.0, "Yes + Consumer Healthcare JV gain matches gold."),
    ("doc132_qa0", 0.75, "Trillium + Array correct (2 of 3); third (Therachon) named only as 'not specified'."),
    ("doc133_qa0", 0.0, "$700M vs gold 77.78: pred answered separation cost not the ratio asked."),
    ("doc134_qa0", 1.0, "Developed Rest of World = Developed Rest of the World (minor article diff)."),
    ("doc135_qa0", 1.0, "Yes + Upjohn separation = gold spinning off Upjohn."),
    ("doc136_qa0", 0.0, "Lists common stock (not a debt security); gold says 'There are none' for Ulta."),
    ("doc137_qa0", 0.0, "Refusal 'passages do not contain information'; gold is definitive 'did not make any acquisitions FY23/FY22'."),
    ("doc138_qa0", 1.0, "Lower marketing expenses + leverage of incentive comp + higher sales = gold drivers."),
    ("doc139_qa0", 1.0, "47 new stores + new brand launches + first-half FY24 inventory = gold."),
    ("doc140_qa0", 0.0, "22.5% vs gold 36%: 38% off."),
    ("doc141_qa0", 0.0, "Decrease contradicts gold (wages as % of net sales increased FY23)."),
    ("doc142_qa0", 0.75, "Cross currency swaps correct; missing $32,502M notional value."),
    ("doc143_qa0", 0.5, "$1,097M pension correct; missing $862M health care/life insurance estimate."),
    ("doc144_qa0", 0.0, "Refusal 'quick ratio not provided'; gold gives definitive No + 0.54 figure."),
    ("doc145_qa0", 0.75, "Yes capital-intensive + $307,689M PP&E correct; missing 2.77 ratio."),
    ("doc146_qa0", 0.25, "Numbers ($150,868M -> $150,639M, $229M decrease) match gold; yes/no answer flipped (pred Yes, gold No)."),
    ("doc147_qa0", 1.0, "Arrives at 42.52 (gold 42.69): 0.4% off, within tolerance."),
    ("doc148_qa0", 0.0, "-0.6% vs gold 0.2%: sign + magnitude off."),
    ("doc149_qa0", 0.0, "3.8% vs gold 6.2%: 39% off."),
]


def main() -> None:
    fpath = Path("results/stage3/judge_queue/financebench__v4t-corpus-tuned__online__seed42/results.jsonl")
    prefix = "financebench__v4t-corpus-tuned__online__"
    # Idempotent: skip already-judged qids
    existing = set()
    if fpath.is_file():
        for line in fpath.open(encoding="utf-8"):
            line = line.strip()
            if line:
                existing.add(json.loads(line)["qid"])
    added = 0
    with fpath.open("a", encoding="utf-8") as fh:
        for qsuffix, score, rationale in JUDGMENTS:
            qid = f"{prefix}{qsuffix}__seed42"
            if qid in existing:
                continue
            fh.write(json.dumps({"qid": qid, "judge_score": score, "rationale": rationale}, ensure_ascii=False) + "\n")
            added += 1
    # Reload + summarize
    scores = []
    with fpath.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                scores.append(json.loads(line)["judge_score"])
    from collections import Counter
    dist = Counter(scores)
    print(f"Appended {added} judgments (total now {len(scores)})")
    print(f"Score distribution: {dict(sorted(dist.items()))}")
    print(f"Mean judge: {sum(scores)/len(scores):.4f}")


if __name__ == "__main__":
    main()
