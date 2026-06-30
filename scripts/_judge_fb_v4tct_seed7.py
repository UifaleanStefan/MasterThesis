"""Apply Claude 1-by-1 judgments for financebench__v4t-corpus-tuned__batch__seed7 (n=150).

Multi-seed replicate (seed 7) of the FinanceBench v4t-corpus-tuned headline cell,
for cross-seed variance. Each (sfx -> (score, rationale)) is a hand judgment by
Claude against the FinanceBench rubric: numeric answers must match the gold value
(small rounding/format diffs -> 0.75-1.0; materially wrong number -> 0.0);
qualitative answers 1.0 fully correct, 0.75 substantially correct, 0.5 partial,
0.25 weakly related, 0.0 wrong/opposite or a refusal when the gold answer exists.
NO heuristics. Writes results.jsonl with judge_model + judge_protocol + rationale.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELL = ROOT / "results" / "stage3" / "judge_queue" / "financebench__v4t-corpus-tuned__batch__seed7"

J: dict[str, tuple[float, str]] = {
 "doc0_qa0": (0.0, "Gold capex $1577M; pred $1501M -- wrong number."),
 "doc1_qa0": (0.75, "Pred 8.738bn vs gold 8.70 -- same figure, minor rounding."),
 "doc2_qa0": (0.0, "Gold No (managing capex efficiently); pred Yes -- opposite."),
 "doc3_qa0": (0.75, "Captures the drivers (litigation, impairment, restructuring); omits the -1.7% magnitude."),
 "doc4_qa0": (1.0, "Names the gold segment (Consumer)."),
 "doc5_qa0": (0.0, "Gold quick ratio 0.96 (No); pred says not provided -- abstains."),
 "doc6_qa0": (0.75, "Lists the three gold notes (2026/2030/2031); omits trading symbols."),
 "doc7_qa0": (1.0, "Matches gold (65 consecutive years of dividend increases)."),
 "doc8_qa0": (0.0, "Gold 24.26; pred 25.73 -- wrong."),
 "doc9_qa0": (0.0, "Gold 1.9%; pred 3.5% -- wrong."),
 "doc10_qa0": (0.0, "Gold 0.66; pred 1.97 -- wrong."),
 "doc11_qa0": (1.0, "Pred's calc yields 65.4% = gold."),
 "doc12_qa0": (0.0, "Gold 0.83; pred 1.23 -- wrong."),
 "doc13_qa0": (0.0, "Gold No (declined 36.8->34.6); pred Yes improving -- opposite."),
 "doc14_qa0": (0.0, "Gold Yes (FCF conversion improved); pred says cannot determine -- abstains."),
 "doc15_qa0": (1.0, "Matches gold (0 restructuring costs)."),
 "doc16_qa0": (0.5, "Pred 9.99x vs gold 9.5x -- right ballpark, method/value differ."),
 "doc17_qa0": (0.0, "Gold ROA -0.02; pred -1.42 -- wrong magnitude."),
 "doc18_qa0": (0.0, "Gold DPO 93.86; pred 30.73 -- wrong."),
 "doc19_qa0": (0.75, "Pred 30.7% vs gold 30.8% -- rounding."),
 "doc20_qa0": (1.0, "Matches gold ($11,588M)."),
 "doc21_qa0": (1.0, "Pred $1,615.9M = gold $1616M."),
 "doc22_qa0": (1.0, "Matches gold (supplemental indentures, Amcor Flexibles NA substituted as issuer)."),
 "doc23_qa0": (0.25, "Gives current assets/liabilities but no quick ratio and no improved/declined verdict; gold improved 0.67->0.69."),
 "doc24_qa0": (0.0, "Gold lists FY2023 acquisitions; pred says no info -- abstains."),
 "doc25_qa0": (1.0, "Matches gold (packaging industry)."),
 "doc26_qa0": (0.75, "Correct direction (declining gross margin); omits gold's 0.8% figure."),
 "doc27_qa0": (0.5, "Captures employee component but misses gold 87% concentration; adds other categories."),
 "doc28_qa0": (1.0, "Matches gold ($2,018M adj. EBITDA)."),
 "doc29_qa0": (0.0, "Gold real growth flat; pred -5% -- wrong."),
 "doc30_qa0": (1.0, "Pred 4.18% = gold 4.2%."),
 "doc31_qa0": (0.0, "Gold quick ratio 1.57 (Yes); pred says not provided -- abstains."),
 "doc32_qa0": (1.0, "Matches gold product list (CPUs/GPUs/DPUs/FPGAs/Adaptive SoC/APUs...)."),
 "doc33_qa0": (1.0, "Matches gold revenue drivers (EPYC, semi-custom, Xilinx embedded)."),
 "doc34_qa0": (1.0, "Matches gold (amortization of Xilinx intangibles)."),
 "doc35_qa0": (1.0, "Matches gold (operations brought in the most cash)."),
 "doc36_qa0": (1.0, "Names gold segment (Data Center)."),
 "doc37_qa0": (1.0, "Matches gold (one customer = 16%)."),
 "doc38_qa0": (0.0, "Gold none; pred lists common shares (not debt securities) -- wrong."),
 "doc39_qa0": (1.0, "Matches gold geographies (US, EMEA, APAC, LACC)."),
 "doc40_qa0": (1.0, "Matches gold (operating margin not the right metric for AMEX)."),
 "doc41_qa0": (1.0, "Matches gold (gross margin not the right metric for AMEX)."),
 "doc42_qa0": (1.0, "Matches gold (24.6% -> 21.6%)."),
 "doc43_qa0": (0.0, "Gold largest liability = customer deposits; pred says long-term debt -- wrong."),
 "doc44_qa0": (1.0, "Matches gold (Yes, retention high)."),
 "doc45_qa0": (0.75, "Pred 0.389bn vs gold $0.40bn -- rounding."),
 "doc46_qa0": (0.75, "Pred 1,829 vs gold 1,832 -- minor diff."),
 "doc47_qa0": (0.5, "Pred's math + conclusion = -$1,561M (negative, matches gold) but opens with a contradictory 'Yes positive'."),
 "doc48_qa0": (0.0, "Gold 2.8%; pred 3.5% -- wrong."),
 "doc49_qa0": (1.0, "Matches gold ($5,409M inventories)."),
 "doc50_qa0": (0.0, "Gold Yes (consistent, 1.1% decline); pred says fluctuated >2% -- opposite."),
 "doc51_qa0": (1.0, "Matches gold (FY2022: Current Health + Yardbird; none FY2023/2021)."),
 "doc52_qa0": (1.0, "Matches gold (operating, ~$1.8bn)."),
 "doc53_qa0": (0.75, "Correct (drop, $1,874M->$1,093M ~ gold's 42%); doesn't state 42%."),
 "doc54_qa0": (0.5, "Correct direction (decline) but numbers off (966/977 vs gold 969/982)."),
 "doc55_qa0": (0.75, "Gaming (9%) is the gold driver; gold frames it as the Entertainment segment."),
 "doc56_qa0": (0.75, "Pred 1.74 vs gold 1.73 -- rounding."),
 "doc57_qa0": (0.75, "Pred 102.0% vs gold 101.5% -- close."),
 "doc58_qa0": (1.0, "Pred $381.6M = gold $382M."),
 "doc59_qa0": (1.0, "Matches gold ($12,645M)."),
 "doc60_qa0": (0.5, "Correct Yes + Commercial Airplanes; misses gold's Defense/Services categories."),
 "doc61_qa0": (1.0, "Matches gold (Lion Air 2018 + Ethiopian 2019 crash litigation)."),
 "doc62_qa0": (0.0, "Gold Yes improving (4.8->5.3%); pred dodges saying gross margin not useful -- wrong."),
 "doc63_qa0": (0.5, "Captures commercial airlines; misses gold's US government (40%)."),
 "doc64_qa0": (1.0, "Matches gold (Yes, cyclical)."),
 "doc65_qa0": (0.75, "Captures 737/787/777X production plans (gold's three aircraft)."),
 "doc66_qa0": (0.0, "Gold FY2022 0.62% vs FY2021 -14.76% (higher); pred says lower -- wrong direction, no values."),
 "doc67_qa0": (0.0, "Gold ROA 0.01 (~1%); pred 1.46% -- wrong."),
 "doc68_qa0": (1.0, "Matches gold (39.7% COGS margin)."),
 "doc69_qa0": (1.0, "Pred 0.80 = gold 0.8."),
 "doc70_qa0": (0.0, "Gold DPO 63.86; pred 66.67 -- wrong."),
 "doc71_qa0": (0.75, "Pred 10.5% vs gold 10.3% -- close."),
 "doc72_qa0": (1.0, "Matches gold (20% -> 23%)."),
 "doc73_qa0": (0.75, "Correct Yes (positive WC); magnitude differs (total vs gold's operating-only $831M)."),
 "doc74_qa0": (1.0, "Matches gold ($59,268M total assets)."),
 "doc75_qa0": (0.0, "Gold 17.98; pred 8.73 -- wrong."),
 "doc76_qa0": (0.5, "Correct Yes but reasoning (PP&E) partly contradicts gold (asset base is mostly goodwill)."),
 "doc77_qa0": (0.75, "Correct Yes + captures usual-and-customary pricing litigation (a gold dispute area)."),
 "doc78_qa0": (1.0, "Matches gold ($0.55/share quarterly, Q2 FY2022)."),
 "doc79_qa0": (1.0, "Matches gold (Mary Dillon, ex-Ulta CEO)."),
 "doc80_qa0": (1.0, "Matches gold (Richard A. Johnson)."),
 "doc81_qa0": (0.0, "Gold CCC -3.7; pred 66.73 -- wrong."),
 "doc82_qa0": (0.75, "Pred 0.69 vs gold 0.68 -- rounding."),
 "doc83_qa0": (0.0, "Gold FCF $3,215M; pred $3,115.4M -- off by ~100M."),
 "doc84_qa0": (1.0, "Matches gold (0.54)."),
 "doc85_qa0": (1.0, "Matches gold (No, 1.3% sales growth)."),
 "doc86_qa0": (0.0, "Gold gives drivers (COVID exit, currency, commodity inflation); pred dodges saying not useful -- wrong."),
 "doc87_qa0": (0.25, "Headline 7.6x is wrong; pred's own inputs compute to 2.72x = gold 2.7."),
 "doc88_qa0": (0.0, "Gold No (decelerate 3.6->3.5); pred Yes accelerate 12.5% -- opposite."),
 "doc89_qa0": (1.0, "Matches gold (US +3.0%, intl -0.6%)."),
 "doc90_qa0": (1.0, "Matches gold (Consumer Health discontinued from 30 Aug 2023)."),
 "doc91_qa0": (1.0, "Matches gold (~$20bn gain)."),
 "doc92_qa0": (1.0, "Matches gold ($13.2bn Kenvue cash proceeds)."),
 "doc93_qa0": (1.0, "Matches gold (Yes, 20.0% -> 20.1%)."),
 "doc94_qa0": (0.0, "Gold Corporate (-$473M); pred Consumer & Community Banking -- wrong."),
 "doc95_qa0": (0.0, "Gold $66.56/share; pred $239.45 -- wrong."),
 "doc96_qa0": (1.0, "Matches gold (gross margin not relevant for a financial institution)."),
 "doc97_qa0": (0.0, "Gold Corporate & Investment Bank ($3,725M); pred Consumer & Community Banking -- wrong."),
 "doc98_qa0": (1.0, "Matches gold (Yes, VaR decreased)."),
 "doc99_qa0": (0.0, "Gold 6.25; pred 3.12 -- wrong."),
 "doc100_qa0": (0.75, "Pred 1.36 vs gold 1.33 -- close."),
 "doc101_qa0": (1.0, "Matches gold ($5,818M net working capital)."),
 "doc102_qa0": (1.0, "Matches gold (0.4% CAGR)."),
 "doc103_qa0": (1.0, "Pred $302.578M = gold $303M."),
 "doc104_qa0": (0.0, "Gold 7.9%; pred -3.5% -- wrong."),
 "doc105_qa0": (1.0, "Matches gold ($0.01/share dividend FY2022)."),
 "doc106_qa0": (1.0, "Matches gold (Las Vegas highest EBITDAR)."),
 "doc107_qa0": (0.0, "Gold zero (negative adj. EBIT); pred 1.61 -- wrong."),
 "doc108_qa0": (1.0, "Names gold region (MGM China, worst topline)."),
 "doc109_qa0": (1.0, "Matches gold (corporate bonds largest short-term investment)."),
 "doc110_qa0": (1.0, "Matches gold ($32,780M COGS)."),
 "doc111_qa0": (0.5, "Explanation says debt decreased (right) but opens with a contradictory 'Yes'; magnitude/scope differ from gold's $2.5bn."),
 "doc112_qa0": (0.0, "Gold 5.4%; pred 4.5% -- wrong."),
 "doc113_qa0": (1.0, "Pred 5,466.3M = gold $5,466M."),
 "doc114_qa0": (0.5, "Pred 56.3% vs gold 55.1% -- off ~1.2pp."),
 "doc115_qa0": (1.0, "Matches gold ($16,525M total current assets)."),
 "doc116_qa0": (0.75, "Pred 3.57 vs gold 3.46 -- close (averaging convention)."),
 "doc117_qa0": (1.0, "Matches gold (operations highest cash flow)."),
 "doc118_qa0": (0.75, "Correct Yes (positive WC); magnitude differs (total vs gold's $1.6bn operating)."),
 "doc119_qa0": (1.0, "Pred $4.625bn = gold $4.60bn."),
 "doc120_qa0": (0.25, "Gives a developed/emerging cut, not the gold geographic-segment list."),
 "doc121_qa0": (0.75, "Conveys no material legal battles (= gold No)."),
 "doc122_qa0": (1.0, "Matches gold ($411M restructuring)."),
 "doc123_qa0": (0.0, "Gold EBITDA-less-capex $9,068M; pred $14,275M (EBITDA, not less capex) -- wrong."),
 "doc124_qa0": (1.0, "Matches gold (16.5% EBITDA margin)."),
 "doc125_qa0": (1.0, "Matches gold (net-zero proposal defeated)."),
 "doc126_qa0": (1.0, "Matches gold ($400M increase)."),
 "doc127_qa0": (1.0, "Matches gold ($8.4bn total borrowing capacity)."),
 "doc128_qa0": (1.0, "Matches gold (strong start to FY2023)."),
 "doc129_qa0": (1.0, "Matches gold (1 percentage point, 8->9%)."),
 "doc130_qa0": (1.0, "Matches gold (Yes, PP&E grew)."),
 "doc131_qa0": (0.75, "Correct event (Consumer Healthcare JV gain); the year/amount detail is muddled."),
 "doc132_qa0": (0.5, "Trillium + Array correct; Upjohn wrong (it was spun off; gold = Therachon)."),
 "doc133_qa0": (0.0, "Gold $77.78M; pred $700M -- wrong."),
 "doc134_qa0": (1.0, "Matches gold (Developed Rest of World)."),
 "doc135_qa0": (1.0, "Matches gold (Yes, spinning off Upjohn)."),
 "doc136_qa0": (1.0, "Matches gold (none)."),
 "doc137_qa0": (0.75, "Correct (no acquisitions), slightly hedged ('not mentioned')."),
 "doc138_qa0": (1.0, "Matches gold (lower marketing + incentive-comp leverage)."),
 "doc139_qa0": (1.0, "Captures gold driver (47 new stores) plus detail."),
 "doc140_qa0": (0.75, "Pred 36.5% vs gold 36% -- close."),
 "doc141_qa0": (0.0, "Gold increased; pred Decrease -- opposite."),
 "doc142_qa0": (1.0, "Names gold instrument (cross currency swaps)."),
 "doc143_qa0": (0.5, "Captures pension $1,097M; misses gold's health/life $862M."),
 "doc144_qa0": (0.0, "Gold quick ratio 0.54 (No); pred says not provided -- abstains."),
 "doc145_qa0": (0.75, "Correct Yes (capital intensive); no reasoning vs gold's 2.77 ratio."),
 "doc146_qa0": (0.75, "Explanation + exact -$229M match gold (decreased); opens with a contradictory 'Yes'."),
 "doc147_qa0": (0.0, "Gold DPO 42.69; pred 36.73 -- wrong."),
 "doc148_qa0": (0.0, "Gold +0.2%; pred -6.0% -- wrong."),
 "doc149_qa0": (0.0, "Gold 6.2%; pred 3.9% -- wrong."),
}


def main() -> int:
    queue = [json.loads(l) for l in (CELL / "queue.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out, missing = [], []
    for q in queue:
        sfx = q["qid"].split("__batch__")[1].replace("__seed7", "")
        if sfx not in J:
            missing.append(sfx); continue
        score, rat = J[sfx]
        out.append({"qid": q["qid"], "judge_score": score, "rationale": rat,
                    "judge_model": "claude-opus-4.7-1m", "judge_protocol": "v1"})
    if missing:
        raise SystemExit(f"missing {len(missing)}: {missing[:10]}")
    (CELL / "results.jsonl").write_text("\n".join(json.dumps(o) for o in out) + "\n", encoding="utf-8")
    mean = sum(o["judge_score"] for o in out) / len(out)
    print(f"wrote {len(out)} judgments; mean judge = {mean:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
