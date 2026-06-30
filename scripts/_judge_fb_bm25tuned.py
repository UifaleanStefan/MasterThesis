"""Apply Claude 1-by-1 judgments for financebench__bm25-corpus-tuned__batch__seed42.

Hand-judged by Claude against the FinanceBench 5-point rubric: 1.0 exact/correct;
0.75 substantially correct (rounding / minor omission); 0.5 partial; 0.25 wrong
value but right ballpark/category; 0.0 wrong, or a refusal/abstention when the
gold answer exists (common here -- BM25 retrieval often missed the needed
financial statement). NO heuristics. Each doc has one question (doc{i}_qa0).
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELL = ROOT / "results" / "stage3" / "judge_queue" / "financebench__bm25-corpus-tuned__batch__seed42"

# index i -> (score, rationale) ; qid suffix is doc{i}_qa0
S = [
 (1.0, "Gold $1577; pred $1,577M -- match."),
 (0.0, "Net PPNE gold $8.70B; pred $1.577B (capex confusion) -- wrong."),
 (0.0, "Capital-intensive: gold No; pred Yes -- opposite."),
 (0.0, "Op-margin drivers exist (gold); pred refuses 'no info'."),
 (0.0, "Gold consumer segment; pred says litigation -- wrong."),
 (0.0, "Quick ratio 0.96 exists; pred refuses."),
 (1.0, "Lists the three registered notes (2026/2030/2031) -- match."),
 (1.0, "65 consecutive years of dividend increases -- match."),
 (0.0, "Fixed-asset turnover gold 24.26; pred 8.06 -- wrong."),
 (0.0, "3yr capex% gold 1.9%; pred 3.0% -- wrong."),
 (0.0, "Op cash flow ratio 0.66 exists; pred refuses."),
 (1.0, "YoY op income +65.4% computed correctly."),
 (0.0, "Op cash flow ratio 0.83 exists; pred refuses."),
 (0.0, "Adobe op margin declined (gold); pred refuses."),
 (0.0, "Adobe FCF conversion improved (gold); pred refuses."),
 (1.0, "Restructuring costs 0 -- match."),
 (0.25, "Inventory turnover gold 9.5; pred computation truncated, no/wrong final value."),
 (0.25, "ROA gold -0.02; pred -1.42% -- ballpark sign right, value off."),
 (0.0, "DPO 93.86 exists; pred refuses."),
 (0.0, "Revenue +30.8% exists; pred refuses."),
 (1.0, "Net income $11,588M -- match."),
 (1.0, "Net AR $1,615.9M ~ gold $1616 -- match."),
 (1.0, "8-K agenda (supplemental indentures, 2026/2028 notes, issuer substitution) -- match."),
 (0.0, "Quick ratio improved 0.67->0.69 (gold); pred refuses."),
 (0.75, "Names all three FY23 acquisitions; mis-years the Czech one."),
 (1.0, "Packaging industry -- match."),
 (0.0, "Gross margin declined 0.8% (gold); pred refuses."),
 (0.5, "Captures employee-cost nature of restructuring liability; misses 87% figure."),
 (0.0, "Adj EBITDA gold $2,018M; pred $2,117M -- wrong."),
 (0.0, "Real sales growth flat (gold); pred -5% -- wrong."),
 (0.0, "D&A margin 4.2% exists; pred refuses."),
 (0.0, "Quick ratio 1.57 exists; pred refuses."),
 (1.0, "AMD product list matches gold."),
 (1.0, "Revenue drivers (EPYC, semi-custom, Xilinx) -- match."),
 (0.0, "Op-margin driver (Xilinx amortization) exists; pred refuses."),
 (1.0, "Operations brought most cash flow -- match."),
 (0.0, "Top segment growth: gold Data Center; pred Gaming -- wrong."),
 (1.0, "Customer concentration: one customer 16% -- match."),
 (0.0, "Debt securities gold 'none'; pred lists common shares -- wrong."),
 (1.0, "Geographies US/EMEA/APAC/LACC -- match."),
 (0.0, "Gold: op margin not the metric; pred refuses instead of saying so."),
 (0.0, "Gold: gross margin not the metric; pred refuses instead of saying so."),
 (1.0, "Effective tax 24.6%->21.6% -- match."),
 (1.0, "Largest liability customer deposits -- match."),
 (1.0, "Card-member retention high -- match."),
 (0.75, "Dividends $0.389B ~ gold $0.40B -- rounds correctly."),
 (1.0, "Unadjusted EBITDA $1,832M -- match."),
 (0.0, "Working capital gold negative -$1561M; pred says positive (its own numbers show negative)."),
 (0.0, "Net profit margin 2.8% exists; pred refuses."),
 (0.0, "Inventories $5409M exist; pred refuses."),
 (0.0, "Gross margins consistent (gold Yes); pred says not consistent -- opposite."),
 (1.0, "Best Buy FY22 acquisitions (Current Health, Yardbird) -- match."),
 (1.0, "Operations most cash flow ($1,824M) -- match."),
 (1.0, "Cash dropped ~42% ($1,874M->$1,093M) -- match."),
 (1.0, "Stores 982->969 -- match."),
 (1.0, "Best category gaming/entertainment +9% -- match."),
 (0.75, "Working-capital ratio gold 1.73; pred 1.74 -- rounding."),
 (0.0, "Revenue growth 101.5% exists; pred refuses."),
 (0.0, "Op cash flow gold $382M; pred $1,000M -- wrong."),
 (0.0, "Net PPE $12,645M exists; pred refuses."),
 (0.0, ">20% revenue categories exist (gold); pred refuses."),
 (1.0, "Legal battles: Lion Air + Ethiopian crashes -- match."),
 (0.0, "Gross margin improved (gold); pred refuses."),
 (0.75, "Customers: airlines + government; misses US-gov 40% detail."),
 (0.75, "Cyclical Yes; omits airline-industry reasoning."),
 (0.75, "Production-rate increases 737/787/777X -- substantially correct."),
 (0.0, "Effective tax 0.62%/-14.76% (gold); pred (31)%/743% -- wrong."),
 (0.25, "ROA gold 0.01; pred 1.46% -- ballpark off."),
 (1.0, "COGS margin 39.7% -- match."),
 (1.0, "Dividend payout 0.80 -- match."),
 (0.0, "DPO 63.86 exists; pred refuses."),
 (0.0, "Op income margin gold 10.3%; pred 13.3% -- wrong."),
 (1.0, "Effective tax 20%->23% -- match."),
 (0.0, "Working capital +$831M (gold); pred refuses."),
 (1.0, "Total assets $59,268M -- match."),
 (0.0, "Fixed-asset turnover 17.98 exists; pred refuses."),
 (0.5, "Capital-intensive Yes (correct) but generic reasoning vs gold metrics."),
 (0.5, "Legal battles Yes; names one of three dispute areas."),
 (0.0, "Dividend $0.55/q exists; pred refuses."),
 (1.0, "New CEO Mary Dillon ex-Ulta CEO -- match."),
 (1.0, "Board nominee Richard A. Johnson, most votes against -- match."),
 (0.0, "CCC -3.7 exists; pred refuses."),
 (1.0, "Working-capital ratio 0.68 -- match."),
 (0.25, "FCF gold $3215M; pred $3,115M -- off ~$100M."),
 (0.25, "Retention ratio gold 0.54; pred 0.49 -- off."),
 (1.0, "JnJ not high-growth (1.3% sales) -- match."),
 (0.0, "Gross-margin drivers exist (gold); pred refuses."),
 (0.0, "Inventory turnover 2.7 exists; pred refuses."),
 (0.0, "Adj EPS decelerates (gold No); pred says accelerate -- opposite."),
 (1.0, "US +3.0% vs intl -0.6% -- match."),
 (1.0, "Consumer Health discontinued from Aug 30 2023 -- match."),
 (1.0, "Separation gain ~$20B -- match."),
 (1.0, "Kenvue cash proceeds $13.2B -- match."),
 (1.0, "Net earnings %sales 20.0%->20.1% -- match."),
 (0.0, "Lowest-revenue segment gold Corporate; pred Commercial Banking -- wrong."),
 (0.0, "Per-share liquidation $66.56 exists; pred refuses."),
 (1.0, "JPM: gross margin not relevant (financial firm) -- match."),
 (0.0, "Highest net income gold Corp & Investment Bank; pred CCB -- wrong."),
 (1.0, "VaR decreased Q2'23 -- match."),
 (0.75, "Inventory turnover gold 6.25; pred 6.20 -- rounding."),
 (0.0, "Asset turnover gold 1.33; pred 0.73 -- wrong."),
 (1.0, "Net working capital $5,818M -- match."),
 (0.0, "Revenue CAGR gold 0.4%; pred 1.4% -- wrong."),
 (1.0, "Accounts payable $302.578M ~ gold $303 -- match."),
 (0.5, "Capex% computation correct per-year but truncated, no final 7.9% stated."),
 (1.0, "MGM dividend $0.01/share -- match."),
 (1.0, "Highest EBITDAR region Las Vegas -- match."),
 (0.0, "Interest coverage gold 0 (negative EBIT); pred 1.61 -- wrong."),
 (1.0, "Worst topline region MGM China -- match."),
 (1.0, "Largest short-term investment corporate bonds -- match."),
 (1.0, "COGS $32,780M -- match."),
 (0.0, "Debt: gold decreased; pred says 'Yes increased' (data shows decrease) -- wrong."),
 (0.0, "EBITDA margin 5.4% exists; pred computes EBITDA but fails the margin."),
 (1.0, "Total current liabilities $5,466.3M ~ gold $5466 -- match."),
 (0.0, "Nike COGS% 55.1% exists; pred refuses."),
 (1.0, "Total current assets $16,525M -- match."),
 (0.0, "Inventory turnover gold 3.46; pred 4.65 -- wrong."),
 (1.0, "Operations most cash flow ($5,841M) -- match."),
 (0.25, "Gold answers Yes (+$1.6B); pred dodges as 'not relevant'."),
 (0.75, "Capex $4.625B ~ gold $4.60B -- rounds correctly."),
 (0.5, "Geographies partial (US/Canada/LatAm/Europe); misses Africa/ME/Asia/etc."),
 (1.0, "No material legal battles -- match."),
 (1.0, "Restructuring costs $411M -- match."),
 (0.0, "EBITDA-less-capex gold $9068M; pred $14,275M -- wrong."),
 (0.0, "EBITDA margin 16.5% exists; pred fails to reach the margin."),
 (1.0, "Net-zero proposal defeated -- match."),
 (1.0, "Credit agreement increase $400M -- match."),
 (1.0, "Total borrow $8.4B -- match."),
 (1.0, "Raised guidance: strong start to FY2023 -- match."),
 (1.0, "EPS guidance +1pp (8%->9%) -- match."),
 (0.5, "PPNE grew Yes (correct) but pred mislabels net-income figures as PPNE."),
 (1.0, "Non-operating gain: Consumer Healthcare JV -- match."),
 (0.5, "Two of three acquisitions correct (Trillium, Array); third hallucinated."),
 (0.0, "Upjohn spin-off cost gold 77.78; pred $700M -- wrong."),
 (1.0, "Biggest revenue drop: Developed Rest of World -- match."),
 (1.0, "Spinning off Upjohn -- match."),
 (1.0, "Ulta debt securities: none -- match."),
 (0.75, "Ulta no acquisitions; pred 'not mentioned' ~ none."),
 (1.0, "SG&A reduction drivers (marketing, incentive comp) -- match."),
 (1.0, "Inventory increase driven by 47 new stores -- match."),
 (0.75, "Stock-repurchase Q4 share gold 36%; pred 36.5%."),
 (0.0, "Wages %sales increased (gold); pred says decrease -- opposite."),
 (1.0, "Highest-notional derivative cross currency swaps -- match."),
 (0.5, "Pension $1,097M correct; misses $862M health/life component."),
 (0.0, "Quick ratio 0.54 exists; pred refuses."),
 (0.75, "Capital-intensive Yes (correct); reasoning via PP&E magnitude."),
 (0.0, "Debt: gold decreased; pred says 'Yes increased' (data shows decrease) -- wrong."),
 (0.0, "Walmart DPO computation used 3M's accounts payable (retrieval error) -- wrong."),
 (0.25, "Op-margin change gold 0.2%; pred 0.1% -- off by rounding."),
 (0.0, "EBITDA margin gold 6.2%; pred 10.5% -- wrong."),
]


def main() -> int:
    queue = [json.loads(l) for l in (CELL / "queue.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_sfx = {f"doc{i}_qa0": S[i] for i in range(len(S))}
    out, missing = [], []
    for q in queue:
        sfx = q["qid"].split("__batch__")[1].replace("__seed42", "")
        if sfx not in by_sfx:
            missing.append(sfx); continue
        score, rat = by_sfx[sfx]
        out.append({"qid": q["qid"], "judge_score": score, "rationale": rat,
                    "judge_model": "claude-opus-4.7-1m", "judge_protocol": "v1"})
    if len(S) != 150 or missing:
        raise SystemExit(f"S has {len(S)} (need 150); missing {missing[:8]}")
    (CELL / "results.jsonl").write_text("\n".join(json.dumps(o) for o in out) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} judgments; mean = {sum(o['judge_score'] for o in out)/len(out):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
