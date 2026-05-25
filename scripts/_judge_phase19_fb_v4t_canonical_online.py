"""
Manual Claude judging — FinanceBench v4t-canonical ONLINE cell.

Canonical θ (grid-world tuned: w_recency=3.78, theta_store=0.29) collapses
on corpus scale: recall@k=0.567 — about half the questions don't even
retrieve their own (just-ingested) gold doc. The judge mean should be
~0.45-0.50, vastly below corpus-tuned's 0.68.

Per evaluation/claude_judge_protocol.md, manually scored 1-by-1 fresh
against the regenerated predictions (prior session predictions differed
due to OpenAI approximate-determinism, ~50/150 textually).
"""
from __future__ import annotations
import json
from pathlib import Path

JUDGE_DIR = Path("results/stage3/judge_queue/financebench__v4t-canonical__online__seed42")
QID_PREFIX = "financebench__v4t-canonical__online__"
QID_SUFFIX = "__seed42"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc0_qa0",   1.0,  "Pred $1,577M = gold $1577M (3M FY18 capex) exact."),
    ("doc1_qa0",   0.0,  "Pred 1.577B vs gold $8.70B (3M FY18 PPNE) — pred appears to confuse with capex; 82% off."),
    ("doc2_qa0",   0.0,  "Y/N flip: pred Yes vs gold No (3M capital-intensive FY22)."),
    ("doc3_qa0",   1.0,  "Pred names SG&A+litigation+PFAS+Russia+restructuring drivers — matches gold's litigation/impairment/Russia/PFAS."),
    ("doc4_qa0",   1.0,  "Pred Consumer = gold Consumer segment dragged growth."),
    ("doc5_qa0",   0.0,  "Pred refuses; gold definitive No (3M quick ratio 0.96)."),
    ("doc6_qa0",   1.0,  "Pred MMM26/30/31 = gold MMM26/30/31 (3M debt securities)."),
    ("doc7_qa0",   1.0,  "Pred Yes 65 consecutive years = gold Yes 65 years (3M dividends)."),
    ("doc8_qa0",   0.0,  "Pred 2.66 vs gold 24.26 (Activision fixed asset turnover) — pred is 9× too low."),
    ("doc9_qa0",   0.0,  "Pred 6.0% vs gold 1.9% (Activision capex/rev) — 3× too high."),
    ("doc10_qa0",  0.0,  "Pred 0.40 vs gold 0.66 (Adobe FY15 OCF ratio) — 39% diff."),
    ("doc11_qa0",  1.0,  "Pred 65.4% = gold 65.4% exactly (Adobe OI change)."),
    ("doc12_qa0",  0.0,  "Pred refuses; gold definitive 0.83 (Adobe FY17 OCF ratio)."),
    ("doc13_qa0",  0.0,  "Pred refuses; gold definitive No (Adobe OM declined 36.8 → 34.6)."),
    ("doc14_qa0",  0.0,  "Pred refuses; gold definitive Yes (Adobe FCF conversion improved 143 → 156)."),
    ("doc15_qa0",  1.0,  "Pred 0 = gold 0 exact (AES restructuring)."),
    ("doc16_qa0",  0.0,  "Pred refuses; gold definitive 9.5x (AES inventory turnover)."),
    ("doc17_qa0",  0.0,  "Pred -1.16 vs gold -0.02 (AES ROA) — 58× too large in magnitude."),
    ("doc18_qa0",  0.25, "Pred truncated mid-calculation (started COGS/AP setup) but no final answer; gold 93.86."),
    ("doc19_qa0",  1.0,  "Pred 29.5% vs gold 30.8% (Amazon revenue growth) — 4.2% diff, within 5%."),
    ("doc20_qa0",  0.0,  "Pred refuses; gold definitive $11,588M (Amazon FY19 net income)."),
    ("doc21_qa0",  0.0,  "Pred refuses; gold definitive $1,616M (Amcor FY20 net AR)."),
    ("doc22_qa0",  0.75, "Pred matches substance (supplemental indentures + AFNA substitution) but says June 30, 2022 (gold says July 1, 2022)."),
    ("doc23_qa0",  0.0,  "Pred refuses; gold definitive improvement 0.67 → 0.69 (Amcor quick ratio)."),
    ("doc24_qa0",  0.75, "Pred names Shanghai + Czech (matches gold's FY23 acquisitions) but assigns wrong fiscal years (Shanghai → FY23 not FY22 per gold) and misses NZ acquisition."),
    ("doc25_qa0",  1.0,  "Pred packaging = gold packaging."),
    ("doc26_qa0",  0.0,  "Pred says GM not useful; gold definitive No (slight 0.8% decline)."),
    ("doc27_qa0",  0.5,  "Pred mentions employee costs ($93M total) but does not give 87%-employee breakdown that gold highlights."),
    ("doc28_qa0",  1.0,  "Pred 2,018M = gold $2,018M exact (Amcor adj EBITDA FY23)."),
    ("doc29_qa0",  0.0,  "Pred -5% vs gold flat (Amcor real sales change) — wrong sign and magnitude."),
    ("doc30_qa0",  0.0,  "Pred refuses; gold definitive 4.2% (AMD FY15 D&A margin)."),
    ("doc31_qa0",  0.0,  "Pred refuses; gold definitive Yes 1.57 (AMD quick ratio FY22)."),
    ("doc32_qa0",  1.0,  "Pred lists matching AMD product portfolio."),
    ("doc33_qa0",  1.0,  "Pred names EPYC + semi-custom + Xilinx — matches gold."),
    ("doc34_qa0",  1.0,  "Pred Xilinx amortization = gold Xilinx amortization (AMD op margin)."),
    ("doc35_qa0",  1.0,  "Pred operations = gold operations (AMD FY22 cashflow source)."),
    ("doc36_qa0",  1.0,  "Pred Data Center = gold Data Center (AMD biggest growth segment)."),
    ("doc37_qa0",  1.0,  "Pred Yes 16% concentration = gold Yes 16%."),
    ("doc38_qa0",  0.0,  "Pred Common Shares (equity) vs gold no debt securities registered (AmEx)."),
    ("doc39_qa0",  1.0,  "Pred US/EMEA/APAC/LACC (+ Other Unallocated) = gold US/EMEA/APAC/LACC."),
    ("doc40_qa0",  1.0,  "Pred OM not useful = gold performance-not-measured-through-OM."),
    ("doc41_qa0",  1.0,  "Pred GM not useful = gold performance-not-measured-through-GM."),
    ("doc42_qa0",  0.0,  "Pred 6.5% → 8.0% vs gold 24.6% → 21.6% (AmEx ETR) — entirely wrong numbers."),
    ("doc43_qa0",  1.0,  "Pred Customer deposits = gold Customer deposits (AmEx largest liability)."),
    ("doc44_qa0",  1.0,  "Pred Yes retention high = gold Yes (AmEx 2022 card retention)."),
    ("doc45_qa0",  0.0,  "Pred refuses; gold definitive $0.40B (AWK FY20 cash dividends)."),
    ("doc46_qa0",  1.0,  "Pred 1,832 = gold $1832M exact (AWK FY21 EBITDA)."),
    ("doc47_qa0",  0.0,  "Pred refuses; gold definitive No -$1561M (AWK FY22 working capital)."),
    ("doc48_qa0",  0.0,  "Pred 3.1% vs gold 2.8% (Best Buy FY15-17 net margin) — 10.7% diff."),
    ("doc49_qa0",  0.0,  "Pred refuses; gold definitive $5,409M (Best Buy FY19 inventories)."),
    ("doc50_qa0",  0.0,  "Y/N flip: pred 'not consistent' vs gold 'consistent' (Best Buy gross margins)."),
    ("doc51_qa0",  1.0,  "Pred Current Health + Yardbird FY22, none FY23/21 — matches gold."),
    ("doc52_qa0",  0.0,  "Pred refuses; gold definitive operations $1.8bn (Best Buy FY23 cashflow)."),
    ("doc53_qa0",  0.0,  "Pred refuses; gold definitive Yes ~42% drop (Best Buy cash FY23 → Q2FY24)."),
    ("doc54_qa0",  0.5,  "Pred 930 → 907 vs gold 982 → 969 (Best Buy store counts) — direction right, specific counts off."),
    ("doc55_qa0",  0.75, "Pred Gaming matches sub-driver but gold names Entertainment segment +9% (Gaming is the Entertainment driver)."),
    ("doc56_qa0",  1.0,  "Pred 1.74 vs gold 1.73 (Block FY16 WC ratio) — 0.6% diff."),
    ("doc57_qa0",  1.0,  "Pred 101.7% vs gold 101.5% (Block FY19-20 revenue growth) — 0.2% diff."),
    ("doc58_qa0",  0.0,  "Pred refuses; gold definitive $382M (Block FY20 OCF)."),
    ("doc59_qa0",  0.0,  "Pred refuses; gold definitive $12,645M (Boeing FY18 net PPE)."),
    ("doc60_qa0",  0.5,  "Pred only Commercial Airplanes; gold says Commercial AND Defense both >20%."),
    ("doc61_qa0",  1.0,  "Pred Lion Air 2018 + Ethiopian 2019 = gold (Boeing legal battles)."),
    ("doc62_qa0",  0.0,  "Pred says GM not useful; gold definitive Yes improving 4.8 → 5.3 (Boeing FY22 gross margin)."),
    ("doc63_qa0",  1.0,  "Pred US gov + airlines = gold US gov + airlines (Boeing primary customers)."),
    ("doc64_qa0",  1.0,  "Pred Yes cyclical = gold Yes (Boeing cyclicality)."),
    ("doc65_qa0",  1.0,  "Pred 787 increase + 737 increase + 777X resume — matches gold."),
    ("doc66_qa0",  0.0,  "Pred refuses; gold definitive 0.62% vs -14.76% (Boeing ETR)."),
    ("doc67_qa0",  0.0,  "Pred 3.06% vs gold 0.01 (Coca-Cola FY17 ROA) — 306× too high."),
    ("doc68_qa0",  0.75, "Pred 37.5% vs gold 39.7% (Coca-Cola COGS margin) — 5.5% diff, just outside tolerance."),
    ("doc69_qa0",  0.0,  "Pred refuses; gold definitive 0.8 (Coca-Cola FY22 dividend payout)."),
    ("doc70_qa0",  0.0,  "Pred refuses; gold definitive 63.86 (Corning FY20 DPO)."),
    ("doc71_qa0",  0.0,  "Pred refuses; gold definitive 10.3% (Corning FY19-21 op income margin)."),
    ("doc72_qa0",  1.0,  "Pred 20% → 23% = gold 20% → 23% exact (Corning ETR change)."),
    ("doc73_qa0",  0.0,  "Pred refuses; gold definitive Yes $831M (Corning FY22 WC)."),
    ("doc74_qa0",  1.0,  "Pred 59,268 = gold $59,268M exact (Costco FY21 total assets)."),
    ("doc75_qa0",  0.0,  "Pred refuses; gold definitive 17.98 (CVS FY18 fixed asset turnover)."),
    ("doc76_qa0",  1.0,  "Pred Yes capital intensive = gold Yes (CVS FY22)."),
    ("doc77_qa0",  1.0,  "Pred opioid + drug pricing + U&C litigation — matches gold's multiple dispute areas."),
    ("doc78_qa0",  1.0,  "Pred Yes $0.55 quarterly = gold Yes $0.55/share (CVS Q2FY22 dividends)."),
    ("doc79_qa0",  1.0,  "Pred Mary Dillon ex-Ulta CEO = gold Yes she was Ulta CEO."),
    ("doc80_qa0",  1.0,  "Pred Richard A. Johnson = gold Richard A. Johnson."),
    ("doc81_qa0",  0.0,  "Pred 36.73 vs gold -3.7 (General Mills CCC FY19) — wrong sign."),
    ("doc82_qa0",  0.0,  "Pred refuses; gold definitive 0.68 (General Mills FY20 WC ratio)."),
    ("doc83_qa0",  0.0,  "Pred refuses; gold definitive $3,215M (General Mills FY20 FCF)."),
    ("doc84_qa0",  0.0,  "Pred 0.11 vs gold 0.54 (General Mills retention rate) — 80% off."),
    ("doc85_qa0",  1.0,  "Pred No (1.3% growth, vs 13.6% prior) = gold No 1.3% (JnJ FY22)."),
    ("doc86_qa0",  1.0,  "Pred names COVID exit + currency + commodity inflation drivers — matches gold."),
    ("doc87_qa0",  0.0,  "Pred refuses (truncated mid-calc, no answer); gold definitive 2.7x."),
    ("doc88_qa0",  0.25, "Y/N flip: pred Yes accelerate but cites 3.5% which matches gold's deceleration figure. Wrong Y/N answer."),
    ("doc89_qa0",  0.25, "Pred US +6.9% (gold says +3.0%) and intl 'negatively impacted by currency' (vague); gold US +3.0% intl -0.6%."),
    ("doc90_qa0",  1.0,  "Pred Consumer Health = gold Consumer Health (JnJ discontinued op)."),
    ("doc91_qa0",  0.0,  "Pred refuses; gold definitive ~$20B (JnJ Consumer Health separation gain)."),
    ("doc92_qa0",  0.0,  "Pred refuses; gold definitive $13.2B (JnJ Kenvue proceeds)."),
    ("doc93_qa0",  1.0,  "Pred Yes 20.0 → 20.1 = gold exact (JnJ Q2FY23 net earnings/sales)."),
    ("doc94_qa0",  0.0,  "Pred CCB vs gold Corporate (JPM lowest net revenue 2021Q1)."),
    ("doc95_qa0",  1.0,  "Pred $66.56 = gold $66.56 exact (JPM bankruptcy per-share)."),
    ("doc96_qa0",  1.0,  "Pred GM not relevant for financial firm = gold."),
    ("doc97_qa0",  1.0,  "Pred Corporate & Investment Bank = gold CIB (JPM highest net income 2022Q2)."),
    ("doc98_qa0",  1.0,  "Pred Yes VaR decreased = gold Yes decreased."),
    ("doc99_qa0",  0.0,  "Pred refuses; gold definitive 6.25 (Kraft Heinz FY19 inventory turnover)."),
    ("doc100_qa0", 0.0,  "Pred refuses; gold definitive 1.33."),
    ("doc101_qa0", 1.0,  "Pred $5,818M = gold $5818M exact (Lockheed FY21 NWC)."),
    ("doc102_qa0", 0.0,  "Pred refuses; gold definitive 0.4% (Lockheed FY20-22 CAGR)."),
    ("doc103_qa0", 0.0,  "Pred refuses; gold definitive $303M (MGM FY18 AP)."),
    ("doc104_qa0", 0.0,  "Pred 10.5% vs gold 7.9% (MGM FY18-20 capex/rev) — 33% diff."),
    ("doc105_qa0", 1.0,  "Pred $0.01/share annual = gold (MGM FY22 dividends)."),
    ("doc106_qa0", 0.0,  "Pred refuses; gold definitive Las Vegas ~90% (MGM EBITDAR)."),
    ("doc107_qa0", 0.0,  "Pred 1.61 vs gold 0 (MGM interest coverage; gold notes EBIT negative → ratio zero)."),
    ("doc108_qa0", 1.0,  "Pred MGM China -44% $674M = gold MGM China -44% (worst topline)."),
    ("doc109_qa0", 1.0,  "Pred corporate bonds = gold corporate bonds (MGM ST investment)."),
    ("doc110_qa0", 0.0,  "Pred refuses; gold definitive $32,780M (Microsoft FY16 COGS)."),
    ("doc111_qa0", 0.25, "Pred 'Yes increased' (wrong Y/N answer) but body details show long-term debt decreased $47,032M → $41,990M (matches gold's decrease). Self-contradictory."),
    ("doc112_qa0", 0.25, "Pred truncated mid-calc with $305,826 OI (no final answer); gold 5.4%."),
    ("doc113_qa0", 0.0,  "Pred refuses; gold definitive $5,466M (Netflix FY17 CL)."),
    ("doc114_qa0", 1.0,  "Pred 56.3% vs gold 55.1% — 2.2% diff, within 5%."),
    ("doc115_qa0", 0.0,  "Pred refuses; gold definitive $16,525M (Nike FY19 CA)."),
    ("doc116_qa0", 0.0,  "Pred refuses; gold definitive 3.46 (Nike FY21 inventory turnover)."),
    ("doc117_qa0", 0.0,  "Pred refuses; gold definitive operations (Nike FY23 cashflow source)."),
    ("doc118_qa0", 0.0,  "Pred refuses; gold definitive Yes $1.6B (Paypal FY22 WC)."),
    ("doc119_qa0", 1.0,  "Pred 4.625B vs gold $4.60B (PepsiCo FY21 capex) — 0.5% diff."),
    ("doc120_qa0", 1.0,  "Pred lists same 10 regions as gold (Pepsico geographies)."),
    ("doc121_qa0", 0.75, "Pred describes ongoing litigation but says no material adverse effect — matches gold's 'No material legal battles' conclusion."),
    ("doc122_qa0", 1.0,  "Pred 411 = gold $411M (Pepsico FY22 restructuring)."),
    ("doc123_qa0", 0.0,  "Pred $14,389M vs gold $9068M (PepsiCo EBITDA less capex) — 58% too high; pred returns gross EBITDA."),
    ("doc124_qa0", 0.0,  "Pred mid-calc with $10,389M EBITDA gives ~12.5% margin (truncated); gold 16.5%."),
    ("doc125_qa0", 1.0,  "Pred defeated = gold defeated (Pepsico net-zero vote)."),
    ("doc126_qa0", 1.0,  "Pred $400M = gold $400M (Pepsico credit increase)."),
    ("doc127_qa0", 0.0,  "Pred $4,950M vs gold $8.4B (Pepsico total credit) — 41% off."),
    ("doc128_qa0", 1.0,  "Pred strong start + momentum = gold strong start."),
    ("doc129_qa0", 1.0,  "Pred 1pp (8 → 9) = gold 1pp exact."),
    ("doc130_qa0", 1.0,  "Pred Yes $13,745M → $14,882M (correct PPNE figures) — matches gold's Yes positive."),
    ("doc131_qa0", 1.0,  "Pred Consumer Healthcare JV gain = gold Consumer Healthcare JV (2019 net income event)."),
    ("doc132_qa0", 1.0,  "Pred Therachon/Trillium/Array = gold Trillium/Array/Therachon (same 3 names)."),
    ("doc133_qa0", 0.0,  "Pred $700M vs gold $77.78M (Pfizer Upjohn payment) — 9× too high."),
    ("doc134_qa0", 1.0,  "Pred Developed Rest of World = gold Developed Rest of the World."),
    ("doc135_qa0", 1.0,  "Pred Yes Upjohn = gold Yes Upjohn."),
    ("doc136_qa0", 0.0,  "Pred common stock NASDAQ (equity) vs gold no debt securities (Ulta)."),
    ("doc137_qa0", 0.75, "Pred refuses 'not in passages'; gold says 'none'. Aligns with truth but lacks confident statement."),
    ("doc138_qa0", 1.0,  "Pred lower marketing + incentive comp leverage = gold."),
    ("doc139_qa0", 0.25, "Pred cites 'change in operating assets and liabilities, decrease of $104,233' (wrong direction in cited line — gold says increase driven by 47 new stores). Pred misses the store-opening cause."),
    ("doc140_qa0", 1.0,  "Pred 36.5% vs gold 36% — 1.4% diff."),
    ("doc141_qa0", 0.0,  "Y/N flip: pred Decrease vs gold increased (Ulta wages %)."),
    ("doc142_qa0", 1.0,  "Pred cross currency swaps = gold (Verizon top derivative)."),
    ("doc143_qa0", 0.75, "Pred $1,097M pension only; gold pension $1097M + health/life $862M."),
    ("doc144_qa0", 0.0,  "Pred refuses; gold definitive No 0.54 (Verizon quick ratio)."),
    ("doc145_qa0", 1.0,  "Pred Yes capital intensive = gold Yes (Verizon FY22)."),
    ("doc146_qa0", 0.25, "Pred 'Yes increased' (wrong) cites 'debt maturing within one year' increased $7,443M → $9,963M; gold says No (total debt decreased). Pred cherry-picks a subset that did increase."),
    ("doc147_qa0", 0.0,  "Pred 36.36 vs gold 42.69 (Walmart FY18 DPO) — 14.8% diff."),
    ("doc148_qa0", 0.0,  "Pred refuses (says FY19 OI not provided); gold definitive 0.2% (Walmart op income margin change)."),
    ("doc149_qa0", 0.0,  "Pred 4.0% vs gold 6.2% (Walmart FY18-20 EBITDA margin) — 35% diff."),
]


def main() -> None:
    results_path = JUDGE_DIR / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict] = {}
    if results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                existing[e["qid"]] = e

    new_records = []
    skipped = 0
    for suffix, score, rationale in JUDGMENTS:
        qid = QID_PREFIX + suffix + QID_SUFFIX
        if qid in existing:
            skipped += 1
            continue
        new_records.append({
            "qid": qid,
            "judge_score": score,
            "rationale": rationale,
            "judge_model": "claude-opus-4.7-1m",
            "judge_protocol": "v1",
        })

    if new_records:
        with results_path.open("a", encoding="utf-8") as f:
            for rec in new_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with results_path.open(encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    scores = [e["judge_score"] for e in lines]
    from collections import Counter
    dist = Counter(scores)
    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"Appended {len(new_records)} (skipped {skipped}, total {len(lines)})")
    print(f"Score distribution: {dict(sorted(dist.items()))}")
    print(f"Mean judge: {mean:.4f}")


if __name__ == "__main__":
    main()
