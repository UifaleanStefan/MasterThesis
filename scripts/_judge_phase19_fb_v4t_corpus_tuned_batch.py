"""
Manual Claude judging — FinanceBench v4t-corpus-tuned BATCH cell.

End-of-corpus re-ask of all 150 questions after all 150 docs ingested.
Recall@k=0.967 → 5 questions lost retrieval after corpus dilution.

Per evaluation/claude_judge_protocol.md, manually scored 1-by-1 against
predicted answer + gold answer + question.

Idempotent append to results.jsonl (skips already-judged qids).
"""
from __future__ import annotations
import json
from pathlib import Path

JUDGE_DIR = Path("results/stage3/judge_queue/financebench__v4t-corpus-tuned__batch__seed42")
QID_PREFIX = "financebench__v4t-corpus-tuned__batch__"
QID_SUFFIX = "__seed42"

# (qid_suffix, judge_score, rationale)
JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc0_qa0",   1.0,  "Pred $1,501M vs gold $1577M (3M FY18 capex) — 4.8% diff, within 5% tolerance."),
    ("doc1_qa0",   1.0,  "Pred $8.738B vs gold $8.70B (3M FY18 net PPNE) — 0.4% diff."),
    ("doc2_qa0",   0.0,  "Y/N flip: pred says Yes (capital-intensive) — gold says No (managing efficiently)."),
    ("doc3_qa0",   0.75, "Pred names litigation + impairment + restructuring drivers (matches gold) but omits 1.7% magnitude and decrease-in-gross-margin component."),
    ("doc4_qa0",   1.0,  "Pred correctly identifies Consumer as the dragging segment."),
    ("doc5_qa0",   0.0,  "Pred refuses (not in passages); gold is definitive (No, quick ratio 0.96)."),
    ("doc6_qa0",   1.0,  "Pred lists same three notes (MMM26/30/31) as gold."),
    ("doc7_qa0",   1.0,  "Pred mentions 65 consecutive years — matches gold."),
    ("doc8_qa0",   0.25, "Pred 25.73 vs gold 24.26 (Activision fixed asset turnover) — 6.1% diff, outside 5%."),
    ("doc9_qa0",   0.0,  "Pred 2.9% vs gold 1.9% (Activision capex/rev) — 53% diff."),
    ("doc10_qa0",  0.0,  "Pred 1.73 vs gold 0.66 (Adobe FY15 OCF ratio) — way off."),
    ("doc11_qa0",  0.75, "Pred shows correct work (590,507/903,095) leading to 65.4% but truncated before stating final number."),
    ("doc12_qa0",  0.0,  "Pred 1.23 vs gold 0.83 (Adobe FY17 OCF ratio) — 48% diff."),
    ("doc13_qa0",  0.0,  "Y/N flip: pred Yes (improving) vs gold No (declined 36.8 → 34.6)."),
    ("doc14_qa0",  0.0,  "Pred refuses; gold definitive Yes (FCF conversion 143% → 156%)."),
    ("doc15_qa0",  1.0,  "Pred 0 = gold 0 exactly (AES restructuring costs)."),
    ("doc16_qa0",  0.75, "Pred 9.99x vs gold 9.5x (AES inventory turnover) — 5.2% diff, just outside 5%."),
    ("doc17_qa0",  0.0,  "Pred -1.42 vs gold -0.02 (AES ROA) — magnitude 71× too large."),
    ("doc18_qa0",  0.0,  "Pred 29.12 vs gold 93.86 (Amazon DPO) — 69% diff."),
    ("doc19_qa0",  1.0,  "Pred 30.7% vs gold 30.8% (Amazon revenue change) — 0.3% diff."),
    ("doc20_qa0",  1.0,  "Pred 11,588 = gold $11,588M exactly."),
    ("doc21_qa0",  1.0,  "Pred $1,615.9M vs gold $1616M (Amcor net AR) — within 5%."),
    ("doc22_qa0",  1.0,  "Pred describes supplemental indentures + AFNA-for-AFUSA substitution — matches gold."),
    ("doc23_qa0",  0.5,  "Pred hedges 'may have improved' (correct direction); gold confirms slight improvement 0.67 → 0.69 but pred lacks specific numbers."),
    ("doc24_qa0",  0.0,  "Pred refuses (not in passages); gold is definitive (Czech, Shanghai, NZ acquisitions)."),
    ("doc25_qa0",  1.0,  "Pred packaging = gold packaging."),
    ("doc26_qa0",  1.0,  "Pred reports declining gross margin profile — matches gold's slight decline 0.8%."),
    ("doc27_qa0",  0.5,  "Pred mentions employee costs alongside fixed asset and other costs — does not give the 87% employee split that gold highlights."),
    ("doc28_qa0",  1.0,  "Pred $2,018M = gold $2,018M exactly (Amcor FY23 adj EBITDA)."),
    ("doc29_qa0",  0.0,  "Pred -5% vs gold flat (Amcor real sales change) — wrong direction and magnitude."),
    ("doc30_qa0",  1.0,  "Pred 4.18% vs gold 4.2% (AMD FY15 D&A margin) — 0.5% diff."),
    ("doc31_qa0",  0.0,  "Pred refuses; gold definitive Yes (quick ratio 1.57)."),
    ("doc32_qa0",  1.0,  "Pred lists matching AMD product portfolio (CPUs/GPUs/DPUs/FPGAs/SoCs/APUs/embedded)."),
    ("doc33_qa0",  1.0,  "Pred names EPYC server, semi-custom, Xilinx embedded growth — matches gold."),
    ("doc34_qa0",  1.0,  "Pred Xilinx amortization = gold Xilinx amortization (AMD operating margin driver)."),
    ("doc35_qa0",  1.0,  "Pred operations $3,565M; gold confirms operations as highest cashflow source."),
    ("doc36_qa0",  0.0,  "Pred Gaming +21% vs gold Data Center (segment with biggest sales growth)."),
    ("doc37_qa0",  1.0,  "Pred Yes 16% concentration = gold (AMD customer concentration)."),
    ("doc38_qa0",  0.0,  "Pred 'Common Shares' is equity not debt; gold says no debt securities are registered (AmEx)."),
    ("doc39_qa0",  1.0,  "Pred lists US/EMEA/APAC/LACC — same as gold."),
    ("doc40_qa0",  1.0,  "Pred OM not useful matches gold (performance not measured through OM)."),
    ("doc41_qa0",  1.0,  "Pred GM not useful matches gold."),
    ("doc42_qa0",  1.0,  "Pred 24.6 → 21.6 = gold 24.6 → 21.6 (AmEx ETR change)."),
    ("doc43_qa0",  0.0,  "Pred Long-term debt vs gold Customer deposits (largest AmEx liability)."),
    ("doc44_qa0",  1.0,  "Pred Yes high retention = gold Yes (AmEx 2022 card retention)."),
    ("doc45_qa0",  1.0,  "Pred 0.389B vs gold $0.40B (AWK cash dividends FY20) — 2.75% diff."),
    ("doc46_qa0",  1.0,  "Pred 1,829 vs gold $1832 (AWK FY21 EBITDA) — 0.2% diff."),
    ("doc47_qa0",  0.25, "Pred starts 'Yes positive WC' (wrong) but body computes -$1561M and says 'indicating negative'. Self-contradictory; first impression wrong."),
    ("doc48_qa0",  0.0,  "Pred 3.9% vs gold 2.8% (Best Buy FY15-17 net margin) — 39% diff."),
    ("doc49_qa0",  1.0,  "Pred 5,409 = gold $5409M exactly (Best Buy FY19 inventories)."),
    ("doc50_qa0",  0.0,  "Y/N flip: pred 'not consistent' vs gold 'consistent' (Best Buy gross margins)."),
    ("doc51_qa0",  1.0,  "Pred Current Health + Yardbird in FY22, none for FY23/21 — matches gold."),
    ("doc52_qa0",  1.0,  "Pred operations $1,824M = gold operations $1.8bn (Best Buy FY23 cashflow source)."),
    ("doc53_qa0",  1.0,  "Pred Yes 1874 → 1093 (drop) — gold confirms ~42% drop. Pred matches direction and magnitude."),
    ("doc54_qa0",  0.5,  "Pred 977 → 966 vs gold 982 → 969 (Best Buy store count) — direction correct but specific counts off."),
    ("doc55_qa0",  0.75, "Pred Gaming +9% matches gold Entertainment +9% (Gaming is the Entertainment sub-driver) but technically gold names Entertainment segment."),
    ("doc56_qa0",  1.0,  "Pred 1.74 vs gold 1.73 (Block FY16 working capital ratio) — 0.6% diff."),
    ("doc57_qa0",  1.0,  "Pred 102.0% vs gold 101.5% (Block FY19-20 revenue growth) — 0.5% diff."),
    ("doc58_qa0",  1.0,  "Pred $381.6M vs gold $382M (Block FY20 OCF) — 0.1% diff."),
    ("doc59_qa0",  1.0,  "Pred $12,645 = gold $12645 exactly (Boeing FY18 net PPE)."),
    ("doc60_qa0",  0.5,  "Pred only Commercial Airplanes; gold says Commercial Airplanes 39% AND Defense 35% (both >20%)."),
    ("doc61_qa0",  1.0,  "Pred Lion Air 2018 + Ethiopian 2019 = gold (Boeing legal battles)."),
    ("doc62_qa0",  0.0,  "Pred refuses 'not useful metric'; gold definitive Yes improving 4.8% → 5.3% (Boeing gross margin)."),
    ("doc63_qa0",  0.5,  "Pred only mentions limited airlines; gold also names US government (40% of revenue)."),
    ("doc64_qa0",  1.0,  "Pred Yes cyclical = gold Yes (Boeing cyclicality)."),
    ("doc65_qa0",  1.0,  "Pred 787 increase + 737 increase + 777X resume — matches gold's 737/777X/787 increases."),
    ("doc66_qa0",  0.5,  "Pred says 'lower' (correct direction) but omits gold's specific 0.62% vs -14.76% comparison (Boeing ETR)."),
    ("doc67_qa0",  0.0,  "Pred 1.43% vs gold 0.01 (Coca-Cola FY17 ROA). Pred is 143× too high."),
    ("doc68_qa0",  1.0,  "Pred 39.7% = gold 39.7% exactly (Coca-Cola FY21 COGS margin)."),
    ("doc69_qa0",  1.0,  "Pred 0.80 = gold 0.8 exactly (Coca-Cola FY22 dividend payout)."),
    ("doc70_qa0",  0.0,  "Pred 56.73 vs gold 63.86 (Corning FY20 DPO) — 11.2% diff."),
    ("doc71_qa0",  1.0,  "Pred 10.5% vs gold 10.3% (Corning FY19-21 op income margin) — 1.9% diff."),
    ("doc72_qa0",  1.0,  "Pred 20% → 23% = gold 20% → 23% exactly (Corning ETR change)."),
    ("doc73_qa0",  0.5,  "Pred Yes $2,278M vs gold Yes $831M (Corning WC). Direction right, magnitude wrong (pred didn't restrict to operating items)."),
    ("doc74_qa0",  1.0,  "Pred 59,268 = gold $59268M exactly (Costco FY21 total assets)."),
    ("doc75_qa0",  0.0,  "Pred 8.73 vs gold 17.98 (CVS FY18 fixed asset turnover) — pred is 51% too low."),
    ("doc76_qa0",  1.0,  "Pred Yes capital intensive = gold Yes (CVS FY22)."),
    ("doc77_qa0",  0.75, "Pred mentions usual-and-customary pricing litigation; gold lists multiple including U&C, opioid, etc. Partial."),
    ("doc78_qa0",  1.0,  "Pred Yes $0.55/share = gold Yes $0.55/share (CVS Q2FY22 dividends)."),
    ("doc79_qa0",  1.0,  "Pred Yes Mary Dillon was Ulta CEO = gold Yes she was Ulta CEO (Foot Locker CEO experience)."),
    ("doc80_qa0",  1.0,  "Pred Richard A. Johnson = gold Richard A. Johnson (Foot Locker board nominee)."),
    ("doc81_qa0",  0.0,  "Pred 66.73 vs gold -3.7 (General Mills FY19 CCC) — pred missed that DPO > DIO+DSO produces negative CCC."),
    ("doc82_qa0",  1.0,  "Pred 0.69 vs gold 0.68 (General Mills FY20 WC ratio) — 1.5% diff."),
    ("doc83_qa0",  1.0,  "Pred $3,115.4M vs gold $3215M (General Mills FY20 FCF) — 3.1% diff."),
    ("doc84_qa0",  0.0,  "Pred 0.46 vs gold 0.54 (General Mills FY22 retention rate) — 14.8% diff."),
    ("doc85_qa0",  1.0,  "Pred No (1.3% sales growth) = gold No 1.3% (JnJ FY22 high growth)."),
    ("doc86_qa0",  0.0,  "Pred refuses 'not useful metric'; gold lists specific drivers (COVID exit costs, FX, inflation)."),
    ("doc87_qa0",  0.0,  "Pred refuses 'not provided'; gold definitive 2.7x (JnJ FY22 inventory turnover)."),
    ("doc88_qa0",  0.0,  "Y/N flip + magnitude: pred Yes +12.5% vs gold No deceleration 3.6 → 3.5 (JnJ adj EPS FY23)."),
    ("doc89_qa0",  1.0,  "Pred US +3.0% intl -0.6% = gold exactly (JnJ FY22 geo growth)."),
    ("doc90_qa0",  1.0,  "Pred Consumer Health = gold Consumer Health (JnJ discontinued op)."),
    ("doc91_qa0",  1.0,  "Pred ~$20B = gold ~$20B (JnJ Consumer Health separation gain)."),
    ("doc92_qa0",  1.0,  "Pred $13.2B = gold $13.2B (JnJ Kenvue cash proceeds)."),
    ("doc93_qa0",  1.0,  "Pred Yes 20.0 → 20.1 = gold Yes 20 → 20.1 (JnJ Q2FY23 net earnings/sales)."),
    ("doc94_qa0",  0.0,  "Pred Consumer & Community Banking vs gold Corporate (JPM lowest net rev 2021Q1)."),
    ("doc95_qa0",  0.0,  "Pred $239.45 vs gold $66.56 (JPM bankruptcy per-share) — 3.6× too high."),
    ("doc96_qa0",  1.0,  "Pred GM not relevant (financial firm) = gold GM not relevant."),
    ("doc97_qa0",  0.0,  "Pred Consumer & Community Banking vs gold Corporate & Investment Bank (JPM highest net income 2022Q2)."),
    ("doc98_qa0",  1.0,  "Pred Yes VaR decreased $7M = gold Yes decreased (JPM Q2 2023 VaR)."),
    ("doc99_qa0",  0.0,  "Pred 3.06 vs gold 6.25 (Kraft Heinz FY19 inventory turnover) — pred is half of gold."),
    ("doc100_qa0", 1.0,  "Pred 1.38 vs gold 1.33 — 3.8% diff, within 5% tolerance."),
    ("doc101_qa0", 1.0,  "Pred 5,818 = gold $5818M exactly (Lockheed FY21 NWC)."),
    ("doc102_qa0", 1.0,  "Pred 0.4% = gold 0.4% exactly (Lockheed FY20-22 revenue CAGR)."),
    ("doc103_qa0", 1.0,  "Pred $302.578M vs gold $303M (MGM FY18 AP) — 0.1% diff."),
    ("doc104_qa0", 0.0,  "Pred -3.5% vs gold 7.9% (MGM FY18-20 capex/rev) — wrong sign and magnitude."),
    ("doc105_qa0", 1.0,  "Pred $0.01/share annual = gold Yes $0.01/share (MGM FY22 dividends)."),
    ("doc106_qa0", 1.0,  "Pred Las Vegas Strip = gold Las Vegas (~90% of EBITDAR)."),
    ("doc107_qa0", 0.0,  "Pred 1.61 vs gold 0 (MGM interest coverage FY22; gold notes EBIT was negative so ratio is zero)."),
    ("doc108_qa0", 1.0,  "Pred MGM China = gold MGM China (worst topline performer FY22)."),
    ("doc109_qa0", 1.0,  "Pred corporate bonds = gold corporate bonds (MGM H1FY23 largest ST investment)."),
    ("doc110_qa0", 1.0,  "Pred $32,780M = gold $32,780M exactly (Microsoft FY16 COGS)."),
    ("doc111_qa0", 0.25, "Pred says 'Yes' but body describes decrease ($47,032M → $41,990M) — Y/N answer is wrong even though numbers point to decrease (matches gold). Self-contradictory."),
    ("doc112_qa0", 0.0,  "Pred 4.51% vs gold 5.4% (Netflix FY15 EBITDA margin) — 16.5% diff."),
    ("doc113_qa0", 1.0,  "Pred 5,466.3M vs gold $5466M (Netflix FY17 total CL) — 0.005% diff."),
    ("doc114_qa0", 1.0,  "Pred 56.3% vs gold 55.1% — 2.2% diff, within 5%."),
    ("doc115_qa0", 1.0,  "Pred $16,525 = gold $16525M exactly (Nike FY19 total CA)."),
    ("doc116_qa0", 1.0,  "Pred 3.61 vs gold 3.46 (Nike FY21 inventory turnover) — 4.3% diff, within 5%."),
    ("doc117_qa0", 1.0,  "Pred operations $5,841M = gold operations highest (Nike FY23 cashflow source)."),
    ("doc118_qa0", 0.5,  "Pred Yes positive WC computes $12,416M; gold Yes $1.6B. Direction matches, magnitude off ~7.7×."),
    ("doc119_qa0", 1.0,  "Pred $4.625B vs gold $4.60B (PepsiCo FY21 capex) — 0.5% diff."),
    ("doc120_qa0", 0.5,  "Pred lists PepsiCo's reporting segments (US/Dev Europe/Dev RoW/Emerging) instead of operating geographies (N.America/Latin America/Europe/etc) — different framing."),
    ("doc121_qa0", 0.75, "Pred matches gold's conclusion (no material legal proceedings) but with more verbose hedging."),
    ("doc122_qa0", 1.0,  "Pred 411 = gold $411M exactly (Pepsico FY22 restructuring)."),
    ("doc123_qa0", 0.0,  "Pred $14,275M vs gold $9068M (PepsiCo FY22 EBITDA less capex) — 57% too high; pred returns gross EBITDA not EBITDA-capex."),
    ("doc124_qa0", 1.0,  "Pred 16.5% = gold 16.5% exactly (PepsiCo FY22 EBITDA margin)."),
    ("doc125_qa0", 1.0,  "Pred defeated = gold defeated (Pepsico net-zero shareholder vote)."),
    ("doc126_qa0", 1.0,  "Pred $400,000,000 = gold $400M (Pepsico May 2023 credit increase)."),
    ("doc127_qa0", 0.25, "Pred 4.2B+4.2B=9.4B (arithmetic error; should be 8.4B which matches gold). Components right, sum wrong."),
    ("doc128_qa0", 1.0,  "Pred strong start + momentum = gold strong start (Pepsico FY23 guidance raise reason)."),
    ("doc129_qa0", 1.0,  "Pred 1pp (8 → 9) = gold 1pp exactly (Pepsico EPS guidance raise)."),
    ("doc130_qa0", 0.25, "Pred Yes (correct Y/N) but cites 'Net income attributable to shareholders' instead of PPNE — wrong concept entirely."),
    ("doc131_qa0", 0.25, "Pred mentions Consumer Healthcare JV but cites '2021' and '$(6) million' — wrong year (should be 2019) and wrong magnitude."),
    ("doc132_qa0", 0.5,  "Pred Trillium + Array correct; Upjohn was a spinoff, not acquisition. Gold says Therachon. 2/3 correct."),
    ("doc133_qa0", 0.0,  "Pred $700M vs gold $77.78M (Pfizer Upjohn spinoff payment) — 9× too high."),
    ("doc134_qa0", 1.0,  "Pred Developed Rest of World = gold Developed Rest of the World."),
    ("doc135_qa0", 1.0,  "Pred Yes Upjohn = gold Yes Upjohn (Pfizer spinoff)."),
    ("doc136_qa0", 0.0,  "Pred common stock NASDAQ is equity not debt; gold says no debt securities registered (Ulta Beauty)."),
    ("doc137_qa0", 0.75, "Pred refuses 'not mentioned'; gold answer is 'none'. Refusal aligns with the correct answer but lacks confidence."),
    ("doc138_qa0", 1.0,  "Pred lower marketing + incentive comp leverage = gold (Ulta SG&A driver FY23)."),
    ("doc139_qa0", 1.0,  "Pred 47 new stores + brand launches = gold 47 new stores (Ulta inventory increase)."),
    ("doc140_qa0", 1.0,  "Pred 36.5% vs gold 36% (Ulta Q4 stock repurchase %) — 1.4% diff."),
    ("doc141_qa0", 0.0,  "Y/N flip: pred Decrease vs gold increased (Ulta wages % FY23)."),
    ("doc142_qa0", 1.0,  "Pred cross currency swaps = gold cross currency swaps (Verizon top notional derivative)."),
    ("doc143_qa0", 0.75, "Pred $1,097M pension only; gold pension $1097M + health/life $862M. Pred misses the health/life component."),
    ("doc144_qa0", 0.0,  "Pred refuses; gold definitive No (Verizon quick ratio 0.54 unhealthy)."),
    ("doc145_qa0", 1.0,  "Pred Yes capital intensive = gold Yes (Verizon FY22)."),
    ("doc146_qa0", 0.25, "Pred says 'Yes' but body describes decrease ($150,868M → $150,639M) — Y/N flipped; numbers match gold's decrease."),
    ("doc147_qa0", 0.0,  "Pred 30.73 vs gold 42.69 (Walmart FY18 DPO) — 28% diff."),
    ("doc148_qa0", 0.0,  "Pred -6.0% vs gold 0.2% (Walmart FY18-19 op income margin change) — wrong sign and magnitude."),
    ("doc149_qa0", 0.0,  "Pred 3.8% vs gold 6.2% (Walmart FY18-20 EBITDA margin) — 39% diff."),
]


def main() -> None:
    results_path = JUDGE_DIR / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing (idempotent skip)
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

    # Re-read and report stats
    with results_path.open(encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    scores = [e["judge_score"] for e in lines]
    from collections import Counter
    dist = Counter(scores)
    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"Appended {len(new_records)} new judgments (skipped {skipped}, total now {len(lines)})")
    print(f"Score distribution: {dict(sorted(dist.items()))}")
    print(f"Mean judge: {mean:.4f}")


if __name__ == "__main__":
    main()
