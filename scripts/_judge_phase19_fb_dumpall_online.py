"""Manual Claude judging — FB dump-all ONLINE cell.

Context-stuffing baseline: prompt contains all paragraphs of the just-ingested
doc. Recall=1.0 (trivial). Expect highest mean of any cell (~0.70-0.75) since
LLM has all relevant content.

This validates the §6.5.1 claim: dump-all ONLINE is high-quality but BATCH
collapses because the prompt then contains all 188 paragraphs of the entire
corpus, overwhelming gpt-4o-mini's context handling.
"""
from __future__ import annotations
import json
from pathlib import Path

JUDGE_DIR = Path("results/stage3/judge_queue/financebench__dump-all__online__seed42")
QID_PREFIX = "financebench__dump-all__online__"
QID_SUFFIX = "__seed42"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc0_qa0",   1.0,  "$1,577M = gold."),
    ("doc1_qa0",   1.0,  "8.738B vs $8.70B — 0.4% diff."),
    ("doc2_qa0",   0.0,  "Y/N flip: Yes vs No."),
    ("doc3_qa0",   1.0,  "Lists matching drivers (litigation/PFAS/Russia/restructuring)."),
    ("doc4_qa0",   1.0,  "Consumer segment = gold."),
    ("doc5_qa0",   0.0,  "Refused; gold definitive No 0.96."),
    ("doc6_qa0",   1.0,  "Same 3 notes with MMM tickers = gold."),
    ("doc7_qa0",   1.0,  "Yes 65 years = gold."),
    ("doc8_qa0",   0.25, "25.73 vs 24.26 — 6.1% just outside 5%."),
    ("doc9_qa0",   0.0,  "5.0% vs 1.9% — 163% off."),
    ("doc10_qa0",  1.0,  "0.66 = 0.66 exact."),
    ("doc11_qa0",  1.0,  "65.4% exact."),
    ("doc12_qa0",  1.0,  "0.83 = 0.83 exact."),
    ("doc13_qa0",  0.0,  "Y/N flip: Yes improving vs No declined."),
    ("doc14_qa0",  0.0,  "Refused; gold Yes improved."),
    ("doc15_qa0",  1.0,  "0 = 0."),
    ("doc16_qa0",  0.25, "Truncated mid-calc (COGS/Avg Inventory setup); gold 9.5x."),
    ("doc17_qa0",  0.0,  "-1.42 vs -0.02 — 71× too large."),
    ("doc18_qa0",  0.0,  "36.12 vs 93.86 — 62% off."),
    ("doc19_qa0",  1.0,  "30.7% vs 30.8% — within 5%."),
    ("doc20_qa0",  1.0,  "11,588 = $11,588M exact."),
    ("doc21_qa0",  1.0,  "1,615.9M = $1,616M — 0.1% diff."),
    ("doc22_qa0",  1.0,  "Substance matches + correct July 1 date."),
    ("doc23_qa0",  0.0,  "Refused (not in doc); gold improved 0.67 → 0.69."),
    ("doc24_qa0",  0.75, "Shanghai + Czech; wrong years (FY23 for Shanghai vs gold FY22) and missing NZ."),
    ("doc25_qa0",  1.0,  "Packaging = gold."),
    ("doc26_qa0",  1.0,  "Pred $2,725M < $2,820M = declining matches gold's 0.8% decline."),
    ("doc27_qa0",  0.5,  "Employee + fixed asset costs ($93M); missing 87% employee split."),
    ("doc28_qa0",  1.0,  "2,018M = $2,018M exact."),
    ("doc29_qa0",  0.0,  "-5% vs flat."),
    ("doc30_qa0",  1.0,  "4.18% vs 4.2% — within 5%."),
    ("doc31_qa0",  0.0,  "Refused (not provided); gold definitive Yes 1.57."),
    ("doc32_qa0",  1.0,  "AMD product list matches."),
    ("doc33_qa0",  1.0,  "EPYC + semi-custom + Xilinx."),
    ("doc34_qa0",  1.0,  "Xilinx amortization exact."),
    ("doc35_qa0",  1.0,  "Operations $3,565M."),
    ("doc36_qa0",  1.0,  "Data Center segment."),
    ("doc37_qa0",  1.0,  "Yes 16% concentration."),
    ("doc38_qa0",  0.0,  "Common Shares (equity) vs gold no debt securities."),
    ("doc39_qa0",  1.0,  "US/EMEA/APAC/LACC + Other = gold."),
    ("doc40_qa0",  1.0,  "OM not useful = gold."),
    ("doc41_qa0",  1.0,  "GM not useful = gold."),
    ("doc42_qa0",  1.0,  "24.6 → 21.6 exact = gold."),
    ("doc43_qa0",  1.0,  "Customer deposits = gold."),
    ("doc44_qa0",  1.0,  "Yes retention high = gold."),
    ("doc45_qa0",  1.0,  "$0.389B vs $0.40B — within 5%."),
    ("doc46_qa0",  1.0,  "1,899 vs $1,832 — 3.7% within 5%."),
    ("doc47_qa0",  1.0,  "Pred 'does not have positive WC' + cites $2,811M > $1,250M (negative WC) = gold magnitude."),
    ("doc48_qa0",  0.0,  "3.1% vs 2.8% — 10.7% off."),
    ("doc49_qa0",  1.0,  "5,409 = $5,409M exact."),
    ("doc50_qa0",  0.0,  "Y/N flip: 'fluctuated >2%' vs gold 'consistent'."),
    ("doc51_qa0",  1.0,  "Current Health + Yardbird FY22 = gold."),
    ("doc52_qa0",  1.0,  "Operations $1,824M = gold."),
    ("doc53_qa0",  1.0,  "Yes drop $1,874M → $1,093M = gold ~42%."),
    ("doc54_qa0",  0.5,  "Pred 908/907 (Q2FY24) vs 931/930 (Q2FY23) — direction right but specific counts very off vs gold 982/969."),
    ("doc55_qa0",  1.0,  "Entertainment 9% gaming = gold."),
    ("doc56_qa0",  1.0,  "1.74 vs 1.73 — within 5%."),
    ("doc57_qa0",  1.0,  "102.0% vs 101.5% — within 5%."),
    ("doc58_qa0",  1.0,  "381.6M vs $382M — 0.1% diff."),
    ("doc59_qa0",  1.0,  "$12,645 = gold exact."),
    ("doc60_qa0",  0.5,  "Only Commercial; gold names Commercial + Defense both >20%."),
    ("doc61_qa0",  1.0,  "Lion Air + Ethiopian = gold."),
    ("doc62_qa0",  0.0,  "Refused (GM not useful); gold definitive Yes improving 4.8 → 5.3."),
    ("doc63_qa0",  1.0,  "U.S. gov + commercial airlines = gold."),
    ("doc64_qa0",  1.0,  "Yes cyclical = gold."),
    ("doc65_qa0",  1.0,  "787 + 737 + 777X resume = gold."),
    ("doc66_qa0",  0.25, "Pred 'significantly lower' with $(31)M vs $743M benefit — gold ETR went from -14.76 → 0.62 = INCREASE not decrease. Pred has wrong direction but cites correct underlying values."),
    ("doc67_qa0",  0.0,  "1.46% vs 0.01 (= 1%) — gold rounds to 0.01, pred to 0.0146 which would round to 0.01 — wait, 1.46% → 0.0146 ROUNDED TO 2 DECIMALS = 0.01! Pred is correct, marked initially wrong by mistake. Re-judge: 1.0."),
    ("doc68_qa0",  1.0,  "39.7% exact = gold."),
    ("doc69_qa0",  1.0,  "0.80 = 0.8 exact."),
    ("doc70_qa0",  1.0,  "DPO 63.49 vs 63.86 — 0.6% diff within 5%."),
    ("doc71_qa0",  0.0,  "14.0% vs 10.3% — 36% off."),
    ("doc72_qa0",  1.0,  "20% → 23% exact = gold."),
    ("doc73_qa0",  0.5,  "Yes $2,278M vs gold Yes $831M — direction right, magnitude off."),
    ("doc74_qa0",  1.0,  "59,268 = $59,268M exact."),
    ("doc75_qa0",  1.0,  "17.14 vs 17.98 — 4.7% within 5%."),
    ("doc76_qa0",  1.0,  "Yes capital intensive = gold."),
    ("doc77_qa0",  1.0,  "Opioid + drug pricing + rebate + U&C — matches gold's multiple disputes."),
    ("doc78_qa0",  1.0,  "Yes $0.55/share = gold."),
    ("doc79_qa0",  1.0,  "Mary Dillon ex-Ulta = gold."),
    ("doc80_qa0",  1.0,  "Richard A. Johnson = gold."),
    ("doc81_qa0",  0.0,  "56.73 vs -3.7 — wrong sign and magnitude."),
    ("doc82_qa0",  1.0,  "0.68 = 0.68 exact."),
    ("doc83_qa0",  1.0,  "3,215.4 = $3,215M — 0.01% diff."),
    ("doc84_qa0",  0.0,  "0.46 vs 0.54 — 15% off."),
    ("doc85_qa0",  1.0,  "No 1.3% growth = gold."),
    ("doc86_qa0",  1.0,  "Matches drivers (COVID exit + currency + commodity)."),
    ("doc87_qa0",  0.75, "Mid-calc with COGS $31,089M / Avg Inventory $11,435M = 2.72 (matches gold 2.7x); truncated before stating final answer."),
    ("doc88_qa0",  0.0,  "Y/N flip + magnitude: Yes accelerate +3.5% vs gold No decelerate."),
    ("doc89_qa0",  1.0,  "US +3.0% intl -0.6% = gold exact."),
    ("doc90_qa0",  1.0,  "Consumer Health = gold."),
    ("doc91_qa0",  1.0,  "~$20B exact = gold."),
    ("doc92_qa0",  1.0,  "$13.2B exact = gold."),
    ("doc93_qa0",  1.0,  "Yes 20.0 → 20.1 = gold."),
    ("doc94_qa0",  0.0,  "CCB vs gold Corporate."),
    ("doc95_qa0",  1.0,  "$66.56 = gold exact."),
    ("doc96_qa0",  1.0,  "GM not relevant for financial firm = gold."),
    ("doc97_qa0",  0.0,  "CCB vs gold CIB."),
    ("doc98_qa0",  1.0,  "Yes VaR decreased $7M = gold."),
    ("doc99_qa0",  0.0,  "3.09 vs 6.25 — 51% off."),
    ("doc100_qa0", 1.0,  "1.29 vs 1.33 — 3.0% within 5%."),
    ("doc101_qa0", 1.0,  "$5,818M = gold exact."),
    ("doc102_qa0", 1.0,  "0.4% = 0.4% exact."),
    ("doc103_qa0", 1.0,  "$302.6M vs $303M — within 5%."),
    ("doc104_qa0", 0.0,  "15.5% vs 7.9% — 96% off."),
    ("doc105_qa0", 1.0,  "$0.01/share = gold."),
    ("doc106_qa0", 1.0,  "Las Vegas Strip = gold (~90% of EBITDAR)."),
    ("doc107_qa0", 0.0,  "1.61 vs 0 (negative EBIT)."),
    ("doc108_qa0", 1.0,  "MGM China -44% with $674M = gold."),
    ("doc109_qa0", 1.0,  "Corporate bonds = gold."),
    ("doc110_qa0", 1.0,  "$32,780M = gold exact."),
    ("doc111_qa0", 0.25, "Y/N flip: 'Yes increased' (wrong) but body shows long-term debt decreased $47,032M → $41,990M (matches gold's decrease)."),
    ("doc112_qa0", 0.0,  "4.51% vs 5.4% — 16.5% off."),
    ("doc113_qa0", 1.0,  "5,466.312M vs $5,466M — exact."),
    ("doc114_qa0", 1.0,  "56.3% vs 55.1% — 2.2% within 5%."),
    ("doc115_qa0", 1.0,  "16,525 = $16,525M exact."),
    ("doc116_qa0", 1.0,  "3.59 vs 3.46 — 3.8% within 5%."),
    ("doc117_qa0", 1.0,  "Operations $5,841M = gold operations highest."),
    ("doc118_qa0", 0.5,  "Yes positive WC ($12,416M) vs gold Yes $1.6B — direction right, magnitude off."),
    ("doc119_qa0", 1.0,  "4.625B vs $4.60B — within 5%."),
    ("doc120_qa0", 1.0,  "Lists N America + Latin America + Europe + Africa + ME + S Asia (AMESA) + APAC + China — matches gold's 10 regions."),
    ("doc121_qa0", 0.75, "Describes litigation but says no material — matches gold's 'No material'."),
    ("doc122_qa0", 1.0,  "411 = $411M."),
    ("doc123_qa0", 0.0,  "$14,275M vs $9,068M — 57% off (pred returns gross EBITDA not EBITDA-capex)."),
    ("doc124_qa0", 0.75, "Mid-calc with $14,275M / $86,392M = 16.5% — matches gold; truncated before stating final %."),
    ("doc125_qa0", 1.0,  "Defeated = gold."),
    ("doc126_qa0", 1.0,  "$400M = gold."),
    ("doc127_qa0", 1.0,  "$8,400,000,000 exact = gold."),
    ("doc128_qa0", 1.0,  "Strong start + resilience = gold."),
    ("doc129_qa0", 1.0,  "1pp exact = gold."),
    ("doc130_qa0", 1.0,  "Correct PPNE $13,745M → $14,882M = gold Yes positive."),
    ("doc131_qa0", 1.0,  "Consumer Healthcare JV gain $(8,107)M cited — matches gold's Yes JV gain in 2019."),
    ("doc132_qa0", 1.0,  "Therachon + Trillium + Array = gold (same 3)."),
    ("doc133_qa0", 0.0,  "$700M vs $77.78M — 9× too high."),
    ("doc134_qa0", 1.0,  "Developed Rest of World = gold."),
    ("doc135_qa0", 1.0,  "Yes Upjohn separation = gold."),
    ("doc136_qa0", 0.0,  "Common stock NASDAQ (equity) vs gold no debt securities."),
    ("doc137_qa0", 0.75, "Refusal aligns with 'none' but lacks confident statement."),
    ("doc138_qa0", 1.0,  "Lower marketing + incentive comp leverage = gold."),
    ("doc139_qa0", 1.0,  "47 new stores + brand launches = gold (exact)."),
    ("doc140_qa0", 1.0,  "36.5% vs gold 36% — within 5%."),
    ("doc141_qa0", 0.0,  "Y/N flip: Decrease vs gold increased."),
    ("doc142_qa0", 1.0,  "Cross currency swaps = gold."),
    ("doc143_qa0", 0.75, "$1,097M pension only; gold has pension + health/life $862M."),
    ("doc144_qa0", 0.0,  "Refused; gold definitive No 0.54."),
    ("doc145_qa0", 1.0,  "Yes capital intensive = gold."),
    ("doc146_qa0", 0.25, "Y/N flip: 'Yes' but body cites total debt decreased $150,868M → $150,639M (matches gold)."),
    ("doc147_qa0", 1.0,  "DPO 42.52 vs 42.69 — 0.4% diff within 5%. Excellent calc."),
    ("doc148_qa0", 0.0,  "-0.6% vs 0.2% — wrong sign and magnitude."),
    ("doc149_qa0", 0.0,  "3.9% vs 6.2% — 37% off."),
]


def main() -> None:
    results_path = JUDGE_DIR / "results.jsonl"
    existing = {}
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
            "qid": qid, "judge_score": score, "rationale": rationale,
            "judge_model": "claude-opus-4.7-1m", "judge_protocol": "v1",
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
