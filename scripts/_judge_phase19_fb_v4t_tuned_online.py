"""
Manual Claude judging — FinanceBench v4t-tuned ONLINE cell.

Per-doc tuned θ (the §5.2 narrow-tuned variant). Recall@k=0.527 —
similar to canonical's 0.567 (per-doc tuning didn't help much on corpus
scale). Expected mean ~0.48-0.52.
"""
from __future__ import annotations
import json
from pathlib import Path

JUDGE_DIR = Path("results/stage3/judge_queue/financebench__v4t-tuned__online__seed42")
QID_PREFIX = "financebench__v4t-tuned__online__"
QID_SUFFIX = "__seed42"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc0_qa0",   1.0,  "Pred $1,577M = gold (3M FY18 capex)."),
    ("doc1_qa0",   0.0,  "Pred $16.4B vs gold $8.70B (3M FY18 PPNE) — 88% off."),
    ("doc2_qa0",   0.0,  "Y/N flip: pred Yes vs gold No."),
    ("doc3_qa0",   1.0,  "Pred names litigation+PFAS+Russia+restructuring drivers — matches gold."),
    ("doc4_qa0",   1.0,  "Pred Consumer = gold."),
    ("doc5_qa0",   0.0,  "Refused; gold definitive No 0.96."),
    ("doc6_qa0",   1.0,  "Same 3 notes."),
    ("doc7_qa0",   1.0,  "Yes 65 years = gold."),
    ("doc8_qa0",   0.0,  "Pred 67.73 vs gold 24.26 — 179% too high."),
    ("doc9_qa0",   0.0,  "6.0% vs 1.9%."),
    ("doc10_qa0",  0.0,  "0.40 vs 0.66."),
    ("doc11_qa0",  1.0,  "65.4% exact."),
    ("doc12_qa0",  0.0,  "2.90 vs 0.83."),
    ("doc13_qa0",  0.0,  "Refused; gold definitive No."),
    ("doc14_qa0",  0.0,  "Refused; gold definitive Yes."),
    ("doc15_qa0",  1.0,  "0 = 0."),
    ("doc16_qa0",  0.0,  "Refused; gold 9.5x."),
    ("doc17_qa0",  0.0,  "-2.29 vs -0.02 — 113× too large."),
    ("doc18_qa0",  0.0,  "Refused (couldn't compute); gold 93.86."),
    ("doc19_qa0",  1.0,  "29.5% vs 30.8% — 4.2% diff."),
    ("doc20_qa0",  0.0,  "Refused; gold $11,588M."),
    ("doc21_qa0",  1.0,  "$1,615.9M = gold $1,616M."),
    ("doc22_qa0",  1.0,  "Substance matches gold + correct July 1 date."),
    ("doc23_qa0",  0.0,  "Refused; gold 0.67 → 0.69."),
    ("doc24_qa0",  0.75, "Shanghai + Czech mentioned; wrong years (FY23 vs FY22) and missing NZ acquisition."),
    ("doc25_qa0",  1.0,  "Packaging = gold."),
    ("doc26_qa0",  0.0,  "Refused (not useful); gold definitive No 0.8% decline."),
    ("doc27_qa0",  0.5,  "Mentions employee + fixed asset costs ($93M); missing 87%-employee detail."),
    ("doc28_qa0",  1.0,  "$2,018M exact."),
    ("doc29_qa0",  0.0,  "-5% vs flat."),
    ("doc30_qa0",  0.0,  "Refused; gold 4.2%."),
    ("doc31_qa0",  0.0,  "Refused; gold Yes 1.57."),
    ("doc32_qa0",  1.0,  "AMD product list matches."),
    ("doc33_qa0",  1.0,  "EPYC + semi-custom + Xilinx."),
    ("doc34_qa0",  1.0,  "Xilinx amortization exact."),
    ("doc35_qa0",  1.0,  "Operations brought most cashflow."),
    ("doc36_qa0",  1.0,  "Data Center segment = gold."),
    ("doc37_qa0",  0.0,  "Refused; gold Yes 16% concentration."),
    ("doc38_qa0",  0.0,  "Common Shares (equity) vs gold none."),
    ("doc39_qa0",  1.0,  "US/EMEA/APAC/LACC + Other = gold."),
    ("doc40_qa0",  1.0,  "OM not useful = gold."),
    ("doc41_qa0",  1.0,  "GM not useful = gold."),
    ("doc42_qa0",  0.0,  "Pred 9.4 → 10.5 vs gold 24.6 → 21.6 — entirely wrong numbers."),
    ("doc43_qa0",  1.0,  "Customer deposits = gold."),
    ("doc44_qa0",  1.0,  "Yes retention high = gold."),
    ("doc45_qa0",  0.0,  "Refused; gold $0.40B."),
    ("doc46_qa0",  0.0,  "Refused; gold $1,832M."),
    ("doc47_qa0",  0.5,  "Pred says 'does not have positive WC' (correct direction) but reasons 'current liabilities not provided'; pred then computes $1,250-$2,811=-$1,561M but only because of citing earlier math. Direction right, partial reasoning."),
    ("doc48_qa0",  0.0,  "3.1% vs 2.8% — 10.7% diff."),
    ("doc49_qa0",  0.0,  "Refused; gold $5,409M."),
    ("doc50_qa0",  0.0,  "Y/N flip: pred not consistent vs gold consistent."),
    ("doc51_qa0",  1.0,  "Current Health + Yardbird FY22 = gold."),
    ("doc52_qa0",  0.0,  "Refused; gold operations $1.8bn."),
    ("doc53_qa0",  0.0,  "Refused; gold definitive Yes ~42% drop."),
    ("doc54_qa0",  0.5,  "Pred 977 → 966 vs gold 982 → 969 (Best Buy store counts) — direction right, specific counts off."),
    ("doc55_qa0",  0.75, "Gaming matches sub-driver; gold names Entertainment +9%."),
    ("doc56_qa0",  1.0,  "1.74 vs 1.73 — within 5%."),
    ("doc57_qa0",  0.0,  "Refused; gold 101.5%."),
    ("doc58_qa0",  0.0,  "Refused; gold $382M."),
    ("doc59_qa0",  0.0,  "Refused; gold $12,645M."),
    ("doc60_qa0",  0.5,  "Pred only Commercial; gold Commercial AND Defense both >20%."),
    ("doc61_qa0",  1.0,  "Lion Air + Ethiopian = gold."),
    ("doc62_qa0",  0.0,  "Refused (GM not useful); gold definitive Yes improving 4.8 → 5.3."),
    ("doc63_qa0",  1.0,  "US gov + airlines = gold."),
    ("doc64_qa0",  1.0,  "Yes cyclical = gold."),
    ("doc65_qa0",  1.0,  "787 + 737 + 777X resume — matches gold's three aircraft increases."),
    ("doc66_qa0",  0.0,  "Refused; gold 0.62% vs -14.76%."),
    ("doc67_qa0",  0.0,  "Pred 0.03 vs gold 0.01 (Coca-Cola FY17 ROA) — 200% diff."),
    ("doc68_qa0",  0.0,  "Pred 36.4% vs gold 39.7% (Coca-Cola COGS) — 8.3% diff, beyond 5%."),
    ("doc69_qa0",  0.0,  "Refused; gold 0.8."),
    ("doc70_qa0",  0.0,  "Refused; gold 63.86."),
    ("doc71_qa0",  0.0,  "Refused; gold 10.3%."),
    ("doc72_qa0",  1.0,  "20% → 23% exact = gold."),
    ("doc73_qa0",  0.0,  "Refused; gold Yes $831M."),
    ("doc74_qa0",  0.0,  "Refused; gold $59,268M."),
    ("doc75_qa0",  0.0,  "Refused; gold 17.98."),
    ("doc76_qa0",  1.0,  "Yes capital intensive = gold."),
    ("doc77_qa0",  1.0,  "Pred opioid + drug pricing + U&C — matches gold."),
    ("doc78_qa0",  1.0,  "Yes $0.55/share = gold."),
    ("doc79_qa0",  1.0,  "Mary Dillon ex-Ulta CEO = gold."),
    ("doc80_qa0",  1.0,  "Richard Johnson = gold."),
    ("doc81_qa0",  0.0,  "Refused; gold -3.7."),
    ("doc82_qa0",  0.0,  "Refused; gold 0.68."),
    ("doc83_qa0",  0.0,  "Refused; gold $3,215M."),
    ("doc84_qa0",  0.0,  "0.11 vs 0.54 — 80% off."),
    ("doc85_qa0",  1.0,  "No 1.3% growth = gold."),
    ("doc86_qa0",  1.0,  "Names COVID exit + currency + commodity inflation drivers = gold."),
    ("doc87_qa0",  0.0,  "Refused mid-calc (no final answer); gold 2.7x."),
    ("doc88_qa0",  0.0,  "Refused; gold definitive No decelerate."),
    ("doc89_qa0",  0.0,  "Refused; gold definitive US +3% intl -0.6%."),
    ("doc90_qa0",  1.0,  "Consumer Health = gold."),
    ("doc91_qa0",  1.0,  "~$20B exact = gold."),
    ("doc92_qa0",  1.0,  "$13.2B exact = gold."),
    ("doc93_qa0",  0.0,  "Refused; gold Yes 20.0 → 20.1."),
    ("doc94_qa0",  0.0,  "Commercial Banking vs gold Corporate (JPM lowest 2021Q1)."),
    ("doc95_qa0",  1.0,  "$66.56 exact = gold."),
    ("doc96_qa0",  1.0,  "GM not relevant (financial firm) = gold."),
    ("doc97_qa0",  1.0,  "Corporate & Investment Bank = gold."),
    ("doc98_qa0",  1.0,  "Yes VaR decreased $7M = gold."),
    ("doc99_qa0",  0.0,  "Refused; gold 6.25."),
    ("doc100_qa0", 0.0,  "0.46 vs 1.33 — 65% off."),
    ("doc101_qa0", 0.0,  "Refused; gold $5,818M."),
    ("doc102_qa0", 0.0,  "Pred 4.9% vs gold 0.4% (Lockheed CAGR)."),
    ("doc103_qa0", 0.0,  "Refused; gold $303M."),
    ("doc104_qa0", 0.0,  "Pred 10.5% vs gold 7.9% — 33% diff."),
    ("doc105_qa0", 1.0,  "$0.01/share annual = gold."),
    ("doc106_qa0", 0.0,  "Refused; gold Las Vegas ~90%."),
    ("doc107_qa0", 0.0,  "Refused (can't calculate); gold 0 (negative EBIT)."),
    ("doc108_qa0", 1.0,  "Pred MGM China -44% $674M = gold MGM China -44%."),
    ("doc109_qa0", 1.0,  "Corporate bonds = gold."),
    ("doc110_qa0", 0.0,  "Refused; gold $32,780M."),
    ("doc111_qa0", 0.0,  "Refused; gold definitive No decreased $2.5bn."),
    ("doc112_qa0", 0.0,  "Pred 4.51% vs gold 5.4% — 16.5% diff."),
    ("doc113_qa0", 0.0,  "Refused; gold $5,466M."),
    ("doc114_qa0", 1.0,  "56.3% vs 55.1% — within 5%."),
    ("doc115_qa0", 0.0,  "Refused; gold $16,525M."),
    ("doc116_qa0", 0.0,  "Refused; gold 3.46."),
    ("doc117_qa0", 0.0,  "Refused; gold operations."),
    ("doc118_qa0", 0.0,  "Refused; gold Yes $1.6B."),
    ("doc119_qa0", 1.0,  "4.625B vs gold $4.60B — within 5%."),
    ("doc120_qa0", 1.0,  "Lists same 10 regions as gold."),
    ("doc121_qa0", 0.75, "Describes ongoing litigation but says no material — matches gold's No material legal battles."),
    ("doc122_qa0", 1.0,  "411 = gold $411M."),
    ("doc123_qa0", 0.0,  "Pred $10,586M vs gold $9,068M — 16.7% diff."),
    ("doc124_qa0", 0.0,  "Truncated mid-calc with $10,389M EBITDA → would yield ~12.5% margin; gold 16.5%."),
    ("doc125_qa0", 1.0,  "Defeated = gold."),
    ("doc126_qa0", 1.0,  "$400M = gold."),
    ("doc127_qa0", 0.0,  "$4,950M vs $8.4B — 41% off."),
    ("doc128_qa0", 1.0,  "Strong start + resilience = gold strong start."),
    ("doc129_qa0", 1.0,  "1pp exact = gold."),
    ("doc130_qa0", 1.0,  "Correct PPNE $13,745M → $14,882M = gold Yes positive."),
    ("doc131_qa0", 0.0,  "Refused; gold definitive Yes Consumer Healthcare JV gain (2019)."),
    ("doc132_qa0", 1.0,  "Therachon + Trillium + Array = gold."),
    ("doc133_qa0", 0.0,  "$700M vs $77.78M — 9× too high."),
    ("doc134_qa0", 1.0,  "Developed Rest of World = gold."),
    ("doc135_qa0", 1.0,  "Yes Upjohn separation = gold."),
    ("doc136_qa0", 0.0,  "Common stock NASDAQ (equity) vs gold no debt securities."),
    ("doc137_qa0", 0.75, "Refusal aligns with gold's 'none' but lacks confident statement."),
    ("doc138_qa0", 1.0,  "Pred lower marketing + incentive comp leverage = gold."),
    ("doc139_qa0", 0.25, "Pred 'decrease of $104,233' wrong direction; gold says increase driven by 47 new stores."),
    ("doc140_qa0", 1.0,  "36.5% vs gold 36% — within 5%."),
    ("doc141_qa0", 0.0,  "Y/N flip: pred Decrease vs gold increased."),
    ("doc142_qa0", 1.0,  "Cross currency swaps = gold."),
    ("doc143_qa0", 0.75, "Pred $1,097M pension only; gold has both pension + health/life $862M."),
    ("doc144_qa0", 0.0,  "Refused; gold definitive No 0.54."),
    ("doc145_qa0", 1.0,  "Yes capital intensive = gold."),
    ("doc146_qa0", 0.25, "Pred Yes increased ($7,443M → $9,963M) — cherry-picks subset; gold No total decreased."),
    ("doc147_qa0", 0.0,  "36.36 vs 42.69 — 14.8% diff."),
    ("doc148_qa0", 0.0,  "Refused (FY19 OI not provided); gold 0.2%."),
    ("doc149_qa0", 0.0,  "4.0% vs 6.2% — 35% diff."),
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
