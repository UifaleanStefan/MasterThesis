"""Manual Claude judging — FB bm25-corpus ONLINE cell.

BM25 sparse retrieval on corpus mode. Recall@k=0.913 — high (BM25 is a
strong retrieval baseline). Expected mean ~0.60-0.65.

Per AGENTS.md §0 + evaluation/claude_judge_protocol.md §HARD RULE: every
judge_score below was produced by Claude reading the (question, gold,
predicted) triple in this session and applying the rubric manually.
"""
from __future__ import annotations
import json
from pathlib import Path

JUDGE_DIR = Path("results/stage3/judge_queue/financebench__bm25-corpus__online__seed42")
QID_PREFIX = "financebench__bm25-corpus__online__"
QID_SUFFIX = "__seed42"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc0_qa0",   1.0,  "$1,577M = gold."),
    ("doc1_qa0",   1.0,  "8.738B vs $8.70B — 0.4% diff."),
    ("doc2_qa0",   0.0,  "Y/N flip: Yes vs No."),
    ("doc3_qa0",   1.0,  "Lists matching drivers (litigation/PFAS/Russia/restructuring)."),
    ("doc4_qa0",   1.0,  "Consumer = gold."),
    ("doc5_qa0",   0.0,  "Refused; gold definitive No 0.96."),
    ("doc6_qa0",   1.0,  "Same 3 notes (MMM26/30/31)."),
    ("doc7_qa0",   1.0,  "Yes 65 consecutive years = gold."),
    ("doc8_qa0",   0.25, "25.73 vs 24.26 — 6.1% just outside 5%."),
    ("doc9_qa0",   0.0,  "3.5% vs 1.9% — 84% off."),
    ("doc10_qa0",  1.0,  "0.66 = 0.66 exact (BM25 nailed Adobe OCF)."),
    ("doc11_qa0",  1.0,  "65.4% exact."),
    ("doc12_qa0",  1.0,  "0.83 = 0.83 exact."),
    ("doc13_qa0",  0.0,  "Y/N flip: Yes improving vs No declined."),
    ("doc14_qa0",  0.0,  "Refused; gold Yes improved."),
    ("doc15_qa0",  1.0,  "0 = 0."),
    ("doc16_qa0",  0.0,  "12.0 vs 9.5 — 26% off."),
    ("doc17_qa0",  0.0,  "-1.42 vs -0.02 — 71× too large."),
    ("doc18_qa0",  0.0,  "36.12 vs 93.86 — 62% off."),
    ("doc19_qa0",  1.0,  "30.8% = 30.8% exact."),
    ("doc20_qa0",  1.0,  "11,588 = $11,588M exact."),
    ("doc21_qa0",  1.0,  "$1,615.9M vs $1,616M — 0.1% diff."),
    ("doc22_qa0",  1.0,  "Substance matches + correct July 1 date."),
    ("doc23_qa0",  0.0,  "Refused (not in doc); gold improved 0.67 → 0.69."),
    ("doc24_qa0",  0.75, "Shanghai + Czech; wrong years (FY23 vs gold FY22 for Czech); missing NZ acquisition."),
    ("doc25_qa0",  1.0,  "Packaging = gold."),
    ("doc26_qa0",  1.0,  "Pred 18.5% (FY23) vs 19.4% (FY22) declining — matches gold's 0.8% decline."),
    ("doc27_qa0",  0.5,  "Employee + costs $93M; missing 87% employee split."),
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
    ("doc45_qa0",  1.0,  "0.389B vs $0.40B — within 5%."),
    ("doc46_qa0",  1.0,  "1,829 vs $1,832 — 0.2% diff."),
    ("doc47_qa0",  1.0,  "Pred 'does not have positive WC' (correct direction) + cites $1,250-$2,811=-$1,561M = gold magnitude (-$1,561M)."),
    ("doc48_qa0",  0.0,  "3.1% vs 2.8% — 10.7% off."),
    ("doc49_qa0",  1.0,  "5,409 = $5,409M exact."),
    ("doc50_qa0",  0.0,  "Y/N flip: 'not consistent' vs gold 'consistent'."),
    ("doc51_qa0",  1.0,  "Current Health + Yardbird FY22 = gold."),
    ("doc52_qa0",  1.0,  "Operations $1,824M = gold operations $1.8bn."),
    ("doc53_qa0",  1.0,  "Yes drop $1,874M → $1,093M = gold ~42%."),
    ("doc54_qa0",  0.5,  "Pred 930 → 907 vs gold 982 → 969 — direction right, numbers off."),
    ("doc55_qa0",  1.0,  "Entertainment +9% gaming = gold (exact match)."),
    ("doc56_qa0",  1.0,  "1.74 vs 1.73 — within 5%."),
    ("doc57_qa0",  0.0,  "Refused; gold 101.5%."),
    ("doc58_qa0",  0.0,  "$1,000 vs $382 — 162% off (Block FY20 OCF)."),
    ("doc59_qa0",  1.0,  "$12,645 = gold exact."),
    ("doc60_qa0",  0.5,  "Only Commercial Airplanes; gold also names Defense >20%."),
    ("doc61_qa0",  1.0,  "Lion Air + Ethiopian = gold."),
    ("doc62_qa0",  0.0,  "Y/N flip: 'does not have improving' vs gold Yes improving 4.8 → 5.3."),
    ("doc63_qa0",  0.5,  "Only 'limited number of commercial airlines'; gold also names US gov 40%."),
    ("doc64_qa0",  1.0,  "Yes cyclical = gold."),
    ("doc65_qa0",  1.0,  "787 + 737 + 777X resume = gold."),
    ("doc66_qa0",  0.0,  "'0.6% lower' — gold 0.62% (FY22) vs -14.76% (FY21) is an INCREASE in ETR (less negative), not a decrease. Wrong direction."),
    ("doc67_qa0",  1.0,  "0.01 = 0.01 exact (BM25 nailed Coca-Cola ROA)."),
    ("doc68_qa0",  1.0,  "Computes 39.7% from $15,357M/$38,655M = gold 39.7% exact."),
    ("doc69_qa0",  0.0,  "0.33 vs 0.8 — 59% off."),
    ("doc70_qa0",  0.0,  "Refused; gold 63.86."),
    ("doc71_qa0",  0.0,  "14.9% vs 10.3% — 45% off."),
    ("doc72_qa0",  1.0,  "20% → 23% exact = gold."),
    ("doc73_qa0",  0.0,  "Refused; gold Yes $831M."),
    ("doc74_qa0",  1.0,  "59,268 = $59,268M exact."),
    ("doc75_qa0",  0.0,  "Refused (not provided); gold 17.98."),
    ("doc76_qa0",  1.0,  "Yes capital intensive = gold."),
    ("doc77_qa0",  1.0,  "Drug pricing + rebate + U&C litigation — matches gold's multiple disputes."),
    ("doc78_qa0",  1.0,  "Yes $0.55/share = gold."),
    ("doc79_qa0",  1.0,  "Mary Dillon ex-Ulta = gold."),
    ("doc80_qa0",  1.0,  "Richard A. Johnson = gold."),
    ("doc81_qa0",  0.0,  "36.73 vs -3.7 — wrong sign + magnitude."),
    ("doc82_qa0",  1.0,  "0.68 = 0.68 exact."),
    ("doc83_qa0",  1.0,  "$3,215.4M vs $3,215M — 0.01% diff."),
    ("doc84_qa0",  0.0,  "0.44 vs 0.54 — 18.5% off."),
    ("doc85_qa0",  1.0,  "No 1.3% growth = gold."),
    ("doc86_qa0",  1.0,  "Lists matching drivers (COVID exit + currency + commodity)."),
    ("doc87_qa0",  0.0,  "89.9 vs 2.7 — wildly wrong (used $346M as average inventory, dramatically wrong calc)."),
    ("doc88_qa0",  0.0,  "Y/N flip + magnitude: Yes accelerate +3.5% vs gold No decelerate."),
    ("doc89_qa0",  1.0,  "US +3.0% intl -0.6% = gold exact."),
    ("doc90_qa0",  1.0,  "Consumer Health = gold."),
    ("doc91_qa0",  1.0,  "~$20B exact = gold."),
    ("doc92_qa0",  1.0,  "$13.2B exact = gold."),
    ("doc93_qa0",  1.0,  "Yes 20.0 → 20.1 = gold."),
    ("doc94_qa0",  0.0,  "CCB vs gold Corporate."),
    ("doc95_qa0",  1.0,  "$66.56 exact = gold."),
    ("doc96_qa0",  1.0,  "GM not relevant for financial firm = gold."),
    ("doc97_qa0",  0.0,  "CCB vs gold CIB."),
    ("doc98_qa0",  1.0,  "Yes VaR decreased $7M = gold."),
    ("doc99_qa0",  0.0,  "3.12 vs 6.25 — 50% off."),
    ("doc100_qa0", 0.0,  "Refused; gold 1.33."),
    ("doc101_qa0", 1.0,  "$5,818M = $5,818M exact."),
    ("doc102_qa0", 1.0,  "0.4% = 0.4% exact."),
    ("doc103_qa0", 1.0,  "$302.6M vs $303M — within 5%."),
    ("doc104_qa0", 0.0,  "12.5% vs 7.9% — 58% off."),
    ("doc105_qa0", 1.0,  "$0.01/share = gold."),
    ("doc106_qa0", 1.0,  "Las Vegas Strip = gold."),
    ("doc107_qa0", 0.0,  "2.43 vs 0 (negative EBIT)."),
    ("doc108_qa0", 1.0,  "MGM China = gold."),
    ("doc109_qa0", 1.0,  "Corporate bonds = gold."),
    ("doc110_qa0", 0.0,  "Refused; gold $32,780M."),
    ("doc111_qa0", 0.25, "Y/N flip: 'Yes increased' but body cites long-term decreased $47,032M → $41,990M (matches gold)."),
    ("doc112_qa0", 0.0,  "Refused; gold 5.4%."),
    ("doc113_qa0", 1.0,  "5,466.3M = $5,466M — 0.005% diff."),
    ("doc114_qa0", 1.0,  "56.2% vs 55.1% — within 5%."),
    ("doc115_qa0", 1.0,  "16,525 = $16,525M exact."),
    ("doc116_qa0", 0.0,  "4.37 vs 3.46 — 26% off."),
    ("doc117_qa0", 1.0,  "Operations $5,841M = gold operations highest."),
    ("doc118_qa0", 0.0,  "Refused; gold Yes $1.6B."),
    ("doc119_qa0", 1.0,  "4.625B vs $4.60B — within 5%."),
    ("doc120_qa0", 0.25, "Lists only 4 regions (US/Canada/LatAm/Europe); gold has 10 regions — major incomplete."),
    ("doc121_qa0", 0.75, "Describes litigation but says no material — matches gold's 'No material'."),
    ("doc122_qa0", 1.0,  "$411M = gold exact."),
    ("doc123_qa0", 0.0,  "$14,275M vs $9,068M — 57% off (pred returns gross EBITDA)."),
    ("doc124_qa0", 1.0,  "16.3% vs 16.5% — within 5%."),
    ("doc125_qa0", 1.0,  "Defeated = gold."),
    ("doc126_qa0", 1.0,  "$400M = gold."),
    ("doc127_qa0", 1.0,  "$8,400,000,000 exact = gold."),
    ("doc128_qa0", 1.0,  "Strong start + resilience = gold."),
    ("doc129_qa0", 1.0,  "1pp exact = gold."),
    ("doc130_qa0", 1.0,  "Correct PPNE $13,745M → $14,882M = gold Yes positive."),
    ("doc131_qa0", 1.0,  "Consumer Healthcare JV gain — matches gold (the specific $(8,107)M figure is the gain magnitude)."),
    ("doc132_qa0", 0.5,  "Trillium + Array + 'commercial stage biopharmaceutical (not named)' — 2/3 (Therachon missing)."),
    ("doc133_qa0", 0.0,  "$700M vs $77.78M — 9× too high."),
    ("doc134_qa0", 1.0,  "Developed Rest of World = gold."),
    ("doc135_qa0", 1.0,  "Yes Upjohn separation = gold."),
    ("doc136_qa0", 0.0,  "Common stock NASDAQ (equity) vs gold no debt securities."),
    ("doc137_qa0", 0.75, "Refusal aligns with 'none' but lacks confident statement."),
    ("doc138_qa0", 1.0,  "Lower marketing + incentive comp leverage = gold."),
    ("doc139_qa0", 1.0,  "47 new stores + brand launches = gold (exact match)."),
    ("doc140_qa0", 0.0,  "21.8% vs 36% — 40% off."),
    ("doc141_qa0", 0.0,  "Y/N flip: Decrease vs gold increased."),
    ("doc142_qa0", 1.0,  "Cross currency swaps = gold."),
    ("doc143_qa0", 0.75, "$1,097M pension only; gold has pension + health/life."),
    ("doc144_qa0", 0.0,  "Refused; gold definitive No 0.54."),
    ("doc145_qa0", 1.0,  "Yes capital intensive = gold."),
    ("doc146_qa0", 0.25, "Y/N flip: 'Yes' but body cites total debt decreased $150,868M → $150,639M (matches gold)."),
    ("doc147_qa0", 0.0,  "Refused; gold 42.69."),
    ("doc148_qa0", 0.0,  "Refused (FY19 OI not provided); gold 0.2%."),
    ("doc149_qa0", 0.0,  "10.5% vs 6.2% — 69% off."),
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
