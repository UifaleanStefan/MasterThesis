"""Manual Claude judging — FB bm25-corpus BATCH cell. recall=0.700."""
from __future__ import annotations
import json
from pathlib import Path

JUDGE_DIR = Path("results/stage3/judge_queue/financebench__bm25-corpus__batch__seed42")
QID_PREFIX = "financebench__bm25-corpus__batch__"
QID_SUFFIX = "__seed42"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc0_qa0",   1.0,  "$1,577M = gold."),
    ("doc1_qa0",   0.0,  "Refused; gold $8.70B."),
    ("doc2_qa0",   0.0,  "Y/N flip: Yes vs No."),
    ("doc3_qa0",   0.0,  "Refused; gold lists drivers."),
    ("doc4_qa0",   0.0,  "Pred 'Combat Arms litigation costs' (not a segment) vs gold Consumer segment."),
    ("doc5_qa0",   0.0,  "Refused; gold No 0.96."),
    ("doc6_qa0",   1.0,  "Same 3 notes (no MMM tickers but the notes themselves)."),
    ("doc7_qa0",   1.0,  "Yes 65 years = gold."),
    ("doc8_qa0",   0.0,  "Refused; gold 24.26."),
    ("doc9_qa0",   0.0,  "6.0% vs 1.9% — 3× too high."),
    ("doc10_qa0",  0.0,  "Refused; gold 0.66."),
    ("doc11_qa0",  0.0,  "Refused; gold 65.4%."),
    ("doc12_qa0",  0.0,  "Refused; gold 0.83."),
    ("doc13_qa0",  0.0,  "Refused; gold No declined."),
    ("doc14_qa0",  0.0,  "Refused; gold Yes improved."),
    ("doc15_qa0",  1.0,  "0 = 0."),
    ("doc16_qa0",  0.0,  "11.4 vs 9.5 — 20% off."),
    ("doc17_qa0",  0.0,  "-4.30% vs -0.02 — way off."),
    ("doc18_qa0",  0.0,  "Refused; gold 93.86."),
    ("doc19_qa0",  0.0,  "Refused; gold 30.8%."),
    ("doc20_qa0",  1.0,  "11,588 = $11,588M exact."),
    ("doc21_qa0",  0.0,  "Refused; gold $1,616M."),
    ("doc22_qa0",  1.0,  "Substance matches July 1 supplemental indentures."),
    ("doc23_qa0",  0.0,  "Refused; gold improved 0.67 → 0.69."),
    ("doc24_qa0",  0.75, "Shanghai + NZ in FY23, Czech in FY22; gold has all 3 in FY23 — partial match."),
    ("doc25_qa0",  1.0,  "Packaging = gold."),
    ("doc26_qa0",  0.0,  "Refused; gold definitive No 0.8% decline."),
    ("doc27_qa0",  0.5,  "Employee + fixed asset + other costs; missing 87% employee."),
    ("doc28_qa0",  1.0,  "$2,018M exact."),
    ("doc29_qa0",  0.0,  "-5% vs flat."),
    ("doc30_qa0",  0.0,  "Refused; gold 4.2%."),
    ("doc31_qa0",  0.0,  "Refused; gold Yes 1.57."),
    ("doc32_qa0",  1.0,  "AMD product list matches."),
    ("doc33_qa0",  1.0,  "EPYC + semi-custom + Xilinx."),
    ("doc34_qa0",  0.0,  "Refused; gold Xilinx amortization."),
    ("doc35_qa0",  1.0,  "Operations $3,565M."),
    ("doc36_qa0",  0.0,  "Gaming vs gold Data Center."),
    ("doc37_qa0",  1.0,  "Yes 16% concentration."),
    ("doc38_qa0",  0.0,  "Common Shares (equity) vs gold no debt securities."),
    ("doc39_qa0",  1.0,  "US/EMEA/APAC/LACC = gold."),
    ("doc40_qa0",  0.0,  "Refused; gold definitive 'OM not measured'."),
    ("doc41_qa0",  0.0,  "Refused; gold definitive 'GM not measured'."),
    ("doc42_qa0",  1.0,  "24.6 → 21.6 exact."),
    ("doc43_qa0",  1.0,  "Customer deposits $110,239M = gold."),
    ("doc44_qa0",  1.0,  "Yes retention high = gold."),
    ("doc45_qa0",  1.0,  "$0.389B vs $0.40B — within 5%."),
    ("doc46_qa0",  1.0,  "1,832 = gold exact."),
    ("doc47_qa0",  0.25, "Y/N flip: 'has positive WC' (wrong) but body shows -$1,561M = gold negative."),
    ("doc48_qa0",  0.0,  "Refused; gold 2.8%."),
    ("doc49_qa0",  0.0,  "Refused; gold $5,409M."),
    ("doc50_qa0",  0.0,  "Y/N flip: 'fluctuated >2%' vs gold 'consistent'."),
    ("doc51_qa0",  1.0,  "Current Health + Yardbird FY22 = gold."),
    ("doc52_qa0",  1.0,  "Operations $1,824M = gold."),
    ("doc53_qa0",  1.0,  "Yes drop $1,874 → $1,093 = gold ~42%."),
    ("doc54_qa0",  1.0,  "982 → 969 = gold exact."),
    ("doc55_qa0",  1.0,  "Entertainment 9% gaming = gold."),
    ("doc56_qa0",  0.0,  "Refused; gold 1.73."),
    ("doc57_qa0",  0.0,  "Refused; gold 101.5%."),
    ("doc58_qa0",  0.0,  "$1,000M vs $382M — 162% off."),
    ("doc59_qa0",  0.0,  "Refused; gold $12,645M."),
    ("doc60_qa0",  0.5,  "Only Commercial; gold also names Defense."),
    ("doc61_qa0",  1.0,  "Lion Air + Ethiopian = gold."),
    ("doc62_qa0",  0.0,  "Refused; gold Yes improving 4.8 → 5.3."),
    ("doc63_qa0",  0.5,  "Only 'limited commercial airlines'; gold also US gov 40%."),
    ("doc64_qa0",  1.0,  "Yes cyclical = gold."),
    ("doc65_qa0",  1.0,  "787 + 737 + 777X resume = gold."),
    ("doc66_qa0",  0.0,  "(31)% vs (743)% — pred references income tax expense numbers, not ETR percentages; gold 0.62 vs -14.76."),
    ("doc67_qa0",  1.0,  "1.42% = 0.0142 ≈ gold 0.01 (rounded to 2 decimals). Correct calc + correct value."),
    ("doc68_qa0",  1.0,  "39.7% = gold exact."),
    ("doc69_qa0",  0.0,  "0.98 vs 0.8 — 22.5% off."),
    ("doc70_qa0",  0.0,  "Refused; gold 63.86."),
    ("doc71_qa0",  0.0,  "13.3% vs 10.3% — 29% off."),
    ("doc72_qa0",  1.0,  "20% → 23% exact = gold."),
    ("doc73_qa0",  0.0,  "Refused; gold Yes $831M."),
    ("doc74_qa0",  1.0,  "$59,268M = gold exact."),
    ("doc75_qa0",  0.0,  "Refused; gold 17.98."),
    ("doc76_qa0",  1.0,  "Yes capital intensive = gold."),
    ("doc77_qa0",  0.75, "U&C litigation only; gold lists multiple."),
    ("doc78_qa0",  0.75, "Yes paid; missing $0.55 specific."),
    ("doc79_qa0",  1.0,  "Mary Dillon ex-Ulta = gold."),
    ("doc80_qa0",  1.0,  "Richard A. Johnson = gold."),
    ("doc81_qa0",  0.0,  "Refused; gold -3.7."),
    ("doc82_qa0",  1.0,  "0.68 = gold exact."),
    ("doc83_qa0",  1.0,  "$3,189.4M vs $3,215M — 0.8% within 5%."),
    ("doc84_qa0",  0.0,  "0.24 vs 0.54 — 56% off."),
    ("doc85_qa0",  1.0,  "No 1.3% growth = gold."),
    ("doc86_qa0",  0.0,  "Refused; gold names drivers."),
    ("doc87_qa0",  0.0,  "Refused; gold 2.7x."),
    ("doc88_qa0",  0.0,  "Y/N flip: Yes accelerate +12.5% vs gold No decelerate."),
    ("doc89_qa0",  1.0,  "US +3.0% intl -0.6% = gold exact."),
    ("doc90_qa0",  1.0,  "Consumer Health = gold."),
    ("doc91_qa0",  1.0,  "~$20B exact = gold."),
    ("doc92_qa0",  1.0,  "$13.2B exact = gold."),
    ("doc93_qa0",  0.75, "Yes increased — correct direction but lacks specific 20.0 → 20.1 magnitudes."),
    ("doc94_qa0",  0.0,  "Commercial Banking vs gold Corporate."),
    ("doc95_qa0",  0.0,  "Refused; gold $66.56."),
    ("doc96_qa0",  1.0,  "GM not relevant for financial firm = gold."),
    ("doc97_qa0",  0.0,  "CCB vs gold CIB."),
    ("doc98_qa0",  1.0,  "Yes VaR decreased $7M = gold."),
    ("doc99_qa0",  0.0,  "2.06 vs 6.25 — 67% off."),
    ("doc100_qa0", 0.0,  "Refused; gold 1.33."),
    ("doc101_qa0", 1.0,  "$5,818M = gold exact."),
    ("doc102_qa0", 1.0,  "0.4% = 0.4% exact."),
    ("doc103_qa0", 1.0,  "$302.6M vs $303M — within 5%."),
    ("doc104_qa0", 0.0,  "Refused; gold 7.9%."),
    ("doc105_qa0", 1.0,  "$0.01/share = gold."),
    ("doc106_qa0", 1.0,  "Las Vegas Strip = gold."),
    ("doc107_qa0", 0.0,  "2.42 vs 0 (negative EBIT)."),
    ("doc108_qa0", 1.0,  "MGM China = gold."),
    ("doc109_qa0", 1.0,  "Corporate bonds = gold."),
    ("doc110_qa0", 0.0,  "Refused; gold $32,780M."),
    ("doc111_qa0", 0.25, "Y/N flip: 'Yes' but body cites long-term decreased $47,032M → $41,990M (matches gold)."),
    ("doc112_qa0", 0.0,  "Refused; gold 5.4%."),
    ("doc113_qa0", 1.0,  "$5,466.3M = gold."),
    ("doc114_qa0", 0.0,  "Refused; gold 55.1%."),
    ("doc115_qa0", 1.0,  "$16,525M = gold."),
    ("doc116_qa0", 0.0,  "1.73 vs 3.46 — exactly half (50% off)."),
    ("doc117_qa0", 1.0,  "Operations $5,841M = gold."),
    ("doc118_qa0", 0.0,  "Refused; gold Yes $1.6B."),
    ("doc119_qa0", 1.0,  "$4.625B vs $4.60B — within 5%."),
    ("doc120_qa0", 0.25, "Only 4 regions (US/Canada/LatAm/Europe); gold has 10."),
    ("doc121_qa0", 0.75, "Litigation but no material — matches gold's 'No material'."),
    ("doc122_qa0", 1.0,  "411 = $411M."),
    ("doc123_qa0", 0.0,  "$12,275M vs $9,068M — 35% off."),
    ("doc124_qa0", 0.0,  "14.5% vs 16.5% — 12% off."),
    ("doc125_qa0", 1.0,  "Defeated = gold."),
    ("doc126_qa0", 1.0,  "$400M = gold."),
    ("doc127_qa0", 1.0,  "$8,400M exact = gold."),
    ("doc128_qa0", 1.0,  "Strong start + resilience = gold."),
    ("doc129_qa0", 1.0,  "1pp = gold."),
    ("doc130_qa0", 0.25, "Yes (correct Y/N) but cites Net Income (wrong concept)."),
    ("doc131_qa0", 0.0,  "Refused; gold Yes Consumer Healthcare JV gain."),
    ("doc132_qa0", 0.5,  "Trillium + Array + Upjohn (Upjohn was spinoff); gold says Therachon. 2/3."),
    ("doc133_qa0", 0.0,  "$700M vs $77.78M — 9× too high."),
    ("doc134_qa0", 1.0,  "Developed Rest of World = gold."),
    ("doc135_qa0", 1.0,  "Yes Upjohn separation = gold."),
    ("doc136_qa0", 1.0,  "'None.' = gold 'There are none'."),
    ("doc137_qa0", 0.75, "Refusal aligns with 'none'."),
    ("doc138_qa0", 1.0,  "Lower marketing + incentive comp leverage = gold."),
    ("doc139_qa0", 1.0,  "47 new stores + brand launches = gold."),
    ("doc140_qa0", 1.0,  "36.5% vs 36% — within 5%."),
    ("doc141_qa0", 0.0,  "Y/N flip: Decrease vs increased."),
    ("doc142_qa0", 1.0,  "Cross currency swaps = gold."),
    ("doc143_qa0", 0.75, "$1,097M pension only; gold has pension + health/life."),
    ("doc144_qa0", 0.0,  "Refused; gold No 0.54."),
    ("doc145_qa0", 1.0,  "Yes capital intensive = gold."),
    ("doc146_qa0", 0.25, "Y/N flip: 'Yes' + body 'decreased $150,868 → $150,639' = gold decrease."),
    ("doc147_qa0", 0.0,  "Refused; gold 42.69."),
    ("doc148_qa0", 0.0,  "-0.5% vs 0.2% — wrong sign + magnitude."),
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
