"""Manual Claude judging — FB attention-corpus-tuned BATCH_CALIB cell."""
from __future__ import annotations
import json
from pathlib import Path

JUDGE_DIR = Path("results/stage3/judge_queue/financebench__attention-corpus-tuned__batch_calib__seed42")
QID_PREFIX = "financebench__attention-corpus-tuned__batch_calib__"
QID_SUFFIX = "__seed42"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc0_qa0",   1.0,  "$1,501M vs $1577M — 4.8% within 5%."),
    ("doc1_qa0",   1.0,  "8.738B vs $8.70B — 0.4% diff."),
    ("doc2_qa0",   0.0,  "Y/N flip: Yes vs No."),
    ("doc3_qa0",   1.0,  "Lists matching drivers."),
    ("doc4_qa0",   1.0,  "Consumer segment = gold."),
    ("doc5_qa0",   0.0,  "Refused; gold No 0.96."),
    ("doc6_qa0",   1.0,  "Same 3 notes."),
    ("doc7_qa0",   1.0,  "Yes 65 years = gold."),
    ("doc8_qa0",   0.0,  "2.56 vs 24.26 — way off."),
    ("doc9_qa0",   0.0,  "3.5% vs 1.9%."),
    ("doc10_qa0",  0.0,  "1.90 vs 0.66 — 188% off."),
    ("doc11_qa0",  0.75, "Mid-calc shows correct 65.4% computation but truncated."),
    ("doc12_qa0",  0.0,  "1.25 vs 0.83 — 51% off."),
    ("doc13_qa0",  0.0,  "Y/N flip: Yes improving vs No declined."),
    ("doc14_qa0",  0.0,  "Refused; gold Yes improved."),
    ("doc15_qa0",  1.0,  "0 = 0."),
    ("doc16_qa0",  0.0,  "12.0 vs 9.5 — 26% off."),
    ("doc17_qa0",  0.0,  "-1.42 vs -0.02 — 71× too large."),
    ("doc18_qa0",  0.0,  "29.12 vs 93.86 — 69% off."),
    ("doc19_qa0",  1.0,  "30.7% vs 30.8% — within 5%."),
    ("doc20_qa0",  1.0,  "$11,588 = gold exact."),
    ("doc21_qa0",  1.0,  "$1,615.9M = gold $1,616M — 0.1% diff."),
    ("doc22_qa0",  1.0,  "Substance matches + correct July 1 date."),
    ("doc23_qa0",  0.0,  "Refused; gold improved 0.67 → 0.69."),
    ("doc24_qa0",  0.0,  "Refused; gold lists Czech/Shanghai/NZ."),
    ("doc25_qa0",  1.0,  "Packaging = gold."),
    ("doc26_qa0",  1.0,  "Declining gross margin profile = gold."),
    ("doc27_qa0",  0.5,  "Employee + fixed asset costs ($93M); missing 87% employee."),
    ("doc28_qa0",  1.0,  "$2,018M = gold exact."),
    ("doc29_qa0",  0.0,  "-5% vs flat."),
    ("doc30_qa0",  1.0,  "4.18% vs 4.2% — within 5%."),
    ("doc31_qa0",  0.0,  "Refused; gold definitive Yes 1.57."),
    ("doc32_qa0",  1.0,  "AMD product list matches."),
    ("doc33_qa0",  1.0,  "EPYC + semi-custom + Xilinx = gold drivers."),
    ("doc34_qa0",  1.0,  "Xilinx amortization exact."),
    ("doc35_qa0",  1.0,  "Operations $3,565M = gold operations."),
    ("doc36_qa0",  0.0,  "Gaming +21% vs gold Data Center."),
    ("doc37_qa0",  1.0,  "Yes 16% concentration."),
    ("doc38_qa0",  0.0,  "Common Shares (equity) vs gold no debt."),
    ("doc39_qa0",  1.0,  "US/EMEA/APAC/LACC = gold."),
    ("doc40_qa0",  1.0,  "OM not useful = gold."),
    ("doc41_qa0",  1.0,  "GM not useful = gold."),
    ("doc42_qa0",  1.0,  "24.6 → 21.6 = gold exact."),
    ("doc43_qa0",  0.0,  "Long-term debt $42,573M vs gold Customer deposits."),
    ("doc44_qa0",  1.0,  "Yes retention high = gold."),
    ("doc45_qa0",  1.0,  "0.389B vs $0.40B — within 5%."),
    ("doc46_qa0",  1.0,  "1,829 vs $1,832 — 0.2% diff."),
    ("doc47_qa0",  0.25, "Y/N flip: pred Yes positive WC but body computes -$1,561M = gold negative magnitude. Self-contradicts."),
    ("doc48_qa0",  0.0,  "3.9% vs 2.8% — 39% off."),
    ("doc49_qa0",  1.0,  "5,409 = $5,409M exact."),
    ("doc50_qa0",  0.0,  "Y/N flip: 'fluctuated >2%' vs gold 'consistent'."),
    ("doc51_qa0",  1.0,  "Current Health + Yardbird FY22 = gold."),
    ("doc52_qa0",  1.0,  "Operations $1,824M = gold."),
    ("doc53_qa0",  1.0,  "Yes drop $1,874M → $1,093M = gold ~42%."),
    ("doc54_qa0",  0.5,  "Pred 930 → 907 vs gold 982 → 969 — direction right, numbers off."),
    ("doc55_qa0",  0.75, "Gaming +9% matches Entertainment +9% (Gaming is sub-driver of Entertainment)."),
    ("doc56_qa0",  1.0,  "1.74 vs 1.73 — within 5%."),
    ("doc57_qa0",  1.0,  "102.0% vs 101.5% — within 5%."),
    ("doc58_qa0",  1.0,  "$381.6M vs $382M — 0.1% diff."),
    ("doc59_qa0",  1.0,  "$12,645 = gold exact."),
    ("doc60_qa0",  0.5,  "Only Commercial; gold also names Defense >20%."),
    ("doc61_qa0",  1.0,  "Lion Air + Ethiopian = gold."),
    ("doc62_qa0",  0.0,  "Refused (GM not useful); gold definitive Yes improving."),
    ("doc63_qa0",  0.5,  "Only 'limited commercial airlines'; gold also US gov 40%."),
    ("doc64_qa0",  1.0,  "Yes cyclical = gold."),
    ("doc65_qa0",  1.0,  "787 + 737 + 777X resume — matches gold's three."),
    ("doc66_qa0",  0.25, "Pred 'lower' wrong direction (gold went -14.76 → 0.62 = INCREASE)."),
    ("doc67_qa0",  1.0,  "1.43% = 0.0143 ≈ gold 0.01 (rounded to 2 decimals)."),
    ("doc68_qa0",  1.0,  "39.7% = gold exact."),
    ("doc69_qa0",  1.0,  "0.80 = 0.8 exact."),
    ("doc70_qa0",  1.0,  "Mid-calc shows DPO ≈ 63.86 (gold). Correct work."),
    ("doc71_qa0",  1.0,  "10.5% vs 10.3% — within 5%."),
    ("doc72_qa0",  1.0,  "20% → 23% = gold exact."),
    ("doc73_qa0",  0.5,  "Yes $2,278M vs gold Yes $831M — direction right, magnitude off."),
    ("doc74_qa0",  1.0,  "59,268 = $59,268M exact."),
    ("doc75_qa0",  0.0,  "8.99 vs 17.98 — exactly half (50% off)."),
    ("doc76_qa0",  1.0,  "Yes capital intensive = gold."),
    ("doc77_qa0",  0.75, "U&C litigation only; gold lists multiple."),
    ("doc78_qa0",  1.0,  "Yes $0.55/share = gold."),
    ("doc79_qa0",  1.0,  "Mary Dillon ex-Ulta = gold."),
    ("doc80_qa0",  1.0,  "Richard A. Johnson = gold."),
    ("doc81_qa0",  0.0,  "66.73 vs -3.7 — wrong sign and magnitude."),
    ("doc82_qa0",  1.0,  "0.69 vs 0.68 — within 5%."),
    ("doc83_qa0",  1.0,  "$3,115.4M vs $3,215M — 3.1% within 5%."),
    ("doc84_qa0",  1.0,  "0.55 vs 0.54 — within 5%."),
    ("doc85_qa0",  1.0,  "No 1.3% growth = gold."),
    ("doc86_qa0",  0.0,  "Pred refuses (GM not useful); gold lists drivers."),
    ("doc87_qa0",  0.0,  "Refused; gold 2.7x."),
    ("doc88_qa0",  0.0,  "Y/N flip + magnitude: Yes accelerate +12.5% vs gold No decelerate."),
    ("doc89_qa0",  1.0,  "US +3.0% intl -0.6% = gold exact."),
    ("doc90_qa0",  1.0,  "Consumer Health = gold."),
    ("doc91_qa0",  1.0,  "~$20B = gold."),
    ("doc92_qa0",  1.0,  "$13.2B = gold."),
    ("doc93_qa0",  1.0,  "Yes 20.0 → 20.1 = gold."),
    ("doc94_qa0",  0.0,  "CCB vs gold Corporate."),
    ("doc95_qa0",  0.0,  "$239.45 vs gold $66.56 — 3.6× too high."),
    ("doc96_qa0",  1.0,  "GM not relevant for financial firm = gold."),
    ("doc97_qa0",  0.0,  "CCB vs gold CIB."),
    ("doc98_qa0",  1.0,  "Yes VaR decreased $7M = gold."),
    ("doc99_qa0",  0.0,  "3.06 vs 6.25 — 51% off."),
    ("doc100_qa0", 1.0,  "1.38 vs 1.33 — within 5%."),
    ("doc101_qa0", 1.0,  "5,818M = gold exact."),
    ("doc102_qa0", 1.0,  "0.4% = gold exact."),
    ("doc103_qa0", 1.0,  "$302.578M vs $303M — within 5%."),
    ("doc104_qa0", 0.0,  "-3.5% vs 7.9% — wrong sign + magnitude."),
    ("doc105_qa0", 1.0,  "$0.01/share = gold."),
    ("doc106_qa0", 1.0,  "Las Vegas Strip = gold."),
    ("doc107_qa0", 0.0,  "1.61 vs 0 (negative EBIT)."),
    ("doc108_qa0", 1.0,  "MGM China -$203,136 = gold MGM China -44%."),
    ("doc109_qa0", 1.0,  "Corporate bonds = gold."),
    ("doc110_qa0", 1.0,  "$32,780 = gold exact."),
    ("doc111_qa0", 0.25, "Y/N flip: 'Yes' but body shows long-term debt decreased (matches gold direction)."),
    ("doc112_qa0", 0.0,  "4.51% vs 5.4% — 16.5% off."),
    ("doc113_qa0", 1.0,  "5,466.3M vs $5,466M — 0.005% diff."),
    ("doc114_qa0", 1.0,  "56.3% vs 55.1% — within 5%."),
    ("doc115_qa0", 1.0,  "16,525 = $16,525M exact."),
    ("doc116_qa0", 1.0,  "3.61 vs 3.46 — within 5%."),
    ("doc117_qa0", 1.0,  "Operations $5,841M = gold."),
    ("doc118_qa0", 0.5,  "Yes positive WC ($57,517M-$45,101M=$12,416M) vs gold Yes $1.6B — direction right, magnitude off."),
    ("doc119_qa0", 1.0,  "$4.625B vs $4.60B — within 5%."),
    ("doc120_qa0", 0.5,  "Lists US/Dev Europe/Dev RoW/Emerging (reporting segments); gold lists 10 actual geographies."),
    ("doc121_qa0", 0.75, "Litigation but says no material — matches gold's 'No material'."),
    ("doc122_qa0", 1.0,  "411 = $411M."),
    ("doc123_qa0", 0.0,  "$14,275M vs $9,068M — 57% off (returns gross EBITDA)."),
    ("doc124_qa0", 0.75, "Truncated mid-calc but $14,275M / $86,392M = 16.5% — matches gold."),
    ("doc125_qa0", 1.0,  "Defeated = gold."),
    ("doc126_qa0", 1.0,  "$400M = gold."),
    ("doc127_qa0", 0.0,  "$9,400M (sum error) vs gold $8,400M — 12% off (arithmetic mistake adding 4.2+4.2 = 9.4)."),
    ("doc128_qa0", 1.0,  "Strong start + resilience = gold."),
    ("doc129_qa0", 1.0,  "1pp = gold."),
    ("doc130_qa0", 0.25, "Y/N correct (Yes) but cites Net Income (wrong concept; should be PPNE)."),
    ("doc131_qa0", 0.5,  "Names JV (correct event) but cites $(6)M (wrong amount); gold confirms JV gain."),
    ("doc132_qa0", 0.5,  "Trillium + Array correct; Upjohn was spinoff not acquisition. 2/3."),
    ("doc133_qa0", 0.0,  "$700M vs $77.78M — 9× too high."),
    ("doc134_qa0", 1.0,  "Developed Rest of World = gold."),
    ("doc135_qa0", 1.0,  "Yes Upjohn separation = gold."),
    ("doc136_qa0", 0.0,  "Common stock NASDAQ (equity) vs gold no debt."),
    ("doc137_qa0", 0.75, "Refusal aligns with 'none' but lacks confident statement."),
    ("doc138_qa0", 1.0,  "Lower marketing + incentive comp leverage = gold."),
    ("doc139_qa0", 1.0,  "47 new stores + brand launches = gold."),
    ("doc140_qa0", 1.0,  "36.5% vs 36% — within 5%."),
    ("doc141_qa0", 0.0,  "Y/N flip: Decrease vs gold increased."),
    ("doc142_qa0", 1.0,  "Cross currency swaps = gold."),
    ("doc143_qa0", 0.75, "$1,097M pension only; gold has pension + health/life $862M."),
    ("doc144_qa0", 0.0,  "Refused; gold definitive No 0.54."),
    ("doc145_qa0", 1.0,  "Yes capital intensive = gold."),
    ("doc146_qa0", 0.25, "Y/N flip: pred 'Yes' but body cites total debt decreased $150,868M → $150,639M (matches gold's small decrease)."),
    ("doc147_qa0", 1.0,  "Mid-calc shows DPO 42.52 vs gold 42.69 — within 5%."),
    ("doc148_qa0", 0.0,  "-6.0% vs 0.2% — wrong sign and magnitude."),
    ("doc149_qa0", 0.0,  "3.8% vs 6.2% — 39% off."),
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
