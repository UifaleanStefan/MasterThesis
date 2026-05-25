"""Manual Claude judging — FB v4t-canonical BATCH_CALIB cell.

Recall@k=0.220 — catastrophic forgetting under canonical θ at corpus scale.
Mostly refusals; ~25 of 150 entries pass.
"""
from __future__ import annotations
import json
from pathlib import Path

JUDGE_DIR = Path("results/stage3/judge_queue/financebench__v4t-canonical__batch_calib__seed42")
QID_PREFIX = "financebench__v4t-canonical__batch_calib__"
QID_SUFFIX = "__seed42"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc0_qa0",   0.0,  "Refused; gold $1,577M."),
    ("doc1_qa0",   0.0,  "Refused; gold $8.70B."),
    ("doc2_qa0",   0.0,  "Refused; gold No."),
    ("doc3_qa0",   0.0,  "Refused; gold lists drivers."),
    ("doc4_qa0",   0.0,  "Pred Corporate and other vs gold Consumer."),
    ("doc5_qa0",   0.0,  "Refused; gold No 0.96."),
    ("doc6_qa0",   0.0,  "Refused; gold MMM26/30/31."),
    ("doc7_qa0",   1.0,  "Yes 65 years = gold."),
    ("doc8_qa0",   0.0,  "Refused; gold 24.26."),
    ("doc9_qa0",   0.0,  "Refused; gold 1.9%."),
    ("doc10_qa0",  0.0,  "Refused; gold 0.66."),
    ("doc11_qa0",  0.0,  "Refused; gold 65.4%."),
    ("doc12_qa0",  0.0,  "Refused; gold 0.83."),
    ("doc13_qa0",  0.0,  "Refused; gold No declined."),
    ("doc14_qa0",  0.0,  "Refused; gold Yes improved."),
    ("doc15_qa0",  1.0,  "0 = 0."),
    ("doc16_qa0",  0.0,  "Refused; gold 9.5x."),
    ("doc17_qa0",  0.0,  "Refused; gold -0.02."),
    ("doc18_qa0",  0.0,  "Refused; gold 93.86."),
    ("doc19_qa0",  0.0,  "Refused; gold 30.8%."),
    ("doc20_qa0",  0.0,  "Refused; gold $11,588M."),
    ("doc21_qa0",  0.0,  "Refused; gold $1,616M."),
    ("doc22_qa0",  0.0,  "Refused; gold details indentures."),
    ("doc23_qa0",  0.0,  "Refused; gold improved 0.67 → 0.69."),
    ("doc24_qa0",  0.0,  "Refused; gold lists Czech/Shanghai/NZ."),
    ("doc25_qa0",  0.0,  "Refused; gold packaging."),
    ("doc26_qa0",  0.0,  "Refused; gold definitive No 0.8% decline."),
    ("doc27_qa0",  0.0,  "Refused; gold 87% employee."),
    ("doc28_qa0",  0.0,  "Refused; gold $2,018M."),
    ("doc29_qa0",  0.0,  "Refused; gold flat."),
    ("doc30_qa0",  0.0,  "Refused; gold 4.2%."),
    ("doc31_qa0",  0.0,  "Refused; gold Yes 1.57."),
    ("doc32_qa0",  0.0,  "Refused; gold lists AMD products."),
    ("doc33_qa0",  0.0,  "Refused; gold names EPYC+semi-custom+Xilinx."),
    ("doc34_qa0",  0.0,  "Refused; gold Xilinx amortization."),
    ("doc35_qa0",  0.0,  "Refused; gold Operations."),
    ("doc36_qa0",  0.0,  "Refused; gold Data Center."),
    ("doc37_qa0",  0.0,  "Refused; gold Yes 16%."),
    ("doc38_qa0",  0.0,  "Refused; gold 'There are none'."),
    ("doc39_qa0",  0.0,  "Refused; gold US/EMEA/APAC/LACC."),
    ("doc40_qa0",  0.0,  "Refused; gold OM not measured."),
    ("doc41_qa0",  0.0,  "Refused; gold GM not measured."),
    ("doc42_qa0",  0.0,  "Refused; gold 24.6 → 21.6."),
    ("doc43_qa0",  0.0,  "Refused; gold Customer deposits."),
    ("doc44_qa0",  1.0,  "Yes retention high = gold."),
    ("doc45_qa0",  0.0,  "Refused; gold $0.40B."),
    ("doc46_qa0",  0.0,  "Refused; gold $1,832M."),
    ("doc47_qa0",  0.0,  "Refused; gold definitive No -$1,561M."),
    ("doc48_qa0",  0.0,  "Refused; gold 2.8%."),
    ("doc49_qa0",  0.0,  "Refused; gold $5,409M."),
    ("doc50_qa0",  0.0,  "Refused; gold Yes consistent."),
    ("doc51_qa0",  0.0,  "Refused; gold lists Current Health + Yardbird."),
    ("doc52_qa0",  1.0,  "Pred 'operating activities brought in most cash flow' = gold operations."),
    ("doc53_qa0",  0.0,  "Refused; gold Yes ~42% drop."),
    ("doc54_qa0",  1.0,  "Pred 982 → 969 = gold exact (Best Buy store change)."),
    ("doc55_qa0",  0.0,  "Refused; gold Entertainment +9%."),
    ("doc56_qa0",  0.0,  "Refused; gold 1.73."),
    ("doc57_qa0",  0.0,  "Refused; gold 101.5%."),
    ("doc58_qa0",  0.0,  "Refused; gold $382M."),
    ("doc59_qa0",  0.0,  "Refused; gold $12,645M."),
    ("doc60_qa0",  0.5,  "Pred only Commercial Airplanes; gold also names Defense >20%."),
    ("doc61_qa0",  1.0,  "Lion Air + Ethiopian = gold."),
    ("doc62_qa0",  0.0,  "Refused; gold Yes improving 4.8 → 5.3."),
    ("doc63_qa0",  0.5,  "Pred 'commercial airlines + government agencies + defense contractors'; gold limited airlines + US gov 40%."),
    ("doc64_qa0",  1.0,  "Yes cyclical = gold."),
    ("doc65_qa0",  0.75, "Pred 787 + 737; missing 777X (gold lists all 3)."),
    ("doc66_qa0",  0.0,  "Refused; gold 0.62% vs -14.76%."),
    ("doc67_qa0",  0.0,  "Refused; gold 0.01."),
    ("doc68_qa0",  0.0,  "Refused; gold 39.7%."),
    ("doc69_qa0",  0.0,  "Refused; gold 0.8."),
    ("doc70_qa0",  0.0,  "Refused; gold 63.86."),
    ("doc71_qa0",  0.0,  "Refused; gold 10.3%."),
    ("doc72_qa0",  0.0,  "Refused; gold 20% → 23%."),
    ("doc73_qa0",  0.0,  "Refused; gold Yes $831M."),
    ("doc74_qa0",  0.0,  "Pred $39,000M vs gold $59,268M — 34% off (incorrect figure)."),
    ("doc75_qa0",  0.0,  "Refused; gold 17.98."),
    ("doc76_qa0",  0.0,  "Refused; gold Yes capital intensive."),
    ("doc77_qa0",  0.0,  "Refused; gold lists CVS legal disputes."),
    ("doc78_qa0",  0.75, "Yes paid (correct direction) but lacks $0.55/share specific."),
    ("doc79_qa0",  1.0,  "Mary Dillon ex-Ulta = gold."),
    ("doc80_qa0",  1.0,  "Richard A. Johnson = gold."),
    ("doc81_qa0",  0.0,  "Refused; gold -3.7."),
    ("doc82_qa0",  0.0,  "Refused; gold 0.68."),
    ("doc83_qa0",  0.0,  "Refused; gold $3,215M."),
    ("doc84_qa0",  0.0,  "Refused; gold 0.54."),
    ("doc85_qa0",  1.0,  "No 1.3% growth = gold."),
    ("doc86_qa0",  0.0,  "Refused; gold names drivers."),
    ("doc87_qa0",  0.0,  "Refused; gold 2.7x."),
    ("doc88_qa0",  0.0,  "Refused; gold definitive No decelerate."),
    ("doc89_qa0",  0.25, "Pred US +1.3% intl 'unfavorable FX'; gold US +3.0% intl -0.6%."),
    ("doc90_qa0",  1.0,  "Consumer Health = gold."),
    ("doc91_qa0",  0.0,  "Refused; gold ~$20B."),
    ("doc92_qa0",  0.0,  "Pred $3.7B vs gold $13.2B — 72% off."),
    ("doc93_qa0",  0.0,  "Refused; gold Yes 20.0 → 20.1."),
    ("doc94_qa0",  0.0,  "Refused; gold Corporate."),
    ("doc95_qa0",  0.0,  "Refused; gold $66.56."),
    ("doc96_qa0",  1.0,  "GM not relevant for financial firm = gold."),
    ("doc97_qa0",  1.0,  "Corporate & Investment Bank = gold CIB."),
    ("doc98_qa0",  1.0,  "Yes VaR decreased $7M = gold."),
    ("doc99_qa0",  0.0,  "Refused; gold 6.25."),
    ("doc100_qa0", 0.0,  "Refused; gold 1.33."),
    ("doc101_qa0", 1.0,  "$5,818M = gold exact."),
    ("doc102_qa0", 0.0,  "Refused; gold 0.4%."),
    ("doc103_qa0", 0.0,  "Refused; gold $303M."),
    ("doc104_qa0", 0.0,  "Refused; gold 7.9%."),
    ("doc105_qa0", 1.0,  "$0.01/share annual = gold."),
    ("doc106_qa0", 0.0,  "Refused; gold Las Vegas ~90%."),
    ("doc107_qa0", 0.0,  "Refused; gold 0 (negative EBIT)."),
    ("doc108_qa0", 0.0,  "Refused; gold MGM China -44%."),
    ("doc109_qa0", 1.0,  "Corporate bonds = gold."),
    ("doc110_qa0", 0.0,  "Refused; gold $32,780M."),
    ("doc111_qa0", 0.0,  "Refused; gold No decreased $2.5bn."),
    ("doc112_qa0", 0.0,  "Refused; gold 5.4%."),
    ("doc113_qa0", 0.0,  "Refused; gold $5,466M."),
    ("doc114_qa0", 1.0,  "56.2% vs 55.1% — within 5%."),
    ("doc115_qa0", 0.0,  "Refused; gold $16,525M."),
    ("doc116_qa0", 0.0,  "Pred 4.29 vs gold 3.46 (Nike inventory turnover) — 24% off."),
    ("doc117_qa0", 0.0,  "Refused; gold operations."),
    ("doc118_qa0", 0.0,  "Refused; gold Yes $1.6B."),
    ("doc119_qa0", 0.0,  "Refused; gold $4.60B."),
    ("doc120_qa0", 0.0,  "Refused; gold lists 10 regions."),
    ("doc121_qa0", 0.75, "Litigation but no material — matches gold's 'No material'."),
    ("doc122_qa0", 0.0,  "Pred 0 vs gold $411M (Pepsico restructuring)."),
    ("doc123_qa0", 0.0,  "Refused; gold $9,068M."),
    ("doc124_qa0", 0.0,  "Refused; gold 16.5%."),
    ("doc125_qa0", 1.0,  "Defeated = gold."),
    ("doc126_qa0", 1.0,  "$400M = gold."),
    ("doc127_qa0", 0.0,  "Pred $4,200M (+possibility $4,950M) vs gold $8,400M — 50% off."),
    ("doc128_qa0", 1.0,  "Strong start + momentum = gold strong start."),
    ("doc129_qa0", 1.0,  "1pp exact = gold."),
    ("doc130_qa0", 0.75, "Yes correct direction but lacks specific numbers (gold just confirms Yes positive)."),
    ("doc131_qa0", 0.5,  "Pred Consumer Healthcare JV gain $(6)M (wrong amount); gold confirms JV gain 2019."),
    ("doc132_qa0", 0.0,  "Refused; gold lists Trillium/Array/Therachon."),
    ("doc133_qa0", 0.0,  "Pred $700M vs gold $77.78M (Pfizer Upjohn) — 9× off."),
    ("doc134_qa0", 1.0,  "Developed Rest of World = gold."),
    ("doc135_qa0", 1.0,  "Yes Upjohn separation = gold (cites $700M cost but conclusion matches)."),
    ("doc136_qa0", 1.0,  "'None.' = gold 'There are none'."),
    ("doc137_qa0", 1.0,  "Pred 'no major acquisitions reported' = gold 'did not make any'."),
    ("doc138_qa0", 1.0,  "Lower marketing + incentive comp leverage = gold."),
    ("doc139_qa0", 0.25, "Pred cites 'decrease of $104,233' (wrong direction); gold increase driven by 47 new stores."),
    ("doc140_qa0", 1.0,  "36.5% vs 36% — within 5%."),
    ("doc141_qa0", 0.0,  "Y/N flip: Decrease vs gold increased."),
    ("doc142_qa0", 1.0,  "Cross currency swaps = gold."),
    ("doc143_qa0", 0.75, "$1,097M pension only; gold has pension + health/life."),
    ("doc144_qa0", 0.0,  "Refused; gold definitive No 0.54."),
    ("doc145_qa0", 1.0,  "Yes capital intensive = gold."),
    ("doc146_qa0", 0.25, "Y/N flip: 'Yes increased' ($7,443M → $9,963M — debt subset) vs gold No total decreased."),
    ("doc147_qa0", 0.0,  "Pred 36.73 vs gold 42.69 — 14% off."),
    ("doc148_qa0", 0.0,  "Refused (FY19 OI not provided); gold 0.2%."),
    ("doc149_qa0", 0.0,  "Pred 4.0% vs gold 6.2% — 35% off."),
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
