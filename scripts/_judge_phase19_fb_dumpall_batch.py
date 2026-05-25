"""Manual Claude judging — FB dump-all BATCH cell.

THE COLLAPSE. End-of-corpus batch with all 188 paragraphs of the full
FB corpus dumped into a single gpt-4o-mini prompt. The model can no
longer locate specific facts in the noise and refuses ~95% of questions.
This is the §6.5.1 headline finding: dump-all batch judge collapses
to ~0.04 while corpus-tuned θ holds at ~0.65. Context-stuffing breaks
at scale; selective retrieval is structurally necessary.

Expected mean: ~0.04, matching §6.5.1's reported 0.037.
"""
from __future__ import annotations
import json
from pathlib import Path

JUDGE_DIR = Path("results/stage3/judge_queue/financebench__dump-all__batch__seed42")
QID_PREFIX = "financebench__dump-all__batch__"
QID_SUFFIX = "__seed42"

# Almost all 150 entries are refusals ("provided passages do not contain...").
# Only a tiny handful pass — the ones where the LLM happened to surface a
# specific fact despite the 188-paragraph noise. Default = 0.0 for refusal.
JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc0_qa0",   0.0,  "Refused; gold $1,577M."),
    ("doc1_qa0",   0.0,  "Refused; gold $8.70B."),
    ("doc2_qa0",   0.0,  "Refused; gold No."),
    ("doc3_qa0",   0.0,  "Refused; gold lists drivers."),
    ("doc4_qa0",   0.0,  "Refused; gold Consumer."),
    ("doc5_qa0",   0.0,  "Refused; gold No 0.96."),
    ("doc6_qa0",   0.0,  "Refused; gold MMM26/30/31."),
    ("doc7_qa0",   0.0,  "Refused; gold Yes 65 years."),
    ("doc8_qa0",   0.0,  "Refused; gold 24.26."),
    ("doc9_qa0",   0.0,  "Refused; gold 1.9%."),
    ("doc10_qa0",  0.0,  "Refused; gold 0.66."),
    ("doc11_qa0",  0.0,  "Refused; gold 65.4%."),
    ("doc12_qa0",  0.0,  "Refused; gold 0.83."),
    ("doc13_qa0",  0.0,  "Refused; gold No declined."),
    ("doc14_qa0",  0.0,  "Refused; gold Yes improved."),
    ("doc15_qa0",  1.0,  "0 = 0 (one of the few non-refusals; gpt-4o-mini found this trivial extraction)."),
    ("doc16_qa0",  0.0,  "Refused; gold 9.5x."),
    ("doc17_qa0",  0.0,  "Refused; gold -0.02."),
    ("doc18_qa0",  0.0,  "Refused; gold 93.86."),
    ("doc19_qa0",  0.0,  "Refused; gold 30.8%."),
    ("doc20_qa0",  0.0,  "Refused; gold $11,588M."),
    ("doc21_qa0",  0.0,  "Refused; gold $1,616M."),
    ("doc22_qa0",  0.0,  "Refused; gold details supplemental indentures."),
    ("doc23_qa0",  0.0,  "Refused; gold improved 0.67 → 0.69."),
    ("doc24_qa0",  0.0,  "Refused; gold lists Czech/Shanghai/NZ."),
    ("doc25_qa0",  0.0,  "Refused; gold packaging."),
    ("doc26_qa0",  0.0,  "Refused; gold No 0.8% decline."),
    ("doc27_qa0",  0.0,  "Refused; gold 87% employee."),
    ("doc28_qa0",  0.0,  "Refused; gold $2,018M."),
    ("doc29_qa0",  0.0,  "Refused; gold flat."),
    ("doc30_qa0",  0.0,  "Refused; gold 4.2%."),
    ("doc31_qa0",  0.0,  "Refused; gold Yes 1.57."),
    ("doc32_qa0",  0.0,  "Refused; gold lists AMD products."),
    ("doc33_qa0",  0.0,  "Refused; gold names EPYC + semi-custom + Xilinx."),
    ("doc34_qa0",  0.0,  "Refused; gold Xilinx amortization."),
    ("doc35_qa0",  0.0,  "Refused; gold Operations."),
    ("doc36_qa0",  0.0,  "Refused; gold Data Center."),
    ("doc37_qa0",  0.0,  "Refused; gold Yes 16%."),
    ("doc38_qa0",  0.0,  "Refused; gold definitive 'There are none' — refusal ≠ 'none'."),
    ("doc39_qa0",  0.0,  "Refused; gold US/EMEA/APAC/LACC."),
    ("doc40_qa0",  0.0,  "Refused; gold OM not measured."),
    ("doc41_qa0",  0.0,  "Refused; gold GM not measured."),
    ("doc42_qa0",  0.0,  "Refused; gold 24.6 → 21.6."),
    ("doc43_qa0",  0.0,  "Pred 'Total debt' vs gold Customer deposits."),
    ("doc44_qa0",  0.0,  "Refused; gold Yes."),
    ("doc45_qa0",  0.0,  "Refused; gold $0.40B."),
    ("doc46_qa0",  0.0,  "Refused; gold $1,832M."),
    ("doc47_qa0",  0.0,  "Refused; gold No -$1,561M."),
    ("doc48_qa0",  0.0,  "Refused; gold 2.8%."),
    ("doc49_qa0",  0.0,  "Refused; gold $5,409M."),
    ("doc50_qa0",  0.0,  "Refused; gold Yes consistent."),
    ("doc51_qa0",  0.0,  "Refused; gold lists Current Health + Yardbird."),
    ("doc52_qa0",  0.0,  "Refused; gold operations $1.8bn."),
    ("doc53_qa0",  0.0,  "Refused; gold Yes ~42%."),
    ("doc54_qa0",  0.0,  "Refused; gold 982 → 969."),
    ("doc55_qa0",  0.0,  "Refused; gold Entertainment +9%."),
    ("doc56_qa0",  0.0,  "Refused; gold 1.73."),
    ("doc57_qa0",  0.0,  "Refused; gold 101.5%."),
    ("doc58_qa0",  0.0,  "Refused; gold $382M."),
    ("doc59_qa0",  0.0,  "Refused; gold $12,645M."),
    ("doc60_qa0",  0.0,  "Refused; gold Commercial + Defense both >20%."),
    ("doc61_qa0",  0.0,  "Refused; gold Lion Air + Ethiopian."),
    ("doc62_qa0",  0.0,  "Refused; gold Yes improving 4.8 → 5.3."),
    ("doc63_qa0",  0.0,  "Refused; gold limited airlines + US gov."),
    ("doc64_qa0",  0.0,  "Refused; gold Yes cyclical."),
    ("doc65_qa0",  0.0,  "Refused; gold 737/777X/787 increases."),
    ("doc66_qa0",  0.0,  "Refused; gold 0.62% vs -14.76%."),
    ("doc67_qa0",  0.0,  "Refused; gold 0.01."),
    ("doc68_qa0",  0.0,  "Refused; gold 39.7%."),
    ("doc69_qa0",  0.0,  "Refused; gold 0.8."),
    ("doc70_qa0",  0.0,  "Refused; gold 63.86."),
    ("doc71_qa0",  0.0,  "Refused; gold 10.3%."),
    ("doc72_qa0",  0.0,  "Refused; gold 20% → 23%."),
    ("doc73_qa0",  0.0,  "Refused; gold Yes $831M."),
    ("doc74_qa0",  0.0,  "Refused; gold $59,268M."),
    ("doc75_qa0",  0.0,  "Refused; gold 17.98."),
    ("doc76_qa0",  0.0,  "Refused; gold Yes capital intensive."),
    ("doc77_qa0",  0.0,  "Refused; gold lists CVS legal disputes."),
    ("doc78_qa0",  0.0,  "Refused; gold Yes $0.55/share."),
    ("doc79_qa0",  0.0,  "Refused; gold Yes Mary Dillon ex-Ulta CEO."),
    ("doc80_qa0",  0.0,  "Refused; gold Richard A. Johnson."),
    ("doc81_qa0",  0.0,  "Refused; gold -3.7."),
    ("doc82_qa0",  0.0,  "Refused; gold 0.68."),
    ("doc83_qa0",  0.0,  "Refused; gold $3,215M."),
    ("doc84_qa0",  0.0,  "Refused; gold 0.54."),
    ("doc85_qa0",  0.0,  "Refused; gold No 1.3% growth."),
    ("doc86_qa0",  0.0,  "Refused; gold names drivers."),
    ("doc87_qa0",  0.0,  "Refused; gold 2.7x."),
    ("doc88_qa0",  0.0,  "Refused; gold No decelerate."),
    ("doc89_qa0",  0.0,  "Refused; gold US +3.0% intl -0.6%."),
    ("doc90_qa0",  1.0,  "Pred Consumer Health = gold (model surfaced this short-form question)."),
    ("doc91_qa0",  0.0,  "Refused; gold ~$20B."),
    ("doc92_qa0",  0.0,  "Refused; gold $13.2B."),
    ("doc93_qa0",  0.0,  "Refused; gold Yes 20.0 → 20.1."),
    ("doc94_qa0",  0.0,  "Refused; gold Corporate."),
    ("doc95_qa0",  0.0,  "Refused; gold $66.56."),
    ("doc96_qa0",  1.0,  "Pred GM not relevant for financial firm = gold (the model surfaced JPM's domain and made the correct meta-claim)."),
    ("doc97_qa0",  0.0,  "Refused; gold CIB."),
    ("doc98_qa0",  0.0,  "Refused; gold Yes VaR decreased."),
    ("doc99_qa0",  0.0,  "Refused; gold 6.25."),
    ("doc100_qa0", 0.0,  "Refused; gold 1.33."),
    ("doc101_qa0", 0.0,  "Refused; gold $5,818M."),
    ("doc102_qa0", 0.0,  "Refused; gold 0.4%."),
    ("doc103_qa0", 0.0,  "Refused; gold $303M."),
    ("doc104_qa0", 0.0,  "Refused; gold 7.9%."),
    ("doc105_qa0", 0.0,  "Refused; gold Yes $0.01/share."),
    ("doc106_qa0", 0.0,  "Refused; gold Las Vegas ~90%."),
    ("doc107_qa0", 0.0,  "Refused; gold 0 (negative EBIT)."),
    ("doc108_qa0", 0.0,  "Refused; gold MGM China -44%."),
    ("doc109_qa0", 0.0,  "Refused; gold corporate bonds."),
    ("doc110_qa0", 0.0,  "Refused; gold $32,780M."),
    ("doc111_qa0", 0.0,  "Refused; gold No decreased $2.5bn."),
    ("doc112_qa0", 0.0,  "Refused; gold 5.4%."),
    ("doc113_qa0", 0.0,  "Refused; gold $5,466M."),
    ("doc114_qa0", 0.0,  "Pred 61.5% (FY16-18) vs gold 55.1% — pred picks wrong years/wrong COGS-vs-margin metric."),
    ("doc115_qa0", 0.0,  "Refused; gold $16,525M."),
    ("doc116_qa0", 0.0,  "Refused; gold 3.46."),
    ("doc117_qa0", 0.0,  "Refused; gold operations."),
    ("doc118_qa0", 0.0,  "Refused; gold Yes $1.6B."),
    ("doc119_qa0", 0.0,  "Refused; gold $4.60B."),
    ("doc120_qa0", 0.0,  "Refused; gold lists 10 regions."),
    ("doc121_qa0", 0.0,  "Refused; gold No material."),
    ("doc122_qa0", 0.0,  "Pred 0 vs gold $411M (Pepsico restructuring)."),
    ("doc123_qa0", 0.0,  "Refused; gold $9,068M."),
    ("doc124_qa0", 0.0,  "Refused; gold 16.5%."),
    ("doc125_qa0", 0.0,  "Refused; gold defeated."),
    ("doc126_qa0", 0.0,  "Refused; gold $400M."),
    ("doc127_qa0", 0.0,  "Refused; gold $8.4B."),
    ("doc128_qa0", 0.0,  "Refused; gold strong start."),
    ("doc129_qa0", 0.0,  "Refused; gold 1pp."),
    ("doc130_qa0", 0.0,  "Refused; gold Yes PPNE positive."),
    ("doc131_qa0", 0.0,  "Refused; gold Yes Consumer Healthcare JV gain."),
    ("doc132_qa0", 0.0,  "Refused; gold Trillium/Array/Therachon."),
    ("doc133_qa0", 0.0,  "Refused; gold $77.78M."),
    ("doc134_qa0", 0.0,  "Refused; gold Developed Rest of World."),
    ("doc135_qa0", 0.0,  "Refused; gold Yes Upjohn."),
    ("doc136_qa0", 0.0,  "Refused; gold 'There are none'."),
    ("doc137_qa0", 0.0,  "Refused; gold 'did not make any'."),
    ("doc138_qa0", 0.0,  "Refused; gold lower marketing + incentive comp."),
    ("doc139_qa0", 0.0,  "Refused; gold 47 new stores."),
    ("doc140_qa0", 0.0,  "Refused; gold 36%."),
    ("doc141_qa0", 0.0,  "Refused; gold increased."),
    ("doc142_qa0", 1.0,  "Pred cross currency swaps = gold (model surfaced this short fact)."),
    ("doc143_qa0", 0.75, "$1,097M pension only; gold has pension + health/life $862M."),
    ("doc144_qa0", 0.0,  "Refused; gold No 0.54."),
    ("doc145_qa0", 1.0,  "Pred Yes capital intensive = gold (model surfaced Verizon's high PP&E)."),
    ("doc146_qa0", 0.0,  "Y/N flip: pred Yes increased (without details) vs gold No (decreased $229M)."),
    ("doc147_qa0", 0.0,  "Pred 36.12 vs gold 42.69 — 15% off."),
    ("doc148_qa0", 0.0,  "-1.4% vs 0.2% — wrong sign."),
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
