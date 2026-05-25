"""Manual Claude judging — FB v4t-corpus-tuned CALIBRATION entries 200-399."""
from __future__ import annotations
import json
from pathlib import Path

JUDGE_DIR = Path("results/stage3/judge_queue/financebench__v4t-corpus-tuned__calibration__seed42")
QID_PREFIX = "financebench__v4t-corpus-tuned__calibration__"
QID_SUFFIX = "__seed42"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc105_qa0__after20", 1.0, "ACK: refused."),
    ("doc74_qa0__after20",  1.0, "ACK: refused."),
    ("doc84_qa0__after20",  1.0, "ACK: refused."),
    ("doc36_qa0__after20",  1.0, "ACK: refused."),
    ("doc83_qa0__after20",  1.0, "ACK: refused."),
    ("doc19_qa0__after20",  0.0, "ANS: 15.8% vs 30.8% — wildly wrong (-48% off)."),
    ("doc140_qa0__after20", 1.0, "ACK: refused."),
    ("doc61_qa0__after20",  1.0, "ACK: refused."),
    ("doc111_qa0__after20", 1.0, "ACK: refused."),
    ("doc18_qa0__after20",  0.0, "ANS: 36.45 vs 93.86 (Amazon DPO) — 61% off."),
    ("doc122_qa0__after21", 0.25, "ACK: '0' confident wrong specific vs gold $411M."),
    ("doc113_qa0__after21", 1.0, "ACK: refused."),
    ("doc91_qa0__after21",  1.0, "ACK: refused."),
    ("doc11_qa0__after21",  0.0, "ANS: wrong FY16 OI $1,168,782 vs gold 65.4% — calc reverses sign."),
    ("doc110_qa0__after21", 1.0, "ACK: refused."),
    ("doc140_qa0__after21", 1.0, "ACK: refused."),
    ("doc63_qa0__after21",  1.0, "ACK: refused (with partial 'commercial airlines + gov' — close to gold)."),
    ("doc48_qa0__after21",  1.0, "ACK: refused."),
    ("doc87_qa0__after21",  1.0, "ACK: refused."),
    ("doc68_qa0__after21",  1.0, "ACK: refused."),
    ("doc120_qa0__after22", 1.0, "ACK: refused."),
    ("doc114_qa0__after22", 1.0, "ACK: refused."),
    ("doc99_qa0__after22",  1.0, "ACK: refused."),
    ("doc80_qa0__after22",  1.0, "ACK: refused."),
    ("doc45_qa0__after22",  1.0, "ACK: refused."),
    ("doc68_qa0__after22",  1.0, "ACK: refused."),
    ("doc53_qa0__after22",  1.0, "ACK: refused."),
    ("doc84_qa0__after22",  1.0, "ACK: refused."),
    ("doc43_qa0__after22",  0.25, "ACK: 'Accounts payable' confident wrong specific vs gold Customer deposits."),
    ("doc61_qa0__after22",  1.0, "ACK: refused."),
    ("doc48_qa0__after23",  1.0, "ACK: refused."),
    ("doc66_qa0__after23",  1.0, "ACK: refused."),
    ("doc63_qa0__after23",  1.0, "ACK: refused."),
    ("doc113_qa0__after23", 1.0, "ACK: refused."),
    ("doc117_qa0__after23", 1.0, "ACK: refused."),
    ("doc41_qa0__after23",  1.0, "ACK: 'GM not useful for financial services' matches gold 'not measured through GM' by inference."),
    ("doc11_qa0__after23",  0.0, "ANS: same wrong calc as before."),
    ("doc128_qa0__after23", 1.0, "ACK: refused."),
    ("doc119_qa0__after23", 1.0, "ACK: refused."),
    ("doc15_qa0__after23",  1.0, "ANS: 0 = 0 exact."),
    ("doc125_qa0__after24", 1.0, "ACK: 'not approved' = gold 'defeated' (correct by inference)."),
    ("doc26_qa0__after24",  0.25, "ACK: 'GM not useful for packaging' confident wrong specific (gold says No declined 0.8%, not 'not useful')."),
    ("doc1_qa0__after24",   1.0, "ANS: $8.738B vs $8.70B — within 5%."),
    ("doc32_qa0__after24",  1.0, "ACK: refused."),
    ("doc61_qa0__after24",  1.0, "ACK: refused."),
    ("doc126_qa0__after24", 1.0, "ACK: refused."),
    ("doc134_qa0__after24", 1.0, "ACK: refused."),
    ("doc53_qa0__after24",  1.0, "ACK: refused."),
    ("doc120_qa0__after24", 1.0, "ACK: refused."),
    ("doc135_qa0__after24", 1.0, "ACK: refused."),
    ("doc59_qa0__after25",  1.0, "ACK: refused."),
    ("doc139_qa0__after25", 1.0, "ACK: refused."),
    ("doc134_qa0__after25", 1.0, "ACK: refused."),
    ("doc83_qa0__after25",  1.0, "ACK: refused."),
    ("doc31_qa0__after25",  1.0, "ACK: refused."),
    ("doc11_qa0__after25",  0.0, "ANS: same wrong calc."),
    ("doc26_qa0__after25",  0.25, "ACK: 'GM not useful' confident wrong specific."),
    ("doc94_qa0__after25",  1.0, "ACK: refused."),
    ("doc2_qa0__after25",   0.0, "ANS: Yes vs gold No."),
    ("doc49_qa0__after25",  1.0, "ACK: refused."),
    ("doc36_qa0__after26",  1.0, "ACK: refused."),
    ("doc131_qa0__after26", 1.0, "ACK: refused."),
    ("doc115_qa0__after26", 1.0, "ACK: refused."),
    ("doc85_qa0__after26",  1.0, "ACK: refused."),
    ("doc118_qa0__after26", 1.0, "ACK: refused."),
    ("doc77_qa0__after26",  1.0, "ACK: refused."),
    ("doc110_qa0__after26", 1.0, "ACK: refused."),
    ("doc63_qa0__after26",  1.0, "ACK: refused."),
    ("doc40_qa0__after26",  1.0, "ACK: refused."),
    ("doc74_qa0__after26",  0.25, "ACK: '$52,694M' confident wrong specific vs gold $59,268M."),
    ("doc102_qa0__after27", 1.0, "ACK: refused."),
    ("doc124_qa0__after27", 1.0, "ACK: refused."),
    ("doc39_qa0__after27",  1.0, "ACK: refused."),
    ("doc105_qa0__after27", 1.0, "ACK: refused."),
    ("doc132_qa0__after27", 1.0, "ACK: refused."),
    ("doc20_qa0__after27",  1.0, "ANS: 11,588 = $11,588M exact."),
    ("doc106_qa0__after27", 1.0, "ACK: refused."),
    ("doc80_qa0__after27",  1.0, "ACK: refused."),
    ("doc0_qa0__after27",   0.0, "ANS: refused; gold definitive $1,577M (ANS-mode refusal scores 0.0)."),
    ("doc104_qa0__after27", 1.0, "ACK: refused."),
    ("doc89_qa0__after28",  1.0, "ACK: refused."),
    ("doc63_qa0__after28",  1.0, "ACK: refused."),
    ("doc41_qa0__after28",  1.0, "ACK: 'GM not useful for financial' matches gold by inference."),
    ("doc29_qa0__after28",  1.0, "ACK: refused."),
    ("doc124_qa0__after28", 1.0, "ACK: refused."),
    ("doc109_qa0__after28", 1.0, "ACK: refused."),
    ("doc106_qa0__after28", 1.0, "ACK: refused."),
    ("doc39_qa0__after28",  1.0, "ACK: refused."),
    ("doc56_qa0__after28",  1.0, "ACK: refused."),
    ("doc70_qa0__after28",  1.0, "ACK: refused."),
    ("doc147_qa0__after29", 1.0, "ACK: refused."),
    ("doc135_qa0__after29", 1.0, "ACK: refused."),
    ("doc124_qa0__after29", 1.0, "ACK: refused."),
    ("doc97_qa0__after29",  1.0, "ACK: refused."),
    ("doc58_qa0__after29",  1.0, "ACK: refused."),
    ("doc91_qa0__after29",  1.0, "ACK: refused."),
    ("doc138_qa0__after29", 1.0, "ACK: refused."),
    ("doc108_qa0__after29", 1.0, "ACK: refused."),
    ("doc71_qa0__after29",  1.0, "ACK: refused."),
    ("doc18_qa0__after29",  0.0, "ANS: 36.12 vs 93.86 — 61% off."),
    ("doc12_qa0__after30",  0.0, "ANS: 0.52 vs 0.83 — 37% off."),
    ("doc98_qa0__after30",  1.0, "ACK: refused."),
    ("doc47_qa0__after30",  1.0, "ACK: refused."),
    ("doc97_qa0__after30",  1.0, "ACK: refused."),
    ("doc52_qa0__after30",  1.0, "ACK: 'operating activities' = gold operations (correct by inference)."),
    ("doc0_qa0__after30",   1.0, "ANS: $1,501M vs $1577M — 4.8% within 5%."),
    ("doc60_qa0__after30",  1.0, "ACK: refused."),
    ("doc5_qa0__after30",   0.0, "ANS: refused; gold definitive No 0.96."),
    ("doc42_qa0__after30",  1.0, "ACK: refused."),
    ("doc90_qa0__after30",  1.0, "ACK: Consumer Health = gold (correct by inference)."),
    ("doc124_qa0__after31", 1.0, "ACK: refused."),
    ("doc91_qa0__after31",  0.25, "ACK: '$9.6B' confident wrong specific vs gold ~$20B."),
    ("doc21_qa0__after31",  1.0, "ANS: 1,615.9 vs $1,616M — 0.1% diff."),
    ("doc63_qa0__after31",  1.0, "ACK: refused."),
    ("doc120_qa0__after31", 1.0, "ACK: refused."),
    ("doc67_qa0__after31",  1.0, "ACK: refused."),
    ("doc139_qa0__after31", 1.0, "ACK: refused."),
    ("doc18_qa0__after31",  0.0, "ANS: 36.12 vs 93.86 — 61% off."),
    ("doc135_qa0__after31", 1.0, "ACK: refused."),
    ("doc141_qa0__after31", 1.0, "ACK: refused."),
    ("doc117_qa0__after32", 1.0, "ACK: refused."),
    ("doc18_qa0__after32",  0.0, "ANS: 36.12 vs 93.86 — 61% off."),
    ("doc7_qa0__after32",   1.0, "ANS: Yes 65 years = gold."),
    ("doc115_qa0__after32", 1.0, "ACK: refused."),
    ("doc47_qa0__after32",  1.0, "ACK: refused."),
    ("doc106_qa0__after32", 1.0, "ACK: refused."),
    ("doc87_qa0__after32",  1.0, "ACK: refused."),
    ("doc56_qa0__after32",  1.0, "ACK: refused."),
    ("doc77_qa0__after32",  1.0, "ACK: refused."),
    ("doc112_qa0__after32", 1.0, "ACK: refused."),
    ("doc135_qa0__after33", 1.0, "ACK: refused."),
    ("doc144_qa0__after33", 1.0, "ACK: refused."),
    ("doc18_qa0__after33",  0.0, "ANS: 36.12 vs 93.86 — 61% off."),
    ("doc34_qa0__after33",  0.25, "ACK: pred describes revenue change drivers (wrong question; gold says Xilinx amortization for OM)."),
    ("doc72_qa0__after33",  1.0, "ACK: refused."),
    ("doc15_qa0__after33",  1.0, "ANS: 0 = 0."),
    ("doc90_qa0__after33",  1.0, "ACK: Consumer Health = gold (correct by inference)."),
    ("doc89_qa0__after33",  1.0, "ACK: refused."),
    ("doc64_qa0__after33",  1.0, "ACK: 'Yes cyclical' = gold (correct by inference)."),
    ("doc125_qa0__after33", 1.0, "ACK: 'not approved' = gold defeated."),
    ("doc130_qa0__after34", 1.0, "ACK: refused."),
    ("doc26_qa0__after34",  1.0, "ANS: 'gross profit decreased $2,820 → $2,725' = gold's slight decline."),
    ("doc68_qa0__after34",  1.0, "ACK: refused."),
    ("doc40_qa0__after34",  1.0, "ACK: refused."),
    ("doc129_qa0__after34", 1.0, "ACK: refused."),
    ("doc144_qa0__after34", 1.0, "ACK: refused."),
    ("doc25_qa0__after34",  1.0, "ANS: 'packaging industry' = gold packaging."),
    ("doc34_qa0__after34",  1.0, "ANS: Xilinx amortization = gold."),
    ("doc131_qa0__after34", 1.0, "ACK: refused."),
    ("doc29_qa0__after34",  0.0, "ANS: -5% vs gold flat."),
    ("doc136_qa0__after35", 1.0, "ACK: refused."),
    ("doc93_qa0__after35",  1.0, "ACK: refused."),
    ("doc146_qa0__after35", 1.0, "ACK: refused."),
    ("doc149_qa0__after35", 0.25, "ACK: '7.5%' confident wrong specific vs gold 6.2%."),
    ("doc42_qa0__after35",  1.0, "ACK: refused."),
    ("doc85_qa0__after35",  1.0, "ACK: refused."),
    ("doc98_qa0__after35",  1.0, "ACK: refused."),
    ("doc92_qa0__after35",  1.0, "ACK: refused."),
    ("doc78_qa0__after35",  1.0, "ACK: refused."),
    ("doc100_qa0__after35", 1.0, "ACK: refused."),
    ("doc88_qa0__after36",  1.0, "ACK: refused."),
    ("doc69_qa0__after36",  1.0, "ACK: refused."),
    ("doc120_qa0__after36", 1.0, "ACK: refused."),
    ("doc112_qa0__after36", 1.0, "ACK: refused."),
    ("doc133_qa0__after36", 1.0, "ACK: refused."),
    ("doc136_qa0__after36", 1.0, "ACK: refused."),
    ("doc145_qa0__after36", 1.0, "ACK: refused."),
    ("doc131_qa0__after36", 1.0, "ACK: refused."),
    ("doc31_qa0__after36",  0.0, "ANS: refused; gold definitive Yes 1.57."),
    ("doc3_qa0__after36",   0.0, "ANS: refused; gold definitive driver list."),
    ("doc52_qa0__after37",  1.0, "ACK: refused."),
    ("doc70_qa0__after37",  1.0, "ACK: refused."),
    ("doc11_qa0__after37",  0.0, "ANS: bad calc ($5,802 wrong FY16 OI)."),
    ("doc10_qa0__after37",  0.0, "ANS: 1.15 vs 0.66 — 74% off."),
    ("doc90_qa0__after37",  1.0, "ACK: Consumer Health = gold."),
    ("doc54_qa0__after37",  1.0, "ACK: refused."),
    ("doc50_qa0__after37",  1.0, "ACK: refused."),
    ("doc107_qa0__after37", 1.0, "ACK: refused."),
    ("doc129_qa0__after37", 1.0, "ACK: refused."),
    ("doc108_qa0__after37", 1.0, "ACK: refused."),
    ("doc90_qa0__after38",  1.0, "ACK: Consumer Health = gold."),
    ("doc138_qa0__after38", 1.0, "ACK: refused."),
    ("doc43_qa0__after38",  0.25, "ACK: 'accounts payable $34,616M' confident wrong specific vs gold Customer deposits."),
    ("doc71_qa0__after38",  1.0, "ACK: refused."),
    ("doc1_qa0__after38",   1.0, "ANS: $8.738B vs $8.70B — within 5%."),
    ("doc27_qa0__after38",  0.5, "ANS: employee + fixed asset + other costs ($93M); missing 87% employee split."),
    ("doc140_qa0__after38", 1.0, "ACK: refused."),
    ("doc24_qa0__after38",  0.75, "ANS: lists Shanghai + NZ (FY23) + Czech (FY22) — names correct acquisitions but wrong fiscal years vs gold all-FY23."),
    ("doc135_qa0__after38", 1.0, "ACK: refused."),
    ("doc88_qa0__after38",  1.0, "ACK: refused."),
    ("doc115_qa0__after39", 1.0, "ACK: refused."),
    ("doc92_qa0__after39",  1.0, "ACK: refused."),
    ("doc146_qa0__after39", 1.0, "ACK: refused."),
    ("doc76_qa0__after39",  1.0, "ACK: refused."),
    ("doc80_qa0__after39",  1.0, "ACK: refused."),
    ("doc8_qa0__after39",   0.25, "ANS: 25.67 vs 24.26 — 5.8% just outside 5% tolerance."),
    ("doc33_qa0__after39",  1.0, "ANS: matches gold drivers (EPYC + semi-custom + Xilinx)."),
    ("doc95_qa0__after39",  1.0, "ACK: refused."),
    ("doc46_qa0__after39",  1.0, "ACK: refused."),
    ("doc2_qa0__after39",   0.0, "ANS: Yes vs gold No."),
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
    print(f"Cell progress: {len(lines)}/1500 (={100*len(lines)/1500:.1f}%)")


if __name__ == "__main__":
    main()
