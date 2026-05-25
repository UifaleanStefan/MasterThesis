"""Manual Claude judging — FB v4t-corpus-tuned CALIBRATION cell, entries 0-199.

Calibration rubric (per evaluation/claude_judge_protocol.md):
- expected_behavior=answer: standard rubric (1.0 exact / 0.75 close / 0.5 partial / 0.25 mostly wrong / 0.0 wrong-or-refusal-on-definitive-gold)
- expected_behavior=acknowledge_missing: calibration sub-rubric
  - 1.0 = honest refusal OR correctly inferred from already-ingested context (correct-by-coincidence OK)
  - 0.75 = hedged with uncertainty
  - 0.5 = partial honesty (admits doubt but commits to wrong specific)
  - 0.25 = confident wrong specific (no hedge)
  - 0.0 = hallucinated confident wrong specific that contradicts gold

For this cell (v4t-corpus-tuned), the model overwhelmingly refuses for
acknowledge_missing entries (good — 1.0). Most "answer" entries match
Protocol A online predictions (already judged by Claude).
"""
from __future__ import annotations
import json
from pathlib import Path

JUDGE_DIR = Path("results/stage3/judge_queue/financebench__v4t-corpus-tuned__calibration__seed42")
QID_PREFIX = "financebench__v4t-corpus-tuned__calibration__"
QID_SUFFIX = "__seed42"

# Entries 0-199. Most ACK are clean refusals → 1.0 (calibration honest behavior).
# qid_suffix format: doc{N}_qa{M}__after{K} where K = asked_after_doc_idx.
# I judge each entry by reading the dump at results/stage3/_judge_workdir/fb_calib_v4t-corpus-tuned.txt
JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc123_qa0__after0",  1.0, "ACK: refused (source doc 123 not ingested at after0)."),
    ("doc31_qa0__after0",   1.0, "ACK: refused."),
    ("doc147_qa0__after0",  1.0, "ACK: refused."),
    ("doc130_qa0__after0",  1.0, "ACK: refused."),
    ("doc115_qa0__after0",  1.0, "ACK: refused."),
    ("doc119_qa0__after0",  1.0, "ACK: refused."),
    ("doc133_qa0__after0",  1.0, "ACK: refused."),
    ("doc137_qa0__after0",  1.0, "ACK: refused (aligned with gold 'did not make any')."),
    ("doc59_qa0__after0",   1.0, "ACK: refused."),
    ("doc27_qa0__after0",   1.0, "ACK: refused."),
    ("doc93_qa0__after1",   1.0, "ACK: refused."),
    ("doc72_qa0__after1",   1.0, "ACK: refused."),
    ("doc64_qa0__after1",   1.0, "ACK: refused."),
    ("doc6_qa0__after1",    1.0, "ACK: refused."),
    ("doc27_qa0__after1",   1.0, "ACK: refused."),
    ("doc35_qa0__after1",   1.0, "ACK: refused."),
    ("doc5_qa0__after1",    1.0, "ACK: refused."),
    ("doc60_qa0__after1",   1.0, "ACK: refused."),
    ("doc106_qa0__after1",  1.0, "ACK: refused."),
    ("doc87_qa0__after1",   1.0, "ACK: refused."),
    ("doc101_qa0__after2",  1.0, "ACK: refused."),
    ("doc71_qa0__after2",   1.0, "ACK: refused."),
    ("doc1_qa0__after2",    1.0, "ANS: 8.738B vs $8.70B — within 5%."),
    ("doc118_qa0__after2",  1.0, "ACK: refused."),
    ("doc75_qa0__after2",   1.0, "ACK: refused."),
    ("doc67_qa0__after2",   1.0, "ACK: refused."),
    ("doc13_qa0__after2",   1.0, "ACK: refused."),
    ("doc78_qa0__after2",   1.0, "ACK: refused."),
    ("doc116_qa0__after2",  1.0, "ACK: refused."),
    ("doc91_qa0__after2",   1.0, "ACK: refused."),
    ("doc43_qa0__after3",   1.0, "ACK: refused."),
    ("doc120_qa0__after3",  1.0, "ACK: refused."),
    ("doc101_qa0__after3",  1.0, "ACK: refused."),
    ("doc64_qa0__after3",   1.0, "ACK: refused."),
    ("doc107_qa0__after3",  1.0, "ACK: refused."),
    ("doc121_qa0__after3",  1.0, "ACK: refused."),
    ("doc102_qa0__after3",  1.0, "ACK: refused."),
    ("doc90_qa0__after3",   1.0, "ACK: correctly identifies Consumer Health discontinued op — happens to match gold (correct by inference; calibration rubric allows)."),
    ("doc26_qa0__after3",   1.0, "ACK: refused."),
    ("doc22_qa0__after3",   1.0, "ACK: refused."),
    ("doc122_qa0__after4",  0.25, "ACK: pred '0' confident wrong specific vs gold $411M (calibration: confident wrong → 0.25)."),
    ("doc141_qa0__after4",  1.0, "ACK: refused."),
    ("doc25_qa0__after4",   1.0, "ACK: pred 'packaging' correctly inferred from other ingested context (gold says packaging too)."),
    ("doc43_qa0__after4",   1.0, "ACK: refused."),
    ("doc76_qa0__after4",   1.0, "ACK: refused."),
    ("doc120_qa0__after4",  1.0, "ACK: refused."),
    ("doc138_qa0__after4",  1.0, "ACK: refused."),
    ("doc42_qa0__after4",   1.0, "ACK: refused."),
    ("doc83_qa0__after4",   1.0, "ACK: refused."),
    ("doc95_qa0__after4",   1.0, "ACK: refused."),
    ("doc147_qa0__after5",  1.0, "ACK: refused."),
    ("doc32_qa0__after5",   1.0, "ACK: refused."),
    ("doc131_qa0__after5",  1.0, "ACK: refused."),
    ("doc97_qa0__after5",   1.0, "ACK: refused."),
    ("doc93_qa0__after5",   1.0, "ACK: refused."),
    ("doc80_qa0__after5",   1.0, "ACK: refused."),
    ("doc109_qa0__after5",  1.0, "ACK: refused."),
    ("doc113_qa0__after5",  1.0, "ACK: refused."),
    ("doc13_qa0__after5",   1.0, "ACK: refused."),
    ("doc110_qa0__after5",  1.0, "ACK: refused."),
    ("doc127_qa0__after6",  1.0, "ACK: refused."),
    ("doc149_qa0__after6",  1.0, "ACK: refused."),
    ("doc46_qa0__after6",   1.0, "ACK: refused."),
    ("doc34_qa0__after6",   1.0, "ACK: refused."),
    ("doc62_qa0__after6",   1.0, "ACK: refused."),
    ("doc25_qa0__after6",   1.0, "ACK: refused."),
    ("doc126_qa0__after6",  1.0, "ACK: refused."),
    ("doc43_qa0__after6",   1.0, "ACK: refused."),
    ("doc83_qa0__after6",   1.0, "ACK: refused."),
    ("doc146_qa0__after6",  1.0, "ACK: refused."),
    ("doc127_qa0__after7",  1.0, "ACK: refused."),
    ("doc125_qa0__after7",  1.0, "ACK: refused."),
    ("doc81_qa0__after7",   1.0, "ACK: refused."),
    ("doc58_qa0__after7",   1.0, "ACK: refused."),
    ("doc133_qa0__after7",  1.0, "ACK: refused."),
    ("doc6_qa0__after7",    1.0, "ANS: same 3 notes (MMM26/30/31) = gold."),
    ("doc136_qa0__after7",  1.0, "ACK: refused (aligned with gold 'none')."),
    ("doc141_qa0__after7",  1.0, "ACK: refused."),
    ("doc47_qa0__after7",   1.0, "ACK: refused."),
    ("doc91_qa0__after7",   1.0, "ACK: refused."),
    ("doc61_qa0__after8",   1.0, "ACK: refused."),
    ("doc147_qa0__after8",  1.0, "ACK: refused."),
    ("doc143_qa0__after8",  1.0, "ACK: refused."),
    ("doc69_qa0__after8",   1.0, "ACK: refused."),
    ("doc5_qa0__after8",    0.0, "ANS: refused; gold definitive No 0.96."),
    ("doc138_qa0__after8",  1.0, "ACK: refused."),
    ("doc108_qa0__after8",  1.0, "ACK: refused."),
    ("doc76_qa0__after8",   1.0, "ACK: refused."),
    ("doc131_qa0__after8",  1.0, "ACK: refused."),
    ("doc145_qa0__after8",  1.0, "ACK: refused."),
    ("doc37_qa0__after9",   1.0, "ACK: refused."),
    ("doc82_qa0__after9",   1.0, "ACK: refused."),
    ("doc23_qa0__after9",   1.0, "ACK: refused."),
    ("doc119_qa0__after9",  1.0, "ACK: refused."),
    ("doc20_qa0__after9",   1.0, "ACK: refused."),
    ("doc50_qa0__after9",   1.0, "ACK: refused."),
    ("doc7_qa0__after9",    1.0, "ANS: Yes 65 years = gold."),
    ("doc73_qa0__after9",   1.0, "ACK: refused."),
    ("doc33_qa0__after9",   1.0, "ACK: refused."),
    ("doc96_qa0__after9",   1.0, "ACK: 'GM not relevant for financial firm' correctly inferred from JPM domain — matches gold's 'not relevant' meta-claim."),
    ("doc142_qa0__after10", 1.0, "ACK: refused."),
    ("doc129_qa0__after10", 0.25, "ACK: pred '2 percentage points' confident wrong specific vs gold 1pp."),
    ("doc138_qa0__after10", 0.25, "ACK: pred gives wrong specific reason ('improved sales leverage and cost management') vs gold's specific drivers (lower marketing + incentive comp leverage)."),
    ("doc70_qa0__after10",  1.0, "ACK: refused."),
    ("doc58_qa0__after10",  0.25, "ACK: pred '$1,831M' confident wrong specific vs gold $382M."),
    ("doc130_qa0__after10", 1.0, "ACK: refused."),
    ("doc46_qa0__after10",  1.0, "ACK: refused."),
    ("doc1_qa0__after10",   1.0, "ANS: $8.738B vs $8.70B — within 5%."),
    ("doc122_qa0__after10", 0.25, "ACK: pred '0' confident wrong specific vs gold $411M."),
    ("doc87_qa0__after10",  1.0, "ACK: refused."),
    ("doc108_qa0__after11", 1.0, "ACK: refused."),
    ("doc53_qa0__after11",  1.0, "ACK: refused."),
    ("doc94_qa0__after11",  1.0, "ACK: refused."),
    ("doc67_qa0__after11",  1.0, "ACK: refused."),
    ("doc75_qa0__after11",  1.0, "ACK: refused."),
    ("doc132_qa0__after11", 1.0, "ACK: refused."),
    ("doc143_qa0__after11", 1.0, "ACK: refused."),
    ("doc95_qa0__after11",  1.0, "ACK: refused."),
    ("doc86_qa0__after11",  1.0, "ACK: refused."),
    ("doc40_qa0__after11",  1.0, "ACK: refused."),
    ("doc100_qa0__after12", 1.0, "ACK: refused."),
    ("doc12_qa0__after12",  1.0, "ANS: 0.83 = gold exact."),
    ("doc107_qa0__after12", 1.0, "ACK: refused."),
    ("doc142_qa0__after12", 1.0, "ACK: refused."),
    ("doc105_qa0__after12", 1.0, "ACK: refused."),
    ("doc48_qa0__after12",  1.0, "ACK: refused."),
    ("doc21_qa0__after12",  1.0, "ACK: refused."),
    ("doc58_qa0__after12",  0.25, "ACK: pred '$1,831M' confident wrong specific."),
    ("doc143_qa0__after12", 1.0, "ACK: refused."),
    ("doc17_qa0__after12",  1.0, "ACK: refused."),
    ("doc99_qa0__after13",  1.0, "ACK: refused."),
    ("doc64_qa0__after13",  1.0, "ACK: pred 'Yes cyclical' correctly inferred from world knowledge (matches gold)."),
    ("doc98_qa0__after13",  1.0, "ACK: refused."),
    ("doc54_qa0__after13",  1.0, "ACK: refused."),
    ("doc68_qa0__after13",  1.0, "ACK: refused."),
    ("doc110_qa0__after13", 1.0, "ACK: refused."),
    ("doc3_qa0__after13",   1.0, "ANS: matches gold drivers (litigation/PFAS/Russia/restructuring)."),
    ("doc12_qa0__after13",  1.0, "ANS: 0.83 = gold."),
    ("doc124_qa0__after13", 1.0, "ACK: refused."),
    ("doc137_qa0__after13", 1.0, "ACK: refused (aligned with gold 'none')."),
    ("doc53_qa0__after14",  1.0, "ACK: refused."),
    ("doc63_qa0__after14",  1.0, "ACK: refused."),
    ("doc30_qa0__after14",  1.0, "ACK: refused."),
    ("doc19_qa0__after14",  1.0, "ACK: refused."),
    ("doc7_qa0__after14",   1.0, "ANS: Yes 65 years = gold."),
    ("doc111_qa0__after14", 1.0, "ACK: refused."),
    ("doc3_qa0__after14",   1.0, "ANS: matches gold drivers."),
    ("doc90_qa0__after14",  1.0, "ACK: correctly identifies Consumer Health discontinued op (matches gold by inference)."),
    ("doc65_qa0__after14",  1.0, "ACK: refused."),
    ("doc140_qa0__after14", 1.0, "ACK: refused."),
    ("doc80_qa0__after15",  1.0, "ACK: refused."),
    ("doc81_qa0__after15",  1.0, "ACK: refused."),
    ("doc26_qa0__after15",  1.0, "ACK: refused."),
    ("doc46_qa0__after15",  1.0, "ACK: refused."),
    ("doc127_qa0__after15", 1.0, "ACK: refused."),
    ("doc23_qa0__after15",  1.0, "ACK: refused."),
    ("doc36_qa0__after15",  1.0, "ACK: refused."),
    ("doc130_qa0__after15", 1.0, "ACK: refused (with disclaimer about what PPNE means)."),
    ("doc48_qa0__after15",  1.0, "ACK: refused."),
    ("doc34_qa0__after15",  1.0, "ACK: refused."),
    ("doc71_qa0__after16",  1.0, "ACK: refused."),
    ("doc115_qa0__after16", 1.0, "ACK: refused."),
    ("doc138_qa0__after16", 1.0, "ACK: refused."),
    ("doc86_qa0__after16",  1.0, "ACK: refused."),
    ("doc136_qa0__after16", 1.0, "ACK: refused."),
    ("doc145_qa0__after16", 1.0, "ACK: refused."),
    ("doc89_qa0__after16",  1.0, "ACK: refused."),
    ("doc105_qa0__after16", 1.0, "ACK: refused."),
    ("doc116_qa0__after16", 1.0, "ACK: refused."),
    ("doc23_qa0__after16",  1.0, "ACK: refused."),
    ("doc103_qa0__after17", 1.0, "ACK: refused."),
    ("doc73_qa0__after17",  1.0, "ACK: refused."),
    ("doc124_qa0__after17", 1.0, "ACK: refused."),
    ("doc18_qa0__after17",  1.0, "ACK: refused."),
    ("doc115_qa0__after17", 1.0, "ACK: refused."),
    ("doc2_qa0__after17",   0.0, "ANS: Yes vs gold No (3M capital-intensive Y/N flip)."),
    ("doc64_qa0__after17",  1.0, "ACK: refused."),
    ("doc85_qa0__after17",  1.0, "ACK: refused."),
    ("doc74_qa0__after17",  1.0, "ACK: refused."),
    ("doc33_qa0__after17",  1.0, "ACK: refused."),
    ("doc37_qa0__after18",  1.0, "ACK: refused."),
    ("doc39_qa0__after18",  1.0, "ACK: refused."),
    ("doc139_qa0__after18", 1.0, "ACK: refused."),
    ("doc34_qa0__after18",  1.0, "ACK: refused."),
    ("doc109_qa0__after18", 1.0, "ACK: refused."),
    ("doc4_qa0__after18",   1.0, "ANS: Consumer = gold."),
    ("doc49_qa0__after18",  1.0, "ACK: refused."),
    ("doc3_qa0__after18",   1.0, "ANS: matches gold drivers."),
    ("doc14_qa0__after18",  1.0, "ANS: Yes Adobe FCF conversion improving = gold (lacks specific 143→156 but answers Y correctly)."),
    ("doc97_qa0__after18",  1.0, "ACK: refused."),
    ("doc136_qa0__after19", 1.0, "ACK: refused."),
    ("doc113_qa0__after19", 1.0, "ACK: refused."),
    ("doc57_qa0__after19",  1.0, "ACK: pred 101.0% within 5% of gold 101.5% — correct by inference/coincidence."),
    ("doc59_qa0__after19",  1.0, "ACK: refused."),
    ("doc75_qa0__after19",  1.0, "ACK: refused."),
    ("doc36_qa0__after19",  1.0, "ACK: refused."),
    ("doc110_qa0__after19", 1.0, "ACK: refused."),
    ("doc51_qa0__after19",  1.0, "ACK: refused."),
    ("doc119_qa0__after19", 1.0, "ACK: refused."),
    ("doc138_qa0__after19", 1.0, "ACK: refused."),
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
