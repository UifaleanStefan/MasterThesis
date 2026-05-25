"""Claude manual judging — Phase 1.9 FB calibration dump-all (entries 715-859)."""

from __future__ import annotations

import json
from pathlib import Path

JUDGE_DIR = Path(
    "results/stage3/judge_queue/financebench__dump-all__calibration__seed42"
)
QID_PREFIX = "financebench__dump-all__calibration__"
QID_SUFFIX = "__seed42"

JUDGMENTS: list[tuple[str, float, str]] = [
    # 715-719
    ("doc139_qa0__after71", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc42_qa0__after71", 0.25, "ANS gold 24.6%→21.6% AMEX; predicted '20.0%→19.0%' — confident wrong specifics."),
    ("doc94_qa0__after71", 1.0, "ACK honest refusal — doc94 not yet ingested."),
    ("doc58_qa0__after71", 0.0, "ANS refusal on definitive gold ($382)."),
    ("doc14_qa0__after71", 0.0, "ANS refusal on definitive gold."),
    # 720-729
    ("doc3_qa0__after72", 0.0, "ANS refusal on definitive gold."),
    ("doc110_qa0__after72", 1.0, "ACK honest refusal — doc110 not yet ingested."),
    ("doc134_qa0__after72", 1.0, "ACK honest refusal — doc134 not yet ingested."),
    ("doc12_qa0__after72", 0.0, "ANS refusal on definitive gold."),
    ("doc71_qa0__after72", 0.0, "ANS gold 10.3%; predicted 15.5% — wrong specific."),
    ("doc52_qa0__after72", 0.0, "ANS refusal on definitive gold."),
    ("doc64_qa0__after72", 1.0, "ANS Yes Boeing cyclical — match."),
    ("doc26_qa0__after72", 0.0, "ANS refusal on definitive gold."),
    ("doc117_qa0__after72", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc119_qa0__after72", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    # 730-739
    ("doc14_qa0__after73", 0.0, "ANS refusal on definitive gold."),
    ("doc106_qa0__after73", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc12_qa0__after73", 0.0, "ANS refusal on definitive gold."),
    ("doc114_qa0__after73", 1.0, "ACK honest refusal — doc114 not yet ingested."),
    ("doc92_qa0__after73", 1.0, "ACK honest refusal — doc92 not yet ingested."),
    ("doc140_qa0__after73", 1.0, "ACK honest refusal — doc140 not yet ingested."),
    ("doc115_qa0__after73", 1.0, "ACK honest refusal — doc115 not yet ingested."),
    ("doc69_qa0__after73", 0.0, "ANS gold 0.8; predicted 1.00 — wrong specific (25% off)."),
    ("doc4_qa0__after73", 0.0, "ANS refusal on definitive gold."),
    ("doc26_qa0__after73", 0.0, "ANS refusal on definitive gold."),
    # 740-749
    ("doc119_qa0__after74", 0.25, "ACK calibration: '$4.2B PepsiCo FY2021 capex' confident wrong (gold $4.60B, 8.7% off)."),
    ("doc117_qa0__after74", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc69_qa0__after74", 0.0, "ANS gold 0.8; predicted 1.08 — wrong specific."),
    ("doc123_qa0__after74", 1.0, "ACK honest refusal — doc123 not yet ingested."),
    ("doc90_qa0__after74", 1.0, "ACK Consumer Health discontinued — correct by inference."),
    ("doc83_qa0__after74", 1.0, "ACK honest refusal — doc83 not yet ingested."),
    ("doc126_qa0__after74", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    ("doc50_qa0__after74", 0.0, "ANS refusal on definitive gold (consistent margins)."),
    ("doc22_qa0__after74", 0.0, "ANS refusal on definitive gold (Amcor 8K)."),
    ("doc6_qa0__after74", 0.0, "ANS refusal on definitive gold (3M debt securities)."),
    # 750-759
    ("doc37_qa0__after75", 0.0, "ANS refusal on definitive gold (16% customer)."),
    ("doc0_qa0__after75", 0.0, "ANS refusal on definitive gold ($1,577)."),
    ("doc122_qa0__after75", 0.25, "ACK '0' confident wrong (gold $411M)."),
    ("doc26_qa0__after75", 0.0, "ANS refusal on definitive gold."),
    ("doc126_qa0__after75", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    ("doc111_qa0__after75", 1.0, "ACK honest refusal — doc111 not yet ingested."),
    ("doc53_qa0__after75", 0.0, "ANS refusal on definitive gold (~42% decline)."),
    ("doc25_qa0__after75", 0.0, "ANS refusal on definitive gold."),
    ("doc121_qa0__after75", 1.0, "ACK honest refusal — doc121 not yet ingested."),
    ("doc133_qa0__after75", 1.0, "ACK honest refusal — doc133 not yet ingested."),
    # 760-769
    ("doc3_qa0__after76", 0.0, "ANS refusal on definitive gold."),
    ("doc41_qa0__after76", 0.0, "ANS refusal on definitive gold (gross margin not useful)."),
    ("doc112_qa0__after76", 1.0, "ACK honest refusal — doc112 not yet ingested."),
    ("doc100_qa0__after76", 1.0, "ACK honest refusal — doc100 not yet ingested."),
    ("doc37_qa0__after76", 0.0, "ANS refusal on definitive gold."),
    ("doc55_qa0__after76", 0.0, "ANS refusal on definitive gold."),
    ("doc18_qa0__after76", 0.0, "ANS refusal on definitive gold (93.86)."),
    ("doc86_qa0__after76", 1.0, "ACK honest refusal — doc86 not yet ingested."),
    ("doc74_qa0__after76", 1.0, "ANS gold $59,268; predicted $59,268 — exact!"),
    ("doc11_qa0__after76", 0.0, "ANS refusal on definitive gold (65.4%)."),
    # 770-779
    ("doc64_qa0__after77", 0.0, "ANS gold Yes Boeing cyclical definitive; predicted refusal — refusal on definitive gold."),
    ("doc60_qa0__after77", 0.0, "ANS refusal on definitive gold (Commercial Airplanes)."),
    ("doc113_qa0__after77", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc44_qa0__after77", 0.0, "ANS gold Yes Amex card retention definitive; predicted refusal — refusal on definitive gold."),
    ("doc87_qa0__after77", 1.0, "ACK honest refusal — doc87 not yet ingested."),
    ("doc82_qa0__after77", 1.0, "ACK honest refusal — doc82 not yet ingested."),
    ("doc52_qa0__after77", 0.0, "ANS refusal on definitive gold."),
    ("doc97_qa0__after77", 1.0, "ACK honest refusal — doc97 not yet ingested."),
    ("doc130_qa0__after77", 1.0, "ACK honest refusal — doc130 not yet ingested."),
    ("doc11_qa0__after77", 0.0, "ANS refusal on definitive gold."),
    # 780-789
    ("doc149_qa0__after78", 1.0, "ACK honest refusal — doc149 not yet ingested."),
    ("doc120_qa0__after78", 1.0, "ACK honest refusal — doc120 not yet ingested."),
    ("doc19_qa0__after78", 0.0, "ANS refusal on definitive gold (30.8%)."),
    ("doc44_qa0__after78", 0.0, "ANS refusal on definitive gold (Yes Amex retention)."),
    ("doc63_qa0__after78", 0.0, "ANS refusal on definitive gold (Boeing customers)."),
    ("doc102_qa0__after78", 1.0, "ACK honest refusal — doc102 not yet ingested."),
    ("doc67_qa0__after78", 0.0, "ANS refusal on definitive gold (0.01)."),
    ("doc40_qa0__after78", 0.0, "ANS refusal on definitive gold."),
    ("doc52_qa0__after78", 0.0, "ANS refusal on definitive gold."),
    ("doc65_qa0__after78", 0.0, "ANS gold Boeing production rates; predicted refusal — refusal on definitive gold."),
    # 790-799
    ("doc146_qa0__after79", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc23_qa0__after79", 0.0, "ANS refusal on definitive gold (quick ratio)."),
    ("doc109_qa0__after79", 1.0, "ACK honest refusal — doc109 not yet ingested."),
    ("doc56_qa0__after79", 0.0, "ANS refusal on definitive gold (1.73 Block WC)."),
    ("doc92_qa0__after79", 1.0, "ACK honest refusal — doc92 not yet ingested."),
    ("doc55_qa0__after79", 0.0, "ANS refusal on definitive gold."),
    ("doc28_qa0__after79", 0.0, "ANS refusal on definitive gold (AMCOR EBITDA)."),
    ("doc83_qa0__after79", 1.0, "ACK honest refusal — doc83 not yet ingested."),
    ("doc2_qa0__after79", 0.0, "ANS refusal on definitive gold."),
    ("doc14_qa0__after79", 0.0, "ANS refusal on definitive gold."),
    # 800-809
    ("doc106_qa0__after80", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc44_qa0__after80", 0.0, "ANS refusal on definitive gold."),
    ("doc82_qa0__after80", 1.0, "ACK honest refusal — doc82 not yet ingested."),
    ("doc25_qa0__after80", 0.0, "ANS refusal on definitive gold (Amcor packaging)."),
    ("doc60_qa0__after80", 0.0, "ANS refusal on definitive gold."),
    ("doc103_qa0__after80", 1.0, "ACK honest refusal — doc103 not yet ingested."),
    ("doc35_qa0__after80", 0.0, "ANS refusal on definitive gold."),
    ("doc12_qa0__after80", 0.0, "ANS refusal on definitive gold."),
    ("doc141_qa0__after80", 1.0, "ACK honest refusal — doc141 not yet ingested."),
    ("doc43_qa0__after80", 0.0, "ANS refusal on definitive gold (Customer deposits)."),
    # 810-819
    ("doc30_qa0__after81", 0.0, "ANS refusal on definitive gold (4.2%)."),
    ("doc75_qa0__after81", 0.25, "ANS gold 17.98 fixed asset turnover; predicted 16.73 — 7% off, slightly outside 5% tolerance."),
    ("doc79_qa0__after81", 1.0, "ANS gold Yes Mary Dillon similar Ulta; predicted Yes Mary Dillon former Ulta CEO — match."),
    ("doc2_qa0__after81", 0.0, "ANS refusal on definitive gold."),
    ("doc138_qa0__after81", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc60_qa0__after81", 0.0, "ANS refusal on definitive gold."),
    ("doc23_qa0__after81", 0.0, "ANS refusal on definitive gold."),
    ("doc59_qa0__after81", 0.0, "ANS refusal on definitive gold."),
    ("doc98_qa0__after81", 1.0, "ACK honest refusal — doc98 not yet ingested."),
    ("doc106_qa0__after81", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    # 820-829
    ("doc79_qa0__after82", 1.0, "ANS Mary Dillon former Ulta CEO — match."),
    ("doc12_qa0__after82", 0.0, "ANS refusal on definitive gold."),
    ("doc125_qa0__after82", 1.0, "ACK 'proposal not approved with 66.5% against' — correct + detail."),
    ("doc28_qa0__after82", 0.0, "ANS refusal on definitive gold."),
    ("doc35_qa0__after82", 0.0, "ANS refusal on definitive gold."),
    ("doc27_qa0__after82", 0.0, "ANS refusal on definitive gold."),
    ("doc43_qa0__after82", 0.25, "ANS 'accounts payable' confident wrong (gold Customer deposits)."),
    ("doc101_qa0__after82", 1.0, "ACK honest refusal — doc101 not yet ingested."),
    ("doc71_qa0__after82", 0.0, "ANS refusal on definitive gold."),
    ("doc144_qa0__after82", 1.0, "ACK honest refusal — doc144 not yet ingested."),
    # 830-839
    ("doc39_qa0__after83", 0.0, "ANS refusal on definitive gold."),
    ("doc3_qa0__after83", 0.0, "ANS refusal on definitive gold."),
    ("doc54_qa0__after83", 0.0, "ANS refusal on definitive gold (982→969 stores)."),
    ("doc42_qa0__after83", 0.0, "ANS refusal on definitive gold."),
    ("doc144_qa0__after83", 1.0, "ACK honest refusal — doc144 not yet ingested."),
    ("doc126_qa0__after83", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    ("doc90_qa0__after83", 1.0, "ACK Consumer Health discontinued — correct by inference."),
    ("doc17_qa0__after83", 0.0, "ANS refusal on definitive gold."),
    ("doc46_qa0__after83", 0.0, "ANS refusal on definitive gold."),
    ("doc57_qa0__after83", 0.0, "ANS refusal on definitive gold (101.5%)."),
    # 840-849
    ("doc148_qa0__after84", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc46_qa0__after84", 0.0, "ANS refusal on definitive gold."),
    ("doc84_qa0__after84", 1.0, "ANS gold 0.54; predicted 0.54 — exact!"),
    ("doc12_qa0__after84", 0.0, "ANS refusal on definitive gold."),
    ("doc77_qa0__after84", 0.75, "ANS gold CVS legal battles (multiple incl usual customary pricing); predicted Yes lawsuits about drug pricing + rebate + opioid settlement — partial (similar topics, plus opioid extra)."),
    ("doc58_qa0__after84", 0.0, "ANS refusal on definitive gold."),
    ("doc29_qa0__after84", 0.0, "ANS refusal on definitive gold (flat real growth)."),
    ("doc124_qa0__after84", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc13_qa0__after84", 0.0, "ANS refusal on definitive gold."),
    ("doc8_qa0__after84", 0.0, "ANS refusal on definitive gold (24.26)."),
    # 850-859
    ("doc18_qa0__after85", 0.0, "ANS refusal on definitive gold."),
    ("doc131_qa0__after85", 1.0, "ACK honest refusal — doc131 not yet ingested."),
    ("doc67_qa0__after85", 0.0, "ANS refusal on definitive gold."),
    ("doc11_qa0__after85", 0.0, "ANS refusal on definitive gold."),
    ("doc118_qa0__after85", 1.0, "ACK honest refusal — doc118 not yet ingested."),
    ("doc48_qa0__after85", 0.0, "ANS refusal on definitive gold (2.8%)."),
    ("doc139_qa0__after85", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc116_qa0__after85", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    ("doc135_qa0__after85", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc119_qa0__after85", 1.0, "ACK honest refusal — doc119 not yet ingested."),
]


def main() -> None:
    results_path = JUDGE_DIR / "results.jsonl"

    existing: dict[str, dict] = {}
    if results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                existing[e["qid"]] = e

    new_records: list[dict] = []
    for suffix, score, rationale in JUDGMENTS:
        qid = QID_PREFIX + suffix + QID_SUFFIX
        if qid in existing:
            continue
        new_records.append({
            "qid": qid,
            "judge_score": score,
            "rationale": rationale,
            "judge_model": "claude-opus-4.7-1m",
            "judge_protocol": "v1",
        })

    with results_path.open("a", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(existing) + len(new_records)
    print(f"Appended {len(new_records)} (skipped {len(JUDGMENTS) - len(new_records)}, total {total})")
    if new_records:
        from collections import Counter
        dist = Counter(r["judge_score"] for r in new_records)
        print(f"Score distribution: {dict(sorted(dist.items()))}")
        mean = sum(r["judge_score"] for r in new_records) / len(new_records)
        print(f"Mean judge: {mean:.4f}")
    print(f"Cell progress: {total}/1500 (={100*total/1500:.1f}%)")


if __name__ == "__main__":
    main()
