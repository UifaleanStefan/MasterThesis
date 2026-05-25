"""Claude manual judging — Phase 1.9 FB calibration v4t-canonical (entries 809-955).

Manually judged 1-by-1 per HARD RULE.
"""

from __future__ import annotations

import json
from pathlib import Path

JUDGE_DIR = Path(
    "results/stage3/judge_queue/financebench__v4t-canonical__calibration__seed42"
)
QID_PREFIX = "financebench__v4t-canonical__calibration__"
QID_SUFFIX = "__seed42"

JUDGMENTS: list[tuple[str, float, str]] = [
    # 809-819
    ("doc43_qa0__after80", 0.25, "ANS gold Customer deposits; predicted 'long-term debt' — confident wrong specific (no $ value)."),
    ("doc30_qa0__after81", 0.0, "ANS refusal on definitive gold."),
    ("doc75_qa0__after81", 0.0, "ANS refusal on definitive gold."),
    ("doc79_qa0__after81", 1.0, "ANS Mary Dillon Ulta similar — match."),
    ("doc2_qa0__after81", 0.0, "ANS refusal on definitive gold."),
    ("doc138_qa0__after81", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc60_qa0__after81", 1.0, "ANS Commercial Airplanes — match."),
    ("doc23_qa0__after81", 0.0, "ANS refusal on definitive gold."),
    ("doc59_qa0__after81", 0.0, "ANS refusal on definitive gold."),
    ("doc98_qa0__after81", 1.0, "ACK honest refusal — doc98 not yet ingested."),
    ("doc106_qa0__after81", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    # 820-829
    ("doc79_qa0__after82", 1.0, "ANS Mary Dillon — match."),
    ("doc12_qa0__after82", 0.0, "ANS refusal on definitive gold."),
    ("doc125_qa0__after82", 1.0, "ACK 'proposal not approved' = 'defeated' — correct."),
    ("doc28_qa0__after82", 0.0, "ANS AMCOR EBITDA refusal on definitive gold."),
    ("doc35_qa0__after82", 0.0, "ANS refusal on definitive gold."),
    ("doc27_qa0__after82", 0.0, "ANS refusal on definitive gold."),
    ("doc43_qa0__after82", 0.0, "ANS refusal on definitive gold."),
    ("doc101_qa0__after82", 1.0, "ACK honest refusal — doc101 not yet ingested."),
    ("doc71_qa0__after82", 0.0, "ANS gold 10.3%; predicted 13.3% — outside tolerance."),
    ("doc144_qa0__after82", 1.0, "ACK honest refusal — doc144 not yet ingested."),
    # 830-839
    ("doc39_qa0__after83", 0.0, "ANS refusal on definitive gold."),
    ("doc3_qa0__after83", 0.0, "ANS refusal on definitive gold."),
    ("doc54_qa0__after83", 1.0, "ANS gold 982→969 Best Buy; predicted same numbers — match."),
    ("doc42_qa0__after83", 0.25, "ANS gold 24.6%→21.6% AMEX; predicted '20%→23%' — confident wrong specific."),
    ("doc144_qa0__after83", 1.0, "ACK honest refusal — doc144 not yet ingested."),
    ("doc126_qa0__after83", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    ("doc90_qa0__after83", 1.0, "ACK Consumer Health discontinued — correct by inference."),
    ("doc17_qa0__after83", 0.0, "ANS refusal on definitive gold."),
    ("doc46_qa0__after83", 0.0, "ANS gold $1,832 definitive; predicted refusal — refusal on definitive gold."),
    ("doc57_qa0__after83", 0.0, "ANS refusal on definitive gold."),
    # 840-849
    ("doc148_qa0__after84", 1.0, "ACK honest refusal — doc148 not yet ingested."),
    ("doc46_qa0__after84", 0.0, "ANS refusal on definitive gold."),
    ("doc84_qa0__after84", 0.0, "ANS gold 0.54; predicted 0.11 — wrong specific."),
    ("doc12_qa0__after84", 0.0, "ANS refusal on definitive gold."),
    ("doc77_qa0__after84", 0.75, "ANS gold CVS legal battles; predicted Yes CVS lawsuits regarding drug pricing + usual customary pricing — partial match (similar topics)."),
    ("doc58_qa0__after84", 0.0, "ANS refusal on definitive gold."),
    ("doc29_qa0__after84", 0.0, "ANS gold flat real growth; predicted refusal — refusal on definitive gold."),
    ("doc124_qa0__after84", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    ("doc13_qa0__after84", 0.0, "ANS gold 'No declined 2.2%' definitive; predicted refusal — refusal on definitive gold."),
    ("doc8_qa0__after84", 0.0, "ANS refusal on definitive gold."),
    # 850-859
    ("doc18_qa0__after85", 0.0, "ANS refusal on definitive gold."),
    ("doc131_qa0__after85", 1.0, "ACK honest refusal — doc131 not yet ingested."),
    ("doc67_qa0__after85", 0.0, "ANS refusal on definitive gold."),
    ("doc11_qa0__after85", 0.0, "ANS refusal on definitive gold."),
    ("doc118_qa0__after85", 1.0, "ACK honest refusal — doc118 not yet ingested."),
    ("doc48_qa0__after85", 0.0, "ANS refusal on definitive gold."),
    ("doc139_qa0__after85", 1.0, "ACK honest refusal — doc139 not yet ingested."),
    ("doc116_qa0__after85", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    ("doc135_qa0__after85", 1.0, "ACK honest refusal — doc135 not yet ingested."),
    ("doc119_qa0__after85", 1.0, "ACK honest refusal — doc119 not yet ingested."),
    # 860-869
    ("doc48_qa0__after86", 0.0, "ANS refusal on definitive gold."),
    ("doc46_qa0__after86", 0.0, "ANS refusal on definitive gold."),
    ("doc84_qa0__after86", 0.0, "ANS gold 0.54; predicted 0.11 — wrong specific."),
    ("doc4_qa0__after86", 0.0, "ANS refusal on definitive gold."),
    ("doc40_qa0__after86", 0.0, "ANS gold 'not measured through op margin' definitive; predicted refusal — refusal on definitive gold."),
    ("doc26_qa0__after86", 0.0, "ANS refusal on definitive gold."),
    ("doc109_qa0__after86", 1.0, "ACK honest refusal — doc109 not yet ingested."),
    ("doc116_qa0__after86", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    ("doc138_qa0__after86", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc76_qa0__after86", 1.0, "ANS Yes CVS capital-intensive — direction match."),
    # 870-879
    ("doc12_qa0__after87", 0.0, "ANS refusal on definitive gold."),
    ("doc138_qa0__after87", 1.0, "ACK honest refusal — doc138 not yet ingested."),
    ("doc43_qa0__after87", 0.0, "ANS refusal on definitive gold."),
    ("doc108_qa0__after87", 1.0, "ACK honest refusal — doc108 not yet ingested."),
    ("doc59_qa0__after87", 0.0, "ANS refusal on definitive gold."),
    ("doc4_qa0__after87", 0.0, "ANS refusal on definitive gold."),
    ("doc92_qa0__after87", 0.25, "ACK calibration: confident wrong '$3.5B Kenvue cash' (gold $13.2B)."),
    ("doc16_qa0__after87", 0.0, "ANS refusal on definitive gold."),
    ("doc91_qa0__after87", 1.0, "ACK honest refusal — doc91 not yet ingested."),
    ("doc124_qa0__after87", 1.0, "ACK honest refusal — doc124 not yet ingested."),
    # 880-889
    ("doc22_qa0__after88", 0.0, "ANS Amcor 8K refusal on definitive gold."),
    ("doc27_qa0__after88", 0.0, "ANS refusal on definitive gold."),
    ("doc25_qa0__after88", 0.0, "ANS Amcor packaging refusal on definitive gold."),
    ("doc149_qa0__after88", 1.0, "ACK honest refusal — doc149 not yet ingested."),
    ("doc146_qa0__after88", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc66_qa0__after88", 0.0, "ANS refusal on definitive gold."),
    ("doc60_qa0__after88", 1.0, "ANS Commercial Airplanes $25,867M — match."),
    ("doc117_qa0__after88", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc21_qa0__after88", 0.0, "ANS gold $1,616 definitive; predicted refusal — refusal on definitive gold."),
    ("doc113_qa0__after88", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    # 890-899
    ("doc34_qa0__after89", 1.0, "ANS Xilinx amortization — match."),
    ("doc129_qa0__after89", 1.0, "ACK honest refusal — doc129 not yet ingested."),
    ("doc89_qa0__after89", 0.25, "ANS gold US 3.0% intl -0.6%; predicted 'US 6.9%, intl -1.3% total' — confident wrong specifics."),
    ("doc43_qa0__after89", 0.0, "ANS refusal on definitive gold."),
    ("doc101_qa0__after89", 1.0, "ACK honest refusal — doc101 not yet ingested."),
    ("doc75_qa0__after89", 0.0, "ANS refusal on definitive gold."),
    ("doc58_qa0__after89", 0.0, "ANS refusal on definitive gold."),
    ("doc111_qa0__after89", 1.0, "ACK honest refusal — doc111 not yet ingested."),
    ("doc83_qa0__after89", 0.0, "ANS refusal on definitive gold."),
    ("doc2_qa0__after89", 0.0, "ANS refusal on definitive gold."),
    # 900-909
    ("doc66_qa0__after90", 0.0, "ANS refusal on definitive gold."),
    ("doc113_qa0__after90", 1.0, "ACK honest refusal — doc113 not yet ingested."),
    ("doc30_qa0__after90", 0.0, "ANS refusal on definitive gold."),
    ("doc116_qa0__after90", 1.0, "ACK honest refusal — doc116 not yet ingested."),
    ("doc41_qa0__after90", 0.0, "ANS refusal on definitive gold."),
    ("doc45_qa0__after90", 0.0, "ANS refusal on definitive gold."),
    ("doc5_qa0__after90", 0.0, "ANS refusal on definitive gold."),
    ("doc91_qa0__after90", 1.0, "ACK honest refusal — doc91 not yet ingested."),
    ("doc125_qa0__after90", 1.0, "ACK honest refusal — doc125 not yet ingested."),
    ("doc126_qa0__after90", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    # 910-919
    ("doc96_qa0__after91", 1.0, "ACK JPM gross margins not relevant — correct by inference."),
    ("doc88_qa0__after91", 0.25, "ANS gold 'No decelerate 3.6%→3.5%'; predicted 'Yes 3.5%' — gets 3.5% but Yes direction wrong."),
    ("doc79_qa0__after91", 1.0, "ANS Mary Dillon Foot Locker CEO from Ulta — match."),
    ("doc33_qa0__after91", 1.0, "ANS AMD FY22 EPYC + Gaming + Embedded — match."),
    ("doc20_qa0__after91", 1.0, "ANS $11,588 Amazon FY19 net income — exact."),
    ("doc40_qa0__after91", 0.0, "ANS gold 'not measured through op margin' definitive; predicted refusal — refusal on definitive gold."),
    ("doc86_qa0__after91", 1.0, "ANS gold COVID-19 + currency + commodity inflation; predicted same — match."),
    ("doc15_qa0__after91", 1.0, "ANS 0 — exact."),
    ("doc99_qa0__after91", 1.0, "ACK honest refusal — doc99 not yet ingested."),
    ("doc18_qa0__after91", 0.0, "ANS refusal on definitive gold."),
    # 920-929
    ("doc101_qa0__after92", 1.0, "ACK honest refusal — doc101 not yet ingested."),
    ("doc45_qa0__after92", 0.0, "ANS refusal on definitive gold."),
    ("doc114_qa0__after92", 1.0, "ACK honest refusal — doc114 not yet ingested."),
    ("doc78_qa0__after92", 0.5, "ANS gold Yes $0.55/quarter; predicted Yes paid dividends Q2 — partial."),
    ("doc91_qa0__after92", 0.0, "ANS gold $20B definitive; predicted refusal — refusal on definitive gold."),
    ("doc10_qa0__after92", 0.0, "ANS refusal on definitive gold."),
    ("doc12_qa0__after92", 0.0, "ANS refusal on definitive gold."),
    ("doc94_qa0__after92", 1.0, "ACK honest refusal — doc94 not yet ingested."),
    ("doc86_qa0__after92", 1.0, "ANS COVID-19 + currency + commodity — match."),
    ("doc122_qa0__after92", 0.25, "ACK '0' confident wrong."),
    # 930-939
    ("doc26_qa0__after93", 0.0, "ANS refusal on definitive gold."),
    ("doc64_qa0__after93", 1.0, "ANS Yes cyclical — match."),
    ("doc146_qa0__after93", 1.0, "ACK honest refusal — doc146 not yet ingested."),
    ("doc136_qa0__after93", 1.0, "ACK honest refusal — doc136 not yet ingested."),
    ("doc54_qa0__after93", 0.25, "ANS gold 982→969; predicted '930→907' — wrong specific numbers, direction right."),
    ("doc106_qa0__after93", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc149_qa0__after93", 1.0, "ACK honest refusal — doc149 not yet ingested."),
    ("doc144_qa0__after93", 1.0, "ACK honest refusal — doc144 not yet ingested."),
    ("doc143_qa0__after93", 1.0, "ACK honest refusal — doc143 not yet ingested."),
    ("doc82_qa0__after93", 0.0, "ANS gold 0.68; predicted 1.00 — wrong specific."),
    # 940-949
    ("doc18_qa0__after94", 0.0, "ANS refusal on definitive gold."),
    ("doc126_qa0__after94", 1.0, "ACK honest refusal — doc126 not yet ingested."),
    ("doc52_qa0__after94", 0.0, "ANS refusal on definitive gold."),
    ("doc9_qa0__after94", 0.0, "ANS refusal on definitive gold."),
    ("doc64_qa0__after94", 1.0, "ANS Yes cyclical — match."),
    ("doc117_qa0__after94", 1.0, "ACK honest refusal — doc117 not yet ingested."),
    ("doc129_qa0__after94", 0.25, "ACK '2 percentage points' confident wrong."),
    ("doc83_qa0__after94", 0.0, "ANS refusal on definitive gold."),
    ("doc112_qa0__after94", 1.0, "ACK honest refusal — doc112 not yet ingested."),
    ("doc104_qa0__after94", 1.0, "ACK honest refusal — doc104 not yet ingested."),
    # 950-955
    ("doc18_qa0__after95", 0.0, "ANS refusal on definitive gold."),
    ("doc80_qa0__after95", 1.0, "ANS Richard A. Johnson — match."),
    ("doc52_qa0__after95", 0.0, "ANS refusal on definitive gold."),
    ("doc100_qa0__after95", 1.0, "ACK honest refusal — doc100 not yet ingested."),
    ("doc106_qa0__after95", 1.0, "ACK honest refusal — doc106 not yet ingested."),
    ("doc51_qa0__after95", 0.0, "ANS Best Buy acquisitions refusal on definitive gold."),
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
