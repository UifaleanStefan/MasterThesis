"""Phase 1.9 Protocol B FB v4t-tuned calibration part 10 (entries 1350-1499) — FINAL BATCH."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__v4t-tuned__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__v4t-tuned__calibration__seed42/results.jsonl")

# All ANS refusals get 0.0; ACK refusals get default 1.0; correct/partial get explicit scores.
ANS_REFUSAL_ZERO = (0.0, "[ANS] source doc seen. PRED refuses on definitive ANS. Refusal on definitive ANS per calibration rubric.")

SPECIAL: dict[str, tuple[float, str]] = {
    "doc55_qa0__after135": ANS_REFUSAL_ZERO,
    "doc60_qa0__after135": ANS_REFUSAL_ZERO,
    "doc102_qa0__after135": ANS_REFUSAL_ZERO,
    "doc88_qa0__after135": ANS_REFUSAL_ZERO,
    "doc86_qa0__after135": ANS_REFUSAL_ZERO,
    "doc81_qa0__after135": ANS_REFUSAL_ZERO,
    "doc118_qa0__after135": ANS_REFUSAL_ZERO,
    "doc127_qa0__after135": (0.25, "[ANS] doc127 seen. GOLD $8,400M. PRED '$4,950M' — wrong specific (59% off)."),
    "doc10_qa0__after135": ANS_REFUSAL_ZERO,
    "doc115_qa0__after136": ANS_REFUSAL_ZERO,
    "doc120_qa0__after136": ANS_REFUSAL_ZERO,
    "doc27_qa0__after136": ANS_REFUSAL_ZERO,
    "doc108_qa0__after136": ANS_REFUSAL_ZERO,
    "doc2_qa0__after136": ANS_REFUSAL_ZERO,
    "doc58_qa0__after136": ANS_REFUSAL_ZERO,
    "doc80_qa0__after136": ANS_REFUSAL_ZERO,
    "doc63_qa0__after136": ANS_REFUSAL_ZERO,
    "doc103_qa0__after136": ANS_REFUSAL_ZERO,
    "doc128_qa0__after137": (1.0, "[ANS] doc128 seen. GOLD 'Pepsico strong start FY2023'. PRED 'PepsiCo raised full year guidance FY2023, organic revenue +8%, EPS +9%'. Correct + detail."),
    "doc39_qa0__after137": ANS_REFUSAL_ZERO,
    "doc60_qa0__after137": ANS_REFUSAL_ZERO,
    "doc88_qa0__after137": ANS_REFUSAL_ZERO,
    "doc134_qa0__after137": (1.0, "[ANS] doc134 seen. GOLD 'Developed Rest of the World'. PRED 'Developed Rest of World'. Match (whitespace/article only)."),
    "doc135_qa0__after137": (1.0, "[ANS] doc135 seen. GOLD 'Yes spinning off Upjohn'. PRED 'Yes Pfizer incurring $700M separating Upjohn'. Correct + detail."),
    "doc113_qa0__after137": ANS_REFUSAL_ZERO,
    "doc126_qa0__after137": (1.0, "[ANS] doc126 seen. GOLD '$400M increase'. PRED 'PepsiCo increased $400M from $3.8B to $4.2B'. Correct + detail."),
    "doc18_qa0__after137": ANS_REFUSAL_ZERO,
    "doc13_qa0__after137": ANS_REFUSAL_ZERO,
    "doc60_qa0__after138": ANS_REFUSAL_ZERO,
    "doc39_qa0__after138": ANS_REFUSAL_ZERO,
    "doc119_qa0__after138": ANS_REFUSAL_ZERO,
    "doc35_qa0__after138": ANS_REFUSAL_ZERO,
    "doc8_qa0__after138": ANS_REFUSAL_ZERO,
    "doc131_qa0__after138": ANS_REFUSAL_ZERO,
    "doc67_qa0__after138": ANS_REFUSAL_ZERO,
    "doc47_qa0__after138": ANS_REFUSAL_ZERO,
    "doc3_qa0__after138": ANS_REFUSAL_ZERO,
    "doc70_qa0__after139": ANS_REFUSAL_ZERO,
    "doc118_qa0__after139": ANS_REFUSAL_ZERO,
    "doc39_qa0__after139": ANS_REFUSAL_ZERO,
    "doc74_qa0__after139": (1.0, "[ANS] doc74 seen. GOLD $59268 Costco. PRED '59,693' — within 0.72% (under 5% tolerance)."),
    "doc12_qa0__after139": ANS_REFUSAL_ZERO,
    "doc24_qa0__after139": ANS_REFUSAL_ZERO,
    "doc25_qa0__after139": ANS_REFUSAL_ZERO,
    "doc0_qa0__after139": ANS_REFUSAL_ZERO,
    "doc92_qa0__after139": ANS_REFUSAL_ZERO,
    "doc5_qa0__after140": ANS_REFUSAL_ZERO,
    "doc135_qa0__after140": (1.0, "[ANS] doc135 seen. Same Pfizer Upjohn correct + $700M detail."),
    "doc76_qa0__after140": ANS_REFUSAL_ZERO,
    "doc26_qa0__after140": ANS_REFUSAL_ZERO,
    "doc55_qa0__after140": ANS_REFUSAL_ZERO,
    "doc58_qa0__after140": ANS_REFUSAL_ZERO,
    "doc105_qa0__after140": ANS_REFUSAL_ZERO,
    "doc31_qa0__after140": ANS_REFUSAL_ZERO,
    "doc123_qa0__after140": ANS_REFUSAL_ZERO,
    "doc3_qa0__after140": ANS_REFUSAL_ZERO,
    "doc62_qa0__after141": ANS_REFUSAL_ZERO,
    "doc3_qa0__after141": ANS_REFUSAL_ZERO,
    "doc38_qa0__after141": (0.75, "[ANS] doc38 seen. GOLD 'There are none' (AMEX debt). PRED 'do not contain info on AMEX debt securities' — equivalent conclusion via different framing."),
    "doc125_qa0__after141": (1.0, "[ANS] doc125 seen. GOLD 'Pepsi net-zero proposal defeated'. PRED 'defeated, 19,718,780 for vs 977,228,788 against'. Exact match with vote counts."),
    "doc87_qa0__after141": ANS_REFUSAL_ZERO,
    "doc63_qa0__after141": ANS_REFUSAL_ZERO,
    "doc69_qa0__after141": ANS_REFUSAL_ZERO,
    "doc124_qa0__after141": ANS_REFUSAL_ZERO,
    "doc17_qa0__after141": ANS_REFUSAL_ZERO,
    "doc34_qa0__after142": ANS_REFUSAL_ZERO,
    "doc102_qa0__after142": ANS_REFUSAL_ZERO,
    "doc127_qa0__after142": (0.5, "[ANS] doc127 seen. GOLD '$8,400M'. PRED '$4,200M + up to $4,950M with increase' — partial (covers one credit agreement, misses total $8.4B)."),
    "doc2_qa0__after142": ANS_REFUSAL_ZERO,
    "doc113_qa0__after142": ANS_REFUSAL_ZERO,
    "doc139_qa0__after142": (0.25, "[ANS] doc139 seen. GOLD '47 new stores'. PRED 'change in operating assets/liabilities, decrease of $104,233' — confident wrong (irrelevant)."),
    "doc74_qa0__after142": ANS_REFUSAL_ZERO,
    "doc132_qa0__after142": (0.5, "[ANS] doc132 seen. GOLD 'Trillium, Array, Therachon'. PRED 'Trillium, Array, Upjohn' — 2/3 correct (Therachon → Upjohn wrong)."),
    "doc107_qa0__after142": ANS_REFUSAL_ZERO,
    "doc63_qa0__after143": ANS_REFUSAL_ZERO,
    "doc45_qa0__after143": ANS_REFUSAL_ZERO,
    "doc4_qa0__after143": ANS_REFUSAL_ZERO,
    "doc141_qa0__after143": (0.0, "[ANS] doc141 seen. GOLD 'Wages expense increased FY2023'. PRED 'Decrease'. Y/N flip / direction error."),
    "doc93_qa0__after143": ANS_REFUSAL_ZERO,
    "doc134_qa0__after143": (1.0, "[ANS] doc134 seen. Same Developed Rest of World match."),
    "doc79_qa0__after143": ANS_REFUSAL_ZERO,
    "doc138_qa0__after143": (1.0, "[ANS] doc138 seen. GOLD 'Lower marketing + leverage incentive'. PRED matches verbatim + offset by deleverage. Correct."),
    "doc11_qa0__after143": ANS_REFUSAL_ZERO,
    "doc7_qa0__after143": ANS_REFUSAL_ZERO,
    "doc86_qa0__after144": ANS_REFUSAL_ZERO,
    "doc31_qa0__after144": ANS_REFUSAL_ZERO,
    "doc139_qa0__after144": (0.25, "[ANS] doc139 seen. Same wrong 'decrease of $104,233' as after142."),
    "doc44_qa0__after144": ANS_REFUSAL_ZERO,
    "doc24_qa0__after144": ANS_REFUSAL_ZERO,
    "doc97_qa0__after144": ANS_REFUSAL_ZERO,
    "doc63_qa0__after144": ANS_REFUSAL_ZERO,
    "doc110_qa0__after144": ANS_REFUSAL_ZERO,
    "doc23_qa0__after144": ANS_REFUSAL_ZERO,
    "doc78_qa0__after144": ANS_REFUSAL_ZERO,
    "doc23_qa0__after145": ANS_REFUSAL_ZERO,
    "doc110_qa0__after145": ANS_REFUSAL_ZERO,
    "doc19_qa0__after145": ANS_REFUSAL_ZERO,
    "doc20_qa0__after145": ANS_REFUSAL_ZERO,
    "doc136_qa0__after145": (0.25, "[ANS] doc136 seen. GOLD 'There are none' (Ulta debt securities). PRED 'Ulta common stock par $0.01 NASDAQ ULTA' — confident-wrong (gives info on common stock, not debt securities)."),
    "doc95_qa0__after145": ANS_REFUSAL_ZERO,
    "doc119_qa0__after145": ANS_REFUSAL_ZERO,
    "doc109_qa0__after145": ANS_REFUSAL_ZERO,
    "doc62_qa0__after145": ANS_REFUSAL_ZERO,
    "doc12_qa0__after145": ANS_REFUSAL_ZERO,
    "doc111_qa0__after146": ANS_REFUSAL_ZERO,
    "doc51_qa0__after146": ANS_REFUSAL_ZERO,
    "doc10_qa0__after146": ANS_REFUSAL_ZERO,
    "doc64_qa0__after146": ANS_REFUSAL_ZERO,
    "doc139_qa0__after146": (0.25, "[ANS] doc139 seen. Same wrong 'decrease of $104,233' as after142."),
    "doc24_qa0__after146": ANS_REFUSAL_ZERO,
    "doc98_qa0__after146": ANS_REFUSAL_ZERO,
    "doc5_qa0__after146": ANS_REFUSAL_ZERO,
    "doc13_qa0__after146": ANS_REFUSAL_ZERO,
    "doc53_qa0__after146": ANS_REFUSAL_ZERO,
    "doc25_qa0__after147": ANS_REFUSAL_ZERO,
    "doc24_qa0__after147": ANS_REFUSAL_ZERO,
    "doc35_qa0__after147": ANS_REFUSAL_ZERO,
    "doc22_qa0__after147": ANS_REFUSAL_ZERO,
    "doc117_qa0__after147": ANS_REFUSAL_ZERO,
    "doc26_qa0__after147": ANS_REFUSAL_ZERO,
    "doc141_qa0__after147": (0.0, "[ANS] doc141 seen. Same 'Decrease' Y/N flip vs gold 'increased'."),
    "doc83_qa0__after147": ANS_REFUSAL_ZERO,
    "doc102_qa0__after147": ANS_REFUSAL_ZERO,
    "doc111_qa0__after147": ANS_REFUSAL_ZERO,
    "doc140_qa0__after148": (1.0, "[ANS] doc140 seen. GOLD 36% Ulta. PRED '36.5%' — within 1.4% (under 5% tolerance)."),
    "doc107_qa0__after148": ANS_REFUSAL_ZERO,
    "doc38_qa0__after148": (0.75, "[ANS] doc38 seen. Same equivalent-conclusion partial as after141."),
    "doc59_qa0__after148": ANS_REFUSAL_ZERO,
    "doc120_qa0__after148": ANS_REFUSAL_ZERO,
    "doc127_qa0__after148": (0.5, "[ANS] doc127 seen. Same partial as after142 — covers one credit agreement, misses total $8.4B."),
    "doc77_qa0__after148": ANS_REFUSAL_ZERO,
    "doc118_qa0__after148": ANS_REFUSAL_ZERO,
    "doc85_qa0__after148": ANS_REFUSAL_ZERO,
    "doc137_qa0__after148": (0.75, "[ANS] doc137 seen. GOLD 'Ulta did not make acquisitions'. PRED 'do not contain info on Ulta acquisitions' — equivalent conclusion via different framing."),
    "doc90_qa0__after149": (1.0, "[ANS] doc90 seen. GOLD 'Consumer Health discontinued Aug 30 2023'. PRED matches with JnJ prefix. Correct."),
    "doc82_qa0__after149": ANS_REFUSAL_ZERO,
    "doc63_qa0__after149": ANS_REFUSAL_ZERO,
    "doc109_qa0__after149": ANS_REFUSAL_ZERO,
    "doc61_qa0__after149": ANS_REFUSAL_ZERO,
    "doc55_qa0__after149": ANS_REFUSAL_ZERO,
    "doc80_qa0__after149": (1.0, "[ANS] doc80 seen. GOLD 'Yes Richard A. Johnson'. PRED 'Yes Richard A. Johnson 16,105,005 against substantially more than others'. Correct + vote count."),
    "doc105_qa0__after149": (1.0, "[ANS] doc105 seen. GOLD 'Yes MGM $0.01 throughout FY2022'. PRED 'Yes MGM Resorts $0.01 throughout 2022'. Correct."),
    "doc108_qa0__after149": ANS_REFUSAL_ZERO,
    "doc128_qa0__after149": (1.0, "[ANS] doc128 seen. GOLD 'Pepsico strong start FY2023'. PRED 'PepsiCo raised full year guidance FY2023, strong start + resilience'. Correct."),
}

ENTRY_SUFFIXES: list[str] = [
    "doc55_qa0__after135", "doc60_qa0__after135", "doc102_qa0__after135", "doc88_qa0__after135",
    "doc86_qa0__after135", "doc81_qa0__after135", "doc118_qa0__after135", "doc139_qa0__after135",
    "doc127_qa0__after135", "doc10_qa0__after135",
    "doc115_qa0__after136", "doc120_qa0__after136", "doc27_qa0__after136", "doc148_qa0__after136",
    "doc108_qa0__after136", "doc2_qa0__after136", "doc58_qa0__after136", "doc80_qa0__after136",
    "doc63_qa0__after136", "doc103_qa0__after136",
    "doc128_qa0__after137", "doc39_qa0__after137", "doc60_qa0__after137", "doc88_qa0__after137",
    "doc134_qa0__after137", "doc135_qa0__after137", "doc113_qa0__after137", "doc126_qa0__after137",
    "doc18_qa0__after137", "doc13_qa0__after137",
    "doc60_qa0__after138", "doc39_qa0__after138", "doc119_qa0__after138", "doc142_qa0__after138",
    "doc35_qa0__after138", "doc8_qa0__after138", "doc131_qa0__after138", "doc67_qa0__after138",
    "doc47_qa0__after138", "doc3_qa0__after138",
    "doc148_qa0__after139", "doc70_qa0__after139", "doc118_qa0__after139", "doc39_qa0__after139",
    "doc74_qa0__after139", "doc12_qa0__after139", "doc24_qa0__after139", "doc25_qa0__after139",
    "doc0_qa0__after139", "doc92_qa0__after139",
    "doc5_qa0__after140", "doc135_qa0__after140", "doc76_qa0__after140", "doc26_qa0__after140",
    "doc55_qa0__after140", "doc58_qa0__after140", "doc105_qa0__after140", "doc31_qa0__after140",
    "doc123_qa0__after140", "doc3_qa0__after140",
    "doc62_qa0__after141", "doc3_qa0__after141", "doc38_qa0__after141", "doc143_qa0__after141",
    "doc125_qa0__after141", "doc87_qa0__after141", "doc63_qa0__after141", "doc69_qa0__after141",
    "doc124_qa0__after141", "doc17_qa0__after141",
    "doc34_qa0__after142", "doc102_qa0__after142", "doc127_qa0__after142", "doc146_qa0__after142",
    "doc2_qa0__after142", "doc113_qa0__after142", "doc139_qa0__after142", "doc74_qa0__after142",
    "doc132_qa0__after142", "doc107_qa0__after142",
    "doc63_qa0__after143", "doc45_qa0__after143", "doc4_qa0__after143", "doc141_qa0__after143",
    "doc93_qa0__after143", "doc134_qa0__after143", "doc79_qa0__after143", "doc138_qa0__after143",
    "doc11_qa0__after143", "doc7_qa0__after143",
    "doc86_qa0__after144", "doc31_qa0__after144", "doc139_qa0__after144", "doc44_qa0__after144",
    "doc24_qa0__after144", "doc97_qa0__after144", "doc63_qa0__after144", "doc110_qa0__after144",
    "doc23_qa0__after144", "doc78_qa0__after144",
    "doc23_qa0__after145", "doc110_qa0__after145", "doc19_qa0__after145", "doc20_qa0__after145",
    "doc136_qa0__after145", "doc95_qa0__after145", "doc119_qa0__after145", "doc109_qa0__after145",
    "doc62_qa0__after145", "doc12_qa0__after145",
    "doc111_qa0__after146", "doc51_qa0__after146", "doc10_qa0__after146", "doc64_qa0__after146",
    "doc139_qa0__after146", "doc24_qa0__after146", "doc98_qa0__after146", "doc5_qa0__after146",
    "doc13_qa0__after146", "doc53_qa0__after146",
    "doc25_qa0__after147", "doc24_qa0__after147", "doc35_qa0__after147", "doc22_qa0__after147",
    "doc117_qa0__after147", "doc26_qa0__after147", "doc141_qa0__after147", "doc83_qa0__after147",
    "doc102_qa0__after147", "doc111_qa0__after147",
    "doc140_qa0__after148", "doc107_qa0__after148", "doc38_qa0__after148", "doc59_qa0__after148",
    "doc120_qa0__after148", "doc127_qa0__after148", "doc77_qa0__after148", "doc118_qa0__after148",
    "doc85_qa0__after148", "doc137_qa0__after148",
    "doc90_qa0__after149", "doc82_qa0__after149", "doc63_qa0__after149", "doc109_qa0__after149",
    "doc61_qa0__after149", "doc55_qa0__after149", "doc80_qa0__after149", "doc105_qa0__after149",
    "doc108_qa0__after149", "doc128_qa0__after149",
]
assert len(ENTRY_SUFFIXES) == 150, f"expected 150 got {len(ENTRY_SUFFIXES)}"

DEFAULT_RATIONALE = "[ACK] source doc not yet seen. PRED honest refusal ('passages do not contain X'). Correctly acknowledges missing info per calibration rubric."

JUDGMENTS: list[tuple[str, float, str]] = []
for suf in ENTRY_SUFFIXES:
    if suf in SPECIAL:
        sc, ra = SPECIAL[suf]
        JUDGMENTS.append((suf, sc, ra))
    else:
        JUDGMENTS.append((suf, 1.0, DEFAULT_RATIONALE))


def main() -> None:
    existing: set[str] = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                existing.add(json.loads(line)["qid"])
            except Exception:
                continue
    added = 0
    scores: list[float] = []
    with RESULTS.open("a", encoding="utf-8") as fh:
        for suffix, score, rationale in JUDGMENTS:
            qid = f"{QID_PREFIX}{suffix}{QID_SUFFIX}"
            if qid in existing:
                continue
            fh.write(
                json.dumps(
                    {
                        "qid": qid,
                        "judge_score": float(score),
                        "rationale": rationale,
                        "judge_model": "claude-opus-4.7-1m",
                        "judge_protocol": "v1",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            added += 1
            scores.append(score)
    dist: dict[str, int] = {}
    for s in scores:
        k = f"{s:.2f}"
        dist[k] = dist.get(k, 0) + 1
    print(f"Added {added}. Dist: {dist}")
    if scores:
        print(f"Mean: {sum(scores) / len(scores):.4f}")
    total = sum(1 for ln in RESULTS.read_text(encoding="utf-8").splitlines() if ln.strip())
    print(f"Total: {total}/1500 ({100 * total / 1500:.1f}%)")


if __name__ == "__main__":
    main()
