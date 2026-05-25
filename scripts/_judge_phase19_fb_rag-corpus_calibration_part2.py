"""Phase 1.9 extension: FB rag-corpus calibration part 2 (entries 150-299)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__rag-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__rag-corpus__calibration__seed42/results.jsonl")

SPECIAL: dict[str, tuple[float, str]] = {
    "doc2_qa0__after17": (0.0, "[ANS] doc2 seen. GOLD 'No, well-managed CAPEX/RoA'. PRED 'Yes, 3M capital-intensive $25,998M PP&E'. Y/N flip + fabricated specific."),
    "doc4_qa0__after18": (0.75, "[ANS] doc4 seen. GOLD 'Consumer segment shrunk 0.9% organically'. PRED 'Consumer segment' — correct segment ID but no quant."),
    "doc3_qa0__after18": (0.75, "[ANS] doc3 seen. GOLD '3M OM -1.7% from Combat Arms/PFAS/Russia'. PRED 'litigation, impairment, restructuring + growth initiatives' — partial, missing PFAS/Russia specifics."),
    "doc14_qa0__after18": (0.0, "[ANS] doc14 seen. GOLD definitive 'Yes Adobe FCF 143% to 156%'. PRED refuses. Refusal on definitive ANS."),
    "doc19_qa0__after20": (0.25, "[ANS] doc19 seen. GOLD 30.8% Amazon YoY revenue. PRED '13.2%' — wrong specific."),
    "doc18_qa0__after20": (0.25, "[ANS] doc18 seen. GOLD 93.86 Amazon DPO. PRED '36.45' — wrong specific."),
    "doc122_qa0__after21": (0.25, "[ACK] doc122 not yet seen. PRED '0' wrong specific vs GOLD $411M."),
    "doc11_qa0__after21": (0.25, "[ANS] doc11 seen. GOLD 65.4% Adobe FY15-FY16. PRED computes -21.7% using WRONG numbers ($1,493,602 for FY15 instead of FY16). Wrong specific."),
    "doc63_qa0__after21": (0.25, "[ACK] doc63 not yet seen. PRED 'commercial airlines + government agencies + defense contractors' — defense contractors fabrication."),
    "doc43_qa0__after22": (0.25, "[ACK] doc43 not yet seen. GOLD 'Customer deposits'. PRED 'Accounts payable' — confident wrong specific."),
    "doc41_qa0__after23": (1.0, "[ANS] doc41 seen. GOLD 'Performance not measured through gross margin (AMEX)'. PRED 'Gross margin not useful for AMEX (financial services, fees/interest)'. Correct reasoning."),
    "doc11_qa0__after23": (0.25, "[ANS] doc11 seen. Same -21.7% wrong specific as after21."),
    "doc15_qa0__after23": (1.0, "[ANS] doc15 seen. GOLD 0. PRED '0'. Match."),
    "doc125_qa0__after24": (0.5, "[ACK] doc125 not yet seen. GOLD 'proposal defeated'. PRED 'not approved' — equivalent meaning, confident-correct from world knowledge."),
    "doc26_qa0__after24": (0.25, "[ANS] doc26 seen. GOLD 'No, gross margin decline 0.8%'. PRED 'gross margin not useful metric for Amcor' — wrong reasoning."),
    "doc1_qa0__after24": (1.0, "[ANS] doc1 seen. GOLD $8.70. PRED '$8.738 billion' within 0.43% tolerance."),
    "doc11_qa0__after25": (0.25, "[ANS] doc11 seen. Same -21.7% wrong specific."),
    "doc26_qa0__after25": (0.25, "[ANS] doc26 seen. Same wrong reasoning."),
    "doc2_qa0__after25": (0.0, "[ANS] doc2 seen. Same Y/N flip + $25,998M PP&E."),
    "doc74_qa0__after26": (0.25, "[ACK] doc74 not yet seen. GOLD $59,268 Costco FY21. PRED '$52,694 million' — wrong specific (10.8% off, beyond tolerance)."),
    "doc20_qa0__after27": (1.0, "[ANS] doc20 seen. GOLD $11588 Amazon FY19 NI. PRED '11,588'. Exact match."),
    "doc0_qa0__after27": (1.0, "[ANS] doc0 seen. GOLD $1577 3M FY18 capex. PRED '$1,501 million' — within 4.8% (under 5% tolerance)."),
    "doc63_qa0__after28": (0.25, "[ACK] doc63 not yet seen. Same 'defense contractors' fabrication."),
    "doc41_qa0__after28": (1.0, "[ANS] doc41 seen. Same correct AMEX gross margin reasoning."),
    "doc29_qa0__after28": (0.75, "[ANS] doc29 seen. GOLD 'Real Growth flat FY2023 vs FY2022'. PRED 'not explicitly provided' — partial honesty (admits can't tell). Hedged."),
    "doc18_qa0__after29": (0.25, "[ANS] doc18 seen. GOLD 93.86. PRED '36.12' wrong specific."),
}

ENTRY_SUFFIXES: list[str] = [
    "doc80_qa0__after15", "doc81_qa0__after15", "doc26_qa0__after15", "doc46_qa0__after15",
    "doc127_qa0__after15", "doc23_qa0__after15", "doc36_qa0__after15", "doc130_qa0__after15",
    "doc48_qa0__after15", "doc34_qa0__after15",
    "doc71_qa0__after16", "doc115_qa0__after16", "doc138_qa0__after16", "doc86_qa0__after16",
    "doc136_qa0__after16", "doc145_qa0__after16", "doc89_qa0__after16", "doc105_qa0__after16",
    "doc116_qa0__after16", "doc23_qa0__after16",
    "doc103_qa0__after17", "doc73_qa0__after17", "doc124_qa0__after17", "doc18_qa0__after17",
    "doc115_qa0__after17", "doc2_qa0__after17", "doc64_qa0__after17", "doc85_qa0__after17",
    "doc74_qa0__after17", "doc33_qa0__after17",
    "doc37_qa0__after18", "doc39_qa0__after18", "doc139_qa0__after18", "doc34_qa0__after18",
    "doc109_qa0__after18", "doc4_qa0__after18", "doc49_qa0__after18", "doc3_qa0__after18",
    "doc14_qa0__after18", "doc97_qa0__after18",
    "doc136_qa0__after19", "doc113_qa0__after19", "doc57_qa0__after19", "doc59_qa0__after19",
    "doc75_qa0__after19", "doc36_qa0__after19", "doc110_qa0__after19", "doc51_qa0__after19",
    "doc119_qa0__after19", "doc138_qa0__after19",
    "doc105_qa0__after20", "doc74_qa0__after20", "doc84_qa0__after20", "doc36_qa0__after20",
    "doc83_qa0__after20", "doc19_qa0__after20", "doc140_qa0__after20", "doc61_qa0__after20",
    "doc111_qa0__after20", "doc18_qa0__after20",
    "doc122_qa0__after21", "doc113_qa0__after21", "doc91_qa0__after21", "doc11_qa0__after21",
    "doc110_qa0__after21", "doc140_qa0__after21", "doc63_qa0__after21", "doc48_qa0__after21",
    "doc87_qa0__after21", "doc68_qa0__after21",
    "doc120_qa0__after22", "doc114_qa0__after22", "doc99_qa0__after22", "doc80_qa0__after22",
    "doc45_qa0__after22", "doc68_qa0__after22", "doc53_qa0__after22", "doc84_qa0__after22",
    "doc43_qa0__after22", "doc61_qa0__after22",
    "doc48_qa0__after23", "doc66_qa0__after23", "doc63_qa0__after23", "doc113_qa0__after23",
    "doc117_qa0__after23", "doc41_qa0__after23", "doc11_qa0__after23", "doc128_qa0__after23",
    "doc119_qa0__after23", "doc15_qa0__after23",
    "doc125_qa0__after24", "doc26_qa0__after24", "doc1_qa0__after24", "doc32_qa0__after24",
    "doc61_qa0__after24", "doc126_qa0__after24", "doc134_qa0__after24", "doc53_qa0__after24",
    "doc120_qa0__after24", "doc135_qa0__after24",
    "doc59_qa0__after25", "doc139_qa0__after25", "doc134_qa0__after25", "doc83_qa0__after25",
    "doc31_qa0__after25", "doc11_qa0__after25", "doc26_qa0__after25", "doc94_qa0__after25",
    "doc2_qa0__after25", "doc49_qa0__after25",
    "doc36_qa0__after26", "doc131_qa0__after26", "doc115_qa0__after26", "doc85_qa0__after26",
    "doc118_qa0__after26", "doc77_qa0__after26", "doc110_qa0__after26", "doc63_qa0__after26",
    "doc40_qa0__after26", "doc74_qa0__after26",
    "doc102_qa0__after27", "doc124_qa0__after27", "doc39_qa0__after27", "doc105_qa0__after27",
    "doc132_qa0__after27", "doc20_qa0__after27", "doc106_qa0__after27", "doc80_qa0__after27",
    "doc0_qa0__after27", "doc104_qa0__after27",
    "doc89_qa0__after28", "doc63_qa0__after28", "doc41_qa0__after28", "doc29_qa0__after28",
    "doc124_qa0__after28", "doc109_qa0__after28", "doc106_qa0__after28", "doc39_qa0__after28",
    "doc56_qa0__after28", "doc70_qa0__after28",
    "doc147_qa0__after29", "doc135_qa0__after29", "doc124_qa0__after29", "doc97_qa0__after29",
    "doc58_qa0__after29", "doc91_qa0__after29", "doc138_qa0__after29", "doc108_qa0__after29",
    "doc71_qa0__after29", "doc18_qa0__after29",
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
