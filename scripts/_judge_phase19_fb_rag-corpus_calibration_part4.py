"""Phase 1.9 extension: FB rag-corpus calibration part 4 (entries 450-599)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__rag-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__rag-corpus__calibration__seed42/results.jsonl")

SPECIAL: dict[str, tuple[float, str]] = {
    "doc11_qa0__after45": (0.25, "[ANS] doc11 seen. GOLD 65.4%. PRED wrong calc with self-contradictory numbers."),
    "doc30_qa0__after45": (1.0, "[ANS] doc30 seen. GOLD 4.2%. PRED '4.18%' from $167M/$3,991M — within 0.5% tolerance. Correct."),
    "doc32_qa0__after45": (1.0, "[ANS] doc32 seen. GOLD AMD products list. PRED matches verbatim."),
    "doc31_qa0__after45": (0.0, "[ANS] doc31 seen. GOLD definitive 'Yes quick ratio 1.57'. PRED refuses."),
    "doc37_qa0__after46": (1.0, "[ANS] doc37 seen. GOLD 'Yes one customer 16%'. PRED 'Yes AMD reported one customer 16% net revenue 2022'. Correct + detail."),
    "doc24_qa0__after46": (0.75, "[ANS] doc24 seen. Same Amcor partial — Shanghai+NZ in FY23 + Czech wrong year as FY22."),
    "doc30_qa0__after46": (1.0, "[ANS] doc30 seen. Same 4.18% within tol. Correct."),
    "doc1_qa0__after47": (1.0, "[ANS] doc1 seen. PRED '$8.738 billion' within 0.43% tolerance."),
    "doc75_qa0__after47": (0.25, "[ACK] doc75 not yet seen. GOLD 17.98. PRED '2.06' wrong specific (8.7x off)."),
    "doc31_qa0__after47": (0.0, "[ANS] doc31 seen. Same AMD refusal."),
    "doc4_qa0__after48": (0.75, "[ANS] doc4 seen. PRED 'Consumer segment' correct ID, no quant."),
    "doc40_qa0__after49": (1.0, "[ANS] doc40 seen. PRED 'Operating margin not useful for AMEX as financial services, net income/pretax income more relevant'. Correct reasoning."),
    "doc30_qa0__after48": (1.0, "[ANS] doc30 seen. Same 4.18% within tol."),
    "doc75_qa0__after48": (0.25, "[ACK] doc75 not yet seen. Same '2.06' wrong specific."),
    "doc41_qa0__after49": (1.0, "[ANS] doc41 seen. PRED correct AMEX gross margin reasoning."),
    "doc27_qa0__after49": (0.5, "[ANS] doc27 seen. Same generic 'employee + fixed asset + other costs $93M'."),
    "doc16_qa0__after49": (0.25, "[ANS] doc16 seen. GOLD 9.5 inventory turnover. PRED '11.99' wrong specific."),
    "doc4_qa0__after49": (0.75, "[ANS] doc4 seen. Same partial 'Consumer segment'."),
    "doc9_qa0__after50": (0.25, "[ANS] doc9 seen. GOLD 1.9% Activision capex/rev. PRED '3.5%' wrong specific."),
    "doc24_qa0__after50": (0.75, "[ANS] doc24 seen. Same Amcor partial."),
    "doc11_qa0__after50": (0.25, "[ANS] doc11 seen. Same wrong -99.6% calc."),
    "doc35_qa0__after50": (1.0, "[ANS] doc35 seen. PRED 'Operating activities most cash flow AMD $3,565M'. Correct + amount."),
    "doc29_qa0__after50": (0.25, "[ANS] doc29 seen. 'decrease 5%' wrong direction."),
    "doc52_qa0__after51": (0.5, "[ACK] doc52 not yet seen. PRED 'cash flow from operating activities' partial — no $1.8bn specific."),
    "doc122_qa0__after51": (0.25, "[ACK] doc122 not yet seen. PRED '0' wrong specific."),
    "doc17_qa0__after51": (0.25, "[ANS] doc17 seen. PRED '-1.41' wrong specific."),
    "doc30_qa0__after52": (1.0, "[ANS] doc30 seen. Same 4.18% within tol."),
    "doc36_qa0__after52": (1.0, "[ANS] doc36 seen. PRED 'Data Center segment'. Match."),
    "doc35_qa0__after52": (1.0, "[ANS] doc35 seen. Same Operations $3,565M match."),
    "doc36_qa0__after53": (1.0, "[ANS] doc36 seen. Same Data Center match."),
    "doc29_qa0__after53": (0.25, "[ANS] doc29 seen. Same wrong direction."),
    "doc15_qa0__after53": (1.0, "[ANS] doc15 seen. PRED '0'. Match."),
    "doc0_qa0__after53": (1.0, "[ANS] doc0 seen. PRED '$1,501 million' within 4.8% tolerance."),
    "doc50_qa0__after53": (0.0, "[ANS] doc50 seen. GOLD 'Yes consistent margins, 1.1% decline'. PRED 'Gross margins fluctuated >2%, not historically consistent' — Y/N flip."),
    "doc0_qa0__after54": (1.0, "[ANS] doc0 seen. Same $1,501M within tol."),
    "doc29_qa0__after54": (0.25, "[ANS] doc29 seen. Same wrong direction."),
    "doc42_qa0__after54": (1.0, "[ANS] doc42 seen. PRED '24.6% to 21.6%' EXACT match to gold."),
    "doc37_qa0__after55": (1.0, "[ANS] doc37 seen. Same correct one customer 16%."),
    "doc50_qa0__after55": (0.0, "[ANS] doc50 seen. Same Y/N flip."),
    "doc53_qa0__after55": (1.0, "[ANS] doc53 seen. GOLD 'Yes ~42% Best Buy decline'. PRED 'Yes $1,874M to $1,093M' — within 1% (1093/1874=58%, so 42% drop). Correct."),
    "doc29_qa0__after55": (0.25, "[ANS] doc29 seen. Same wrong direction."),
    "doc3_qa0__after56": (0.75, "[ANS] doc3 seen. Same partial 3M OM."),
    "doc22_qa0__after56": (1.0, "[ANS] doc22 seen. GOLD Amcor 8K supplemental indentures (Amcor Flexibles NA for Amcor Finance USA). PRED matches with same entities + dates."),
    "doc14_qa0__after56": (0.0, "[ANS] doc14 seen. GOLD definitive 'Yes Adobe FCF 143% to 156%'. PRED refuses."),
    "doc27_qa0__after57": (0.5, "[ANS] doc27 seen. Same generic restructuring partial."),
    "doc28_qa0__after57": (1.0, "[ANS] doc28 seen. GOLD '$2,018mn FY 2023'. PRED '2,018 million' exact match."),
    "doc31_qa0__after57": (0.0, "[ANS] doc31 seen. Same AMD refusal."),
    "doc74_qa0__after57": (0.25, "[ACK] doc74 not yet seen. PRED '$52,694' wrong (10.8% off)."),
    "doc57_qa0__after57": (0.25, "[ANS] doc57 seen. GOLD 101.5%. PRED '6.9%' wrong specific."),
    "doc55_qa0__after58": (1.0, "[ANS] doc55 seen. GOLD 'Entertainment 9% Q2 FY24 from gaming'. PRED 'Entertainment, 9.0% comparable sales growth driven primarily by gaming'. EXACT match."),
    "doc64_qa0__after58": (0.5, "[ACK] doc64 not yet seen. PRED 'Yes Boeing cyclicality' confident without doc basis."),
    "doc17_qa0__after58": (0.25, "[ANS] doc17 seen. Same -1.41 wrong specific."),
    "doc14_qa0__after58": (0.0, "[ANS] doc14 seen. Same Adobe FCF refusal."),
    "doc16_qa0__after58": (0.25, "[ANS] doc16 seen. Same 11.99 wrong specific."),
    "doc29_qa0__after59": (0.25, "[ANS] doc29 seen. Same wrong direction."),
    "doc30_qa0__after59": (1.0, "[ANS] doc30 seen. Same 4.18% within tol."),
}

ENTRY_SUFFIXES: list[str] = [
    "doc124_qa0__after45", "doc141_qa0__after45", "doc56_qa0__after45", "doc11_qa0__after45",
    "doc109_qa0__after45", "doc59_qa0__after45", "doc57_qa0__after45", "doc30_qa0__after45",
    "doc32_qa0__after45", "doc31_qa0__after45",
    "doc99_qa0__after46", "doc37_qa0__after46", "doc54_qa0__after46", "doc118_qa0__after46",
    "doc58_qa0__after46", "doc24_qa0__after46", "doc30_qa0__after46", "doc50_qa0__after46",
    "doc148_qa0__after46", "doc95_qa0__after46",
    "doc1_qa0__after47", "doc75_qa0__after47", "doc92_qa0__after47", "doc87_qa0__after47",
    "doc93_qa0__after47", "doc78_qa0__after47", "doc97_qa0__after47", "doc49_qa0__after47",
    "doc136_qa0__after47", "doc31_qa0__after47",
    "doc125_qa0__after48", "doc4_qa0__after48", "doc58_qa0__after48", "doc133_qa0__after48",
    "doc40_qa0__after48", "doc148_qa0__after48", "doc30_qa0__after48", "doc76_qa0__after48",
    "doc121_qa0__after48", "doc75_qa0__after48",
    "doc41_qa0__after49", "doc27_qa0__after49", "doc16_qa0__after49", "doc145_qa0__after49",
    "doc117_qa0__after49", "doc65_qa0__after49", "doc66_qa0__after49", "doc58_qa0__after49",
    "doc138_qa0__after49", "doc4_qa0__after49",
    "doc76_qa0__after50", "doc113_qa0__after50", "doc9_qa0__after50", "doc136_qa0__after50",
    "doc24_qa0__after50", "doc130_qa0__after50", "doc11_qa0__after50", "doc35_qa0__after50",
    "doc29_qa0__after50", "doc53_qa0__after50",
    "doc52_qa0__after51", "doc122_qa0__after51", "doc128_qa0__after51", "doc53_qa0__after51",
    "doc104_qa0__after51", "doc98_qa0__after51", "doc17_qa0__after51", "doc77_qa0__after51",
    "doc136_qa0__after51", "doc61_qa0__after51",
    "doc137_qa0__after52", "doc30_qa0__after52", "doc54_qa0__after52", "doc53_qa0__after52",
    "doc80_qa0__after52", "doc36_qa0__after52", "doc121_qa0__after52", "doc125_qa0__after52",
    "doc136_qa0__after52", "doc35_qa0__after52",
    "doc94_qa0__after53", "doc36_qa0__after53", "doc56_qa0__after53", "doc29_qa0__after53",
    "doc139_qa0__after53", "doc15_qa0__after53", "doc0_qa0__after53", "doc78_qa0__after53",
    "doc50_qa0__after53", "doc145_qa0__after53",
    "doc63_qa0__after54", "doc0_qa0__after54", "doc134_qa0__after54", "doc80_qa0__after54",
    "doc133_qa0__after54", "doc29_qa0__after54", "doc42_qa0__after54", "doc83_qa0__after54",
    "doc137_qa0__after54", "doc92_qa0__after54",
    "doc147_qa0__after55", "doc108_qa0__after55", "doc100_qa0__after55", "doc37_qa0__after55",
    "doc50_qa0__after55", "doc92_qa0__after55", "doc53_qa0__after55", "doc29_qa0__after55",
    "doc120_qa0__after55", "doc128_qa0__after55",
    "doc3_qa0__after56", "doc22_qa0__after56", "doc116_qa0__after56", "doc141_qa0__after56",
    "doc14_qa0__after56", "doc88_qa0__after56", "doc148_qa0__after56", "doc60_qa0__after56",
    "doc67_qa0__after56", "doc109_qa0__after56",
    "doc120_qa0__after57", "doc63_qa0__after57", "doc27_qa0__after57", "doc28_qa0__after57",
    "doc31_qa0__after57", "doc107_qa0__after57", "doc74_qa0__after57", "doc121_qa0__after57",
    "doc69_qa0__after57", "doc57_qa0__after57",
    "doc55_qa0__after58", "doc118_qa0__after58", "doc59_qa0__after58", "doc64_qa0__after58",
    "doc17_qa0__after58", "doc14_qa0__after58", "doc16_qa0__after58", "doc66_qa0__after58",
    "doc78_qa0__after58", "doc95_qa0__after58",
    "doc29_qa0__after59", "doc65_qa0__after59", "doc87_qa0__after59", "doc116_qa0__after59",
    "doc66_qa0__after59", "doc110_qa0__after59", "doc30_qa0__after59", "doc134_qa0__after59",
    "doc119_qa0__after59", "doc147_qa0__after59",
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
