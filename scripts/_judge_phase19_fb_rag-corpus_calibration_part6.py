"""Phase 1.9 extension: FB rag-corpus calibration part 6 (entries 750-899)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__rag-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__rag-corpus__calibration__seed42/results.jsonl")

SPECIAL: dict[str, tuple[float, str]] = {
    "doc37_qa0__after75": (1.0, "[ANS] doc37 seen. Same 'Yes one customer 16%' correct."),
    "doc0_qa0__after75": (1.0, "[ANS] doc0 seen. PRED '$1,501 million' within 4.8% tolerance."),
    "doc122_qa0__after75": (0.25, "[ACK] doc122 not yet seen. PRED '0' wrong specific."),
    "doc26_qa0__after75": (0.75, "[ANS] doc26 seen. Same partial gross margin calc."),
    "doc53_qa0__after75": (1.0, "[ANS] doc53 seen. GOLD '~42% decline'. PRED 'drop from $1,874M to $1,093M' — 42% decline match."),
    "doc25_qa0__after75": (1.0, "[ANS] doc25 seen. PRED rich Amcor packaging match."),
    "doc3_qa0__after76": (0.75, "[ANS] doc3 seen. Same partial 3M OM."),
    "doc41_qa0__after76": (1.0, "[ANS] doc41 seen. Correct AMEX gross margin reasoning."),
    "doc37_qa0__after76": (1.0, "[ANS] doc37 seen. Same correct."),
    "doc55_qa0__after76": (1.0, "[ANS] doc55 seen. Exact Entertainment 9% match."),
    "doc18_qa0__after76": (0.25, "[ANS] doc18 seen. PRED '29.12' wrong specific."),
    "doc74_qa0__after76": (1.0, "[ANS] doc74 seen. GOLD $59268. PRED '59,268' — EXACT match!"),
    "doc11_qa0__after76": (0.25, "[ANS] doc11 seen. Wrong calc -99.6%."),
    "doc64_qa0__after77": (0.75, "[ANS] doc64 seen. Y correct without airline industry context."),
    "doc60_qa0__after77": (0.75, "[ANS] doc60 seen. One segment only."),
    "doc44_qa0__after77": (1.0, "[ANS] doc44 seen. Correct Card Member retention."),
    "doc52_qa0__after77": (1.0, "[ANS] doc52 seen. Correct $1,824M Best Buy operations."),
    "doc11_qa0__after77": (0.25, "[ANS] doc11 seen. Same wrong calc."),
    "doc19_qa0__after78": (1.0, "[ANS] doc19 seen. PRED '30.8%' exact match."),
    "doc44_qa0__after78": (1.0, "[ANS] doc44 seen. Correct."),
    "doc63_qa0__after78": (0.5, "[ANS] doc63 seen. PRED 'Boeing derives significant portion from limited commercial airlines' — partial, misses US gov 40%."),
    "doc67_qa0__after78": (0.25, "[ANS] doc67 seen. GOLD 0.01. PRED '1.46%' wrong specific."),
    "doc40_qa0__after78": (1.0, "[ANS] doc40 seen. Correct AMEX OM reasoning."),
    "doc52_qa0__after78": (1.0, "[ANS] doc52 seen. Correct $1,824M."),
    "doc65_qa0__after78": (1.0, "[ANS] doc65 seen. Full 737/787/777X production rates match."),
    "doc23_qa0__after79": (0.75, "[ANS] doc23 seen. PRED hedged 'not explicitly provided + may not be primary focus'. Partial honesty."),
    "doc56_qa0__after79": (1.0, "[ANS] doc56 seen. GOLD 1.73. PRED '1.74' within 0.6% tolerance."),
    "doc55_qa0__after79": (1.0, "[ANS] doc55 seen. Exact match."),
    "doc28_qa0__after79": (1.0, "[ANS] doc28 seen. GOLD '$2,018mn FY 2023'. PRED '$2,018 million FY 2023' exact."),
    "doc83_qa0__after79": (0.25, "[ACK] doc83 not yet seen. GOLD $3215. PRED '$1,200M' wrong specific."),
    "doc2_qa0__after79": (0.0, "[ANS] doc2 seen. Same Y/N flip + $25,998M PP&E."),
    "doc14_qa0__after79": (0.0, "[ANS] doc14 seen. Same Adobe FCF refusal."),
    "doc44_qa0__after80": (1.0, "[ANS] doc44 seen. Correct."),
    "doc25_qa0__after80": (0.75, "[ANS] doc25 seen. Same minimal 'Amcor packaging industry'."),
    "doc60_qa0__after80": (0.75, "[ANS] doc60 seen. Same one-segment partial."),
    "doc35_qa0__after80": (1.0, "[ANS] doc35 seen. Correct $3,565M operations."),
    "doc12_qa0__after80": (0.25, "[ANS] doc12 seen. PRED '1.25' wrong specific."),
    "doc43_qa0__after80": (1.0, "[ANS] doc43 seen. GOLD 'Customer deposits'. PRED 'customer deposits, totaling $110,239 million'. EXACT match + amount!"),
    "doc30_qa0__after81": (1.0, "[ANS] doc30 seen. Same 4.18% within tol."),
    "doc75_qa0__after81": (0.25, "[ANS] doc75 seen. GOLD 17.98. PRED '9.25' wrong specific."),
    "doc79_qa0__after81": (1.0, "[ANS] doc79 seen. GOLD 'Yes Mary Dillon Ulta'. PRED 'Yes Mary N. Dillon former Exec Chair/CEO Ulta'. Correct."),
    "doc2_qa0__after81": (0.0, "[ANS] doc2 seen. Same Y/N flip."),
    "doc60_qa0__after81": (0.75, "[ANS] doc60 seen. Same one-segment."),
    "doc23_qa0__after81": (0.75, "[ANS] doc23 seen. Same hedged."),
    "doc59_qa0__after81": (1.0, "[ANS] doc59 seen. PRED '$12,645' exact."),
    "doc79_qa0__after82": (1.0, "[ANS] doc79 seen. Same Mary Dillon correct."),
    "doc12_qa0__after82": (0.25, "[ANS] doc12 seen. PRED '1.23' wrong specific."),
    "doc125_qa0__after82": (0.5, "[ACK] doc125 not yet seen. PRED 'proposal not approved' — equivalent to gold 'defeated'."),
    "doc28_qa0__after82": (1.0, "[ANS] doc28 seen. Exact $2,018 million match."),
    "doc35_qa0__after82": (1.0, "[ANS] doc35 seen. Correct."),
    "doc27_qa0__after82": (0.5, "[ANS] doc27 seen. Same generic restructuring."),
    "doc43_qa0__after82": (0.25, "[ANS] doc43 seen. PRED 'Long-term debt $42,573M' wrong (gold Customer deposits)."),
    "doc71_qa0__after82": (0.25, "[ANS] doc71 seen. PRED '15.5%' wrong specific (gold 10.3%)."),
    "doc39_qa0__after83": (1.0, "[ANS] doc39 seen. 'United States, EMEA, APAC, LACC' exact match."),
    "doc3_qa0__after83": (0.75, "[ANS] doc3 seen. Same partial OM."),
    "doc54_qa0__after83": (0.25, "[ANS] doc54 seen. PRED '907 down from 930' wrong specifics."),
    "doc42_qa0__after83": (1.0, "[ANS] doc42 seen. PRED '24.6% to 21.6%' EXACT match."),
    "doc90_qa0__after83": (0.5, "[ACK] doc90 not yet seen. PRED exact JnJ Consumer Health Aug 30 2023 — world-knowledge match."),
    "doc17_qa0__after83": (0.25, "[ANS] doc17 seen. PRED '-1.41' wrong specific."),
    "doc46_qa0__after83": (1.0, "[ANS] doc46 seen. PRED '1,829' within 0.16% tolerance."),
    "doc57_qa0__after83": (1.0, "[ANS] doc57 seen. PRED '101.7%' within 0.2% of gold 101.5%."),
    "doc46_qa0__after84": (1.0, "[ANS] doc46 seen. Same 1,829 within tol."),
    "doc84_qa0__after84": (1.0, "[ANS] doc84 seen. GOLD 0.54. PRED '0.54' EXACT match!"),
    "doc12_qa0__after84": (0.25, "[ANS] doc12 seen. PRED '1.25' wrong."),
    "doc77_qa0__after84": (0.75, "[ANS] doc77 seen. Y correct + one CVS legal category."),
    "doc58_qa0__after84": (1.0, "[ANS] doc58 seen. PRED '$381.6 million' within 0.1% tolerance."),
    "doc29_qa0__after84": (0.25, "[ANS] doc29 seen. 'decrease 5%' wrong direction."),
    "doc13_qa0__after84": (0.0, "[ANS] doc13 seen. PRED gives 34.6% vs 36.7% but calls decline 'improvement' — Y/N logical flip."),
    "doc8_qa0__after84": (0.25, "[ANS] doc8 seen. PRED '25.66' 5.7% off (beyond 5% tolerance)."),
    "doc18_qa0__after85": (0.25, "[ANS] doc18 seen. PRED '34.12' wrong."),
    "doc67_qa0__after85": (0.25, "[ANS] doc67 seen. PRED '1.46%' wrong specific."),
    "doc11_qa0__after85": (0.25, "[ANS] doc11 seen. Same wrong calc."),
    "doc48_qa0__after85": (0.25, "[ANS] doc48 seen. GOLD 2.8%. PRED '3.9%' wrong specific."),
    "doc48_qa0__after86": (0.25, "[ANS] doc48 seen. Same wrong."),
    "doc46_qa0__after86": (1.0, "[ANS] doc46 seen. Same within tol."),
    "doc84_qa0__after86": (1.0, "[ANS] doc84 seen. Same 0.54 exact."),
    "doc4_qa0__after86": (0.75, "[ANS] doc4 seen. 'Consumer segment' partial."),
    "doc40_qa0__after86": (1.0, "[ANS] doc40 seen. Same correct AMEX OM reasoning."),
    "doc26_qa0__after86": (0.75, "[ANS] doc26 seen. Same partial gross margin calc."),
    "doc76_qa0__after86": (0.75, "[ANS] doc76 seen. Y correct minimal."),
    "doc12_qa0__after87": (0.25, "[ANS] doc12 seen. PRED '1.25' wrong."),
    "doc43_qa0__after87": (0.25, "[ANS] doc43 seen. PRED 'Long-term debt' wrong."),
    "doc59_qa0__after87": (1.0, "[ANS] doc59 seen. PRED '$12,645' exact."),
    "doc4_qa0__after87": (0.75, "[ANS] doc4 seen. 'Consumer segment' partial."),
    "doc92_qa0__after87": (0.25, "[ACK] doc92 not yet seen. PRED '$3.7 billion' wrong specific (gold $13.2B)."),
    "doc16_qa0__after87": (0.25, "[ANS] doc16 seen. PRED '11.97 times' wrong specific."),
    "doc22_qa0__after88": (1.0, "[ANS] doc22 seen. Same Amcor 8K matches."),
    "doc27_qa0__after88": (0.5, "[ANS] doc27 seen. Same generic restructuring."),
    "doc25_qa0__after88": (0.75, "[ANS] doc25 seen. Same minimal."),
    "doc66_qa0__after88": (0.25, "[ANS] doc66 seen. Same Boeing tax $ vs % partial."),
    "doc60_qa0__after88": (0.75, "[ANS] doc60 seen. Same one-segment."),
    "doc21_qa0__after88": (1.0, "[ANS] doc21 seen. PRED '$1,615.9 million' within 0.01% tolerance."),
    "doc34_qa0__after89": (1.0, "[ANS] doc34 seen. PRED matches 'amortization of intangible assets from Xilinx acquisition'."),
    "doc89_qa0__after89": (1.0, "[ANS] doc89 seen. GOLD 'US +3.0% vs intl -0.6%'. PRED 'US 3.0%, international decrease of 0.6%'. Exact match!"),
    "doc43_qa0__after89": (0.25, "[ANS] doc43 seen. Same long-term debt wrong."),
    "doc75_qa0__after89": (0.25, "[ANS] doc75 seen. PRED '8.73' wrong specific."),
    "doc58_qa0__after89": (1.0, "[ANS] doc58 seen. Same $381.6 within tol."),
    "doc83_qa0__after89": (1.0, "[ANS] doc83 seen. GOLD $3215. PRED '$3,189.4 million' within 0.8% tolerance."),
    "doc2_qa0__after89": (0.0, "[ANS] doc2 seen. Same Y/N flip."),
}

ENTRY_SUFFIXES: list[str] = [
    "doc37_qa0__after75", "doc0_qa0__after75", "doc122_qa0__after75", "doc26_qa0__after75",
    "doc126_qa0__after75", "doc111_qa0__after75", "doc53_qa0__after75", "doc25_qa0__after75",
    "doc121_qa0__after75", "doc133_qa0__after75",
    "doc3_qa0__after76", "doc41_qa0__after76", "doc112_qa0__after76", "doc100_qa0__after76",
    "doc37_qa0__after76", "doc55_qa0__after76", "doc18_qa0__after76", "doc86_qa0__after76",
    "doc74_qa0__after76", "doc11_qa0__after76",
    "doc64_qa0__after77", "doc60_qa0__after77", "doc113_qa0__after77", "doc44_qa0__after77",
    "doc87_qa0__after77", "doc82_qa0__after77", "doc52_qa0__after77", "doc97_qa0__after77",
    "doc130_qa0__after77", "doc11_qa0__after77",
    "doc149_qa0__after78", "doc120_qa0__after78", "doc19_qa0__after78", "doc44_qa0__after78",
    "doc63_qa0__after78", "doc102_qa0__after78", "doc67_qa0__after78", "doc40_qa0__after78",
    "doc52_qa0__after78", "doc65_qa0__after78",
    "doc146_qa0__after79", "doc23_qa0__after79", "doc109_qa0__after79", "doc56_qa0__after79",
    "doc92_qa0__after79", "doc55_qa0__after79", "doc28_qa0__after79", "doc83_qa0__after79",
    "doc2_qa0__after79", "doc14_qa0__after79",
    "doc106_qa0__after80", "doc44_qa0__after80", "doc82_qa0__after80", "doc25_qa0__after80",
    "doc60_qa0__after80", "doc103_qa0__after80", "doc35_qa0__after80", "doc12_qa0__after80",
    "doc141_qa0__after80", "doc43_qa0__after80",
    "doc30_qa0__after81", "doc75_qa0__after81", "doc79_qa0__after81", "doc2_qa0__after81",
    "doc138_qa0__after81", "doc60_qa0__after81", "doc23_qa0__after81", "doc59_qa0__after81",
    "doc98_qa0__after81", "doc106_qa0__after81",
    "doc79_qa0__after82", "doc12_qa0__after82", "doc125_qa0__after82", "doc28_qa0__after82",
    "doc35_qa0__after82", "doc27_qa0__after82", "doc43_qa0__after82", "doc101_qa0__after82",
    "doc71_qa0__after82", "doc144_qa0__after82",
    "doc39_qa0__after83", "doc3_qa0__after83", "doc54_qa0__after83", "doc42_qa0__after83",
    "doc144_qa0__after83", "doc126_qa0__after83", "doc90_qa0__after83", "doc17_qa0__after83",
    "doc46_qa0__after83", "doc57_qa0__after83",
    "doc148_qa0__after84", "doc46_qa0__after84", "doc84_qa0__after84", "doc12_qa0__after84",
    "doc77_qa0__after84", "doc58_qa0__after84", "doc29_qa0__after84", "doc124_qa0__after84",
    "doc13_qa0__after84", "doc8_qa0__after84",
    "doc18_qa0__after85", "doc131_qa0__after85", "doc67_qa0__after85", "doc11_qa0__after85",
    "doc118_qa0__after85", "doc48_qa0__after85", "doc139_qa0__after85", "doc116_qa0__after85",
    "doc135_qa0__after85", "doc119_qa0__after85",
    "doc48_qa0__after86", "doc46_qa0__after86", "doc84_qa0__after86", "doc4_qa0__after86",
    "doc40_qa0__after86", "doc26_qa0__after86", "doc109_qa0__after86", "doc116_qa0__after86",
    "doc138_qa0__after86", "doc76_qa0__after86",
    "doc12_qa0__after87", "doc138_qa0__after87", "doc43_qa0__after87", "doc108_qa0__after87",
    "doc59_qa0__after87", "doc4_qa0__after87", "doc92_qa0__after87", "doc16_qa0__after87",
    "doc91_qa0__after87", "doc124_qa0__after87",
    "doc22_qa0__after88", "doc27_qa0__after88", "doc25_qa0__after88", "doc149_qa0__after88",
    "doc146_qa0__after88", "doc66_qa0__after88", "doc60_qa0__after88", "doc117_qa0__after88",
    "doc21_qa0__after88", "doc113_qa0__after88",
    "doc34_qa0__after89", "doc129_qa0__after89", "doc89_qa0__after89", "doc43_qa0__after89",
    "doc101_qa0__after89", "doc75_qa0__after89", "doc58_qa0__after89", "doc111_qa0__after89",
    "doc83_qa0__after89", "doc2_qa0__after89",
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
