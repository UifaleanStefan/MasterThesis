"""Phase 1.9 extension: FB rag-corpus calibration part 8 (entries 1050-1199)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__rag-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__rag-corpus__calibration__seed42/results.jsonl")

SPECIAL: dict[str, tuple[float, str]] = {
    "doc22_qa0__after105": (1.0, "[ANS] doc22 seen. Same Amcor 8K supplemental indentures match."),
    "doc25_qa0__after105": (0.75, "[ANS] doc25 seen. Same minimal 'Amcor packaging industry'."),
    "doc62_qa0__after105": (0.25, "[ANS] doc62 seen. 'Gross margin not useful for Boeing' wrong reasoning + missed Y."),
    "doc98_qa0__after105": (1.0, "[ANS] doc98 seen. Correct JPM VaR decreased $7M."),
    "doc1_qa0__after105": (1.0, "[ANS] doc1 seen. PRED '$8.738 billion' within tolerance."),
    "doc76_qa0__after105": (0.75, "[ANS] doc76 seen. 'Yes CVS capital-intensive' Y correct minimal."),
    "doc70_qa0__after106": (0.25, "[ANS] doc70 seen. GOLD 63.86. PRED '56.73 days' wrong specific."),
    "doc28_qa0__after106": (1.0, "[ANS] doc28 seen. Exact $2,018 million match."),
    "doc30_qa0__after106": (1.0, "[ANS] doc30 seen. Same 4.18% within tol."),
    "doc85_qa0__after106": (1.0, "[ANS] doc85 seen. 'No 1.3% vs 13.6% 2021' rich match."),
    "doc87_qa0__after106": (0.5, "[ANS] doc87 seen. PRED computation cut off mid-sentence — partial."),
    "doc49_qa0__after106": (1.0, "[ANS] doc49 seen. GOLD $5409. PRED '5,409' EXACT."),
    "doc9_qa0__after107": (0.25, "[ANS] doc9 seen. PRED '6.0%' wrong specific (gold 1.9%)."),
    "doc13_qa0__after107": (0.0, "[ANS] doc13 seen. PRED 'Yes improving margin' — Y/N flip vs gold 'No declining'."),
    "doc122_qa0__after107": (0.25, "[ACK] doc122 not yet seen. PRED '0' wrong."),
    "doc103_qa0__after107": (1.0, "[ANS] doc103 seen. PRED '$302.578 million' within 0.14% of $303."),
    "doc87_qa0__after107": (0.75, "[ANS] doc87 seen. Hedged response 'not provided + conventional inventory may not be meaningful'."),
    "doc75_qa0__after108": (0.25, "[ANS] doc75 seen. PRED '8.73' wrong specific."),
    "doc90_qa0__after108": (1.0, "[ANS] doc90 seen. PRED matches JnJ Consumer Health Aug 30 2023."),
    "doc98_qa0__after108": (1.0, "[ANS] doc98 seen. Correct VaR."),
    "doc42_qa0__after108": (1.0, "[ANS] doc42 seen. '24.6% to 21.6%' exact."),
    "doc43_qa0__after108": (0.25, "[ANS] doc43 seen. 'Long-term debt' wrong (gold Customer deposits)."),
    "doc51_qa0__after108": (1.0, "[ANS] doc51 seen. Correct Best Buy acquisitions with amounts."),
    "doc68_qa0__after108": (1.0, "[ANS] doc68 seen. GOLD 39.7%. PRED '39.7%' EXACT match!"),
    "doc45_qa0__after108": (1.0, "[ANS] doc45 seen. PRED '$0.389 billion' within 2.75% (under 5% tolerance)."),
    "doc108_qa0__after108": (0.75, "[ANS] doc108 seen. 'MGM China worst' correct ID, no -44% specific."),
    "doc26_qa0__after109": (0.75, "[ANS] doc26 seen. Same partial gross margin calc."),
    "doc7_qa0__after109": (1.0, "[ANS] doc7 seen. 65 years correct."),
    "doc14_qa0__after109": (0.0, "[ANS] doc14 seen. Same Adobe FCF refusal."),
    "doc44_qa0__after109": (1.0, "[ANS] doc44 seen. Correct Card Member retention."),
    "doc102_qa0__after109": (0.25, "[ANS] doc102 seen. PRED '1.3%' wrong specific (gold 0.4%, 3.25x off)."),
    "doc65_qa0__after109": (1.0, "[ANS] doc65 seen. Full 737/787/777X."),
    "doc18_qa0__after109": (0.25, "[ANS] doc18 seen. PRED '30.73' wrong."),
    "doc7_qa0__after110": (1.0, "[ANS] doc7 seen. Correct."),
    "doc72_qa0__after110": (1.0, "[ANS] doc72 seen. PRED '20% to 23%' EXACT match to gold."),
    "doc35_qa0__after110": (1.0, "[ANS] doc35 seen. Correct $3,565M."),
    "doc99_qa0__after110": (1.0, "[ANS] doc99 seen. GOLD 6.25. PRED '6.20' within 1% tolerance."),
    "doc33_qa0__after110": (1.0, "[ANS] doc33 seen. Rich AMD drivers match."),
    "doc90_qa0__after110": (1.0, "[ANS] doc90 seen. PRED matches."),
    "doc97_qa0__after110": (0.25, "[ANS] doc97 seen. 'Consumer & Community Banking' wrong segment."),
    "doc122_qa0__after110": (0.25, "[ACK] doc122 not yet seen. PRED '0' wrong."),
    "doc26_qa0__after111": (0.75, "[ANS] doc26 seen. Same partial."),
    "doc30_qa0__after111": (1.0, "[ANS] doc30 seen. 4.18% within tol."),
    "doc82_qa0__after111": (1.0, "[ANS] doc82 seen. GOLD 0.68. PRED '0.69' within 1.5% tolerance."),
    "doc36_qa0__after111": (1.0, "[ANS] doc36 seen. 'Data Center segment' correct."),
    "doc72_qa0__after111": (1.0, "[ANS] doc72 seen. '20% to 23%' exact."),
    "doc37_qa0__after111": (1.0, "[ANS] doc37 seen. Correct."),
    "doc101_qa0__after111": (1.0, "[ANS] doc101 seen. '$5,818 million' exact."),
    "doc35_qa0__after112": (1.0, "[ANS] doc35 seen. Correct."),
    "doc52_qa0__after112": (1.0, "[ANS] doc52 seen. Correct $1,824M."),
    "doc23_qa0__after112": (0.5, "[ANS] doc23 seen. PRED hedged 'cannot calculate directly + may focus on other liquidity measures'. Partial."),
    "doc21_qa0__after112": (1.0, "[ANS] doc21 seen. '$1,615.9 million' within tol."),
    "doc59_qa0__after112": (1.0, "[ANS] doc59 seen. '$12,645' exact."),
    "doc92_qa0__after112": (1.0, "[ANS] doc92 seen. '$13.2 billion' exact."),
    "doc89_qa0__after112": (1.0, "[ANS] doc89 seen. 'US 3.0%, international decline of 0.6%' exact."),
    "doc122_qa0__after112": (0.25, "[ACK] doc122 not yet seen. PRED '0' wrong."),
    "doc104_qa0__after113": (0.25, "[ANS] doc104 seen. GOLD 7.9%. PRED '-3.5%' wrong direction + magnitude."),
    "doc82_qa0__after113": (1.0, "[ANS] doc82 seen. '0.69' within tol."),
    "doc60_qa0__after113": (0.75, "[ANS] doc60 seen. One segment partial."),
    "doc89_qa0__after113": (1.0, "[ANS] doc89 seen. Same correct US/intl match."),
    "doc47_qa0__after113": (0.75, "[ANS] doc47 seen. Same self-contradictory with correct -$1,561M."),
    "doc56_qa0__after113": (1.0, "[ANS] doc56 seen. '1.74' within 0.6% tolerance."),
    "doc98_qa0__after113": (1.0, "[ANS] doc98 seen. VaR correct."),
    "doc24_qa0__after114": (0.0, "[ANS] doc24 seen. GOLD definitive Amcor acquisitions list. PRED refuses 'do not contain info'. Refusal on definitive ANS."),
    "doc113_qa0__after114": (1.0, "[ANS] doc113 seen. GOLD $5466. PRED '5,466.3 million' EXACT."),
    "doc27_qa0__after114": (0.5, "[ANS] doc27 seen. Same generic restructuring partial."),
    "doc97_qa0__after114": (0.25, "[ANS] doc97 seen. 'Consumer & Community Banking' wrong segment."),
    "doc99_qa0__after114": (1.0, "[ANS] doc99 seen. '6.20' within tol."),
    "doc19_qa0__after114": (1.0, "[ANS] doc19 seen. GOLD 30.8%. PRED '30.7%' within 0.3% tolerance."),
    "doc98_qa0__after114": (1.0, "[ANS] doc98 seen. VaR correct."),
    "doc12_qa0__after114": (0.25, "[ANS] doc12 seen. '1.23' wrong specific."),
    "doc80_qa0__after115": (1.0, "[ANS] doc80 seen. Correct + vote count."),
    "doc23_qa0__after115": (0.5, "[ANS] doc23 seen. PRED hedged 'cannot calculate + current assets $5,853 to $5,308'. Partial with computation but no quick ratio."),
    "doc111_qa0__after115": (0.25, "[ANS] doc111 seen. 'Yes increased debt' Y/N flip + self-contradictory ($47,032M to $41,990M decrease)."),
    "doc33_qa0__after115": (1.0, "[ANS] doc33 seen. Rich correct."),
    "doc87_qa0__after115": (0.75, "[ANS] doc87 seen. Hedged."),
    "doc81_qa0__after115": (0.25, "[ANS] doc81 seen. GOLD -3.7. PRED '66.73 days' wrong direction + magnitude."),
    "doc68_qa0__after115": (1.0, "[ANS] doc68 seen. GOLD 39.7%. PRED '39.5%' within 0.5% tolerance."),
    "doc4_qa0__after116": (0.75, "[ANS] doc4 seen. 'Consumer segment' partial."),
    "doc66_qa0__after116": (0.25, "[ANS] doc66 seen. Same Boeing tax $ amounts partial."),
    "doc88_qa0__after116": (0.0, "[ANS] doc88 seen. '+12.5% accelerate' Y/N flip."),
    "doc93_qa0__after116": (1.0, "[ANS] doc93 seen. 'Yes JnJ 20.0% to 20.1%' exact match!"),
    "doc105_qa0__after116": (1.0, "[ANS] doc105 seen. 'Yes MGM $0.01 throughout 2022' correct."),
    "doc44_qa0__after116": (1.0, "[ANS] doc44 seen. Correct."),
    "doc104_qa0__after116": (0.25, "[ANS] doc104 seen. Same '-3.5%' wrong."),
    "doc21_qa0__after116": (1.0, "[ANS] doc21 seen. '$1,615.9 million' within tol."),
    "doc6_qa0__after117": (0.75, "[ANS] doc6 seen. PRED lists 3 notes without MMM tickers."),
    "doc44_qa0__after117": (1.0, "[ANS] doc44 seen. Correct."),
    "doc42_qa0__after117": (1.0, "[ANS] doc42 seen. '24.6% to 21.6%' exact."),
    "doc54_qa0__after117": (1.0, "[ANS] doc54 seen. 'Yes 982 to 969' EXACT numbers."),
    "doc2_qa0__after117": (0.0, "[ANS] doc2 seen. Same Y/N flip + $26,459M (slightly different)."),
    "doc3_qa0__after117": (0.75, "[ANS] doc3 seen. Same partial 3M OM."),
    "doc107_qa0__after118": (0.25, "[ANS] doc107 seen. GOLD 'Coverage zero'. PRED '1.61' wrong specific."),
    "doc93_qa0__after118": (1.0, "[ANS] doc93 seen. Same JnJ 20.0% to 20.1% match."),
    "doc4_qa0__after118": (0.75, "[ANS] doc4 seen. 'Consumer segment' partial."),
    "doc22_qa0__after118": (1.0, "[ANS] doc22 seen. Same Amcor 8K match."),
    "doc37_qa0__after118": (1.0, "[ANS] doc37 seen. Correct."),
    "doc73_qa0__after118": (0.75, "[ANS] doc73 seen. GOLD '$831M WC'. PRED 'positive WC, $7,453M - $5,175M = $2,278M' — Y direction correct, different specific (uses inventories)."),
    "doc45_qa0__after118": (1.0, "[ANS] doc45 seen. '$0.389 billion' within tol."),
    "doc41_qa0__after118": (1.0, "[ANS] doc41 seen. Correct."),
    "doc34_qa0__after118": (1.0, "[ANS] doc34 seen. PRED matches verbatim."),
    "doc15_qa0__after119": (1.0, "[ANS] doc15 seen. '0' match."),
    "doc45_qa0__after119": (1.0, "[ANS] doc45 seen. '$0.389 billion' within tol."),
    "doc49_qa0__after119": (1.0, "[ANS] doc49 seen. '5,409' exact."),
    "doc68_qa0__after119": (1.0, "[ANS] doc68 seen. '39.7%' exact."),
    "doc48_qa0__after119": (0.25, "[ANS] doc48 seen. '3.9%' wrong."),
    "doc25_qa0__after119": (0.75, "[ANS] doc25 seen. Minimal."),
    "doc59_qa0__after119": (1.0, "[ANS] doc59 seen. Exact."),
    "doc52_qa0__after119": (1.0, "[ANS] doc52 seen. Correct."),
}

ENTRY_SUFFIXES: list[str] = [
    "doc22_qa0__after105", "doc119_qa0__after105", "doc25_qa0__after105", "doc146_qa0__after105",
    "doc62_qa0__after105", "doc98_qa0__after105", "doc1_qa0__after105", "doc138_qa0__after105",
    "doc123_qa0__after105", "doc76_qa0__after105",
    "doc124_qa0__after106", "doc70_qa0__after106", "doc28_qa0__after106", "doc30_qa0__after106",
    "doc85_qa0__after106", "doc130_qa0__after106", "doc87_qa0__after106", "doc135_qa0__after106",
    "doc148_qa0__after106", "doc49_qa0__after106",
    "doc9_qa0__after107", "doc134_qa0__after107", "doc13_qa0__after107", "doc142_qa0__after107",
    "doc127_qa0__after107", "doc122_qa0__after107", "doc133_qa0__after107", "doc103_qa0__after107",
    "doc139_qa0__after107", "doc87_qa0__after107",
    "doc75_qa0__after108", "doc90_qa0__after108", "doc98_qa0__after108", "doc140_qa0__after108",
    "doc42_qa0__after108", "doc43_qa0__after108", "doc51_qa0__after108", "doc68_qa0__after108",
    "doc45_qa0__after108", "doc108_qa0__after108",
    "doc26_qa0__after109", "doc7_qa0__after109", "doc119_qa0__after109", "doc14_qa0__after109",
    "doc44_qa0__after109", "doc102_qa0__after109", "doc65_qa0__after109", "doc133_qa0__after109",
    "doc18_qa0__after109", "doc134_qa0__after109",
    "doc120_qa0__after110", "doc7_qa0__after110", "doc72_qa0__after110", "doc35_qa0__after110",
    "doc99_qa0__after110", "doc33_qa0__after110", "doc145_qa0__after110", "doc90_qa0__after110",
    "doc97_qa0__after110", "doc122_qa0__after110",
    "doc117_qa0__after111", "doc120_qa0__after111", "doc26_qa0__after111", "doc113_qa0__after111",
    "doc30_qa0__after111", "doc82_qa0__after111", "doc36_qa0__after111", "doc72_qa0__after111",
    "doc37_qa0__after111", "doc101_qa0__after111",
    "doc35_qa0__after112", "doc52_qa0__after112", "doc23_qa0__after112", "doc120_qa0__after112",
    "doc21_qa0__after112", "doc59_qa0__after112", "doc114_qa0__after112", "doc92_qa0__after112",
    "doc89_qa0__after112", "doc122_qa0__after112",
    "doc139_qa0__after113", "doc104_qa0__after113", "doc136_qa0__after113", "doc82_qa0__after113",
    "doc60_qa0__after113", "doc89_qa0__after113", "doc47_qa0__after113", "doc137_qa0__after113",
    "doc56_qa0__after113", "doc98_qa0__after113",
    "doc24_qa0__after114", "doc113_qa0__after114", "doc27_qa0__after114", "doc124_qa0__after114",
    "doc97_qa0__after114", "doc99_qa0__after114", "doc131_qa0__after114", "doc19_qa0__after114",
    "doc98_qa0__after114", "doc12_qa0__after114",
    "doc80_qa0__after115", "doc23_qa0__after115", "doc111_qa0__after115", "doc131_qa0__after115",
    "doc33_qa0__after115", "doc87_qa0__after115", "doc140_qa0__after115", "doc81_qa0__after115",
    "doc121_qa0__after115", "doc68_qa0__after115",
    "doc4_qa0__after116", "doc66_qa0__after116", "doc120_qa0__after116", "doc138_qa0__after116",
    "doc88_qa0__after116", "doc93_qa0__after116", "doc105_qa0__after116", "doc44_qa0__after116",
    "doc104_qa0__after116", "doc21_qa0__after116",
    "doc146_qa0__after117", "doc131_qa0__after117", "doc6_qa0__after117", "doc44_qa0__after117",
    "doc42_qa0__after117", "doc54_qa0__after117", "doc2_qa0__after117", "doc148_qa0__after117",
    "doc121_qa0__after117", "doc3_qa0__after117",
    "doc107_qa0__after118", "doc93_qa0__after118", "doc4_qa0__after118", "doc133_qa0__after118",
    "doc22_qa0__after118", "doc37_qa0__after118", "doc73_qa0__after118", "doc45_qa0__after118",
    "doc41_qa0__after118", "doc34_qa0__after118",
    "doc15_qa0__after119", "doc142_qa0__after119", "doc45_qa0__after119", "doc49_qa0__after119",
    "doc68_qa0__after119", "doc48_qa0__after119", "doc25_qa0__after119", "doc146_qa0__after119",
    "doc59_qa0__after119", "doc52_qa0__after119",
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
