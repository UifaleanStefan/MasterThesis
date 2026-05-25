"""Phase 1.9 extension: FB rag-corpus calibration part 7 (entries 900-1049)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__rag-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__rag-corpus__calibration__seed42/results.jsonl")

SPECIAL: dict[str, tuple[float, str]] = {
    "doc66_qa0__after90": (0.25, "[ANS] doc66 seen. GOLD '0.62% vs -14.76%'. PRED $ amounts ($-31M vs $743M benefit) — wrong format + numbers."),
    "doc30_qa0__after90": (1.0, "[ANS] doc30 seen. PRED '4.18%' within tol."),
    "doc41_qa0__after90": (1.0, "[ANS] doc41 seen. Correct AMEX gross margin reasoning."),
    "doc45_qa0__after90": (0.25, "[ANS] doc45 seen. GOLD $0.40. PRED '$0.353B' — 12% off, above tolerance."),
    "doc5_qa0__after90": (0.0, "[ANS] doc5 seen. Same 3M quick ratio refusal."),
    "doc91_qa0__after90": (0.5, "[ACK] doc91 not yet seen. PRED 'Approximately $20 billion' — exact match without doc basis. World knowledge."),
    "doc125_qa0__after90": (0.5, "[ACK] doc125 not yet seen. PRED 'proposal not approved' equivalent."),
    "doc96_qa0__after91": (1.0, "[ACK] doc96 not yet seen. PRED 'gross margins not relevant for JPM (NIM/ROE/NI)'. Universally correct."),
    "doc88_qa0__after91": (0.0, "[ANS] doc88 seen. GOLD 'No, decelerate 3.6% to 3.5%'. PRED '+12.5% accelerate' Y/N flip + fabricated."),
    "doc79_qa0__after91": (1.0, "[ANS] doc79 seen. PRED 'Yes Mary N. Dillon former Exec Chair/CEO Ulta, similar retail'. Correct."),
    "doc33_qa0__after91": (1.0, "[ANS] doc33 seen. Same rich AMD drivers match."),
    "doc20_qa0__after91": (1.0, "[ANS] doc20 seen. PRED '11,588' exact."),
    "doc40_qa0__after91": (1.0, "[ANS] doc40 seen. Correct AMEX OM reasoning."),
    "doc86_qa0__after91": (1.0, "[ANS] doc86 seen. Full JnJ gross margin drivers match (COVID/currency/commodity/supply chain)."),
    "doc15_qa0__after91": (1.0, "[ANS] doc15 seen. '0' match."),
    "doc18_qa0__after91": (0.25, "[ANS] doc18 seen. '34.12' wrong specific."),
    "doc45_qa0__after92": (0.25, "[ANS] doc45 seen. Same $0.353B 12% off."),
    "doc78_qa0__after92": (0.75, "[ANS] doc78 seen. 'Yes CVS paid dividends Q2 FY2022' — Y correct but no $0.55 specific."),
    "doc91_qa0__after92": (1.0, "[ANS] doc91 seen. 'Approximately $20 billion' exact."),
    "doc10_qa0__after92": (0.25, "[ANS] doc10 seen. PRED '1.69' wrong specific."),
    "doc12_qa0__after92": (0.25, "[ANS] doc12 seen. PRED '1.23' wrong specific."),
    "doc86_qa0__after92": (0.0, "[ANS] doc86 seen. Refusal on definitive ANS (previous correct match was world-knowledge based)."),
    "doc122_qa0__after92": (0.25, "[ACK] doc122 not yet seen. PRED '0' wrong specific."),
    "doc26_qa0__after93": (0.75, "[ANS] doc26 seen. Same partial gross margin calc."),
    "doc64_qa0__after93": (0.75, "[ANS] doc64 seen. Y partial."),
    "doc54_qa0__after93": (1.0, "[ANS] doc54 seen. GOLD '982 to 969 -1.32%'. PRED '982 at Q2 FY23 to 969 at Q2 FY24' EXACT numbers."),
    "doc82_qa0__after93": (0.25, "[ANS] doc82 seen. GOLD 0.68. PRED '0.72' — 6% off, above 5% tolerance."),
    "doc18_qa0__after94": (0.25, "[ANS] doc18 seen. '25.36' wrong."),
    "doc52_qa0__after94": (1.0, "[ANS] doc52 seen. Correct $1,824M operations."),
    "doc9_qa0__after94": (0.25, "[ANS] doc9 seen. GOLD 1.9%. PRED '6.0%' wrong specific."),
    "doc64_qa0__after94": (0.75, "[ANS] doc64 seen. Y partial."),
    "doc83_qa0__after94": (1.0, "[ANS] doc83 seen. GOLD $3215. PRED '$3,189.9 million' within 0.8% tolerance."),
    "doc18_qa0__after95": (0.25, "[ANS] doc18 seen. '29.12' wrong."),
    "doc80_qa0__after95": (1.0, "[ANS] doc80 seen. PRED 'Yes Richard A. Johnson 16,105,005 against'. Correct + votes."),
    "doc52_qa0__after95": (1.0, "[ANS] doc52 seen. Correct $1,824M."),
    "doc51_qa0__after95": (1.0, "[ANS] doc51 seen. Correct Best Buy acquisitions with amounts."),
    "doc122_qa0__after95": (0.25, "[ACK] doc122 not yet seen. PRED '0' wrong."),
    "doc8_qa0__after95": (0.25, "[ANS] doc8 seen. '25.66' 5.7% off, above tolerance."),
    "doc17_qa0__after95": (0.25, "[ANS] doc17 seen. '-1.41' wrong."),
    "doc86_qa0__after96": (0.0, "[ANS] doc86 seen. PRED 'Gross margin not useful for JnJ (pharma/medical)' — wrong reasoning (gold has specific drivers)."),
    "doc80_qa0__after96": (1.0, "[ANS] doc80 seen. Same correct match."),
    "doc94_qa0__after96": (0.25, "[ANS] doc94 seen. GOLD 'Corporate'. PRED 'Consumer & Community Banking' wrong segment."),
    "doc15_qa0__after96": (1.0, "[ANS] doc15 seen. '0' match."),
    "doc95_qa0__after96": (0.25, "[ANS] doc95 seen. GOLD '$66.56/share'. PRED '$292.3 billion' — wrong (gives stockholders equity not per-share)."),
    "doc53_qa0__after96": (1.0, "[ANS] doc53 seen. PRED '$1,874M to $1,093M' (42% drop)."),
    "doc52_qa0__after96": (1.0, "[ANS] doc52 seen. Correct."),
    "doc50_qa0__after96": (0.0, "[ANS] doc50 seen. Same Y/N flip."),
    "doc39_qa0__after96": (1.0, "[ANS] doc39 seen. 'US, EMEA, APAC, LACC, Other Unallocated' — core 4 correct + extra. Match."),
    "doc63_qa0__after97": (0.5, "[ANS] doc63 seen. 'Boeing significant portion limited commercial airlines' partial — misses US gov 40%."),
    "doc8_qa0__after97": (0.25, "[ANS] doc8 seen. Same '25.66' above tol."),
    "doc47_qa0__after97": (0.75, "[ANS] doc47 seen. Same self-contradictory but final conclusion correct + -$1,561M."),
    "doc125_qa0__after97": (0.5, "[ACK] doc125 not yet seen. 'not approved' equivalent."),
    "doc95_qa0__after97": (0.25, "[ANS] doc95 seen. PRED '$292.3 billion stockholders equity' — wrong (no per-share calc)."),
    "doc37_qa0__after97": (1.0, "[ANS] doc37 seen. Correct."),
    "doc6_qa0__after97": (0.75, "[ANS] doc6 seen. PRED lists 3 notes without MMM tickers — partial."),
    "doc50_qa0__after97": (0.0, "[ANS] doc50 seen. Same Y/N flip."),
    "doc42_qa0__after98": (1.0, "[ANS] doc42 seen. '24.6% to 21.6%' exact."),
    "doc80_qa0__after98": (1.0, "[ANS] doc80 seen. Same correct match."),
    "doc91_qa0__after98": (1.0, "[ANS] doc91 seen. '$20 billion' exact."),
    "doc60_qa0__after98": (0.75, "[ANS] doc60 seen. One segment partial."),
    "doc108_qa0__after98": (0.25, "[ACK] doc108 not yet seen. GOLD 'MGM China worst -44%'. PRED 'International region 11.5%' wrong specific."),
    "doc97_qa0__after98": (0.25, "[ANS] doc97 seen. GOLD 'Corporate & Investment Bank $3725M'. PRED 'Consumer & Community Banking' wrong segment."),
    "doc16_qa0__after98": (0.25, "[ANS] doc16 seen. '11.97' wrong."),
    "doc11_qa0__after99": (0.25, "[ANS] doc11 seen. -99.6% wrong calc."),
    "doc40_qa0__after99": (1.0, "[ANS] doc40 seen. Correct AMEX OM reasoning."),
    "doc108_qa0__after99": (0.25, "[ACK] doc108 not yet seen. Same wrong specific."),
    "doc43_qa0__after99": (0.25, "[ANS] doc43 seen. 'Long-term debt $42,573M' wrong (gold Customer deposits)."),
    "doc71_qa0__after99": (1.0, "[ANS] doc71 seen. GOLD 10.3%. PRED '10.5%' within 2% (under 5% tolerance)."),
    "doc5_qa0__after100": (0.0, "[ANS] doc5 seen. Same 3M quick ratio refusal."),
    "doc10_qa0__after100": (0.25, "[ANS] doc10 seen. PRED '1.69' wrong."),
    "doc90_qa0__after100": (1.0, "[ANS] doc90 seen. PRED matches JnJ Consumer Health Aug 30 2023."),
    "doc15_qa0__after100": (1.0, "[ANS] doc15 seen. '0' match."),
    "doc67_qa0__after100": (0.25, "[ANS] doc67 seen. '1.46%' wrong."),
    "doc65_qa0__after100": (1.0, "[ANS] doc65 seen. Full 737/787/777X."),
    "doc63_qa0__after100": (0.5, "[ANS] doc63 seen. Same partial 'limited commercial airlines' missing US gov."),
    "doc81_qa0__after101": (0.25, "[ANS] doc81 seen. GOLD -3.7. PRED '66.73 days' wrong specific."),
    "doc35_qa0__after101": (1.0, "[ANS] doc35 seen. Correct $3,565M."),
    "doc41_qa0__after101": (1.0, "[ANS] doc41 seen. Correct AMEX gross margin reasoning."),
    "doc100_qa0__after101": (1.0, "[ANS] doc100 seen. GOLD 1.33. PRED '1.30' within 2.3% tolerance."),
    "doc98_qa0__after101": (1.0, "[ANS] doc98 seen. 'Yes VaR decreased $7M three months June 30 2023'. Correct."),
    "doc78_qa0__after101": (0.75, "[ANS] doc78 seen. Y CVS paid dividends, no $0.55 specific."),
    "doc75_qa0__after101": (0.25, "[ANS] doc75 seen. PRED '8.73' wrong."),
    "doc96_qa0__after101": (1.0, "[ANS] doc96 seen. Correct JPM gross margin reasoning."),
    "doc125_qa0__after101": (0.5, "[ACK] doc125 not yet seen. 'not approved' equivalent."),
    "doc31_qa0__after102": (0.0, "[ANS] doc31 seen. Same AMD refusal."),
    "doc39_qa0__after102": (1.0, "[ANS] doc39 seen. 'US, EMEA, APAC, LACC, Other Unallocated' match."),
    "doc24_qa0__after102": (0.75, "[ANS] doc24 seen. Same partial Amcor acquisitions."),
    "doc68_qa0__after102": (1.0, "[ANS] doc68 seen. GOLD 39.7%. PRED computes COGS/Revenue * 100 = 39.7%. Correct calc!"),
    "doc44_qa0__after102": (1.0, "[ANS] doc44 seen. Correct."),
    "doc36_qa0__after102": (1.0, "[ANS] doc36 seen. 'Data Center segment' correct."),
    "doc59_qa0__after102": (1.0, "[ANS] doc59 seen. '$12,645' exact."),
    "doc46_qa0__after102": (1.0, "[ANS] doc46 seen. GOLD $1832. PRED '1,832' EXACT match!"),
    "doc108_qa0__after102": (0.25, "[ACK] doc108 not yet seen. Same wrong specific."),
    "doc108_qa0__after103": (0.0, "[ACK] doc108 not yet seen. PRED refuses 'do not provide regional topline performance'. Honest refusal. → 1.0 actually"),
    "doc61_qa0__after103": (1.0, "[ANS] doc61 seen. Full Lion Air/Ethiopian Airlines lawsuits detail."),
    "doc60_qa0__after103": (0.75, "[ANS] doc60 seen. One segment partial."),
    "doc36_qa0__after103": (1.0, "[ANS] doc36 seen. Correct."),
    "doc51_qa0__after103": (1.0, "[ANS] doc51 seen. Correct Best Buy acquisitions."),
    "doc85_qa0__after103": (1.0, "[ANS] doc85 seen. 'No, sales 1.3% vs 13.6% 2021' rich match with comparison."),
    "doc105_qa0__after103": (0.0, "[ANS] doc105 seen. GOLD definitive 'Yes MGM $0.01 throughout FY2022'. PRED refuses. Refusal on definitive."),
    "doc71_qa0__after103": (1.0, "[ANS] doc71 seen. '10.5%' within tol."),
    "doc46_qa0__after104": (1.0, "[ANS] doc46 seen. '1,829' within 0.16% tol."),
    "doc96_qa0__after104": (1.0, "[ANS] doc96 seen. Correct JPM reasoning."),
    "doc16_qa0__after104": (0.25, "[ANS] doc16 seen. PRED '~12.0' wrong specific."),
    "doc80_qa0__after104": (1.0, "[ANS] doc80 seen. Correct + votes."),
    "doc31_qa0__after104": (0.0, "[ANS] doc31 seen. Same refusal."),
    "doc14_qa0__after104": (0.0, "[ANS] doc14 seen. Same Adobe FCF refusal."),
    "doc101_qa0__after104": (1.0, "[ANS] doc101 seen. GOLD $5818. PRED '$5,818 million' EXACT!"),
    "doc103_qa0__after104": (1.0, "[ANS] doc103 seen. GOLD $303. PRED '$302.6 million' within 0.1% tolerance."),
}

# Fix doc108_qa0__after103 (was incorrectly given 0.0, should be 1.0 for honest refusal at [ACK])
SPECIAL["doc108_qa0__after103"] = (1.0, "[ACK] doc108 not yet seen. PRED 'do not provide regional topline performance'. Honest refusal.")

ENTRY_SUFFIXES: list[str] = [
    "doc66_qa0__after90", "doc113_qa0__after90", "doc30_qa0__after90", "doc116_qa0__after90",
    "doc41_qa0__after90", "doc45_qa0__after90", "doc5_qa0__after90", "doc91_qa0__after90",
    "doc125_qa0__after90", "doc126_qa0__after90",
    "doc96_qa0__after91", "doc88_qa0__after91", "doc79_qa0__after91", "doc33_qa0__after91",
    "doc20_qa0__after91", "doc40_qa0__after91", "doc86_qa0__after91", "doc15_qa0__after91",
    "doc99_qa0__after91", "doc18_qa0__after91",
    "doc101_qa0__after92", "doc45_qa0__after92", "doc114_qa0__after92", "doc78_qa0__after92",
    "doc91_qa0__after92", "doc10_qa0__after92", "doc12_qa0__after92", "doc94_qa0__after92",
    "doc86_qa0__after92", "doc122_qa0__after92",
    "doc26_qa0__after93", "doc64_qa0__after93", "doc146_qa0__after93", "doc136_qa0__after93",
    "doc54_qa0__after93", "doc106_qa0__after93", "doc149_qa0__after93", "doc144_qa0__after93",
    "doc143_qa0__after93", "doc82_qa0__after93",
    "doc18_qa0__after94", "doc126_qa0__after94", "doc52_qa0__after94", "doc9_qa0__after94",
    "doc64_qa0__after94", "doc117_qa0__after94", "doc129_qa0__after94", "doc83_qa0__after94",
    "doc112_qa0__after94", "doc104_qa0__after94",
    "doc18_qa0__after95", "doc80_qa0__after95", "doc52_qa0__after95", "doc100_qa0__after95",
    "doc106_qa0__after95", "doc51_qa0__after95", "doc142_qa0__after95", "doc122_qa0__after95",
    "doc8_qa0__after95", "doc17_qa0__after95",
    "doc86_qa0__after96", "doc80_qa0__after96", "doc94_qa0__after96", "doc15_qa0__after96",
    "doc95_qa0__after96", "doc127_qa0__after96", "doc53_qa0__after96", "doc52_qa0__after96",
    "doc50_qa0__after96", "doc39_qa0__after96",
    "doc133_qa0__after97", "doc63_qa0__after97", "doc118_qa0__after97", "doc8_qa0__after97",
    "doc47_qa0__after97", "doc125_qa0__after97", "doc95_qa0__after97", "doc37_qa0__after97",
    "doc6_qa0__after97", "doc50_qa0__after97",
    "doc42_qa0__after98", "doc141_qa0__after98", "doc80_qa0__after98", "doc91_qa0__after98",
    "doc60_qa0__after98", "doc149_qa0__after98", "doc108_qa0__after98", "doc97_qa0__after98",
    "doc138_qa0__after98", "doc16_qa0__after98",
    "doc113_qa0__after99", "doc11_qa0__after99", "doc40_qa0__after99", "doc127_qa0__after99",
    "doc108_qa0__after99", "doc145_qa0__after99", "doc43_qa0__after99", "doc71_qa0__after99",
    "doc124_qa0__after99", "doc116_qa0__after99",
    "doc5_qa0__after100", "doc129_qa0__after100", "doc10_qa0__after100", "doc90_qa0__after100",
    "doc148_qa0__after100", "doc15_qa0__after100", "doc67_qa0__after100", "doc127_qa0__after100",
    "doc65_qa0__after100", "doc63_qa0__after100",
    "doc81_qa0__after101", "doc114_qa0__after101", "doc35_qa0__after101", "doc41_qa0__after101",
    "doc100_qa0__after101", "doc98_qa0__after101", "doc78_qa0__after101", "doc75_qa0__after101",
    "doc96_qa0__after101", "doc125_qa0__after101",
    "doc31_qa0__after102", "doc39_qa0__after102", "doc24_qa0__after102", "doc68_qa0__after102",
    "doc119_qa0__after102", "doc44_qa0__after102", "doc36_qa0__after102", "doc59_qa0__after102",
    "doc46_qa0__after102", "doc108_qa0__after102",
    "doc108_qa0__after103", "doc61_qa0__after103", "doc135_qa0__after103", "doc60_qa0__after103",
    "doc36_qa0__after103", "doc51_qa0__after103", "doc85_qa0__after103", "doc105_qa0__after103",
    "doc71_qa0__after103", "doc137_qa0__after103",
    "doc46_qa0__after104", "doc136_qa0__after104", "doc121_qa0__after104", "doc96_qa0__after104",
    "doc16_qa0__after104", "doc80_qa0__after104", "doc31_qa0__after104", "doc14_qa0__after104",
    "doc101_qa0__after104", "doc103_qa0__after104",
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
