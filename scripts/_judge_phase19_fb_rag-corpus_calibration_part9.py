"""Phase 1.9 extension: FB rag-corpus calibration part 9 (entries 1200-1349)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__rag-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__rag-corpus__calibration__seed42/results.jsonl")

SPECIAL: dict[str, tuple[float, str]] = {
    "doc54_qa0__after120": (1.0, "[ANS] doc54 seen. '982 to 969' EXACT numbers match."),
    "doc121_qa0__after120": (0.25, "[ACK] doc121 not yet seen. GOLD 'No, not material legal'. PRED 'Yes, lawsuits re: drug pricing' wrong (gold says No + fabricates lawsuits for PepsiCo which isn't pharma)."),
    "doc112_qa0__after120": (0.25, "[ANS] doc112 seen. PRED '4.5%' wrong specific (17% off from 5.4%)."),
    "doc117_qa0__after120": (1.0, "[ANS] doc117 seen. 'Cash from operations highest for Nike FY2023 $5,841M'. Correct + amount."),
    "doc3_qa0__after120": (0.75, "[ANS] doc3 seen. Same partial 3M OM."),
    "doc0_qa0__after120": (1.0, "[ANS] doc0 seen. PRED '$1,501M' within 4.8% tolerance."),
    "doc99_qa0__after120": (1.0, "[ANS] doc99 seen. '6.20' within 1% tolerance."),
    "doc88_qa0__after120": (0.0, "[ANS] doc88 seen. '+12.5% accelerate' Y/N flip."),
    "doc34_qa0__after120": (1.0, "[ANS] doc34 seen. Verbatim AMD Xilinx amortization match."),
    "doc72_qa0__after120": (1.0, "[ANS] doc72 seen. '20% to 23%' EXACT."),
    "doc114_qa0__after121": (1.0, "[ANS] doc114 seen. PRED '56.3%' within 2.2% tolerance."),
    "doc127_qa0__after121": (0.25, "[ACK] doc127 not yet seen. PRED '$4.0 billion' wrong specific (gold $8.4B)."),
    "doc11_qa0__after121": (1.0, "[ANS] doc11 seen. PRED computes 65.4% with CORRECT numbers ($903,095/$1,493,602) — finally right!"),
    "doc46_qa0__after121": (1.0, "[ANS] doc46 seen. '1,829' within 0.16% tolerance."),
    "doc92_qa0__after121": (1.0, "[ANS] doc92 seen. '$13.2 billion' exact."),
    "doc115_qa0__after121": (1.0, "[ANS] doc115 seen. GOLD $16525. PRED '16,525' EXACT."),
    "doc49_qa0__after121": (1.0, "[ANS] doc49 seen. '5,409' exact."),
    "doc72_qa0__after121": (1.0, "[ANS] doc72 seen. '20% to 23%' exact."),
    "doc106_qa0__after121": (0.75, "[ANS] doc106 seen. 'Las Vegas Strip Resorts' partial."),
    "doc12_qa0__after122": (0.25, "[ANS] doc12 seen. PRED '1.23' wrong."),
    "doc125_qa0__after122": (0.25, "[ACK] doc125 not yet seen. PRED 'not approved, 66.4% against' — direction right but 66.4% wrong (real ~98% against)."),
    "doc67_qa0__after122": (0.25, "[ANS] doc67 seen. PRED '1.46%' wrong specific."),
    "doc31_qa0__after122": (0.0, "[ANS] doc31 seen. Same AMD refusal."),
    "doc90_qa0__after122": (1.0, "[ANS] doc90 seen. PRED matches."),
    "doc85_qa0__after122": (1.0, "[ANS] doc85 seen. 'No 1.3% vs 13.6%' rich match."),
    "doc27_qa0__after122": (0.5, "[ANS] doc27 seen. Same generic restructuring."),
    "doc63_qa0__after122": (0.5, "[ANS] doc63 seen. Same partial 'limited commercial airlines'."),
    "doc83_qa0__after123": (1.0, "[ANS] doc83 seen. GOLD $3215. PRED '$3,115.4M' within 3.1% tolerance."),
    "doc0_qa0__after123": (1.0, "[ANS] doc0 seen. '$1,501M' within tol."),
    "doc96_qa0__after123": (1.0, "[ANS] doc96 seen. Correct JPM reasoning."),
    "doc47_qa0__after123": (0.75, "[ANS] doc47 seen. Same self-contradictory with correct -$1,561M."),
    "doc67_qa0__after123": (0.25, "[ANS] doc67 seen. '1.46%' wrong."),
    "doc2_qa0__after123": (0.0, "[ANS] doc2 seen. Same Y/N flip."),
    "doc100_qa0__after123": (1.0, "[ANS] doc100 seen. '1.30' within tol."),
    "doc45_qa0__after123": (1.0, "[ANS] doc45 seen. '0.389 billion' within tol."),
    "doc30_qa0__after123": (1.0, "[ANS] doc30 seen. 4.18% within tol."),
    "doc117_qa0__after123": (1.0, "[ANS] doc117 seen. Correct."),
    "doc20_qa0__after124": (1.0, "[ANS] doc20 seen. '11,588' exact."),
    "doc55_qa0__after124": (0.75, "[ANS] doc55 seen. 'Gaming 9.0% domestic' — gaming + 9% mentioned but no Entertainment segment ID."),
    "doc24_qa0__after124": (0.0, "[ANS] doc24 seen. PRED refuses. Refusal on definitive ANS."),
    "doc3_qa0__after124": (0.75, "[ANS] doc3 seen. Same partial."),
    "doc85_qa0__after124": (1.0, "[ANS] doc85 seen. Same rich correct."),
    "doc58_qa0__after124": (1.0, "[ANS] doc58 seen. '$381.6M' within tol."),
    "doc71_qa0__after124": (1.0, "[ANS] doc71 seen. '10.5%' within tol."),
    "doc81_qa0__after124": (0.25, "[ANS] doc81 seen. '66.73 days' wrong specific (gold -3.7)."),
    "doc1_qa0__after125": (1.0, "[ANS] doc1 seen. '$8.738B' within tol."),
    "doc59_qa0__after125": (1.0, "[ANS] doc59 seen. '$12,645' exact."),
    "doc97_qa0__after125": (0.25, "[ANS] doc97 seen. 'Consumer & Community Banking' wrong segment."),
    "doc101_qa0__after125": (1.0, "[ANS] doc101 seen. '$5,818 million' exact."),
    "doc47_qa0__after125": (0.75, "[ANS] doc47 seen. Same self-contradictory + correct."),
    "doc19_qa0__after125": (1.0, "[ANS] doc19 seen. '30.7%' within 0.3% tolerance."),
    "doc77_qa0__after125": (0.75, "[ANS] doc77 seen. Y + one CVS legal category."),
    "doc34_qa0__after125": (1.0, "[ANS] doc34 seen. Verbatim."),
    "doc32_qa0__after125": (1.0, "[ANS] doc32 seen. Verbatim AMD products."),
    "doc41_qa0__after126": (1.0, "[ANS] doc41 seen. Correct."),
    "doc40_qa0__after126": (1.0, "[ANS] doc40 seen. Correct."),
    "doc66_qa0__after126": (0.25, "[ANS] doc66 seen. Same Boeing tax $ amounts partial."),
    "doc99_qa0__after126": (1.0, "[ANS] doc99 seen. '6.20' within tol."),
    "doc7_qa0__after126": (1.0, "[ANS] doc7 seen. 65 years correct."),
    "doc98_qa0__after126": (1.0, "[ANS] doc98 seen. VaR correct."),
    "doc103_qa0__after126": (1.0, "[ANS] doc103 seen. '$302.578 million' within 0.14%."),
    "doc28_qa0__after127": (1.0, "[ANS] doc28 seen. $2,018M exact."),
    "doc62_qa0__after127": (0.25, "[ANS] doc62 seen. 'Not useful for Boeing' wrong reasoning."),
    "doc25_qa0__after127": (0.75, "[ANS] doc25 seen. Minimal."),
    "doc26_qa0__after127": (0.75, "[ANS] doc26 seen. Same partial."),
    "doc80_qa0__after127": (1.0, "[ANS] doc80 seen. Correct + votes."),
    "doc100_qa0__after127": (1.0, "[ANS] doc100 seen. '1.31' within tol."),
    "doc123_qa0__after127": (0.25, "[ANS] doc123 seen. GOLD $9068. PRED '$14,275 million' wrong (57% off)."),
    "doc14_qa0__after127": (0.0, "[ANS] doc14 seen. Same Adobe FCF refusal."),
    "doc72_qa0__after128": (1.0, "[ANS] doc72 seen. '20% to 23%' exact."),
    "doc106_qa0__after128": (0.75, "[ANS] doc106 seen. 'Las Vegas Strip Resorts' partial."),
    "doc39_qa0__after128": (1.0, "[ANS] doc39 seen. Match."),
    "doc117_qa0__after128": (1.0, "[ANS] doc117 seen. Correct."),
    "doc32_qa0__after128": (1.0, "[ANS] doc32 seen. Verbatim."),
    "doc98_qa0__after128": (1.0, "[ANS] doc98 seen. VaR correct."),
    "doc41_qa0__after128": (1.0, "[ANS] doc41 seen. Correct."),
    "doc79_qa0__after128": (1.0, "[ANS] doc79 seen. Correct."),
    "doc42_qa0__after129": (1.0, "[ANS] doc42 seen. Exact match."),
    "doc85_qa0__after129": (0.75, "[ANS] doc85 seen. 'No 1.3%' less detail than other ANS."),
    "doc124_qa0__after129": (0.25, "[ANS] doc124 seen. GOLD 16.5% margin. PRED computes EBITDA $14,275M but no % — wrong format/answer."),
    "doc59_qa0__after129": (1.0, "[ANS] doc59 seen. Exact."),
    "doc123_qa0__after129": (0.25, "[ANS] doc123 seen. Same $14,275M wrong."),
    "doc0_qa0__after129": (1.0, "[ANS] doc0 seen. Within tol."),
    "doc38_qa0__after129": (0.25, "[ANS] doc38 seen. 'Common Shares par $0.20' wrong (gold 'none' for debt securities)."),
    "doc100_qa0__after129": (1.0, "[ANS] doc100 seen. '1.30' within tol."),
    "doc122_qa0__after130": (1.0, "[ANS] doc122 seen. '411' exact match!"),
    "doc17_qa0__after130": (0.25, "[ANS] doc17 seen. '-1.41' wrong."),
    "doc78_qa0__after130": (1.0, "[ANS] doc78 seen. 'Yes CVS $0.55 quarterly Q2 FY2022' EXACT + amount."),
    "doc38_qa0__after130": (0.25, "[ANS] doc38 seen. Same Common Shares wrong."),
    "doc74_qa0__after130": (1.0, "[ANS] doc74 seen. GOLD $59268. PRED '59,268' EXACT!"),
    "doc86_qa0__after130": (0.0, "[ANS] doc86 seen. 'Gross margin not useful for JnJ (pharma/medical)' — wrong reasoning."),
    "doc37_qa0__after130": (1.0, "[ANS] doc37 seen. Correct."),
    "doc42_qa0__after130": (1.0, "[ANS] doc42 seen. Exact."),
    "doc10_qa0__after130": (0.25, "[ANS] doc10 seen. PRED '1.96' wrong."),
    "doc101_qa0__after130": (1.0, "[ANS] doc101 seen. Exact."),
    "doc26_qa0__after131": (0.75, "[ANS] doc26 seen. Same partial."),
    "doc89_qa0__after131": (1.0, "[ANS] doc89 seen. 'US 3.0%, intl -0.6%' exact."),
    "doc3_qa0__after131": (0.75, "[ANS] doc3 seen. Same partial OM."),
    "doc58_qa0__after131": (1.0, "[ANS] doc58 seen. '$381.6M' within tol."),
    "doc71_qa0__after131": (1.0, "[ANS] doc71 seen. '10.5%' within tol."),
    "doc94_qa0__after131": (0.25, "[ANS] doc94 seen. 'Commercial Banking' wrong (gold 'Corporate')."),
    "doc9_qa0__after131": (0.25, "[ANS] doc9 seen. '3.5%' wrong."),
    "doc18_qa0__after131": (0.25, "[ANS] doc18 seen. '34.12' wrong."),
    "doc97_qa0__after131": (0.25, "[ANS] doc97 seen. 'Consumer & Community Banking' wrong segment."),
    "doc61_qa0__after131": (1.0, "[ANS] doc61 seen. Full Lion Air/Ethiopian Airlines detail."),
    "doc0_qa0__after132": (1.0, "[ANS] doc0 seen. Within tol."),
    "doc120_qa0__after132": (0.5, "[ANS] doc120 seen. GOLD lists 10 geographies. PRED 'Americas, Europe, Africa, ME, S Asia, AP, Australia, NZ, China' — 'Americas' aggregates NA+LatAm. Partial overlap."),
    "doc32_qa0__after132": (1.0, "[ANS] doc32 seen. Verbatim."),
    "doc112_qa0__after132": (0.25, "[ANS] doc112 seen. '4.51%' wrong specific."),
    "doc43_qa0__after132": (0.25, "[ANS] doc43 seen. 'Long-term debt' wrong."),
    "doc4_qa0__after132": (0.75, "[ANS] doc4 seen. Partial."),
    "doc126_qa0__after132": (1.0, "[ANS] doc126 seen. '$400,000,000 increase from $3,800,000,000 to $4,200,000,000' EXACT."),
    "doc93_qa0__after132": (1.0, "[ANS] doc93 seen. '20.0% to 20.1%' exact."),
    "doc9_qa0__after132": (0.25, "[ANS] doc9 seen. '3.5%' wrong."),
    "doc75_qa0__after133": (0.25, "[ANS] doc75 seen. '8.73' wrong."),
    "doc84_qa0__after133": (0.25, "[ANS] doc84 seen. '0.46' 15% off from gold 0.54."),
    "doc19_qa0__after133": (1.0, "[ANS] doc19 seen. '30.7%' within tol."),
    "doc120_qa0__after133": (0.5, "[ANS] doc120 seen. Same 'Americas' grouping partial."),
    "doc76_qa0__after133": (0.75, "[ANS] doc76 seen. Y correct minimal."),
    "doc11_qa0__after133": (1.0, "[ANS] doc11 seen. PRED computes 65.4% with CORRECT numbers."),
    "doc86_qa0__after133": (0.0, "[ANS] doc86 seen. Same wrong reasoning."),
    "doc131_qa0__after133": (0.5, "[ANS] doc131 seen. 'Yes gain on Consumer Healthcare JV $(6)M' — Y match + odd negative number."),
    "doc117_qa0__after133": (1.0, "[ANS] doc117 seen. Correct."),
    "doc80_qa0__after134": (1.0, "[ANS] doc80 seen. Correct."),
    "doc20_qa0__after134": (1.0, "[ANS] doc20 seen. '11,588' exact."),
    "doc107_qa0__after134": (0.25, "[ANS] doc107 seen. '1.61' wrong specific."),
    "doc15_qa0__after134": (1.0, "[ANS] doc15 seen. '0' match."),
    "doc134_qa0__after134": (1.0, "[ANS] doc134 seen. 'Developed Rest of World' match."),
    "doc108_qa0__after134": (1.0, "[ANS] doc108 seen. 'MGM China worst $(203,136) thousand' correct ID + amount."),
    "doc114_qa0__after134": (1.0, "[ANS] doc114 seen. '56.3%' within tol."),
    "doc109_qa0__after134": (1.0, "[ANS] doc109 seen. 'Corporate bonds $416,420,000 largest'. Correct + amount."),
    "doc25_qa0__after134": (0.75, "[ANS] doc25 seen. Minimal."),
}

ENTRY_SUFFIXES: list[str] = [
    "doc54_qa0__after120", "doc121_qa0__after120", "doc112_qa0__after120", "doc117_qa0__after120",
    "doc3_qa0__after120", "doc0_qa0__after120", "doc99_qa0__after120", "doc88_qa0__after120",
    "doc34_qa0__after120", "doc72_qa0__after120",
    "doc114_qa0__after121", "doc127_qa0__after121", "doc11_qa0__after121", "doc136_qa0__after121",
    "doc46_qa0__after121", "doc92_qa0__after121", "doc115_qa0__after121", "doc49_qa0__after121",
    "doc72_qa0__after121", "doc106_qa0__after121",
    "doc12_qa0__after122", "doc125_qa0__after122", "doc128_qa0__after122", "doc67_qa0__after122",
    "doc31_qa0__after122", "doc134_qa0__after122", "doc90_qa0__after122", "doc85_qa0__after122",
    "doc27_qa0__after122", "doc63_qa0__after122",
    "doc83_qa0__after123", "doc0_qa0__after123", "doc96_qa0__after123", "doc47_qa0__after123",
    "doc67_qa0__after123", "doc2_qa0__after123", "doc100_qa0__after123", "doc45_qa0__after123",
    "doc30_qa0__after123", "doc117_qa0__after123",
    "doc20_qa0__after124", "doc128_qa0__after124", "doc55_qa0__after124", "doc24_qa0__after124",
    "doc3_qa0__after124", "doc139_qa0__after124", "doc85_qa0__after124", "doc58_qa0__after124",
    "doc71_qa0__after124", "doc81_qa0__after124",
    "doc1_qa0__after125", "doc59_qa0__after125", "doc97_qa0__after125", "doc143_qa0__after125",
    "doc101_qa0__after125", "doc47_qa0__after125", "doc19_qa0__after125", "doc77_qa0__after125",
    "doc34_qa0__after125", "doc32_qa0__after125",
    "doc132_qa0__after126", "doc130_qa0__after126", "doc41_qa0__after126", "doc40_qa0__after126",
    "doc66_qa0__after126", "doc99_qa0__after126", "doc7_qa0__after126", "doc142_qa0__after126",
    "doc98_qa0__after126", "doc103_qa0__after126",
    "doc28_qa0__after127", "doc130_qa0__after127", "doc62_qa0__after127", "doc25_qa0__after127",
    "doc26_qa0__after127", "doc80_qa0__after127", "doc135_qa0__after127", "doc100_qa0__after127",
    "doc123_qa0__after127", "doc14_qa0__after127",
    "doc72_qa0__after128", "doc131_qa0__after128", "doc106_qa0__after128", "doc39_qa0__after128",
    "doc117_qa0__after128", "doc141_qa0__after128", "doc32_qa0__after128", "doc98_qa0__after128",
    "doc41_qa0__after128", "doc79_qa0__after128",
    "doc134_qa0__after129", "doc42_qa0__after129", "doc85_qa0__after129", "doc124_qa0__after129",
    "doc59_qa0__after129", "doc123_qa0__after129", "doc0_qa0__after129", "doc38_qa0__after129",
    "doc100_qa0__after129", "doc146_qa0__after129",
    "doc122_qa0__after130", "doc17_qa0__after130", "doc78_qa0__after130", "doc38_qa0__after130",
    "doc74_qa0__after130", "doc86_qa0__after130", "doc37_qa0__after130", "doc42_qa0__after130",
    "doc10_qa0__after130", "doc101_qa0__after130",
    "doc26_qa0__after131", "doc89_qa0__after131", "doc3_qa0__after131", "doc58_qa0__after131",
    "doc71_qa0__after131", "doc94_qa0__after131", "doc9_qa0__after131", "doc18_qa0__after131",
    "doc97_qa0__after131", "doc61_qa0__after131",
    "doc0_qa0__after132", "doc120_qa0__after132", "doc32_qa0__after132", "doc141_qa0__after132",
    "doc112_qa0__after132", "doc43_qa0__after132", "doc4_qa0__after132", "doc126_qa0__after132",
    "doc93_qa0__after132", "doc9_qa0__after132",
    "doc75_qa0__after133", "doc84_qa0__after133", "doc19_qa0__after133", "doc120_qa0__after133",
    "doc76_qa0__after133", "doc11_qa0__after133", "doc86_qa0__after133", "doc131_qa0__after133",
    "doc148_qa0__after133", "doc117_qa0__after133",
    "doc80_qa0__after134", "doc143_qa0__after134", "doc20_qa0__after134", "doc107_qa0__after134",
    "doc15_qa0__after134", "doc134_qa0__after134", "doc108_qa0__after134", "doc114_qa0__after134",
    "doc109_qa0__after134", "doc25_qa0__after134",
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
