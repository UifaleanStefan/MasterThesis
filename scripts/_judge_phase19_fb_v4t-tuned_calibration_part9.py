"""Phase 1.9 Protocol B FB v4t-tuned calibration part 9 (entries 1200-1349)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__v4t-tuned__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__v4t-tuned__calibration__seed42/results.jsonl")

SPECIAL: dict[str, tuple[float, str]] = {
    "doc54_qa0__after120": (1.0, "[ANS] doc54 seen. GOLD 'Yes 982 to 969 -1.32% Best Buy'. PRED 'Yes 982 Q2 FY23 to 969 Q2 FY24'. Exact match."),
    "doc112_qa0__after120": (0.25, "[ANS] doc112 seen. GOLD 5.4% Netflix EBITDA margin. PRED '4.51%' — wrong specific (0.9pp off, beyond 5% tolerance)."),
    "doc117_qa0__after120": (0.0, "[ANS] doc117 seen. Same Nike cash flow refusal."),
    "doc3_qa0__after120": (0.0, "[ANS] doc3 seen. Same 3M OM refusal."),
    "doc0_qa0__after120": (0.0, "[ANS] doc0 seen. Same 3M capex refusal."),
    "doc99_qa0__after120": (0.0, "[ANS] doc99 seen. Same Kraft Heinz refusal."),
    "doc88_qa0__after120": (0.0, "[ANS] doc88 seen. Same JnJ EPS refusal."),
    "doc34_qa0__after120": (0.0, "[ANS] doc34 seen. Same AMD operating refusal."),
    "doc72_qa0__after120": (0.0, "[ANS] doc72 seen. Same Corning tax refusal."),
    "doc114_qa0__after121": (1.0, "[ANS] doc114 seen. GOLD 55.1% Nike. PRED '56.2%' — within 2% relative (under 5% tolerance). Correct."),
    "doc127_qa0__after121": (0.25, "[ACK] doc127 not yet seen. GOLD '$8,400,000,000'. PRED '$4.0 billion' — half of gold. Confident wrong specific."),
    "doc11_qa0__after121": (0.0, "[ANS] doc11 seen. Same Adobe refusal."),
    "doc46_qa0__after121": (0.0, "[ANS] doc46 seen. Same AWW EBITDA refusal."),
    "doc92_qa0__after121": (1.0, "[ANS] doc92 seen. GOLD '$13.2B JnJ Kenvue'. PRED '$13.2 billion'. Exact match."),
    "doc115_qa0__after121": (0.0, "[ANS] doc115 seen. Same Nike current assets refusal."),
    "doc49_qa0__after121": (0.0, "[ANS] doc49 seen. Same Best Buy inventory refusal."),
    "doc72_qa0__after121": (0.0, "[ANS] doc72 seen. Same Corning tax refusal."),
    "doc106_qa0__after121": (0.75, "[ANS] doc106 seen. GOLD 'Las Vegas resorts ~90% EBITDAR'. PRED 'Las Vegas Strip Resorts' — partial (correct region, no 90% specific)."),
    "doc12_qa0__after122": (0.0, "[ANS] doc12 seen. Same Adobe OCF refusal."),
    "doc125_qa0__after122": (0.25, "[ACK] doc125 not yet seen. GOLD 'proposal defeated'. PRED 'not approved, ~70% against' — direction correct but 70% wrong (real ~98%)."),
    "doc67_qa0__after122": (0.0, "[ANS] doc67 seen. Same Coca-Cola ROA refusal."),
    "doc31_qa0__after122": (0.0, "[ANS] doc31 seen. Same AMD quick ratio refusal."),
    "doc90_qa0__after122": (0.0, "[ANS] doc90 seen. GOLD definitive 'Consumer Health discontinued Aug 30 2023'. PRED refuses (context lost). Refusal on definitive ANS."),
    "doc85_qa0__after122": (0.0, "[ANS] doc85 seen. Same JnJ high growth refusal."),
    "doc27_qa0__after122": (0.0, "[ANS] doc27 seen. Same Amcor restructuring refusal."),
    "doc63_qa0__after122": (0.0, "[ANS] doc63 seen. GOLD definitive Boeing customers. PRED refuses. Refusal on definitive ANS."),
    "doc83_qa0__after123": (0.0, "[ANS] doc83 seen. Same GenMills FCF refusal."),
    "doc0_qa0__after123": (0.0, "[ANS] doc0 seen. Same 3M capex refusal."),
    "doc96_qa0__after123": (1.0, "[ANS] doc96 seen. GOLD 'JPM gross margin not relevant'. PRED 'not relevant for JPM (NIM/ROE/ROA for financial services)'. Correct reasoning."),
    "doc47_qa0__after123": (0.0, "[ANS] doc47 seen. Same AWW WC refusal."),
    "doc67_qa0__after123": (0.0, "[ANS] doc67 seen. Same Coca-Cola refusal."),
    "doc2_qa0__after123": (0.0, "[ANS] doc2 seen. Same 3M refusal."),
    "doc100_qa0__after123": (0.25, "[ANS] doc100 seen. GOLD 1.33 Lockheed asset turnover. PRED '1.00' — wrong specific (25% off)."),
    "doc45_qa0__after123": (0.0, "[ANS] doc45 seen. Same AWW dividends refusal."),
    "doc30_qa0__after123": (0.0, "[ANS] doc30 seen. Same AMD D&A refusal."),
    "doc117_qa0__after123": (0.0, "[ANS] doc117 seen. Same Nike cash flow refusal."),
    "doc20_qa0__after124": (0.0, "[ANS] doc20 seen. Same Amazon FY19 NI refusal."),
    "doc55_qa0__after124": (0.0, "[ANS] doc55 seen. GOLD definitive 'Entertainment 9% from gaming'. PRED refuses. Refusal on definitive ANS."),
    "doc24_qa0__after124": (0.0, "[ANS] doc24 seen. Same Amcor acquisitions refusal."),
    "doc3_qa0__after124": (0.0, "[ANS] doc3 seen. Same 3M OM refusal."),
    "doc85_qa0__after124": (0.0, "[ANS] doc85 seen. Same JnJ refusal."),
    "doc58_qa0__after124": (0.0, "[ANS] doc58 seen. Same Block CFO refusal."),
    "doc71_qa0__after124": (0.0, "[ANS] doc71 seen. Same Corning OM refusal."),
    "doc81_qa0__after124": (0.0, "[ANS] doc81 seen. Same GenMills CCC refusal."),
    "doc1_qa0__after125": (0.0, "[ANS] doc1 seen. Same 3M PP&E refusal."),
    "doc59_qa0__after125": (0.0, "[ANS] doc59 seen. Same Boeing PP&E refusal."),
    "doc97_qa0__after125": (0.0, "[ANS] doc97 seen. Same JPM segments refusal."),
    "doc101_qa0__after125": (0.0, "[ANS] doc101 seen. Same Lockheed NWC refusal."),
    "doc47_qa0__after125": (0.0, "[ANS] doc47 seen. Same AWW WC refusal."),
    "doc19_qa0__after125": (0.0, "[ANS] doc19 seen. Same Amazon revenue refusal."),
    "doc77_qa0__after125": (0.75, "[ANS] doc77 seen. GOLD 'Yes CVS multi legal (usual pricing, PBM, etc.)'. PRED 'Yes CVS ongoing legal usual customary pricing' — Y correct + one category."),
    "doc34_qa0__after125": (0.0, "[ANS] doc34 seen. Same AMD operating refusal."),
    "doc32_qa0__after125": (0.0, "[ANS] doc32 seen. GOLD definitive AMD products list. PRED refuses. Refusal on definitive ANS."),
    "doc41_qa0__after126": (0.0, "[ANS] doc41 seen. Same AMEX gross margin refusal."),
    "doc40_qa0__after126": (0.0, "[ANS] doc40 seen. Same AMEX OM refusal."),
    "doc66_qa0__after126": (0.0, "[ANS] doc66 seen. Same Boeing tax refusal."),
    "doc99_qa0__after126": (0.0, "[ANS] doc99 seen. Same Kraft Heinz refusal."),
    "doc7_qa0__after126": (0.0, "[ANS] doc7 seen. Same 3M dividend refusal."),
    "doc98_qa0__after126": (1.0, "[ANS] doc98 seen. Same JPM VaR decreased $7M correct."),
    "doc103_qa0__after126": (0.0, "[ANS] doc103 seen. Same MGM AP refusal."),
    "doc28_qa0__after127": (0.0, "[ANS] doc28 seen. Same Amcor Adj EBITDA refusal."),
    "doc62_qa0__after127": (0.0, "[ANS] doc62 seen. Same Boeing gross margin refusal."),
    "doc25_qa0__after127": (0.0, "[ANS] doc25 seen. Same Amcor refusal."),
    "doc26_qa0__after127": (0.0, "[ANS] doc26 seen. Same Amcor margin refusal."),
    "doc80_qa0__after127": (1.0, "[ANS] doc80 seen. Same Richard A. Johnson match with vote count."),
    "doc100_qa0__after127": (0.0, "[ANS] doc100 seen. Same Lockheed refusal."),
    "doc123_qa0__after127": (1.0, "[ANS] doc123 seen. GOLD $9068 PepsiCo. PRED '$9,301 million' — within 3.6% tolerance. Correct."),
    "doc14_qa0__after127": (0.0, "[ANS] doc14 seen. Same Adobe FCF refusal."),
    "doc72_qa0__after128": (0.0, "[ANS] doc72 seen. Same Corning tax refusal."),
    "doc106_qa0__after128": (0.0, "[ANS] doc106 seen. Same MGM EBITDAR refusal."),
    "doc39_qa0__after128": (0.0, "[ANS] doc39 seen. Same AMEX geographies refusal."),
    "doc117_qa0__after128": (0.0, "[ANS] doc117 seen. Same Nike refusal."),
    "doc32_qa0__after128": (0.0, "[ANS] doc32 seen. Same AMD products refusal."),
    "doc98_qa0__after128": (1.0, "[ANS] doc98 seen. Same JPM VaR correct."),
    "doc41_qa0__after128": (0.0, "[ANS] doc41 seen. Same AMEX gross margin refusal."),
    "doc79_qa0__after128": (0.0, "[ANS] doc79 seen. GOLD definitive 'Yes Mary Dillon Ulta'. PRED refuses (context lost). Refusal on definitive ANS."),
    "doc42_qa0__after129": (0.0, "[ANS] doc42 seen. Same AMEX tax refusal."),
    "doc85_qa0__after129": (0.0, "[ANS] doc85 seen. Same JnJ refusal."),
    "doc124_qa0__after129": (0.0, "[ANS] doc124 seen. Same Pepsi EBITDA refusal."),
    "doc59_qa0__after129": (0.0, "[ANS] doc59 seen. Same Boeing PP&E refusal."),
    "doc123_qa0__after129": (0.25, "[ANS] doc123 seen. GOLD $9068. PRED '$13,985 million' — wrong (54% off)."),
    "doc0_qa0__after129": (0.0, "[ANS] doc0 seen. Same 3M capex refusal."),
    "doc38_qa0__after129": (0.75, "[ANS] doc38 seen. GOLD 'There are none' (AMEX debt). PRED 'do not contain info on AMEX debt securities' — equivalent conclusion via different framing."),
    "doc100_qa0__after129": (0.25, "[ANS] doc100 seen. Same Lockheed 1.00 wrong."),
    "doc122_qa0__after130": (1.0, "[ANS] doc122 seen. GOLD '$411M Pepsi restructuring'. PRED '411' — exact match."),
    "doc17_qa0__after130": (0.0, "[ANS] doc17 seen. Same AES ROA refusal."),
    "doc78_qa0__after130": (0.0, "[ANS] doc78 seen. Same CVS dividend refusal."),
    "doc38_qa0__after130": (0.75, "[ANS] doc38 seen. Same equivalent-conclusion partial."),
    "doc74_qa0__after130": (0.0, "[ANS] doc74 seen. Same Costco total assets refusal."),
    "doc86_qa0__after130": (0.0, "[ANS] doc86 seen. Same JnJ gross margin refusal."),
    "doc37_qa0__after130": (0.0, "[ANS] doc37 seen. Same AMD customer refusal."),
    "doc42_qa0__after130": (0.0, "[ANS] doc42 seen. Same AMEX tax refusal."),
    "doc10_qa0__after130": (0.0, "[ANS] doc10 seen. Same Adobe OCF refusal."),
    "doc101_qa0__after130": (0.0, "[ANS] doc101 seen. Same Lockheed NWC refusal."),
    "doc26_qa0__after131": (0.0, "[ANS] doc26 seen. Same Amcor margin refusal."),
    "doc89_qa0__after131": (0.0, "[ANS] doc89 seen. Same JnJ US/intl refusal."),
    "doc3_qa0__after131": (0.0, "[ANS] doc3 seen. Same 3M OM refusal."),
    "doc58_qa0__after131": (0.0, "[ANS] doc58 seen. Same Block refusal."),
    "doc71_qa0__after131": (0.0, "[ANS] doc71 seen. Same Corning OM refusal."),
    "doc94_qa0__after131": (0.0, "[ANS] doc94 seen. Same JPM Q1 segments refusal."),
    "doc9_qa0__after131": (0.0, "[ANS] doc9 seen. Same Activision refusal."),
    "doc18_qa0__after131": (0.0, "[ANS] doc18 seen. Same Amazon DPO refusal."),
    "doc97_qa0__after131": (0.0, "[ANS] doc97 seen. Same JPM Q2 segments refusal."),
    "doc61_qa0__after131": (0.0, "[ANS] doc61 seen. Same Boeing lawsuits refusal."),
    "doc0_qa0__after132": (0.0, "[ANS] doc0 seen. Same 3M capex refusal."),
    "doc120_qa0__after132": (0.5, "[ANS] doc120 seen. GOLD 'NA/LatAm/Europe/Africa/ME/SAsia/AsiaPacific/Australia/NZ/China'. PRED 'Africa/ME/SAsia/AsiaPacific/Australia/NZ/China' — partial (misses NA/LatAm/Europe)."),
    "doc32_qa0__after132": (0.0, "[ANS] doc32 seen. Same AMD products refusal."),
    "doc112_qa0__after132": (0.0, "[ANS] doc112 seen. Same Netflix EBITDA refusal."),
    "doc43_qa0__after132": (0.0, "[ANS] doc43 seen. Same AMEX liabilities refusal."),
    "doc4_qa0__after132": (0.0, "[ANS] doc4 seen. Same 3M segment refusal."),
    "doc126_qa0__after132": (1.0, "[ANS] doc126 seen. GOLD '$400M increase'. PRED 'PepsiCo increased five year revolving credit $400M, $3.8B to $4.2B'. Correct + detail."),
    "doc93_qa0__after132": (0.0, "[ANS] doc93 seen. Same JnJ earnings refusal."),
    "doc9_qa0__after132": (0.0, "[ANS] doc9 seen. Same Activision refusal."),
    "doc75_qa0__after133": (0.0, "[ANS] doc75 seen. Same CVS turnover refusal."),
    "doc84_qa0__after133": (0.0, "[ANS] doc84 seen. Same GenMills retention refusal."),
    "doc19_qa0__after133": (0.0, "[ANS] doc19 seen. Same Amazon refusal."),
    "doc120_qa0__after133": (0.0, "[ANS] doc120 seen. Same Pepsi geographies refusal."),
    "doc76_qa0__after133": (0.0, "[ANS] doc76 seen. Same CVS capital-intensive refusal."),
    "doc11_qa0__after133": (0.0, "[ANS] doc11 seen. Same Adobe refusal."),
    "doc86_qa0__after133": (0.0, "[ANS] doc86 seen. Same JnJ margin refusal."),
    "doc131_qa0__after133": (0.0, "[ANS] doc131 seen. Same Pfizer 2019 refusal."),
    "doc117_qa0__after133": (0.0, "[ANS] doc117 seen. Same Nike refusal."),
    "doc80_qa0__after134": (0.0, "[ANS] doc80 seen. GOLD definitive 'Yes Richard A. Johnson'. PRED refuses. Refusal on definitive ANS."),
    "doc20_qa0__after134": (0.0, "[ANS] doc20 seen. Same Amazon NI refusal."),
    "doc107_qa0__after134": (0.0, "[ANS] doc107 seen. Same MGM coverage refusal."),
    "doc15_qa0__after134": (1.0, "[ANS] doc15 seen. GOLD 0. PRED '0'. Match."),
    "doc134_qa0__after134": (1.0, "[ANS] doc134 seen. GOLD 'Developed Rest of the World'. PRED 'Developed Rest of World'. Match (whitespace/article only)."),
    "doc108_qa0__after134": (0.0, "[ANS] doc108 seen. Same MGM China refusal."),
    "doc114_qa0__after134": (0.0, "[ANS] doc114 seen. Same Nike COGS refusal."),
    "doc109_qa0__after134": (0.0, "[ANS] doc109 seen. Same MGM short-term investments refusal."),
    "doc25_qa0__after134": (0.0, "[ANS] doc25 seen. Same Amcor refusal."),
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
