"""Phase 1.9 extension: FB rag-corpus calibration part 3 (entries 300-449)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__rag-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__rag-corpus__calibration__seed42/results.jsonl")

SPECIAL: dict[str, tuple[float, str]] = {
    "doc12_qa0__after30": (0.25, "[ANS] doc12 seen. GOLD 0.83. PRED '0.51' wrong specific."),
    "doc52_qa0__after30": (0.5, "[ACK] doc52 not yet seen. GOLD 'Best Buy operations $1.8bn'. PRED 'cash flow from operating activities brought in the most' — confident-correct on which without $1.8bn. Partial honesty."),
    "doc0_qa0__after30": (0.0, "[ANS] doc0 seen. GOLD $1577. PRED 'not explicitly provided' — refusal on definitive ANS."),
    "doc5_qa0__after30": (0.0, "[ANS] doc5 seen. Same 3M quick ratio refusal."),
    "doc90_qa0__after30": (0.5, "[ACK] doc90 not yet seen. PRED exact 'JnJ Consumer Health Aug 30 2023' world-knowledge match."),
    "doc21_qa0__after31": (1.0, "[ANS] doc21 seen. GOLD $1616. PRED '$1,615.9 million' within 0.01% tolerance."),
    "doc63_qa0__after31": (0.25, "[ACK] doc63 not yet seen. Same 'defense contractors' fabrication."),
    "doc18_qa0__after31": (0.25, "[ANS] doc18 seen. GOLD 93.86. PRED '36.45' wrong specific."),
    "doc18_qa0__after32": (0.25, "[ANS] doc18 seen. PRED '36.12' wrong specific."),
    "doc7_qa0__after32": (1.0, "[ANS] doc7 seen. GOLD '65 consecutive years 3M dividend'. PRED matches. Correct."),
    "doc18_qa0__after33": (0.25, "[ANS] doc18 seen. PRED '36.12' wrong specific."),
    "doc34_qa0__after33": (0.25, "[ACK] doc34 not yet seen. GOLD 'AMD op income decrease from Xilinx amortization'. PRED 'revenue increase from Data Center 64% + Gaming 21% + Embedded' — wrong topic (revenue drivers vs OM drivers)."),
    "doc15_qa0__after33": (1.0, "[ANS] doc15 seen. GOLD 0. PRED '0'. Match."),
    "doc90_qa0__after33": (0.5, "[ACK] doc90 not yet seen. Same JnJ Consumer Health world-knowledge match."),
    "doc64_qa0__after33": (0.5, "[ACK] doc64 not yet seen. PRED 'Yes Boeing cyclicality' confident-correct without doc basis."),
    "doc125_qa0__after33": (0.5, "[ACK] doc125 not yet seen. PRED 'proposal not approved' equivalent to gold 'defeated'."),
    "doc26_qa0__after34": (0.75, "[ANS] doc26 seen. GOLD 'No, gross margin decline 0.8%'. PRED 'Amcor gross profit FY2023 $2,725M lower than $2,820M FY2022, declining gross margin profile' — direction correct + supporting calc, missing 0.8% specific."),
    "doc25_qa0__after34": (1.0, "[ANS] doc25 seen. GOLD 'Amcor packaging leader for various use cases'. PRED 'Amcor packaging industry, food/beverage/pharma/medical/home/personal-care'. Correct + rich detail."),
    "doc34_qa0__after34": (1.0, "[ANS] doc34 seen. GOLD 'AMD op income decrease from Xilinx amortization'. PRED matches verbatim. Correct."),
    "doc29_qa0__after34": (0.25, "[ANS] doc29 seen. GOLD 'Real Growth flat FY2023 vs FY2022'. PRED 'decrease of 5%' wrong direction."),
    "doc31_qa0__after36": (0.0, "[ANS] doc31 seen. Same AMD quick ratio refusal."),
    "doc3_qa0__after36": (0.75, "[ANS] doc3 seen. GOLD '3M OM -1.7% from Combat Arms/PFAS/Russia'. PRED 'litigation, impairment, restructuring + SG&A' — partial."),
    "doc11_qa0__after37": (0.25, "[ANS] doc11 seen. GOLD 65.4%. PRED gives 36.9% with wrong numbers ($5,802M vs $4,237M)."),
    "doc10_qa0__after37": (0.25, "[ANS] doc10 seen. GOLD 0.66. PRED '1.29' wrong specific."),
    "doc90_qa0__after37": (0.5, "[ACK] doc90 not yet seen. Same JnJ Consumer Health match."),
    "doc90_qa0__after38": (0.5, "[ACK] doc90 not yet seen. Same JnJ Consumer Health match."),
    "doc43_qa0__after38": (0.25, "[ACK] doc43 not yet seen. GOLD 'Customer deposits'. PRED 'accounts payable' confident wrong specific."),
    "doc1_qa0__after38": (1.0, "[ANS] doc1 seen. PRED '$8.738 billion' within 0.43% tolerance."),
    "doc27_qa0__after38": (0.5, "[ANS] doc27 seen. GOLD '87% Employee liabilities'. PRED generic 'employee + fixed asset + other costs $93M' — no 87% specific."),
    "doc24_qa0__after38": (0.75, "[ANS] doc24 seen. GOLD lists 3 Amcor FY23 acquisitions (Czech, Shanghai, NZ). PRED gives Shanghai $60M + NZ $45M in FY23 + Czech in FY22 — wrong year for Czech but covers all 3 with amounts. Partial."),
    "doc8_qa0__after39": (0.25, "[ANS] doc8 seen. GOLD 24.26 Activision FY19 FAT ratio. PRED '25.66' — 5.7% off, beyond 5% tolerance."),
    "doc33_qa0__after39": (1.0, "[ANS] doc33 seen. GOLD 'AMD revenue EPYC + semi-custom + Xilinx embedded'. PRED 'Data Center 64% (EPYC) + Gaming 21% (semi-custom) + Embedded segment' — rich match."),
    "doc2_qa0__after39": (0.0, "[ANS] doc2 seen. Same Y/N flip."),
    "doc16_qa0__after40": (0.25, "[ANS] doc16 seen. GOLD 9.5 AES inventory turnover. PRED '11.99' — wrong specific (26% off)."),
    "doc11_qa0__after40": (0.25, "[ANS] doc11 seen. PRED gives -99.6% with self-contradictory numbers."),
    "doc21_qa0__after41": (1.0, "[ANS] doc21 seen. Same $1,615.9M within tol match."),
    "doc36_qa0__after42": (1.0, "[ANS] doc36 seen. GOLD 'Data Center'. PRED 'Data Center segment'. Match."),
    "doc25_qa0__after43": (1.0, "[ANS] doc25 seen. Same Amcor packaging rich match."),
    "doc27_qa0__after43": (0.5, "[ANS] doc27 seen. Same generic restructuring partial."),
    "doc122_qa0__after43": (0.25, "[ACK] doc122 not yet seen. PRED '0' wrong specific."),
    "doc24_qa0__after43": (0.75, "[ANS] doc24 seen. Same Amcor acquisitions partial with wrong Czech year."),
    "doc35_qa0__after44": (1.0, "[ANS] doc35 seen. GOLD 'AMD operations most cashflow FY22'. PRED 'Operating activities most cash flow $3,565M'. Correct."),
    "doc17_qa0__after44": (0.25, "[ANS] doc17 seen. GOLD -0.02. PRED '-1.41' — wrong specific (70x off)."),
    "doc30_qa0__after44": (1.0, "[ANS] doc30 seen. GOLD 4.2%. PRED computes '≈4.18%' from $167M / $3,991M — within 0.5% tolerance. Correct calc."),
}

ENTRY_SUFFIXES: list[str] = [
    "doc12_qa0__after30", "doc98_qa0__after30", "doc47_qa0__after30", "doc97_qa0__after30",
    "doc52_qa0__after30", "doc0_qa0__after30", "doc60_qa0__after30", "doc5_qa0__after30",
    "doc42_qa0__after30", "doc90_qa0__after30",
    "doc124_qa0__after31", "doc91_qa0__after31", "doc21_qa0__after31", "doc63_qa0__after31",
    "doc120_qa0__after31", "doc67_qa0__after31", "doc139_qa0__after31", "doc18_qa0__after31",
    "doc135_qa0__after31", "doc141_qa0__after31",
    "doc117_qa0__after32", "doc18_qa0__after32", "doc7_qa0__after32", "doc115_qa0__after32",
    "doc47_qa0__after32", "doc106_qa0__after32", "doc87_qa0__after32", "doc56_qa0__after32",
    "doc77_qa0__after32", "doc112_qa0__after32",
    "doc135_qa0__after33", "doc144_qa0__after33", "doc18_qa0__after33", "doc34_qa0__after33",
    "doc72_qa0__after33", "doc15_qa0__after33", "doc90_qa0__after33", "doc89_qa0__after33",
    "doc64_qa0__after33", "doc125_qa0__after33",
    "doc130_qa0__after34", "doc26_qa0__after34", "doc68_qa0__after34", "doc40_qa0__after34",
    "doc129_qa0__after34", "doc144_qa0__after34", "doc25_qa0__after34", "doc34_qa0__after34",
    "doc131_qa0__after34", "doc29_qa0__after34",
    "doc136_qa0__after35", "doc93_qa0__after35", "doc146_qa0__after35", "doc149_qa0__after35",
    "doc42_qa0__after35", "doc85_qa0__after35", "doc98_qa0__after35", "doc92_qa0__after35",
    "doc78_qa0__after35", "doc100_qa0__after35",
    "doc88_qa0__after36", "doc69_qa0__after36", "doc120_qa0__after36", "doc112_qa0__after36",
    "doc133_qa0__after36", "doc136_qa0__after36", "doc145_qa0__after36", "doc131_qa0__after36",
    "doc31_qa0__after36", "doc3_qa0__after36",
    "doc52_qa0__after37", "doc70_qa0__after37", "doc11_qa0__after37", "doc10_qa0__after37",
    "doc90_qa0__after37", "doc54_qa0__after37", "doc50_qa0__after37", "doc107_qa0__after37",
    "doc129_qa0__after37", "doc108_qa0__after37",
    "doc90_qa0__after38", "doc138_qa0__after38", "doc43_qa0__after38", "doc71_qa0__after38",
    "doc1_qa0__after38", "doc27_qa0__after38", "doc140_qa0__after38", "doc24_qa0__after38",
    "doc135_qa0__after38", "doc88_qa0__after38",
    "doc115_qa0__after39", "doc92_qa0__after39", "doc146_qa0__after39", "doc76_qa0__after39",
    "doc80_qa0__after39", "doc8_qa0__after39", "doc33_qa0__after39", "doc95_qa0__after39",
    "doc46_qa0__after39", "doc2_qa0__after39",
    "doc16_qa0__after40", "doc93_qa0__after40", "doc128_qa0__after40", "doc110_qa0__after40",
    "doc59_qa0__after40", "doc54_qa0__after40", "doc135_qa0__after40", "doc11_qa0__after40",
    "doc53_qa0__after40", "doc57_qa0__after40",
    "doc85_qa0__after41", "doc88_qa0__after41", "doc53_qa0__after41", "doc61_qa0__after41",
    "doc46_qa0__after41", "doc124_qa0__after41", "doc84_qa0__after41", "doc134_qa0__after41",
    "doc21_qa0__after41", "doc87_qa0__after41",
    "doc106_qa0__after42", "doc124_qa0__after42", "doc98_qa0__after42", "doc56_qa0__after42",
    "doc36_qa0__after42", "doc51_qa0__after42", "doc111_qa0__after42", "doc60_qa0__after42",
    "doc148_qa0__after42", "doc50_qa0__after42",
    "doc25_qa0__after43", "doc114_qa0__after43", "doc133_qa0__after43", "doc141_qa0__after43",
    "doc55_qa0__after43", "doc85_qa0__after43", "doc27_qa0__after43", "doc94_qa0__after43",
    "doc122_qa0__after43", "doc24_qa0__after43",
    "doc76_qa0__after44", "doc35_qa0__after44", "doc17_qa0__after44", "doc30_qa0__after44",
    "doc66_qa0__after44", "doc101_qa0__after44", "doc95_qa0__after44", "doc67_qa0__after44",
    "doc53_qa0__after44", "doc141_qa0__after44",
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
