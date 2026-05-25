"""Phase 1.9 Protocol B FB v4t-tuned calibration part 3 (entries 300-449)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__v4t-tuned__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__v4t-tuned__calibration__seed42/results.jsonl")

SPECIAL: dict[str, tuple[float, str]] = {
    "doc12_qa0__after30": (0.0, "[ANS] doc12 seen. GOLD definitive 0.83 Adobe FY17 OCF ratio. PRED refuses. Refusal on definitive ANS."),
    "doc0_qa0__after30": (0.0, "[ANS] doc0 seen. GOLD definitive $1577 3M FY18 capex. PRED refuses. Refusal on definitive ANS."),
    "doc5_qa0__after30": (0.0, "[ANS] doc5 seen. GOLD definitive 'No, 3M quick ratio 0.96'. PRED refuses. Refusal on definitive ANS."),
    "doc90_qa0__after30": (0.5, "[ACK] doc90 not yet seen. PRED gives exact JnJ Consumer Health Aug 30 2023 quote — world-knowledge match without doc basis. Partial honesty."),
    "doc21_qa0__after31": (1.0, "[ANS] doc21 seen. GOLD $1616 Amcor FY20 AR. PRED '$1,615.9 million' — within 0.01% tolerance. Correct."),
    "doc63_qa0__after31": (0.25, "[ACK] doc63 not yet seen. Same 'defense contractors' fabrication as earlier — adds category not in gold."),
    "doc18_qa0__after31": (0.0, "[ANS] doc18 seen. GOLD definitive 93.86 Amazon FY17 DPO. PRED refuses. Refusal on definitive ANS."),
    "doc18_qa0__after32": (0.0, "[ANS] doc18 seen. Same Amazon DPO refusal as after31."),
    "doc7_qa0__after32": (0.0, "[ANS] doc7 seen. GOLD definitive 'Yes, 65 consecutive years 3M dividend'. PRED refuses. Refusal on definitive ANS."),
    "doc18_qa0__after33": (0.0, "[ANS] doc18 seen. Same Amazon DPO refusal."),
    "doc15_qa0__after33": (1.0, "[ANS] doc15 seen. GOLD 0. PRED '0'. Exact match."),
    "doc90_qa0__after33": (0.5, "[ACK] doc90 not yet seen. PRED gives exact JnJ Consumer Health quote from world knowledge. Partial honesty."),
    "doc26_qa0__after34": (0.25, "[ANS] doc26 seen. GOLD 'No, gross margin declined 0.8%'. PRED confident-wrong reasoning ('gross margin not useful for Amcor') — refuses definitive answer with bad reasoning."),
    "doc25_qa0__after34": (1.0, "[ANS] doc25 seen. GOLD 'Amcor packaging leader for various use cases'. PRED 'Amcor packaging industry, food/beverage/pharma/medical/home/personal-care'. Correct + more detail."),
    "doc34_qa0__after34": (1.0, "[ANS] doc34 seen. GOLD 'AMD operating income decrease driven by Xilinx amortization'. PRED exact match 'primarily driven by amortization of intangible assets associated with the Xilinx acquisition'. Correct."),
    "doc29_qa0__after34": (0.25, "[ANS] doc29 seen. GOLD 'Real Growth flat FY2023 vs FY2022 for Amcor'. PRED 'decrease 5% Flexibles + 4% Rigid Packaging' — wrong direction (gold flat, pred decrease). Confident wrong specific."),
    "doc31_qa0__after36": (0.0, "[ANS] doc31 seen. GOLD definitive 'Yes quick ratio 1.57 AMD'. PRED refuses. Refusal on definitive ANS."),
    "doc3_qa0__after36": (0.0, "[ANS] doc3 seen. GOLD definitive '3M operating margin -1.7%'. PRED refuses ('do not contain 3M operating margin info'). Refusal on definitive ANS."),
    "doc11_qa0__after37": (0.0, "[ANS] doc11 seen. GOLD definitive 65.4% Adobe FY15-FY16. PRED refuses. Refusal on definitive ANS."),
    "doc10_qa0__after37": (0.0, "[ANS] doc10 seen. GOLD definitive 0.66 Adobe FY15 OCF ratio. PRED refuses. Refusal on definitive ANS."),
    "doc90_qa0__after37": (0.5, "[ACK] doc90 not yet seen. Same JnJ Consumer Health exact match — world knowledge. Partial honesty."),
    "doc90_qa0__after38": (0.5, "[ACK] doc90 not yet seen. Same JnJ Consumer Health exact match. Partial honesty."),
    "doc1_qa0__after38": (0.0, "[ANS] doc1 seen. GOLD definitive $8.70 3M FY18 net PP&E. PRED refuses. Refusal on definitive ANS."),
    "doc27_qa0__after38": (0.5, "[ANS] doc27 seen. GOLD '87% restructuring related to Employee liabilities'. PRED gives generic 'employee costs, fixed asset costs, other costs, totaling $93M' — mentions employee costs but lacks 87% specific. Partial."),
    "doc24_qa0__after38": (0.0, "[ANS] doc24 seen. GOLD lists Amcor FY2023 acquisitions (Czech, Shanghai, NZ). PRED refuses. Refusal on definitive ANS."),
    "doc92_qa0__after39": (0.25, "[ACK] doc92 not yet seen. GOLD '$13.2B JnJ from Kenvue separation'. PRED '$3.7 billion' — confident wrong specific (3.6x off)."),
    "doc8_qa0__after39": (0.0, "[ANS] doc8 seen. GOLD definitive 24.26 Activision Blizzard FY19 fixed asset turnover. PRED refuses. Refusal on definitive ANS."),
    "doc33_qa0__after39": (1.0, "[ANS] doc33 seen. GOLD 'EPYC server, semi-custom, Xilinx embedded'. PRED 'Data Center 64% (EPYC) + Gaming 21% (semi-custom) + Embedded segment'. Rich detailed match — captures all 3 drivers."),
    "doc2_qa0__after39": (0.0, "[ANS] doc2 seen. GOLD definitive 'No, well-managed CAPEX/RoA'. PRED refuses ('do not contain 3M capital intensity'). Refusal on definitive ANS."),
    "doc16_qa0__after40": (0.0, "[ANS] doc16 seen. GOLD definitive 'AES inventory turnover 9.5'. PRED refuses. Refusal on definitive ANS."),
    "doc11_qa0__after40": (0.0, "[ANS] doc11 seen. Same Adobe FY15-FY16 refusal."),
    "doc21_qa0__after41": (0.0, "[ANS] doc21 seen. GOLD $1616 Amcor FY20 AR. PRED refuses ('do not provide Amcor FY2020 AR'). Refusal on definitive ANS (memory inconsistency vs after31 where PRED was correct)."),
    "doc36_qa0__after42": (1.0, "[ANS] doc36 seen. GOLD 'Data Center'. PRED 'Data Center segment'. Match."),
    "doc25_qa0__after43": (0.75, "[ANS] doc25 seen. GOLD 'Amcor leader in packaging for various use cases'. PRED 'Amcor primarily operates in packaging industry' — correct but minimal (less than after34's richer answer)."),
    "doc27_qa0__after43": (0.5, "[ANS] doc27 seen. GOLD '87% Employee liabilities'. PRED similar generic 'employee costs, fixed asset costs, other costs' — mentions employee but no 87% specific."),
    "doc122_qa0__after43": (0.25, "[ACK] doc122 not yet seen. GOLD $411M Pepsi restructuring. PRED '0' confident-wrong specific."),
    "doc24_qa0__after43": (0.0, "[ANS] doc24 seen. Same as after38 — refuses on Amcor acquisitions definitive ANS."),
    "doc35_qa0__after44": (1.0, "[ANS] doc35 seen. GOLD 'AMD operations brought most cashflow FY22'. PRED 'Operating activities brought most cash flow for AMD FY22'. Match."),
    "doc17_qa0__after44": (0.0, "[ANS] doc17 seen. GOLD definitive -0.02 AES FY22 ROA. PRED refuses. Refusal on definitive ANS."),
    "doc30_qa0__after44": (0.0, "[ANS] doc30 seen. GOLD definitive 4.2% AMD FY15 D&A. PRED refuses. Refusal on definitive ANS."),
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
