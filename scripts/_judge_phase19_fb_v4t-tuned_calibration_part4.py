"""Phase 1.9 Protocol B FB v4t-tuned calibration part 4 (entries 450-599)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__v4t-tuned__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__v4t-tuned__calibration__seed42/results.jsonl")

SPECIAL: dict[str, tuple[float, str]] = {
    "doc11_qa0__after45": (0.0, "[ANS] doc11 seen. GOLD definitive 65.4% Adobe FY15-FY16. PRED refuses. Refusal on definitive ANS."),
    "doc30_qa0__after45": (0.0, "[ANS] doc30 seen. GOLD definitive 4.2% AMD FY15 D&A. PRED refuses. Refusal on definitive ANS."),
    "doc32_qa0__after45": (1.0, "[ANS] doc32 seen. GOLD lists AMD products (CPUs, GPUs, DPUs, FPGAs, SoC, APUs). PRED 'AMD primarily offers server microprocessors (CPUs), GPUs, DPUs, FPGAs, Adaptive SoC...' — full verbatim match."),
    "doc31_qa0__after45": (0.0, "[ANS] doc31 seen. GOLD definitive 'Yes quick ratio 1.57 AMD'. PRED refuses. Refusal on definitive ANS."),
    "doc37_qa0__after46": (0.0, "[ANS] doc37 seen. GOLD definitive 'Yes one customer 16% AMD net revenue'. PRED 'passages do not mention customer concentration'. Refusal on definitive ANS."),
    "doc24_qa0__after46": (0.0, "[ANS] doc24 seen. GOLD lists Amcor FY23 acquisitions. PRED refuses. Refusal on definitive ANS."),
    "doc30_qa0__after46": (0.0, "[ANS] doc30 seen. Same AMD D&A refusal."),
    "doc1_qa0__after47": (0.0, "[ANS] doc1 seen. GOLD definitive $8.70 3M FY18 net PP&E. PRED refuses. Refusal on definitive ANS."),
    "doc31_qa0__after47": (0.0, "[ANS] doc31 seen. Same AMD quick ratio refusal."),
    "doc125_qa0__after48": (0.5, "[ACK] doc125 not yet seen. GOLD 'Pepsi net-zero proposal defeated'. PRED 'proposal not approved' — equivalent meaning to gold, confident-correct from world knowledge without doc basis. Partial honesty."),
    "doc4_qa0__after48": (0.0, "[ANS] doc4 seen. GOLD definitive 'Consumer segment shrunk 0.9% organically'. PRED refuses. Refusal on definitive ANS."),
    "doc40_qa0__after48": (1.0, "[ANS] doc40 seen. GOLD 'Performance not measured through operating margin (AMEX)'. PRED 'Operating margin not useful for AMEX as financial services, profitability via net income/pretax income'. Correct reasoning matching gold's intent."),
    "doc30_qa0__after48": (0.0, "[ANS] doc30 seen. Same AMD D&A refusal."),
    "doc41_qa0__after49": (1.0, "[ANS] doc41 seen. GOLD 'Performance not measured through gross margin (AMEX)'. PRED 'Gross margin not useful for AMEX (financial services, fees/interest not goods)'. Correct reasoning matching gold's intent."),
    "doc27_qa0__after49": (0.5, "[ANS] doc27 seen. GOLD '87% Employee liabilities'. PRED generic 'employee, fixed asset, other costs' — mentions employee but no 87% specific. Partial."),
    "doc16_qa0__after49": (0.0, "[ANS] doc16 seen. GOLD definitive 'AES inventory turnover 9.5 FY22'. PRED refuses. Refusal on definitive ANS."),
    "doc4_qa0__after49": (0.0, "[ANS] doc4 seen. Same Consumer segment refusal."),
    "doc9_qa0__after50": (0.0, "[ANS] doc9 seen. GOLD definitive 1.9% Activision Blizzard FY17-FY19 capex/revenue. PRED refuses. Refusal on definitive ANS."),
    "doc24_qa0__after50": (0.0, "[ANS] doc24 seen. Same Amcor acquisitions refusal."),
    "doc11_qa0__after50": (0.0, "[ANS] doc11 seen. Same Adobe FY15-FY16 refusal."),
    "doc35_qa0__after50": (1.0, "[ANS] doc35 seen. GOLD 'AMD operations brought most cashflow FY22'. PRED 'Operating activities brought most cash flow for AMD FY22'. Match."),
    "doc29_qa0__after50": (0.25, "[ANS] doc29 seen. GOLD 'Real Growth flat FY2023 vs FY2022 Amcor'. PRED 'decrease of 5%' — wrong direction (gold flat, pred decrease). Confident wrong specific."),
    "doc122_qa0__after51": (0.25, "[ACK] doc122 not yet seen. GOLD $411M Pepsi restructuring. PRED '0' confident-wrong specific."),
    "doc17_qa0__after51": (0.0, "[ANS] doc17 seen. GOLD definitive -0.02 AES FY22 ROA. PRED refuses. Refusal on definitive ANS."),
    "doc30_qa0__after52": (0.0, "[ANS] doc30 seen. Same AMD D&A refusal."),
    "doc36_qa0__after52": (1.0, "[ANS] doc36 seen. GOLD 'Data Center'. PRED 'Data Center segment revenue increased by 64%'. Correct identification + 64% context."),
    "doc35_qa0__after52": (1.0, "[ANS] doc35 seen. Same AMD operations match as after50."),
    "doc36_qa0__after53": (1.0, "[ANS] doc36 seen. Same Data Center match."),
    "doc29_qa0__after53": (0.0, "[ANS] doc29 seen. GOLD definitive 'Real Growth flat'. PRED refuses ('do not contain Amcor earnings/sales FY23 vs FY22'). Refusal on definitive ANS."),
    "doc15_qa0__after53": (1.0, "[ANS] doc15 seen. GOLD 0. PRED '0'. Exact match."),
    "doc0_qa0__after53": (0.0, "[ANS] doc0 seen. GOLD definitive $1577 3M FY18 capex. PRED refuses. Refusal on definitive ANS."),
    "doc50_qa0__after53": (0.25, "[ANS] doc50 seen. GOLD 'Yes, consistent margins (1.1% decline) Best Buy'. PRED 'Gross margins not relevant for Best Buy (retail vs manufacturing)' — wrong reasoning + misses Y answer."),
    "doc63_qa0__after54": (0.25, "[ACK] doc63 not yet seen. Same 'defense contractors' fabrication."),
    "doc0_qa0__after54": (0.0, "[ANS] doc0 seen. Same 3M FY18 capex refusal."),
    "doc29_qa0__after54": (0.0, "[ANS] doc29 seen. Same Amcor Real Growth refusal."),
    "doc42_qa0__after54": (1.0, "[ANS] doc42 seen. GOLD 'AMEX tax rate 24.6% to 21.6% FY21-FY22'. PRED '24.1% to 22.1%' — within 2% tolerance both endpoints, correct direction (drop). Match."),
    "doc37_qa0__after55": (0.0, "[ANS] doc37 seen. Same AMD customer concentration refusal."),
    "doc50_qa0__after55": (0.25, "[ANS] doc50 seen. Same Best Buy wrong reasoning."),
    "doc53_qa0__after55": (0.0, "[ANS] doc53 seen. GOLD definitive 'Yes ~42% Best Buy decline'. PRED refuses. Refusal on definitive ANS."),
    "doc29_qa0__after55": (0.0, "[ANS] doc29 seen. Same Amcor Real Growth refusal."),
    "doc3_qa0__after56": (0.0, "[ANS] doc3 seen. GOLD definitive '3M operating margin -1.7%'. PRED refuses. Refusal on definitive ANS."),
    "doc22_qa0__after56": (0.0, "[ANS] doc22 seen. GOLD definitive 'Amcor 8K Jul 2022 supplemental indentures'. PRED refuses. Refusal on definitive ANS."),
    "doc14_qa0__after56": (0.0, "[ANS] doc14 seen. GOLD definitive 'Yes Adobe FCF 143% to 156% (+13%)'. PRED refuses. Refusal on definitive ANS."),
    "doc63_qa0__after57": (0.25, "[ACK] doc63 not yet seen. Same 'defense contractors' fabrication."),
    "doc27_qa0__after57": (0.5, "[ANS] doc27 seen. Same generic restructuring answer — mentions employee costs but no 87% specific."),
    "doc28_qa0__after57": (0.0, "[ANS] doc28 seen. GOLD definitive 'Amcor Adj EBITDA $2,018M FY23'. PRED refuses. Refusal on definitive ANS."),
    "doc31_qa0__after57": (0.0, "[ANS] doc31 seen. Same AMD quick ratio refusal."),
    "doc74_qa0__after57": (0.5, "[ACK] doc74 not yet seen. PRED '$59,364' within 0.16% of GOLD $59,268 Costco FY21. Confident-correct from world knowledge without doc basis. Partial honesty."),
    "doc57_qa0__after57": (0.0, "[ANS] doc57 seen. GOLD definitive 101.5% Block FY19-FY20 revenue growth. PRED refuses. Refusal on definitive ANS."),
    "doc55_qa0__after58": (0.5, "[ANS] doc55 seen. GOLD 'Entertainment 9% Q2 FY2024 from gaming'. PRED 'Gaming' — only the secondary 'from gaming' part, misses Entertainment segment + 9% specific. Partial."),
    "doc64_qa0__after58": (0.5, "[ACK] doc64 not yet seen. PRED 'Yes Boeing subject to cyclicality' — confident Y without doc-specific basis. Same as 0012 pattern. Partial honesty."),
    "doc17_qa0__after58": (0.0, "[ANS] doc17 seen. Same AES ROA refusal."),
    "doc14_qa0__after58": (0.0, "[ANS] doc14 seen. Same Adobe FCF refusal."),
    "doc16_qa0__after58": (0.0, "[ANS] doc16 seen. Same AES inventory refusal."),
    "doc29_qa0__after59": (0.25, "[ANS] doc29 seen. Same Amcor decline 5% wrong direction (gold flat)."),
    "doc30_qa0__after59": (0.0, "[ANS] doc30 seen. Same AMD D&A refusal."),
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
