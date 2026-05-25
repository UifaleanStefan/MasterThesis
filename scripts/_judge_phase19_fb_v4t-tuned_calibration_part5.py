"""Phase 1.9 Protocol B FB v4t-tuned calibration part 5 (entries 600-749)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__v4t-tuned__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__v4t-tuned__calibration__seed42/results.jsonl")

SPECIAL: dict[str, tuple[float, str]] = {
    "doc13_qa0__after60": (0.0, "[ANS] doc13 seen. GOLD definitive 'No, Adobe OM declined 36.8% to 34.6% (-2.2%)'. PRED refuses. Refusal on definitive ANS."),
    "doc59_qa0__after60": (0.0, "[ANS] doc59 seen. GOLD definitive $12645 Boeing FY18 PP&E. PRED refuses. Refusal on definitive ANS."),
    "doc47_qa0__after60": (0.0, "[ANS] doc47 seen. GOLD 'No, AWW negative WC -$1561M FY2022'. PRED 'Yes, positive WC ($1,250M current assets)'. Y/N flip + fabricated."),
    "doc18_qa0__after60": (0.0, "[ANS] doc18 seen. GOLD definitive 93.86 Amazon DPO. PRED refuses. Refusal on definitive ANS."),
    "doc7_qa0__after60": (1.0, "[ANS] doc7 seen. GOLD '65 consecutive years 3M dividend'. PRED '65th consecutive year increases'. Match."),
    "doc50_qa0__after61": (0.25, "[ANS] doc50 seen. Same Best Buy wrong reasoning ('gross margins not relevant for retail') as earlier."),
    "doc20_qa0__after61": (0.0, "[ANS] doc20 seen. GOLD definitive $11588 Amazon FY19 net income. PRED refuses. Refusal on definitive ANS."),
    "doc12_qa0__after61": (0.0, "[ANS] doc12 seen. Same Adobe OCF ratio refusal."),
    "doc54_qa0__after61": (1.0, "[ANS] doc54 seen. GOLD 'Yes, 1.32% decline 982 to 969 Best Buy stores'. PRED 'Yes, 966 stores Q2 FY24 down from 977 Q2 FY23' — Y direction correct + numbers within 1% tolerance."),
    "doc47_qa0__after62": (0.75, "[ANS] doc47 seen. GOLD 'No, AWW negative WC'. PRED concludes 'does not have positive WC' — matches Y/N direction even with incomplete data. Partial — correct conclusion, lacks -$1561M specific."),
    "doc40_qa0__after62": (1.0, "[ANS] doc40 seen. GOLD 'Performance not measured through OM (AMEX)'. PRED 'Operating margin not useful for AMEX (financial services, net card fees/retention)'. Correct reasoning."),
    "doc61_qa0__after63": (1.0, "[ANS] doc61 seen. GOLD 'Yes, multiple lawsuits Lion Air + Ethiopian Airlines'. PRED 'Yes, multiple legal actions for Lion Air Flight 610 and Ethiopian Airlines Flight 302'. Exact match with detail."),
    "doc33_qa0__after63": (1.0, "[ANS] doc33 seen. Same rich AMD revenue drivers match (Data Center 64% EPYC, Gaming 21% semi-custom, Embedded)."),
    "doc39_qa0__after63": (0.0, "[ANS] doc39 seen. GOLD definitive 'US, EMEA, APAC, LACC' AMEX geographies. PRED refuses. Refusal on definitive ANS."),
    "doc25_qa0__after63": (0.0, "[ANS] doc25 seen. GOLD definitive 'Amcor leader in packaging'. PRED 'do not contain info about Amcor'. Refusal on definitive ANS."),
    "doc60_qa0__after64": (0.75, "[ANS] doc60 seen. GOLD 'Yes, Commercial Airplanes 39%, Defence 35%, Services'. PRED 'Yes, Commercial Airplanes >20% ($25,867M)' — Y correct + one segment but misses Defence/Services."),
    "doc24_qa0__after64": (0.0, "[ANS] doc24 seen. Same Amcor acquisitions refusal."),
    "doc36_qa0__after64": (1.0, "[ANS] doc36 seen. Same Data Center 64% match."),
    "doc27_qa0__after64": (0.0, "[ANS] doc27 seen. GOLD definitive '87% Employee liabilities'. PRED refuses. Refusal on definitive ANS."),
    "doc41_qa0__after64": (1.0, "[ANS] doc41 seen. Same correct AMEX gross margin reasoning."),
    "doc26_qa0__after65": (0.0, "[ANS] doc26 seen. GOLD definitive 'No, gross margin decline 0.8%'. PRED refuses. Refusal on definitive ANS."),
    "doc18_qa0__after65": (0.0, "[ANS] doc18 seen. Same Amazon DPO refusal."),
    "doc38_qa0__after65": (0.75, "[ANS] doc38 seen. GOLD 'There are none' (AMEX debt securities). PRED 'do not contain info on AMEX debt securities registered'. Both reach same conclusion via different framing — partial honesty."),
    "doc55_qa0__after66": (0.5, "[ANS] doc55 seen. GOLD 'Entertainment 9% Q2 FY2024 from gaming'. PRED 'Gaming' — partial (gold's secondary detail only, misses Entertainment segment + 9%)."),
    "doc51_qa0__after66": (1.0, "[ANS] doc51 seen. GOLD 'Best Buy Current Health + Two Peaks/Yardbird FY2022'. PRED 'FY2022 Current Health $389M + Yardbird $79M'. Match + amounts."),
    "doc62_qa0__after66": (0.25, "[ANS] doc62 seen. GOLD 'Yes, Boeing improving gross margin (4.8%→5.3%)'. PRED 'Gross margin not useful for Boeing (cyclical/competitive)' — wrong reasoning + misses Y answer."),
    "doc66_qa0__after66": (0.0, "[ANS] doc66 seen. GOLD definitive 'Effective tax rate 0.62% vs -14.76%'. PRED refuses. Refusal on definitive ANS."),
    "doc17_qa0__after66": (0.0, "[ANS] doc17 seen. Same AES ROA refusal."),
    "doc74_qa0__after67": (0.5, "[ACK] doc74 not yet seen. PRED '$59,364' within 0.16% of GOLD $59,268. Confident-correct from world knowledge. Partial honesty."),
    "doc25_qa0__after67": (0.75, "[ANS] doc25 seen. GOLD 'Amcor leader packaging various use'. PRED 'Amcor primarily operates in packaging industry' — correct but minimal."),
    "doc2_qa0__after67": (0.0, "[ANS] doc2 seen. GOLD definitive 'No, well-managed CAPEX/RoA'. PRED refuses. Refusal on definitive ANS."),
    "doc3_qa0__after67": (0.0, "[ANS] doc3 seen. Same 3M OM refusal."),
    "doc35_qa0__after67": (0.25, "[ANS] doc35 seen. GOLD 'operations'. PRED 'investing activities brought in most cash flow for AMD FY22' — wrong specific (investing vs operating)."),
    "doc39_qa0__after67": (0.0, "[ANS] doc39 seen. Same AMEX geographies refusal."),
    "doc66_qa0__after68": (0.0, "[ANS] doc66 seen. Same Boeing tax refusal."),
    "doc25_qa0__after68": (0.75, "[ANS] doc25 seen. Same Amcor minimal answer."),
    "doc24_qa0__after68": (0.0, "[ANS] doc24 seen. Same Amcor acquisitions refusal."),
    "doc32_qa0__after68": (1.0, "[ANS] doc32 seen. Same verbatim AMD products match."),
    "doc15_qa0__after68": (1.0, "[ANS] doc15 seen. GOLD 0. PRED '0'. Match."),
    "doc30_qa0__after69": (0.0, "[ANS] doc30 seen. Same AMD D&A refusal."),
    "doc108_qa0__after69": (0.25, "[ACK] doc108 not yet seen. GOLD 'MGM China worst (-44%)'. PRED 'Las Vegas Strip' — confident wrong specific (gold says MGM China)."),
    "doc32_qa0__after69": (1.0, "[ANS] doc32 seen. Same AMD products match."),
    "doc87_qa0__after69": (0.75, "[ACK] doc87 not yet seen. GOLD 'JnJ inventory turnover 2.7'. PRED hedged 'not provided... conventional inventory may not be meaningful due to nature of business' — hedged + admits uncertainty."),
    "doc65_qa0__after69": (0.75, "[ANS] doc65 seen. GOLD 'Boeing 737/777X/787 production rate increase 2023'. PRED '787 to 5/month + 737 based on demand' — partial (covers 737/787, misses 777X)."),
    "doc16_qa0__after69": (0.0, "[ANS] doc16 seen. Same AES inventory refusal."),
    "doc26_qa0__after70": (0.0, "[ANS] doc26 seen. Same Amcor gross margin refusal."),
    "doc66_qa0__after70": (0.0, "[ANS] doc66 seen. Same Boeing tax refusal."),
    "doc65_qa0__after70": (1.0, "[ANS] doc65 seen. Now includes 777X (paused, expected to resume 2023). Richer match covering all 3 — Correct."),
    "doc10_qa0__after71": (0.0, "[ANS] doc10 seen. GOLD definitive 0.66 Adobe FY15 OCF ratio. PRED refuses. Refusal on definitive ANS."),
    "doc46_qa0__after71": (0.0, "[ANS] doc46 seen. GOLD definitive $1832 AWW FY21 EBITDA. PRED refuses. Refusal on definitive ANS."),
    "doc59_qa0__after71": (0.0, "[ANS] doc59 seen. Same Boeing PP&E refusal."),
    "doc55_qa0__after71": (0.5, "[ANS] doc55 seen. Same 'Gaming' partial."),
    "doc42_qa0__after71": (0.0, "[ANS] doc42 seen. GOLD definitive 'AMEX tax rate 24.6%→21.6%'. PRED refuses. Refusal on definitive ANS."),
    "doc58_qa0__after71": (0.0, "[ANS] doc58 seen. GOLD definitive $382 Block FY20 CFO. PRED refuses. Refusal on definitive ANS."),
    "doc14_qa0__after71": (0.0, "[ANS] doc14 seen. Same Adobe FCF refusal."),
    "doc3_qa0__after72": (0.0, "[ANS] doc3 seen. Same 3M OM refusal."),
    "doc12_qa0__after72": (0.0, "[ANS] doc12 seen. Same Adobe OCF refusal."),
    "doc71_qa0__after72": (0.25, "[ANS] doc71 seen. GOLD 10.3% Corning op margin. PRED '14.0%' — wrong specific (35% off)."),
    "doc52_qa0__after72": (0.0, "[ANS] doc52 seen. GOLD definitive 'Best Buy operations $1.8bn FY2023'. PRED refuses. Refusal on definitive ANS."),
    "doc64_qa0__after72": (0.75, "[ANS] doc64 seen. GOLD 'Yes, Boeing cyclicality due to airline industry'. PRED 'Yes, cyclicality' — Y correct but lacks airline industry context."),
    "doc26_qa0__after72": (0.0, "[ANS] doc26 seen. Same Amcor gross margin refusal."),
    "doc14_qa0__after73": (0.0, "[ANS] doc14 seen. Same Adobe FCF refusal."),
    "doc12_qa0__after73": (0.0, "[ANS] doc12 seen. Same Adobe OCF refusal."),
    "doc69_qa0__after73": (0.25, "[ANS] doc69 seen. GOLD 0.8 Coca-Cola FY22 dividend payout. PRED '0.20' — wrong specific (75% off)."),
    "doc4_qa0__after73": (0.25, "[ANS] doc4 seen. GOLD 'Consumer segment shrunk 0.9%'. PRED 'Health Care segment dragged 3M growth 2022' — wrong segment."),
    "doc26_qa0__after73": (0.0, "[ANS] doc26 seen. Same Amcor gross margin refusal."),
    "doc69_qa0__after74": (0.25, "[ANS] doc69 seen. Same Coca-Cola wrong specific (0.19 vs 0.8)."),
    "doc90_qa0__after74": (0.5, "[ACK] doc90 not yet seen. PRED exact JnJ Consumer Health Aug 30 2023 quote. World knowledge match. Partial honesty."),
    "doc50_qa0__after74": (0.0, "[ANS] doc50 seen. GOLD definitive 'Yes, consistent margins, 1.1% decline Best Buy'. PRED refuses. Refusal on definitive ANS."),
    "doc22_qa0__after74": (0.0, "[ANS] doc22 seen. Same Amcor 8K refusal."),
    "doc6_qa0__after74": (0.0, "[ANS] doc6 seen. GOLD definitive '3M debt securities MMM26/MMM30/MMM31'. PRED refuses. Refusal on definitive ANS."),
}

ENTRY_SUFFIXES: list[str] = [
    "doc13_qa0__after60", "doc59_qa0__after60", "doc47_qa0__after60", "doc67_qa0__after60",
    "doc130_qa0__after60", "doc18_qa0__after60", "doc133_qa0__after60", "doc7_qa0__after60",
    "doc137_qa0__after60", "doc134_qa0__after60",
    "doc50_qa0__after61", "doc20_qa0__after61", "doc96_qa0__after61", "doc69_qa0__after61",
    "doc12_qa0__after61", "doc54_qa0__after61", "doc126_qa0__after61", "doc106_qa0__after61",
    "doc142_qa0__after61", "doc75_qa0__after61",
    "doc47_qa0__after62", "doc40_qa0__after62", "doc101_qa0__after62", "doc140_qa0__after62",
    "doc87_qa0__after62", "doc121_qa0__after62", "doc83_qa0__after62", "doc72_qa0__after62",
    "doc147_qa0__after62", "doc126_qa0__after62",
    "doc126_qa0__after63", "doc64_qa0__after63", "doc115_qa0__after63", "doc77_qa0__after63",
    "doc143_qa0__after63", "doc123_qa0__after63", "doc61_qa0__after63", "doc33_qa0__after63",
    "doc39_qa0__after63", "doc25_qa0__after63",
    "doc132_qa0__after64", "doc60_qa0__after64", "doc134_qa0__after64", "doc107_qa0__after64",
    "doc68_qa0__after64", "doc24_qa0__after64", "doc36_qa0__after64", "doc117_qa0__after64",
    "doc27_qa0__after64", "doc41_qa0__after64",
    "doc105_qa0__after65", "doc146_qa0__after65", "doc26_qa0__after65", "doc18_qa0__after65",
    "doc89_qa0__after65", "doc114_qa0__after65", "doc102_qa0__after65", "doc38_qa0__after65",
    "doc94_qa0__after65", "doc145_qa0__after65",
    "doc55_qa0__after66", "doc51_qa0__after66", "doc62_qa0__after66", "doc139_qa0__after66",
    "doc142_qa0__after66", "doc149_qa0__after66", "doc116_qa0__after66", "doc103_qa0__after66",
    "doc66_qa0__after66", "doc17_qa0__after66",
    "doc74_qa0__after67", "doc76_qa0__after67", "doc25_qa0__after67", "doc71_qa0__after67",
    "doc113_qa0__after67", "doc2_qa0__after67", "doc3_qa0__after67", "doc141_qa0__after67",
    "doc35_qa0__after67", "doc39_qa0__after67",
    "doc66_qa0__after68", "doc25_qa0__after68", "doc99_qa0__after68", "doc85_qa0__after68",
    "doc24_qa0__after68", "doc126_qa0__after68", "doc32_qa0__after68", "doc15_qa0__after68",
    "doc82_qa0__after68", "doc121_qa0__after68",
    "doc105_qa0__after69", "doc85_qa0__after69", "doc139_qa0__after69", "doc30_qa0__after69",
    "doc108_qa0__after69", "doc32_qa0__after69", "doc87_qa0__after69", "doc93_qa0__after69",
    "doc65_qa0__after69", "doc16_qa0__after69",
    "doc26_qa0__after70", "doc66_qa0__after70", "doc93_qa0__after70", "doc138_qa0__after70",
    "doc129_qa0__after70", "doc71_qa0__after70", "doc135_qa0__after70", "doc65_qa0__after70",
    "doc104_qa0__after70", "doc91_qa0__after70",
    "doc10_qa0__after71", "doc46_qa0__after71", "doc59_qa0__after71", "doc95_qa0__after71",
    "doc55_qa0__after71", "doc139_qa0__after71", "doc42_qa0__after71", "doc94_qa0__after71",
    "doc58_qa0__after71", "doc14_qa0__after71",
    "doc3_qa0__after72", "doc110_qa0__after72", "doc134_qa0__after72", "doc12_qa0__after72",
    "doc71_qa0__after72", "doc52_qa0__after72", "doc64_qa0__after72", "doc26_qa0__after72",
    "doc117_qa0__after72", "doc119_qa0__after72",
    "doc14_qa0__after73", "doc106_qa0__after73", "doc12_qa0__after73", "doc114_qa0__after73",
    "doc92_qa0__after73", "doc140_qa0__after73", "doc115_qa0__after73", "doc69_qa0__after73",
    "doc4_qa0__after73", "doc26_qa0__after73",
    "doc119_qa0__after74", "doc117_qa0__after74", "doc69_qa0__after74", "doc123_qa0__after74",
    "doc90_qa0__after74", "doc83_qa0__after74", "doc126_qa0__after74", "doc50_qa0__after74",
    "doc22_qa0__after74", "doc6_qa0__after74",
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
