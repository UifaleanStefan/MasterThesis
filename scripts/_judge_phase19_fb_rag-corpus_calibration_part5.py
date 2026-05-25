"""Phase 1.9 extension: FB rag-corpus calibration part 5 (entries 600-749)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__rag-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__rag-corpus__calibration__seed42/results.jsonl")

SPECIAL: dict[str, tuple[float, str]] = {
    "doc13_qa0__after60": (0.0, "[ANS] doc13 seen. GOLD 'No Adobe OM declined 36.8% to 34.6%'. PRED 'OM 34.6% indicating improving margin' — Y/N flip + wrong reasoning (compares income not margin)."),
    "doc59_qa0__after60": (1.0, "[ANS] doc59 seen. GOLD $12645 Boeing FY18 PP&E. PRED '$12,645'. EXACT match."),
    "doc47_qa0__after60": (0.75, "[ANS] doc47 seen. GOLD 'No, AWW negative WC -$1561M'. PRED self-contradictory: opens 'Yes positive' but computes -$1,561M and concludes 'does not have positive WC'. Correct final conclusion + correct -$1561M specific."),
    "doc18_qa0__after60": (0.25, "[ANS] doc18 seen. GOLD 93.86. PRED '36.12' wrong specific."),
    "doc7_qa0__after60": (1.0, "[ANS] doc7 seen. PRED '65th consecutive year increases'. Correct."),
    "doc50_qa0__after61": (0.0, "[ANS] doc50 seen. GOLD 'Yes consistent margins'. PRED 'Gross margins fluctuated >2%, not historically consistent' — Y/N flip."),
    "doc20_qa0__after61": (1.0, "[ANS] doc20 seen. GOLD $11588. PRED '$11,588'. Exact."),
    "doc12_qa0__after61": (0.25, "[ANS] doc12 seen. GOLD 0.83. PRED '1.25' wrong specific."),
    "doc54_qa0__after61": (0.25, "[ANS] doc54 seen. GOLD 'Yes 982 to 969 (-1.32%) Best Buy stores'. PRED '907 down from 930' — wrong specific numbers."),
    "doc75_qa0__after61": (0.25, "[ACK] doc75 not yet seen. PRED '2.06' wrong specific (gold 17.98, 8.7x off)."),
    "doc47_qa0__after62": (0.75, "[ANS] doc47 seen. Same self-contradictory but correct -$1561M + final conclusion."),
    "doc40_qa0__after62": (1.0, "[ANS] doc40 seen. PRED correct AMEX OM reasoning (non-interest income, card fees)."),
    "doc72_qa0__after62": (0.25, "[ACK] doc72 not yet seen. GOLD '20% to 23%'. PRED '15.6% to 24.8%' wrong specifics."),
    "doc64_qa0__after63": (0.5, "[ACK] doc64 not yet seen. PRED 'Yes Boeing cyclicality, commercial airlines, economic cycles' confident-correct with detail but no doc basis. Partial honesty."),
    "doc61_qa0__after63": (1.0, "[ANS] doc61 seen. Same Lion Air/Ethiopian Airlines crashes match. Correct."),
    "doc33_qa0__after63": (1.0, "[ANS] doc33 seen. Same rich AMD revenue drivers match."),
    "doc39_qa0__after63": (1.0, "[ANS] doc39 seen. GOLD 'US, EMEA, APAC, LACC'. PRED 'United States, EMEA, APAC, LACC' with explanations. Exact + detail."),
    "doc25_qa0__after63": (0.75, "[ANS] doc25 seen. PRED 'Amcor primarily operates in packaging industry' minimal."),
    "doc60_qa0__after64": (0.75, "[ANS] doc60 seen. PRED 'Yes Commercial Airplanes $25,867M >20%' — one segment, missing Defence/Services."),
    "doc24_qa0__after64": (0.75, "[ANS] doc24 seen. Same partial Amcor acquisitions."),
    "doc36_qa0__after64": (0.25, "[ANS] doc36 seen. GOLD 'Data Center'. PRED 'Gaming segment' — wrong segment."),
    "doc27_qa0__after64": (0.5, "[ANS] doc27 seen. Same generic restructuring partial."),
    "doc41_qa0__after64": (1.0, "[ANS] doc41 seen. Same correct AMEX gross margin reasoning."),
    "doc26_qa0__after65": (0.75, "[ANS] doc26 seen. PRED 'gross profit decreased $2,820 to $2,725, declining gross margin profile' — direction correct + supporting calc, missing 0.8% specific."),
    "doc18_qa0__after65": (0.25, "[ANS] doc18 seen. Same Amazon DPO wrong specific."),
    "doc38_qa0__after65": (0.25, "[ANS] doc38 seen. GOLD 'There are none' (AMEX debt). PRED 'Common Shares par value $0.20' — gives common stock instead of acknowledging no debt securities."),
    "doc55_qa0__after66": (1.0, "[ANS] doc55 seen. EXACT 'Entertainment, 9.0% comparable sales growth driven primarily by gaming'. Correct."),
    "doc51_qa0__after66": (1.0, "[ANS] doc51 seen. GOLD 'Best Buy: Current Health + Two Peaks/Yardbird FY22'. PRED 'Current Health $389M + Yardbird $79M FY22, no FY23/FY21'. Match + amounts."),
    "doc62_qa0__after66": (0.25, "[ANS] doc62 seen. GOLD 'Yes Boeing improving gross margin'. PRED 'Gross margin not useful for Boeing (cyclical/competitive)' — wrong reasoning + missed Y."),
    "doc66_qa0__after66": (0.25, "[ANS] doc66 seen. GOLD '0.62% vs -14.76%'. PRED 'lower, $(31)M vs $743M benefit' — uses $ amounts when % asked + direction correct but wrong format."),
    "doc17_qa0__after66": (0.25, "[ANS] doc17 seen. PRED '-1.41' wrong specific."),
    "doc74_qa0__after67": (0.5, "[ACK] doc74 not yet seen. PRED '$59,364 million' within 0.16% — world knowledge match."),
    "doc25_qa0__after67": (0.75, "[ANS] doc25 seen. Same minimal Amcor packaging."),
    "doc2_qa0__after67": (0.0, "[ANS] doc2 seen. Same Y/N flip + $25,998M PP&E."),
    "doc3_qa0__after67": (0.75, "[ANS] doc3 seen. Same partial OM (litigation/impairment/restructuring without PFAS/Russia specifics)."),
    "doc35_qa0__after67": (1.0, "[ANS] doc35 seen. PRED 'Operating activities $3,565M AMD FY22'. Correct + amount."),
    "doc39_qa0__after67": (1.0, "[ANS] doc39 seen. 'United States, EMEA, APAC, LACC' exact."),
    "doc66_qa0__after68": (0.25, "[ANS] doc66 seen. Same Boeing tax partial."),
    "doc25_qa0__after68": (1.0, "[ANS] doc25 seen. PRED 'Amcor packaging industry, food/beverage/pharma/medical/home/personal-care' — rich match."),
    "doc24_qa0__after68": (0.75, "[ANS] doc24 seen. Same partial Amcor."),
    "doc32_qa0__after68": (1.0, "[ANS] doc32 seen. Same AMD products verbatim match."),
    "doc15_qa0__after68": (1.0, "[ANS] doc15 seen. '0'. Match."),
    "doc82_qa0__after68": (0.25, "[ACK] doc82 not yet seen. GOLD 0.68 GenMills WC ratio. PRED '1.14' wrong specific."),
    "doc30_qa0__after69": (1.0, "[ANS] doc30 seen. Same 4.18% within tol."),
    "doc32_qa0__after69": (1.0, "[ANS] doc32 seen. Same AMD products match."),
    "doc65_qa0__after69": (1.0, "[ANS] doc65 seen. GOLD 'Boeing 737/777X/787 production rates 2023'. PRED '787 to 5/month + 737 + 777X resume 2023'. Covers all 3 aircraft."),
    "doc16_qa0__after69": (0.25, "[ANS] doc16 seen. Same 11.99 wrong specific."),
    "doc26_qa0__after70": (0.75, "[ANS] doc26 seen. Same partial gross margin calc."),
    "doc66_qa0__after70": (0.25, "[ANS] doc66 seen. Same Boeing tax partial."),
    "doc71_qa0__after70": (0.25, "[ACK] doc71 not yet seen. GOLD 10.3%. PRED '4.5%' wrong specific."),
    "doc65_qa0__after70": (1.0, "[ANS] doc65 seen. Same 737/787/777X correct."),
    "doc10_qa0__after71": (0.25, "[ANS] doc10 seen. GOLD 0.66. PRED '1.69' wrong specific."),
    "doc46_qa0__after71": (1.0, "[ANS] doc46 seen. GOLD $1832. PRED '1,829' within 0.16% tolerance."),
    "doc59_qa0__after71": (1.0, "[ANS] doc59 seen. PRED '$12,645' exact."),
    "doc55_qa0__after71": (1.0, "[ANS] doc55 seen. Same Entertainment 9% exact."),
    "doc42_qa0__after71": (1.0, "[ANS] doc42 seen. '24.6% to 21.6%' exact."),
    "doc58_qa0__after71": (1.0, "[ANS] doc58 seen. GOLD $382 Block CFO. PRED '$381.6 million' within 0.1% tolerance."),
    "doc14_qa0__after71": (0.0, "[ANS] doc14 seen. Same Adobe FCF refusal."),
    "doc3_qa0__after72": (0.75, "[ANS] doc3 seen. Same partial OM."),
    "doc12_qa0__after72": (0.25, "[ANS] doc12 seen. PRED '1.25' wrong specific."),
    "doc71_qa0__after72": (0.25, "[ANS] doc71 seen. GOLD 10.3%. PRED '15.5%' wrong specific."),
    "doc52_qa0__after72": (1.0, "[ANS] doc52 seen. GOLD 'Best Buy operations $1.8bn FY23'. PRED 'Operating activities $1,824M Best Buy FY2023'. Exact + amount."),
    "doc64_qa0__after72": (0.75, "[ANS] doc64 seen. PRED 'Yes cyclicality' Y correct but lacks airline industry context."),
    "doc26_qa0__after72": (0.75, "[ANS] doc26 seen. Same partial."),
    "doc117_qa0__after72": (0.5, "[ACK] doc117 not yet seen. PRED 'Operating activities brought in most cash flow for Nike FY2023' — confident-correct from world knowledge."),
    "doc14_qa0__after73": (0.0, "[ANS] doc14 seen. Same Adobe FCF refusal."),
    "doc12_qa0__after73": (0.25, "[ANS] doc12 seen. PRED '1.25' wrong specific."),
    "doc69_qa0__after73": (1.0, "[ANS] doc69 seen. GOLD 0.8. PRED '0.80' exact."),
    "doc4_qa0__after73": (0.75, "[ANS] doc4 seen. 'Consumer segment' partial."),
    "doc26_qa0__after73": (0.75, "[ANS] doc26 seen. Same partial with FY21 context."),
    "doc117_qa0__after74": (0.5, "[ACK] doc117 not yet seen. Same confident-correct."),
    "doc69_qa0__after74": (1.0, "[ANS] doc69 seen. '0.80' exact."),
    "doc90_qa0__after74": (1.0, "[ANS] doc90 seen. GOLD 'Consumer Health discontinued Aug 30 2023'. PRED matches with JnJ prefix. Correct."),
    "doc83_qa0__after74": (0.25, "[ACK] doc83 not yet seen. GOLD $3215. PRED '$2,343M' wrong specific (27% off)."),
    "doc50_qa0__after74": (0.0, "[ANS] doc50 seen. Same Y/N flip."),
    "doc22_qa0__after74": (1.0, "[ANS] doc22 seen. PRED matches gold (Amcor Flexibles NA for Amcor Finance USA supplemental indentures)."),
    "doc6_qa0__after74": (0.75, "[ANS] doc6 seen. GOLD lists 3 3M notes with MMM26/MMM30/MMM31 tickers. PRED lists rates+years (1.500%/2026, 1.750%/2030, 1.500%/2031) without tickers — content match, partial detail."),
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
