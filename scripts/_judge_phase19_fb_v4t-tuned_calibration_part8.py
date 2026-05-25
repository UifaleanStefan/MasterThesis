"""Phase 1.9 Protocol B FB v4t-tuned calibration part 8 (entries 1050-1199)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__v4t-tuned__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__v4t-tuned__calibration__seed42/results.jsonl")

SPECIAL: dict[str, tuple[float, str]] = {
    "doc22_qa0__after105": (0.0, "[ANS] doc22 seen. Same Amcor 8K refusal."),
    "doc25_qa0__after105": (0.0, "[ANS] doc25 seen. Same Amcor refusal."),
    "doc62_qa0__after105": (0.0, "[ANS] doc62 seen. Same Boeing gross margin refusal."),
    "doc98_qa0__after105": (1.0, "[ANS] doc98 seen. Same JPM VaR decreased $7M correct."),
    "doc1_qa0__after105": (0.0, "[ANS] doc1 seen. Same 3M PP&E refusal."),
    "doc76_qa0__after105": (0.0, "[ANS] doc76 seen. GOLD definitive 'Yes CVS capital-intensive'. PRED refuses. Refusal on definitive ANS."),
    "doc70_qa0__after106": (0.0, "[ANS] doc70 seen. Same Corning DPO refusal."),
    "doc28_qa0__after106": (0.0, "[ANS] doc28 seen. Same Amcor Adj EBITDA refusal."),
    "doc30_qa0__after106": (0.0, "[ANS] doc30 seen. Same AMD D&A refusal."),
    "doc85_qa0__after106": (0.0, "[ANS] doc85 seen. Same JnJ high growth refusal."),
    "doc87_qa0__after106": (0.75, "[ANS] doc87 seen. GOLD 'JnJ 2.7 inventory turnover'. PRED hedged 'turnover not provided + conventional inventory may not be meaningful due to nature of business'. Partial honesty (hedged + admits uncertainty)."),
    "doc49_qa0__after106": (0.0, "[ANS] doc49 seen. Same Best Buy inventory refusal."),
    "doc9_qa0__after107": (0.0, "[ANS] doc9 seen. Same Activision refusal."),
    "doc13_qa0__after107": (0.0, "[ANS] doc13 seen. Same Adobe OM refusal."),
    "doc122_qa0__after107": (0.25, "[ACK] doc122 not yet seen. PRED '0' confident wrong specific."),
    "doc103_qa0__after107": (0.0, "[ANS] doc103 seen. Same MGM AP refusal."),
    "doc87_qa0__after107": (0.75, "[ANS] doc87 seen. Same hedged response."),
    "doc75_qa0__after108": (0.0, "[ANS] doc75 seen. Same CVS turnover refusal."),
    "doc90_qa0__after108": (1.0, "[ANS] doc90 seen. GOLD 'Consumer Health discontinued Aug 30 2023'. PRED matches with JnJ prefix. Correct."),
    "doc98_qa0__after108": (1.0, "[ANS] doc98 seen. Same JPM VaR correct."),
    "doc42_qa0__after108": (0.0, "[ANS] doc42 seen. Same AMEX tax refusal."),
    "doc43_qa0__after108": (0.0, "[ANS] doc43 seen. Same AMEX liabilities refusal."),
    "doc51_qa0__after108": (0.0, "[ANS] doc51 seen. Same Best Buy acquisitions refusal."),
    "doc68_qa0__after108": (0.0, "[ANS] doc68 seen. Same Coca-Cola COGS refusal."),
    "doc45_qa0__after108": (0.0, "[ANS] doc45 seen. Same AWW dividends refusal."),
    "doc108_qa0__after108": (1.0, "[ANS] doc108 seen. GOLD 'MGM China worst (-44%)'. PRED 'MGM China worst $674M -44%'. Correct + specific."),
    "doc26_qa0__after109": (0.0, "[ANS] doc26 seen. Same Amcor margin refusal."),
    "doc7_qa0__after109": (0.0, "[ANS] doc7 seen. Same 3M dividend refusal."),
    "doc14_qa0__after109": (0.0, "[ANS] doc14 seen. Same Adobe FCF refusal."),
    "doc44_qa0__after109": (1.0, "[ANS] doc44 seen. GOLD 'Yes' Card Member retention. PRED 'Yes, Card Member retention remained high'. Correct."),
    "doc102_qa0__after109": (0.25, "[ANS] doc102 seen. GOLD 0.4% Lockheed CAGR. PRED '7.5%' — wrong specific (18x off)."),
    "doc65_qa0__after109": (0.0, "[ANS] doc65 seen. Same Boeing production refusal."),
    "doc18_qa0__after109": (0.0, "[ANS] doc18 seen. Same Amazon DPO refusal."),
    "doc7_qa0__after110": (0.0, "[ANS] doc7 seen. Same 3M dividend refusal."),
    "doc72_qa0__after110": (0.0, "[ANS] doc72 seen. Same Corning tax refusal."),
    "doc35_qa0__after110": (0.25, "[ANS] doc35 seen. GOLD 'operations'. PRED 'investing activities brought most cash flow for AMD FY22' — wrong specific."),
    "doc99_qa0__after110": (0.5, "[ANS] doc99 seen. GOLD 6.25 Kraft Heinz inventory turnover. PRED 'COGS $16,830M but average inventory not provided, cannot calculate' — partial reasoning, no answer."),
    "doc33_qa0__after110": (0.0, "[ANS] doc33 seen. Same AMD revenue refusal."),
    "doc90_qa0__after110": (1.0, "[ANS] doc90 seen. Same JnJ Consumer Health match."),
    "doc97_qa0__after110": (0.75, "[ANS] doc97 seen. GOLD 'Corporate & Investment Bank, $3725M'. PRED 'Corporate & Investment Bank' — correct segment, no amount."),
    "doc122_qa0__after110": (0.25, "[ACK] doc122 not yet seen. Same '0' wrong specific."),
    "doc26_qa0__after111": (0.0, "[ANS] doc26 seen. Same Amcor refusal."),
    "doc30_qa0__after111": (0.0, "[ANS] doc30 seen. Same AMD D&A refusal."),
    "doc82_qa0__after111": (0.0, "[ANS] doc82 seen. Same GenMills WC refusal."),
    "doc36_qa0__after111": (0.0, "[ANS] doc36 seen. Same Data Center refusal."),
    "doc72_qa0__after111": (0.0, "[ANS] doc72 seen. Same Corning tax refusal."),
    "doc37_qa0__after111": (0.0, "[ANS] doc37 seen. Same AMD customer refusal."),
    "doc101_qa0__after111": (0.0, "[ANS] doc101 seen. Same Lockheed NWC refusal."),
    "doc35_qa0__after112": (0.0, "[ANS] doc35 seen. Same AMD refusal."),
    "doc52_qa0__after112": (0.0, "[ANS] doc52 seen. Same Best Buy refusal."),
    "doc23_qa0__after112": (0.0, "[ANS] doc23 seen. Same Amcor quick ratio refusal."),
    "doc21_qa0__after112": (0.0, "[ANS] doc21 seen. Same Amcor AR refusal."),
    "doc59_qa0__after112": (0.0, "[ANS] doc59 seen. Same Boeing PP&E refusal."),
    "doc92_qa0__after112": (1.0, "[ANS] doc92 seen. GOLD '$13.2B JnJ Kenvue'. PRED '$13.2 billion'. Exact match."),
    "doc89_qa0__after112": (0.0, "[ANS] doc89 seen. Same JnJ US/intl sales refusal."),
    "doc122_qa0__after112": (0.25, "[ACK] doc122 not yet seen. Same '0' wrong specific."),
    "doc139_qa0__after113": (0.25, "[ACK] doc139 not yet seen. GOLD '47 new stores'. PRED 'strategic decision to increase inventory levels to support sales growth' — confident-fabricated reasoning, misses 47 stores."),
    "doc104_qa0__after113": (0.25, "[ANS] doc104 seen. GOLD 7.9% MGM capex. PRED '10.5%' — wrong specific (33% off)."),
    "doc82_qa0__after113": (0.0, "[ANS] doc82 seen. Same GenMills WC refusal."),
    "doc60_qa0__after113": (0.0, "[ANS] doc60 seen. Same Boeing segments refusal."),
    "doc89_qa0__after113": (0.0, "[ANS] doc89 seen. Same refusal."),
    "doc47_qa0__after113": (0.0, "[ANS] doc47 seen. GOLD definitive 'No AWW negative WC -$1561M'. PRED refuses (no longer Y/N flip — context lost). Refusal on definitive ANS."),
    "doc56_qa0__after113": (0.0, "[ANS] doc56 seen. Same Block WC refusal."),
    "doc98_qa0__after113": (1.0, "[ANS] doc98 seen. Same JPM VaR correct."),
    "doc24_qa0__after114": (0.0, "[ANS] doc24 seen. Same Amcor acquisitions refusal."),
    "doc113_qa0__after114": (0.0, "[ANS] doc113 seen. GOLD definitive $5466 Netflix current liab. PRED refuses. Refusal on definitive ANS."),
    "doc27_qa0__after114": (0.0, "[ANS] doc27 seen. Same Amcor restructuring refusal."),
    "doc97_qa0__after114": (0.0, "[ANS] doc97 seen. Same JPM segments refusal."),
    "doc99_qa0__after114": (0.0, "[ANS] doc99 seen. Same Kraft Heinz refusal."),
    "doc19_qa0__after114": (0.0, "[ANS] doc19 seen. Same Amazon revenue refusal."),
    "doc98_qa0__after114": (1.0, "[ANS] doc98 seen. Same JPM VaR correct."),
    "doc12_qa0__after114": (0.0, "[ANS] doc12 seen. Same Adobe OCF refusal."),
    "doc80_qa0__after115": (1.0, "[ANS] doc80 seen. Same Richard A. Johnson with 16,105,005 votes."),
    "doc23_qa0__after115": (0.0, "[ANS] doc23 seen. Same Amcor quick ratio refusal."),
    "doc111_qa0__after115": (0.0, "[ANS] doc111 seen. GOLD definitive 'No Microsoft -$2.5bn debt'. PRED refuses. Refusal on definitive ANS."),
    "doc33_qa0__after115": (0.0, "[ANS] doc33 seen. Same AMD revenue refusal."),
    "doc87_qa0__after115": (0.0, "[ANS] doc87 seen. Same JnJ inventory refusal."),
    "doc81_qa0__after115": (0.0, "[ANS] doc81 seen. Same GenMills CCC refusal."),
    "doc68_qa0__after115": (0.0, "[ANS] doc68 seen. Same Coca-Cola COGS refusal."),
    "doc4_qa0__after116": (0.0, "[ANS] doc4 seen. Same 3M segment refusal."),
    "doc66_qa0__after116": (0.0, "[ANS] doc66 seen. Same Boeing tax refusal."),
    "doc88_qa0__after116": (0.0, "[ANS] doc88 seen. GOLD definitive 'No, JnJ EPS decelerate 3.6% to 3.5%'. PRED refuses. Refusal on definitive ANS."),
    "doc93_qa0__after116": (0.0, "[ANS] doc93 seen. Same JnJ earnings refusal."),
    "doc105_qa0__after116": (1.0, "[ANS] doc105 seen. GOLD 'Yes MGM $0.01 throughout FY2022'. PRED 'Yes MGM Resorts $0.01 throughout 2022'. Correct."),
    "doc44_qa0__after116": (0.0, "[ANS] doc44 seen. GOLD definitive 'Yes' Card Member retention. PRED refuses. Refusal on definitive ANS."),
    "doc104_qa0__after116": (0.25, "[ANS] doc104 seen. Same wrong 10.5% specific."),
    "doc21_qa0__after116": (0.0, "[ANS] doc21 seen. Same Amcor AR refusal."),
    "doc6_qa0__after117": (0.0, "[ANS] doc6 seen. Same 3M debt securities refusal."),
    "doc44_qa0__after117": (0.0, "[ANS] doc44 seen. Same Card Member refusal."),
    "doc42_qa0__after117": (0.0, "[ANS] doc42 seen. Same AMEX tax refusal."),
    "doc54_qa0__after117": (1.0, "[ANS] doc54 seen. GOLD 'Yes 982 to 969 stores -1.32%'. PRED 'Yes 982 in Q2 FY2023 to 969 in Q2 FY2024'. Exact numbers correct."),
    "doc2_qa0__after117": (0.0, "[ANS] doc2 seen. Same 3M refusal."),
    "doc3_qa0__after117": (0.0, "[ANS] doc3 seen. Same 3M OM refusal."),
    "doc107_qa0__after118": (0.25, "[ANS] doc107 seen. GOLD 'Coverage ratio zero (negative EBIT)'. PRED '5.82' — wrong specific (confident different from 1.61/2.42)."),
    "doc93_qa0__after118": (0.0, "[ANS] doc93 seen. Same JnJ earnings refusal."),
    "doc4_qa0__after118": (0.0, "[ANS] doc4 seen. Same 3M segment refusal."),
    "doc22_qa0__after118": (0.0, "[ANS] doc22 seen. Same Amcor 8K refusal."),
    "doc37_qa0__after118": (0.0, "[ANS] doc37 seen. Same AMD customer refusal."),
    "doc73_qa0__after118": (0.0, "[ANS] doc73 seen. GOLD definitive 'Yes Corning positive WC $831M'. PRED refuses. Refusal on definitive ANS."),
    "doc45_qa0__after118": (0.0, "[ANS] doc45 seen. Same AWW dividends refusal."),
    "doc41_qa0__after118": (0.0, "[ANS] doc41 seen. Same AMEX gross margin refusal."),
    "doc34_qa0__after118": (0.0, "[ANS] doc34 seen. Same AMD operating refusal."),
    "doc15_qa0__after119": (1.0, "[ANS] doc15 seen. GOLD 0. PRED '0'. Match."),
    "doc45_qa0__after119": (0.0, "[ANS] doc45 seen. Same AWW dividends refusal."),
    "doc49_qa0__after119": (0.0, "[ANS] doc49 seen. Same Best Buy inventory refusal."),
    "doc68_qa0__after119": (0.0, "[ANS] doc68 seen. Same Coca-Cola COGS refusal."),
    "doc48_qa0__after119": (0.0, "[ANS] doc48 seen. Same Best Buy NPM refusal."),
    "doc25_qa0__after119": (0.0, "[ANS] doc25 seen. Same Amcor refusal."),
    "doc59_qa0__after119": (0.0, "[ANS] doc59 seen. Same Boeing PP&E refusal."),
    "doc52_qa0__after119": (0.0, "[ANS] doc52 seen. Same Best Buy refusal."),
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
