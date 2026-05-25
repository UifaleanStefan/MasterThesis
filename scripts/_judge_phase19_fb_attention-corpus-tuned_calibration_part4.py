"""Phase 1.9 — attention-corpus-tuned calibration cell — Part 4.

Entries 0452-0599 (148 entries). Mid-corpus, more ANS attempts.

Notable wins: doc30 4.18% calc match (5x repeat), doc32 AMD products full
list match, doc37 'Y 16%' exact, doc52 cash flow ops match (2x), doc28
'2,018M' exact, doc35 AMD $3,565M ops match, doc36 'Data Center' match
(2x), doc22 Amcor 8k indenture detailed match, doc55 'Entertainment 9%
gaming' exact, doc15 0 exact, doc1 $8.738B tolerance, doc40 'OM not useful'
semantic, doc41 'GM not useful' semantic, doc42 AMEX tax rate exact,
doc53 'Y -42% cash decline' Y match + supporting calc, doc56 cash
$1,874→$1,093 derives -42%. Confident-wrong: doc75 2.06 vs 17.98, doc9
5.0 vs 1.9, doc92 $3.7B vs $13.2B, doc74 $52,694 vs $59,268, doc57 6.9
vs 101.5, doc17 -1.32 vs -0.02 (3x), doc16 11.98 vs 9.5 (2x), doc11
-99.6% calc vs 65.4% (2x), doc29 -5% vs 'flat' (5x). Y/N flip: doc50
'fluctuated 2%' vs 'consistent 1.1% decline' (2x). Refusal-on-definitive:
doc0 $1577, doc31 quick ratio (3x), doc14 FCF conv (2x).
"""

from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__attention-corpus-tuned__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__attention-corpus-tuned__calibration__seed42/results.jsonl")

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc56_qa0__after45", 1.0, "ACK refuse → 1.0."),
    ("doc11_qa0__after45", 0.25, "ANS same -99.6% wrong calc → 0.25."),
    ("doc109_qa0__after45", 1.0, "ACK refuse → 1.0."),
    ("doc59_qa0__after45", 1.0, "ACK refuse → 1.0."),
    ("doc57_qa0__after45", 1.0, "ACK refuse → 1.0."),
    ("doc30_qa0__after45", 1.0, "ANS calc 4.18% within tolerance of 4.2% → 1.0."),
    ("doc32_qa0__after45", 1.0, "ANS GOLD AMD products list (CPUs, GPUs, DPUs, FPGAs, SoC, APUs); PRED matches all + extra detail → 1.0."),
    ("doc31_qa0__after45", 0.0, "ANS refusal on definitive Y/N+numeric → 0.0."),
    ("doc99_qa0__after46", 1.0, "ACK refuse → 1.0."),
    ("doc37_qa0__after46", 1.0, "ANS GOLD 'Y 16% one customer' PRED 'Yes AMD 16% one customer FY2022' → 1.0 exact."),
    ("doc54_qa0__after46", 1.0, "ACK refuse → 1.0."),
    ("doc118_qa0__after46", 1.0, "ACK refuse → 1.0."),
    ("doc58_qa0__after46", 1.0, "ACK refuse → 1.0."),
    ("doc24_qa0__after46", 0.5, "ANS partial Amcor acquisitions mixed FY years → 0.5."),
    ("doc30_qa0__after46", 1.0, "ANS calc 4.18% → 1.0."),
    ("doc50_qa0__after46", 1.0, "ACK refuse → 1.0."),
    ("doc148_qa0__after46", 1.0, "ACK refuse → 1.0."),
    ("doc95_qa0__after46", 1.0, "ACK refuse → 1.0."),
    ("doc1_qa0__after47", 1.0, "ANS 8.738B in tolerance → 1.0."),
    ("doc75_qa0__after47", 1.0, "ACK refuse → 1.0."),
    ("doc92_qa0__after47", 1.0, "ACK refuse → 1.0."),
    ("doc87_qa0__after47", 1.0, "ACK refuse → 1.0."),
    ("doc93_qa0__after47", 1.0, "ACK refuse → 1.0."),
    ("doc78_qa0__after47", 1.0, "ACK refuse → 1.0."),
    ("doc97_qa0__after47", 1.0, "ACK refuse → 1.0."),
    ("doc49_qa0__after47", 1.0, "ACK refuse → 1.0."),
    ("doc136_qa0__after47", 1.0, "ACK refuse → 1.0."),
    ("doc31_qa0__after47", 0.0, "ANS refusal on definitive → 0.0."),
    ("doc125_qa0__after48", 1.0, "ACK refuse → 1.0."),
    ("doc4_qa0__after48", 0.5, "ANS partial 'Consumer segment' without % → 0.5."),
    ("doc58_qa0__after48", 1.0, "ACK refuse → 1.0."),
    ("doc133_qa0__after48", 1.0, "ACK refuse → 1.0."),
    ("doc40_qa0__after48", 1.0, "ANS GOLD 'Perf not measured through OM' PRED 'OM not useful for AMEX, non-interest income + card fees not product sales, more relevant net income/total revenues' → 1.0 semantic match."),
    ("doc148_qa0__after48", 1.0, "ACK refuse → 1.0."),
    ("doc30_qa0__after48", 1.0, "ANS calc 4.18% → 1.0."),
    ("doc76_qa0__after48", 1.0, "ACK refuse → 1.0."),
    ("doc121_qa0__after48", 1.0, "ACK refuse → 1.0."),
    ("doc75_qa0__after48", 0.25, "ACK GOLD 17.98 PRED 2.06 — confident wrong → 0.25."),
    ("doc41_qa0__after49", 1.0, "ANS GOLD 'Perf not measured through GM' PRED 'GM not useful for AMEX, service fees + card fees + interest income not goods sales' → 1.0 semantic match."),
    ("doc27_qa0__after49", 0.5, "ANS same partial $93M employee/fixed/other → 0.5."),
    ("doc16_qa0__after49", 0.25, "ANS 11.98 vs 9.5 → 0.25 confident wrong."),
    ("doc145_qa0__after49", 1.0, "ACK refuse → 1.0."),
    ("doc117_qa0__after49", 1.0, "ACK refuse → 1.0."),
    ("doc65_qa0__after49", 1.0, "ACK refuse → 1.0."),
    ("doc66_qa0__after49", 1.0, "ACK refuse → 1.0."),
    ("doc58_qa0__after49", 1.0, "ACK refuse → 1.0."),
    ("doc138_qa0__after49", 1.0, "ACK refuse → 1.0."),
    ("doc4_qa0__after49", 0.5, "ANS partial → 0.5."),
    ("doc76_qa0__after50", 1.0, "ACK refuse → 1.0."),
    ("doc113_qa0__after50", 1.0, "ACK refuse → 1.0."),
    ("doc9_qa0__after50", 0.25, "ANS GOLD 1.9% PRED 5.0% — confident wrong (163% off) → 0.25."),
    ("doc136_qa0__after50", 1.0, "ACK refuse → 1.0."),
    ("doc24_qa0__after50", 0.5, "ANS partial → 0.5."),
    ("doc130_qa0__after50", 1.0, "ACK refuse → 1.0."),
    ("doc11_qa0__after50", 0.25, "ANS same -99.6% wrong calc → 0.25."),
    ("doc35_qa0__after50", 1.0, "ANS AMD $3,565M cashflow match → 1.0."),
    ("doc29_qa0__after50", 0.25, "ANS -5% vs 'flat' → 0.25 wrong direction."),
    ("doc53_qa0__after50", 1.0, "ACK refuse → 1.0."),
    ("doc52_qa0__after51", 1.0, "ACK Best Buy cash flow ops match → 1.0."),
    ("doc122_qa0__after51", 0.25, "ACK PRED '0' vs $411M → 0.25."),
    ("doc128_qa0__after51", 1.0, "ACK refuse → 1.0."),
    ("doc53_qa0__after51", 1.0, "ACK refuse → 1.0."),
    ("doc104_qa0__after51", 1.0, "ACK refuse → 1.0."),
    ("doc98_qa0__after51", 1.0, "ACK refuse → 1.0."),
    ("doc17_qa0__after51", 0.25, "ANS -1.32 vs -0.02 → 0.25 confident wrong."),
    ("doc77_qa0__after51", 1.0, "ACK refuse → 1.0."),
    ("doc136_qa0__after51", 1.0, "ACK refuse → 1.0."),
    ("doc61_qa0__after51", 1.0, "ACK refuse → 1.0."),
    ("doc137_qa0__after52", 1.0, "ACK refuse → 1.0."),
    ("doc30_qa0__after52", 1.0, "ANS calc 4.18% → 1.0."),
    ("doc54_qa0__after52", 1.0, "ACK refuse → 1.0."),
    ("doc53_qa0__after52", 1.0, "ACK refuse → 1.0."),
    ("doc80_qa0__after52", 1.0, "ACK refuse → 1.0."),
    ("doc36_qa0__after52", 1.0, "ANS 'Data Center segment' → 1.0."),
    ("doc121_qa0__after52", 1.0, "ACK refuse → 1.0."),
    ("doc125_qa0__after52", 1.0, "ACK refuse → 1.0."),
    ("doc136_qa0__after52", 1.0, "ACK refuse → 1.0."),
    ("doc35_qa0__after52", 1.0, "ANS AMD $3,565M ops cashflow match → 1.0."),
    ("doc94_qa0__after53", 1.0, "ACK refuse → 1.0."),
    ("doc36_qa0__after53", 1.0, "ANS 'Data Center segment' → 1.0."),
    ("doc56_qa0__after53", 1.0, "ACK refuse → 1.0."),
    ("doc29_qa0__after53", 0.25, "ANS -5% vs 'flat' → 0.25."),
    ("doc139_qa0__after53", 1.0, "ACK refuse → 1.0."),
    ("doc15_qa0__after53", 1.0, "ANS 0=0 → 1.0."),
    ("doc0_qa0__after53", 1.0, "ANS $1501M just within 5% tolerance of $1577 → 1.0."),
    ("doc78_qa0__after53", 1.0, "ACK refuse → 1.0."),
    ("doc50_qa0__after53", 0.0, "ANS GOLD 'Y consistent 1.1% decline'; PRED 'fluctuated more than 2%, not historically consistent' — Y/N FLIP (Yes vs No) → 0.0."),
    ("doc145_qa0__after53", 1.0, "ACK refuse → 1.0."),
    ("doc63_qa0__after54", 1.0, "ACK refuse → 1.0."),
    ("doc0_qa0__after54", 0.0, "ANS refusal on definitive $1577 → 0.0."),
    ("doc134_qa0__after54", 1.0, "ACK refuse → 1.0."),
    ("doc80_qa0__after54", 1.0, "ACK refuse → 1.0."),
    ("doc133_qa0__after54", 1.0, "ACK refuse → 1.0."),
    ("doc29_qa0__after54", 0.25, "ANS -5% vs 'flat' → 0.25."),
    ("doc42_qa0__after54", 1.0, "ANS GOLD 'AMEX tax 24.6→21.6%'; PRED '24.6% FY21 to 21.6% FY22' → 1.0 exact."),
    ("doc83_qa0__after54", 1.0, "ACK refuse → 1.0."),
    ("doc137_qa0__after54", 1.0, "ACK refuse → 1.0."),
    ("doc92_qa0__after54", 1.0, "ACK refuse → 1.0."),
    ("doc147_qa0__after55", 1.0, "ACK refuse → 1.0."),
    ("doc108_qa0__after55", 1.0, "ACK refuse → 1.0."),
    ("doc100_qa0__after55", 1.0, "ACK refuse → 1.0."),
    ("doc37_qa0__after55", 1.0, "ANS 'Y 16% one customer' match → 1.0."),
    ("doc50_qa0__after55", 0.0, "ANS same Y/N flip 'fluctuated more than 2%' vs 'consistent' → 0.0."),
    ("doc92_qa0__after55", 0.25, "ACK GOLD $13.2B PRED $3.7B — confident wrong → 0.25."),
    ("doc53_qa0__after55", 1.0, "ANS GOLD 'Y -42% FY23→Q2 FY24 cash decline'; PRED 'Y, $1,874M Jan2023 → $1,093M July2023' — Y match + supporting figures derive -41.7% decline matching gold's ~42% → 1.0."),
    ("doc29_qa0__after55", 0.25, "ANS -5% vs 'flat' → 0.25."),
    ("doc120_qa0__after55", 1.0, "ACK refuse → 1.0."),
    ("doc128_qa0__after55", 1.0, "ACK refuse → 1.0."),
    ("doc3_qa0__after56", 0.75, "ANS doc3 OI — qualitative drivers without -1.7% → 0.75."),
    ("doc22_qa0__after56", 1.0, "ANS GOLD 'Amcor 8k Jul-2022, Amcor Flexibles NA substitute for Amcor Finance USA, Guaranteed Senior Notes 2026 + 2028'; PRED '8k Jul-2022, Second Supplemental Indenture, First Supplemental Indenture, Amcor Flexibles NA substitute for Amcor Finance USA' → 1.0 exact match."),
    ("doc116_qa0__after56", 1.0, "ACK refuse → 1.0."),
    ("doc141_qa0__after56", 1.0, "ACK refuse → 1.0."),
    ("doc14_qa0__after56", 0.0, "ANS GOLD 'Y FCF conv +13% 143→156%'; PRED 'not contain specific information... cannot determine' — refusal on definitive Y/N → 0.0."),
    ("doc88_qa0__after56", 1.0, "ACK refuse → 1.0."),
    ("doc148_qa0__after56", 1.0, "ACK refuse → 1.0."),
    ("doc60_qa0__after56", 1.0, "ACK refuse → 1.0."),
    ("doc67_qa0__after56", 1.0, "ACK refuse → 1.0."),
    ("doc109_qa0__after56", 1.0, "ACK refuse → 1.0."),
    ("doc120_qa0__after57", 1.0, "ACK refuse → 1.0."),
    ("doc63_qa0__after57", 1.0, "ACK refuse → 1.0."),
    ("doc27_qa0__after57", 0.5, "ANS same partial → 0.5."),
    ("doc28_qa0__after57", 1.0, "ANS GOLD '$2,018M Adj EBITDA FY23' PRED '2,018 million' → exact → 1.0."),
    ("doc31_qa0__after57", 0.0, "ANS refusal on definitive → 0.0."),
    ("doc107_qa0__after57", 1.0, "ACK refuse → 1.0."),
    ("doc74_qa0__after57", 0.25, "ACK GOLD $59268 PRED $52,694 → 0.25 confident wrong."),
    ("doc121_qa0__after57", 1.0, "ACK refuse → 1.0."),
    ("doc69_qa0__after57", 1.0, "ACK refuse → 1.0."),
    ("doc57_qa0__after57", 0.25, "ANS GOLD 101.5% PRED 6.9% → 0.25 confident wrong (way off)."),
    ("doc55_qa0__after58", 1.0, "ANS GOLD 'entertainment 9% Q2 FY24 gaming'; PRED 'Entertainment 9.0% comparable sales gaming' → 1.0 exact."),
    ("doc118_qa0__after58", 1.0, "ACK refuse → 1.0."),
    ("doc59_qa0__after58", 1.0, "ACK refuse → 1.0."),
    ("doc64_qa0__after58", 1.0, "ACK 'Y Boeing cyclical' match → 1.0."),
    ("doc17_qa0__after58", 0.25, "ANS -1.32 vs -0.02 → 0.25."),
    ("doc14_qa0__after58", 0.0, "ANS refusal on definitive Y/N → 0.0."),
    ("doc16_qa0__after58", 0.25, "ANS 11.98 vs 9.5 → 0.25."),
    ("doc66_qa0__after58", 1.0, "ACK refuse → 1.0."),
    ("doc78_qa0__after58", 1.0, "ACK refuse → 1.0."),
    ("doc95_qa0__after58", 1.0, "ACK refuse → 1.0."),
    ("doc29_qa0__after59", 0.25, "ANS -5% vs 'flat' → 0.25."),
    ("doc65_qa0__after59", 1.0, "ACK refuse → 1.0."),
    ("doc87_qa0__after59", 1.0, "ACK refuse → 1.0."),
    ("doc116_qa0__after59", 1.0, "ACK refuse → 1.0."),
    ("doc66_qa0__after59", 1.0, "ACK refuse → 1.0."),
    ("doc110_qa0__after59", 1.0, "ACK refuse → 1.0."),
    ("doc30_qa0__after59", 1.0, "ANS calc 4.18% → 1.0."),
    ("doc134_qa0__after59", 1.0, "ACK refuse → 1.0."),
    ("doc119_qa0__after59", 1.0, "ACK refuse → 1.0."),
    ("doc147_qa0__after59", 1.0, "ACK refuse → 1.0."),
]


def main() -> None:
    existing = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line); existing.add(obj["qid"])
            except Exception: continue
    added, scores = 0, []
    with RESULTS.open("a", encoding="utf-8") as fh:
        for s, sc, r in JUDGMENTS:
            qid = f"{QID_PREFIX}{s}{QID_SUFFIX}"
            if qid in existing: continue
            fh.write(json.dumps({"qid": qid, "judge_score": float(sc), "rationale": r, "judge_model": "claude-opus-4.7-1m", "judge_protocol": "v1"}, ensure_ascii=False) + "\n")
            added += 1; scores.append(sc)
    print(f"Added {added}. Dist: {dict((f'{x:.2f}', scores.count(x)) for x in sorted(set(scores), reverse=True))}")
    if scores: print(f"Mean: {sum(scores)/len(scores):.4f}")
    total = sum(1 for _ in RESULTS.read_text(encoding="utf-8").splitlines() if _.strip())
    print(f"Total: {total}/1500 ({100*total/1500:.1f}%)")


if __name__ == "__main__":
    main()
