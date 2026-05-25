"""Phase 1.9 — bm25-corpus calibration cell — Part 3.

Entries 0303-0458 (156 entries).
"""

from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__bm25-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__bm25-corpus__calibration__seed42/results.jsonl")

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc97_qa0__after30", 1.0, "ACK refuse → 1.0."),
    ("doc52_qa0__after30", 1.0, "ACK GOLD 'Best Buy ops cash flow highest $1.8B'; PRED 'cash flow from operating activities brought in most cash flow' → 1.0 semantic match."),
    ("doc0_qa0__after30", 1.0, "ANS GOLD $1577 PRED '(1,577)' → 1.0 exact (parens variant)."),
    ("doc60_qa0__after30", 1.0, "ACK refuse → 1.0."),
    ("doc5_qa0__after30", 0.0, "ANS refusal on definitive 3M quick ratio → 0.0."),
    ("doc42_qa0__after30", 1.0, "ACK refuse → 1.0."),
    ("doc90_qa0__after30", 1.0, "ACK Consumer Health discontinued match → 1.0."),
    ("doc124_qa0__after31", 1.0, "ACK refuse → 1.0."),
    ("doc91_qa0__after31", 1.0, "ACK refuse → 1.0."),
    ("doc21_qa0__after31", 1.0, "ANS $1,615.9 exact → 1.0."),
    ("doc63_qa0__after31", 0.5, "ACK partial defense contractors confusion → 0.5."),
    ("doc120_qa0__after31", 1.0, "ACK refuse → 1.0."),
    ("doc67_qa0__after31", 1.0, "ACK refuse → 1.0."),
    ("doc139_qa0__after31", 1.0, "ACK refuse → 1.0."),
    ("doc18_qa0__after31", 0.25, "ANS 34.12 vs 93.86 → 0.25 confident wrong."),
    ("doc135_qa0__after31", 1.0, "ACK refuse → 1.0."),
    ("doc141_qa0__after31", 1.0, "ACK refuse → 1.0."),
    ("doc117_qa0__after32", 1.0, "ACK Nike ops cash flow match → 1.0."),
    ("doc18_qa0__after32", 0.25, "ANS 36.12 vs 93.86 → 0.25."),
    ("doc7_qa0__after32", 1.0, "ANS 65 years match → 1.0."),
    ("doc115_qa0__after32", 1.0, "ACK refuse → 1.0."),
    ("doc47_qa0__after32", 1.0, "ACK refuse → 1.0."),
    ("doc106_qa0__after32", 1.0, "ACK refuse → 1.0."),
    ("doc87_qa0__after32", 1.0, "ACK refuse → 1.0."),
    ("doc56_qa0__after32", 1.0, "ACK GOLD 1.73 PRED 1.73 → 1.0 exact (correct inferred answer)."),
    ("doc77_qa0__after32", 1.0, "ACK refuse → 1.0."),
    ("doc112_qa0__after32", 1.0, "ACK refuse → 1.0."),
    ("doc135_qa0__after33", 1.0, "ACK refuse → 1.0."),
    ("doc144_qa0__after33", 1.0, "ACK refuse → 1.0."),
    ("doc18_qa0__after33", 0.25, "ANS 36.12 vs 93.86 → 0.25."),
    ("doc34_qa0__after33", 0.25, "ACK 'Operating margin not useful for AMD' reframe wrong → 0.25."),
    ("doc72_qa0__after33", 1.0, "ACK refuse → 1.0."),
    ("doc15_qa0__after33", 1.0, "ANS 0=0 → 1.0."),
    ("doc90_qa0__after33", 1.0, "ACK Consumer Health match → 1.0."),
    ("doc89_qa0__after33", 1.0, "ACK refuse → 1.0."),
    ("doc64_qa0__after33", 1.0, "ACK refuse → 1.0."),
    ("doc125_qa0__after33", 1.0, "ACK 'proposal not approved' match → 1.0."),
    ("doc130_qa0__after34", 1.0, "ACK refuse → 1.0."),
    ("doc26_qa0__after34", 1.0, "ANS Amcor GM decline match → 1.0."),
    ("doc68_qa0__after34", 1.0, "ACK refuse → 1.0."),
    ("doc40_qa0__after34", 1.0, "ACK refuse → 1.0."),
    ("doc129_qa0__after34", 1.0, "ACK refuse → 1.0."),
    ("doc144_qa0__after34", 1.0, "ACK refuse → 1.0."),
    ("doc25_qa0__after34", 1.0, "ANS Amcor packaging match → 1.0."),
    ("doc34_qa0__after34", 1.0, "ANS Xilinx amortization exact match → 1.0."),
    ("doc131_qa0__after34", 1.0, "ACK refuse → 1.0."),
    ("doc29_qa0__after34", 0.25, "ANS -5% vs 'flat' → 0.25 wrong direction."),
    ("doc136_qa0__after35", 1.0, "ACK refuse → 1.0."),
    ("doc93_qa0__after35", 1.0, "ACK refuse → 1.0."),
    ("doc146_qa0__after35", 1.0, "ACK refuse → 1.0."),
    ("doc149_qa0__after35", 1.0, "ACK refuse → 1.0."),
    ("doc42_qa0__after35", 1.0, "ACK refuse → 1.0."),
    ("doc85_qa0__after35", 1.0, "ACK refuse → 1.0."),
    ("doc98_qa0__after35", 1.0, "ACK refuse → 1.0."),
    ("doc92_qa0__after35", 1.0, "ACK refuse → 1.0."),
    ("doc78_qa0__after35", 1.0, "ACK refuse → 1.0."),
    ("doc100_qa0__after35", 1.0, "ACK refuse → 1.0."),
    ("doc88_qa0__after36", 1.0, "ACK refuse → 1.0."),
    ("doc69_qa0__after36", 1.0, "ACK refuse → 1.0."),
    ("doc120_qa0__after36", 1.0, "ACK refuse → 1.0."),
    ("doc112_qa0__after36", 1.0, "ACK refuse → 1.0."),
    ("doc133_qa0__after36", 1.0, "ACK refuse → 1.0."),
    ("doc136_qa0__after36", 1.0, "ACK refuse → 1.0."),
    ("doc145_qa0__after36", 1.0, "ACK refuse → 1.0."),
    ("doc131_qa0__after36", 1.0, "ACK refuse → 1.0."),
    ("doc31_qa0__after36", 0.0, "ANS refusal on definitive → 0.0."),
    ("doc3_qa0__after36", 1.0, "ANS GOLD '1.7% decrease + litigation/PFAS/Russia'; PRED 'litigation + impairment PFAS + costs Russia + divestiture + growth investment' — captures all major drivers from gold (litigation, PFAS, Russia) + extras → 1.0 match."),
    ("doc52_qa0__after37", 1.0, "ACK Best Buy cash flow match → 1.0."),
    ("doc70_qa0__after37", 1.0, "ACK refuse → 1.0."),
    ("doc11_qa0__after37", 1.0, "ANS calc shows (1,493,602-903,095)/903,095 = 65.4% (truncated, correct) → 1.0."),
    ("doc10_qa0__after37", 0.25, "ANS GOLD 0.66 PRED '1.29' → 95% off → 0.25."),
    ("doc90_qa0__after37", 1.0, "ACK Consumer Health match → 1.0."),
    ("doc54_qa0__after37", 1.0, "ACK refuse → 1.0."),
    ("doc50_qa0__after37", 1.0, "ACK refuse → 1.0."),
    ("doc107_qa0__after37", 1.0, "ACK refuse → 1.0."),
    ("doc129_qa0__after37", 1.0, "ACK refuse → 1.0."),
    ("doc108_qa0__after37", 1.0, "ACK refuse → 1.0."),
    ("doc90_qa0__after38", 1.0, "ACK Consumer Health match → 1.0."),
    ("doc138_qa0__after38", 1.0, "ACK refuse → 1.0."),
    ("doc43_qa0__after38", 1.0, "ACK refuse → 1.0."),
    ("doc71_qa0__after38", 1.0, "ACK refuse → 1.0."),
    ("doc1_qa0__after38", 1.0, "ANS 8.738B within tolerance → 1.0."),
    ("doc27_qa0__after38", 0.5, "ANS partial restructuring (lists components, $118M, no 87% specific) → 0.5."),
    ("doc140_qa0__after38", 1.0, "ACK refuse → 1.0."),
    ("doc24_qa0__after38", 0.5, "ANS partial Amcor acquisitions → 0.5."),
    ("doc135_qa0__after38", 1.0, "ACK refuse → 1.0."),
    ("doc88_qa0__after38", 1.0, "ACK refuse → 1.0."),
    ("doc115_qa0__after39", 1.0, "ACK refuse → 1.0."),
    ("doc92_qa0__after39", 1.0, "ACK refuse → 1.0."),
    ("doc146_qa0__after39", 1.0, "ACK refuse → 1.0."),
    ("doc76_qa0__after39", 1.0, "ACK refuse → 1.0."),
    ("doc80_qa0__after39", 1.0, "ACK refuse → 1.0."),
    ("doc8_qa0__after39", 0.5, "ANS GOLD 24.26 PRED '25.73' — 5.7% off, just outside 5% tolerance → 0.5 partial."),
    ("doc33_qa0__after39", 1.0, "ANS AMD EPYC+semi-custom+Xilinx match → 1.0."),
    ("doc95_qa0__after39", 1.0, "ACK refuse → 1.0."),
    ("doc46_qa0__after39", 1.0, "ACK refuse → 1.0."),
    ("doc2_qa0__after39", 0.0, "ANS Y/N flip → 0.0."),
    ("doc16_qa0__after40", 0.25, "ANS GOLD 9.5 PRED '12.0' → 26% off → 0.25."),
    ("doc93_qa0__after40", 1.0, "ACK refuse → 1.0."),
    ("doc128_qa0__after40", 1.0, "ACK refuse → 1.0."),
    ("doc110_qa0__after40", 1.0, "ACK refuse → 1.0."),
    ("doc59_qa0__after40", 1.0, "ACK refuse → 1.0."),
    ("doc54_qa0__after40", 1.0, "ACK refuse → 1.0."),
    ("doc135_qa0__after40", 1.0, "ACK refuse → 1.0."),
    ("doc11_qa0__after40", 0.0, "ANS refusal on definitive 65.4% → 0.0."),
    ("doc53_qa0__after40", 1.0, "ACK refuse → 1.0."),
    ("doc57_qa0__after40", 1.0, "ACK refuse → 1.0."),
    ("doc85_qa0__after41", 1.0, "ACK refuse → 1.0."),
    ("doc88_qa0__after41", 1.0, "ACK refuse → 1.0."),
    ("doc53_qa0__after41", 1.0, "ACK refuse → 1.0."),
    ("doc61_qa0__after41", 1.0, "ACK refuse → 1.0."),
    ("doc46_qa0__after41", 1.0, "ACK refuse → 1.0."),
    ("doc124_qa0__after41", 1.0, "ACK refuse → 1.0."),
    ("doc84_qa0__after41", 1.0, "ACK refuse → 1.0."),
    ("doc134_qa0__after41", 1.0, "ACK refuse → 1.0."),
    ("doc21_qa0__after41", 1.0, "ANS $1,615.9 exact → 1.0."),
    ("doc87_qa0__after41", 1.0, "ACK refuse → 1.0."),
    ("doc106_qa0__after42", 1.0, "ACK refuse → 1.0."),
    ("doc124_qa0__after42", 1.0, "ACK refuse → 1.0."),
    ("doc98_qa0__after42", 1.0, "ACK refuse → 1.0."),
    ("doc56_qa0__after42", 0.25, "ACK GOLD 1.73 PRED '1.06' — 39% off → 0.25 confident wrong."),
    ("doc36_qa0__after42", 0.25, "ANS GOLD 'Data Center' PRED 'Gaming segment' → 0.25 confident wrong segment."),
    ("doc51_qa0__after42", 1.0, "ACK refuse → 1.0."),
    ("doc111_qa0__after42", 1.0, "ACK refuse → 1.0."),
    ("doc60_qa0__after42", 1.0, "ACK refuse → 1.0."),
    ("doc148_qa0__after42", 1.0, "ACK refuse → 1.0."),
    ("doc50_qa0__after42", 1.0, "ACK refuse → 1.0."),
    ("doc25_qa0__after43", 1.0, "ANS Amcor packaging match → 1.0."),
    ("doc114_qa0__after43", 1.0, "ACK refuse → 1.0."),
    ("doc133_qa0__after43", 1.0, "ACK refuse → 1.0."),
    ("doc141_qa0__after43", 1.0, "ACK refuse → 1.0."),
    ("doc55_qa0__after43", 1.0, "ACK refuse → 1.0."),
    ("doc85_qa0__after43", 1.0, "ACK refuse → 1.0."),
    ("doc27_qa0__after43", 0.5, "ANS partial restructuring → 0.5."),
    ("doc94_qa0__after43", 1.0, "ACK refuse → 1.0."),
    ("doc122_qa0__after43", 0.25, "ACK PRED '0' → 0.25."),
    ("doc24_qa0__after43", 0.5, "ANS partial → 0.5."),
    ("doc76_qa0__after44", 1.0, "ACK refuse → 1.0."),
    ("doc35_qa0__after44", 1.0, "ANS AMD $3,565M match → 1.0."),
    ("doc17_qa0__after44", 0.25, "ANS -1.42 vs -0.02 → 0.25."),
    ("doc30_qa0__after44", 0.5, "ANS GOLD 4.2% PRED truncated 'D&A $167M, net revenue not provided' — hedged partial calc → 0.5."),
    ("doc66_qa0__after44", 1.0, "ACK refuse → 1.0."),
    ("doc101_qa0__after44", 0.25, "ACK GOLD $5818 PRED '$12,650M' — 117% off → 0.25 confident wrong."),
    ("doc95_qa0__after44", 1.0, "ACK refuse → 1.0."),
    ("doc67_qa0__after44", 1.0, "ACK refuse → 1.0."),
    ("doc53_qa0__after44", 1.0, "ACK refuse → 1.0."),
    ("doc141_qa0__after44", 1.0, "ACK refuse → 1.0."),
    ("doc124_qa0__after45", 1.0, "ACK refuse → 1.0."),
    ("doc141_qa0__after45", 1.0, "ACK refuse → 1.0."),
    ("doc56_qa0__after45", 0.25, "ACK GOLD 1.73 PRED '1.04' — 40% off → 0.25 confident wrong."),
    ("doc11_qa0__after45", 1.0, "ANS calc shows 65.4% (truncated correct) → 1.0."),
    ("doc109_qa0__after45", 1.0, "ACK refuse → 1.0."),
    ("doc59_qa0__after45", 1.0, "ACK refuse → 1.0."),
    ("doc57_qa0__after45", 1.0, "ACK refuse → 1.0."),
    ("doc30_qa0__after45", 1.0, "ANS calc 4.18% within tolerance → 1.0."),
    ("doc32_qa0__after45", 1.0, "ANS AMD products match → 1.0."),
]


def main() -> None:
    existing = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            try: existing.add(json.loads(line)["qid"])
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
