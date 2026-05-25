"""Phase 1.9 — bm25-corpus calibration cell — Part 1.

Entries 0-152 (153 entries, early corpus).

HARD RULE: each judge_score from Claude reading the (question, gold, predicted)
triple manually. NO heuristic / auto-judging.

Early-corpus pattern: similar to attention-corpus-tuned — overwhelmingly
honest refusals on ACK (1.0). Notable wins: doc1 8.738B within tolerance,
doc6 3M debt securities full list match, doc7 Y 65 years match, doc12 0.83
exact, doc90 J&J Consumer Health discontinued match, doc96 JPM gross
margin semantic match, doc125 PepsiCo proposal not approved match.
Confident-wrong: doc122 PRED '0' vs $411M (twice), doc58 wrong $ amounts
$1,831M/$1,031M vs $382, doc75 2.06 vs 17.98, doc82 2.50 vs 0.68, doc129
2pp vs 1pp, doc138 wrong reasons (twice), doc12 0.27 wrong. Partial:
doc3 OI drivers without -1.7%, doc63 Boeing customers without 40%.
Refusal-on-definitive (ANS mode): doc5 quick ratio (the only 0.0).
"""

from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__bm25-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__bm25-corpus__calibration__seed42/results.jsonl")

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc123_qa0__after0", 1.0, "ACK refuse → 1.0."),
    ("doc31_qa0__after0", 1.0, "ACK refuse → 1.0."),
    ("doc147_qa0__after0", 1.0, "ACK refuse → 1.0."),
    ("doc130_qa0__after0", 1.0, "ACK refuse → 1.0."),
    ("doc115_qa0__after0", 1.0, "ACK refuse → 1.0."),
    ("doc119_qa0__after0", 1.0, "ACK refuse → 1.0."),
    ("doc133_qa0__after0", 1.0, "ACK refuse → 1.0."),
    ("doc137_qa0__after0", 1.0, "ACK refuse → 1.0."),
    ("doc59_qa0__after0", 1.0, "ACK refuse → 1.0."),
    ("doc27_qa0__after0", 1.0, "ACK refuse → 1.0."),
    ("doc93_qa0__after1", 1.0, "ACK refuse → 1.0."),
    ("doc72_qa0__after1", 1.0, "ACK refuse → 1.0."),
    ("doc64_qa0__after1", 1.0, "ACK refuse → 1.0."),
    ("doc6_qa0__after1", 1.0, "ACK refuse → 1.0."),
    ("doc27_qa0__after1", 1.0, "ACK refuse → 1.0."),
    ("doc35_qa0__after1", 1.0, "ACK refuse → 1.0."),
    ("doc5_qa0__after1", 1.0, "ACK refuse → 1.0."),
    ("doc60_qa0__after1", 1.0, "ACK refuse → 1.0."),
    ("doc106_qa0__after1", 1.0, "ACK refuse → 1.0."),
    ("doc87_qa0__after1", 1.0, "ACK refuse → 1.0."),
    ("doc101_qa0__after2", 1.0, "ACK refuse → 1.0."),
    ("doc71_qa0__after2", 1.0, "ACK refuse → 1.0."),
    ("doc1_qa0__after2", 1.0, "ANS GOLD $8.70 PRED '8.738 billion' — within tolerance → 1.0."),
    ("doc118_qa0__after2", 1.0, "ACK refuse → 1.0."),
    ("doc75_qa0__after2", 1.0, "ACK refuse → 1.0."),
    ("doc67_qa0__after2", 1.0, "ACK refuse → 1.0."),
    ("doc13_qa0__after2", 1.0, "ACK refuse → 1.0."),
    ("doc78_qa0__after2", 1.0, "ACK refuse → 1.0."),
    ("doc116_qa0__after2", 1.0, "ACK refuse → 1.0."),
    ("doc91_qa0__after2", 1.0, "ACK refuse → 1.0."),
    ("doc43_qa0__after3", 1.0, "ACK refuse → 1.0."),
    ("doc120_qa0__after3", 1.0, "ACK refuse → 1.0."),
    ("doc101_qa0__after3", 1.0, "ACK refuse → 1.0."),
    ("doc64_qa0__after3", 1.0, "ACK refuse → 1.0."),
    ("doc107_qa0__after3", 1.0, "ACK refuse → 1.0."),
    ("doc121_qa0__after3", 1.0, "ACK refuse → 1.0."),
    ("doc102_qa0__after3", 1.0, "ACK refuse → 1.0."),
    ("doc90_qa0__after3", 1.0, "ACK 'Consumer Health discontinued Aug 30 2023' match → 1.0 correct inferred."),
    ("doc26_qa0__after3", 1.0, "ACK refuse → 1.0."),
    ("doc22_qa0__after3", 1.0, "ACK refuse → 1.0."),
    ("doc122_qa0__after4", 0.25, "ACK PRED '0' vs $411M → 0.25 confident wrong."),
    ("doc141_qa0__after4", 1.0, "ACK refuse → 1.0."),
    ("doc25_qa0__after4", 1.0, "ACK refuse → 1.0."),
    ("doc43_qa0__after4", 1.0, "ACK refuse → 1.0."),
    ("doc76_qa0__after4", 1.0, "ACK refuse → 1.0."),
    ("doc120_qa0__after4", 1.0, "ACK refuse → 1.0."),
    ("doc138_qa0__after4", 1.0, "ACK refuse → 1.0."),
    ("doc42_qa0__after4", 1.0, "ACK refuse → 1.0."),
    ("doc83_qa0__after4", 1.0, "ACK refuse → 1.0."),
    ("doc95_qa0__after4", 1.0, "ACK refuse → 1.0."),
    ("doc147_qa0__after5", 1.0, "ACK refuse → 1.0."),
    ("doc32_qa0__after5", 1.0, "ACK refuse → 1.0."),
    ("doc131_qa0__after5", 1.0, "ACK refuse → 1.0."),
    ("doc97_qa0__after5", 1.0, "ACK refuse → 1.0."),
    ("doc93_qa0__after5", 1.0, "ACK refuse → 1.0."),
    ("doc80_qa0__after5", 1.0, "ACK refuse → 1.0."),
    ("doc109_qa0__after5", 1.0, "ACK refuse → 1.0."),
    ("doc113_qa0__after5", 1.0, "ACK refuse → 1.0."),
    ("doc13_qa0__after5", 1.0, "ACK refuse → 1.0."),
    ("doc110_qa0__after5", 1.0, "ACK refuse → 1.0."),
    ("doc127_qa0__after6", 1.0, "ACK refuse → 1.0."),
    ("doc149_qa0__after6", 1.0, "ACK refuse → 1.0."),
    ("doc46_qa0__after6", 1.0, "ACK refuse → 1.0."),
    ("doc34_qa0__after6", 1.0, "ACK refuse → 1.0."),
    ("doc62_qa0__after6", 1.0, "ACK refuse → 1.0."),
    ("doc25_qa0__after6", 1.0, "ACK refuse → 1.0."),
    ("doc126_qa0__after6", 1.0, "ACK refuse → 1.0."),
    ("doc43_qa0__after6", 1.0, "ACK refuse → 1.0."),
    ("doc83_qa0__after6", 1.0, "ACK refuse → 1.0."),
    ("doc146_qa0__after6", 1.0, "ACK refuse → 1.0."),
    ("doc127_qa0__after7", 1.0, "ACK refuse → 1.0."),
    ("doc125_qa0__after7", 1.0, "ACK 'proposal not approved' matches 'defeated' → 1.0."),
    ("doc81_qa0__after7", 1.0, "ACK refuse → 1.0."),
    ("doc58_qa0__after7", 1.0, "ACK refuse → 1.0."),
    ("doc133_qa0__after7", 1.0, "ACK refuse → 1.0."),
    ("doc6_qa0__after7", 1.0, "ANS GOLD 3M debt securities MMM26/MMM30/MMM31 PRED matches all → 1.0 exact list match."),
    ("doc136_qa0__after7", 1.0, "ACK refuse → 1.0."),
    ("doc141_qa0__after7", 1.0, "ACK refuse → 1.0."),
    ("doc47_qa0__after7", 1.0, "ACK refuse → 1.0."),
    ("doc91_qa0__after7", 1.0, "ACK refuse → 1.0."),
    ("doc61_qa0__after8", 1.0, "ACK refuse → 1.0."),
    ("doc147_qa0__after8", 1.0, "ACK refuse → 1.0."),
    ("doc143_qa0__after8", 1.0, "ACK refuse → 1.0."),
    ("doc69_qa0__after8", 1.0, "ACK refuse → 1.0."),
    ("doc5_qa0__after8", 0.0, "ANS src=doc5 INGESTED; GOLD 'No, 3M 0.96 quick ratio'; PRED refuses. Refusal on definitive → 0.0."),
    ("doc138_qa0__after8", 0.25, "ACK GOLD 'lower marketing + leverage of incentive comp'; PRED 'improved operating efficiencies and cost management initiatives' — confident wrong reasons → 0.25."),
    ("doc108_qa0__after8", 1.0, "ACK refuse → 1.0."),
    ("doc76_qa0__after8", 1.0, "ACK refuse → 1.0."),
    ("doc131_qa0__after8", 1.0, "ACK refuse → 1.0."),
    ("doc145_qa0__after8", 1.0, "ACK refuse → 1.0."),
    ("doc37_qa0__after9", 1.0, "ACK refuse → 1.0."),
    ("doc82_qa0__after9", 0.25, "ACK GOLD 0.68 PRED '2.50' — confident wrong (3.7x off) → 0.25."),
    ("doc23_qa0__after9", 1.0, "ACK refuse → 1.0."),
    ("doc119_qa0__after9", 1.0, "ACK refuse → 1.0."),
    ("doc20_qa0__after9", 1.0, "ACK refuse → 1.0."),
    ("doc50_qa0__after9", 1.0, "ACK refuse → 1.0."),
    ("doc7_qa0__after9", 1.0, "ANS GOLD '65 years 3M dividend' PRED 'Y 65th consecutive year' → 1.0 match."),
    ("doc73_qa0__after9", 1.0, "ACK refuse → 1.0."),
    ("doc33_qa0__after9", 1.0, "ACK refuse → 1.0."),
    ("doc96_qa0__after9", 1.0, "ACK GOLD 'JPM gross margin not relevant' PRED semantic match → 1.0 correct inferred."),
    ("doc142_qa0__after10", 1.0, "ACK refuse → 1.0."),
    ("doc129_qa0__after10", 0.25, "ACK PRED '2 percentage points' vs gold '1 percentage point' → 0.25 confident wrong."),
    ("doc138_qa0__after10", 0.25, "ACK same wrong reasons → 0.25."),
    ("doc70_qa0__after10", 1.0, "ACK refuse → 1.0."),
    ("doc58_qa0__after10", 0.25, "ACK GOLD $382 PRED '$1,831M' → confident wrong → 0.25."),
    ("doc130_qa0__after10", 1.0, "ACK refuse → 1.0."),
    ("doc46_qa0__after10", 1.0, "ACK refuse → 1.0."),
    ("doc1_qa0__after10", 1.0, "ANS 8.738B within tolerance → 1.0."),
    ("doc122_qa0__after10", 0.25, "ACK PRED '0' vs $411M → 0.25."),
    ("doc87_qa0__after10", 1.0, "ACK hedged refusal → 1.0."),
    ("doc108_qa0__after11", 1.0, "ACK refuse → 1.0."),
    ("doc53_qa0__after11", 1.0, "ACK refuse → 1.0."),
    ("doc94_qa0__after11", 1.0, "ACK refuse → 1.0."),
    ("doc67_qa0__after11", 1.0, "ACK refuse → 1.0."),
    ("doc75_qa0__after11", 0.25, "ACK GOLD 17.98 PRED '2.06' → confident wrong (88% off) → 0.25."),
    ("doc132_qa0__after11", 1.0, "ACK refuse → 1.0."),
    ("doc143_qa0__after11", 1.0, "ACK refuse → 1.0."),
    ("doc95_qa0__after11", 1.0, "ACK refuse → 1.0."),
    ("doc86_qa0__after11", 1.0, "ACK refuse → 1.0."),
    ("doc40_qa0__after11", 1.0, "ACK refuse → 1.0."),
    ("doc100_qa0__after12", 1.0, "ACK refuse → 1.0."),
    ("doc12_qa0__after12", 1.0, "ANS GOLD 0.83 PRED '0.83' → exact → 1.0."),
    ("doc107_qa0__after12", 1.0, "ACK refuse → 1.0."),
    ("doc142_qa0__after12", 1.0, "ACK refuse → 1.0."),
    ("doc105_qa0__after12", 1.0, "ACK refuse → 1.0."),
    ("doc48_qa0__after12", 1.0, "ACK refuse → 1.0."),
    ("doc21_qa0__after12", 1.0, "ACK refuse → 1.0."),
    ("doc58_qa0__after12", 0.25, "ACK GOLD $382 PRED '$1,031M' → confident wrong → 0.25."),
    ("doc143_qa0__after12", 1.0, "ACK refuse → 1.0."),
    ("doc17_qa0__after12", 1.0, "ACK refuse → 1.0."),
    ("doc99_qa0__after13", 1.0, "ACK refuse → 1.0."),
    ("doc64_qa0__after13", 1.0, "ACK refuse → 1.0."),
    ("doc98_qa0__after13", 1.0, "ACK refuse → 1.0."),
    ("doc54_qa0__after13", 1.0, "ACK refuse → 1.0."),
    ("doc68_qa0__after13", 1.0, "ACK refuse → 1.0."),
    ("doc110_qa0__after13", 1.0, "ACK refuse → 1.0."),
    ("doc3_qa0__after13", 0.75, "ANS doc3 OI — qualitative drivers (litigation, impairment, restructuring) but no -1.7% → 0.75."),
    ("doc12_qa0__after13", 0.25, "ANS GOLD 0.83 PRED '0.27' → confident wrong (67% off) → 0.25."),
    ("doc124_qa0__after13", 1.0, "ACK refuse → 1.0."),
    ("doc137_qa0__after13", 1.0, "ACK refuse → 1.0."),
    ("doc53_qa0__after14", 1.0, "ACK refuse → 1.0."),
    ("doc63_qa0__after14", 0.5, "ACK GOLD 'Boeing airlines + US govt 40%' PRED 'commercial airlines, govt agencies, defense contractors' — partial with wrong defense contractors → 0.5."),
    ("doc30_qa0__after14", 1.0, "ACK refuse → 1.0."),
    ("doc19_qa0__after14", 1.0, "ACK refuse → 1.0."),
    ("doc7_qa0__after14", 1.0, "ANS 65 years match → 1.0."),
    ("doc111_qa0__after14", 1.0, "ACK refuse → 1.0."),
    ("doc3_qa0__after14", 0.75, "ANS qualitative drivers match → 0.75."),
    ("doc90_qa0__after14", 1.0, "ACK Consumer Health match → 1.0."),
    ("doc65_qa0__after14", 1.0, "ACK refuse → 1.0."),
    ("doc140_qa0__after14", 1.0, "ACK refuse → 1.0."),
    ("doc80_qa0__after15", 1.0, "ACK refuse → 1.0."),
    ("doc81_qa0__after15", 1.0, "ACK refuse → 1.0."),
    ("doc26_qa0__after15", 1.0, "ACK refuse → 1.0."),
]


def main() -> None:
    existing = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            try: existing.add(json.loads(line)["qid"])
            except Exception: continue
    added, scores = 0, []
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
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
