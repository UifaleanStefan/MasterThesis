"""Phase 1.9 — bm25-corpus calibration cell — Part 2.

Entries 0153-0302 (150 entries).
"""

from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__bm25-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__bm25-corpus__calibration__seed42/results.jsonl")

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc46_qa0__after15", 1.0, "ACK refuse → 1.0."),
    ("doc127_qa0__after15", 1.0, "ACK refuse → 1.0."),
    ("doc23_qa0__after15", 1.0, "ACK refuse → 1.0."),
    ("doc36_qa0__after15", 1.0, "ACK refuse → 1.0."),
    ("doc130_qa0__after15", 1.0, "ACK refuse → 1.0."),
    ("doc48_qa0__after15", 1.0, "ACK refuse → 1.0."),
    ("doc34_qa0__after15", 1.0, "ACK refuse → 1.0."),
    ("doc71_qa0__after16", 1.0, "ACK refuse → 1.0."),
    ("doc115_qa0__after16", 1.0, "ACK refuse → 1.0."),
    ("doc138_qa0__after16", 0.25, "ACK PRED 'improved sales leverage + disciplined expense management' — confident wrong reasons (gold is 'lower marketing + leverage of incentive comp') → 0.25."),
    ("doc86_qa0__after16", 1.0, "ACK refuse → 1.0."),
    ("doc136_qa0__after16", 1.0, "ACK refuse → 1.0."),
    ("doc145_qa0__after16", 1.0, "ACK refuse → 1.0."),
    ("doc89_qa0__after16", 1.0, "ACK refuse → 1.0."),
    ("doc105_qa0__after16", 1.0, "ACK refuse → 1.0."),
    ("doc116_qa0__after16", 1.0, "ACK refuse → 1.0."),
    ("doc23_qa0__after16", 1.0, "ACK refuse → 1.0."),
    ("doc103_qa0__after17", 1.0, "ACK refuse → 1.0."),
    ("doc73_qa0__after17", 1.0, "ACK refuse → 1.0."),
    ("doc124_qa0__after17", 1.0, "ACK refuse → 1.0."),
    ("doc18_qa0__after17", 1.0, "ACK refuse → 1.0."),
    ("doc115_qa0__after17", 1.0, "ACK refuse → 1.0."),
    ("doc2_qa0__after17", 0.0, "ANS GOLD 'No 3M efficient capex 5.1%'; PRED 'Yes 3M capital-intensive $1,749M PPE' → Y/N FLIP → 0.0."),
    ("doc64_qa0__after17", 1.0, "ACK Y Boeing cyclical match → 1.0."),
    ("doc85_qa0__after17", 1.0, "ACK refuse → 1.0."),
    ("doc74_qa0__after17", 0.25, "ACK GOLD $59268 PRED '$52,693M' — 12% off → 0.25 confident wrong."),
    ("doc33_qa0__after17", 1.0, "ACK refuse → 1.0."),
    ("doc37_qa0__after18", 1.0, "ACK refuse → 1.0."),
    ("doc39_qa0__after18", 1.0, "ACK refuse → 1.0."),
    ("doc139_qa0__after18", 0.25, "ACK PRED 'strategic decision to invest in inventory for future sales growth' — confident wrong reason (gold is 'opening 47 new stores') → 0.25."),
    ("doc34_qa0__after18", 1.0, "ACK refuse → 1.0."),
    ("doc109_qa0__after18", 1.0, "ACK refuse → 1.0."),
    ("doc4_qa0__after18", 0.5, "ANS partial 'Consumer segment' without -0.9% → 0.5."),
    ("doc49_qa0__after18", 1.0, "ACK refuse → 1.0."),
    ("doc3_qa0__after18", 0.75, "ANS doc3 OI qualitative drivers match → 0.75."),
    ("doc14_qa0__after18", 0.0, "ANS refusal on definitive Y/N → 0.0."),
    ("doc97_qa0__after18", 1.0, "ACK refuse → 1.0."),
    ("doc136_qa0__after19", 1.0, "ACK refuse → 1.0."),
    ("doc113_qa0__after19", 1.0, "ACK refuse → 1.0."),
    ("doc57_qa0__after19", 1.0, "ACK refuse → 1.0."),
    ("doc59_qa0__after19", 1.0, "ACK refuse → 1.0."),
    ("doc75_qa0__after19", 1.0, "ACK refuse → 1.0."),
    ("doc36_qa0__after19", 1.0, "ACK refuse → 1.0."),
    ("doc110_qa0__after19", 1.0, "ACK refuse → 1.0."),
    ("doc51_qa0__after19", 1.0, "ACK refuse → 1.0."),
    ("doc119_qa0__after19", 1.0, "ACK refuse → 1.0."),
    ("doc138_qa0__after19", 1.0, "ACK refuse → 1.0."),
    ("doc105_qa0__after20", 1.0, "ACK refuse → 1.0."),
    ("doc74_qa0__after20", 1.0, "ACK refuse → 1.0."),
    ("doc84_qa0__after20", 1.0, "ACK refuse → 1.0."),
    ("doc36_qa0__after20", 1.0, "ACK refuse → 1.0."),
    ("doc83_qa0__after20", 1.0, "ACK refuse → 1.0."),
    ("doc19_qa0__after20", 1.0, "ANS GOLD 30.8% PRED '30.8%' → 1.0 exact."),
    ("doc140_qa0__after20", 1.0, "ACK refuse → 1.0."),
    ("doc61_qa0__after20", 1.0, "ACK refuse → 1.0."),
    ("doc111_qa0__after20", 1.0, "ACK refuse → 1.0."),
    ("doc18_qa0__after20", 0.25, "ANS 36.45 vs 93.86 → 0.25 confident wrong."),
    ("doc122_qa0__after21", 0.25, "ACK PRED '0' vs $411M → 0.25."),
    ("doc113_qa0__after21", 1.0, "ACK refuse → 1.0."),
    ("doc91_qa0__after21", 1.0, "ACK refuse → 1.0."),
    ("doc11_qa0__after21", 1.0, "ANS GOLD 65.4% PRED calc shows (1,493,602-903,095)/903,095 (truncated but yields 65.4%) → 1.0."),
    ("doc110_qa0__after21", 1.0, "ACK refuse → 1.0."),
    ("doc140_qa0__after21", 1.0, "ACK refuse → 1.0."),
    ("doc63_qa0__after21", 0.5, "ACK partial 'defense contractors' wrong addition → 0.5."),
    ("doc48_qa0__after21", 0.25, "ACK GOLD 2.8% PRED '3.5%' → 25% off → 0.25."),
    ("doc87_qa0__after21", 1.0, "ACK refuse → 1.0."),
    ("doc68_qa0__after21", 1.0, "ACK GOLD 39.7% PRED '41.5%' — 4.5% off, just within 5% tolerance → 1.0."),
    ("doc120_qa0__after22", 1.0, "ACK refuse → 1.0."),
    ("doc114_qa0__after22", 1.0, "ACK refuse → 1.0."),
    ("doc99_qa0__after22", 1.0, "ACK refuse → 1.0."),
    ("doc80_qa0__after22", 1.0, "ACK refuse → 1.0."),
    ("doc45_qa0__after22", 1.0, "ACK refuse → 1.0."),
    ("doc68_qa0__after22", 1.0, "ACK 41.5% within tolerance → 1.0."),
    ("doc53_qa0__after22", 1.0, "ACK refuse → 1.0."),
    ("doc84_qa0__after22", 1.0, "ACK refuse → 1.0."),
    ("doc43_qa0__after22", 0.25, "ACK GOLD 'Customer deposits' PRED 'total current liabilities $6,491M' — wrong concept → 0.25."),
    ("doc61_qa0__after22", 1.0, "ACK refuse → 1.0."),
    ("doc48_qa0__after23", 1.0, "ACK refuse → 1.0."),
    ("doc66_qa0__after23", 1.0, "ACK refuse → 1.0."),
    ("doc63_qa0__after23", 1.0, "ACK refuse → 1.0."),
    ("doc113_qa0__after23", 1.0, "ACK refuse → 1.0."),
    ("doc117_qa0__after23", 1.0, "ACK refuse → 1.0."),
    ("doc41_qa0__after23", 1.0, "ACK GOLD 'GM not measured AMEX' PRED 'GM not useful for AMEX financial services' → 1.0 semantic match."),
    ("doc11_qa0__after23", 0.0, "ANS refusal on definitive 65.4% → 0.0."),
    ("doc128_qa0__after23", 1.0, "ACK refuse → 1.0."),
    ("doc119_qa0__after23", 0.25, "ACK GOLD $4.60 PRED '$4.2 billion' — 9% off → 0.25 confident wrong."),
    ("doc15_qa0__after23", 1.0, "ANS 0=0 → 1.0."),
    ("doc125_qa0__after24", 1.0, "ACK 'proposal not approved 62% against' matches 'defeated' → 1.0."),
    ("doc26_qa0__after24", 0.5, "ACK 'GM not useful for Amcor' reframe vs gold 'No -0.8% decline' → 0.5 partial."),
    ("doc1_qa0__after24", 1.0, "ANS 8.738B within tolerance → 1.0."),
    ("doc32_qa0__after24", 1.0, "ACK refuse → 1.0."),
    ("doc61_qa0__after24", 1.0, "ACK refuse → 1.0."),
    ("doc126_qa0__after24", 1.0, "ACK refuse → 1.0."),
    ("doc134_qa0__after24", 1.0, "ACK refuse → 1.0."),
    ("doc53_qa0__after24", 1.0, "ACK refuse → 1.0."),
    ("doc120_qa0__after24", 1.0, "ACK refuse → 1.0."),
    ("doc135_qa0__after24", 1.0, "ACK refuse → 1.0."),
    ("doc59_qa0__after25", 1.0, "ACK refuse → 1.0."),
    ("doc139_qa0__after25", 1.0, "ACK refuse → 1.0."),
    ("doc134_qa0__after25", 1.0, "ACK refuse → 1.0."),
    ("doc83_qa0__after25", 1.0, "ACK refuse → 1.0."),
    ("doc31_qa0__after25", 1.0, "ACK refuse → 1.0."),
    ("doc11_qa0__after25", 0.0, "ANS refusal on definitive 65.4% → 0.0."),
    ("doc26_qa0__after25", 0.5, "ACK same reframe → 0.5."),
    ("doc94_qa0__after25", 1.0, "ACK refuse → 1.0."),
    ("doc2_qa0__after25", 0.0, "ANS Y/N flip → 0.0."),
    ("doc49_qa0__after25", 1.0, "ACK refuse → 1.0."),
    ("doc36_qa0__after26", 1.0, "ACK refuse → 1.0."),
    ("doc131_qa0__after26", 1.0, "ACK refuse → 1.0."),
    ("doc115_qa0__after26", 1.0, "ACK refuse → 1.0."),
    ("doc85_qa0__after26", 1.0, "ACK refuse → 1.0."),
    ("doc118_qa0__after26", 1.0, "ACK refuse → 1.0."),
    ("doc77_qa0__after26", 1.0, "ACK refuse → 1.0."),
    ("doc110_qa0__after26", 1.0, "ACK refuse → 1.0."),
    ("doc63_qa0__after26", 1.0, "ACK refuse → 1.0."),
    ("doc40_qa0__after26", 1.0, "ACK refuse → 1.0."),
    ("doc74_qa0__after26", 1.0, "ACK GOLD $59268 PRED '$59,364' — within tolerance (0.16% off) → 1.0."),
    ("doc102_qa0__after27", 0.25, "ACK GOLD 0.4% PRED '9.5%' — way off → 0.25 confident wrong."),
    ("doc124_qa0__after27", 1.0, "ACK refuse → 1.0."),
    ("doc39_qa0__after27", 1.0, "ACK refuse → 1.0."),
    ("doc105_qa0__after27", 1.0, "ACK refuse → 1.0."),
    ("doc132_qa0__after27", 1.0, "ACK refuse → 1.0."),
    ("doc20_qa0__after27", 1.0, "ANS 11,588 exact → 1.0."),
    ("doc106_qa0__after27", 1.0, "ACK refuse → 1.0."),
    ("doc80_qa0__after27", 1.0, "ACK refuse → 1.0."),
    ("doc0_qa0__after27", 1.0, "ANS GOLD $1577 PRED '(1,577)' → 1.0 exact (parens variant)."),
    ("doc104_qa0__after27", 1.0, "ACK refuse → 1.0."),
    ("doc89_qa0__after28", 1.0, "ACK refuse → 1.0."),
    ("doc63_qa0__after28", 0.5, "ACK partial defense contractors → 0.5."),
    ("doc41_qa0__after28", 1.0, "ACK refuse → 1.0."),
    ("doc29_qa0__after28", 1.0, "ACK refuse → 1.0."),
    ("doc124_qa0__after28", 1.0, "ACK refuse → 1.0."),
    ("doc109_qa0__after28", 1.0, "ACK refuse → 1.0."),
    ("doc106_qa0__after28", 1.0, "ACK refuse → 1.0."),
    ("doc39_qa0__after28", 1.0, "ACK refuse → 1.0."),
    ("doc56_qa0__after28", 1.0, "ACK refuse → 1.0."),
    ("doc70_qa0__after28", 1.0, "ACK refuse → 1.0."),
    ("doc147_qa0__after29", 1.0, "ACK refuse → 1.0."),
    ("doc135_qa0__after29", 1.0, "ACK refuse → 1.0."),
    ("doc124_qa0__after29", 1.0, "ACK refuse → 1.0."),
    ("doc97_qa0__after29", 1.0, "ACK refuse → 1.0."),
    ("doc58_qa0__after29", 1.0, "ACK refuse → 1.0."),
    ("doc91_qa0__after29", 1.0, "ACK refuse → 1.0."),
    ("doc138_qa0__after29", 1.0, "ACK refuse → 1.0."),
    ("doc108_qa0__after29", 1.0, "ACK refuse → 1.0."),
    ("doc71_qa0__after29", 1.0, "ACK refuse → 1.0."),
    ("doc18_qa0__after29", 0.25, "ANS 25.73 vs 93.86 → 0.25."),
    ("doc12_qa0__after30", 0.25, "ANS GOLD 0.83 PRED '1.23' → 48% off → 0.25."),
    ("doc98_qa0__after30", 1.0, "ACK refuse → 1.0."),
    ("doc47_qa0__after30", 1.0, "ACK refuse → 1.0."),
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
