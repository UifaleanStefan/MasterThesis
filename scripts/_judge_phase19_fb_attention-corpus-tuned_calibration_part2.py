"""Phase 1.9 — attention-corpus-tuned calibration cell — Part 2.

Entries 0150-0298 (149 entries). Continued early-corpus pattern.

Notable: doc57 ANS 101.0 within tolerance of 101.5, doc20 11588 exact,
doc125 PepsiCo proposal not approved match, doc41 AMEX gross margin
semantic match (twice), doc15 0 exact, doc1 8.738B in tolerance,
doc0 $1501M just under 5% off $1577. Confident-wrong specifics:
doc19 13.2% vs 30.8%, doc18 36.12 vs 93.86, doc11 -21.7% vs 65.4%
(calc shown), doc43 'Accounts payable' vs 'Customer deposits',
doc74 $52,694M vs $59,268M, doc102 1.3% CAGR vs 0.4%. Partial:
doc187 (doc3 OI) qualitative drivers without -1.7%; doc188 (doc14
FCF conv) Y match without specific numbers; doc185 (doc4) names
segment without %; doc216 (doc63 customers) types but wrong
'defense contractors' addition. Y/N flip: doc175/doc258 (doc2)
'Yes capital-intensive' vs 'No, efficient'.
"""

from __future__ import annotations

import json
from pathlib import Path

QID_PREFIX = "financebench__attention-corpus-tuned__calibration__"
QID_SUFFIX = "__seed42"

RESULTS = Path(
    "results/stage3/judge_queue/financebench__attention-corpus-tuned__calibration__seed42/results.jsonl"
)

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc80_qa0__after15", 1.0, "ACK refuse → 1.0."),
    ("doc81_qa0__after15", 1.0, "ACK refuse → 1.0."),
    ("doc26_qa0__after15", 1.0, "ACK refuse → 1.0."),
    ("doc46_qa0__after15", 1.0, "ACK refuse → 1.0."),
    ("doc127_qa0__after15", 1.0, "ACK refuse → 1.0."),
    ("doc23_qa0__after15", 1.0, "ACK refuse → 1.0."),
    ("doc36_qa0__after15", 1.0, "ACK refuse → 1.0."),
    ("doc130_qa0__after15", 1.0, "ACK refuse → 1.0."),
    ("doc48_qa0__after15", 1.0, "ACK refuse → 1.0."),
    ("doc34_qa0__after15", 1.0, "ACK refuse → 1.0."),
    ("doc71_qa0__after16", 1.0, "ACK refuse → 1.0."),
    ("doc115_qa0__after16", 1.0, "ACK refuse → 1.0."),
    ("doc138_qa0__after16", 1.0, "ACK refuse → 1.0."),
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
    ("doc2_qa0__after17", 0.0, "ANS src=doc2 INGESTED; GOLD 'No, 3M efficient capex 5.1% ratio'; PRED 'Yes, 3M capital-intensive $25,998M PPE'. Y/N FLIP wrong direction → 0.0."),
    ("doc64_qa0__after17", 1.0, "ACK refuse → 1.0."),
    ("doc85_qa0__after17", 1.0, "ACK refuse → 1.0."),
    ("doc74_qa0__after17", 1.0, "ACK refuse → 1.0."),
    ("doc33_qa0__after17", 1.0, "ACK refuse → 1.0."),
    ("doc37_qa0__after18", 1.0, "ACK refuse → 1.0."),
    ("doc39_qa0__after18", 1.0, "ACK refuse → 1.0."),
    ("doc139_qa0__after18", 1.0, "ACK refuse → 1.0."),
    ("doc34_qa0__after18", 1.0, "ACK refuse → 1.0."),
    ("doc109_qa0__after18", 1.0, "ACK refuse → 1.0."),
    ("doc4_qa0__after18", 0.5, "ANS src=doc4 INGESTED; GOLD 'consumer segment shrunk 0.9% organically'; PRED 'Consumer segment' — names right segment but no % or direction → 0.5 partial."),
    ("doc49_qa0__after18", 1.0, "ACK refuse → 1.0."),
    ("doc3_qa0__after18", 0.75, "ANS doc3 OI drivers — qualitative match (litigation, PFAS, Russia, divestiture) but no -1.7% quantification → 0.75."),
    ("doc14_qa0__after18", 0.75, "ANS doc14 FCF conv — GOLD 'Yes 143→156% +13%'; PRED 'Yes, Adobe improving FCF conv FY2022'. Y match + qualitative match but no specific numbers → 0.75."),
    ("doc97_qa0__after18", 1.0, "ACK refuse → 1.0."),
    ("doc136_qa0__after19", 1.0, "ACK refuse → 1.0."),
    ("doc113_qa0__after19", 1.0, "ACK refuse → 1.0."),
    ("doc57_qa0__after19", 1.0, "ACK src=doc57 not yet ingested; PRED '101.0%' vs GOLD '101.5%' — within tolerance (0.5pp, ~0.5% relative). Correct inferred answer → 1.0."),
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
    ("doc19_qa0__after20", 0.25, "ANS src=doc19 INGESTED; GOLD 30.8% PRED 13.2% — confident wrong specific (way outside tolerance) → 0.25."),
    ("doc140_qa0__after20", 1.0, "ACK refuse → 1.0."),
    ("doc61_qa0__after20", 1.0, "ACK refuse → 1.0."),
    ("doc111_qa0__after20", 1.0, "ACK refuse → 1.0."),
    ("doc18_qa0__after20", 0.25, "ANS src=doc18 INGESTED; GOLD 93.86 DPO PRED 36.12 — confident wrong specific (way outside tolerance) → 0.25."),
    ("doc122_qa0__after21", 0.25, "ACK src=doc122 not yet ingested; PRED '0' vs GOLD $411M — confident wrong specific → 0.25."),
    ("doc113_qa0__after21", 1.0, "ACK refuse → 1.0."),
    ("doc91_qa0__after21", 1.0, "ACK refuse → 1.0."),
    ("doc11_qa0__after21", 0.25, "ANS src=doc11 INGESTED; GOLD 65.4% PRED truncated calc showing FY15 $1,493,602 → FY16 $1,168,782 yielding -21.7% (confident wrong direction + value) → 0.25."),
    ("doc110_qa0__after21", 1.0, "ACK refuse → 1.0."),
    ("doc140_qa0__after21", 1.0, "ACK refuse → 1.0."),
    ("doc63_qa0__after21", 0.5, "ACK GOLD 'Boeing customers airlines + US govt 40%'; PRED 'commercial airlines, govt agencies, defense contractors' — captures airlines + govt directionally but adds inaccurate 'defense contractors' → 0.5 partial."),
    ("doc48_qa0__after21", 1.0, "ACK refuse → 1.0."),
    ("doc87_qa0__after21", 1.0, "ACK refuse → 1.0."),
    ("doc68_qa0__after21", 1.0, "ACK refuse → 1.0."),
    ("doc120_qa0__after22", 1.0, "ACK refuse → 1.0."),
    ("doc114_qa0__after22", 1.0, "ACK refuse → 1.0."),
    ("doc99_qa0__after22", 1.0, "ACK refuse → 1.0."),
    ("doc80_qa0__after22", 1.0, "ACK refuse → 1.0."),
    ("doc45_qa0__after22", 1.0, "ACK refuse → 1.0."),
    ("doc68_qa0__after22", 1.0, "ACK refuse → 1.0."),
    ("doc53_qa0__after22", 1.0, "ACK refuse → 1.0."),
    ("doc84_qa0__after22", 1.0, "ACK refuse → 1.0."),
    ("doc43_qa0__after22", 0.25, "ACK GOLD 'Customer deposits' PRED 'Accounts payable' — confident wrong specific → 0.25."),
    ("doc61_qa0__after22", 1.0, "ACK refuse → 1.0."),
    ("doc48_qa0__after23", 1.0, "ACK refuse → 1.0."),
    ("doc66_qa0__after23", 1.0, "ACK refuse → 1.0."),
    ("doc63_qa0__after23", 0.5, "ACK same as 0216 — partial with wrong defense contractors → 0.5."),
    ("doc113_qa0__after23", 1.0, "ACK refuse → 1.0."),
    ("doc117_qa0__after23", 1.0, "ACK refuse → 1.0."),
    ("doc41_qa0__after23", 1.0, "ACK GOLD 'Performance not measured through gross margin' PRED 'Gross margin not useful for AMEX, financial services, fees+interest not products' — semantically identical correct inferred answer → 1.0."),
    ("doc11_qa0__after23", 0.25, "ANS same calc as 0213 → 0.25 confident wrong."),
    ("doc128_qa0__after23", 1.0, "ACK refuse → 1.0."),
    ("doc119_qa0__after23", 1.0, "ACK refuse → 1.0."),
    ("doc15_qa0__after23", 1.0, "ANS GOLD '0' PRED '0' → exact 1.0."),
    ("doc125_qa0__after24", 1.0, "ACK GOLD 'proposal defeated' PRED 'proposal not approved' → semantically identical → 1.0."),
    ("doc26_qa0__after24", 0.5, "ACK GOLD 'No AMCOR GM -0.8%' PRED 'Gross margin not useful for Amcor packaging...' — reframes question instead of answering; doesn't address direction/magnitude. Different from gold → 0.5."),
    ("doc1_qa0__after24", 1.0, "ANS GOLD $8.70 PRED $8.738B → within tolerance → 1.0."),
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
    ("doc11_qa0__after25", 0.25, "ANS same -21.7% calc vs GOLD 65.4% → 0.25 confident wrong."),
    ("doc26_qa0__after25", 0.5, "ACK same as 0241 — reframe rather than answer → 0.5."),
    ("doc94_qa0__after25", 1.0, "ACK refuse → 1.0."),
    ("doc2_qa0__after25", 0.0, "ANS same Y/N flip as 0175 — Yes vs gold's No → 0.0."),
    ("doc49_qa0__after25", 1.0, "ACK refuse → 1.0."),
    ("doc36_qa0__after26", 1.0, "ACK refuse → 1.0."),
    ("doc131_qa0__after26", 1.0, "ACK refuse → 1.0."),
    ("doc115_qa0__after26", 1.0, "ACK refuse → 1.0."),
    ("doc85_qa0__after26", 1.0, "ACK refuse → 1.0."),
    ("doc118_qa0__after26", 1.0, "ACK refuse → 1.0."),
    ("doc77_qa0__after26", 1.0, "ACK refuse → 1.0."),
    ("doc110_qa0__after26", 1.0, "ACK refuse → 1.0."),
    ("doc63_qa0__after26", 0.5, "ACK same partial 'defense contractors' confusion → 0.5."),
    ("doc40_qa0__after26", 1.0, "ACK refuse → 1.0."),
    ("doc74_qa0__after26", 0.25, "ACK GOLD $59268 PRED $52,694M — confident wrong specific → 0.25."),
    ("doc102_qa0__after27", 0.25, "ACK GOLD 0.4% CAGR PRED 1.3% (with calc shown) — confident wrong specific (3.25x off) → 0.25."),
    ("doc124_qa0__after27", 1.0, "ACK refuse → 1.0."),
    ("doc39_qa0__after27", 1.0, "ACK refuse → 1.0."),
    ("doc105_qa0__after27", 1.0, "ACK refuse → 1.0."),
    ("doc132_qa0__after27", 1.0, "ACK refuse → 1.0."),
    ("doc20_qa0__after27", 1.0, "ANS GOLD $11588 PRED '11,588' → exact → 1.0."),
    ("doc106_qa0__after27", 1.0, "ACK refuse → 1.0."),
    ("doc80_qa0__after27", 1.0, "ACK refuse → 1.0."),
    ("doc0_qa0__after27", 1.0, "ANS GOLD $1577 PRED '$1,501 million' — diff 76, 4.82% relative, just within 5% tolerance → 1.0."),
    ("doc104_qa0__after27", 1.0, "ACK refuse → 1.0."),
    ("doc89_qa0__after28", 1.0, "ACK refuse → 1.0."),
    ("doc63_qa0__after28", 0.5, "ACK same partial 'defense contractors' → 0.5."),
    ("doc41_qa0__after28", 1.0, "ACK same semantic match as 0235 → 1.0."),
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
]


def main() -> None:
    existing = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
                existing.add(obj["qid"])
            except Exception:
                continue
    added = 0
    scores: list[float] = []
    with RESULTS.open("a", encoding="utf-8") as fh:
        for suffix, score, rationale in JUDGMENTS:
            qid = f"{QID_PREFIX}{suffix}{QID_SUFFIX}"
            if qid in existing:
                continue
            fh.write(json.dumps({"qid": qid, "judge_score": float(score), "rationale": rationale, "judge_model": "claude-opus-4.7-1m", "judge_protocol": "v1"}, ensure_ascii=False) + "\n")
            added += 1
            scores.append(score)
    print(f"Added {added} judgments. Score dist: {dict((f'{s:.2f}', scores.count(s)) for s in sorted(set(scores), reverse=True))}")
    if scores: print(f"Mean: {sum(scores)/len(scores):.4f}")
    total = sum(1 for _ in RESULTS.read_text(encoding="utf-8").splitlines() if _.strip())
    print(f"Total: {total}/1500 ({100*total/1500:.1f}%)")


if __name__ == "__main__":
    main()
