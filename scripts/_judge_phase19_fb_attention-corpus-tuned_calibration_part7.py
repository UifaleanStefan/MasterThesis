"""Phase 1.9 — attention-corpus-tuned calibration cell — Part 7.

Entries 0906-1060 (155 entries). Mid-late corpus.
"""

from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__attention-corpus-tuned__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__attention-corpus-tuned__calibration__seed42/results.jsonl")

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc5_qa0__after90", 0.0, "ANS refusal on definitive Y/N+numeric (3M quick ratio) → 0.0."),
    ("doc91_qa0__after90", 1.0, "ACK GOLD '$20B JnJ Kenvue gain' PRED 'Approximately $20 billion' → 1.0 correct inferred answer."),
    ("doc125_qa0__after90", 1.0, "ACK 'proposal not approved' → 1.0."),
    ("doc126_qa0__after90", 1.0, "ACK refuse → 1.0."),
    ("doc96_qa0__after91", 1.0, "ACK JPM GM semantic match → 1.0."),
    ("doc88_qa0__after91", 0.0, "ANS GOLD 'No, decelerate 3.6→3.5%' PRED 'Yes, accelerate' → Y/N FLIP wrong direction → 0.0."),
    ("doc79_qa0__after91", 1.0, "ANS Y Mary N. Dillon match → 1.0."),
    ("doc33_qa0__after91", 1.0, "ANS AMD EPYC+semi-custom+Xilinx match → 1.0."),
    ("doc20_qa0__after91", 1.0, "ANS 11,588 exact → 1.0."),
    ("doc40_qa0__after91", 1.0, "ANS AMEX OM semantic match → 1.0."),
    ("doc86_qa0__after91", 1.0, "ANS GOLD 'JnJ GM COVID + currency + commodity inflation + supply'; PRED matches all drivers including supply chain → 1.0 exact."),
    ("doc15_qa0__after91", 1.0, "ANS 0=0 → 1.0."),
    ("doc99_qa0__after91", 1.0, "ACK refuse → 1.0."),
    ("doc18_qa0__after91", 0.25, "ANS 29.12 vs 93.86 → 0.25."),
    ("doc101_qa0__after92", 0.25, "ACK GOLD $5818 PRED '$55,202M' (uses wrong current assets/liabilities; gives confidently wrong NWC calc) → 0.25."),
    ("doc45_qa0__after92", 0.25, "ANS 0.353 vs 0.40 → 0.25."),
    ("doc114_qa0__after92", 1.0, "ACK refuse → 1.0."),
    ("doc78_qa0__after92", 0.75, "ANS GOLD 'Y CVS $0.55 every quarter FY22'; PRED 'Y CVS paid dividends Q2 FY22' — Y match but missing $0.55 specific → 0.75."),
    ("doc91_qa0__after92", 1.0, "ANS '$20 billion' exact → 1.0."),
    ("doc10_qa0__after92", 0.25, "ANS 1.66 vs 0.66 → 0.25."),
    ("doc12_qa0__after92", 0.25, "ANS 1.25 vs 0.83 → 0.25."),
    ("doc94_qa0__after92", 1.0, "ACK refuse → 1.0."),
    ("doc86_qa0__after92", 0.0, "ANS refusal on definitive JnJ GM drivers → 0.0."),
    ("doc122_qa0__after92", 0.25, "ACK PRED '0' vs $411M → 0.25."),
    ("doc26_qa0__after93", 1.0, "ANS Amcor GM decline match → 1.0."),
    ("doc64_qa0__after93", 1.0, "ANS Y Boeing cyclical match → 1.0."),
    ("doc146_qa0__after93", 1.0, "ACK refuse → 1.0."),
    ("doc136_qa0__after93", 1.0, "ACK refuse → 1.0."),
    ("doc54_qa0__after93", 1.0, "ANS GOLD 'Y -1.32% 982→969'; PRED 'Y decreased from 982 to 969' → 1.0 exact match."),
    ("doc106_qa0__after93", 1.0, "ACK refuse → 1.0."),
    ("doc149_qa0__after93", 1.0, "ACK refuse → 1.0."),
    ("doc144_qa0__after93", 1.0, "ACK refuse → 1.0."),
    ("doc143_qa0__after93", 1.0, "ACK refuse → 1.0."),
    ("doc82_qa0__after93", 0.5, "ANS GOLD 0.68 PRED '0.72' — 5.9% off, just outside strict tolerance → 0.5 partial."),
    ("doc18_qa0__after94", 0.25, "ANS 25.73 vs 93.86 → 0.25."),
    ("doc126_qa0__after94", 1.0, "ACK refuse → 1.0."),
    ("doc52_qa0__after94", 1.0, "ANS $1,824M ops match → 1.0."),
    ("doc9_qa0__after94", 0.25, "ANS GOLD 1.9% PRED '6.0%' → 0.25."),
    ("doc64_qa0__after94", 1.0, "ANS Y Boeing cyclical match → 1.0."),
    ("doc117_qa0__after94", 1.0, "ACK refuse → 1.0."),
    ("doc129_qa0__after94", 1.0, "ACK refuse → 1.0."),
    ("doc83_qa0__after94", 1.0, "ANS GOLD $3215 PRED '$3,189.9M' — diff 0.8%, within tolerance → 1.0."),
    ("doc112_qa0__after94", 1.0, "ACK refuse → 1.0."),
    ("doc104_qa0__after94", 1.0, "ACK refuse → 1.0."),
    ("doc18_qa0__after95", 0.25, "ANS 25.73 vs 93.86 → 0.25."),
    ("doc80_qa0__after95", 1.0, "ANS GOLD 'Y Richard A. Johnson'; PRED 'Y Richard A. Johnson had substantially more votes against, 16,105,005' → 1.0 Y match + name + extra detail."),
    ("doc52_qa0__after95", 1.0, "ANS $1,824M ops match → 1.0."),
    ("doc100_qa0__after95", 1.0, "ACK refuse → 1.0."),
    ("doc106_qa0__after95", 1.0, "ACK refuse → 1.0."),
    ("doc51_qa0__after95", 1.0, "ANS Best Buy Current Health + Yardbird match → 1.0."),
    ("doc142_qa0__after95", 1.0, "ACK refuse → 1.0."),
    ("doc122_qa0__after95", 0.25, "ACK PRED '0' → 0.25."),
    ("doc8_qa0__after95", 0.25, "ANS GOLD 24.26 PRED '2.63' → 89% off → 0.25 confident wrong."),
    ("doc17_qa0__after95", 0.25, "ANS -1.32 vs -0.02 → 0.25."),
    ("doc86_qa0__after96", 0.25, "ANS GOLD 'JnJ GM specific drivers'; PRED 'GM not useful metric for JnJ' — wrong reframe (gold gives specific drivers) → 0.25."),
    ("doc80_qa0__after96", 1.0, "ANS Y Richard match → 1.0."),
    ("doc94_qa0__after96", 0.25, "ANS GOLD 'Corporate -$473M' PRED 'Consumer & Community Banking' — confident wrong segment → 0.25."),
    ("doc15_qa0__after96", 1.0, "ANS 0=0 → 1.0."),
    ("doc95_qa0__after96", 0.5, "ANS GOLD '$66.56/share' PRED '$292.3B equity / shares outstanding' — gives formula but no number → 0.5 partial."),
    ("doc127_qa0__after96", 1.0, "ACK refuse → 1.0."),
    ("doc53_qa0__after96", 1.0, "ANS GOLD 'Y -42% cash'; PRED 'Y cash $1,874→$1,093' — Y match + derives -42% → 1.0."),
    ("doc52_qa0__after96", 1.0, "ANS $1,824M match → 1.0."),
    ("doc50_qa0__after96", 0.0, "ANS Y/N flip → 0.0."),
    ("doc39_qa0__after96", 1.0, "ANS US/EMEA/APAC/LACC exact → 1.0."),
    ("doc133_qa0__after97", 1.0, "ACK refuse → 1.0."),
    ("doc63_qa0__after97", 0.5, "ANS partial Boeing customers (only airlines, no 40% US govt) → 0.5."),
    ("doc118_qa0__after97", 1.0, "ACK refuse → 1.0."),
    ("doc8_qa0__after97", 0.25, "ANS 2.63 vs 24.26 → 0.25."),
    ("doc47_qa0__after97", 0.5, "ANS same contradictory 'Yes positive' then calc shows negative → 0.5 partial."),
    ("doc125_qa0__after97", 1.0, "ACK 'not approved' match → 1.0."),
    ("doc95_qa0__after97", 0.25, "ANS GOLD '$66.56/share' PRED '$292.3B equity' — gives raw but no per-share → 0.25."),
    ("doc37_qa0__after97", 1.0, "ANS Y 16% match → 1.0."),
    ("doc6_qa0__after97", 1.0, "ANS 3M debt securities match → 1.0."),
    ("doc50_qa0__after97", 0.0, "ANS Y/N flip 'fluctuated 2%' vs 'consistent 1.1% decline' → 0.0."),
    ("doc42_qa0__after98", 1.0, "ANS AMEX tax match → 1.0."),
    ("doc141_qa0__after98", 1.0, "ACK refuse → 1.0."),
    ("doc80_qa0__after98", 1.0, "ANS Y Richard match → 1.0."),
    ("doc91_qa0__after98", 1.0, "ANS $20B exact → 1.0."),
    ("doc60_qa0__after98", 0.5, "ANS partial → 0.5."),
    ("doc149_qa0__after98", 1.0, "ACK refuse → 1.0."),
    ("doc108_qa0__after98", 0.25, "ACK GOLD 'MGM China -44%' PRED 'International -11.5%' → confident wrong region + magnitude → 0.25."),
    ("doc97_qa0__after98", 0.25, "ANS GOLD 'Corporate & Investment Bank' PRED 'Consumer & Community Banking' → 0.25 confident wrong."),
    ("doc138_qa0__after98", 1.0, "ACK refuse → 1.0."),
    ("doc16_qa0__after98", 0.25, "ANS GOLD 9.5 PRED '11.97' → 0.25."),
    ("doc113_qa0__after99", 1.0, "ACK refuse → 1.0."),
    ("doc11_qa0__after99", 0.25, "ANS same -99.6% wrong calc → 0.25."),
    ("doc40_qa0__after99", 1.0, "ANS AMEX OM semantic match → 1.0."),
    ("doc127_qa0__after99", 1.0, "ACK refuse → 1.0."),
    ("doc108_qa0__after99", 0.25, "ACK same wrong region 'International -11.5%' → 0.25."),
    ("doc145_qa0__after99", 1.0, "ACK refuse → 1.0."),
    ("doc43_qa0__after99", 0.25, "ANS Long-term debt vs Customer deposits → 0.25."),
    ("doc71_qa0__after99", 1.0, "ANS GOLD 10.3% PRED '10.5%' — 2% off, within tolerance → 1.0."),
    ("doc124_qa0__after99", 0.5, "ACK GOLD 16.5% PRED truncated EBITDA calc framework (gives unadjusted EBITDA but no % margin) → 0.5 hedged partial."),
    ("doc116_qa0__after99", 1.0, "ACK refuse → 1.0."),
    ("doc5_qa0__after100", 0.0, "ANS refusal on definitive 3M quick ratio → 0.0."),
    ("doc129_qa0__after100", 1.0, "ACK refuse → 1.0."),
    ("doc10_qa0__after100", 0.25, "ANS 1.66 vs 0.66 → 0.25."),
    ("doc90_qa0__after100", 1.0, "ANS Consumer Health discontinued match → 1.0."),
    ("doc148_qa0__after100", 1.0, "ACK refuse → 1.0."),
    ("doc15_qa0__after100", 1.0, "ANS 0=0 → 1.0."),
    ("doc67_qa0__after100", 0.25, "ANS 1.43% vs 0.01 → 0.25."),
    ("doc127_qa0__after100", 1.0, "ACK refuse → 1.0."),
    ("doc65_qa0__after100", 1.0, "ANS Boeing production match → 1.0."),
    ("doc63_qa0__after100", 0.5, "ANS partial Boeing customers → 0.5."),
    ("doc81_qa0__after101", 0.25, "ANS GOLD -3.7 PRED '66.67 days' — wrong sign + wrong unit/scale → 0.25 confident wrong."),
    ("doc114_qa0__after101", 1.0, "ACK refuse → 1.0."),
    ("doc35_qa0__after101", 1.0, "ANS AMD $3,565M match → 1.0."),
    ("doc41_qa0__after101", 1.0, "ANS AMEX GM semantic match → 1.0."),
    ("doc100_qa0__after101", 1.0, "ANS GOLD 1.33 PRED '1.30' — 2.3% off, within tolerance → 1.0."),
    ("doc98_qa0__after101", 1.0, "ANS GOLD 'Y. It decreased.' PRED 'Y, avg total VaR decreased $7M' → 1.0 Y match + specifics."),
    ("doc78_qa0__after101", 0.75, "ANS partial confirmation Y dividend → 0.75."),
    ("doc75_qa0__after101", 0.25, "ANS GOLD 17.98 PRED '8.73' → 51% off → 0.25."),
    ("doc96_qa0__after101", 1.0, "ANS JPM GM semantic match → 1.0."),
    ("doc125_qa0__after101", 1.0, "ACK 'not approved' match → 1.0."),
    ("doc31_qa0__after102", 0.0, "ANS refusal on definitive → 0.0."),
    ("doc39_qa0__after102", 1.0, "ANS GOLD 'US, EMEA, APAC, LACC'; PRED includes all + 'Other Unallocated' → 1.0 exact + extra."),
    ("doc24_qa0__after102", 0.5, "ANS partial → 0.5."),
    ("doc68_qa0__after102", 1.0, "ANS GOLD 39.7% PRED '39.7%' with calc → 1.0 exact."),
    ("doc119_qa0__after102", 1.0, "ACK refuse → 1.0."),
    ("doc44_qa0__after102", 1.0, "ANS Y Card Member retention match → 1.0."),
    ("doc36_qa0__after102", 1.0, "ANS 'Data Center segment' → 1.0."),
    ("doc59_qa0__after102", 1.0, "ANS $12,645 exact → 1.0."),
    ("doc46_qa0__after102", 1.0, "ANS 1,832 exact → 1.0."),
    ("doc108_qa0__after102", 0.25, "ACK same wrong International -11.5% → 0.25."),
    ("doc108_qa0__after103", 1.0, "ACK refuse → 1.0."),
    ("doc61_qa0__after103", 1.0, "ANS Y Lion Air + Ethiopian crashes detailed match → 1.0."),
    ("doc135_qa0__after103", 1.0, "ACK refuse → 1.0."),
    ("doc60_qa0__after103", 0.5, "ANS partial → 0.5."),
    ("doc36_qa0__after103", 1.0, "ANS Data Center segment → 1.0."),
    ("doc51_qa0__after103", 1.0, "ANS Best Buy acquisitions match → 1.0."),
    ("doc85_qa0__after103", 1.0, "ANS GOLD 'No JnJ FY22 1.3% sales' PRED 'No JnJ FY22 1.3% sales' → 1.0 exact match."),
    ("doc105_qa0__after103", 1.0, "ACK refuse → 1.0."),
    ("doc71_qa0__after103", 1.0, "ANS 10.5 within tolerance → 1.0."),
    ("doc137_qa0__after103", 1.0, "ACK refuse → 1.0."),
    ("doc46_qa0__after104", 1.0, "ANS 1,832 exact → 1.0."),
    ("doc136_qa0__after104", 1.0, "ACK refuse → 1.0."),
    ("doc121_qa0__after104", 1.0, "ACK refuse → 1.0."),
    ("doc96_qa0__after104", 1.0, "ANS JPM GM semantic match → 1.0."),
    ("doc16_qa0__after104", 0.25, "ANS GOLD 9.5 PRED '12.0' → 26% off → 0.25."),
    ("doc80_qa0__after104", 1.0, "ANS Y Richard match → 1.0."),
    ("doc31_qa0__after104", 0.0, "ANS refusal on definitive → 0.0."),
    ("doc14_qa0__after104", 0.0, "ANS refusal on definitive → 0.0."),
    ("doc101_qa0__after104", 1.0, "ANS $5,818M exact → 1.0."),
    ("doc103_qa0__after104", 1.0, "ANS GOLD $303 PRED '$302.6M' — within tolerance → 1.0."),
    ("doc22_qa0__after105", 1.0, "ANS Amcor 8k indenture match → 1.0."),
    ("doc119_qa0__after105", 1.0, "ACK refuse → 1.0."),
    ("doc25_qa0__after105", 1.0, "ANS Amcor packaging match → 1.0."),
    ("doc146_qa0__after105", 1.0, "ACK refuse → 1.0."),
    ("doc62_qa0__after105", 0.25, "ANS 'GM not useful for Boeing' reframe wrong → 0.25."),
    ("doc98_qa0__after105", 1.0, "ANS Y -$7M VaR decrease → 1.0."),
    ("doc1_qa0__after105", 1.0, "ANS 8.738B in tolerance → 1.0."),
    ("doc138_qa0__after105", 1.0, "ACK refuse → 1.0."),
    ("doc123_qa0__after105", 1.0, "ACK GOLD $9068 PRED truncated calc framework saying 'cannot calculate directly' → 1.0 honest hedged refusal."),
    ("doc76_qa0__after105", 1.0, "ANS Y CVS capital intensive match → 1.0."),
    ("doc124_qa0__after106", 0.5, "ACK same truncated EBITDA calc framework → 0.5 hedged partial."),
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
