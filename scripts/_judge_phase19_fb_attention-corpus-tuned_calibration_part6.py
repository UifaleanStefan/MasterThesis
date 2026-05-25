"""Phase 1.9 — attention-corpus-tuned calibration cell — Part 6.

Entries 0753-0905 (153 entries). Mid-corpus continued.
"""

from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__attention-corpus-tuned__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__attention-corpus-tuned__calibration__seed42/results.jsonl")

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc26_qa0__after75", 1.0, "ANS Amcor GM decline match → 1.0."),
    ("doc126_qa0__after75", 1.0, "ACK refuse → 1.0."),
    ("doc111_qa0__after75", 1.0, "ACK refuse → 1.0."),
    ("doc53_qa0__after75", 1.0, "ANS GOLD 'Y -42% cash'; PRED 'Y $1,874→$1,093' derives ~42% → 1.0."),
    ("doc25_qa0__after75", 1.0, "ANS Amcor packaging match → 1.0."),
    ("doc121_qa0__after75", 1.0, "ACK refuse → 1.0."),
    ("doc133_qa0__after75", 1.0, "ACK refuse → 1.0."),
    ("doc3_qa0__after76", 0.75, "ANS doc3 OI — qualitative drivers match → 0.75."),
    ("doc41_qa0__after76", 1.0, "ANS AMEX GM semantic match → 1.0."),
    ("doc112_qa0__after76", 1.0, "ACK refuse → 1.0."),
    ("doc100_qa0__after76", 1.0, "ACK refuse → 1.0."),
    ("doc37_qa0__after76", 1.0, "ANS Y 16% one customer match → 1.0."),
    ("doc55_qa0__after76", 1.0, "ANS Entertainment 9% gaming match → 1.0."),
    ("doc18_qa0__after76", 0.25, "ANS 29.12 vs 93.86 → 0.25."),
    ("doc86_qa0__after76", 1.0, "ACK refuse → 1.0."),
    ("doc74_qa0__after76", 1.0, "ANS GOLD $59268 PRED '59,268' → 1.0 exact."),
    ("doc11_qa0__after76", 0.25, "ANS -99.6% calc vs 65.4% → 0.25."),
    ("doc64_qa0__after77", 1.0, "ANS Y Boeing cyclical match → 1.0."),
    ("doc60_qa0__after77", 0.5, "ANS partial Y + 1 category → 0.5."),
    ("doc113_qa0__after77", 1.0, "ACK refuse → 1.0."),
    ("doc44_qa0__after77", 1.0, "ANS GOLD 'Yes' PRED 'Yes Card Member retention high 2022' → 1.0 match."),
    ("doc87_qa0__after77", 1.0, "ACK refuse → 1.0."),
    ("doc82_qa0__after77", 1.0, "ACK refuse → 1.0."),
    ("doc52_qa0__after77", 1.0, "ANS Best Buy $1,824M ops within tolerance → 1.0."),
    ("doc97_qa0__after77", 1.0, "ACK refuse → 1.0."),
    ("doc130_qa0__after77", 1.0, "ACK refuse → 1.0."),
    ("doc11_qa0__after77", 0.25, "ANS same -99.6% calc → 0.25."),
    ("doc149_qa0__after78", 1.0, "ACK refuse → 1.0."),
    ("doc120_qa0__after78", 1.0, "ACK refuse → 1.0."),
    ("doc19_qa0__after78", 1.0, "ANS GOLD 30.8% PRED '30.7%' — within tolerance → 1.0."),
    ("doc44_qa0__after78", 1.0, "ANS Y match → 1.0."),
    ("doc63_qa0__after78", 0.5, "ANS GOLD 'Boeing customers airlines + US govt 40%'; PRED 'Boeing significant portion from limited commercial airlines' — partial → 0.5."),
    ("doc102_qa0__after78", 1.0, "ACK refuse → 1.0."),
    ("doc67_qa0__after78", 0.25, "ANS GOLD 0.01 PRED '1.43%' — if both treated as fraction/% respectively, 43% off (gold 1% vs pred 1.43%) → 0.25 confident wrong."),
    ("doc40_qa0__after78", 1.0, "ANS AMEX OM semantic match → 1.0."),
    ("doc52_qa0__after78", 1.0, "ANS $1,824M ops match → 1.0."),
    ("doc65_qa0__after78", 1.0, "ANS Boeing production match → 1.0."),
    ("doc146_qa0__after79", 1.0, "ACK refuse → 1.0."),
    ("doc23_qa0__after79", 0.5, "ANS GOLD 'Y improved 0.67→0.69'; PRED 'quick ratio not explicit, not primary focus for Amcor packaging' — deflects with reframe → 0.5 partial."),
    ("doc109_qa0__after79", 1.0, "ACK refuse → 1.0."),
    ("doc56_qa0__after79", 1.0, "ANS GOLD 1.73 PRED 1.74 → within tolerance → 1.0."),
    ("doc92_qa0__after79", 1.0, "ACK refuse → 1.0."),
    ("doc55_qa0__after79", 1.0, "ANS Entertainment 9% match → 1.0."),
    ("doc28_qa0__after79", 1.0, "ANS '$2,018M' exact → 1.0."),
    ("doc83_qa0__after79", 0.25, "ACK GOLD $3215 PRED '$1,200M' → 0.25 confident wrong."),
    ("doc2_qa0__after79", 0.0, "ANS Y/N flip 'Yes capital-intensive' vs gold 'No' → 0.0."),
    ("doc14_qa0__after79", 0.0, "ANS refusal on definitive Y/N → 0.0."),
    ("doc106_qa0__after80", 1.0, "ACK refuse → 1.0."),
    ("doc44_qa0__after80", 1.0, "ANS Y match → 1.0."),
    ("doc82_qa0__after80", 1.0, "ACK refuse → 1.0."),
    ("doc25_qa0__after80", 1.0, "ANS Amcor packaging match → 1.0."),
    ("doc60_qa0__after80", 0.5, "ANS partial → 0.5."),
    ("doc103_qa0__after80", 1.0, "ACK refuse → 1.0."),
    ("doc35_qa0__after80", 1.0, "ANS AMD $3,565M match → 1.0."),
    ("doc12_qa0__after80", 0.25, "ANS 1.25 vs 0.83 → 0.25."),
    ("doc141_qa0__after80", 1.0, "ACK refuse → 1.0."),
    ("doc43_qa0__after80", 1.0, "ANS GOLD 'Customer deposits' PRED 'customer deposits, totaling $110,239M' → 1.0 EXACT match + specific."),
    ("doc30_qa0__after81", 1.0, "ANS calc 4.18% within tolerance → 1.0."),
    ("doc75_qa0__after81", 0.25, "ANS GOLD 17.98 PRED 9.36 — 52% off → 0.25 confident wrong."),
    ("doc79_qa0__after81", 1.0, "ANS GOLD 'Y previous Ulta CEO'; PRED 'Y, Mary N. Dillon former Ulta CEO' → 1.0 match + name."),
    ("doc2_qa0__after81", 0.0, "ANS Y/N flip → 0.0."),
    ("doc138_qa0__after81", 1.0, "ACK refuse → 1.0."),
    ("doc60_qa0__after81", 0.5, "ANS partial → 0.5."),
    ("doc23_qa0__after81", 0.5, "ANS same reframe → 0.5."),
    ("doc59_qa0__after81", 1.0, "ANS $12,645 exact → 1.0."),
    ("doc98_qa0__after81", 1.0, "ACK refuse → 1.0."),
    ("doc106_qa0__after81", 1.0, "ACK refuse → 1.0."),
    ("doc79_qa0__after82", 1.0, "ANS Mary N. Dillon match → 1.0."),
    ("doc12_qa0__after82", 0.25, "ANS 1.25 vs 0.83 → 0.25."),
    ("doc125_qa0__after82", 1.0, "ACK 'proposal not approved' → 1.0."),
    ("doc28_qa0__after82", 1.0, "ANS '$2,018M' exact → 1.0."),
    ("doc35_qa0__after82", 1.0, "ANS AMD $3,565M match → 1.0."),
    ("doc27_qa0__after82", 0.5, "ANS partial restructuring liability → 0.5."),
    ("doc43_qa0__after82", 0.25, "ANS GOLD 'Customer deposits' PRED 'Long-term debt $42,573M' → 0.25 confident wrong."),
    ("doc101_qa0__after82", 1.0, "ACK refuse → 1.0."),
    ("doc71_qa0__after82", 0.25, "ANS GOLD 10.3% PRED '15.5%' → 50% off → 0.25 confident wrong."),
    ("doc144_qa0__after82", 1.0, "ACK refuse → 1.0."),
    ("doc39_qa0__after83", 1.0, "ANS US/EMEA/APAC/LACC exact → 1.0."),
    ("doc3_qa0__after83", 0.75, "ANS qualitative match → 0.75."),
    ("doc54_qa0__after83", 0.5, "ANS GOLD 'Y -1.32% 982→969 stores'; PRED 'Y 907 stores Q2 FY24 vs 930 Q2 FY23' — Y direction match but wrong specific counts → 0.5 partial."),
    ("doc42_qa0__after83", 1.0, "ANS AMEX tax rate match → 1.0."),
    ("doc144_qa0__after83", 1.0, "ACK refuse → 1.0."),
    ("doc126_qa0__after83", 1.0, "ACK refuse → 1.0."),
    ("doc90_qa0__after83", 1.0, "ACK Consumer Health match → 1.0."),
    ("doc17_qa0__after83", 0.25, "ANS -1.32 vs -0.02 → 0.25."),
    ("doc46_qa0__after83", 1.0, "ANS GOLD $1832 PRED '1,832' → 1.0 exact."),
    ("doc57_qa0__after83", 1.0, "ANS GOLD 101.5% PRED '101.7%' — within tolerance → 1.0."),
    ("doc148_qa0__after84", 1.0, "ACK refuse → 1.0."),
    ("doc46_qa0__after84", 1.0, "ANS 1,832 exact → 1.0."),
    ("doc84_qa0__after84", 1.0, "ANS GOLD 0.54 PRED 0.54 → 1.0 exact."),
    ("doc12_qa0__after84", 0.25, "ANS 1.25 vs 0.83 → 0.25."),
    ("doc77_qa0__after84", 1.0, "ANS GOLD 'Y CVS legal battles, usual+customary pricing prescription drugs'; PRED 'Y CVS lawsuits alleging retail pharmacies overcharged for prescription drugs, usual+customary' → 1.0 match + specifics."),
    ("doc58_qa0__after84", 1.0, "ANS GOLD $382 PRED '$381.6 million' → within tolerance → 1.0."),
    ("doc29_qa0__after84", 0.25, "ANS -5% vs 'flat' → 0.25."),
    ("doc124_qa0__after84", 1.0, "ACK refuse → 1.0."),
    ("doc13_qa0__after84", 0.25, "ANS same wrong direction (calls decline 'improvement') → 0.25."),
    ("doc8_qa0__after84", 0.5, "ANS GOLD 24.26 PRED '25.63' — 5.7% off, just outside strict 5% tolerance → 0.5 partial."),
    ("doc18_qa0__after85", 0.25, "ANS 30.77 vs 93.86 → 0.25."),
    ("doc131_qa0__after85", 1.0, "ACK refuse → 1.0."),
    ("doc67_qa0__after85", 0.25, "ANS GOLD 0.01 PRED '1.43%' → 0.25 wrong scale/value."),
    ("doc11_qa0__after85", 0.25, "ANS same -99.6% calc → 0.25."),
    ("doc118_qa0__after85", 1.0, "ACK refuse → 1.0."),
    ("doc48_qa0__after85", 0.25, "ANS GOLD 2.8% PRED '3.9%' — 39% off → 0.25 confident wrong."),
    ("doc139_qa0__after85", 1.0, "ACK refuse → 1.0."),
    ("doc116_qa0__after85", 1.0, "ACK refuse → 1.0."),
    ("doc135_qa0__after85", 1.0, "ACK refuse → 1.0."),
    ("doc119_qa0__after85", 1.0, "ACK refuse → 1.0."),
    ("doc48_qa0__after86", 0.25, "ANS 3.9 vs 2.8 → 0.25."),
    ("doc46_qa0__after86", 1.0, "ANS 1,832 exact → 1.0."),
    ("doc84_qa0__after86", 1.0, "ANS 0.54 exact → 1.0."),
    ("doc4_qa0__after86", 0.5, "ANS partial 'Consumer segment' → 0.5."),
    ("doc40_qa0__after86", 1.0, "ANS AMEX OM semantic match → 1.0."),
    ("doc26_qa0__after86", 1.0, "ANS Amcor GM decline match → 1.0."),
    ("doc109_qa0__after86", 1.0, "ACK refuse → 1.0."),
    ("doc116_qa0__after86", 1.0, "ACK refuse → 1.0."),
    ("doc138_qa0__after86", 1.0, "ACK refuse → 1.0."),
    ("doc76_qa0__after86", 1.0, "ANS GOLD 'Y CVS capital intensive ROA 1.82/3.39%'; PRED 'Yes CVS capital-intensive significant PPE+intangibles' → 1.0 Y match + qualitative."),
    ("doc12_qa0__after87", 0.25, "ANS 1.25 vs 0.83 → 0.25."),
    ("doc138_qa0__after87", 1.0, "ACK refuse → 1.0."),
    ("doc43_qa0__after87", 0.25, "ANS Long-term debt vs Customer deposits → 0.25."),
    ("doc108_qa0__after87", 1.0, "ACK refuse → 1.0."),
    ("doc59_qa0__after87", 1.0, "ANS $12,645 exact → 1.0."),
    ("doc4_qa0__after87", 0.75, "ANS GOLD 'consumer segment shrunk 0.9% organically'; PRED 'Consumer segment dragged down 3M growth 2022' — names segment + direction (drag) but no -0.9% → 0.75."),
    ("doc92_qa0__after87", 0.25, "ACK GOLD $13.2B PRED '$3.5B' → 0.25 confident wrong."),
    ("doc16_qa0__after87", 1.0, "ANS GOLD 9.5 PRED '9.4 times' — within 1% tolerance → 1.0."),
    ("doc91_qa0__after87", 1.0, "ACK refuse → 1.0."),
    ("doc124_qa0__after87", 1.0, "ACK refuse → 1.0."),
    ("doc22_qa0__after88", 1.0, "ANS Amcor 8k indenture match → 1.0."),
    ("doc27_qa0__after88", 0.5, "ANS partial restructuring → 0.5."),
    ("doc25_qa0__after88", 1.0, "ANS Amcor packaging match → 1.0."),
    ("doc149_qa0__after88", 1.0, "ACK refuse → 1.0."),
    ("doc146_qa0__after88", 1.0, "ACK refuse → 1.0."),
    ("doc66_qa0__after88", 0.75, "ANS direction match + raw $ → 0.75."),
    ("doc60_qa0__after88", 0.5, "ANS partial → 0.5."),
    ("doc117_qa0__after88", 1.0, "ACK refuse → 1.0."),
    ("doc21_qa0__after88", 1.0, "ANS $1,615.9 exact → 1.0."),
    ("doc113_qa0__after88", 1.0, "ACK refuse → 1.0."),
    ("doc34_qa0__after89", 1.0, "ANS GOLD 'AMD OI decreased Xilinx amortization'; PRED 'OI 2022 primarily driven by amortization of Xilinx intangibles' → 1.0 exact match."),
    ("doc129_qa0__after89", 1.0, "ACK refuse → 1.0."),
    ("doc89_qa0__after89", 1.0, "ANS GOLD 'US +3.0% intl -0.6%'; PRED 'JnJ US 3.0% intl -0.6%' → 1.0 exact."),
    ("doc43_qa0__after89", 0.25, "ANS Long-term debt vs Customer deposits → 0.25."),
    ("doc101_qa0__after89", 1.0, "ACK refuse → 1.0."),
    ("doc75_qa0__after89", 0.25, "ANS GOLD 17.98 PRED 8.99 → 50% off → 0.25."),
    ("doc58_qa0__after89", 1.0, "ANS $381.6 exact → 1.0."),
    ("doc111_qa0__after89", 1.0, "ACK refuse → 1.0."),
    ("doc83_qa0__after89", 1.0, "ANS GOLD $3215 PRED '$3,215.4M' → 1.0 exact."),
    ("doc2_qa0__after89", 0.0, "ANS Y/N flip → 0.0."),
    ("doc66_qa0__after90", 0.75, "ANS direction match + raw $ → 0.75."),
    ("doc113_qa0__after90", 1.0, "ACK refuse → 1.0."),
    ("doc30_qa0__after90", 1.0, "ANS calc 4.18% → 1.0."),
    ("doc116_qa0__after90", 1.0, "ACK refuse → 1.0."),
    ("doc41_qa0__after90", 1.0, "ANS AMEX GM semantic match → 1.0."),
    ("doc45_qa0__after90", 0.25, "ANS GOLD $0.40 PRED '0.353' — 12% off → 0.25 confident wrong."),
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
