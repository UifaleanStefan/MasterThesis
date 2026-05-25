"""Phase 1.9 — attention-corpus-tuned calibration cell — Part 5.

Entries 0600-0752 (153 entries).
"""

from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__attention-corpus-tuned__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__attention-corpus-tuned__calibration__seed42/results.jsonl")

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc13_qa0__after60", 0.25, "ANS GOLD 'No, OM declined 36.8→34.6% -2.2%'; PRED states 'improvement from previous year' despite correct numbers showing decline → 0.25 confident wrong qualitative direction."),
    ("doc59_qa0__after60", 1.0, "ANS GOLD $12645 PRED $12,645 → exact → 1.0."),
    ("doc47_qa0__after60", 0.5, "ANS GOLD 'No, AWK -$1561M'; PRED 'Yes positive' then calc shows -$1561M then 'negative... does not have'. Self-contradicting; correct calc but wrong opening direction → 0.5 partial."),
    ("doc67_qa0__after60", 1.0, "ACK refuse → 1.0."),
    ("doc130_qa0__after60", 1.0, "ACK refuse → 1.0."),
    ("doc18_qa0__after60", 0.25, "ANS 30.77 vs 93.86 → 0.25 confident wrong."),
    ("doc133_qa0__after60", 1.0, "ACK refuse → 1.0."),
    ("doc7_qa0__after60", 1.0, "ANS 65 years match → 1.0."),
    ("doc137_qa0__after60", 1.0, "ACK refuse → 1.0."),
    ("doc134_qa0__after60", 1.0, "ACK refuse → 1.0."),
    ("doc50_qa0__after61", 0.0, "ANS Y/N flip 'fluctuated 2%' vs 'consistent' → 0.0."),
    ("doc20_qa0__after61", 1.0, "ANS 11,588 exact → 1.0."),
    ("doc96_qa0__after61", 1.0, "ACK JPM GM semantic match → 1.0."),
    ("doc69_qa0__after61", 1.0, "ACK refuse → 1.0."),
    ("doc12_qa0__after61", 0.25, "ANS 1.25 vs 0.83 → 51% off → 0.25 confident wrong."),
    ("doc54_qa0__after61", 0.5, "ANS GOLD 'Y -1.32% 982→969'; PRED 'Y, decrease, 908→931 stores' — Y/N direction match but wrong store counts → 0.5 partial."),
    ("doc126_qa0__after61", 1.0, "ACK refuse → 1.0."),
    ("doc106_qa0__after61", 1.0, "ACK refuse → 1.0."),
    ("doc142_qa0__after61", 1.0, "ACK refuse → 1.0."),
    ("doc75_qa0__after61", 1.0, "ACK refuse → 1.0."),
    ("doc47_qa0__after62", 0.5, "ANS same contradictory → 0.5."),
    ("doc40_qa0__after62", 1.0, "ANS 'OM not useful' semantic match → 1.0."),
    ("doc101_qa0__after62", 1.0, "ACK refuse → 1.0."),
    ("doc140_qa0__after62", 1.0, "ACK refuse → 1.0."),
    ("doc87_qa0__after62", 1.0, "ACK refuse → 1.0."),
    ("doc121_qa0__after62", 1.0, "ACK refuse → 1.0."),
    ("doc83_qa0__after62", 1.0, "ACK refuse → 1.0."),
    ("doc72_qa0__after62", 0.5, "ACK GOLD 'Corning tax 20→23%'; PRED '15.6%→24.8%' — direction matches (increase) but values off → 0.5 partial."),
    ("doc147_qa0__after62", 1.0, "ACK refuse → 1.0."),
    ("doc126_qa0__after62", 1.0, "ACK refuse → 1.0."),
    ("doc126_qa0__after63", 1.0, "ACK refuse → 1.0."),
    ("doc64_qa0__after63", 1.0, "ACK 'Y Boeing cyclical' match → 1.0."),
    ("doc115_qa0__after63", 1.0, "ACK refuse → 1.0."),
    ("doc77_qa0__after63", 1.0, "ACK refuse → 1.0."),
    ("doc143_qa0__after63", 1.0, "ACK refuse → 1.0."),
    ("doc123_qa0__after63", 1.0, "ACK refuse → 1.0."),
    ("doc61_qa0__after63", 1.0, "ANS GOLD 'Y Lion Air + Ethiopian crashes'; PRED 'Y Boeing legal actions Lion Air Flight 610 + Ethiopian Airlines Flight 302' → 1.0 exact match."),
    ("doc33_qa0__after63", 1.0, "ANS AMD EPYC+semi-custom+Xilinx match → 1.0."),
    ("doc39_qa0__after63", 1.0, "ANS US/EMEA/APAC/LACC match → 1.0."),
    ("doc25_qa0__after63", 1.0, "ANS Amcor packaging match → 1.0."),
    ("doc132_qa0__after64", 1.0, "ACK refuse → 1.0."),
    ("doc60_qa0__after64", 0.5, "ANS GOLD 'Y multiple Commercial Airplanes 39%, Defence 35%, Services'; PRED 'Y, Commercial Airplanes $25,867M' — Y match + 1 category but missing other 2 → 0.5 partial."),
    ("doc134_qa0__after64", 1.0, "ACK refuse → 1.0."),
    ("doc107_qa0__after64", 1.0, "ACK refuse → 1.0."),
    ("doc68_qa0__after64", 1.0, "ACK refuse → 1.0."),
    ("doc24_qa0__after64", 0.5, "ANS partial Amcor acquisitions → 0.5."),
    ("doc36_qa0__after64", 0.25, "ANS GOLD 'Data Center' PRED 'Gaming segment' → confident wrong specific (different segment) → 0.25."),
    ("doc117_qa0__after64", 1.0, "ACK refuse → 1.0."),
    ("doc27_qa0__after64", 0.5, "ANS partial → 0.5."),
    ("doc41_qa0__after64", 1.0, "ANS 'GM not useful' semantic match → 1.0."),
    ("doc105_qa0__after65", 1.0, "ACK refuse → 1.0."),
    ("doc146_qa0__after65", 1.0, "ACK refuse → 1.0."),
    ("doc26_qa0__after65", 1.0, "ANS Amcor GM decline match → 1.0."),
    ("doc18_qa0__after65", 0.25, "ANS 29.73 vs 93.86 → 0.25."),
    ("doc89_qa0__after65", 1.0, "ACK refuse → 1.0."),
    ("doc114_qa0__after65", 1.0, "ACK refuse → 1.0."),
    ("doc102_qa0__after65", 1.0, "ACK refuse → 1.0."),
    ("doc38_qa0__after65", 0.25, "ANS GOLD 'There are none' PRED 'Common Shares par $0.20' — gives common shares info instead of saying no debt securities → 0.25 confident wrong topic."),
    ("doc94_qa0__after65", 1.0, "ACK refuse → 1.0."),
    ("doc145_qa0__after65", 1.0, "ACK refuse → 1.0."),
    ("doc55_qa0__after66", 1.0, "ANS Entertainment 9% gaming match → 1.0."),
    ("doc51_qa0__after66", 1.0, "ANS GOLD 'Best Buy Current Health + Yardbird FY22'; PRED 'Current Health $389M + Yardbird $79M FY22, no FY23/FY21' — exact match + amounts → 1.0."),
    ("doc62_qa0__after66", 0.25, "ANS GOLD 'Y Boeing GM improved 4.8→5.3%'; PRED 'GM not useful for Boeing cyclicality' — reframes question with wrong premise → 0.25 confident wrong reframe."),
    ("doc139_qa0__after66", 1.0, "ACK refuse → 1.0."),
    ("doc142_qa0__after66", 1.0, "ACK refuse → 1.0."),
    ("doc149_qa0__after66", 1.0, "ACK refuse → 1.0."),
    ("doc116_qa0__after66", 1.0, "ACK refuse → 1.0."),
    ("doc103_qa0__after66", 1.0, "ACK refuse → 1.0."),
    ("doc66_qa0__after66", 0.75, "ANS GOLD 'tax 0.62% vs -14.76%'; PRED 'lower FY22, $(31)M expense vs $743M benefit' — direction matches + raw $ figures but no rate calc → 0.75."),
    ("doc17_qa0__after66", 0.25, "ANS -1.32 vs -0.02 → 0.25."),
    ("doc74_qa0__after67", 1.0, "ACK GOLD $59268 PRED $59,364 — diff $96, 0.16% relative, well within tolerance → 1.0."),
    ("doc76_qa0__after67", 1.0, "ACK refuse → 1.0."),
    ("doc25_qa0__after67", 1.0, "ANS Amcor packaging match → 1.0."),
    ("doc71_qa0__after67", 1.0, "ACK refuse → 1.0."),
    ("doc113_qa0__after67", 1.0, "ACK refuse → 1.0."),
    ("doc2_qa0__after67", 0.0, "ANS Y/N flip 'Yes capital-intensive' vs gold 'No' → 0.0."),
    ("doc3_qa0__after67", 0.75, "ANS qualitative OI drivers match without -1.7% → 0.75."),
    ("doc141_qa0__after67", 1.0, "ACK refuse → 1.0."),
    ("doc35_qa0__after67", 1.0, "ANS AMD $3,565M match → 1.0."),
    ("doc39_qa0__after67", 1.0, "ANS US/EMEA/APAC/LACC exact → 1.0."),
    ("doc66_qa0__after68", 0.75, "ANS same direction match + raw $ → 0.75."),
    ("doc25_qa0__after68", 1.0, "ANS Amcor packaging match → 1.0."),
    ("doc99_qa0__after68", 1.0, "ACK refuse → 1.0."),
    ("doc85_qa0__after68", 1.0, "ACK refuse → 1.0."),
    ("doc24_qa0__after68", 0.5, "ANS partial Amcor acquisitions → 0.5."),
    ("doc126_qa0__after68", 1.0, "ACK refuse → 1.0."),
    ("doc32_qa0__after68", 1.0, "ANS AMD products match → 1.0."),
    ("doc15_qa0__after68", 1.0, "ANS 0=0 → 1.0."),
    ("doc82_qa0__after68", 0.25, "ACK GOLD 0.68 PRED 1.14 → 68% off → 0.25 confident wrong."),
    ("doc121_qa0__after68", 1.0, "ACK refuse → 1.0."),
    ("doc105_qa0__after69", 1.0, "ACK refuse → 1.0."),
    ("doc85_qa0__after69", 1.0, "ACK refuse → 1.0."),
    ("doc139_qa0__after69", 1.0, "ACK refuse → 1.0."),
    ("doc30_qa0__after69", 1.0, "ANS calc 4.18% within tolerance → 1.0."),
    ("doc108_qa0__after69", 1.0, "ACK refuse → 1.0."),
    ("doc32_qa0__after69", 1.0, "ANS AMD products match → 1.0."),
    ("doc87_qa0__after69", 1.0, "ACK refuse → 1.0."),
    ("doc93_qa0__after69", 1.0, "ACK refuse → 1.0."),
    ("doc65_qa0__after69", 1.0, "ANS GOLD 'Boeing increase 737/777X/787 production FY23'; PRED '787 5/mo, 737 gradual, 777X resume 2023' → 1.0 exact match all three planes."),
    ("doc16_qa0__after69", 0.25, "ANS truncated calc framework same as ~11.98 vs 9.5 → 0.25 confident wrong."),
    ("doc26_qa0__after70", 1.0, "ANS Amcor GM decline match → 1.0."),
    ("doc66_qa0__after70", 0.75, "ANS same direction match + raw $ → 0.75."),
    ("doc93_qa0__after70", 1.0, "ACK refuse → 1.0."),
    ("doc138_qa0__after70", 1.0, "ACK refuse → 1.0."),
    ("doc129_qa0__after70", 1.0, "ACK refuse → 1.0."),
    ("doc71_qa0__after70", 0.25, "ACK GOLD 10.3% PRED '4.4%' — confident wrong (57% off) → 0.25."),
    ("doc135_qa0__after70", 1.0, "ACK refuse → 1.0."),
    ("doc65_qa0__after70", 1.0, "ANS same Boeing production match → 1.0."),
    ("doc104_qa0__after70", 1.0, "ACK refuse → 1.0."),
    ("doc91_qa0__after70", 1.0, "ACK refuse → 1.0."),
    ("doc10_qa0__after71", 0.25, "ANS GOLD 0.66 PRED 1.66 → 0.25 confident wrong."),
    ("doc46_qa0__after71", 1.0, "ANS GOLD $1832 PRED '1,829' — within tolerance → 1.0."),
    ("doc59_qa0__after71", 1.0, "ANS $12,645 exact → 1.0."),
    ("doc95_qa0__after71", 1.0, "ACK refuse → 1.0."),
    ("doc55_qa0__after71", 1.0, "ANS Entertainment 9% gaming match → 1.0."),
    ("doc139_qa0__after71", 1.0, "ACK refuse → 1.0."),
    ("doc42_qa0__after71", 1.0, "ANS AMEX tax rate match → 1.0."),
    ("doc94_qa0__after71", 1.0, "ACK refuse → 1.0."),
    ("doc58_qa0__after71", 1.0, "ANS GOLD $382 PRED '$381.6 million' — within tolerance → 1.0."),
    ("doc14_qa0__after71", 0.0, "ANS refusal on definitive Y/N → 0.0."),
    ("doc3_qa0__after72", 0.75, "ANS qualitative match → 0.75."),
    ("doc110_qa0__after72", 1.0, "ACK refuse → 1.0."),
    ("doc134_qa0__after72", 1.0, "ACK refuse → 1.0."),
    ("doc12_qa0__after72", 0.25, "ANS 1.25 vs 0.83 → 0.25."),
    ("doc71_qa0__after72", 0.25, "ANS GOLD 10.3% PRED '15.5%' → 50% off → 0.25 confident wrong."),
    ("doc52_qa0__after72", 1.0, "ANS GOLD '$1.8B ops'; PRED '$1,824M ops' — within tolerance → 1.0."),
    ("doc64_qa0__after72", 1.0, "ANS Y Boeing cyclical match → 1.0."),
    ("doc26_qa0__after72", 1.0, "ANS Amcor GM decline → 1.0."),
    ("doc117_qa0__after72", 1.0, "ACK 'Operating activities most cash flow Nike FY23' semantic match → 1.0."),
    ("doc119_qa0__after72", 1.0, "ACK refuse → 1.0."),
    ("doc14_qa0__after73", 0.0, "ANS refusal → 0.0."),
    ("doc106_qa0__after73", 1.0, "ACK refuse → 1.0."),
    ("doc12_qa0__after73", 0.25, "ANS 1.25 vs 0.83 → 0.25."),
    ("doc114_qa0__after73", 1.0, "ACK refuse → 1.0."),
    ("doc92_qa0__after73", 0.25, "ACK GOLD $13.2B PRED $3.7B → 0.25 confident wrong."),
    ("doc140_qa0__after73", 1.0, "ACK refuse → 1.0."),
    ("doc115_qa0__after73", 1.0, "ACK refuse → 1.0."),
    ("doc69_qa0__after73", 1.0, "ANS 0.80 vs 0.8 → 1.0."),
    ("doc4_qa0__after73", 0.5, "ANS partial 'Consumer segment' → 0.5."),
    ("doc26_qa0__after73", 1.0, "ANS Amcor GM decline match → 1.0."),
    ("doc119_qa0__after74", 1.0, "ACK refuse → 1.0."),
    ("doc117_qa0__after74", 1.0, "ACK Nike ops cash match → 1.0."),
    ("doc69_qa0__after74", 1.0, "ANS 0.80 → 1.0."),
    ("doc123_qa0__after74", 1.0, "ACK refuse → 1.0."),
    ("doc90_qa0__after74", 1.0, "ACK Consumer Health match → 1.0."),
    ("doc83_qa0__after74", 0.25, "ACK GOLD $3215 PRED '$1,000M' → 69% off → 0.25 confident wrong."),
    ("doc126_qa0__after74", 1.0, "ACK refuse → 1.0."),
    ("doc50_qa0__after74", 0.0, "ANS Y/N flip → 0.0."),
    ("doc22_qa0__after74", 1.0, "ANS Amcor 8k indenture match → 1.0."),
    ("doc6_qa0__after74", 1.0, "ANS 3M debt securities list match → 1.0."),
    ("doc37_qa0__after75", 1.0, "ANS Y 16% one customer match → 1.0."),
    ("doc0_qa0__after75", 1.0, "ANS $1501M just within 5% tolerance of $1577 → 1.0."),
    ("doc122_qa0__after75", 0.25, "ACK PRED '0' vs $411M → 0.25."),
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
