"""Phase 1.9 — attention-corpus-tuned calibration cell — Part 10 FINAL.

Entries 1374-1499 (126 entries). Final batch — cell completes at 1500/1500.
"""

from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__attention-corpus-tuned__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__attention-corpus-tuned__calibration__seed42/results.jsonl")

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc134_qa0__after137", 1.0, "ANS 'Developed Rest of World' match → 1.0."),
    ("doc135_qa0__after137", 1.0, "ANS GOLD 'Y Upjohn spin-off' PRED 'Y Pfizer separating Upjohn' → 1.0 match."),
    ("doc113_qa0__after137", 1.0, "ANS '5,466.3M' within tolerance of $5466 → 1.0."),
    ("doc126_qa0__after137", 1.0, "ANS $400M increase exact + breakdown → 1.0."),
    ("doc18_qa0__after137", 0.25, "ANS 30.73 vs 93.86 → 0.25."),
    ("doc13_qa0__after137", 0.0, "ANS Y/N flip 'Yes improving' vs 'No declined' → 0.0."),
    ("doc60_qa0__after138", 0.5, "ANS partial Boeing categories (only 1 of 3+) → 0.5."),
    ("doc39_qa0__after138", 1.0, "ANS US/EMEA/APAC/LACC match + expansions → 1.0."),
    ("doc119_qa0__after138", 1.0, "ANS GOLD $4.60 PRED '$4.625 billion' — within 0.5% tolerance → 1.0."),
    ("doc142_qa0__after138", 1.0, "ACK refuse → 1.0."),
    ("doc35_qa0__after138", 1.0, "ANS AMD $3,565M match → 1.0."),
    ("doc8_qa0__after138", 0.25, "ANS GOLD 24.26 PRED '2.54' → 90% off → 0.25."),
    ("doc131_qa0__after138", 1.0, "ANS GOLD 'Y JV gain' PRED 'Y $(8,107)M JV gain' → 1.0 Y match + specific."),
    ("doc67_qa0__after138", 0.25, "ANS 1.43% vs 0.01 → 0.25."),
    ("doc47_qa0__after138", 0.5, "ANS contradictory self-correcting → 0.5."),
    ("doc3_qa0__after138", 0.75, "ANS qualitative match → 0.75."),
    ("doc148_qa0__after139", 1.0, "ACK refuse → 1.0."),
    ("doc70_qa0__after139", 1.0, "ANS GOLD 63.86 PRED '66.67' — within tolerance → 1.0."),
    ("doc118_qa0__after139", 0.75, "ANS Y PayPal $12,416M (vs gold $1.6B) — Y match but wrong magnitude → 0.75."),
    ("doc39_qa0__after139", 1.0, "ANS US/EMEA/APAC/LACC exact → 1.0."),
    ("doc74_qa0__after139", 1.0, "ANS '59,268' exact → 1.0."),
    ("doc12_qa0__after139", 0.25, "ANS 1.25 vs 0.83 → 0.25."),
    ("doc24_qa0__after139", 0.0, "ANS refusal on definitive Amcor acquisitions → 0.0."),
    ("doc25_qa0__after139", 1.0, "ANS Amcor packaging match → 1.0."),
    ("doc0_qa0__after139", 1.0, "ANS $1501 within tolerance → 1.0."),
    ("doc92_qa0__after139", 1.0, "ANS $13.2B exact → 1.0."),
    ("doc5_qa0__after140", 0.0, "ANS refusal on definitive 3M quick ratio → 0.0."),
    ("doc135_qa0__after140", 1.0, "ANS Y Upjohn match → 1.0."),
    ("doc76_qa0__after140", 1.0, "ANS Y CVS capital intensive match → 1.0."),
    ("doc26_qa0__after140", 1.0, "ANS Amcor GM decline match → 1.0."),
    ("doc55_qa0__after140", 1.0, "ANS gaming 9.0% match → 1.0."),
    ("doc58_qa0__after140", 1.0, "ANS $381.6 within tolerance → 1.0."),
    ("doc105_qa0__after140", 1.0, "ANS Y MGM $0.01 match → 1.0."),
    ("doc31_qa0__after140", 0.0, "ANS refusal on definitive → 0.0."),
    ("doc123_qa0__after140", 0.25, "ANS $14,275 vs $9068 → 0.25."),
    ("doc3_qa0__after140", 0.75, "ANS qualitative match → 0.75."),
    ("doc62_qa0__after141", 0.25, "ANS 'GM not useful for Boeing' reframe wrong → 0.25."),
    ("doc3_qa0__after141", 0.75, "ANS qualitative match → 0.75."),
    ("doc38_qa0__after141", 0.25, "ANS Common Shares wrong → 0.25."),
    ("doc143_qa0__after141", 1.0, "ACK refuse → 1.0."),
    ("doc125_qa0__after141", 1.0, "ANS GOLD 'proposal defeated' PRED 'proposal defeated with 19,718,780 for vs 977,228,788 against' → 1.0 exact + specifics."),
    ("doc87_qa0__after141", 0.25, "ANS 7.6 vs 2.7 → 0.25."),
    ("doc63_qa0__after141", 0.5, "ANS partial Boeing customers → 0.5."),
    ("doc69_qa0__after141", 1.0, "ANS 0.80 vs 0.8 → 1.0."),
    ("doc124_qa0__after141", 1.0, "ANS GOLD 16.5% PRED 'EBITDA $14,275M → 16.5% margin' → 1.0 exact match!"),
    ("doc17_qa0__after141", 0.25, "ANS -1.42 vs -0.02 → 0.25."),
    ("doc34_qa0__after142", 1.0, "ANS Xilinx amortization exact → 1.0."),
    ("doc102_qa0__after142", 1.0, "ANS 0.4% exact → 1.0."),
    ("doc127_qa0__after142", 1.0, "ANS $8.4B + breakdown → 1.0."),
    ("doc146_qa0__after142", 1.0, "ACK refuse → 1.0."),
    ("doc2_qa0__after142", 0.0, "ANS Y/N flip → 0.0."),
    ("doc113_qa0__after142", 1.0, "ANS '5,466.3M' within tolerance → 1.0."),
    ("doc139_qa0__after142", 1.0, "ANS GOLD '47 new stores'; PRED 'opening 47 new stores + brand launches + cost increases' → 1.0 match + specifics."),
    ("doc74_qa0__after142", 1.0, "ANS '59,268' exact → 1.0."),
    ("doc132_qa0__after142", 0.5, "ANS GOLD 'Trillium, Array, Therachon' PRED 'Trillium, Array, Upjohn' — 2/3 match but wrong 3rd → 0.5."),
    ("doc107_qa0__after142", 0.25, "ANS 1.61 vs 0 → 0.25 wrong."),
    ("doc63_qa0__after143", 0.5, "ANS partial → 0.5."),
    ("doc45_qa0__after143", 0.25, "ANS $0.389B vs $0.40 wrong scale → 0.25."),
    ("doc4_qa0__after143", 0.5, "ANS partial 'Consumer segment' → 0.5."),
    ("doc141_qa0__after143", 0.0, "ANS 'Decrease' Y/N flip vs gold 'increased' → 0.0."),
    ("doc93_qa0__after143", 1.0, "ANS Y 20→20.1% match → 1.0."),
    ("doc134_qa0__after143", 1.0, "ANS 'Developed Rest of World' → 1.0."),
    ("doc79_qa0__after143", 1.0, "ANS Y Mary N. Dillon match → 1.0."),
    ("doc138_qa0__after143", 1.0, "ANS GOLD 'lower marketing + leverage of incentive comp'; PRED 'lower marketing + leverage of incentive comp due to higher sales' → 1.0 EXACT."),
    ("doc11_qa0__after143", 1.0, "ANS GOLD 65.4% PRED calc shows (1,493,602-903,095)/903,095 = 65.4% — calculation correct (truncated display) → 1.0."),
    ("doc7_qa0__after143", 1.0, "ANS 65 years match → 1.0."),
    ("doc86_qa0__after144", 0.25, "ANS 'GM not useful for JnJ' reframe wrong → 0.25."),
    ("doc31_qa0__after144", 0.0, "ANS refusal on definitive → 0.0."),
    ("doc139_qa0__after144", 1.0, "ANS 47 new stores match → 1.0."),
    ("doc44_qa0__after144", 1.0, "ANS Y match → 1.0."),
    ("doc24_qa0__after144", 0.0, "ANS refusal on definitive Amcor acquisitions → 0.0."),
    ("doc97_qa0__after144", 0.25, "ANS Consumer Banking wrong → 0.25."),
    ("doc63_qa0__after144", 0.5, "ANS partial → 0.5."),
    ("doc110_qa0__after144", 1.0, "ANS GOLD $32780 PRED '$32,780' → 1.0 exact."),
    ("doc23_qa0__after144", 0.5, "ANS Amcor reframe → 0.5."),
    ("doc78_qa0__after144", 1.0, "ANS Y CVS $0.55 exact → 1.0."),
    ("doc23_qa0__after145", 0.5, "ANS Amcor reframe → 0.5."),
    ("doc110_qa0__after145", 1.0, "ANS '$32,780 million' exact → 1.0."),
    ("doc19_qa0__after145", 1.0, "ANS 30.7 within tolerance → 1.0."),
    ("doc20_qa0__after145", 1.0, "ANS '11,588' exact → 1.0."),
    ("doc136_qa0__after145", 1.0, "ANS GOLD 'There are none' PRED 'None.' → 1.0 exact match."),
    ("doc95_qa0__after145", 0.25, "ANS GOLD '$66.56/share' PRED '$239.45/share' → 3.6x off → 0.25 confident wrong."),
    ("doc119_qa0__after145", 1.0, "ANS $4.625B within tolerance → 1.0."),
    ("doc109_qa0__after145", 0.75, "ANS GOLD 'corporate bonds 82%' PRED 'Corporate bonds' — correct category but no % share → 0.75."),
    ("doc62_qa0__after145", 0.25, "ANS reframe wrong → 0.25."),
    ("doc12_qa0__after145", 0.25, "ANS 1.25 vs 0.83 → 0.25."),
    ("doc111_qa0__after146", 0.5, "ANS GOLD 'No MSFT decreased $2.5bn'; PRED 'Yes, Microsoft has decreased long-term debt $47,032M→$41,990M' — Y/N label says 'Yes' but body explains 'decreased', confusing self-correcting answer with right direction → 0.5 partial."),
    ("doc51_qa0__after146", 1.0, "ANS Best Buy acquisitions match → 1.0."),
    ("doc10_qa0__after146", 0.25, "ANS GOLD 0.66 PRED '1.87' → 0.25."),
    ("doc64_qa0__after146", 1.0, "ANS Y Boeing cyclical match → 1.0."),
    ("doc139_qa0__after146", 1.0, "ANS 47 new stores match → 1.0."),
    ("doc24_qa0__after146", 0.0, "ANS refusal → 0.0."),
    ("doc98_qa0__after146", 1.0, "ANS Y -$7M VaR match → 1.0."),
    ("doc5_qa0__after146", 0.0, "ANS refusal on definitive → 0.0."),
    ("doc13_qa0__after146", 0.0, "ANS Y/N flip → 0.0."),
    ("doc53_qa0__after146", 1.0, "ANS GOLD 'Y -42% cash'; PRED 'Y, cash $1,874M Jan→$1,093M July' — Y match + supporting figures → 1.0."),
    ("doc25_qa0__after147", 1.0, "ANS Amcor packaging match → 1.0."),
    ("doc24_qa0__after147", 0.0, "ANS refusal → 0.0."),
    ("doc35_qa0__after147", 1.0, "ANS AMD $3,565M match → 1.0."),
    ("doc22_qa0__after147", 1.0, "ANS Amcor 8k indenture match → 1.0."),
    ("doc117_qa0__after147", 1.0, "ANS Nike ops $5,841M match → 1.0."),
    ("doc26_qa0__after147", 1.0, "ANS Amcor GM decline match → 1.0."),
    ("doc141_qa0__after147", 0.0, "ANS 'Decrease' Y/N flip → 0.0."),
    ("doc83_qa0__after147", 1.0, "ANS $3,115.4M within tolerance of $3215 → 1.0."),
    ("doc102_qa0__after147", 0.25, "ANS GOLD 0.4% PRED '1.3%' → 0.25 wrong."),
    ("doc111_qa0__after147", 0.5, "ANS same Y/N label flip with right direction in body → 0.5."),
    ("doc140_qa0__after148", 1.0, "ANS GOLD '36%' PRED 'Approximately 36.5%' — within tolerance → 1.0."),
    ("doc107_qa0__after148", 0.25, "ANS 1.61 vs 0 → 0.25."),
    ("doc38_qa0__after148", 0.25, "ANS Common Shares wrong → 0.25."),
    ("doc59_qa0__after148", 1.0, "ANS $12,645 exact → 1.0."),
    ("doc120_qa0__after148", 0.25, "ANS wrong PepsiCo geographies (Pfizer-like regions) → 0.25."),
    ("doc127_qa0__after148", 1.0, "ANS $8.4B exact + breakdown → 1.0."),
    ("doc77_qa0__after148", 1.0, "ANS Y CVS legal usual+customary match → 1.0."),
    ("doc118_qa0__after148", 0.75, "ANS Y PayPal $12,416M (vs $1.6B) → 0.75."),
    ("doc85_qa0__after148", 1.0, "ANS exact 'No JnJ FY22 1.3%' match → 1.0."),
    ("doc137_qa0__after148", 1.0, "ANS GOLD 'Ulta no acquisitions FY23/22'; PRED 'do not mention any major acquisitions' — semantically matches (PRED's 'no mention' = gold's 'no acquisitions') → 1.0."),
    ("doc90_qa0__after149", 1.0, "ANS Consumer Health match → 1.0."),
    ("doc82_qa0__after149", 1.0, "ANS GOLD 0.68 PRED '0.69' — within tolerance → 1.0."),
    ("doc63_qa0__after149", 0.5, "ANS partial → 0.5."),
    ("doc109_qa0__after149", 0.75, "ANS 'Corporate bonds' without % → 0.75."),
    ("doc61_qa0__after149", 1.0, "ANS Y Lion Air + Ethiopian detailed match → 1.0."),
    ("doc55_qa0__after149", 1.0, "ANS gaming 9.0% match → 1.0."),
    ("doc80_qa0__after149", 1.0, "ANS Y Richard match → 1.0."),
    ("doc105_qa0__after149", 1.0, "ANS Y MGM $0.01 match → 1.0."),
    ("doc108_qa0__after149", 1.0, "ANS MGM China + raw $ match → 1.0."),
    ("doc128_qa0__after149", 1.0, "ANS PepsiCo strong start match → 1.0."),
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
    if scores: print(f"Batch mean: {sum(scores)/len(scores):.4f}")
    all_lines = [_ for _ in RESULTS.read_text(encoding="utf-8").splitlines() if _.strip()]
    total = len(all_lines)
    print(f"Total: {total}/1500 ({100*total/1500:.1f}%)")
    if total >= 1500:
        all_scores = []
        for line in all_lines:
            try: all_scores.append(json.loads(line)["judge_score"])
            except Exception: continue
        if all_scores:
            print(f"CELL FINAL MEAN: {sum(all_scores)/len(all_scores):.4f}")


if __name__ == "__main__":
    main()
