"""Phase 1.9 Protocol B FB bm25-corpus calibration part 10 (entries 1396-1499, final batch).

Hand-judged by Claude per evaluation/claude_judge_protocol.md (5-point rubric +
calibration sub-rubric for [ACK] entries). Idempotent append to results.jsonl.
"""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "financebench__bm25-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__bm25-corpus__calibration__seed42/results.jsonl")

# (suffix-without-prefix-or-suffix, judge_score, rationale)
JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc24_qa0__after139", 0.5, "GOLD lists 3 Amcor FY2023 acquisitions (Czech, Shanghai, NZ-based). PRED gets Shanghai+NZ in FY2023 but attributes Czech to FY2022 incorrectly. Partial coverage."),
    ("doc25_qa0__after139", 1.0, "GOLD 'Amcor packaging leader for various use cases'. PRED 'Amcor primarily operates in packaging industry, food/beverage/pharma/medical/home/personal-care'. Correct."),
    ("doc0_qa0__after139", 1.0, "GOLD $1577. PRED '$1,577 million'. Exact match."),
    ("doc92_qa0__after139", 1.0, "GOLD '$13.2B from Kenvue separation'. PRED '$13.2 billion'. Correct."),
    ("doc5_qa0__after140", 0.0, "GOLD definitive 'No, quick ratio 0.96 for 3M Jun23'. PRED refuses ('passages do not contain info'). Refusal on definitive ANS."),
    ("doc135_qa0__after140", 1.0, "GOLD 'Yes, spinning off Upjohn'. PRED 'Yes, Pfizer separating Upjohn spin-off'. Correct."),
    ("doc76_qa0__after140", 0.75, "GOLD 'Yes CVS capital-intensive (ROA 1.82% 2022)'. PRED 'Yes, capital-intensive based on FY2022'. Y match but no quantitative support."),
    ("doc26_qa0__after140", 0.0, "GOLD definitive 'No, slight decline 0.8%'. PRED refuses. Refusal on definitive ANS."),
    ("doc55_qa0__after140", 1.0, "GOLD 'Entertainment 9% Q2 FY2024 from gaming'. PRED 'Entertainment, 9.0%, gaming'. Correct."),
    ("doc58_qa0__after140", 0.25, "GOLD $382. PRED '$1,000 million Square FY2020 CFO'. Wrong number (2.6x off)."),
    ("doc105_qa0__after140", 1.0, "GOLD 'Yes MGM $0.01 throughout FY2022'. PRED 'Yes MGM Resorts $0.01 throughout 2022'. Correct."),
    ("doc31_qa0__after140", 0.0, "GOLD definitive 'Yes quick ratio 1.57'. PRED refuses. Refusal on definitive ANS."),
    ("doc123_qa0__after140", 0.25, "GOLD $9068. PRED '$10,811 million'. Wrong (19% off, beyond 5% tolerance)."),
    ("doc3_qa0__after140", 0.0, "GOLD definitive 'Operating margin decreased 1.7%'. PRED refuses. Refusal on definitive ANS."),
    ("doc62_qa0__after141", 0.0, "GOLD definitive 'Yes, Boeing gross margin improved 4.8% to 5.3%'. PRED refuses. Refusal on definitive ANS."),
    ("doc3_qa0__after141", 0.0, "GOLD definitive 'Operating margin decreased 1.7%'. PRED refuses. Refusal on definitive ANS."),
    ("doc38_qa0__after141", 0.0, "GOLD 'There are none'. PRED hallucinates 'Common Shares AXP NYSE'. Confident wrong."),
    ("doc143_qa0__after141", 1.0, "[ACK] src=doc143 not yet seen (seen=142). PRED honest refusal 'passages do not contain Verizon expected payments for retirees 2024'. Correctly acknowledges missing info."),
    ("doc125_qa0__after141", 1.0, "GOLD 'Net-zero emissions proposal defeated'. PRED 'defeated, 19,718,780 for vs 977,228,788 against'. Correct + detail."),
    ("doc87_qa0__after141", 0.0, "GOLD definitive 'JnJ inventory turnover 2.7'. PRED refuses. Refusal on definitive ANS."),
    ("doc63_qa0__after141", 0.5, "GOLD 'Boeing customers: limited commercial airlines + US gov (40%)'. PRED 'Boeing derives significant portion from limited commercial airlines'. Partial — misses US gov."),
    ("doc69_qa0__after141", 0.25, "GOLD 0.8 dividend payout ratio. PRED 2.00. Wrong (2.5x off)."),
    ("doc124_qa0__after141", 0.25, "GOLD 16.5% EBITDA margin. PRED 13.9%. Wrong (2.6pp off)."),
    ("doc17_qa0__after141", 0.25, "GOLD -0.02. PRED -4.32%. Different unit/magnitude."),
    ("doc34_qa0__after142", 0.0, "GOLD definitive 'AMD operating income decrease from Xilinx amortization'. PRED refuses. Refusal on definitive ANS."),
    ("doc102_qa0__after142", 0.25, "GOLD 0.4%. PRED 0.1%. Off by 0.3pp (75% relative error)."),
    ("doc127_qa0__after142", 1.0, "GOLD '$8,400,000,000 total unsecured revolving'. PRED '$8.4B ($4.2B 2023 364-day + $4.2B 2023 5yr)'. Correct + breakdown."),
    ("doc146_qa0__after142", 1.0, "[ACK] src=doc146 not yet seen (seen=143). PRED honest refusal 'passages do not contain Verizon debt info'. Correctly acknowledges missing."),
    ("doc2_qa0__after142", 0.0, "GOLD 'No, well-managed CAPEX/RoA (5.1%/12.4%)'. PRED 'Yes, capital-intensive ($1,749M PP&E)'. Y/N flip."),
    ("doc113_qa0__after142", 1.0, "GOLD $5466. PRED '$5,466.3 million'. Within 0.01% tolerance."),
    ("doc139_qa0__after142", 1.0, "GOLD 'Increase due to 47 new stores'. PRED 'Increase from 47 new stores + brand launches + cost increases'. Correct + more detail."),
    ("doc74_qa0__after142", 1.0, "GOLD $59268. PRED '59,268 million'. Match."),
    ("doc132_qa0__after142", 0.5, "GOLD 'Trillium, Array, Therachon'. PRED 'Trillium, Array, Upjohn'. 2/3 correct (Therachon→Upjohn wrong)."),
    ("doc107_qa0__after142", 0.25, "GOLD 'Coverage ratio zero (negative EBIT)'. PRED '1.61'. Wrong (confident specific number)."),
    ("doc63_qa0__after143", 0.5, "GOLD 'Boeing customers: limited commercial airlines + US gov (40%)'. PRED 'limited commercial airlines'. Partial — misses US gov."),
    ("doc45_qa0__after143", 1.0, "GOLD $0.40 (billion). PRED '$0.389 billion'. Within 2.75% (under 5% tolerance)."),
    ("doc4_qa0__after143", 0.25, "GOLD 'Consumer segment shrunk 0.9%'. PRED 'Litigation costs Combat Arms Earplugs'. Wrong topic (segment vs charge)."),
    ("doc141_qa0__after143", 0.0, "GOLD 'Wages expense increased FY2023'. PRED 'Decrease'. Y/N flip / direction error."),
    ("doc93_qa0__after143", 0.75, "GOLD 'Yes, 20% to 20.1%'. PRED 'Yes, increased Q2 FY2023 vs Q2 FY2022'. Correct direction but no quant."),
    ("doc134_qa0__after143", 1.0, "GOLD 'Developed Rest of the World'. PRED 'Developed Rest of World'. Match (whitespace/article only)."),
    ("doc79_qa0__after143", 1.0, "GOLD 'Yes, Mary Dillon prev CEO Ulta'. PRED 'Yes, Mary N. Dillon former Exec Chair/CEO Ulta'. Correct + more."),
    ("doc138_qa0__after143", 1.0, "GOLD 'Lower marketing + leverage incentive due to higher sales'. PRED verbatim match + 'offset by deleverage corporate overhead'. Correct."),
    ("doc11_qa0__after143", 0.0, "GOLD definitive 65.4%. PRED refuses ('do not contain info about FY2015/FY2016'). Refusal on definitive ANS."),
    ("doc7_qa0__after143", 1.0, "GOLD 'Yes, 65 consecutive years dividend'. PRED 'Yes, 65th consecutive year increases'. Correct."),
    ("doc86_qa0__after144", 0.0, "GOLD definitive 'JnJ drivers: COVID vaccine exit, currency, commodity inflation'. PRED refuses. Refusal on definitive ANS."),
    ("doc31_qa0__after144", 0.0, "GOLD definitive 'Yes quick ratio 1.57'. PRED refuses. Refusal on definitive ANS."),
    ("doc139_qa0__after144", 1.0, "Same as doc139 after142. PRED correct."),
    ("doc44_qa0__after144", 1.0, "GOLD 'Yes' (Card Member retention). PRED 'Yes, Card Member retention remained high in 2022'. Correct + context."),
    ("doc24_qa0__after144", 0.5, "Same Amcor FY2023 acquisitions Q. Partial as before."),
    ("doc97_qa0__after144", 0.25, "GOLD 'Corporate & Investment Bank, $3725M Q2 2022'. PRED 'Consumer & Community Banking, $3,100M'. Wrong segment, wrong number."),
    ("doc63_qa0__after144", 0.5, "Boeing customers — same partial as before (limited airlines, misses US gov)."),
    ("doc110_qa0__after144", 0.0, "GOLD definitive $32780 Microsoft FY2016 COGS. PRED refuses ('do not contain info on FY2016 COGS'). Refusal on definitive ANS."),
    ("doc23_qa0__after144", 0.0, "GOLD definitive 'Improved 0.67 to 0.69 (3.4% jump)'. PRED refuses. Refusal on definitive ANS."),
    ("doc78_qa0__after144", 0.75, "GOLD 'Yes, CVS paid $0.55 quarterly FY2022'. PRED 'Yes CVS paid dividends Q2 FY2022'. Y correct but lacks $ amount."),
    ("doc23_qa0__after145", 0.0, "Same as doc23 after144. PRED refuses on definitive ANS."),
    ("doc110_qa0__after145", 0.0, "Same as doc110 after144. PRED refuses on definitive ANS."),
    ("doc19_qa0__after145", 0.0, "GOLD definitive 30.8% Amazon FY2016-FY2017 YoY revenue change. PRED refuses ('do not contain Amazon FY2016/FY2017 info'). Refusal on definitive ANS."),
    ("doc20_qa0__after145", 1.0, "GOLD $11588. PRED '11,588'. Match."),
    ("doc136_qa0__after145", 1.0, "GOLD 'There are none'. PRED 'None.'. Match."),
    ("doc95_qa0__after145", 0.0, "GOLD definitive $66.56/share. PRED refuses ('passages do not provide JPMorgan assets/shareholder distribution'). Refusal on definitive ANS."),
    ("doc119_qa0__after145", 1.0, "GOLD $4.60 (billion). PRED '$4.625 billion'. Within 0.55% (under 5% tolerance)."),
    ("doc109_qa0__after145", 1.0, "GOLD 'Corporate bonds ~82%'. PRED 'Corporate bonds, $416,420,000 H1 FY2023'. Correct identification."),
    ("doc62_qa0__after145", 0.0, "Same as doc62 after141. PRED refuses on definitive ANS."),
    ("doc12_qa0__after145", 0.0, "GOLD definitive 0.83 Adobe FY2017 operating cash flow ratio. PRED refuses. Refusal on definitive ANS."),
    ("doc111_qa0__after146", 0.0, "GOLD 'No, Microsoft decreased debt by $2.5bn FY23 vs FY22'. PRED 'Yes, Microsoft increased long-term debt from $41,990M to $41,990M' (self-contradictory — same number both years). Y/N flip + nonsensical claim."),
    ("doc51_qa0__after146", 1.0, "GOLD 'Best Buy: Current Health Ltd + Two Peaks/Yardbird in FY2022, both partially owned before'. PRED 'FY2022: Current Health $389M + Yardbird $79M; no FY23/FY21 acquisitions'. Correct + acquisition amounts."),
    ("doc10_qa0__after146", 0.0, "GOLD definitive 0.66 Adobe FY2015 operating cash flow ratio. PRED refuses. Refusal on definitive ANS."),
    ("doc64_qa0__after146", 0.75, "GOLD 'Yes, cyclicality due to airline industry'. PRED 'Yes, cyclicality'. Y correct but lacks airline industry context."),
    ("doc139_qa0__after146", 1.0, "Same as doc139 above. PRED correct."),
    ("doc24_qa0__after146", 0.5, "Same Amcor FY2023 acquisitions. Partial."),
    ("doc98_qa0__after146", 1.0, "GOLD 'Yes, it decreased' (VaR). PRED 'Yes, avg total VaR decreased $7M for three months ended June 30 2023'. Correct + quant."),
    ("doc5_qa0__after146", 0.0, "Same as doc5 after140. PRED refuses on definitive ANS."),
    ("doc13_qa0__after146", 0.0, "GOLD definitive 'No, operating margins declined 36.8% to 34.6% (Adobe)'. PRED refuses. Refusal on definitive ANS."),
    ("doc53_qa0__after146", 1.0, "GOLD 'Yes, decline ~42% FY2023 vs Q2 FY2024'. PRED 'Yes, $1,874M to $1,093M' ($1093/$1874=58%, so 42% drop). Correct."),
    ("doc25_qa0__after147", 1.0, "Same as doc25 after139. Correct."),
    ("doc24_qa0__after147", 0.5, "Same Amcor partial."),
    ("doc35_qa0__after147", 1.0, "GOLD 'In 2022 AMD operations brought most cashflow'. PRED 'Operations, $3,565M'. Correct identification."),
    ("doc22_qa0__after147", 1.0, "GOLD 'Amcor 8K Jul 1 2022: supplemental indentures, substitution of Amcor Flexibles NA for Amcor Finance USA on Guaranteed Senior Notes 2026/2028'. PRED 'Entry into supplemental indentures re: substitution of subsidiary for former issuer'. Correct summary."),
    ("doc117_qa0__after147", 1.0, "GOLD 'Nike FY2023 operations highest cashflow'. PRED 'Operations, $5,841M'. Correct."),
    ("doc26_qa0__after147", 0.0, "Same as doc26 after140. PRED refuses on definitive ANS."),
    ("doc141_qa0__after147", 0.0, "Same as doc141 after143. PRED 'Decrease' Y/N flip."),
    ("doc83_qa0__after147", 1.0, "GOLD $3215. PRED '$3,215.4 million'. Within 0.01% tolerance."),
    ("doc102_qa0__after147", 1.0, "GOLD 0.4%. PRED '0.4%'. Exact match (different from doc102_after142 where PRED was 0.1%)."),
    ("doc111_qa0__after147", 0.25, "GOLD 'No, decreased by $2.5bn'. PRED 'Yes, long-term debt decreased $47,032M to $41,990M' — internally contradictory ('Yes' label + decrease description). Numerically right direction but Y/N flip."),
    ("doc140_qa0__after148", 1.0, "GOLD 36% Ulta. PRED '36.5%'. Within 5% tolerance (1.4% relative)."),
    ("doc107_qa0__after148", 0.25, "GOLD 'Coverage ratio zero (negative EBIT)'. PRED '2.42'. Wrong (confident specific)."),
    ("doc38_qa0__after148", 0.0, "Same as doc38 after141. PRED hallucinates 'AXP NYSE Common Shares'. Confident wrong on 'none'."),
    ("doc59_qa0__after148", 0.0, "GOLD definitive $12645 Boeing FY2018 net PP&E. PRED refuses. Refusal on definitive ANS."),
    ("doc120_qa0__after148", 0.5, "GOLD 'NA/LatAm/Europe/Africa/ME/SAsia/AsiaPacific/Australia/NZ/China'. PRED 'US/Canada/LatAm/Europe'. Partial — misses Asia/Africa/ME entirely."),
    ("doc127_qa0__after148", 1.0, "Same as doc127 after142. Correct + breakdown."),
    ("doc77_qa0__after148", 0.75, "GOLD 'Yes, CVS multiple legal: usual pricing, opioids, PBM'. PRED 'Yes, named defendant re: retail pharmacies overcharging usual & customary'. Partial — one category correctly covered."),
    ("doc118_qa0__after148", 0.0, "GOLD definitive 'Yes PayPal positive WC $1.6Bn FY2022'. PRED refuses. Refusal on definitive ANS."),
    ("doc85_qa0__after148", 1.0, "GOLD 'No, sales grew 1.3% FY2022'. PRED 'No, sales growth 1.3%, EPS decline 13.8%'. Correct + EPS detail."),
    ("doc137_qa0__after148", 0.75, "GOLD 'Ulta did not make acquisitions FY2023/FY2022'. PRED 'passages do not mention any major acquisitions'. Both reach same conclusion via different framings — partial honesty (PRED says 'do not mention' rather than 'none' but answer-equivalent)."),
    ("doc90_qa0__after149", 1.0, "GOLD 'Consumer Health discontinued from Aug 30, 2023'. PRED matches verbatim with JnJ prefix. Correct."),
    ("doc82_qa0__after149", 1.0, "GOLD 0.68. PRED '0.68'. Exact match."),
    ("doc63_qa0__after149", 0.5, "Same Boeing customers partial."),
    ("doc109_qa0__after149", 1.0, "Same as doc109 after145. Correct."),
    ("doc61_qa0__after149", 1.0, "GOLD 'Yes, multiple lawsuits from 2018 Lion Air + 2019 Ethiopian Airlines crashes'. PRED matches with dates. Correct."),
    ("doc55_qa0__after149", 1.0, "Same Entertainment 9%. Correct."),
    ("doc80_qa0__after149", 1.0, "GOLD 'Yes, Richard A. Johnson'. PRED 'Yes, Richard A. Johnson 16,105,005 votes against'. Correct + vote count."),
    ("doc105_qa0__after149", 1.0, "Same MGM dividend. Correct."),
    ("doc108_qa0__after149", 1.0, "GOLD 'MGM China worst (-44%)'. PRED 'MGM China worst FY2022'. Correct identification."),
    ("doc128_qa0__after149", 1.0, "GOLD 'Pepsico strong start FY2023'. PRED 'Pepsico raised full-year guidance FY2023 due to strong start'. Correct."),
]


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
