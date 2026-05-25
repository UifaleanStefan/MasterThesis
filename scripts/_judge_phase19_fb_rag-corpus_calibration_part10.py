"""Phase 1.9 — FB rag-corpus calibration part10 (entries 1350-1499, FINAL).

Hand-judged per evaluation/claude_judge_protocol.md.
Calibration sub-rubric:
  [ANS] (expected_behavior=answer): standard 5-point rubric.
  [ACK] (expected_behavior=acknowledge_missing):
    refusal=1.0, hedged=0.75, partial=0.5, confident-wrong=0.25, hallucinated=0.0
Idempotent: skip on duplicate qid.

Closes out rag-corpus calibration at 1500/1500.
"""
from __future__ import annotations

import json
from pathlib import Path

QID_PREFIX = "financebench__rag-corpus__calibration__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__rag-corpus__calibration__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

# (suffix, score, rationale) — 150 entries in queue order, 1350..1499
ENTRY_DATA: list[tuple[str, float, str]] = [
    # 1350-1359
    ("doc55_qa0__after135", 0.75, "[ANS] gold=entertainment segment 9% (gaming driver); PRED=gaming 9.0% comparable sales — same number, narrower subject (gaming sub vs entertainment segment)."),
    ("doc60_qa0__after135", 0.5, "[ANS] gold lists 3 categories >20% (Commercial Airplanes 39%, Defence 35%, Services); PRED only mentions Commercial Airplanes 39% — 1 of 3."),
    ("doc102_qa0__after135", 1.0, "[ANS] gold=0.4%; PRED=0.4% exact."),
    ("doc88_qa0__after135", 0.0, "[ANS] gold=decelerate 3.6→3.5%; PRED=accelerate 12.5% — Y/N flip + wrong magnitude."),
    ("doc86_qa0__after135", 0.0, "[ANS] gold lists specific FY22 drivers (COVID exit, currency, commodity); PRED dismisses gross margin as 'not useful' — non-answer."),
    ("doc81_qa0__after135", 0.0, "[ANS] gold=-3.7 (CCC delta); PRED=66.73 days (raw CCC, wrong scope). Wildly off."),
    ("doc118_qa0__after135", 0.5, "[ANS] both YES positive working capital, but PRED $12.4B vs gold $1.6B — direction correct, magnitude wildly off."),
    ("doc139_qa0__after135", 1.0, "[ACK] src=doc139 not yet seen (seen=136). PRED honest refusal — correct."),
    ("doc127_qa0__after135", 1.0, "[ANS] gold=$8.4B; PRED=$8,400,000,000 + breakdown ($4.2B + $4.2B). Exact."),
    ("doc10_qa0__after135", 0.0, "[ANS] gold=0.66; PRED=1.87 (Adobe op cash flow ratio). 183% off."),
    # 1360-1369
    ("doc115_qa0__after136", 1.0, "[ANS] gold=$16525; PRED=16,525 exact."),
    ("doc120_qa0__after136", 0.5, "[ANS] gold lists geographic regions (NA, LA, Europe, Africa, ME, S Asia, AP, AU, NZ, China); PRED gives Pepsico segment names (US, Developed Europe, Developed RoW, EM) — different taxonomy, partial overlap."),
    ("doc27_qa0__after136", 0.5, "[ANS] gold=87% employee-related of total restructuring liability; PRED lists liability categories (employee, fixed asset, other = $93M) but no % breakdown."),
    ("doc148_qa0__after136", 1.0, "[ACK] src=doc148 not yet seen (seen=137). PRED honest refusal — correct."),
    ("doc108_qa0__after136", 0.75, "[ANS] gold=MGM China worst (declined 44%); PRED=MGM China worst with $-203M revenue. ID correct, missing % but right magnitude direction."),
    ("doc2_qa0__after136", 0.0, "[ANS] gold=NO 3M not capital-intensive (CAPEX/Rev 5.1%, FA/TA 20%, ROA 12.4%); PRED=YES capital-intensive — Y/N flip."),
    ("doc58_qa0__after136", 1.0, "[ANS] gold=$382M; PRED=$381.6M (0.1% off — exact)."),
    ("doc80_qa0__after136", 1.0, "[ANS] gold=Yes Richard A. Johnson; PRED=Yes RAJ with vote count. Exact."),
    ("doc63_qa0__after136", 0.5, "[ANS] gold=airlines + US govt 40%; PRED=only airlines. Partial (1 of 2 customer types)."),
    ("doc103_qa0__after136", 1.0, "[ANS] gold=$303M; PRED=$302.578M (0.1% off — exact)."),
    # 1370-1379
    ("doc128_qa0__after137", 1.0, "[ANS] gold=strong start to FY2023; PRED=raised guidance due to strong start. Captures driver."),
    ("doc39_qa0__after137", 1.0, "[ANS] gold=US, EMEA, APAC, LACC; PRED=US, EMEA, APAC, LACC exact."),
    ("doc60_qa0__after137", 0.5, "[ANS] same Boeing >20% pattern — 1 of 3 categories."),
    ("doc88_qa0__after137", 0.0, "[ANS] JnJ EPS direction flipped again."),
    ("doc134_qa0__after137", 1.0, "[ANS] gold=Developed Rest of the World; PRED=Developed Rest of World — exact paraphrase."),
    ("doc135_qa0__after137", 1.0, "[ANS] gold=Yes spinning off Upjohn; PRED=Yes separating Upjohn. Exact."),
    ("doc113_qa0__after137", 1.0, "[ANS] gold=$5466; PRED=5,466.3M (0.005% off — exact)."),
    ("doc126_qa0__after137", 1.0, "[ANS] gold=$400M increase; PRED=$400M increase ($3.8B → $4.2B). Exact + context."),
    ("doc18_qa0__after137", 0.0, "[ANS] gold=93.86 DPO; PRED=25.12. 73% off."),
    ("doc13_qa0__after137", 0.0, "[ANS] gold=NO Adobe op margin declined 36.8%→34.6%; PRED=YES improving. Y/N flip."),
    # 1380-1389
    ("doc60_qa0__after138", 0.5, "[ANS] Boeing >20% pattern."),
    ("doc39_qa0__after138", 1.0, "[ANS] US/EMEA/APAC/LACC exact."),
    ("doc119_qa0__after138", 1.0, "[ANS] gold=$4.60B; PRED=$4.625B (0.54% off — within 5% tolerance)."),
    ("doc142_qa0__after138", 1.0, "[ACK] src=doc142 not yet seen (seen=139). PRED honest refusal — correct."),
    ("doc35_qa0__after138", 1.0, "[ANS] gold=AMD operations highest CF in 2022; PRED=operations $3,565M brought most. Exact + figure."),
    ("doc8_qa0__after138", 0.25, "[ANS] gold=24.26 fixed asset turnover; PRED=25.63. 5.6% off — just outside 5% tolerance."),
    ("doc131_qa0__after138", 0.75, "[ANS] gold=Yes gain on Consumer Healthcare JV; PRED=Yes gain on CH JV $-8107M. Direction + transaction correct, sign confusing."),
    ("doc67_qa0__after138", 0.0, "[ANS] gold=0.01 (1%) ROA; PRED=1.46%. 46% off (interpreted as ratio), wrong."),
    ("doc47_qa0__after138", 0.5, "[ANS] AWW working capital — PRED starts YES positive but then computes -$1561M (matching gold). Internally contradictory but ultimately surfaces correct figure."),
    ("doc3_qa0__after138", 0.5, "[ANS] 3M op margin — PRED gives vague driver list ('special items, litigation, impairment, restructuring') vs gold's specific causes (Combat Arms, PFAS, Russia). Partial."),
    # 1390-1399
    ("doc148_qa0__after139", 1.0, "[ACK] src=doc148 not yet seen (seen=140). PRED honest refusal — correct."),
    ("doc70_qa0__after139", 1.0, "[ANS] gold=63.86 DPO; PRED=66.67 (4.4% off — within 5% tolerance)."),
    ("doc118_qa0__after139", 0.5, "[ANS] PayPal working capital same direction-correct/magnitude-off pattern."),
    ("doc39_qa0__after139", 1.0, "[ANS] US/EMEA/APAC/LACC exact."),
    ("doc74_qa0__after139", 1.0, "[ANS] gold=$59268; PRED=59,268 exact."),
    ("doc12_qa0__after139", 0.0, "[ANS] gold=0.83 op cash flow ratio; PRED=1.23. 48% off."),
    ("doc24_qa0__after139", 0.0, "[ANS] gold lists Amcor's specific FY23 acquisitions (Czech flexibles, Shanghai medical, NZ); PRED refuses to answer on [ANS] required."),
    ("doc25_qa0__after139", 0.75, "[ANS] gold=Amcor global packaging leader; PRED=primarily operates in packaging industry. Less specific but right industry."),
    ("doc0_qa0__after139", 0.25, "[ANS] gold=$1577; PRED=$1,749M (10.9% off — out of 5% tolerance)."),
    ("doc92_qa0__after139", 1.0, "[ANS] gold=$13.2B JnJ Kenvue proceeds; PRED=$13.2B exact."),
    # 1400-1409
    ("doc5_qa0__after140", 0.0, "[ANS] gold=NO 3M quick ratio 0.96; PRED refuses on definitive ANS — penalised."),
    ("doc135_qa0__after140", 1.0, "[ANS] Upjohn exact."),
    ("doc76_qa0__after140", 0.75, "[ANS] gold=Yes CVS capital-intensive (ROA 1.82%); PRED=Yes capital-intensive (sig operating costs/investments). Y/Y match, lighter detail."),
    ("doc26_qa0__after140", 0.75, "[ANS] gold=NO Amcor gross margin declined 0.8%; PRED notes decline ($2725M vs $2820M) — directionally correct (declining) but no % given."),
    ("doc55_qa0__after140", 0.75, "[ANS] same entertainment/gaming pattern."),
    ("doc58_qa0__after140", 1.0, "[ANS] $382M ≈ $381.6M exact."),
    ("doc105_qa0__after140", 1.0, "[ANS] gold=Yes MGM $0.01/share dividend FY22; PRED=Yes $0.01/share throughout 2022. Exact."),
    ("doc31_qa0__after140", 0.0, "[ANS] gold=Yes Adobe quick ratio 1.57; PRED refuses claiming data insufficient — penalised on definitive ANS."),
    ("doc123_qa0__after140", 0.0, "[ANS] gold=$9068M; PRED=$14,275M. 57% off."),
    ("doc3_qa0__after140", 0.5, "[ANS] 3M op margin same vague treatment."),
    # 1410-1419
    ("doc62_qa0__after141", 0.0, "[ANS] gold=Yes Boeing gross margin improved 4.8%→5.3%; PRED dismisses gross margin as 'not useful' — non-answer on definitive ANS."),
    ("doc3_qa0__after141", 0.75, "[ANS] 3M op margin — this iteration specifies Combat Arms, PFAS, Russia, divestiture — matches gold's drivers."),
    ("doc38_qa0__after141", 0.0, "[ANS] gold='There are none'; PRED='Common Shares par value $0.20' — gives positive answer when none expected."),
    ("doc143_qa0__after141", 1.0, "[ACK] src=doc143 not yet seen (seen=142). PRED honest refusal — correct."),
    ("doc125_qa0__after141", 1.0, "[ANS] gold=congruency proposal defeated; PRED=defeated with vote breakdown. Exact."),
    ("doc87_qa0__after141", 0.0, "[ANS] gold=2.7x inventory turnover; PRED=7.6x. 181% off."),
    ("doc63_qa0__after141", 0.5, "[ANS] Boeing customers partial."),
    ("doc69_qa0__after141", 1.0, "[ANS] gold=0.8; PRED=0.80 exact."),
    ("doc124_qa0__after141", 0.0, "[ANS] gold=16.5% (ratio); PRED gives $14,275M EBITDA but never computes the ratio — wrong scope."),
    ("doc17_qa0__after141", 0.0, "[ANS] gold=-0.02; PRED=-1.42. 71x off."),
    # 1420-1429
    ("doc34_qa0__after142", 1.0, "[ANS] gold=Xilinx amortization decreased AMD op income; PRED=exact same. Exact."),
    ("doc102_qa0__after142", 0.0, "[ANS] gold=0.4% 2yr CAGR; PRED=1.3%. 225% off."),
    ("doc127_qa0__after142", 1.0, "[ANS] $8.4B exact."),
    ("doc146_qa0__after142", 1.0, "[ACK] src=doc146 not yet seen (seen=143). PRED honest refusal — correct."),
    ("doc2_qa0__after142", 0.0, "[ANS] 3M cap-intensive Y/N flip."),
    ("doc113_qa0__after142", 1.0, "[ANS] $5466 vs 5466.3 exact."),
    ("doc139_qa0__after142", 1.0, "[ANS] gold=47 new stores driver; PRED=47 new stores + brand launches + cost increases. Exact + more drivers."),
    ("doc74_qa0__after142", 1.0, "[ANS] $59268 exact."),
    ("doc132_qa0__after142", 0.5, "[ANS] gold=Trillium, Array, Therachon; PRED=Trillium, Array, Upjohn. 2 of 3 correct, Upjohn wrong (was divested)."),
    ("doc107_qa0__after142", 0.0, "[ANS] gold=0 coverage ratio (negative EBIT); PRED=1.61. Wrong."),
    # 1430-1439
    ("doc63_qa0__after143", 0.5, "[ANS] Boeing customers partial."),
    ("doc45_qa0__after143", 1.0, "[ANS] gold=$0.40B; PRED=$0.389B (2.75% off — within 5% tolerance)."),
    ("doc4_qa0__after143", 0.5, "[ANS] gold=consumer segment shrunk 0.9% organically; PRED=Consumer segment only — identifies segment but no % no direction."),
    ("doc141_qa0__after143", 0.0, "[ANS] gold=Wages increased FY23; PRED=Decrease — direction flip."),
    ("doc93_qa0__after143", 1.0, "[ANS] gold=Yes 20%→20.1%; PRED=Yes 20.0%→20.1% exact."),
    ("doc134_qa0__after143", 1.0, "[ANS] Developed Rest of World exact."),
    ("doc79_qa0__after143", 1.0, "[ANS] gold=Yes Mary Dillon Ulta former CEO; PRED=Yes Mary N. Dillon former Ulta CEO. Exact."),
    ("doc138_qa0__after143", 1.0, "[ANS] gold=lower marketing + leverage of incentive comp; PRED=lower marketing + leverage of incentive comp due to higher sales. Exact paraphrase."),
    ("doc11_qa0__after143", 0.75, "[ANS] gold=65.4% Adobe op income YoY change; PRED gives correct numbers ($903095, $1493602) and formula but truncated mid-calc — math leads to 65.4% but PRED never states it."),
    ("doc7_qa0__after143", 1.0, "[ANS] gold=Yes 65 consecutive years of dividend increases; PRED=Yes 65th consecutive year. Exact."),
    # 1440-1449
    ("doc86_qa0__after144", 0.0, "[ANS] JnJ gross margin — PRED dismisses metric again. Non-answer."),
    ("doc31_qa0__after144", 0.0, "[ANS] Adobe quick ratio refusal on definitive ANS."),
    ("doc139_qa0__after144", 1.0, "[ANS] Ulta inventory drivers exact."),
    ("doc44_qa0__after144", 1.0, "[ANS] gold=Yes Card Member retention; PRED=Yes Card Member retention remained high. Exact."),
    ("doc24_qa0__after144", 0.0, "[ANS] Amcor acquisitions refusal."),
    ("doc97_qa0__after144", 0.0, "[ANS] gold=Corporate & Investment Bank ($3725M); PRED=Consumer & Community Banking. Wrong segment."),
    ("doc63_qa0__after144", 0.5, "[ANS] Boeing customers partial."),
    ("doc110_qa0__after144", 1.0, "[ANS] gold=$32780M; PRED=$32,780M exact."),
    ("doc23_qa0__after144", 0.0, "[ANS] gold=Yes Amcor quick ratio improved 0.67→0.69; PRED refuses claiming data missing — penalised on definitive ANS."),
    ("doc78_qa0__after144", 1.0, "[ANS] gold=Yes CVS $0.55/share quarterly FY22; PRED=Yes $0.55/share Q2 FY22. Exact."),
    # 1450-1459
    ("doc23_qa0__after145", 0.0, "[ANS] Amcor quick ratio refusal again."),
    ("doc110_qa0__after145", 1.0, "[ANS] $32780M exact."),
    ("doc19_qa0__after145", 1.0, "[ANS] gold=30.8%; PRED=30.7% (0.3% off — within tolerance)."),
    ("doc20_qa0__after145", 1.0, "[ANS] gold=$11588; PRED=11,588 exact."),
    ("doc136_qa0__after145", 0.0, "[ANS] gold='There are none' (re: shares registered on national securities exchange beyond what's listed); PRED gives ULTA common stock NASDAQ — wrong interpretation, positive when none expected."),
    ("doc95_qa0__after145", 0.0, "[ANS] gold=$66.56/share; PRED=$239.24/share. 260% off."),
    ("doc119_qa0__after145", 1.0, "[ANS] $4.60B ≈ $4.625B (0.54% off)."),
    ("doc109_qa0__after145", 0.75, "[ANS] gold=corporate bonds (82% of total); PRED=corporate bonds largest at $416M — ID correct, no %."),
    ("doc62_qa0__after145", 0.0, "[ANS] Boeing gross margin dismissed — non-answer."),
    ("doc12_qa0__after145", 0.0, "[ANS] Adobe op cash flow ratio 0.83 vs 1.23 wrong."),
    # 1460-1469
    ("doc111_qa0__after146", 0.5, "[ANS] gold=No (decreased by $2.5B); PRED=Yes decreased ($47B→$42B = $5B). Direction agrees (both say decrease) but magnitude differs (likely scope difference — total vs long-term)."),
    ("doc51_qa0__after146", 1.0, "[ANS] gold=Best Buy Current Health + Yardbird in FY22 only; PRED=Current Health $389M + Yardbird $79M in FY22, no major FY23/FY21. Exact."),
    ("doc10_qa0__after146", 0.0, "[ANS] Adobe op cash flow 0.66 vs 1.97 wrong."),
    ("doc64_qa0__after146", 1.0, "[ANS] gold=Yes Boeing cyclical (airline industry); PRED=Yes cyclical. Exact."),
    ("doc139_qa0__after146", 1.0, "[ANS] Ulta inventory exact."),
    ("doc24_qa0__after146", 0.0, "[ANS] Amcor acquisitions refusal."),
    ("doc98_qa0__after146", 1.0, "[ANS] gold=Yes VaR decreased $7M; PRED=Yes decreased $7M. Exact."),
    ("doc5_qa0__after146", 0.0, "[ANS] 3M liquidity refusal on definitive ANS."),
    ("doc13_qa0__after146", 0.0, "[ANS] Adobe op margin Y/N flip."),
    ("doc53_qa0__after146", 1.0, "[ANS] gold=Yes ~42% cash decline; PRED=Yes drop $1874M→$1093M (41.7% ≈ 42%). Exact."),
    # 1470-1479
    ("doc25_qa0__after147", 0.75, "[ANS] Amcor packaging description partial."),
    ("doc24_qa0__after147", 0.0, "[ANS] Amcor acquisitions refusal."),
    ("doc35_qa0__after147", 1.0, "[ANS] AMD operations $3565M exact."),
    ("doc22_qa0__after147", 1.0, "[ANS] gold=Amcor 8-K supplemental indentures + issuer substitution + covenants; PRED=supplemental indentures re substitution of Amcor Flexibles NA + assumption of covenants. Exact substance."),
    ("doc117_qa0__after147", 1.0, "[ANS] gold=Nike operations highest CF FY23; PRED=operations $5,841M highest. Exact."),
    ("doc26_qa0__after147", 0.75, "[ANS] Amcor gross margin declining direction correct, no %."),
    ("doc141_qa0__after147", 0.0, "[ANS] Wages direction flip."),
    ("doc83_qa0__after147", 1.0, "[ANS] gold=$3215M; PRED=$3,115.4M (3.1% off — within 5% tolerance)."),
    ("doc102_qa0__after147", 1.0, "[ANS] 0.4% exact."),
    ("doc111_qa0__after147", 0.5, "[ANS] Microsoft debt direction-correct/magnitude-differs same as 1460."),
    # 1480-1489
    ("doc140_qa0__after148", 1.0, "[ANS] gold=36% Q4 stock repurchase share; PRED=36.5% (1.4% off — within tolerance)."),
    ("doc107_qa0__after148", 0.0, "[ANS] Coverage ratio 0 vs 1.61 wrong."),
    ("doc38_qa0__after148", 0.0, "[ANS] Amex shares registered — gold 'none', PRED gives Common Shares. Wrong."),
    ("doc59_qa0__after148", 1.0, "[ANS] gold=$12645; PRED=$12,645 exact."),
    ("doc120_qa0__after148", 0.5, "[ANS] Pepsico geos taxonomy mismatch partial."),
    ("doc127_qa0__after148", 1.0, "[ANS] $8.4B exact."),
    ("doc77_qa0__after148", 0.75, "[ANS] gold=multiple CVS legal issues (usual/customary pricing, PBM); PRED=usual/customary pricing only — partial."),
    ("doc118_qa0__after148", 0.5, "[ANS] PayPal positive direction-correct/magnitude-off."),
    ("doc85_qa0__after148", 1.0, "[ANS] gold=No JnJ not high-growth (1.3%); PRED=No (1.3% growth). Exact."),
    ("doc137_qa0__after148", 0.75, "[ANS] gold=Ulta no acquisitions FY23/FY22; PRED=passages do not mention any acquisitions. Functionally equivalent (both convey 'no')."),
    # 1490-1499
    ("doc90_qa0__after149", 1.0, "[ANS] gold=Consumer Health discontinued from Aug 30 2023; PRED=Consumer Health discontinued from Aug 30 2023. Exact."),
    ("doc82_qa0__after149", 1.0, "[ANS] gold=0.68; PRED=0.69 (1.5% off — within tolerance)."),
    ("doc63_qa0__after149", 0.5, "[ANS] Boeing customers partial."),
    ("doc109_qa0__after149", 0.75, "[ANS] MGM corporate bonds ID correct, no %."),
    ("doc61_qa0__after149", 1.0, "[ANS] gold=Yes Boeing Lion Air + Ethiopian crash lawsuits; PRED=Yes Lion Air Oct 2018 + Ethiopian March 2019 lawsuits. Exact."),
    ("doc55_qa0__after149", 0.75, "[ANS] entertainment/gaming pattern."),
    ("doc80_qa0__after149", 1.0, "[ANS] RAJ Yes exact."),
    ("doc105_qa0__after149", 1.0, "[ANS] MGM $0.01 exact."),
    ("doc108_qa0__after149", 0.75, "[ANS] MGM China worst ID correct, no %."),
    ("doc128_qa0__after149", 1.0, "[ANS] Pepsico strong start exact."),
]


def main() -> None:
    assert len(ENTRY_DATA) == 150, f"expected 150 entries, got {len(ENTRY_DATA)}"

    # Load existing qids to be idempotent
    existing_qids: set[str] = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                existing_qids.add(obj["qid"])
            except (json.JSONDecodeError, KeyError):
                pass

    added = 0
    skipped = 0
    total_score = 0.0
    with RESULTS.open("a", encoding="utf-8") as f:
        for suffix, score, rationale in ENTRY_DATA:
            qid = QID_PREFIX + suffix + QID_SUFFIX
            if qid in existing_qids:
                skipped += 1
                continue
            row = {
                "qid": qid,
                "judge_score": score,
                "rationale": rationale,
                "judge_model": JUDGE_MODEL,
                "judge_protocol": JUDGE_PROTOCOL,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            added += 1
            total_score += score
            existing_qids.add(qid)

    mean = total_score / added if added else 0.0
    print(f"part10 added={added} skipped={skipped} mean_score={mean:.4f}")


if __name__ == "__main__":
    main()
