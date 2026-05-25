"""Phase 1.9 — FB rag-corpus batch_calib (end-of-corpus 150 questions).

All [ANS] expected_behavior. Hand-judged per evaluation/claude_judge_protocol.md.
Idempotent (skip-on-duplicate by qid).
"""
from __future__ import annotations

import json
from pathlib import Path

QID_PREFIX = "financebench__rag-corpus__batch__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/financebench__rag-corpus__batch_calib__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

# (suffix, score, rationale) — 150 entries doc0..doc149
ENTRY_DATA: list[tuple[str, float, str]] = [
    ("doc0_qa0", 1.0, "[ANS] gold=$1577 capex; pred=$1,501M (4.8% off — within 5% tolerance)."),
    ("doc1_qa0", 0.25, "[ANS] gold=$8.70B PPNE; pred=$9.211B (5.9% off — just outside tolerance)."),
    ("doc2_qa0", 0.0, "[ANS] gold=NO 3M not cap-intensive (CAPEX/Rev 5.1%); pred=YES — Y/N flip."),
    ("doc3_qa0", 0.5, "[ANS] gold=specific 3M FY22 drivers (Combat Arms, PFAS, Russia); pred=vague (special items, litigation, impairment). Partial."),
    ("doc4_qa0", 0.5, "[ANS] gold=consumer segment shrunk 0.9% organically; pred=Consumer segment only. Identifies segment but no %/direction."),
    ("doc5_qa0", 0.0, "[ANS] gold=NO 3M quick ratio 0.96; pred=refuses on definitive ANS — penalised."),
    ("doc6_qa0", 1.0, "[ANS] gold=1.500% Notes 2026, 1.750% Notes 2030, 1.500% Notes 2031; pred=exact 3 notes."),
    ("doc7_qa0", 1.0, "[ANS] gold=Yes 65 consecutive years; pred=Yes 65th consecutive year. Exact."),
    ("doc8_qa0", 0.25, "[ANS] gold=24.26; pred=25.73 (6.1% off — just outside tolerance)."),
    ("doc9_qa0", 0.0, "[ANS] gold=1.9% 3yr avg capex/rev; pred=3.5% (84% off)."),
    ("doc10_qa0", 0.0, "[ANS] gold=0.66 op CF ratio; pred=1.98 (200% off)."),
    ("doc11_qa0", 0.75, "[ANS] gold=65.4% Adobe op income YoY; pred shows correct numbers + diff $590,507 but truncated before % calc."),
    ("doc12_qa0", 0.0, "[ANS] gold=0.83; pred=1.23 (48% off)."),
    ("doc13_qa0", 0.0, "[ANS] gold=NO Adobe op margin declined 36.8→34.6%; pred=YES improving — Y/N flip."),
    ("doc14_qa0", 0.0, "[ANS] gold=Yes Adobe FCF improved 143%→156%; pred=refuses on definitive ANS — penalised."),
    ("doc15_qa0", 1.0, "[ANS] gold=0; pred=0 exact."),
    ("doc16_qa0", 0.5, "[ANS] gold=9.5x inventory turnover; pred shows correct COGS/inventory data but truncated before final ratio."),
    ("doc17_qa0", 0.0, "[ANS] gold=-0.02 ROA; pred=-1.42 (71x off)."),
    ("doc18_qa0", 0.0, "[ANS] gold=93.86 DPO; pred=36.12 (61% off)."),
    ("doc19_qa0", 1.0, "[ANS] gold=30.8%; pred=30.7% (0.3% off — exact)."),
    ("doc20_qa0", 1.0, "[ANS] gold=$11588; pred=11,588 exact."),
    ("doc21_qa0", 1.0, "[ANS] gold=$1616M; pred=$1,615.9M (0.006% off — exact)."),
    ("doc22_qa0", 1.0, "[ANS] gold=Amcor 8-K supplemental indentures + issuer substitution; pred=same substance. Exact."),
    ("doc23_qa0", 0.0, "[ANS] gold=Yes Amcor quick ratio improved 0.67→0.69; pred=refuses on definitive ANS."),
    ("doc24_qa0", 0.0, "[ANS] gold=Amcor specific FY23 acquisitions (Czech, Shanghai, NZ); pred=refuses on definitive ANS."),
    ("doc25_qa0", 0.75, "[ANS] gold=Amcor packaging leader; pred=packaging + food/beverage/pharma details — more specific, captures industry."),
    ("doc26_qa0", 0.75, "[ANS] gold=NO Amcor gross margin declined 0.8%; pred=declining ($2,725 vs $2,820) — direction correct, no %."),
    ("doc27_qa0", 0.5, "[ANS] gold=87% employee of restructuring liability; pred lists categories ($93M total) but no % breakdown."),
    ("doc28_qa0", 1.0, "[ANS] gold=$2,018M Amcor Adj EBITDA; pred=$2,018M exact."),
    ("doc29_qa0", 0.0, "[ANS] gold=flat Real Growth Amcor; pred=decrease 5% — wrong direction."),
    ("doc30_qa0", 1.0, "[ANS] gold=4.2% AMD D&A margin; pred=4.18% (0.5% off — exact)."),
    ("doc31_qa0", 0.0, "[ANS] gold=Yes AMD quick ratio 1.57; pred=refuses on definitive ANS."),
    ("doc32_qa0", 1.0, "[ANS] gold=AMD CPUs/GPUs/DPUs/FPGAs/SoCs/APUs; pred=exact list."),
    ("doc33_qa0", 1.0, "[ANS] gold=EPYC + semi-custom + Xilinx Embedded; pred=Data Center (EPYC) + Gaming (semi-custom) + Embedded. Exact substance."),
    ("doc34_qa0", 1.0, "[ANS] gold=Xilinx amortization; pred=Xilinx amortization exact."),
    ("doc35_qa0", 1.0, "[ANS] gold=AMD operations highest CF; pred=operations $3,565M. Exact."),
    ("doc36_qa0", 0.0, "[ANS] gold=Data Center (largest segment increase); pred=Gaming (21%) — wrong segment."),
    ("doc37_qa0", 1.0, "[ANS] gold=Yes 16% customer concentration; pred=Yes 16% exact."),
    ("doc38_qa0", 0.0, "[ANS] gold='There are none'; pred=Common Shares — wrong (positive answer when none expected)."),
    ("doc39_qa0", 1.0, "[ANS] gold=US/EMEA/APAC/LACC; pred=exact."),
    ("doc40_qa0", 1.0, "[ANS] gold=op margin not measured for AMEX; pred=op margin not useful for AMEX. Exact paraphrase."),
    ("doc41_qa0", 1.0, "[ANS] gold=gross margin not measured for AMEX; pred=gross margin not useful. Exact paraphrase."),
    ("doc42_qa0", 1.0, "[ANS] gold=24.6%→21.6%; pred=24.6%→21.6% exact."),
    ("doc43_qa0", 0.0, "[ANS] gold=Customer deposits (largest AMEX liability); pred=Long-term debt $42,573M — wrong category."),
    ("doc44_qa0", 1.0, "[ANS] gold=Yes Card Member retention; pred=Yes high retention. Exact."),
    ("doc45_qa0", 1.0, "[ANS] gold=$0.40B; pred=$0.389B (2.75% off — within tolerance)."),
    ("doc46_qa0", 1.0, "[ANS] gold=$1832M; pred=1,829M (0.16% off — exact)."),
    ("doc47_qa0", 0.5, "[ANS] AWW pattern — PRED leads YES then computes -$1561M (matching gold's negative). Internally contradictory but surfaces correct figure."),
    ("doc48_qa0", 0.0, "[ANS] gold=2.8% net profit margin; pred=3.9% (39% off)."),
    ("doc49_qa0", 1.0, "[ANS] gold=$5409M; pred=5,409 exact."),
    ("doc50_qa0", 0.0, "[ANS] gold=Yes Best Buy gross margins consistent (1.1% decline); pred=No fluctuates >2% — Y/N flip."),
    ("doc51_qa0", 1.0, "[ANS] gold=Best Buy Current Health + Yardbird FY22; pred=Current Health $389M + Yardbird $79M FY22, no FY23/FY21. Exact."),
    ("doc52_qa0", 1.0, "[ANS] gold=Best Buy operations $1.8B; pred=Operations $1,824M. Exact."),
    ("doc53_qa0", 1.0, "[ANS] gold=Yes ~42% cash decline; pred=Yes $1874→$1093M (41.7% ≈ 42%). Exact."),
    ("doc54_qa0", 0.5, "[ANS] gold=Yes Best Buy stores decreased 982→969 (1.32%); pred=Yes 930→907. Direction correct but specific store counts wrong (likely total vs domestic scope diff)."),
    ("doc55_qa0", 0.75, "[ANS] entertainment/gaming 9% pattern."),
    ("doc56_qa0", 1.0, "[ANS] gold=1.73 Block working cap ratio; pred=1.74 (0.58% off — exact)."),
    ("doc57_qa0", 1.0, "[ANS] gold=101.5% Block growth; pred=102.0% (0.49% off — exact)."),
    ("doc58_qa0", 1.0, "[ANS] gold=$382M; pred=$381.6M (0.1% off — exact)."),
    ("doc59_qa0", 1.0, "[ANS] gold=$12645M; pred=$12,645 exact."),
    ("doc60_qa0", 0.5, "[ANS] Boeing >20% — only Commercial Airplanes mentioned, gold lists 3."),
    ("doc61_qa0", 1.0, "[ANS] Boeing Lion Air + Ethiopian Airlines crash lawsuits. Exact."),
    ("doc62_qa0", 0.0, "[ANS] gold=Yes Boeing gross margin improved 4.8→5.3%; pred=dismisses as 'not useful' — non-answer on definitive ANS."),
    ("doc63_qa0", 0.5, "[ANS] Boeing customers airlines only (no US govt). Partial."),
    ("doc64_qa0", 1.0, "[ANS] gold=Yes cyclical (airline industry); pred=Yes cyclical. Exact."),
    ("doc65_qa0", 0.5, "[ANS] gold=Boeing increasing 737/777X/787 production; pred=787 increase + 737 increase + 777X PAUSED (not increase). Partial — 2 of 3 directional matches."),
    ("doc66_qa0", 0.0, "[ANS] gold=0.62% vs -14.76% (FY22 HIGHER than FY21 since -14.76 < 0.62); pred='lower in FY22' — directionally wrong."),
    ("doc67_qa0", 0.0, "[ANS] gold=0.01 ROA Coca-Cola; pred=1.46% — wrong scale."),
    ("doc68_qa0", 1.0, "[ANS] gold=39.7% COGS margin; pred=39.7% exact."),
    ("doc69_qa0", 1.0, "[ANS] gold=0.8 dividend payout; pred=0.80 exact."),
    ("doc70_qa0", 1.0, "[ANS] gold=63.86 DPO Corning; pred=66.67 (4.4% off — within tolerance)."),
    ("doc71_qa0", 1.0, "[ANS] gold=10.3% op income margin Corning; pred=10.5% (1.9% off — within tolerance)."),
    ("doc72_qa0", 1.0, "[ANS] gold=20%→23% Corning ETR; pred=20%→23% exact."),
    ("doc73_qa0", 0.5, "[ANS] gold=Yes Corning $831M working capital (operating items only); pred=Yes $2,278M (all current items). Direction correct, magnitude differs."),
    ("doc74_qa0", 1.0, "[ANS] gold=$59268M Costco; pred=59,268 exact."),
    ("doc75_qa0", 0.0, "[ANS] gold=17.98 CVS fixed asset turnover; pred=8.73 (51% off)."),
    ("doc76_qa0", 0.75, "[ANS] gold=Yes CVS cap-intensive (ROA 1.82%); pred=Yes cap-intensive (sig operating costs/investments). Y/Y match, lighter detail."),
    ("doc77_qa0", 0.75, "[ANS] gold=multiple CVS lawsuits (usual/customary, PBM); pred=usual/customary only. Partial."),
    ("doc78_qa0", 1.0, "[ANS] gold=Yes CVS $0.55/qtr FY22; pred=Yes $0.55/qtr Q2 FY22. Exact."),
    ("doc79_qa0", 1.0, "[ANS] gold=Yes Mary Dillon former Ulta CEO; pred=Yes Mary N. Dillon Ulta CEO. Exact."),
    ("doc80_qa0", 1.0, "[ANS] Richard A. Johnson exact."),
    ("doc81_qa0", 0.0, "[ANS] gold=-3.7 CCC delta; pred=66.73 days (raw CCC) — wrong scope."),
    ("doc82_qa0", 1.0, "[ANS] gold=0.68; pred=0.69 (1.5% off — exact)."),
    ("doc83_qa0", 1.0, "[ANS] gold=$3215M FCF; pred=$3,115.4M (3.1% off — within tolerance)."),
    ("doc84_qa0", 0.25, "[ANS] gold=0.54 retention ratio; pred=0.46 (14.8% off — out of tolerance)."),
    ("doc85_qa0", 1.0, "[ANS] gold=No JnJ not high-growth (1.3%); pred=No (1.3%). Exact."),
    ("doc86_qa0", 0.0, "[ANS] JnJ gross margin — pred dismisses metric. Non-answer on definitive ANS."),
    ("doc87_qa0", 0.0, "[ANS] gold=2.7x JnJ inventory turnover; pred=refuses. Non-answer."),
    ("doc88_qa0", 0.0, "[ANS] JnJ EPS decel vs accel — Y/N flip."),
    ("doc89_qa0", 1.0, "[ANS] gold=US 3.0% vs Intl -0.6%; pred=US 3.0% vs Intl -0.6% exact."),
    ("doc90_qa0", 1.0, "[ANS] gold=Consumer Health discontinued Aug 30 2023; pred=exact."),
    ("doc91_qa0", 1.0, "[ANS] gold=~$20B Kenvue gain; pred=~$20B exact."),
    ("doc92_qa0", 1.0, "[ANS] $13.2B Kenvue proceeds exact."),
    ("doc93_qa0", 1.0, "[ANS] gold=20%→20.1%; pred=20.0%→20.1% exact."),
    ("doc94_qa0", 0.0, "[ANS] gold=Corporate (lowest revenue -$473M); pred=Consumer & Community Banking — wrong segment."),
    ("doc95_qa0", 0.0, "[ANS] gold=$66.56/share; pred=$239.45/share (260% off)."),
    ("doc96_qa0", 1.0, "[ANS] gold=JPM gross margin not relevant; pred=not relevant. Exact paraphrase."),
    ("doc97_qa0", 0.0, "[ANS] gold=Corporate & Investment Bank (highest net income); pred=Consumer & Community Banking — wrong segment."),
    ("doc98_qa0", 1.0, "[ANS] gold=Yes VaR decreased $7M; pred=Yes decreased $7M exact."),
    ("doc99_qa0", 0.0, "[ANS] gold=6.25 Kraft Heinz inventory turnover; pred=3.06 (51% off)."),
    ("doc100_qa0", 1.0, "[ANS] gold=1.33; pred=1.38 (3.8% off — within tolerance)."),
    ("doc101_qa0", 1.0, "[ANS] gold=$5818M Lockheed NWC; pred=5,818M exact."),
    ("doc102_qa0", 0.0, "[ANS] gold=0.4% Lockheed CAGR; pred=1.3% (225% off)."),
    ("doc103_qa0", 1.0, "[ANS] $303M MGM AP exact."),
    ("doc104_qa0", 0.0, "[ANS] gold=7.9% MGM capex/rev 3yr; pred=-3.5% (wrong sign + off)."),
    ("doc105_qa0", 1.0, "[ANS] MGM $0.01 dividend exact."),
    ("doc106_qa0", 0.75, "[ANS] gold=Las Vegas resorts ~90% EBITDAR; pred=Las Vegas Strip Resorts — ID correct, no %."),
    ("doc107_qa0", 0.0, "[ANS] gold=0 (negative EBIT); pred=1.61 — wrong."),
    ("doc108_qa0", 0.75, "[ANS] MGM China worst ID correct, no %."),
    ("doc109_qa0", 0.75, "[ANS] Corporate bonds ID correct, no %."),
    ("doc110_qa0", 1.0, "[ANS] $32780M MSFT COGS exact."),
    ("doc111_qa0", 0.5, "[ANS] Microsoft debt direction-correct/magnitude-differs."),
    ("doc112_qa0", 0.25, "[ANS] gold=5.4% Netflix EBITDA margin; pred=4.5% (17% off)."),
    ("doc113_qa0", 1.0, "[ANS] $5466 Netflix current liab exact."),
    ("doc114_qa0", 1.0, "[ANS] gold=55.1% Nike margin; pred=56.3% (2.2% off — within tolerance)."),
    ("doc115_qa0", 1.0, "[ANS] $16525 Nike current assets exact."),
    ("doc116_qa0", 1.0, "[ANS] gold=3.46 Nike inventory turnover; pred=3.61 (4.3% off — within tolerance)."),
    ("doc117_qa0", 1.0, "[ANS] Nike operations $5841M exact."),
    ("doc118_qa0", 0.5, "[ANS] PayPal positive direction-correct/magnitude-off ($1.6B vs $12.4B)."),
    ("doc119_qa0", 1.0, "[ANS] $4.60B PepsiCo capex; pred=$4.625B (0.54% off — within tolerance)."),
    ("doc120_qa0", 0.5, "[ANS] Pepsico geos taxonomy mismatch — segment names vs geographic."),
    ("doc121_qa0", 0.75, "[ANS] gold=No PepsiCo not in material legal battles; pred=PepsiCo party to litigation but management believes no material adverse effect — same substantive conclusion."),
    ("doc122_qa0", 1.0, "[ANS] gold=$411M PepsiCo restructuring; pred=411 exact."),
    ("doc123_qa0", 0.0, "[ANS] gold=$9068M PepsiCo EBITDA less capex; pred=$14,275M (57% off — doesn't subtract capex)."),
    ("doc124_qa0", 0.0, "[ANS] gold=16.5% EBITDA margin; pred gives EBITDA $14,275M but doesn't compute the ratio — wrong scope."),
    ("doc125_qa0", 1.0, "[ANS] PepsiCo congruency proposal defeated. Exact."),
    ("doc126_qa0", 1.0, "[ANS] $400M PepsiCo credit increase exact."),
    ("doc127_qa0", 1.0, "[ANS] $8.4B PepsiCo total borrowing exact."),
    ("doc128_qa0", 1.0, "[ANS] PepsiCo strong start to FY23. Exact paraphrase."),
    ("doc129_qa0", 1.0, "[ANS] gold=1pp PepsiCo EPS guidance raise; pred=1pp (from 8% to 9%). Exact."),
    ("doc130_qa0", 0.25, "[ANS] gold=Yes Pfizer PPNE positive YoY; pred=Yes but uses net income figures (not PPNE) — wrong metric, right direction."),
    ("doc131_qa0", 0.25, "[ANS] gold=Yes Pfizer JV gain 2019; pred=Yes JV gain but specifies 2021 with $-6M — wrong year and amount."),
    ("doc132_qa0", 0.5, "[ANS] Pfizer acquisitions — 2 of 3 correct (Trillium, Array), Upjohn wrong (was divested)."),
    ("doc133_qa0", 0.0, "[ANS] gold=$77.78M Upjohn spinoff cost; pred=$700M (800% off)."),
    ("doc134_qa0", 1.0, "[ANS] Developed Rest of World exact."),
    ("doc135_qa0", 1.0, "[ANS] Pfizer separating Upjohn exact."),
    ("doc136_qa0", 0.0, "[ANS] gold=Ulta 'There are none' (no debt securities); pred=Ulta common stock NASDAQ — wrong interpretation, positive answer when none expected."),
    ("doc137_qa0", 0.75, "[ANS] gold=Ulta no acquisitions FY23/FY22; pred=passages do not mention any acquisitions. Functionally equivalent."),
    ("doc138_qa0", 1.0, "[ANS] gold=Ulta SG&A reduction (marketing + incentive comp); pred=same + deleverage details. Exact + more."),
    ("doc139_qa0", 1.0, "[ANS] Ulta inventory 47 new stores driver exact."),
    ("doc140_qa0", 1.0, "[ANS] 36% Q4 stock repurchase exact (1.4% off)."),
    ("doc141_qa0", 0.0, "[ANS] gold=Ulta wages increased FY23; pred=Decrease — direction flip."),
    ("doc142_qa0", 0.75, "[ANS] gold=Cross currency swaps $32,502M; pred=Cross currency swaps only (no $). ID correct."),
    ("doc143_qa0", 0.5, "[ANS] gold=$1097M pension + $862M health benefits; pred=$1,097M (pension only). Partial — 1 of 2."),
    ("doc144_qa0", 0.0, "[ANS] gold=No Verizon quick ratio 0.54; pred=refuses on definitive ANS — penalised."),
    ("doc145_qa0", 0.75, "[ANS] gold=Yes Verizon cap-intensive (ratio 2.77); pred=Yes (PP&E $307,689M). Y/Y match, different ratio metric."),
    ("doc146_qa0", 0.5, "[ANS] gold=No Verizon debt decreased $229M; pred=Yes but text shows decrease $229M ($150,868→$150,639). Internally contradictory; substantive $ matches gold."),
    ("doc147_qa0", 0.0, "[ANS] gold=42.69 Walmart DPO; pred=30.73 (28% off)."),
    ("doc148_qa0", 0.0, "[ANS] gold=0.2% Walmart op margin change; pred=-6.0% (wrong sign + off)."),
    ("doc149_qa0", 0.0, "[ANS] gold=6.2% Walmart EBITDA margin; pred=3.8% (39% off)."),
]


def main() -> None:
    assert len(ENTRY_DATA) == 150, f"expected 150 entries, got {len(ENTRY_DATA)}"

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
    print(f"batch_calib added={added} skipped={skipped} mean_score={mean:.4f}")


if __name__ == "__main__":
    main()
