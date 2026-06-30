"""Apply Claude 1-by-1 judgments for financebench__v4t-corpus-tuned__batch__seed100 (n=150).

Third multi-seed replicate. 103/150 predictions are byte-identical to seed 7
(end-of-corpus batch retrieval is seed-invariant for those questions), so those
carry the seed-7 judgment by definition (same gold + same answer => same score).
The 47 predictions that differ (doc-order changed retrieval) are judged fresh
here, 1-by-1 by Claude against the FinanceBench rubric, in DIFF below.
NO heuristics. Writes results.jsonl with judge_model + judge_protocol + rationale.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from _judge_fb_v4tct_seed7 import J as J7  # seed-7 1-by-1 judgments (all 150)

CELL = ROOT / "results" / "stage3" / "judge_queue" / "financebench__v4t-corpus-tuned__batch__seed100"

# 47 entries whose seed-100 prediction differs from seed-7 -> judged fresh.
DIFF: dict[str, tuple[float, str]] = {
 "doc0_qa0": (0.0, "Gold capex $1577M; pred $1,749M -- wrong."),
 "doc101_qa0": (1.0, "Matches gold ($5,818M)."),
 "doc102_qa0": (1.0, "Matches gold (0.4% CAGR)."),
 "doc10_qa0": (0.0, "Gold 0.66; pred 1.98 -- wrong."),
 "doc110_qa0": (1.0, "Matches gold ($32,780M COGS)."),
 "doc115_qa0": (1.0, "Matches gold ($16,525M)."),
 "doc118_qa0": (0.75, "Correct Yes (positive working capital); gold gives $1.6bn, pred affirms without the figure."),
 "doc11_qa0": (1.0, "Pred's calc = 65.4% = gold."),
 "doc124_qa0": (1.0, "Matches gold (16.5% EBITDA margin)."),
 "doc125_qa0": (1.0, "Matches gold (net-zero proposal defeated)."),
 "doc12_qa0": (0.0, "Gold 0.83; pred 1.25 -- wrong."),
 "doc136_qa0": (1.0, "Matches gold (no debt securities registered)."),
 "doc138_qa0": (1.0, "Captures gold drivers (lower marketing + incentive-comp leverage) plus detail."),
 "doc13_qa0": (0.0, "Gold No (operating margin declined 36.8->34.6); pred Yes improving -- opposite."),
 "doc147_qa0": (0.75, "Pred computes DPO 42.52 vs gold 42.69 -- correct method, minor input rounding."),
 "doc148_qa0": (0.0, "Gold +0.2%; pred -1.3% -- wrong."),
 "doc16_qa0": (0.0, "Gold 9.5x; pred 12.00x -- wrong (~26% off)."),
 "doc18_qa0": (0.0, "Gold DPO 93.86; pred 25.36 -- wrong."),
 "doc1_qa0": (0.0, "Gold $8.70; pred 9.211bn -- off ~6%, wrong."),
 "doc23_qa0": (0.0, "Gold improved 0.67->0.69; pred says cannot determine -- abstains."),
 "doc24_qa0": (0.0, "Gold lists FY2023 acquisitions; pred says no info -- abstains."),
 "doc27_qa0": (0.5, "Captures employee component; misses gold 87% concentration."),
 "doc2_qa0": (0.0, "Gold No (capex efficient); pred Yes capital-intensive -- opposite."),
 "doc30_qa0": (1.0, "Pred 4.18% = gold 4.2%."),
 "doc31_qa0": (0.0, "Gold quick ratio 1.57 (Yes); pred says not provided -- abstains."),
 "doc36_qa0": (0.0, "Gold Data Center; pred Gaming -- wrong segment."),
 "doc40_qa0": (1.0, "Matches gold (operating margin not the right metric for AMEX)."),
 "doc41_qa0": (1.0, "Matches gold (gross margin not the right metric for AMEX)."),
 "doc46_qa0": (0.0, "Gold $1,832M; pred 1,896 -- off ~3.5%, wrong."),
 "doc47_qa0": (0.5, "Math + final 'negative working capital' = -$1,561M (matches gold) but opens with a contradictory 'Yes'."),
 "doc48_qa0": (0.0, "Gold 2.8%; pred 3.9% -- wrong."),
 "doc50_qa0": (0.0, "Gold Yes consistent (1.1% decline); pred says fluctuated >2% -- opposite (its own 21.4/22.5/22.4 are actually <2%)."),
 "doc52_qa0": (1.0, "Matches gold (operating, ~$1.8bn)."),
 "doc54_qa0": (0.5, "Correct direction (decline) but numbers off (966/977 vs gold 969/982)."),
 "doc57_qa0": (0.75, "Pred 101.7% vs gold 101.5% -- close."),
 "doc70_qa0": (0.0, "Gold DPO 63.86; pred 66.67 -- wrong."),
 "doc79_qa0": (1.0, "Matches gold (Mary Dillon, ex-Ulta CEO)."),
 "doc82_qa0": (0.75, "Pred 0.69 vs gold 0.68 -- rounding."),
 "doc84_qa0": (0.75, "Pred 0.55 vs gold 0.54 -- minor diff."),
 "doc85_qa0": (1.0, "Matches gold (No, 1.3% growth)."),
 "doc86_qa0": (0.0, "Gold gives drivers (COVID exit, currency, commodity inflation); pred dodges saying not useful -- wrong."),
 "doc87_qa0": (0.25, "Headline 7.6x wrong; pred's own inputs compute to 2.72x = gold 2.7."),
 "doc8_qa0": (0.0, "Gold 24.26; pred 2.56 -- wrong."),
 "doc90_qa0": (1.0, "Matches gold (Consumer Health discontinued from 30 Aug 2023)."),
 "doc95_qa0": (0.0, "Gold $66.56/share; pred $239.80 -- wrong."),
 "doc96_qa0": (1.0, "Matches gold (gross margin not relevant for a financial institution)."),
 "doc99_qa0": (0.0, "Gold 6.25; pred 3.06 -- wrong."),
}


def main() -> int:
    queue = [json.loads(l) for l in (CELL / "queue.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out, missing = [], []
    for q in queue:
        sfx = q["qid"].split("__batch__")[1].replace("__seed100", "")
        if sfx in DIFF:
            score, rat = DIFF[sfx]
        elif sfx in J7:
            score, rat = J7[sfx]  # identical prediction -> seed-7 judgment applies
        else:
            missing.append(sfx); continue
        out.append({"qid": q["qid"], "judge_score": score, "rationale": rat,
                    "judge_model": "claude-opus-4.7-1m", "judge_protocol": "v1"})
    if missing:
        raise SystemExit(f"missing {len(missing)}: {missing[:10]}")
    (CELL / "results.jsonl").write_text("\n".join(json.dumps(o) for o in out) + "\n", encoding="utf-8")
    mean = sum(o["judge_score"] for o in out) / len(out)
    print(f"wrote {len(out)} judgments ({len(DIFF)} fresh, {len(out)-len(DIFF)} inherited); mean judge = {mean:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
