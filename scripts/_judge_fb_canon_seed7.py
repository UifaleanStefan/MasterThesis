"""Apply Claude 1-by-1 judgments for financebench__v4t-canonical__batch__seed7 (n=150).

Multi-seed replicate of the FB v4t-CANONICAL cell (the weak, untuned theta), for
cross-seed variance. 103/150 predictions are byte-identical to seed 42 and inherit
that cell's 1-by-1 judgment (same gold + same answer => same score). The 47 that
differ (doc-order changed retrieval) are judged fresh in DIFF below; for the
canonical config these are overwhelmingly abstentions ("passages do not contain
...") because the untuned theta retrieves poorly -> 0.0 when the gold exists.
NO heuristics. judge_model=claude-opus-4.7-1m, rationale field.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELL = ROOT / "results" / "stage3" / "judge_queue" / "financebench__v4t-canonical__batch__seed7"
S42 = ROOT / "results" / "stage3" / "judge_queue" / "financebench__v4t-canonical__batch__seed42" / "results.jsonl"

# 47 entries whose seed-7 prediction differs from seed-42 -> judged fresh.
DIFF: dict[str, tuple[float, str]] = {
 "doc102_qa0": (0.0, "Gold 0.4% CAGR; pred abstains (no data)."),
 "doc104_qa0": (0.0, "Gold 7.9%; pred abstains."),
 "doc108_qa0": (0.0, "Gold MGM China; pred abstains."),
 "doc112_qa0": (0.0, "Gold 5.4%; pred abstains."),
 "doc116_qa0": (0.0, "Gold 3.46; pred abstains."),
 "doc11_qa0": (0.0, "Gold 65.4%; pred abstains."),
 "doc121_qa0": (0.75, "Conveys no material legal battles (= gold No)."),
 "doc123_qa0": (0.0, "Gold $9,068M; pred abstains."),
 "doc124_qa0": (0.0, "Gold 16.5%; pred abstains."),
 "doc127_qa0": (0.25, "Captures one $4.2bn agreement; gold total is $8.4bn -- wrong total."),
 "doc129_qa0": (1.0, "Matches gold (1 percentage point, 8->9%)."),
 "doc131_qa0": (0.75, "Correct event (Consumer Healthcare JV gain); amount detail muddled."),
 "doc13_qa0": (0.0, "Gold No (op margin declined); pred abstains."),
 "doc140_qa0": (0.75, "Pred 36.5% vs gold 36% -- close."),
 "doc145_qa0": (0.75, "Correct Yes (capital intensive); reasoning figure off."),
 "doc146_qa0": (0.0, "Gold No (debt decreased $229M); pred Yes increased -- wrong."),
 "doc148_qa0": (0.0, "Gold +0.2%; pred cannot compute (FY2019 missing) -- abstains."),
 "doc16_qa0": (0.0, "Gold 9.5x; pred abstains."),
 "doc17_qa0": (0.0, "Gold ROA -0.02; pred abstains."),
 "doc1_qa0": (0.0, "Gold $8.70; pred abstains."),
 "doc30_qa0": (0.0, "Gold 4.2%; pred abstains."),
 "doc34_qa0": (0.0, "Gold Xilinx amortization; pred abstains."),
 "doc35_qa0": (1.0, "Matches gold (operations brought in the most cash)."),
 "doc40_qa0": (0.0, "Gold 'not measured through operating margin'; pred abstains (no info) -- misses the insight."),
 "doc44_qa0": (1.0, "Matches gold (Yes, retention high)."),
 "doc49_qa0": (0.0, "Gold $5,409M; pred abstains."),
 "doc53_qa0": (0.0, "Gold Yes ~42% drop; pred abstains."),
 "doc56_qa0": (0.0, "Gold 1.73; pred abstains."),
 "doc65_qa0": (0.5, "Captures 737/787 production increase; misses gold 777X."),
 "doc67_qa0": (0.0, "Gold ROA 0.01; pred abstains."),
 "doc68_qa0": (0.0, "Gold 39.7%; pred abstains."),
 "doc69_qa0": (0.0, "Gold 0.8; pred abstains."),
 "doc70_qa0": (0.0, "Gold DPO 63.86; pred abstains."),
 "doc73_qa0": (0.0, "Gold Yes $831M; pred abstains."),
 "doc74_qa0": (0.0, "Gold $59,268M; pred $39,000 -- wrong."),
 "doc76_qa0": (0.0, "Gold Yes capital-intensive; pred abstains."),
 "doc79_qa0": (1.0, "Matches gold (Mary Dillon, ex-Ulta CEO)."),
 "doc81_qa0": (0.0, "Gold CCC -3.7; pred abstains."),
 "doc85_qa0": (1.0, "Matches gold (No, 1.3% growth)."),
 "doc86_qa0": (0.0, "Gold gives gross-margin drivers; pred abstains."),
 "doc89_qa0": (0.25, "Direction (US>intl) right but wrong numbers (pred US 1.3% is total, not the gold 3.0%/-0.6%)."),
 "doc8_qa0": (0.0, "Gold 24.26; pred abstains."),
 "doc92_qa0": (0.0, "Gold $13.2bn; pred says not provided -- abstains."),
 "doc95_qa0": (0.0, "Gold $66.56/share; pred abstains."),
 "doc96_qa0": (1.0, "Matches gold (gross margin not relevant for a financial institution)."),
 "doc99_qa0": (0.0, "Gold 6.25; pred abstains."),
 "doc9_qa0": (0.0, "Gold 1.9%; pred abstains."),
}


def main() -> int:
    # seed-42 canonical judgments, keyed by config-independent sfx
    s42 = {}
    for l in S42.read_text(encoding="utf-8").splitlines():
        if l.strip():
            j = json.loads(l)
            sfx = j["qid"].split("__batch__")[1].replace("__seed42", "")
            s42[sfx] = (float(j["judge_score"]), j.get("rationale", ""))
    queue = [json.loads(l) for l in (CELL / "queue.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out, missing = [], []
    for q in queue:
        sfx = q["qid"].split("__batch__")[1].replace("__seed7", "")
        if sfx in DIFF:
            score, rat = DIFF[sfx]
        elif sfx in s42:
            score, base = s42[sfx]
            rat = f"[identical prediction to seed42] {base}"
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
