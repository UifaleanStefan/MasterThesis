"""Apply Claude 1-by-1 judgments for financebench__v4t-canonical__batch__seed100 (n=150).

Third seed of the FB v4t-CANONICAL cell. 98/150 predictions are byte-identical to
seed 42 and inherit that cell's 1-by-1 judgment; the 52 that differ are judged
fresh in DIFF below (again overwhelmingly abstentions for the untuned theta).
NO heuristics. judge_model=claude-opus-4.7-1m, rationale field.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELL = ROOT / "results" / "stage3" / "judge_queue" / "financebench__v4t-canonical__batch__seed100"
S42 = ROOT / "results" / "stage3" / "judge_queue" / "financebench__v4t-canonical__batch__seed42" / "results.jsonl"

DIFF: dict[str, tuple[float, str]] = {
 "doc100_qa0": (0.0, "Gold 1.33; pred abstains."),
 "doc104_qa0": (0.0, "Gold 7.9%; pred abstains."),
 "doc106_qa0": (0.0, "Gold Las Vegas ~90% EBITDAR; pred abstains."),
 "doc107_qa0": (0.0, "Gold zero (negative adj. EBIT); pred abstains."),
 "doc10_qa0": (0.0, "Gold 0.66; pred abstains."),
 "doc112_qa0": (0.0, "Gold 5.4%; pred abstains."),
 "doc116_qa0": (0.0, "Gold 3.46; pred 4.29 -- wrong."),
 "doc121_qa0": (0.75, "Conveys no material legal battles (= gold No)."),
 "doc124_qa0": (0.0, "Gold 16.5%; pred abstains."),
 "doc127_qa0": (0.25, "Captures one $4.2bn agreement; gold total is $8.4bn -- wrong total."),
 "doc128_qa0": (1.0, "Matches gold (strong start to FY2023)."),
 "doc12_qa0": (0.0, "Gold 0.83; pred abstains."),
 "doc130_qa0": (1.0, "Matches gold (Yes, PP&E grew)."),
 "doc131_qa0": (0.75, "Correct event (Consumer Healthcare JV gain); amount detail muddled."),
 "doc13_qa0": (0.0, "Gold No (op margin declined); pred abstains."),
 "doc140_qa0": (0.75, "Pred 36.5% vs gold 36% -- close."),
 "doc145_qa0": (0.75, "Correct Yes (capital intensive); reasoning figure off."),
 "doc146_qa0": (0.0, "Gold No (debt decreased $229M); pred Yes increased -- wrong."),
 "doc147_qa0": (0.0, "Gold DPO 42.69; pred 36.73 -- wrong."),
 "doc148_qa0": (0.0, "Gold +0.2%; pred cannot compute (FY2019 missing) -- abstains."),
 "doc16_qa0": (0.0, "Gold 9.5x; pred abstains."),
 "doc17_qa0": (0.0, "Gold ROA -0.02; pred abstains."),
 "doc1_qa0": (0.0, "Gold $8.70; pred abstains."),
 "doc21_qa0": (0.0, "Gold $1,616M net AR; pred abstains."),
 "doc30_qa0": (0.0, "Gold 4.2%; pred abstains."),
 "doc34_qa0": (0.0, "Gold Xilinx amortization; pred abstains."),
 "doc35_qa0": (1.0, "Matches gold (operations brought in the most cash)."),
 "doc40_qa0": (0.0, "Gold 'not measured through operating margin'; pred abstains."),
 "doc41_qa0": (0.0, "Gold 'not measured through gross margin'; pred abstains."),
 "doc42_qa0": (0.0, "Gold 24.6->21.6%; pred abstains."),
 "doc49_qa0": (0.0, "Gold $5,409M; pred abstains."),
 "doc53_qa0": (0.0, "Gold Yes ~42% drop; pred abstains."),
 "doc55_qa0": (0.0, "Gold Entertainment/gaming; pred abstains."),
 "doc56_qa0": (0.0, "Gold 1.73; pred abstains."),
 "doc65_qa0": (0.5, "Captures 737/787 production increase; misses gold 777X."),
 "doc67_qa0": (0.0, "Gold ROA 0.01; pred abstains."),
 "doc68_qa0": (0.0, "Gold 39.7%; pred abstains."),
 "doc69_qa0": (0.0, "Gold 0.8; pred abstains."),
 "doc70_qa0": (0.0, "Gold DPO 63.86; pred abstains."),
 "doc73_qa0": (0.0, "Gold Yes $831M; pred abstains."),
 "doc74_qa0": (0.0, "Gold $59,268M; pred $39,000 -- wrong."),
 "doc75_qa0": (0.0, "Gold 17.98; pred abstains."),
 "doc76_qa0": (0.0, "Gold Yes capital-intensive; pred abstains."),
 "doc81_qa0": (0.0, "Gold CCC -3.7; pred abstains."),
 "doc83_qa0": (0.0, "Gold $3,215M FCF; pred abstains."),
 "doc86_qa0": (0.0, "Gold gross-margin drivers; pred abstains."),
 "doc89_qa0": (0.25, "Direction (US>intl) right but wrong numbers (pred US 1.3% is total, not gold 3.0%/-0.6%)."),
 "doc90_qa0": (1.0, "Matches gold (Consumer Health discontinued from 30 Aug 2023)."),
 "doc95_qa0": (0.0, "Gold $66.56/share; pred abstains."),
 "doc96_qa0": (1.0, "Matches gold (gross margin not relevant for a financial institution)."),
 "doc99_qa0": (0.0, "Gold 6.25; pred abstains."),
 "doc9_qa0": (0.0, "Gold 1.9%; pred abstains."),
}


def main() -> int:
    s42 = {}
    for l in S42.read_text(encoding="utf-8").splitlines():
        if l.strip():
            j = json.loads(l)
            sfx = j["qid"].split("__batch__")[1].replace("__seed42", "")
            s42[sfx] = (float(j["judge_score"]), j.get("rationale", ""))
    queue = [json.loads(l) for l in (CELL / "queue.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out, missing = [], []
    for q in queue:
        sfx = q["qid"].split("__batch__")[1].replace("__seed100", "")
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
