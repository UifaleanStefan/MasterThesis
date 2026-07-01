"""Merge Workflow part-score files into results.jsonl for head-to-head cells.

Reads results/stage3/_judge_workdir/rejudge_scores__<cell>__part*.json (each a
{qid: [score, rationale]} object written by the judging Workflow's verify stage)
and writes results/stage3/judge_queue/<cell>/results.jsonl with the standard
schema (judge_model, judge_protocol, rationale). Validates queue<->scores parity
and the {0,0.25,0.5,0.75,1.0} score domain.

Usage: python scripts/_h2h_merge.py <cell1> <cell2> ...
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WD = ROOT / "results" / "stage3" / "_judge_workdir"
QUEUE = ROOT / "results" / "stage3" / "judge_queue"
VALID = {0.0, 0.25, 0.5, 0.75, 1.0}


def merge_cell(cell: str) -> tuple[int, float]:
    scores: dict[str, tuple[float, str]] = {}
    for pf in sorted(WD.glob(f"rejudge_scores__{cell}__part*.json")):
        obj = json.loads(pf.read_text(encoding="utf-8"))
        for qid, sr in obj.items():
            score = float(sr[0])
            rat = str(sr[1]) if len(sr) > 1 else ""
            if score not in VALID:
                raise SystemExit(f"{cell}: bad score {score} for {qid}")
            scores[qid] = (score, rat)
    queue = [json.loads(l) for l in (QUEUE / cell / "queue.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out, missing = [], []
    for q in queue:
        qid = q["qid"]
        if qid not in scores:
            missing.append(qid); continue
        s, r = scores[qid]
        out.append({"qid": qid, "judge_score": s, "rationale": r,
                    "judge_model": "claude-opus-4.7-1m", "judge_protocol": "v1"})
    if missing:
        raise SystemExit(f"{cell}: missing {len(missing)} judgments, e.g. {missing[:5]}")
    (QUEUE / cell / "results.jsonl").write_text("\n".join(json.dumps(o) for o in out) + "\n", encoding="utf-8")
    mean = sum(o["judge_score"] for o in out) / len(out)
    return len(out), mean


def main() -> int:
    for cell in sys.argv[1:]:
        n, mean = merge_cell(cell)
        print(f"{cell}: wrote {n} judgments; mean judge = {mean:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
