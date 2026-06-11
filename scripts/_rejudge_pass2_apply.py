"""
Pass-2 re-judge applier (critique remediation, Track C1).

Pass 1 wrote template/heuristic scores for ~9,486 content-judgment entries
(see results/stage3/_judge_workdir/rejudge_targets.json). Pass 2 is Claude
re-reading every one of those entries individually and re-scoring per the
5-point rubric in evaluation/claude_judge_protocol.md.

This script ONLY persists already-made judgments: it reads
results/stage3/_judge_workdir/rejudge_scores__<cell>__partNN.json files
({qid: [score, rationale]}), rewrites the matching lines of the cell's
results.jsonl in place (all other lines untouched), and stamps each
rewritten line with judge_pass="2_manual" so the provenance audit can
report first-pass vs re-judged shares. It contains NO scoring logic.

Usage:
    python scripts/_rejudge_pass2_apply.py <cell-name> [<cell-name> ...]
    python scripts/_rejudge_pass2_apply.py --all   # every cell with score files
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WD = ROOT / "results" / "stage3" / "_judge_workdir"
JQ = ROOT / "results" / "stage3" / "judge_queue"

VALID = {0.0, 0.25, 0.5, 0.75, 1.0}


def collect_scores(cell: str) -> dict[str, tuple[float, str]]:
    scores: dict[str, tuple[float, str]] = {}
    for f in sorted(WD.glob(f"rejudge_scores__{cell}__part*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for qid, (score, rationale) in data.items():
            if float(score) not in VALID:
                raise ValueError(f"{f.name}: invalid score {score} for {qid}")
            if not str(rationale).strip():
                raise ValueError(f"{f.name}: empty rationale for {qid}")
            scores[qid] = (float(score), str(rationale))
    return scores


def apply_cell(cell: str) -> dict:
    scores = collect_scores(cell)
    if not scores:
        return {"cell": cell, "status": "no score files"}
    path = JQ / cell / "results.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    n_updated = 0
    deltas = []
    out_lines = []
    for line in lines:
        rec = json.loads(line)
        qid = rec.get("qid")
        if qid in scores:
            new_score, rationale = scores[qid]
            deltas.append(new_score - float(rec.get("judge_score", 0.0)))
            rec["judge_score"] = new_score
            rec["rationale"] = rationale
            rec["judge_model"] = "claude-opus-4.7-1m"
            rec["judge_pass"] = "2_manual"
            n_updated += 1
        out_lines.append(json.dumps(rec, ensure_ascii=False))
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    return {
        "cell": cell,
        "n_scored": len(scores),
        "n_updated": n_updated,
        "n_missing_in_results": len(scores) - n_updated,
        "mean_score_delta": round(mean_delta, 4),
        "n_changed": sum(1 for d in deltas if abs(d) > 1e-9),
    }


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    if args == ["--all"]:
        cells = sorted({
            f.name.split("__part")[0].removeprefix("rejudge_scores__")
            for f in WD.glob("rejudge_scores__*__part*.json")
        })
    else:
        cells = args
    for cell in cells:
        info = apply_cell(cell)
        print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
