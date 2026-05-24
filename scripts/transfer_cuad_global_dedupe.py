"""Globally deduplicate CUAD judgments by (qid_suffix, predicted_text).

After the byte-equality transfer from p4_main, ~1,178 CUAD entries remain
unjudged across k-sweep buckets. But many of those entries share the same
(question, predicted answer) tuple with an entry that IS already judged
in some other cell (e.g., p4_k4 v4-canonical predicting the same text as
p4_k8 v4-canonical for the same qid). Same (qid_suffix, predicted) tuple
deterministically deserves the same Claude judge_score.

This script:
1. Walks every CUAD cell's results.jsonl and queue.jsonl.
2. Builds {(qid_suffix, predicted_text): judgment} from ALL already-judged entries.
3. For every unjudged entry in every cell, checks if its key is in the dict;
   if so, transfers the judgment (annotated "[transferred from global CUAD dedupe]").
4. Reports per-cell transfer counts and the remaining unjudged tally.

Idempotent: re-running adds zero new entries once stable.

Run via:
    python scripts/transfer_cuad_global_dedupe.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QUEUE_ROOT = REPO_ROOT / "results" / "stage3" / "judge_queue"


def main() -> None:
    cuad_cells = sorted(
        d
        for d in QUEUE_ROOT.iterdir()
        if d.is_dir() and "__cuad__" in d.name
    )
    print(f"CUAD cells: {len(cuad_cells)}")

    # Step 1: build global judgment dictionary
    # key = (qid_suffix, predicted_text); value = (judge_score, rationale)
    judgments: dict[tuple[str, str], tuple[float, str]] = {}
    preds_by_cell: dict[str, dict[str, str]] = {}

    for d in cuad_cells:
        qfile = d / "queue.jsonl"
        rfile = d / "results.jsonl"
        if not qfile.is_file():
            continue

        cell_preds: dict[str, str] = {}
        with qfile.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                cell_preds[e["qid"]] = e["predicted"]
        preds_by_cell[d.name] = cell_preds

        if rfile.is_file() and rfile.stat().st_size > 0:
            with rfile.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    j = json.loads(line)
                    qid = j["qid"]
                    if qid not in cell_preds:
                        continue
                    pred = cell_preds[qid]
                    qsuffix = qid.split("__")[-1]
                    key = (qsuffix, pred)
                    if key not in judgments:
                        judgments[key] = (j["judge_score"], j["rationale"])

    print(f"Global (qid_suffix, predicted) judgments collected: {len(judgments)}")

    # Step 2: walk every cell, find unjudged entries with matching key, append
    total_transferred = 0
    total_remaining = 0
    for d in cuad_cells:
        cell_preds = preds_by_cell.get(d.name, {})
        rfile = d / "results.jsonl"

        existing_judged = set()
        if rfile.is_file() and rfile.stat().st_size > 0:
            with rfile.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        existing_judged.add(json.loads(line)["qid"])

        new_entries = []
        unjudged_remaining = 0
        for qid, pred in cell_preds.items():
            if qid in existing_judged:
                continue
            qsuffix = qid.split("__")[-1]
            j = judgments.get((qsuffix, pred))
            if j is not None:
                score, rationale = j
                new_entries.append({
                    "qid": qid,
                    "judge_score": score,
                    "rationale": rationale + " [transferred from global CUAD dedupe]"
                    if "transferred" not in rationale else rationale,
                })
            else:
                unjudged_remaining += 1

        if new_entries:
            with rfile.open("a", encoding="utf-8") as fh:
                for j in new_entries:
                    fh.write(json.dumps(j, ensure_ascii=False) + "\n")

        total_transferred += len(new_entries)
        total_remaining += unjudged_remaining
        if new_entries or unjudged_remaining:
            print(f"  {d.name}: +{len(new_entries)} transferred, {unjudged_remaining} still unjudged")

    print()
    print(f"TOTAL: +{total_transferred} transferred from global dedupe")
    print(f"       {total_remaining} entries still need fresh manual judging")


if __name__ == "__main__":
    main()
