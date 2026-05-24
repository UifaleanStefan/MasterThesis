"""Generic byte-equality judgment-transfer for any benchmark.

Walks every cell whose dir-name contains `__<bench>__`, builds a global
{(qid_suffix, predicted_text): (judge_score, rationale)} from already-
judged entries, and propagates judgments wherever the (qid, predicted)
tuple matches.

Idempotent. Works for any benchmark in {cuad, qasper, hotpotqa,
longmemeval, narrativeqa, financebench, ...}.

Usage:
    python scripts/transfer_bench_global_dedupe.py <benchmark>

Example:
    python scripts/transfer_bench_global_dedupe.py hotpotqa
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QUEUE_ROOT = REPO_ROOT / "results" / "stage3" / "judge_queue"


def main(bench: str) -> None:
    cells = sorted(
        d
        for d in QUEUE_ROOT.iterdir()
        if d.is_dir() and f"__{bench}__" in d.name
    )
    print(f"{bench} cells: {len(cells)}")

    judgments: dict[tuple[str, str], tuple[float, str]] = {}
    preds_by_cell: dict[str, dict[str, str]] = {}

    for d in cells:
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

    total_transferred = 0
    total_remaining = 0
    for d in cells:
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
                    "rationale": rationale + f" [transferred from global {bench} dedupe]"
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
    print(f"TOTAL: +{total_transferred} transferred from global {bench} dedupe")
    print(f"       {total_remaining} entries still need fresh manual judging")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/transfer_bench_global_dedupe.py <benchmark>")
    main(sys.argv[1])
