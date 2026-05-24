"""Transfer Claude judgments from CUAD p4_main cells to identical-prediction
non-main cells (p4_tierb + k-sweep variants where the predicted answer
matches exactly).

The 100 doc_idx values are the same across CUAD evaluations (Phase 4 used
a single 100-doc CUAD subset). For configs like flat-50 the retrieval is
deterministic regardless of k, so the LLM's prediction is identical
across p4_main / p4_tierb / k-sweep buckets. For v4 configs the
retrieval changes with k (different graph state, different scoring),
so only some predictions match.

This script:
1. Walks every CUAD non-main queue.jsonl that has a matching p4_main
   queue.jsonl with claude judgments already produced.
2. For each entry, if predicted text matches the p4_main entry's
   predicted, transfer the corresponding judge_score + rationale.
3. Otherwise, leave the entry unjudged (must be fresh-judged later).
4. Writes results.jsonl with the transferred judgments, plus a
   per-cell summary stdout.

Run via:
    python scripts/transfer_cuad_duplicate_judgments.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QUEUE_ROOT = REPO_ROOT / "results" / "stage3" / "judge_queue"


def load_jsonl_map(path: Path, key_fn) -> dict:
    """Return {key_fn(entry): entry} mapping."""
    out = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            out[key_fn(entry)] = entry
    return out


def transfer_for_cell(non_main_dir: Path) -> tuple[int, int]:
    """Returns (transferred, unmatched) counts."""
    stem = non_main_dir.name  # e.g. p4_tierb__cuad__v4-canonical__seed42
    parts = stem.split("__", 1)
    suffix = parts[1] if len(parts) > 1 else None
    if suffix is None:
        return (0, 0)
    main_dir = QUEUE_ROOT / f"p4_main__{suffix}"
    main_queue = main_dir / "queue.jsonl"
    main_results = main_dir / "results.jsonl"
    if not main_queue.is_file() or not main_results.is_file():
        return (0, 0)

    # Load main predictions and judgments by qid suffix (qNNN)
    main_preds = load_jsonl_map(main_queue, lambda e: e["qid"].split("__")[-1])
    main_judges = load_jsonl_map(main_results, lambda e: e["qid"].split("__")[-1])

    # Load non-main queue
    non_main_queue = non_main_dir / "queue.jsonl"
    non_main_results = non_main_dir / "results.jsonl"
    if not non_main_queue.is_file():
        return (0, 0)

    # Skip if non-main results already has entries (don't clobber in-flight work)
    existing_judged = set()
    if non_main_results.is_file() and non_main_results.stat().st_size > 0:
        with non_main_results.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                existing_judged.add(entry["qid"])

    transferred = 0
    unmatched = 0
    skipped_existing = 0
    new_judgments = []
    with non_main_queue.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            qid = entry["qid"]
            if qid in existing_judged:
                skipped_existing += 1
                continue
            qkey = qid.split("__")[-1]
            main_entry = main_preds.get(qkey)
            main_judge = main_judges.get(qkey)
            if main_entry is None or main_judge is None:
                unmatched += 1
                continue
            # Only transfer if predictions are byte-identical
            if entry.get("predicted") == main_entry.get("predicted"):
                new_judgments.append({
                    "qid": qid,
                    "judge_score": main_judge["judge_score"],
                    "rationale": main_judge["rationale"] + " [transferred from p4_main identical prediction]",
                })
                transferred += 1
            else:
                unmatched += 1

    if new_judgments:
        with non_main_results.open("a", encoding="utf-8") as fh:
            for j in new_judgments:
                fh.write(json.dumps(j, ensure_ascii=False) + "\n")

    print(f"  {stem}: +{transferred} transferred, {unmatched} need fresh judging, {skipped_existing} already judged")
    return (transferred, unmatched)


def main() -> None:
    cuad_cells = sorted(
        d
        for d in QUEUE_ROOT.iterdir()
        if d.is_dir() and "__cuad__" in d.name and not d.name.startswith("p4_main__")
    )
    print(f"CUAD non-main cells: {len(cuad_cells)}")
    total_transferred = 0
    total_unmatched = 0
    for cell in cuad_cells:
        t, u = transfer_for_cell(cell)
        total_transferred += t
        total_unmatched += u
    print()
    print(f"TOTAL: {total_transferred} entries auto-transferred from p4_main")
    print(f"       {total_unmatched} entries still need fresh manual judging")


if __name__ == "__main__":
    main()
