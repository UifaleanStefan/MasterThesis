"""Extract Phase 4 FinanceBench per-question records into Claude-judge queues.

Reads results/stage3/cells/financebench__*.json (10 files) and writes one
queue.jsonl per cell to results/stage3/judge_queue/finbench_p4__{cfg}__{seed}/.

Each queue entry is enough for a stand-alone Claude judgment:
  qid, benchmark, config, seed, doc_idx, question, gold_answer, predicted,
  gpt4omini_judge_score (existing — for cross-judge comparison), retrieved_steps, k

Run via:
    python scripts/build_finbench_phase4_judge_queue.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CELLS_DIR = REPO_ROOT / "results" / "stage3" / "cells"
QUEUE_ROOT = REPO_ROOT / "results" / "stage3" / "judge_queue"


def main() -> None:
    fb_cells = sorted(CELLS_DIR.glob("financebench__*.json"))
    print(f"Found {len(fb_cells)} FinanceBench Phase 4 cells")

    total_entries = 0
    for cell_path in fb_cells:
        d = json.loads(cell_path.read_text(encoding="utf-8"))
        # Parse "financebench__{config}__seed{N}.json" filename
        stem = cell_path.stem  # e.g. "financebench__v4-canonical__seed42"
        m = re.match(r"financebench__(?P<config>.+)__seed(?P<seed>\d+)$", stem)
        if not m:
            print(f"  [SKIP] {stem!r}: bad filename format")
            continue
        config = m.group("config")
        seed = int(m.group("seed"))

        out_dir = QUEUE_ROOT / f"finbench_p4__{config}__seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        queue_path = out_dir / "queue.jsonl"
        results_path = out_dir / "results.jsonl"

        questions = d.get("questions", [])
        n_with_pred = 0
        with queue_path.open("w", encoding="utf-8") as fh:
            for q_idx, q in enumerate(questions):
                pred = q.get("predicted")
                gold = q.get("gold_answer")
                if pred is None or gold is None:
                    continue
                entry = {
                    "qid": f"finbench_p4__{config}__seed{seed}__q{q_idx:03d}",
                    "benchmark": "financebench",
                    "config": config,
                    "seed": seed,
                    "mode": "phase4_per_doc",
                    "doc_idx": q.get("doc_idx"),
                    "question": q.get("question", ""),
                    "gold_answer": gold,
                    "predicted": pred,
                    "gpt4omini_judge_score": q.get("judge_score"),  # preserve for comparison
                    "retrieved_steps": q.get("retrieved_steps", []),
                    "k": 8,
                }
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n_with_pred += 1
        # Don't overwrite results.jsonl if it exists (resume capability)
        if not results_path.exists():
            results_path.touch()
        total_entries += n_with_pred
        print(f"  [OK] {stem}: {n_with_pred} entries -> {out_dir.relative_to(REPO_ROOT)}")

    print()
    print(f"Total Phase 4 FB queue entries: {total_entries}")


if __name__ == "__main__":
    main()
