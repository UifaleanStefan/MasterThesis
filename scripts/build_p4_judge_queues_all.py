"""Build Claude-judge queues for ALL non-FinanceBench Phase 4 + Phase 1.7 cells.

Replaces GPT-4o-mini auto-judge with manual cross-vendor Claude Opus 4.7 max
1-by-1 judging per `evaluation/claude_judge_protocol.md`. Walks every
cells/ + cells_k{4,8,16,32}/ + cells_k16_seed{7,100}/ + cells_tier_b/
directory, skips financebench (already done), and emits one queue.jsonl
per cell to results/stage3/judge_queue/{bucket_tag}__{cell_stem}/.

bucket_tag is one of:
    p4_main      — cells/{bench}__{config}__seed{seed}.json (Phase 4 + Phase 1.7 supp)
    p4_k4 / p4_k8 / p4_k16 / p4_k32 — cells_k{k}/ (k-sweep cells)
    p4_k16_s7 / p4_k16_s100        — cells_k16_seed7/, cells_k16_seed100/
    p4_tierb    — cells_tier_b/

Each queue entry is enough for stand-alone judgment:
    qid, bucket, benchmark, config, seed, k, doc_idx, question, gold_answer,
    predicted, gpt4omini_judge_score (preserved), retrieved_steps

Run via:
    python scripts/build_p4_judge_queues_all.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE3 = REPO_ROOT / "results" / "stage3"
QUEUE_ROOT = STAGE3 / "judge_queue"

# Map (cells_dir → bucket_tag, default_k)
BUCKETS = [
    ("cells", "p4_main", 8),
    ("cells_k4", "p4_k4", 4),
    ("cells_k8", "p4_k8", 8),
    ("cells_k16", "p4_k16", 16),
    ("cells_k32", "p4_k32", 32),
    ("cells_k16_seed7", "p4_k16_s7", 16),
    ("cells_k16_seed100", "p4_k16_s100", 16),
    ("cells_tier_b", "p4_tierb", 8),
]

CELL_RE = re.compile(r"(?P<bench>[a-z]+)__(?P<config>.+)__seed(?P<seed>\d+)$")


def build_for_dir(cells_subdir: str, bucket_tag: str, default_k: int) -> int:
    cells_dir = STAGE3 / cells_subdir
    if not cells_dir.is_dir():
        return 0

    total = 0
    n_cells = 0
    for cell_path in sorted(cells_dir.glob("*.json")):
        stem = cell_path.stem
        # Skip FinanceBench cells — already cross-vendor judged
        if stem.startswith("financebench"):
            continue
        m = CELL_RE.match(stem)
        if not m:
            print(f"  [SKIP] {bucket_tag}/{stem}: bad filename format")
            continue
        bench = m.group("bench")
        config = m.group("config")
        seed = int(m.group("seed"))

        d = json.loads(cell_path.read_text(encoding="utf-8"))
        questions = d.get("questions", [])
        if not questions:
            continue

        out_dir = QUEUE_ROOT / f"{bucket_tag}__{stem}"
        out_dir.mkdir(parents=True, exist_ok=True)
        queue_path = out_dir / "queue.jsonl"
        results_path = out_dir / "results.jsonl"

        # Skip if queue exists and is non-empty — don't clobber in-flight work
        if queue_path.exists() and queue_path.stat().st_size > 0:
            print(f"  [SKIP-EXISTS] {bucket_tag}__{stem}: queue.jsonl already present")
            continue

        n_pred = 0
        with queue_path.open("w", encoding="utf-8") as fh:
            for q_idx, q in enumerate(questions):
                pred = q.get("predicted")
                gold = q.get("gold_answer")
                if pred is None or gold is None:
                    continue
                entry = {
                    "qid": f"{bucket_tag}__{stem}__q{q_idx:03d}",
                    "bucket": bucket_tag,
                    "benchmark": bench,
                    "config": config,
                    "seed": seed,
                    "k": d.get("k", default_k),
                    "doc_idx": q.get("doc_idx"),
                    "question": q.get("question", ""),
                    "gold_answer": gold,
                    "predicted": pred,
                    "gpt4omini_judge_score": q.get("judge_score"),
                    "retrieved_steps": q.get("retrieved_steps", []),
                }
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n_pred += 1

        if not results_path.exists():
            results_path.touch()

        total += n_pred
        n_cells += 1
        print(f"  [OK] {bucket_tag}__{stem}: {n_pred}")

    print(f"  {bucket_tag} TOTAL: {n_cells} cells, {total} entries")
    print()
    return total


def main() -> None:
    grand_total = 0
    for cells_subdir, bucket_tag, default_k in BUCKETS:
        print(f"=== {cells_subdir} ({bucket_tag}) ===")
        grand_total += build_for_dir(cells_subdir, bucket_tag, default_k)
    print(f"GRAND TOTAL: {grand_total} entries across all buckets")


if __name__ == "__main__":
    main()
