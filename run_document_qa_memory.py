"""
DocumentQA memory-quality evaluation — recall@k for all memory systems (no LLM).

Usage (PowerShell):
    python run_document_qa_memory.py
    python run_document_qa_memory.py RAGMemory   # skip RAGMemory (e.g. if sentence_transformers broken)

Stores each paragraph as an event during the reading phase, then for each
question retrieves top-k and computes whether any relevant paragraph was retrieved.
Outputs results to results/document_qa_memory_results.json.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from evaluation.document_qa_memory import (
    run_document_qa_memory_eval,
    print_document_qa_table,
    save_document_qa_results,
)


def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    skip_systems = list(argv)

    print("DocumentQA memory-quality evaluation (recall@k, no LLM)")
    print("Document: fantasy_lore | k=8")
    if skip_systems:
        print(f"Skipping systems: {', '.join(skip_systems)}")

    results = run_document_qa_memory_eval(
        document_name="fantasy_lore",
        k=8,
        seed=0,
        skip_systems=skip_systems if skip_systems else None,
    )
    print_document_qa_table(results)
    out_path = Path("results") / "document_qa_memory_results.json"
    out_path.parent.mkdir(exist_ok=True)
    save_document_qa_results(results, out_path)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
