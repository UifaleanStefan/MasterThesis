"""
Claude-in-the-loop judge queue infrastructure (Block E / critique #11).

The Phase 1.7 LLM judge used gpt-4o-mini both as answer generator AND
as judge — a self-bias known to inflate scores. The corpus-mode Phase 2
runs separate the two: gpt-4o-mini generates answers (cheap, automated),
then Claude scores them via the in-session rubric defined in
`claude_judge_protocol.md`.

Pipeline:

    scripts/run_corpus_qa.py                     # generates answers
       │
       ▼
    results/stage3/judge_queue/<run_id>/
        queue.jsonl                              # one entry per QA
       │
       ▼ (Claude reads, judges per rubric,
       ▼  writes results in same dir)
        results.jsonl                            # one judge_score per QA
       │
       ▼
    merge_judge_results(run_id) → updates qa_online.json / qa_batch.json
                                  with judge_score + rationale fields

This module provides:
  * `write_judge_queue(run_id, items)` — emit queue.jsonl
  * `read_judge_results(run_id)` — load Claude's judgments back
  * `merge_judge_results_into_qa(run_id, qa_path)` — splice scores into
    the per-cell QA JSON written by run_corpus_qa.py

The run_id format is `<benchmark>__<config>__<mode>__<seed>` (or similar);
the queue/results files live under results/stage3/judge_queue/<run_id>/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
QUEUE_ROOT = ROOT / "results" / "stage3" / "judge_queue"


def queue_dir_for(run_id: str) -> Path:
    return QUEUE_ROOT / run_id


def write_judge_queue(run_id: str, items: Iterable[dict]) -> Path:
    """Emit a queue.jsonl that Claude will score in-session.

    Each `item` should be a dict with at least:
        qid, question, scoped_question, gold_answer, predicted

    Optional but useful fields (preserved verbatim in the queue, ignored
    by Claude's rubric):
        benchmark, config, mode, doc_idx, qa_idx_in_doc,
        retrieved_steps, k, current_step
    """
    qdir = queue_dir_for(run_id)
    qdir.mkdir(parents=True, exist_ok=True)
    path = qdir / "queue.jsonl"
    n_written = 0
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            if "qid" not in item:
                raise ValueError(f"queue item missing 'qid': {item}")
            fh.write(json.dumps(item, default=str) + "\n")
            n_written += 1
    print(f"[claude_judge_queue] wrote {n_written} items to {path}")
    return path


def items_lines(path: Path) -> Iterable[dict]:
    """Yield JSON objects line-by-line from a JSONL file."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_judge_queue(run_id: str) -> list[dict]:
    return list(items_lines(queue_dir_for(run_id) / "queue.jsonl"))


def read_judge_results(run_id: str) -> dict[str, dict]:
    """Return {qid: {judge_score, rationale}} for all Claude-judged entries."""
    path = queue_dir_for(run_id) / "results.jsonl"
    by_qid: dict[str, dict] = {}
    for item in items_lines(path):
        qid = item.get("qid")
        if qid is None:
            continue
        by_qid[qid] = item
    return by_qid


def merge_judge_results_into_qa(
    run_id: str,
    qa_json_path: Path,
    qid_field: str = "qid",
) -> dict:
    """Read judge results for `run_id` and splice judge_score/rationale into
    the per-question QA records at `qa_json_path` (in place).

    Returns a summary dict with counts.

    Each QA record in qa_json_path must have a `qid` field matching the
    judge_queue entry, OR the canonical fields (benchmark, config, mode,
    doc_idx, qa_idx_in_doc) — in which case the qid is reconstructed.
    """
    if not qa_json_path.exists():
        raise FileNotFoundError(f"QA JSON not found: {qa_json_path}")
    qa_records: list[dict] = json.loads(qa_json_path.read_text())
    if not isinstance(qa_records, list):
        raise ValueError(f"Expected a list of QA records at {qa_json_path}")

    results = read_judge_results(run_id)
    matched = 0
    missing_qid = 0
    not_in_results = 0
    for rec in qa_records:
        # Reconstruct qid if missing
        qid = rec.get(qid_field)
        if qid is None:
            missing_qid += 1
            continue
        judge = results.get(qid)
        if judge is None:
            not_in_results += 1
            continue
        rec["judge_score"] = judge.get("judge_score")
        rec["judge_rationale"] = judge.get("rationale")
        matched += 1

    qa_json_path.write_text(json.dumps(qa_records, indent=2, default=str))
    return {
        "qa_records": len(qa_records),
        "judge_results": len(results),
        "matched": matched,
        "missing_qid_in_qa": missing_qid,
        "not_yet_judged": not_in_results,
        "qa_path": str(qa_json_path),
    }


def queue_summary(run_id: str) -> dict:
    qdir = queue_dir_for(run_id)
    q = qdir / "queue.jsonl"
    r = qdir / "results.jsonl"
    n_queue = sum(1 for _ in items_lines(q)) if q.exists() else 0
    n_results = sum(1 for _ in items_lines(r)) if r.exists() else 0
    return {
        "run_id": run_id,
        "queue_path": str(q),
        "results_path": str(r),
        "n_queued": n_queue,
        "n_judged": n_results,
        "pct_complete": (n_results / n_queue * 100) if n_queue else 0.0,
    }
