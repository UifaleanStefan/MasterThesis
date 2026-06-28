"""
BM25 corpus-mode tuner — fairness pass (audit critique B6).

The headline comparison tunes V4ₜ's θ on a corpus-cumulative recall@k
objective, but the BM25 baseline was run at rank-bm25's stock defaults
(k1=1.5, b=0.75). That is an unfair comparison: a stock baseline vs. a
tuned system. This tuner removes the asymmetry by optimizing BM25's two
hyperparameters (k1, b) on the *same* corpus-cumulative recall@k
objective used for V4ₜ.

Because BM25 has only two hyperparameters over a small, well-understood
range, we tune by an EXHAUSTIVE GRID rather than CMA-ES. Exhaustive
search returns the global optimum over the grid, so the resulting claim
is the strongest possible fairness statement: "BM25 tuned to its optimum
on this objective still {does/does not} match V4ₜ".

No LLM, no API — pure recall@k of gold evidence, identical objective to
tuning/tune_v4t_corpus.py.

Output: results/stage3/tuned_bm25_corpus_<bench>.json

Usage:
    python -m tuning.tune_bm25_corpus --benchmarks cuad qasper
    python -m tuning.tune_bm25_corpus --benchmarks financebench --limit-docs 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from environment.benchmarks import ADAPTERS, get_adapter
from memory.bm25_memory import BM25Memory
from memory.event import Event
from results.manifest import build_manifest

# Default Okapi hyperparameters (rank-bm25), used as the "canonical" baseline.
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

# Exhaustive grid. k1 controls term-frequency saturation; b controls
# length normalization. These ranges bracket the values used in the IR
# literature (k1 ∈ [0.5, 3], b ∈ [0, 1]).
GRID_K1 = [0.5, 0.8, 1.2, 1.5, 2.0, 2.5, 3.0]
GRID_B = [0.0, 0.25, 0.5, 0.75, 1.0]


def build_eval_tasks(benchmark: str, limit_docs: int, seed: int):
    """Pre-compute (question, gold_global_steps, current_step) eval tasks and
    the ordered (global_step, observation) ingestion stream.

    Mirrors tuning/tune_v4t_corpus.make_corpus_eval_fn so the objective is
    identical: online recall@k of the just-ingested doc's gold paragraphs,
    against the full cumulative memory.
    """
    adapter = get_adapter(benchmark)
    docs = list(adapter.iter_documents(limit=limit_docs))
    if not docs:
        raise RuntimeError(f"adapter {benchmark!r} yielded no docs")

    eval_tasks: list[tuple[str, set[int], int]] = []
    ingest: list[tuple[int, str]] = []  # (global_step, observation)
    global_step = 0
    for doc_idx, doc in enumerate(docs):
        doc_start = global_step
        paragraphs = doc.get("paragraphs", [])
        doc_title = str(doc.get("title", f"doc_{doc_idx}"))[:120]
        for para_idx, paragraph in enumerate(paragraphs):
            obs = f"[{doc_title}] {paragraph}" if para_idx == 0 else paragraph
            ingest.append((global_step, obs))
            global_step += 1
        end_of_doc = global_step
        for qa in (doc.get("qa_pairs", []) or [])[:1]:
            relevant = qa.get("relevant_paragraphs", []) or []
            if not relevant:
                continue
            gold = {doc_start + i for i in relevant if 0 <= i < len(paragraphs)}
            if not gold:
                continue
            scoped_q = f"[Regarding {doc_title}] {qa['question']}"
            eval_tasks.append((scoped_q, gold, end_of_doc))

    if not eval_tasks:
        raise RuntimeError(
            f"adapter {benchmark!r} produced no QAs with gold under "
            f"limit_docs={limit_docs}"
        )
    return docs, ingest, eval_tasks


def recall_at_k(ingest, eval_tasks, k1: float, b: float, k: int) -> float:
    """Mean recall@k under BM25(k1, b) over the cumulative corpus."""
    mem = BM25Memory(k1=k1, b=b)
    for gstep, obs in ingest:
        mem.add_event(Event(step=gstep, observation=obs, action="read"))
    hits: list[float] = []
    for question, gold, current_step in eval_tasks:
        retrieved = mem.get_relevant_events(question, current_step=current_step, k=k)
        steps = {ev.step for ev in retrieved}
        hits.append(1.0 if (steps & gold) else 0.0)
    return float(np.mean(hits)) if hits else 0.0


def tune_one_benchmark(benchmark, limit_docs, seed, k) -> dict:
    print()
    print("=" * 78)
    print(f"  BM25 corpus tuning for {benchmark!r}  (limit_docs={limit_docs}, k={k})")
    print("=" * 78)
    docs, ingest, eval_tasks = build_eval_tasks(benchmark, limit_docs, seed)
    print(f"  {len(docs)} docs, {len(ingest)} paragraphs, {len(eval_tasks)} eval QAs")

    canonical = recall_at_k(ingest, eval_tasks, DEFAULT_K1, DEFAULT_B, k)
    print(f"  canonical BM25 (k1={DEFAULT_K1}, b={DEFAULT_B}) recall@{k} = {canonical:.4f}")

    t0 = time.time()
    history = []
    best = {"k1": DEFAULT_K1, "b": DEFAULT_B, "recall": canonical}
    for k1 in GRID_K1:
        for b in GRID_B:
            r = recall_at_k(ingest, eval_tasks, k1, b, k)
            history.append({"k1": k1, "b": b, "recall": r})
            if r > best["recall"]:
                best = {"k1": k1, "b": b, "recall": r}
    elapsed = time.time() - t0

    improvement = best["recall"] - canonical
    print(f"  tuned BM25 (k1={best['k1']}, b={best['b']}) recall@{k} = {best['recall']:.4f}")
    print(f"  improvement = {improvement:+.4f}   ({len(history)} grid points, {elapsed:.1f}s)")

    return {
        "benchmark": benchmark,
        "status": "ok",
        "variant": "bm25_corpus",
        "limit_docs": limit_docs,
        "k": k,
        "n_eval_questions": len(eval_tasks),
        "canonical_k1": DEFAULT_K1,
        "canonical_b": DEFAULT_B,
        "canonical_recall": canonical,
        "k1": best["k1"],
        "b": best["b"],
        "tuned_recall": best["recall"],
        "improvement": improvement,
        "grid": history,
        "elapsed_seconds": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="*", default=["cuad", "qasper"],
                        choices=sorted(ADAPTERS.keys()))
    parser.add_argument("--limit-docs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--out-dir", default="results/stage3")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for name in args.benchmarks:
        try:
            result = tune_one_benchmark(name, args.limit_docs, args.seed, args.k)
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {"benchmark": name, "status": "error", "error": repr(e)}
        result["_manifest"] = build_manifest(seed=args.seed, extra={
            "experiment": "stage3_bm25_corpus_tuning",
            "benchmark": name, "limit_docs": args.limit_docs, "k": args.k,
        })
        out_path = out_dir / f"tuned_bm25_corpus_{name}.json"
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"  -> saved {out_path}")
        summary.append(result)

    print()
    print("=" * 78)
    print("  BM25 CORPUS-TUNING SUMMARY")
    print("=" * 78)
    print(f"  {'benchmark':<14} {'status':>7} {'canon':>8} {'tuned':>8} {'lift':>8} {'k1':>5} {'b':>5}")
    for s in summary:
        if s.get("status") == "ok":
            print(f"  {s['benchmark']:<14} {'ok':>7} {s['canonical_recall']:>8.4f} "
                  f"{s['tuned_recall']:>8.4f} {s['improvement']:>+8.4f} {s['k1']:>5} {s['b']:>5}")
        else:
            print(f"  {s['benchmark']:<14} {'error':>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
