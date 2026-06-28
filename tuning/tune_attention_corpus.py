"""
AttentionMemory corpus-mode tuner — fairness pass (audit critique B6).

Companion to tuning/tune_bm25_corpus.py. The original AttentionMemory
baseline was tuned per-doc (tuning/tune_attention_per_benchmark.py),
not under the corpus-cumulative regime that V4ₜ is tuned on. This tuner
sweeps the single temperature hyperparameter τ on the *same*
corpus-cumulative recall@k objective, so the attention baseline is tuned
on an equal footing.

τ is one scalar, so we sweep an exhaustive 1-D grid (global optimum on
the grid). No LLM, no API — pure recall@k of gold evidence.

Output: results/stage3/tuned_attention_corpus_<bench>.json

Usage:
    python -m tuning.tune_attention_corpus --benchmarks cuad qasper
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

from environment.benchmarks import ADAPTERS
from memory.attention_memory import AttentionMemory
from memory.event import Event
from results.manifest import build_manifest
from tuning.tune_bm25_corpus import build_eval_tasks

DEFAULT_TAU = 0.5
GRID_TAU = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]


def recall_at_k(ingest, eval_tasks, tau: float, k: int) -> float:
    mem = AttentionMemory(temperature=tau)
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
    print(f"  Attention corpus tuning for {benchmark!r}  (limit_docs={limit_docs}, k={k})")
    print("=" * 78)
    docs, ingest, eval_tasks = build_eval_tasks(benchmark, limit_docs, seed)
    print(f"  {len(docs)} docs, {len(ingest)} paragraphs, {len(eval_tasks)} eval QAs")

    canonical = recall_at_k(ingest, eval_tasks, DEFAULT_TAU, k)
    print(f"  canonical Attention (τ={DEFAULT_TAU}) recall@{k} = {canonical:.4f}")

    t0 = time.time()
    history = []
    best = {"tau": DEFAULT_TAU, "recall": canonical}
    for tau in GRID_TAU:
        r = recall_at_k(ingest, eval_tasks, tau, k)
        history.append({"tau": tau, "recall": r})
        if r > best["recall"]:
            best = {"tau": tau, "recall": r}
    elapsed = time.time() - t0

    improvement = best["recall"] - canonical
    print(f"  tuned Attention (τ={best['tau']}) recall@{k} = {best['recall']:.4f}")
    print(f"  improvement = {improvement:+.4f}   ({len(history)} grid points, {elapsed:.1f}s)")

    return {
        "benchmark": benchmark,
        "status": "ok",
        "variant": "attention_corpus",
        "limit_docs": limit_docs,
        "k": k,
        "n_eval_questions": len(eval_tasks),
        "canonical_temperature": DEFAULT_TAU,
        "canonical_recall": canonical,
        "tuned_temperature": best["tau"],
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
            "experiment": "stage3_attention_corpus_tuning",
            "benchmark": name, "limit_docs": args.limit_docs, "k": args.k,
        })
        out_path = out_dir / f"tuned_attention_corpus_{name}.json"
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"  -> saved {out_path}")
        summary.append(result)

    print()
    print("=" * 78)
    print("  ATTENTION CORPUS-TUNING SUMMARY")
    print("=" * 78)
    print(f"  {'benchmark':<14} {'status':>7} {'canon':>8} {'tuned':>8} {'lift':>8} {'tau':>5}")
    for s in summary:
        if s.get("status") == "ok":
            print(f"  {s['benchmark']:<14} {'ok':>7} {s['canonical_recall']:>8.4f} "
                  f"{s['tuned_recall']:>8.4f} {s['improvement']:>+8.4f} {s['tuned_temperature']:>5}")
        else:
            print(f"  {s['benchmark']:<14} {'error':>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
