"""
V4ₜ corpus-mode CMA-ES tuner — Stage 3 Phase 2 / critique #6.

Re-tunes the 10-D θ vector under the corpus-cumulative regime with
`text_mode_entities=True`. This is the methodologically-honest counter
to the original Phase 1.5/1.6 tuning, which optimized θ in PER-DOC
isolated mode with the grid-world entity gate enabled.

The interesting question this tuner answers: **does w_graph rise from
near-zero when the optimizer faces a corpus-cumulative objective?**

In per-doc mode the graph never has time to develop cross-doc
structure, so graph traversal contributes nothing — the optimizer
correctly sets w_graph ≈ 0. In corpus mode, by the time the agent
hits doc 200, recurring entities span hundreds of mention edges; if
the graph signal is ever going to be useful, this is where.

If the tuner finds w_graph ~ 0 even here, then V4ₜ corpus mode is
empirically "selective storage + recency-weighted embedding lookup"
with no graph contribution — honestly reportable as a thesis finding.

If w_graph > 0 with measurable recall lift, then graph traversal is a
real contributor at corpus scale, and Phase 2's claim that V4ₜ
encodes a learned domain representation has stronger empirical
grounding.

No LLM, no API. Pure CMA-ES over recall@k = the same objective the
Phase 1.5/1.6 tuner used, just in corpus mode with V4ₜ.

Output: `results/stage3/tuned_theta_v4t_corpus_<bench>.json`.

Usage:
    python -m tuning.tune_v4t_corpus --benchmarks cuad
    python -m tuning.tune_v4t_corpus --benchmarks qasper --limit-docs 50 --n-generations 8
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
from memory.event import Event
from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4
from optimization.cma_es import run_cmaes_optimization
from results.manifest import build_manifest
from tuning.tune_v4_per_benchmark import (
    CANONICAL_THETA_VEC,
    vec_to_params,
)


def make_corpus_eval_fn(
    benchmark: str,
    limit_docs: int,
    k: int,
    sample_qa_per_doc: int = 1,
    seed: int = 42,
):
    """Build the CMA-ES objective: mean recall@k under V4ₜ corpus mode.

    For each θ candidate:
      1. Build one V4 instance with text_mode_entities=True and that θ.
      2. Ingest every paragraph of `limit_docs` docs sequentially,
         using GLOBAL step indexing.
      3. For each doc, sample `sample_qa_per_doc` questions and
         compute recall@k at `current_step = end_of_doc_K` (online
         retrieval semantics — measures whether the cumulative
         graph helps answer the just-ingested doc's questions).
      4. Return mean recall@k across all sampled questions with gold.
    """
    adapter = get_adapter(benchmark)
    docs = list(adapter.iter_documents(limit=limit_docs))
    if not docs:
        raise RuntimeError(f"adapter {benchmark!r} yielded no docs")

    # Pre-compute (doc_idx, qa_idx, question, gold_global_steps,
    # current_step) for each sampled QA. Doc paragraphs are fixed,
    # so global step boundaries are too.
    eval_tasks: list[tuple[int, int, str, set[int], int]] = []
    global_step = 0
    for doc_idx, doc in enumerate(docs):
        doc_start = global_step
        paragraphs = doc.get("paragraphs", [])
        global_step += len(paragraphs)
        end_of_doc = global_step
        qa_pairs = doc.get("qa_pairs", []) or []
        if not qa_pairs:
            continue
        for qa_idx, qa in enumerate(qa_pairs[:sample_qa_per_doc]):
            relevant = qa.get("relevant_paragraphs", []) or []
            if not relevant:
                continue
            global_relevant = {doc_start + i for i in relevant if 0 <= i < len(paragraphs)}
            doc_title = str(doc.get("title", f"doc_{doc_idx}"))[:120]
            scoped_q = f"[Regarding {doc_title}] {qa['question']}"
            eval_tasks.append((doc_idx, qa_idx, scoped_q, global_relevant, end_of_doc))

    n_total = len(eval_tasks)
    print(f"  [eval_fn] {benchmark}: {len(docs)} docs, {n_total} eval QAs with gold")
    if n_total == 0:
        raise RuntimeError(f"adapter {benchmark!r} produced no QAs with gold under limit_docs={limit_docs}")

    def eval_fn(vec: np.ndarray) -> float:
        params = vec_to_params(vec)
        params.text_mode_entities = True
        memory = GraphMemoryV4(params)
        # Re-ingest the full corpus under THIS θ.
        gstep = 0
        doc_iter = iter(docs)
        for doc_idx, doc in enumerate(docs):
            paragraphs = doc.get("paragraphs", [])
            doc_title = str(doc.get("title", f"doc_{doc_idx}"))[:120]
            for para_idx, paragraph in enumerate(paragraphs):
                obs = f"[{doc_title}] {paragraph}" if para_idx == 0 else paragraph
                memory.add_event(Event(step=gstep, observation=obs, action="read"),
                                 episode_seed=seed)
                gstep += 1

        # Score: mean recall@k across eval_tasks.
        recalls: list[float] = []
        for _, _, question, gold_global, current_step in eval_tasks:
            retrieved = memory.get_relevant_events(question, current_step=current_step, k=k)
            retrieved_steps = {ev.step for ev in retrieved}
            recall = 1.0 if (retrieved_steps & gold_global) else 0.0
            recalls.append(recall)
        return float(np.mean(recalls))

    return eval_fn, n_total


def tune_one_benchmark(
    benchmark: str,
    limit_docs: int,
    n_generations: int,
    sigma: float,
    seed: int,
    k: int,
) -> dict:
    print()
    print("=" * 78)
    print(f"  V4t corpus tuning for {benchmark!r}")
    print(f"  limit_docs={limit_docs}  n_generations={n_generations}  "
          f"sigma={sigma}  seed={seed}  k={k}")
    print("=" * 78)

    eval_fn, n_total = make_corpus_eval_fn(benchmark, limit_docs, k, seed=seed)

    # Baseline: canonical θ + text_mode_entities=True.
    canonical_recall = eval_fn(CANONICAL_THETA_VEC.copy())
    print(f"  Canonical V4t corpus recall@k={k}: {canonical_recall:.4f}")

    t0 = time.time()
    best_vec, history = run_cmaes_optimization(
        eval_fn,
        n_params=10,
        n_generations=n_generations,
        sigma=sigma,
        seed=seed,
        clip_to_unit=True,
        verbose=True,
    )
    elapsed = time.time() - t0
    tuned_recall = float(history[-1]["best_fitness"]) if history else 0.0
    tuned_params = vec_to_params(best_vec)
    improvement = tuned_recall - canonical_recall

    print(f"\n  V4t corpus tuning done in {elapsed:.1f}s")
    print(f"  canonical recall = {canonical_recall:.4f}")
    print(f"  tuned     recall = {tuned_recall:.4f}")
    print(f"  improvement      = {improvement:+.4f}")
    print(f"\n  Tuned theta (corpus mode):")
    for attr in ["theta_store", "theta_novel", "theta_erich", "theta_surprise",
                 "theta_entity", "theta_temporal", "theta_decay",
                 "w_graph", "w_embed", "w_recency"]:
        print(f"    {attr:<14} = {getattr(tuned_params, attr):.4f}")

    return {
        "benchmark": benchmark,
        "status": "ok",
        "variant": "V4t_corpus",
        "limit_docs": limit_docs,
        "n_eval_questions": n_total,
        "n_generations": n_generations,
        "sigma": sigma,
        "k": k,
        "canonical_recall": canonical_recall,
        "tuned_recall": tuned_recall,
        "improvement": improvement,
        "tuned_theta_vec": [float(v) for v in best_vec],
        "tuned_params": {
            attr: float(getattr(tuned_params, attr))
            for attr in ["theta_store", "theta_novel", "theta_erich",
                         "theta_surprise", "theta_entity", "theta_temporal",
                         "theta_decay", "w_graph", "w_embed", "w_recency"]
        },
        "elapsed_seconds": elapsed,
        "history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks", nargs="*", default=["cuad", "qasper"],
        choices=sorted(ADAPTERS.keys()),
        help="Benchmarks to tune (default: cuad + qasper — the two domain-coherent "
             "long-haystack benchmarks where the original V4 tuning showed real lift).",
    )
    parser.add_argument(
        "--limit-docs", type=int, default=50,
        help="Number of docs to ingest per CMA-ES evaluation (default 50; "
             "smaller = faster tuning, larger = more representative of full corpus).",
    )
    parser.add_argument(
        "--n-generations", type=int, default=8,
        help="CMA-ES generation count (default 8).",
    )
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument(
        "--out-dir", default="results/stage3",
        help="Output dir for tuned_theta_v4t_corpus_<bench>.json files.",
    )
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    summary = []
    for name in args.benchmarks:
        try:
            result = tune_one_benchmark(
                benchmark=name,
                limit_docs=args.limit_docs,
                n_generations=args.n_generations,
                sigma=args.sigma,
                seed=args.seed,
                k=args.k,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {"benchmark": name, "status": "error", "error": repr(e)}

        out_path = out_dir / f"tuned_theta_v4t_corpus_{name}.json"
        result["_manifest"] = build_manifest(seed=args.seed, extra={
            "experiment": "stage3_v4t_corpus_tuning",
            "benchmark": name,
            "limit_docs": args.limit_docs,
            "n_generations": args.n_generations,
            "sigma": args.sigma,
            "k": args.k,
        })
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"  -> saved {out_path}")
        summary.append({
            "benchmark": name,
            "status": result.get("status"),
            "canonical_recall": result.get("canonical_recall"),
            "tuned_recall": result.get("tuned_recall"),
            "improvement": result.get("improvement"),
            "w_graph_tuned": (result.get("tuned_params") or {}).get("w_graph"),
        })

    print()
    print("=" * 78)
    print("  V4t CORPUS-MODE TUNING SUMMARY")
    print("=" * 78)
    print(f"  Total elapsed: {time.time() - t_total:.1f}s")
    print(f"  {'benchmark':<14}  {'status':>8}  {'canonical':>10}  {'tuned':>10}  {'lift':>10}  {'w_graph':>10}")
    for s in summary:
        if s["status"] == "ok":
            print(
                f"  {s['benchmark']:<14}  {s['status']:>8}  "
                f"{s['canonical_recall']:>10.4f}  {s['tuned_recall']:>10.4f}  "
                f"{s['improvement']:>+10.4f}  {s['w_graph_tuned']:>10.4f}"
            )
        else:
            print(f"  {s['benchmark']:<14}  {s['status']:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
