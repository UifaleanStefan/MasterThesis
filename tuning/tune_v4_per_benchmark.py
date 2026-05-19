"""
CMA-ES per-benchmark V4 theta tuning — Stage 3 Phase 1.5 workstream C.

Motivation: V4's canonical theta (`w_recency=3.777`, `theta_store=0.293`) was
tuned on grid-world (MultiHop-KeyDoor) where recency dominates the
task signal. On document QA — where the agent reads a static document
in order and then must retrieve the right paragraph for a question —
that recency bias collapses retrieval onto the last-k paragraphs
regardless of the question. Running the Phase-1.5 retrieval study with
only the canonical theta would understate V4's ceiling.

This module tunes a fresh per-benchmark theta on each adapter using CMA-ES
on mean recall@k as the objective, with no LLM in the loop ($0 cost).
The resulting per-benchmark theta vectors are saved to
`results/stage3/tuned_theta_<benchmark>.json` and consumed by
`evaluation/benchmark_memory_eval.py` when `--load-tuned-thetas` is set.

Reused infrastructure:
  * `optimization.cma_es.run_cmaes_optimization` — the CMA-ES driver.
  * `memory.graph_memory_v4.MemoryParamsV4` — theta container.
  * `evaluation.document_qa_memory._run_reading_phase, _recall_at_k_for_qa`
    — reading + recall helpers (unchanged).
  * `run_graphmemory_v4_cmaes.vec_to_params_v4` (re-implemented locally
    for self-containment) — [0,1]^10 vec -> MemoryParamsV4 with w_* × 4.

The "per-benchmark" framing is what makes this thesis-defensible: we
show V4's *task-adapted* performance, not its grid-world projection.
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
from environment.document_qa import DocumentQA
from evaluation.document_qa_memory import _recall_at_k_for_qa, _run_reading_phase
from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4
from optimization.cma_es import run_cmaes_optimization
from results.manifest import build_manifest

# Canonical (grid-world-tuned) baseline — same as the rest of the codebase
# uses in `evaluation/document_qa_memory._make_document_qa_memory_systems`.
CANONICAL_THETA_VEC = np.array([
    0.293,    # theta_store
    0.908,    # theta_novel
    0.198,    # theta_erich
    0.785,    # theta_surprise
    0.285,    # theta_entity
    0.278,    # theta_temporal
    0.668,    # theta_decay
    0.0  / 4, # w_graph    (=0.0)
    1.079/ 4, # w_embed    (=1.079)
    3.777/ 4, # w_recency  (=3.777, clamped to [0,1])
], dtype=np.float64)


def vec_to_params(v: np.ndarray) -> MemoryParamsV4:
    """[0,1]^10 vector -> V4 params with w_* × 4.0 (matching the established convention)."""
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    return MemoryParamsV4(
        theta_store=float(v[0]),
        theta_novel=float(v[1]),
        theta_erich=float(v[2]),
        theta_surprise=float(v[3]),
        theta_entity=float(v[4]),
        theta_temporal=float(v[5]),
        theta_decay=float(v[6]),
        w_graph=float(v[7]) * 4.0,
        w_embed=float(v[8]) * 4.0,
        w_recency=float(v[9]) * 4.0,
        mode="learnable",
    )


def make_eval_fn(adapter_name: str, n_docs: int, k: int = 8):
    """Build the CMA-ES objective: mean recall@k across n_docs from this benchmark.

    Higher is better. Returns 0.0 when zero qa_pairs have gold (NarrativeQA
    case — tuning isn't meaningful, but we don't crash).

    The doc set is pulled ONCE per make_eval_fn call (cached) so that every
    candidate evaluates on the same docs. This means the only thing changing
    between candidates is theta, which is what makes the search well-posed.
    """
    adapter = get_adapter(adapter_name)
    # Materialize the docs once — cache across all evaluations.
    docs = list(adapter.iter_documents(limit=n_docs))
    if not docs:
        raise RuntimeError(f"Adapter {adapter_name!r} yielded no docs")

    # Pre-extract just the gold qa_pairs (skip empty-gold ones for speed).
    eval_tasks: list[tuple[dict, list[tuple[int, str, list[int]]]]] = []
    for doc in docs:
        qas_with_gold: list[tuple[int, str, list[int]]] = []
        for qa_idx, qa in enumerate(doc["qa_pairs"]):
            rel = qa.get("relevant_paragraphs", []) or []
            if rel:
                qas_with_gold.append((qa_idx, qa["question"], rel))
        if qas_with_gold:
            eval_tasks.append((doc, qas_with_gold))

    n_total_gold_questions = sum(len(t) for _, t in eval_tasks)
    print(f"  [eval_fn] {adapter_name}: {len(eval_tasks)} docs, "
          f"{n_total_gold_questions} qa_pairs with gold")

    def eval_fn(theta_vec: np.ndarray) -> float:
        params = vec_to_params(theta_vec)
        recalls: list[float] = []
        for doc, qas_with_gold in eval_tasks:
            n_paragraphs = len(doc["paragraphs"])
            memory = GraphMemoryV4(params)
            env = DocumentQA(document=doc, seed=42, question_shuffle=False)
            _run_reading_phase(env, memory, episode_seed=42)
            for qa_idx, question, relevant in qas_with_gold:
                current_step = n_paragraphs + qa_idx
                r = _recall_at_k_for_qa(
                    memory, question, relevant,
                    k=k, current_step=current_step,
                )
                recalls.append(r)
        return float(np.mean(recalls)) if recalls else 0.0

    return eval_fn, n_total_gold_questions


def tune_one_benchmark(
    name: str,
    n_docs: int,
    n_generations: int,
    sigma: float,
    seed: int,
    k: int = 8,
) -> dict:
    """Run CMA-ES on one benchmark; return result dict."""
    print()
    print("=" * 78)
    print(f"  Tuning V4 theta for {name!r}")
    print(f"  n_docs={n_docs}  n_generations={n_generations}  sigma={sigma}  seed={seed}")
    print("=" * 78)
    eval_fn, n_gold_qs = make_eval_fn(name, n_docs=n_docs, k=k)
    if n_gold_qs == 0:
        print(f"  [SKIP] {name}: 0 qa_pairs with gold relevance — cannot tune retrieval.")
        return {
            "name": name,
            "status": "skipped",
            "reason": "no_gold_questions",
        }

    # Baseline recall under the canonical (grid-world) theta.
    canonical_recall = eval_fn(CANONICAL_THETA_VEC)
    print(f"  canonical theta recall@k={k}: {canonical_recall:.4f}")

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
    improvement = tuned_recall - canonical_recall

    tuned_params = vec_to_params(best_vec)
    print()
    print(f"  --- {name} tuning complete in {elapsed:.1f}s ---")
    print(f"  canonical recall = {canonical_recall:.4f}")
    print(f"  tuned     recall = {tuned_recall:.4f}")
    print(f"  improvement      = {improvement:+.4f}")
    print(f"  tuned theta          = {[round(x, 3) for x in best_vec.tolist()]}")
    print(f"  -> MemoryParamsV4 = "
          f"theta_store={tuned_params.theta_store:.3f} "
          f"w_embed={tuned_params.w_embed:.3f} "
          f"w_recency={tuned_params.w_recency:.3f}")

    return {
        "name": name,
        "status": "ok",
        "n_docs": n_docs,
        "n_gold_questions": n_gold_qs,
        "n_generations": n_generations,
        "sigma": sigma,
        "k": k,
        "canonical_theta_vec": CANONICAL_THETA_VEC.tolist(),
        "canonical_recall": canonical_recall,
        "tuned_theta_vec": best_vec.tolist(),
        "tuned_theta_params": {
            "theta_store": tuned_params.theta_store,
            "theta_novel": tuned_params.theta_novel,
            "theta_erich": tuned_params.theta_erich,
            "theta_surprise": tuned_params.theta_surprise,
            "theta_entity": tuned_params.theta_entity,
            "theta_temporal": tuned_params.theta_temporal,
            "theta_decay": tuned_params.theta_decay,
            "w_graph": tuned_params.w_graph,
            "w_embed": tuned_params.w_embed,
            "w_recency": tuned_params.w_recency,
        },
        "tuned_recall": tuned_recall,
        "improvement": improvement,
        "elapsed_s": elapsed,
        "history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks", nargs="*", default=sorted(ADAPTERS.keys()),
        help="Subset of benchmarks to tune. Default: all 6.",
    )
    parser.add_argument(
        "--n-docs", type=int, default=10,
        help="Number of documents per benchmark for tuning eval (default 10). "
             "Higher = more stable objective but slower.",
    )
    parser.add_argument(
        "--n-generations", type=int, default=15,
        help="CMA-ES generations (default 15).",
    )
    parser.add_argument(
        "--sigma", type=float, default=0.3,
        help="CMA-ES initial step size (default 0.3).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode for smoke testing: n_docs=5, n_generations=5.",
    )
    parser.add_argument(
        "--out-dir", default="results/stage3",
        help="Output directory for per-benchmark tuning JSON files.",
    )
    parser.add_argument(
        "--out-suffix", default="",
        help="Optional suffix for output filenames: tuned_theta_{suffix_}{bench}.json. "
             "Use to preserve a prior narrow run while running a wider second pass.",
    )
    args = parser.parse_args()

    if args.quick:
        args.n_docs = 5
        args.n_generations = 5
        print("[quick mode] n_docs=5, n_generations=5")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    summary: list[dict] = []
    for name in args.benchmarks:
        if name not in ADAPTERS:
            print(f"  [SKIP] unknown benchmark: {name!r}")
            continue
        try:
            result = tune_one_benchmark(
                name=name,
                n_docs=args.n_docs,
                n_generations=args.n_generations,
                sigma=args.sigma,
                seed=args.seed,
                k=args.k,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {"name": name, "status": "error", "error": repr(e)}

        # Save per-benchmark file.
        suffix = f"{args.out_suffix}_" if args.out_suffix else ""
        out_path = out_dir / f"tuned_theta_{suffix}{name}.json"
        result["_manifest"] = build_manifest(seed=args.seed, extra={
            "experiment": "stage3_theta_tuning",
            "benchmark": name,
            "n_docs": args.n_docs,
            "n_generations": args.n_generations,
            "sigma": args.sigma,
            "k": args.k,
        })
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"  -> saved {out_path}")
        summary.append({
            "name": name,
            "status": result.get("status"),
            "canonical_recall": result.get("canonical_recall"),
            "tuned_recall": result.get("tuned_recall"),
            "improvement": result.get("improvement"),
            "elapsed_s": result.get("elapsed_s"),
        })

    total_elapsed = time.time() - t_total
    print()
    print("=" * 78)
    print("  TUNING SUMMARY")
    print("=" * 78)
    print(f"  Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  {'benchmark':<14}  {'canonical':>10}  {'tuned':>10}  {'lift':>10}  {'status':>10}")
    n_with_lift = 0
    for s in summary:
        if s.get("status") == "ok":
            imp = s["improvement"]
            if imp >= 0.10:
                n_with_lift += 1
            print(
                f"  {s['name']:<14}  {s['canonical_recall']:>10.4f}  "
                f"{s['tuned_recall']:>10.4f}  {imp:>+10.4f}  {s['status']:>10}"
            )
        else:
            print(f"  {s['name']:<14}  {'n/a':>10}  {'n/a':>10}  {'n/a':>10}  {s.get('status'):>10}")

    print(f"\n  Benchmarks with >=0.10 recall lift: {n_with_lift} / {len(summary)}")
    if n_with_lift >= 4:
        print("  [OK] target met (>= 4 of 6 with measurable lift)")
        return 0
    elif n_with_lift > 0:
        print("  [PARTIAL] some benchmarks lifted; consider --n-generations 25+ or --n-docs 15")
        return 0  # not a hard failure
    else:
        print("  [FAIL] no benchmark lifted — investigate canonical theta or tuning config")
        return 1


if __name__ == "__main__":
    sys.exit(main())
