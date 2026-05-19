"""
AttentionMemory temperature tuning — fair-comparison baseline (Phase 1.7).

Addresses critique #2 from the adversarial review: V4 received 165+
CMA-ES evaluations to find its best theta, while the 12 reference
memory systems all ran with default hyperparameters. The fairest
single comparison we can make is to give ONE alternative memory system
the same tuning budget V4 received.

AttentionMemory exposes exactly one tunable hyperparameter
(`temperature: float = 0.5`) per `memory/attention_memory.py:58`,
which controls the softmax sharpness of its retrieval scoring. We
search this 1-D parameter via CMA-ES (overkill for 1-D — grid search
would suffice — but uses identical infrastructure for fair
comparison with V4).

Output: `results/stage3/tuned_temperature_<benchmark>.json` keyed by
benchmark name. The orchestrator's `_load_tuned_attention_temps`
picks them up automatically; `--configs attention-tuned` then runs
AttentionMemory(temperature=tuned_tau) at eval time.

Usage:
    python -m tuning.tune_attention_per_benchmark --benchmarks qasper cuad
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
from memory.attention_memory import AttentionMemory
from optimization.cma_es import run_cmaes_optimization
from results.manifest import build_manifest

# AttentionMemory's `temperature` is clipped to [0, 4] downstream.
# CMA-ES searches in [0, 1]^n by default; we multiply by 4 at param
# construction (same convention as V4's w_* dimensions).
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TEMPERATURE_VEC = DEFAULT_TEMPERATURE / 4.0  # [0,1] vec form


def vec_to_temperature(v: np.ndarray) -> float:
    """[0,1]^1 vector -> AttentionMemory temperature in [0,4]."""
    return float(np.clip(v[0], 0.0, 1.0) * 4.0)


def make_eval_fn(adapter_name: str, n_docs: int, k: int = 8):
    """Build the CMA-ES objective for AttentionMemory's temperature.

    Mirrors `tuning.tune_v4_per_benchmark.make_eval_fn` exactly — same
    docs, same gold-qa filter, same recall@k metric — so the comparison
    to V4-tuned is apples-to-apples.
    """
    adapter = get_adapter(adapter_name)
    docs = list(adapter.iter_documents(limit=n_docs))
    if not docs:
        raise RuntimeError(f"Adapter {adapter_name!r} yielded no docs")

    eval_tasks: list[tuple[dict, list[tuple[int, str, list[int]]]]] = []
    for doc in docs:
        qas_with_gold = []
        for qa_idx, qa in enumerate(doc["qa_pairs"]):
            rel = qa.get("relevant_paragraphs", []) or []
            if rel:
                qas_with_gold.append((qa_idx, qa["question"], rel))
        if qas_with_gold:
            eval_tasks.append((doc, qas_with_gold))

    n_total_gold = sum(len(t) for _, t in eval_tasks)
    print(f"  [eval_fn] {adapter_name}: {len(eval_tasks)} docs, "
          f"{n_total_gold} qa_pairs with gold")

    def eval_fn(vec: np.ndarray) -> float:
        temperature = vec_to_temperature(vec)
        recalls: list[float] = []
        for doc, qas_with_gold in eval_tasks:
            n_paragraphs = len(doc["paragraphs"])
            memory = AttentionMemory(temperature=temperature)
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

    return eval_fn, n_total_gold


def tune_one_benchmark(
    name: str, n_docs: int, n_generations: int, sigma: float, seed: int, k: int = 8,
) -> dict:
    print()
    print("=" * 78)
    print(f"  Tuning AttentionMemory temperature for {name!r}")
    print(f"  n_docs={n_docs}  n_generations={n_generations}  sigma={sigma}  seed={seed}")
    print("=" * 78)
    eval_fn, n_gold_qs = make_eval_fn(name, n_docs=n_docs, k=k)
    if n_gold_qs == 0:
        print(f"  [SKIP] {name}: 0 qa_pairs with gold — cannot tune retrieval.")
        return {"name": name, "status": "skipped", "reason": "no_gold_questions"}

    # Baseline at default temperature (0.5).
    canonical_recall = eval_fn(np.array([DEFAULT_TEMPERATURE_VEC]))
    print(f"  default AttentionMemory(tau=0.5) recall@k={k}: {canonical_recall:.4f}")

    t0 = time.time()
    # CMA-ES on 1-D is overkill but uses identical infrastructure for fair compare.
    best_vec, history = run_cmaes_optimization(
        eval_fn,
        n_params=1,
        n_generations=n_generations,
        sigma=sigma,
        seed=seed,
        clip_to_unit=True,
        verbose=True,
    )
    elapsed = time.time() - t0
    tuned_recall = float(history[-1]["best_fitness"]) if history else 0.0
    tuned_temp = vec_to_temperature(best_vec)
    improvement = tuned_recall - canonical_recall

    print(f"\n  --- {name} AttentionMemory tuning done in {elapsed:.1f}s ---")
    print(f"  default tau=0.5 recall = {canonical_recall:.4f}")
    print(f"  tuned tau={tuned_temp:.3f} recall = {tuned_recall:.4f}")
    print(f"  improvement = {improvement:+.4f}")

    return {
        "name": name,
        "status": "ok",
        "n_docs": n_docs,
        "n_gold_questions": n_gold_qs,
        "n_generations": n_generations,
        "sigma": sigma,
        "k": k,
        "default_temperature": DEFAULT_TEMPERATURE,
        "default_recall": canonical_recall,
        "tuned_temperature": tuned_temp,
        "tuned_recall": tuned_recall,
        "improvement": improvement,
        "elapsed_s": elapsed,
        "history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks", nargs="*", default=["qasper", "cuad"],
        help="Subset of benchmarks to tune. Default: qasper + cuad (the two "
             "where V4 found meaningful headroom).",
    )
    parser.add_argument("--n-docs", type=int, default=10)
    parser.add_argument("--n-generations", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--out-dir", default="results/stage3")
    args = parser.parse_args()

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
                name=name, n_docs=args.n_docs, n_generations=args.n_generations,
                sigma=args.sigma, seed=args.seed, k=args.k,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {"name": name, "status": "error", "error": repr(e)}

        out_path = out_dir / f"tuned_temperature_{name}.json"
        result["_manifest"] = build_manifest(seed=args.seed, extra={
            "experiment": "stage3_attention_tuning",
            "benchmark": name,
            "n_docs": args.n_docs,
            "n_generations": args.n_generations,
            "sigma": args.sigma,
            "k": args.k,
        })
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"  -> saved {out_path}")
        summary.append({
            "name": name, "status": result.get("status"),
            "default_recall": result.get("default_recall"),
            "tuned_recall": result.get("tuned_recall"),
            "tuned_temperature": result.get("tuned_temperature"),
            "improvement": result.get("improvement"),
        })

    print()
    print("=" * 78)
    print("  ATTENTION-MEMORY TUNING SUMMARY")
    print("=" * 78)
    print(f"  Total elapsed: {time.time() - t_total:.1f}s")
    print(f"  {'benchmark':<14}  {'default':>10}  {'tuned':>10}  {'tuned_tau':>10}  {'lift':>10}")
    for s in summary:
        if s.get("status") == "ok":
            print(
                f"  {s['name']:<14}  {s['default_recall']:>10.4f}  "
                f"{s['tuned_recall']:>10.4f}  {s['tuned_temperature']:>10.3f}  "
                f"{s['improvement']:>+10.4f}"
            )
        else:
            print(f"  {s['name']:<14}  {s.get('status', '?'):>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
