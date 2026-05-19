"""
Adapter-aware retrieval-only evaluator — Stage 3 Phase 1.5 workstream A.

Parallel to ``evaluation.document_qa_memory.run_document_qa_memory_eval``
but operates on the new ``environment.benchmarks`` adapters rather than
the hand-authored ``_DOCUMENTS`` registry.

For each (system, document) cell, runs the reading phase, then for each
qa_pair retrieves top-k and computes recall@k against the document's
``relevant_paragraphs``. Aggregates across docs to mean recall per
system. No LLM in the loop — $0 cost.

Public surface:

    from evaluation.benchmark_memory_eval import run_benchmark_memory_eval
    result = run_benchmark_memory_eval(
        adapter_name="hotpotqa", n_docs=30, k=8,
        tuned_thetas={"hotpotqa": [...]},  # optional
    )

Returns:
    {
      system_name: {
        "mean_recall": float,
        "std_recall": float,
        "n_questions_total": int,
        "n_questions_with_gold": int,
        "per_doc_means": list[float],   # one entry per doc, mean of its gold-qa_pairs
        "eval_seconds": float,
      }
    }
"""

from __future__ import annotations

import os
import statistics
import time
from typing import Any, Callable

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from environment.benchmarks import get_adapter
from environment.document_qa import DocumentQA
from evaluation.document_qa_memory import (
    _make_document_qa_memory_systems,
    _recall_at_k_for_qa,
    _run_reading_phase,
)
from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4


def _build_tuned_v4_factory(theta_vec: list[float]) -> Callable[[], GraphMemoryV4]:
    """Return a no-arg factory producing GraphMemoryV4 with the given vector.

    The vector is the [0,1]^10 form that the CMA-ES tuner produces.
    Same w_* * 4 scaling convention as `tuning.tune_v4_per_benchmark.vec_to_params`.
    """
    import numpy as np
    v = np.clip(np.asarray(theta_vec, dtype=np.float64), 0.0, 1.0)
    params = MemoryParamsV4(
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
    return lambda: GraphMemoryV4(params)


def run_benchmark_memory_eval(
    adapter_name: str,
    n_docs: int = 30,
    k: int = 8,
    memory_factories: dict[str, Callable[[], Any]] | None = None,
    seed: int = 42,
    tuned_thetas: dict[str, list[float]] | None = None,
    skip_systems: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, dict]:
    """Evaluate all memory systems on ``n_docs`` of ``adapter_name``.

    Parameters
    ----------
    adapter_name : str
        One of the keys in ``environment.benchmarks.ADAPTERS``.
    n_docs : int
        Number of documents to pull from the adapter (default 30).
    k : int
        Retrieval top-k for recall@k (default 8).
    memory_factories : dict, optional
        ``{system_name: factory}`` mapping. Defaults to the 12-system
        suite from ``_make_document_qa_memory_systems()``.
    seed : int
        Episode/RNG seed for determinism.
    tuned_thetas : dict, optional
        ``{benchmark_name: theta_vec_10D}`` — when provided AND
        ``adapter_name`` is a key, adds an extra system named
        ``V4-tuned-{adapter_name}`` with that theta.
    skip_systems : list, optional
        System names to exclude from the eval.

    Returns
    -------
    dict
        Per-system aggregate stats (see module docstring).
    """
    adapter = get_adapter(adapter_name)
    docs = list(adapter.iter_documents(limit=n_docs))
    if not docs:
        raise RuntimeError(f"Adapter {adapter_name!r} yielded no documents")
    if verbose:
        n_para_avg = sum(len(d["paragraphs"]) for d in docs) // len(docs)
        n_qa_total = sum(len(d["qa_pairs"]) for d in docs)
        print(f"  [{adapter_name}] {len(docs)} docs, "
              f"avg n_paragraphs={n_para_avg}, total qa_pairs={n_qa_total}")

    if memory_factories is None:
        memory_factories = _make_document_qa_memory_systems()
    else:
        memory_factories = dict(memory_factories)

    # Inject tuned V4 if available for this benchmark.
    if tuned_thetas and adapter_name in tuned_thetas:
        sys_name = f"V4-tuned-{adapter_name}"
        memory_factories[sys_name] = _build_tuned_v4_factory(tuned_thetas[adapter_name])
        if verbose:
            print(f"  [{adapter_name}] added system {sys_name}")

    results: dict[str, dict] = {}
    skip = set(skip_systems or [])

    for sys_name, factory in memory_factories.items():
        if sys_name in skip:
            continue
        t0 = time.time()
        per_doc_means: list[float] = []
        all_recalls: list[float] = []
        n_with_gold = 0
        n_total = 0
        for doc in docs:
            memory = factory()
            env = DocumentQA(document=doc, seed=seed, question_shuffle=False)
            _run_reading_phase(env, memory, episode_seed=seed)
            n_paragraphs = len(doc["paragraphs"])
            doc_recalls_with_gold: list[float] = []
            for qa_idx, qa in enumerate(doc["qa_pairs"]):
                n_total += 1
                relevant = qa.get("relevant_paragraphs", []) or []
                if not relevant:
                    continue
                n_with_gold += 1
                current_step = n_paragraphs + qa_idx
                r = _recall_at_k_for_qa(
                    memory, qa["question"], relevant,
                    k=k, current_step=current_step,
                )
                all_recalls.append(r)
                doc_recalls_with_gold.append(r)
            if doc_recalls_with_gold:
                per_doc_means.append(statistics.mean(doc_recalls_with_gold))
            else:
                per_doc_means.append(float("nan"))

        elapsed = time.time() - t0
        if all_recalls:
            mean_recall = statistics.mean(all_recalls)
            std_recall = statistics.stdev(all_recalls) if len(all_recalls) > 1 else 0.0
        else:
            mean_recall = 0.0
            std_recall = 0.0

        results[sys_name] = {
            "mean_recall": mean_recall,
            "std_recall": std_recall,
            "n_questions_total": n_total,
            "n_questions_with_gold": n_with_gold,
            "per_doc_means": per_doc_means,
            "eval_seconds": elapsed,
        }
        if verbose:
            print(
                f"    {sys_name:<22}  mean={mean_recall:.4f}  "
                f"std={std_recall:.3f}  n_gold={n_with_gold}/{n_total}  "
                f"({elapsed:.1f}s)"
            )

    return results


def aggregate_across_benchmarks(
    per_benchmark: dict[str, dict[str, dict]],
) -> dict[str, dict]:
    """Roll up per-benchmark results into a {system: {benchmark: mean_recall}} table.

    Used by `scripts/run_stage3_retrieval.py` to produce the headline summary
    JSON consumed by the frontend.
    """
    all_systems: set[str] = set()
    for r in per_benchmark.values():
        all_systems.update(r.keys())

    summary: dict[str, dict] = {sys_name: {} for sys_name in sorted(all_systems)}
    for benchmark, sys_results in per_benchmark.items():
        for sys_name, stats in sys_results.items():
            summary[sys_name][benchmark] = stats["mean_recall"]
        # Mark missing systems as None for fairness in the table.
        for sys_name in all_systems:
            if sys_name not in sys_results:
                summary[sys_name][benchmark] = None
    return summary
