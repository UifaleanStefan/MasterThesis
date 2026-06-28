"""
Generate (corpus-descriptor, tuned-θ) training pairs from corpus SUB-SAMPLES.

Why: the θ-from-task predictability question (plan Phase 4) cannot be answered
from 5 benchmark-level θ vectors. This harness manufactures many pairs by tuning
θ (CMA-ES on recall@k — no LLM) on random doc slices, recording the slice's
descriptor and the θ it induces. Output feeds optimization/theta_predict.py,
which runs a leave-one-benchmark-out predictability test.

Each record (one JSONL line, resumable — re-running skips done (benchmark, slice)):
  benchmark, slice_idx, slice_doc_indices, descriptor (9-D), n_eval_q,
  canonical_recall, tuned_recall, improvement, tuned_theta_vec (10-D).

Usage:
    python -m tuning.gen_theta_subsamples --benchmarks cuad qasper financebench \
        hotpotqa longmemeval --slices 8 --slice-size 30 --n-generations 6
"""
from __future__ import annotations

import argparse
import json
import os
import random
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
from memory.graph_memory_v4 import GraphMemoryV4
from optimization.cma_es import run_cmaes_optimization
from optimization.theta_descriptor import descriptor_vector, descriptors_from_docs
from tuning.tune_v4_per_benchmark import CANONICAL_THETA_VEC, vec_to_params

OUT = ROOT / "results" / "stage3" / "theta_subsamples.jsonl"


def make_slice_eval_fn(slice_docs, k, seed):
    """Mean recall@k under V4ₜ over an arbitrary doc slice (mirrors
    tuning.tune_v4t_corpus.make_corpus_eval_fn but on a given docs list)."""
    eval_tasks = []
    gstep = 0
    for doc_idx, doc in enumerate(slice_docs):
        doc_start = gstep
        paras = doc.get("paragraphs", [])
        gstep += len(paras)
        end_of_doc = gstep
        title = str(doc.get("title", f"doc_{doc_idx}"))[:120]
        for qa in (doc.get("qa_pairs", []) or [])[:1]:
            rel = qa.get("relevant_paragraphs", []) or []
            gold = {doc_start + i for i in rel if 0 <= i < len(paras)}
            if gold:
                eval_tasks.append((f"[Regarding {title}] {qa['question']}", gold, end_of_doc))

    def eval_fn(vec):
        params = vec_to_params(vec)
        params.text_mode_entities = True
        mem = GraphMemoryV4(params)
        g = 0
        for doc_idx, doc in enumerate(slice_docs):
            title = str(doc.get("title", f"doc_{doc_idx}"))[:120]
            for p_idx, para in enumerate(doc.get("paragraphs", [])):
                obs = f"[{title}] {para}" if p_idx == 0 else para
                mem.add_event(Event(step=g, observation=obs, action="read"), episode_seed=seed)
                g += 1
        hits = []
        for q, gold, cur in eval_tasks:
            steps = {e.step for e in mem.get_relevant_events(q, current_step=cur, k=k)}
            hits.append(1.0 if (steps & gold) else 0.0)
        return float(np.mean(hits)) if hits else 0.0

    return eval_fn, len(eval_tasks)


def done_keys() -> set:
    if not OUT.exists():
        return set()
    keys = set()
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            keys.add((r["benchmark"], r["slice_idx"]))
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", nargs="+",
                    default=["cuad", "qasper", "financebench", "hotpotqa", "longmemeval"],
                    choices=sorted(ADAPTERS.keys()))
    ap.add_argument("--slices", type=int, default=8, help="random slices per benchmark")
    ap.add_argument("--slice-size", type=int, default=30, help="docs per slice")
    ap.add_argument("--pool", type=int, default=150, help="docs to sample slices from")
    ap.add_argument("--n-generations", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys()
    print(f"Resuming: {len(done)} pairs already recorded in {OUT.name}")

    for bench in args.benchmarks:
        docs_pool = list(get_adapter(bench).iter_documents(limit=args.pool))
        n = len(docs_pool)
        size = min(args.slice_size, n)
        print(f"\n=== {bench}: pool={n} docs, slice_size={size} ===")
        for s in range(args.slices):
            if (bench, s) in done:
                print(f"  slice {s}: already done, skip")
                continue
            rng = random.Random(1000 * s + hash(bench) % 997)
            idx = sorted(rng.sample(range(n), size))
            slice_docs = [docs_pool[i] for i in idx]
            eval_fn, n_eval = make_slice_eval_fn(slice_docs, args.k, seed=42)
            if n_eval < 5:
                print(f"  slice {s}: only {n_eval} eval QAs, skip")
                continue
            t0 = time.time()
            canon = eval_fn(CANONICAL_THETA_VEC.copy())
            best_vec, hist = run_cmaes_optimization(
                eval_fn, n_params=10, n_generations=args.n_generations,
                sigma=args.sigma, seed=42, clip_to_unit=True, verbose=False)
            tuned = float(hist[-1]["best_fitness"]) if hist else canon
            rec = {
                "benchmark": bench, "slice_idx": s, "slice_doc_indices": idx,
                "slice_size": size, "n_eval_q": n_eval, "k": args.k,
                "descriptor": descriptors_from_docs(slice_docs),
                "descriptor_vec": descriptor_vector(descriptors_from_docs(slice_docs)),
                "canonical_recall": canon, "tuned_recall": tuned,
                "improvement": tuned - canon,
                "tuned_theta_vec": [float(v) for v in best_vec],
            }
            with OUT.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
            print(f"  slice {s}: n_eval={n_eval} canon={canon:.3f} tuned={tuned:.3f} "
                  f"lift={tuned - canon:+.3f}  ({time.time() - t0:.0f}s)")

    total = len(done_keys())
    print(f"\nTotal (descriptor, theta) pairs now: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
