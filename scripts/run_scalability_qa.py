"""
Judged accuracy-vs-corpus-size harness — audit critique B2.

The existing scalability analysis (scripts/analyze_scalability.py) is
ARITHMETIC ONLY: it shows the dump-all prompt grows O(N) and exceeds the
128K window at N≈11 CUAD contracts. It does NOT show that selective memory
keeps *answering correctly* at large N while dump-all degrades. This harness
closes that gap.

Design: a FIXED probe of questions drawn from the first `probe-docs`
contracts (so every probe question's gold evidence is in memory at every N).
For each N ∈ {50,150,300,510} and each config:
  1. Ingest docs[0:N] cumulatively (global step indexing, doc-title prefix).
  2. Ask each probe question at end-of-corpus (current_step = end), retrieve,
     answer with gpt-4o-mini. dump-all sends the full context (truncated /
     skipped when it exceeds the window); selective sends top-k.
  3. Emit a judge queue per (config, N): scaling__<bench>__<config>__N<N>__seed<seed>.

After judging 1-by-1 (Claude), plot judged accuracy vs N: selective stays
flat; dump-all collapses once truncation bites. Reuses run_corpus_qa's
answer_one_qa + write_judge_queue so the schema matches existing cells.

Usage:
    # validate plumbing without API cost:
    python scripts/run_scalability_qa.py --benchmark cuad --n-values 5 10 \
        --probe-docs 5 --probe-size 3 --dry-run
    # real run (Phase 1):
    python scripts/run_scalability_qa.py --benchmark cuad \
        --n-values 50 150 300 510 --probe-docs 50 --probe-size 100 \
        --configs v4t-corpus-tuned dump-all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from environment.benchmarks import get_adapter
from evaluation.claude_judge_queue import write_judge_queue
from memory.event import Event
from results.manifest import build_manifest
from scripts.run_corpus_qa import (
    LLMAgent, MAX_ANSWER_TOKENS, answer_one_qa, build_memory, _make_token_counter,
)


def build_probe(docs, probe_docs: int, probe_size: int):
    """Pick up to `probe_size` questions from the first `probe_docs` docs,
    each carrying its gold global-step set (stable across all N because these
    docs are ingested first). Returns (probe, doc_start_steps)."""
    doc_start = []
    step = 0
    for doc in docs:
        doc_start.append(step)
        step += len(doc.get("paragraphs", []))
    probe = []
    for doc_idx in range(min(probe_docs, len(docs))):
        doc = docs[doc_idx]
        paras = doc.get("paragraphs", [])
        title = str(doc.get("title", f"doc_{doc_idx}"))[:120]
        for qa in (doc.get("qa_pairs", []) or [])[:1]:
            rel = qa.get("relevant_paragraphs", []) or []
            gold = {doc_start[doc_idx] + i for i in rel if 0 <= i < len(paras)}
            if not gold:
                continue
            probe.append({
                "qid": f"probe_d{doc_idx}_q0",
                "question": qa["question"],
                "gold_answer": qa.get("answer", qa.get("gold_answer", "")),
                "doc_title": title,
                "gold": gold,
            })
            if len(probe) >= probe_size:
                return probe
    return probe


def ingest_n(memory, docs, n: int, seed: int) -> int:
    """Ingest docs[0:n] cumulatively; return total global steps."""
    step = 0
    for doc_idx in range(min(n, len(docs))):
        doc = docs[doc_idx]
        title = str(doc.get("title", f"doc_{doc_idx}"))[:120]
        for p_idx, para in enumerate(doc.get("paragraphs", [])):
            obs = f"[{title}] {para}" if p_idx == 0 else para
            memory.add_event(Event(step=step, observation=obs, action="read"),
                             episode_seed=seed)
            step += 1
    return step


def run_cell(benchmark, config, n, docs, probe, k, model, seed, dry_run):
    print(f"\n  [{config} @ N={n}] ingesting {n} docs...")
    memory, _ = build_memory(config, benchmark)
    total_steps = ingest_n(memory, docs, n, seed)
    uncapped = (config == "dump-all")
    agent = None if dry_run else LLMAgent(model=model, temperature=0.0,
                                          max_tokens=MAX_ANSWER_TOKENS, seed=seed)
    token_count = _make_token_counter(model)
    results = []
    overflow = 0
    for p in probe:
        if dry_run:
            # exercise retrieval + overflow logic without the API call
            retrieved = memory.get_relevant_events(
                p["question"], current_step=total_steps, k=k)
            results.append({"qid": p["qid"], "question": p["question"],
                            "scoped_question": p["question"],
                            "gold_answer": p["gold_answer"],
                            "predicted": "[DRY_RUN]",
                            "recall_at_k": 1.0 if ({e.step for e in retrieved} & p["gold"]) else 0.0,
                            "retrieved_steps": [e.step for e in retrieved], "k": k,
                            "context_overflow": False, "answer_fallback": False})
            continue
        r = answer_one_qa(
            agent=agent, memory=memory, question=p["question"],
            gold_answer=p["gold_answer"], doc_title=p["doc_title"],
            relevant_global_steps=p["gold"], current_step=total_steps,
            k=k, model=model, token_count=token_count,
            skip_judge=True, uncapped_context=uncapped,
        )
        r["qid"] = p["qid"]
        if r.get("context_overflow"):
            overflow += 1
        results.append(r)
    # Judge queue: exclude overflow / fallback (not model output).
    judgeable = [r for r in results
                 if not r.get("context_overflow") and not r.get("answer_fallback")]
    run_id = f"scaling__{benchmark}__{config}__N{n}__seed{seed}"
    if not dry_run:
        write_judge_queue(run_id, (
            {"qid": r["qid"], "benchmark": benchmark, "config": config,
             "mode": "scaling", "docs_seen": n, "question": r["question"],
             "scoped_question": r.get("scoped_question"),
             "gold_answer": r["gold_answer"], "predicted": r["predicted"],
             "retrieved_steps": r.get("retrieved_steps"), "k": r.get("k")}
            for r in judgeable))
    recalls = [r["recall_at_k"] for r in results if r.get("recall_at_k") is not None]
    print(f"    total_steps={total_steps}  probe={len(probe)}  "
          f"overflow={overflow}  judgeable={len(judgeable)}  "
          f"recall@{k}={sum(recalls)/len(recalls):.3f}" if recalls else "")
    return {"config": config, "n": n, "total_steps": total_steps,
            "probe_size": len(probe), "overflow": overflow,
            "judgeable": len(judgeable), "run_id": run_id,
            "mean_recall_at_k": (sum(recalls) / len(recalls)) if recalls else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="cuad")
    ap.add_argument("--n-values", type=int, nargs="+", default=[50, 150, 300, 510])
    ap.add_argument("--configs", nargs="+", default=["v4t-corpus-tuned", "dump-all"])
    ap.add_argument("--probe-docs", type=int, default=50)
    ap.add_argument("--probe-size", type=int, default=100)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip the LLM call; exercise ingestion+retrieval+queue plumbing only")
    ap.add_argument("--out-dir", default="results/stage3")
    args = ap.parse_args()

    adapter = get_adapter(args.benchmark)
    docs = list(adapter.iter_documents(limit=max(args.n_values)))
    print(f"Loaded {len(docs)} docs (requested up to {max(args.n_values)}).")
    probe = build_probe(docs, args.probe_docs, args.probe_size)
    print(f"Probe: {len(probe)} questions from first {args.probe_docs} docs.")

    t0 = time.time()
    cells = []
    for config in args.configs:
        for n in args.n_values:
            cells.append(run_cell(args.benchmark, config, n, docs, probe,
                                  args.k, args.model, args.seed, args.dry_run))

    summary = {
        "benchmark": args.benchmark, "n_values": args.n_values,
        "configs": args.configs, "probe_docs": args.probe_docs,
        "probe_size": len(probe), "k": args.k, "seed": args.seed,
        "dry_run": args.dry_run, "cells": cells,
        "elapsed_seconds": time.time() - t0,
        "_manifest": build_manifest(seed=args.seed, extra={
            "experiment": "stage3_scalability_judged", "benchmark": args.benchmark}),
    }
    out = ROOT / args.out_dir / f"scalability_judged_{args.benchmark}.json"
    if not args.dry_run:
        out.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\n-> saved {out}")
    print(f"\nDone in {summary['elapsed_seconds']:.1f}s. Cells:")
    for c in cells:
        print(f"  {c['config']:<18} N={c['n']:<4} steps={c['total_steps']:<6} "
              f"overflow={c['overflow']:<4} judgeable={c['judgeable']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
