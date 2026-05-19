"""
Stage 3 — Layer 3 API smoke (NOT a pytest; costs ~$0.10).

For each of the 6 benchmark adapters: pull 1 document, build V4 memory,
run the reading phase, retrieve top-k=8 events for the first qa_pair's
question, call ``LLMAgent.answer_question`` against gpt-4o-mini, then
score with ``llm_judge_score_multi_ref``.

Confirms end-to-end:
  * Adapter integrates with DocumentQA cleanly.
  * Retrieval feeds the LLM with the right shape (Event[]).
  * LLM produces a non-empty answer.
  * LLM judge produces a numeric score in [0, 1].
  * Token accounting tracks per-question cost.

Without ``OPENAI_API_KEY``, the script still runs — but uses the
heuristic fallback for both the agent and the judge. That path is the
CI-safe smoke check (proves the wiring works without spending money).

Usage:
    # Real API (requires OPENAI_API_KEY)
    python scripts/smoke_stage3_api.py

    # Subset
    python scripts/smoke_stage3_api.py --benchmarks hotpotqa qasper

    # Fallback-only (no API call, free)
    OPENAI_API_KEY= python scripts/smoke_stage3_api.py

Output: ``results/stage3_api_smoke.json`` with per-benchmark
{question, predicted, gold, judge_score, prompt_tokens, completion_tokens,
cost_usd, retrieved_steps} + manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Smoke benchmarks load real data — keep HF offline.
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.llm_agent import LLMAgent
from environment.benchmarks import ADAPTERS, document_fingerprint, get_adapter
from environment.document_qa import DocumentQA
from evaluation.document_qa_llm_judge import (
    get_judge_stats,
    llm_judge_score_multi_ref,
    reset_judge_stats,
)
from evaluation.document_qa_memory import _run_reading_phase
from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4
from results.manifest import build_manifest

# Same V4 smoke params as tests/test_benchmark_smoke.py — embedding-only,
# no theta_store filtering. This is the "pipeline works" baseline; Phase 4
# will compare multiple V4 / V5 / V6 thetas + other memory systems.
_V4_SMOKE_PARAMS = MemoryParamsV4(
    theta_store=0.0,
    theta_novel=0.0, theta_erich=0.0, theta_surprise=0.0,
    theta_entity=0.0, theta_temporal=0.0, theta_decay=0.0,
    w_graph=0.0, w_embed=1.0, w_recency=0.0,
    mode="learnable",
)


def smoke_one_benchmark(name: str, k: int = 8, model: str = "gpt-4o-mini") -> dict:
    """Run one end-to-end question through the full Stage 3 pipeline."""
    print()
    print("=" * 78)
    print(f"  {name}")
    print("=" * 78)
    adapter = get_adapter(name)
    docs = list(adapter.iter_documents(limit=1))
    if not docs:
        return {"name": name, "ok": False, "reason": "no docs returned"}
    doc = docs[0]
    doc_fp = document_fingerprint(doc)
    n_paragraphs = len(doc["paragraphs"])
    qa = doc["qa_pairs"][0]
    question = qa["question"]
    gold_answer = qa["answer"]
    print(f"  title: {doc['title'][:80]}")
    print(f"  n_paragraphs = {n_paragraphs}  n_qa_pairs = {len(doc['qa_pairs'])}")
    print(f"  question: {question[:100]!r}")
    if isinstance(gold_answer, list):
        print(f"  gold ({len(gold_answer)} refs): {[g[:60] for g in gold_answer]}")
    else:
        print(f"  gold: {gold_answer[:120]!r}")

    # Reading phase
    memory = GraphMemoryV4(_V4_SMOKE_PARAMS)
    env = DocumentQA(document=doc, seed=42, question_shuffle=False)
    t_read0 = time.monotonic()
    _run_reading_phase(env, memory, episode_seed=42)
    t_read = time.monotonic() - t_read0
    stats = memory.get_stats()
    print(f"  reading phase: {stats.get('n_events', 0)} events stored in {t_read:.1f}s")

    # Retrieve top-k for the question
    current_step = n_paragraphs  # QA phase begins right after reading
    retrieved = memory.get_relevant_events(question, current_step=current_step, k=k)
    retrieved_steps = [ev.step for ev in retrieved]
    relevant = qa.get("relevant_paragraphs", [])
    hit = bool(set(retrieved_steps) & set(relevant)) if relevant else None
    print(
        f"  retrieval: top-{k} steps = {retrieved_steps}  "
        f"gold_in_topk = {hit if relevant else '(no gold)'}"
    )

    # Call the LLM agent
    agent = LLMAgent(model=model, temperature=0.0, max_tokens=10, seed=42)
    t_llm0 = time.monotonic()
    predicted = agent.answer_question(question, past_events=retrieved)
    t_llm = time.monotonic() - t_llm0
    stats_after = agent.session_stats
    print(f"  predicted ({t_llm:.2f}s): {predicted[:200]!r}")
    print(
        f"  cost so far: ${agent.session_cost_usd:.6f}  "
        f"prompt_tokens={stats_after.total_prompt_tokens}  "
        f"completion_tokens={stats_after.total_completion_tokens}"
    )

    # Score with judge
    t_j0 = time.monotonic()
    judge_score = llm_judge_score_multi_ref(predicted, gold_answer, model=model)
    t_j = time.monotonic() - t_j0
    judge_stats = get_judge_stats()
    print(
        f"  judge score: {judge_score:.3f} ({t_j:.2f}s)  "
        f"judge_calls={judge_stats['total_judge_calls']}  "
        f"judge_cost=${judge_stats['total_judge_cost_usd']:.6f}"
    )

    return {
        "name": name,
        "ok": True,
        "schema_version": adapter.SCHEMA_VERSION,
        "dataset_fingerprint": adapter.dataset_fingerprint(),
        "document_fingerprint": doc_fp,
        "title": doc["title"],
        "n_paragraphs": n_paragraphs,
        "question": question,
        "gold_answer": gold_answer,
        "predicted": predicted,
        "judge_score": judge_score,
        "retrieved_steps": retrieved_steps,
        "relevant_paragraphs": relevant,
        "gold_in_topk": hit,
        "n_events_stored": stats.get("n_events", 0),
        "reading_seconds": t_read,
        "llm_seconds": t_llm,
        "judge_seconds": t_j,
        "agent_cost_usd": agent.session_cost_usd,
        "agent_prompt_tokens": stats_after.total_prompt_tokens,
        "agent_completion_tokens": stats_after.total_completion_tokens,
        "judge_cost_usd_cumulative": judge_stats["total_judge_cost_usd"],
        "judge_calls_cumulative": judge_stats["total_judge_calls"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks", nargs="*", default=sorted(ADAPTERS.keys()),
        help="Subset of benchmarks to smoke. Default: all 6.",
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model name for both agent and judge.",
    )
    parser.add_argument("--k", type=int, default=8, help="Retrieval top-k.")
    parser.add_argument(
        "--out", default="results/stage3_api_smoke.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]:
        print(f"[API key DETECTED] running smoke against {args.model}")
    else:
        print("[NO API key] running in fallback / heuristic mode (free)")

    reset_judge_stats()

    results: list[dict] = []
    t0 = time.time()
    for name in args.benchmarks:
        if name not in ADAPTERS:
            print(f"  [SKIP] unknown benchmark: {name!r}")
            continue
        try:
            r = smoke_one_benchmark(name, k=args.k, model=args.model)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = {"name": name, "ok": False, "reason": repr(e)}
        results.append(r)
    elapsed = time.time() - t0

    print()
    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    n_ok = sum(1 for r in results if r.get("ok"))
    n_total = len(results)
    total_agent_cost = sum(r.get("agent_cost_usd", 0.0) for r in results)
    judge_stats = get_judge_stats()
    print(
        f"  {n_ok}/{n_total} benchmarks completed   "
        f"agent_cost=${total_agent_cost:.4f}   "
        f"judge_cost=${judge_stats['total_judge_cost_usd']:.4f}   "
        f"elapsed={elapsed:.1f}s"
    )
    for r in results:
        if r.get("ok"):
            print(
                f"  [OK] {r['name']:<14} "
                f"judge={r['judge_score']:.3f}  "
                f"gold_in_topk={r.get('gold_in_topk')!s:<5}  "
                f"agent_cost=${r['agent_cost_usd']:.5f}"
            )
        else:
            print(f"  [FAIL] {r['name']:<14} {r.get('reason', '?')[:80]}")

    # Persist
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_manifest": build_manifest(seed=42, extra={
            "experiment": "stage3_api_smoke",
            "model": args.model,
            "k": args.k,
            "benchmarks": args.benchmarks,
            "has_api_key": bool(os.environ.get("OPENAI_API_KEY", "")),
        }),
        "experiment": "stage3_api_smoke",
        "config": {
            "model": args.model,
            "k": args.k,
            "benchmarks": args.benchmarks,
        },
        "total_agent_cost_usd": total_agent_cost,
        "total_judge_cost_usd": judge_stats["total_judge_cost_usd"],
        "elapsed_s": elapsed,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  Output: {out_path}")

    # Soft assertions for sanity (not pytest, but signal exit code).
    failures: list[str] = []
    for r in results:
        if not r.get("ok"):
            failures.append(r["name"])
            continue
        if not r.get("predicted", "").strip():
            failures.append(f"{r['name']}: empty predicted answer")
        score = r.get("judge_score", -1)
        if not (0.0 <= score <= 1.0):
            failures.append(f"{r['name']}: judge_score={score} out of [0,1]")
    if failures:
        print(f"\n  [FAIL] soft assertions failed: {failures}")
        return 1
    print(f"\n  [OK] all {n_ok} benchmarks produced valid responses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
