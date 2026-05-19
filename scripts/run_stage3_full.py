"""
Stage 3 Phase 4 orchestrator — the loop Phase 4 will run when the API
key is available. Built now in Phase 1.5 so we can dry-run the entire
pipeline cost-free and produce a real cost projection.

Three modes:
  * ``--mode retrieval`` — recall@k only, no LLM. Equivalent to
    ``scripts/run_stage3_retrieval.py`` but emitted per-cell instead of
    per-benchmark.
  * ``--mode dry-run`` — full pipeline but tiktoken-counts the prompts
    that WOULD be sent to the OpenAI API. Produces a cost projection in
    ``results/stage3/cost_projection.json``. The fallback heuristic from
    ``LLMAgent._fallback_decide`` still runs so the per-cell JSONs have
    real (heuristic) predicted answers.
  * ``--mode full`` — real OpenAI calls. Requires ``OPENAI_API_KEY``.

Per-cell output: ``results/stage3/cells/{benchmark}__{config}__{seed}.json``
containing per-question prediction, gold answer, judge score, retrieval
recall@k, token counts, cost.

Aggregate output: ``results/stage3/stage3_runs.json`` with the cross-tab
summary (mean judge score x recall x cost per (benchmark, config)).

Reused infrastructure:
  * `agent.llm_agent.LLMAgent` for full mode.
  * `evaluation.document_qa_llm_judge.llm_judge_score_multi_ref` for scoring.
  * `evaluation.document_qa_memory._run_reading_phase, _recall_at_k_for_qa`
    for the retrieval loop.
  * `results.manifest.build_manifest` for provenance.

Usage:
    # Cost projection only — runs in minutes for free.
    python scripts/run_stage3_full.py --mode dry-run --benchmarks all \\
        --configs v4-canonical v4-tuned flat-50 --n-questions 30

    # Full run (real API, $$$).
    $env:OPENAI_API_KEY = "sk-..."
    python scripts/run_stage3_full.py --mode full --benchmarks all \\
        --configs v4-tuned --n-questions 30
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.llm_agent import LLMAgent
from environment.benchmarks import ADAPTERS, get_adapter, document_fingerprint
from environment.document_qa import DocumentQA
from evaluation.document_qa_llm_judge import (
    get_judge_stats,
    llm_judge_score_multi_ref,
    reset_judge_stats,
)
from evaluation.document_qa_memory import _recall_at_k_for_qa, _run_reading_phase
from memory.flat_memory import FlatMemory
from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4
from results.manifest import build_manifest


# Pricing (USD per 1M tokens) — mirrors agent/llm_agent.py._PRICING.
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

# Token budget per QA-answer completion (matches LLMAgent.answer_question's max_tokens=150).
MAX_ANSWER_TOKENS = 150


# ---------------------------------------------------------------------------
# Config registry — memory system factories per Phase-4 config name.
# ---------------------------------------------------------------------------


_CANONICAL_V4_PARAMS = MemoryParamsV4(
    theta_store=0.293, theta_novel=0.908, theta_erich=0.198, theta_surprise=0.785,
    theta_entity=0.285, theta_temporal=0.278, theta_decay=0.668,
    w_graph=0.0, w_embed=1.079, w_recency=3.777, mode="learnable",
)


def _build_v4_tuned_factory(theta_vec: list[float]) -> Callable[[], GraphMemoryV4]:
    """Build a V4 factory from a [0,1]^10 tuned-theta vector."""
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


def _load_tuned_thetas(out_dir: Path) -> dict[str, list[float]]:
    tuned: dict[str, list[float]] = {}
    for path in out_dir.glob("tuned_theta_*.json"):
        try:
            data = json.loads(path.read_text())
            if data.get("status") != "ok":
                continue
            vec = data.get("tuned_theta_vec")
            name = data.get("name")
            if name and isinstance(vec, list) and len(vec) == 10:
                tuned[name] = vec
        except Exception:
            pass
    return tuned


def build_config_factories(
    benchmark_name: str, tuned_thetas: dict[str, list[float]],
) -> dict[str, Callable[[], Any]]:
    """Return ``{config_name: memory_factory}`` for the Phase-4 configs."""
    factories: dict[str, Callable[[], Any]] = {
        "v4-canonical": lambda: GraphMemoryV4(_CANONICAL_V4_PARAMS),
        "flat-50":      lambda: FlatMemory(window_size=50),
    }
    if benchmark_name in tuned_thetas:
        factories["v4-tuned"] = _build_v4_tuned_factory(tuned_thetas[benchmark_name])
    return factories


# ---------------------------------------------------------------------------
# Token counter (tiktoken, with no-op fallback if unavailable)
# ---------------------------------------------------------------------------


def _get_token_counter(model: str) -> Callable[[str], int]:
    """Return a token-counting function for ``model``. Falls back to len()/4 if tiktoken missing."""
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            # gpt-4o-mini not always registered in older tiktoken; use o200k_base.
            enc = tiktoken.get_encoding("o200k_base")
        return lambda s: len(enc.encode(s))
    except ImportError:
        # Rough fallback: ~4 chars per token. Better than crashing.
        return lambda s: max(1, len(s) // 4)


# ---------------------------------------------------------------------------
# Per-cell processing
# ---------------------------------------------------------------------------


_QA_SYSTEM_PROMPT = (
    "You are a question-answering assistant. You receive a question and relevant document passages. "
    "Answer the question concisely using only the provided context. "
    "Output only the answer text, no preamble or explanation."
)


def _build_qa_prompt(question: str, retrieved_events) -> tuple[str, str]:
    """Reproduce LLMAgent.answer_question's prompt for token counting / actual use."""
    context_lines = []
    for ev in retrieved_events:
        context_lines.append(f"step {ev.step}: {ev.observation}")
    memory_context = "\n".join(context_lines)
    user_msg = f"Relevant passages:\n{memory_context}\n\nQuestion: {question}\n\nAnswer:"
    return _QA_SYSTEM_PROMPT, user_msg


def run_cell(
    benchmark_name: str,
    config_name: str,
    memory_factory: Callable[[], Any],
    n_questions: int,
    k: int,
    mode: str,
    model: str,
    seed: int,
    token_count: Callable[[str], int],
) -> dict:
    """Run one (benchmark, config) cell for `n_questions` questions.

    In `retrieval` mode: only computes recall@k.
    In `dry-run` mode: also counts tokens, computes projected cost via heuristic answer.
    In `full` mode: makes real LLM + judge calls.

    Returns a dict with per-question results + aggregates.
    """
    adapter = get_adapter(benchmark_name)
    docs = list(adapter.iter_documents(limit=n_questions))  # 1 question per doc (most adapters)
    if not docs:
        return {"ok": False, "reason": "no docs"}

    pricing = PRICING.get(model, PRICING["gpt-4o-mini"])

    questions: list[dict] = []
    total_prompt_tokens = 0
    total_completion_tokens_max = 0  # upper bound (max_tokens cap)
    total_recall = 0.0
    total_judge_score = 0.0
    n_with_gold = 0
    n_with_judge = 0

    agent = None
    if mode == "full":
        agent = LLMAgent(model=model, temperature=0.0, max_tokens=MAX_ANSWER_TOKENS, seed=seed)

    t0 = time.time()
    for doc_idx, doc in enumerate(docs):
        # We exercise just the FIRST qa_pair per doc — Phase 4 will typically
        # ask 1 question per doc since reading is the dominant cost.
        qa = doc["qa_pairs"][0]
        question = qa["question"]
        gold_answer = qa["answer"]
        relevant = qa.get("relevant_paragraphs", []) or []

        # Reading phase + retrieval
        memory = memory_factory()
        env = DocumentQA(document=doc, seed=seed, question_shuffle=False)
        _run_reading_phase(env, memory, episode_seed=seed)
        n_paragraphs = len(doc["paragraphs"])
        retrieved = memory.get_relevant_events(question, current_step=n_paragraphs, k=k)
        retrieved_steps = [ev.step for ev in retrieved]
        recall = 1.0 if (relevant and set(retrieved_steps) & set(relevant)) else 0.0
        if relevant:
            n_with_gold += 1
            total_recall += recall

        # Build the prompt we WOULD send to the API (or are sending in full mode).
        sys_prompt, user_msg = _build_qa_prompt(question, retrieved)
        prompt_token_count = token_count(sys_prompt) + token_count(user_msg)
        total_prompt_tokens += prompt_token_count
        total_completion_tokens_max += MAX_ANSWER_TOKENS

        predicted = ""
        judge_score = None
        cell_cost = 0.0
        if mode == "full":
            predicted = agent.answer_question(question, past_events=retrieved)
            judge_score = llm_judge_score_multi_ref(predicted, gold_answer, model=model)
            n_with_judge += 1
            total_judge_score += judge_score
        elif mode == "dry-run":
            # Use the heuristic fallback as a placeholder — keeps the JSON usable.
            agent_h = LLMAgent(model=model, temperature=0.0, max_tokens=MAX_ANSWER_TOKENS, seed=seed)
            # _has_openai is set by _try_init_client; if no key set, fallback runs.
            predicted = agent_h.answer_question(question, past_events=retrieved)
            judge_score = llm_judge_score_multi_ref(predicted, gold_answer, model=model)
            n_with_judge += 1
            total_judge_score += judge_score
            # Projected cost for THIS cell (not actually spent in dry-run).
            cell_cost = (prompt_token_count * pricing["input"]
                         + MAX_ANSWER_TOKENS * pricing["output"]) / 1_000_000

        questions.append({
            "doc_idx": doc_idx,
            "doc_fingerprint": document_fingerprint(doc),
            "n_paragraphs": n_paragraphs,
            "question": question,
            "gold_answer": gold_answer,
            "predicted": predicted,
            "judge_score": judge_score,
            "recall_at_k": recall if relevant else None,
            "retrieved_steps": retrieved_steps,
            "relevant_paragraphs": relevant,
            "prompt_tokens_estimated": prompt_token_count,
            "projected_cost_usd": cell_cost,
        })

    elapsed = time.time() - t0
    mean_recall = total_recall / n_with_gold if n_with_gold else 0.0
    mean_judge = total_judge_score / n_with_judge if n_with_judge else None
    projected_cost = ((total_prompt_tokens * pricing["input"]
                       + total_completion_tokens_max * pricing["output"]) / 1_000_000)

    actual_cost_usd = None
    if mode == "full" and agent is not None:
        actual_cost_usd = agent.session_cost_usd

    return {
        "ok": True,
        "benchmark": benchmark_name,
        "config": config_name,
        "mode": mode,
        "n_questions": len(questions),
        "n_with_gold": n_with_gold,
        "n_with_judge": n_with_judge,
        "mean_recall_at_k": mean_recall,
        "mean_judge_score": mean_judge,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens_max": total_completion_tokens_max,
        "projected_cost_usd": projected_cost,
        "actual_cost_usd": actual_cost_usd,
        "elapsed_seconds": elapsed,
        "questions": questions,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["retrieval", "dry-run", "full"], default="dry-run",
        help="retrieval = recall@k only; dry-run = count tokens + heuristic; full = real API.",
    )
    parser.add_argument(
        "--benchmarks", nargs="*", default=["all"],
        help="Subset or 'all' (default).",
    )
    parser.add_argument(
        "--configs", nargs="*",
        default=["v4-canonical", "v4-tuned", "flat-50"],
        help="Phase-4 configurations to evaluate per benchmark.",
    )
    parser.add_argument("--n-questions", type=int, default=30)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--out-dir", default="results/stage3",
        help="Output dir (cells/ subdir, cost_projection.json, stage3_runs.json).",
    )
    args = parser.parse_args()

    if len(args.benchmarks) == 1 and args.benchmarks[0].lower() == "all":
        args.benchmarks = sorted(ADAPTERS.keys())

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    tuned_thetas = _load_tuned_thetas(out_dir)
    if "v4-tuned" in args.configs:
        print(f"[tuned thetas] loaded for: {sorted(tuned_thetas.keys())}")

    token_count = _get_token_counter(args.model)
    if args.mode == "full":
        if not os.environ.get("OPENAI_API_KEY"):
            print("[full mode] WARNING: OPENAI_API_KEY not set — LLM calls will fall back to heuristic.")
    reset_judge_stats()

    print()
    print("=" * 78)
    print(f"  Stage 3 orchestrator — mode={args.mode}")
    print(f"  benchmarks={args.benchmarks}")
    print(f"  configs={args.configs}")
    print(f"  n_questions={args.n_questions}, k={args.k}, model={args.model}")
    print("=" * 78)

    cells: list[dict] = []
    t_total = time.time()
    for benchmark in args.benchmarks:
        if benchmark not in ADAPTERS:
            print(f"  [SKIP] unknown benchmark: {benchmark!r}")
            continue
        factories = build_config_factories(benchmark, tuned_thetas)
        for config in args.configs:
            if config not in factories:
                if config == "v4-tuned":
                    print(f"  [SKIP] {benchmark}/{config}: no tuned theta available "
                          f"(run tuning/tune_v4_per_benchmark.py first)")
                else:
                    print(f"  [SKIP] {benchmark}/{config}: unknown config")
                continue
            print()
            print(f"---- {benchmark} x {config} ----")
            try:
                cell = run_cell(
                    benchmark_name=benchmark,
                    config_name=config,
                    memory_factory=factories[config],
                    n_questions=args.n_questions,
                    k=args.k,
                    mode=args.mode,
                    model=args.model,
                    seed=args.seed,
                    token_count=token_count,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                cell = {"ok": False, "benchmark": benchmark, "config": config, "reason": repr(e)}

            if cell.get("ok"):
                print(
                    f"    n={cell['n_questions']}  recall@{args.k}={cell['mean_recall_at_k']:.3f}  "
                    f"judge={cell['mean_judge_score']:.3f}  "
                    f"prompt_tok={cell['total_prompt_tokens']:,}  "
                    f"projected=${cell['projected_cost_usd']:.4f}  "
                    f"({cell['elapsed_seconds']:.1f}s)"
                )
            # Write per-cell file
            cell_path = cells_dir / f"{benchmark}__{config}__seed{args.seed}.json"
            cell_payload = {
                "_manifest": build_manifest(seed=args.seed, extra={
                    "experiment": "stage3_orchestrator",
                    "mode": args.mode,
                    "model": args.model,
                    "benchmark": benchmark,
                    "config": config,
                    "k": args.k,
                    "n_questions": args.n_questions,
                }),
                **cell,
            }
            cell_path.write_text(json.dumps(cell_payload, indent=2, default=str))
            cells.append(cell)

    elapsed = time.time() - t_total

    # Aggregate cross-tab + cost projection
    print()
    print("=" * 78)
    print("  ORCHESTRATOR SUMMARY")
    print("=" * 78)
    total_projected = sum(c.get("projected_cost_usd", 0.0) for c in cells if c.get("ok"))
    total_actual = sum(c.get("actual_cost_usd") or 0.0 for c in cells if c.get("ok"))
    total_prompt_tokens = sum(c.get("total_prompt_tokens", 0) for c in cells if c.get("ok"))
    print(f"  total cells: {len(cells)}  elapsed: {elapsed:.1f}s")
    print(f"  total prompt tokens (est): {total_prompt_tokens:,}")
    print(f"  projected cost (gpt-4o-mini, full mode): ${total_projected:.4f}")
    if args.mode == "full":
        print(f"  ACTUAL cost: ${total_actual:.4f}")
        judge_stats = get_judge_stats()
        print(f"  judge cost: ${judge_stats['total_judge_cost_usd']:.4f}")

    # Save the cross-tab summary
    summary = {
        "_manifest": build_manifest(seed=args.seed, extra={
            "experiment": "stage3_orchestrator_summary",
            "mode": args.mode,
            "model": args.model,
            "benchmarks": args.benchmarks,
            "configs": args.configs,
            "n_questions": args.n_questions,
            "k": args.k,
        }),
        "config": {
            "mode": args.mode,
            "model": args.model,
            "benchmarks": args.benchmarks,
            "configs": args.configs,
            "n_questions": args.n_questions,
            "k": args.k,
        },
        "total_projected_cost_usd": total_projected,
        "total_actual_cost_usd": total_actual if args.mode == "full" else None,
        "total_prompt_tokens": total_prompt_tokens,
        "elapsed_s": elapsed,
        "cells": [
            {
                "benchmark": c["benchmark"], "config": c["config"],
                "mean_recall_at_k": c.get("mean_recall_at_k"),
                "mean_judge_score": c.get("mean_judge_score"),
                "total_prompt_tokens": c.get("total_prompt_tokens"),
                "projected_cost_usd": c.get("projected_cost_usd"),
                "actual_cost_usd": c.get("actual_cost_usd"),
            }
            for c in cells if c.get("ok")
        ],
    }
    summary_path = out_dir / "stage3_runs.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  -> {summary_path}")

    # If dry-run, also emit a dedicated cost_projection.json for convenience
    if args.mode == "dry-run":
        cost_path = out_dir / "cost_projection.json"
        cost_path.write_text(json.dumps({
            "_manifest": summary["_manifest"],
            "model": args.model,
            "n_questions": args.n_questions,
            "n_benchmarks": len(args.benchmarks),
            "n_configs": len(args.configs),
            "total_projected_cost_usd": total_projected,
            "total_prompt_tokens": total_prompt_tokens,
            "per_cell": summary["cells"],
            "pricing": PRICING[args.model],
        }, indent=2, default=str))
        print(f"  -> {cost_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
