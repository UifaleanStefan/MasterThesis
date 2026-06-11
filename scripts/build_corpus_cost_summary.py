"""
Corpus-mode cost aggregation + lambda-sweep (critique remediation).

The thesis's contribution #2 frames Stage 3 as operationalizing
J = QA_score - lambda * cost_usd, but the corpus-mode sections report no
cost numbers at all: every corpus_traces/**/qa_summary.json carries
total_cost_usd (answerer spend, accumulated per-call from the OpenAI
usage object), yet none of it reaches the chapter. This script closes
that gap from existing files — no new LLM calls.

Outputs results/stage3/corpus_cost_summary.json with:
  * per-run cost rows (benchmark, config, protocol, n questions,
    total_cost_usd, cost_per_question)
  * per-benchmark and project-wide totals (the "~$5 total spend" claims
    in the abstract/section 7.1 cover only Phase 4 + 1.7; corpus mode
    adds ~$19 on top)
  * a lambda-sweep over the FinanceBench batch configs:
    J(lambda) = claude_judge_mean - lambda * cost_per_question,
    showing where cost-sensitivity reorders the config ranking
    (dump-all's per-question cost is ~100x selective retrieval's).

Archived `*-capped12` runs (the pre-fix dump-all artifacts) are listed
separately and excluded from totals-by-config tables.

Usage:
    python scripts/build_corpus_cost_summary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / "results" / "stage3" / "corpus_traces"
MULTI = ROOT / "results" / "stage3" / "multi_corpus_summary.json"
OUT = ROOT / "results" / "stage3" / "corpus_cost_summary.json"

LAMBDAS = [0.0, 1.0, 5.0, 10.0, 25.0, 50.0]


def main() -> int:
    rows: list[dict] = []
    archived: list[dict] = []

    for d in sorted(TRACES.iterdir()):
        summary_path = d / "qa_summary.json"
        if not summary_path.exists():
            continue
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        n_online = (s.get("online") or {}).get("n", 0) or 0
        n_batch = (s.get("batch") or {}).get("n", 0) or 0
        n_calib = (s.get("calibration") or {}).get("n", 0) or 0
        n_q = n_online + n_batch + n_calib
        cost = float(s.get("total_cost_usd") or 0.0)
        row = {
            "run": d.name,
            "benchmark": s.get("benchmark"),
            "config": s.get("config"),
            "protocol": s.get("protocol"),
            "n_questions": n_q,
            "total_cost_usd": round(cost, 4),
            "cost_per_question_usd": round(cost / n_q, 6) if n_q else None,
        }
        if "-capped12" in d.name:
            archived.append(row)
        else:
            rows.append(row)

    by_bench: dict[str, float] = {}
    for r in rows:
        by_bench[r["benchmark"]] = by_bench.get(r["benchmark"], 0.0) + r["total_cost_usd"]
    total = sum(r["total_cost_usd"] for r in rows)
    total_archived = sum(r["total_cost_usd"] for r in archived)

    # Lambda-sweep over FinanceBench batch configs (judge means from the
    # cross-benchmark summary; per-question cost approximated as the run's
    # pooled cost over all its questions — online and batch share a run).
    sweep = None
    if MULTI.exists():
        multi = json.loads(MULTI.read_text(encoding="utf-8"))
        fb = (multi.get("benchmarks") or {}).get("financebench", {}).get("configs", {})
        cost_by_cfg = {
            r["config"]: r["cost_per_question_usd"]
            for r in rows
            if r["benchmark"] == "financebench" and r["protocol"] == "online_batch"
        }
        entries = []
        for cfg, cdata in fb.items():
            batch = (cdata.get("modes") or {}).get("batch")
            cpq = cost_by_cfg.get(cfg)
            if not batch or cpq is None or batch.get("claude_judge_mean") is None:
                continue
            entries.append({
                "config": cfg,
                "judge_mean_batch": batch["claude_judge_mean"],
                "cost_per_question_usd": cpq,
                "J_at_lambda": {
                    str(lam): round(batch["claude_judge_mean"] - lam * cpq, 4)
                    for lam in LAMBDAS
                },
            })
        if entries:
            winners = {
                str(lam): max(entries,
                              key=lambda e: e["J_at_lambda"][str(lam)])["config"]
                for lam in LAMBDAS
            }
            sweep = {"lambdas": LAMBDAS, "configs": entries,
                     "winner_by_lambda": winners,
                     "note": "cost_per_question pooled over the run's online+batch "
                             "calls (per-mode split not recorded)."}

    report = {
        "runs": rows,
        "archived_capped12_runs": archived,
        "total_corpus_cost_usd": round(total, 2),
        "total_archived_cost_usd": round(total_archived, 2),
        "by_benchmark_usd": {k: round(v, 2) for k, v in sorted(by_bench.items())},
        "fb_batch_lambda_sweep": sweep,
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(f"[build_corpus_cost_summary] wrote {OUT}")
    print(f"  total corpus-mode spend: ${total:.2f} "
          f"(+ ${total_archived:.2f} archived capped12 runs)")
    for b, v in sorted(by_bench.items()):
        print(f"  {b:<14} ${v:.2f}")
    if sweep:
        print("  FB batch winner by lambda:",
              {k: v for k, v in sweep["winner_by_lambda"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
