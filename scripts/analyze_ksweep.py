"""
Tier C k-sweep analysis — computes Pareto frontier across retrieval budgets.

Reads:
  results/stage3/cells_k{4,8,16,32}/{benchmark}__{config}__seed42.json
  results/stage3/k_sweep_k{4,8,16,32}_runs.json   — orchestrator summaries

Filters to the configs Tier C actually ran (v4-canonical, v4-tuned).
Computes per-k, per-(benchmark, config): mean_recall, mean_judge,
mean_cost_per_question. Identifies the Pareto-optimal k per benchmark.

Output:
  results/stage3/ksweep_analysis.json      — full per-cell table
  web/public/data/stage3_ksweep.json       — frontend-consumable subset

Usage:
    python scripts/analyze_ksweep.py
    python scripts/analyze_ksweep.py --ks 4 8 16 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STAGE3_DIR = ROOT / "results" / "stage3"

# Only configs actually run in Tier C — filter out stale Tier B flat-50 snapshots.
TIER_C_CONFIGS = ["v4-canonical", "v4-tuned"]


def _load_cell(k: int, bench: str, cfg: str, seed: int) -> dict | None:
    """Read cells_k{k}/{bench}__{cfg}__seed{seed}.json if present."""
    path = STAGE3_DIR / f"cells_k{k}" / f"{bench}__{cfg}__seed{seed}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if not data.get("ok"):
        return None
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ks", nargs="+", type=int, default=[4, 8, 16, 32],
    )
    parser.add_argument(
        "--benchmarks", nargs="*",
        default=["cuad", "qasper", "hotpotqa", "longmemeval", "financebench", "narrativeqa"],
    )
    parser.add_argument(
        "--configs", nargs="*", default=TIER_C_CONFIGS,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", default="results/stage3/ksweep_analysis.json",
    )
    parser.add_argument(
        "--frontend-out", default="web/public/data/stage3_ksweep.json",
    )
    args = parser.parse_args()

    # Build cube: results[k][bench][cfg] = {recall, judge, cost_per_q, ...}
    results: dict[int, dict[str, dict[str, dict]]] = {}
    available_ks: list[int] = []

    for k in args.ks:
        cell_dir = STAGE3_DIR / f"cells_k{k}"
        if not cell_dir.exists():
            print(f"  [SKIP] cells_k{k}/ missing — k={k} not yet run")
            continue
        available_ks.append(k)
        results[k] = {}
        for bench in args.benchmarks:
            results[k][bench] = {}
            for cfg in args.configs:
                cell = _load_cell(k, bench, cfg, args.seed)
                if cell is None:
                    continue
                n_q = cell.get("n_questions") or 0
                cost = cell.get("actual_cost_usd") or 0.0
                results[k][bench][cfg] = {
                    "mean_judge": cell.get("mean_judge_score"),
                    "mean_recall": cell.get("mean_recall_at_k"),
                    "n_questions": n_q,
                    "n_with_gold": cell.get("n_with_gold"),
                    "total_prompt_tokens": cell.get("total_prompt_tokens"),
                    "actual_cost_usd": cost,
                    "cost_per_q": cost / n_q if n_q else 0.0,
                }

    # Pareto-optimal k per (benchmark, config): the k where each new k step's
    # marginal judge gain falls below 0.01 (1 percentage point). That's the
    # "elbow" of the cost-quality curve.
    pareto_k: dict[str, dict[str, int | None]] = {}
    for bench in args.benchmarks:
        pareto_k[bench] = {}
        for cfg in args.configs:
            per_k = []
            for k in available_ks:
                cell = results[k][bench].get(cfg)
                if cell is None or cell.get("mean_judge") is None:
                    continue
                per_k.append((k, cell["mean_judge"], cell["cost_per_q"]))
            per_k.sort()
            elbow = None
            for i in range(1, len(per_k)):
                k_prev, j_prev, _ = per_k[i - 1]
                k_cur, j_cur, _ = per_k[i]
                gain = j_cur - j_prev
                if gain < 0.01:  # diminishing returns
                    elbow = k_prev  # the k value BEFORE the diminishing return
                    break
            if elbow is None and per_k:
                elbow = per_k[-1][0]  # no diminishing return seen — pick max k
            pareto_k[bench][cfg] = elbow

    # Pretty-print
    print()
    print("=" * 100)
    print(f"  K-SWEEP PARETO ANALYSIS — seed={args.seed}, available k={available_ks}")
    print("=" * 100)
    print()
    for bench in args.benchmarks:
        print(f"  --- {bench} ---")
        print(f"  {'k':>4} | {'config':<14} | {'recall':>8} | {'judge':>8} | {'cost/q':>10}")
        for k in available_ks:
            for cfg in args.configs:
                cell = results[k][bench].get(cfg)
                if cell is None:
                    continue
                r = cell.get("mean_recall")
                j = cell.get("mean_judge")
                cpq = cell.get("cost_per_q") or 0.0
                r_str = f"{r:.3f}" if r is not None else "—"
                j_str = f"{j:.3f}" if j is not None else "—"
                print(f"  {k:>4} | {cfg:<14} | {r_str:>8} | {j_str:>8} | ${cpq:>9.5f}")
        for cfg in args.configs:
            elbow = pareto_k[bench].get(cfg)
            if elbow is not None:
                print(f"  Pareto elbow {cfg!r}: k={elbow}")
        print()

    payload = {
        "config": vars(args),
        "available_ks": available_ks,
        "results": results,
        "pareto_elbows": pareto_k,
    }
    out_path = ROOT / args.out
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  Saved: {out_path}")

    # Frontend subset — flatten for chart consumption.
    rows: list[dict] = []
    for k in available_ks:
        for bench in args.benchmarks:
            for cfg in args.configs:
                cell = results[k][bench].get(cfg)
                if cell is None:
                    continue
                rows.append({
                    "k": k, "benchmark": bench, "config": cfg,
                    "judge": cell.get("mean_judge"),
                    "recall": cell.get("mean_recall"),
                    "cost_per_q": cell.get("cost_per_q"),
                })
    frontend = {
        "available_ks": available_ks,
        "benchmarks": args.benchmarks,
        "configs": args.configs,
        "rows": rows,
        "pareto_elbows": pareto_k,
    }
    fp = ROOT / args.frontend_out
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(frontend, indent=2, default=str))
    print(f"  Frontend data: {fp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
