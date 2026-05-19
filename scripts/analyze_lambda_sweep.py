"""
Lambda-sensitivity analysis — post-hoc on existing Phase-4 cells, no API.

The thesis Stage 3 chapter (§7.4 Future Work, §5.6 Pareto frontier) calls
for a joint objective `J(λ) = judge_score − λ × cost_per_question`. As λ
ranges from 0 (quality-only) to large values (cost-dominated), the
winning config shifts.

This script:
  1. Reads every `results/stage3/cells/{bench}__{cfg}__seed{seed}.json`.
  2. Computes per-cell (mean_judge, mean_cost_per_q).
  3. For a sweep of λ values, ranks configs per benchmark by J(λ).
  4. Outputs the "λ at which V4-tuned stops being the winner" — the
     elasticity quantification that thesis Section 5.6 needs.

Output:
  results/stage3/lambda_sweep.json
  web/public/data/stage3_lambda.json (frontend-consumable subset)

Usage:
    python scripts/analyze_lambda_sweep.py
    python scripts/analyze_lambda_sweep.py --lambdas 0 0.5 1 2 5 10 50 100 500 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STAGE3_DIR = ROOT / "results" / "stage3"
CELLS_DIR = STAGE3_DIR / "cells"


def _load_cell_means(bench: str, cfg: str, seeds: list[int]) -> dict | None:
    """Aggregate per-question judge + cost across the given seeds.

    Note: in `--mode full` runs, per-question `projected_cost_usd` is 0.
    Real costs live at the cell level (`actual_cost_usd` and
    `projected_cost_usd`). We compute per-question cost as
    `cell.actual_cost_usd / cell.n_questions`, which is the mean per-q.
    """
    pooled_judge: list[float] = []
    total_cost = 0.0
    total_n = 0
    n_seeds_loaded = 0
    for seed in seeds:
        path = CELLS_DIR / f"{bench}__{cfg}__seed{seed}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not data.get("ok"):
            continue
        for q in data.get("questions", []):
            j = q.get("judge_score")
            if j is not None:
                pooled_judge.append(j)
        cell_cost = data.get("actual_cost_usd") or data.get("projected_cost_usd") or 0.0
        cell_n = data.get("n_questions") or 0
        if cell_n > 0:
            total_cost += cell_cost
            total_n += cell_n
        n_seeds_loaded += 1
    if not pooled_judge:
        return None
    return {
        "mean_judge": float(np.mean(pooled_judge)),
        "mean_cost_per_q": float(total_cost / total_n) if total_n else 0.0,
        "n_questions": len(pooled_judge),
        "n_seeds": n_seeds_loaded,
        "total_cost_usd": total_cost,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lambdas", nargs="*", type=float,
        default=[0.0, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0],
        help="Lambda values for J = judge - lambda * cost_per_q.",
    )
    parser.add_argument(
        "--benchmarks", nargs="*",
        default=["cuad", "qasper", "hotpotqa", "longmemeval", "narrativeqa", "financebench"],
    )
    parser.add_argument(
        "--configs", nargs="*",
        default=["v4-canonical", "v4-tuned", "flat-50"],
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 7, 100])
    parser.add_argument(
        "--out", default="results/stage3/lambda_sweep.json",
    )
    parser.add_argument(
        "--frontend-out", default="web/public/data/stage3_lambda.json",
    )
    args = parser.parse_args()

    # 1. Load per-cell aggregates.
    cells: dict[str, dict[str, dict]] = {}
    for bench in args.benchmarks:
        cells[bench] = {}
        for cfg in args.configs:
            agg = _load_cell_means(bench, cfg, args.seeds)
            if agg is not None:
                cells[bench][cfg] = agg

    # 2. For each λ, rank configs per benchmark by J = judge - λ * cost_per_q.
    by_lambda: dict[str, dict[str, list]] = {}
    crossovers: dict[str, dict] = {}  # benchmark -> "lambda at which v4-tuned loses to flat-50" etc

    for lam in args.lambdas:
        by_lambda[str(lam)] = {}
        for bench in args.benchmarks:
            scores = []
            for cfg, agg in cells[bench].items():
                j = agg["mean_judge"]
                cpq = agg["mean_cost_per_q"]
                jv = j - lam * cpq
                scores.append({"config": cfg, "J": jv, "judge": j, "cost_per_q": cpq})
            scores.sort(key=lambda x: -x["J"])
            by_lambda[str(lam)][bench] = scores

    # 3. Detect crossovers: at what λ does the leader change for each benchmark?
    for bench in args.benchmarks:
        prev_leader = None
        crossover_lambdas = []
        for lam in sorted(args.lambdas):
            ranking = by_lambda[str(lam)][bench]
            if not ranking:
                continue
            leader = ranking[0]["config"]
            if prev_leader is not None and leader != prev_leader:
                crossover_lambdas.append({
                    "lambda": lam,
                    "leader_before": prev_leader,
                    "leader_after": leader,
                })
            prev_leader = leader
        crossovers[bench] = {
            "crossovers": crossover_lambdas,
            "leader_at_lambda_0": by_lambda[str(args.lambdas[0])][bench][0]["config"] if by_lambda[str(args.lambdas[0])][bench] else None,
            "leader_at_max_lambda": by_lambda[str(max(args.lambdas))][bench][0]["config"] if by_lambda[str(max(args.lambdas))][bench] else None,
        }

    payload = {
        "config": vars(args),
        "cells": cells,
        "by_lambda": by_lambda,
        "crossovers": crossovers,
    }

    # Pretty-print.
    print()
    print("=" * 100)
    print(f"  Lambda-sensitivity sweep — J(lambda) = judge - lambda * cost_per_q")
    print(f"  seeds: {args.seeds}  configs: {args.configs}")
    print("=" * 100)
    print()
    for bench in args.benchmarks:
        if not cells[bench]:
            continue
        print(f"  --- {bench} ---")
        print(f"  {'config':<14} | {'judge':>8} | {'cost/q':>10}")
        for cfg in args.configs:
            if cfg in cells[bench]:
                a = cells[bench][cfg]
                print(f"  {cfg:<14} | {a['mean_judge']:>8.3f} | ${a['mean_cost_per_q']:>9.5f}")
        cb = crossovers[bench]
        print(f"  leader@lambda=0: {cb['leader_at_lambda_0']!r}")
        print(f"  leader@lambda={max(args.lambdas)}: {cb['leader_at_max_lambda']!r}")
        if cb["crossovers"]:
            for x in cb["crossovers"]:
                print(f"  crossover at lambda={x['lambda']}: {x['leader_before']} -> {x['leader_after']}")
        else:
            print(f"  (no crossover in lambda sweep)")
        print()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  Saved: {out_path}")

    # Frontend subset.
    frontend = {
        "lambdas": args.lambdas,
        "benchmarks": args.benchmarks,
        "configs": args.configs,
        "cells": cells,
        "by_lambda": by_lambda,
        "crossovers": crossovers,
    }
    fp = ROOT / args.frontend_out
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(frontend, indent=2, default=str))
    print(f"  Frontend data: {fp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
