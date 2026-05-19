"""
Aggregate Phase-4 Stage-3 results across seeds — Stage 3 Phase 1.7 analysis.

Reads:
  results/stage3/cells/{benchmark}__{config}__seed{seed}.json     — per-cell detail
  results/stage3/tier_b_runs_seed{seed}.json                       — orchestrator summaries

Produces:
  results/stage3/phase4_summary.json — aggregate cross-tab with:
    - per-(benchmark, config) mean/std/CI for recall@k and judge_score
    - paired t-test of V4-tuned vs V4-canonical per benchmark
    - cost totals + per-cell breakdown
    - rank-order of configs per benchmark
  web/public/data/stage3_phase4.json — frontend-consumable headline table

Usage:
    python scripts/aggregate_stage3_results.py
    python scripts/aggregate_stage3_results.py --seeds 42 7 100
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `evaluation.statistics` resolves
STAGE3_DIR = ROOT / "results" / "stage3"
CELLS_DIR = STAGE3_DIR / "cells"


def _load_cell(benchmark: str, config: str, seed: int) -> dict | None:
    path = CELLS_DIR / f"{benchmark}__{config}__seed{seed}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  [WARN] could not parse {path.name}: {e!r}")
        return None


def _bootstrap_ci(values: list[float], n_boot: int = 1000, ci: float = 0.95, rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Bootstrap percentile CI for the mean of `values`."""
    if not values:
        return (0.0, 0.0)
    if len(values) < 2:
        return (float(values[0]), float(values[0]))
    rng = rng or np.random.default_rng(42)
    arr = np.asarray(values, dtype=np.float64)
    boots = np.empty(n_boot)
    n = len(arr)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boots[i] = sample.mean()
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return (lo, hi)


def _paired_ttest(diffs: list[float]) -> dict:
    """One-sample paired t-test on differences (H0: mean=0)."""
    if len(diffs) < 2:
        return {"t": None, "p_two_sided": None, "n": len(diffs)}
    arr = np.asarray(diffs, dtype=np.float64)
    mean_diff = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd < 1e-12:
        return {
            "t": float("inf") if mean_diff != 0 else 0.0,
            "p_two_sided": 0.0 if mean_diff != 0 else 1.0,
            "n": len(arr),
            "mean_diff": mean_diff,
        }
    t_stat = mean_diff / (sd / np.sqrt(len(arr)))
    # Approximate p-value via normal (n is small but this is a fast estimate).
    from math import erf, sqrt
    p_two_sided = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))
    return {
        "t": float(t_stat),
        "p_two_sided": float(p_two_sided),
        "n": len(arr),
        "mean_diff": mean_diff,
        "std_diff": sd,
    }


def aggregate_cells(
    benchmarks: list[str], configs: list[str], seeds: list[int]
) -> dict:
    """Build the aggregate cross-tab.

    Per (benchmark, config) cell:
      - per-seed: mean_recall, mean_judge, n_questions, prompt_tokens, actual_cost
      - across seeds: mean ± std, 95% bootstrap CI
    Per benchmark:
      - paired t-test of (V4-tuned − V4-canonical) judge scores across seeds × questions
    """
    summary: dict[str, dict[str, dict]] = {}
    for bench in benchmarks:
        summary[bench] = {}
        for cfg in configs:
            per_seed: list[dict] = []
            for seed in seeds:
                cell = _load_cell(bench, cfg, seed)
                if cell is None or not cell.get("ok"):
                    continue
                per_seed.append({
                    "seed": seed,
                    "mean_recall": cell.get("mean_recall_at_k"),
                    "mean_judge": cell.get("mean_judge_score"),
                    "n_questions": cell.get("n_questions"),
                    "n_with_gold": cell.get("n_with_gold"),
                    "prompt_tokens": cell.get("total_prompt_tokens"),
                    "actual_cost_usd": cell.get("actual_cost_usd"),
                    "elapsed_seconds": cell.get("elapsed_seconds"),
                    # Pull per-question judge for finer-grained CI / paired-test.
                    "per_q_judge": [q.get("judge_score") for q in cell.get("questions", [])
                                    if q.get("judge_score") is not None],
                    "per_q_recall": [q.get("recall_at_k") for q in cell.get("questions", [])
                                     if q.get("recall_at_k") is not None],
                })
            if not per_seed:
                summary[bench][cfg] = {"status": "missing"}
                continue

            # Per-seed mean judge scores; across-seed mean ± std.
            seed_means_judge = [s["mean_judge"] for s in per_seed if s["mean_judge"] is not None]
            seed_means_recall = [s["mean_recall"] for s in per_seed if s["mean_recall"] is not None]

            # Pool all per-question judge scores across seeds for tighter CI.
            pooled_judge = [v for s in per_seed for v in s["per_q_judge"]]
            pooled_recall = [v for s in per_seed for v in s["per_q_recall"]]

            judge_ci = _bootstrap_ci(pooled_judge) if pooled_judge else (None, None)
            recall_ci = _bootstrap_ci(pooled_recall) if pooled_recall else (None, None)

            summary[bench][cfg] = {
                "status": "ok",
                "n_seeds": len(per_seed),
                "n_questions_per_seed": per_seed[0]["n_questions"] if per_seed else None,
                "n_questions_pooled": len(pooled_judge),
                "mean_judge_across_seeds": float(np.mean(seed_means_judge)) if seed_means_judge else None,
                "std_judge_across_seeds": float(np.std(seed_means_judge, ddof=1)) if len(seed_means_judge) >= 2 else 0.0,
                "judge_95ci_pooled": list(judge_ci),
                "mean_recall_across_seeds": float(np.mean(seed_means_recall)) if seed_means_recall else None,
                "std_recall_across_seeds": float(np.std(seed_means_recall, ddof=1)) if len(seed_means_recall) >= 2 else 0.0,
                "recall_95ci_pooled": list(recall_ci),
                "total_cost_usd": sum(s["actual_cost_usd"] or 0.0 for s in per_seed),
                "total_prompt_tokens": sum(s["prompt_tokens"] or 0 for s in per_seed),
                "per_seed": [{k: v for k, v in s.items() if k not in ("per_q_judge", "per_q_recall")}
                             for s in per_seed],
            }

    # Per-benchmark paired tests + Cohen's d: V4-tuned vs V4-canonical on
    # pooled per-question judge scores. Now also runs Wilcoxon signed-rank
    # (more appropriate than t-test for the discrete bounded judge
    # distribution) and Holm-Bonferroni correction across benchmarks.
    from evaluation.statistics import (
        cluster_bootstrap_ci,
        cohens_d,
        holm_bonferroni,
        wilcoxon_signed_rank,
    )

    paired_tests: dict[str, dict] = {}
    raw_pvalues_ttest: list[tuple[str, float]] = []
    raw_pvalues_wilcoxon: list[tuple[str, float]] = []
    for bench in benchmarks:
        canonical_qs: list[float] = []
        tuned_qs: list[float] = []
        cluster_ids: list[int] = []  # doc_idx for cluster-bootstrap
        for seed in seeds:
            c_cell = _load_cell(bench, "v4-canonical", seed)
            t_cell = _load_cell(bench, "v4-tuned", seed)
            if c_cell is None or t_cell is None or not c_cell.get("ok") or not t_cell.get("ok"):
                continue
            c_qs = [(q.get("doc_idx"), q.get("judge_score")) for q in c_cell.get("questions", [])
                    if q.get("judge_score") is not None]
            t_qs = [(q.get("doc_idx"), q.get("judge_score")) for q in t_cell.get("questions", [])
                    if q.get("judge_score") is not None]
            # Pair by doc_idx (the same docs are pulled for both configs at the same seed).
            c_map = dict(c_qs)
            for doc_idx, t_score in t_qs:
                if doc_idx in c_map:
                    canonical_qs.append(c_map[doc_idx])
                    tuned_qs.append(t_score)
                    # Cluster ID = (seed, doc_idx) to avoid spurious cross-seed clustering.
                    cluster_ids.append((seed, doc_idx))
        diffs = [t - c for c, t in zip(canonical_qs, tuned_qs)]
        paired_t = _paired_ttest(diffs)
        wilcoxon = wilcoxon_signed_rank(canonical_qs, tuned_qs) if canonical_qs else None
        d_stat = cohens_d(canonical_qs, tuned_qs) if len(canonical_qs) >= 2 else None
        diff_ci = (
            cluster_bootstrap_ci(diffs, cluster_ids, n_resamples=1000, seed=42)
            if diffs and len(set(cluster_ids)) >= 2 else None
        )
        paired_tests[bench] = {
            "n_pairs": len(diffs),
            **paired_t,
            "wilcoxon": wilcoxon,
            "cohens_d": d_stat,
            "lift_cluster_ci": diff_ci,
            "canonical_mean": float(np.mean(canonical_qs)) if canonical_qs else None,
            "tuned_mean": float(np.mean(tuned_qs)) if tuned_qs else None,
            "lift": float(np.mean(diffs)) if diffs else None,
        }
        if paired_t.get("p_two_sided") is not None:
            raw_pvalues_ttest.append((bench, paired_t["p_two_sided"]))
        if wilcoxon and wilcoxon.get("p_two_sided") is not None:
            raw_pvalues_wilcoxon.append((bench, wilcoxon["p_two_sided"]))

    # Holm-Bonferroni correction across the benchmarks tested.
    if raw_pvalues_ttest:
        adjusted_t = holm_bonferroni([p for _, p in raw_pvalues_ttest], alpha=0.05)
        for (bench, _), adj in zip(raw_pvalues_ttest, adjusted_t):
            paired_tests[bench]["p_holm_t"] = adj["p_adjusted"]
            paired_tests[bench]["significant_holm_t"] = adj["significant"]
    if raw_pvalues_wilcoxon:
        adjusted_w = holm_bonferroni([p for _, p in raw_pvalues_wilcoxon], alpha=0.05)
        for (bench, _), adj in zip(raw_pvalues_wilcoxon, adjusted_w):
            paired_tests[bench]["p_holm_wilcoxon"] = adj["p_adjusted"]
            paired_tests[bench]["significant_holm_wilcoxon"] = adj["significant"]

    # Rank-order configs per benchmark by mean judge score.
    rankings: dict[str, list[tuple[str, float]]] = {}
    for bench in benchmarks:
        ranked = []
        for cfg in configs:
            d = summary[bench].get(cfg, {})
            if d.get("status") != "ok":
                continue
            v = d.get("mean_judge_across_seeds")
            if v is not None:
                ranked.append((cfg, v))
        ranked.sort(key=lambda x: -x[1])
        rankings[bench] = ranked

    total_cost = sum(d.get("total_cost_usd", 0.0)
                     for bench_d in summary.values() for d in bench_d.values()
                     if d.get("status") == "ok")

    return {
        "config": {"benchmarks": benchmarks, "configs": configs, "seeds": seeds},
        "summary": summary,
        "paired_ttests_judge": paired_tests,
        "rankings_by_judge": rankings,
        "total_cost_usd": total_cost,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 7, 100])
    parser.add_argument(
        "--benchmarks", nargs="*",
        default=["cuad", "financebench", "hotpotqa", "longmemeval", "narrativeqa", "qasper"],
    )
    parser.add_argument(
        "--configs", nargs="*",
        default=["v4-canonical", "v4-tuned", "flat-50"],
    )
    parser.add_argument(
        "--out", default="results/stage3/phase4_summary.json",
    )
    parser.add_argument(
        "--frontend-out", default="web/public/data/stage3_phase4.json",
    )
    args = parser.parse_args()

    result = aggregate_cells(args.benchmarks, args.configs, args.seeds)

    # Pretty-print headline table.
    print()
    print("=" * 100)
    print(f"  Phase 4 Stage 3 — Headline Aggregate (seeds: {args.seeds})")
    print("=" * 100)
    print()
    header = (
        f"  {'benchmark':<14} | {'config':<14} "
        f"| {'mean_recall':>11} | {'mean_judge':>11} | {'judge_95ci':>16} | {'n_seeds':>8} | {'n_q_pooled':>11}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for bench in args.benchmarks:
        for cfg in args.configs:
            d = result["summary"][bench].get(cfg, {})
            if d.get("status") != "ok":
                print(f"  {bench:<14} | {cfg:<14} | {'missing':>11}")
                continue
            mean_r = d["mean_recall_across_seeds"]
            mean_j = d["mean_judge_across_seeds"]
            lo, hi = d.get("judge_95ci_pooled", (None, None))
            ci_str = f"[{lo:.3f}, {hi:.3f}]" if lo is not None and hi is not None else "n/a"
            print(
                f"  {bench:<14} | {cfg:<14} "
                f"| {mean_r:>11.3f} | {mean_j:>11.3f} | {ci_str:>16} "
                f"| {d['n_seeds']:>8} | {d['n_questions_pooled']:>11}"
            )

    # Paired tests with Holm-Bonferroni correction + Wilcoxon + Cohen's d.
    print()
    print("=" * 116)
    print("  Paired tests: V4-tuned vs V4-canonical (per-benchmark, pooled across seeds + Holm-corrected)")
    print("=" * 116)
    print()
    print(f"  {'benchmark':<14} | {'n':>5} | {'tuned':>7} | {'canon':>7} | {'lift':>7} | "
          f"{'p_t':>7} | {'p_holm_t':>9} | {'p_wilcox':>9} | {'p_holm_w':>9} | {'d':>6}")
    print("  " + "-" * 110)
    for bench, t in result["paired_ttests_judge"].items():
        if t.get("n_pairs", 0) == 0:
            continue
        n = t["n_pairs"]
        tuned = t.get("tuned_mean", 0.0)
        canonical = t.get("canonical_mean", 0.0)
        lift = t.get("lift", 0.0)
        p_t = t.get("p_two_sided")
        p_holm_t = t.get("p_holm_t")
        wlc = t.get("wilcoxon") or {}
        p_w = wlc.get("p_two_sided")
        p_holm_w = t.get("p_holm_wilcoxon")
        d_block = t.get("cohens_d") or {}
        d_val = d_block.get("d")

        def fmt_p(v):
            return f"{v:7.4f}" if isinstance(v, (int, float)) else "    n/a"

        def sig_marker(v):
            if v is None: return " "
            if v < 0.001: return "***"
            if v < 0.01: return "**"
            if v < 0.05: return "*"
            return " "

        print(
            f"  {bench:<14} | {n:>5} | {tuned:>7.3f} | {canonical:>7.3f} | {lift:>+7.3f} | "
            f"{fmt_p(p_t)} | {fmt_p(p_holm_t)}{sig_marker(p_holm_t):<2}| {fmt_p(p_w)} | "
            f"{fmt_p(p_holm_w)}{sig_marker(p_holm_w):<2}| "
            f"{(d_val if isinstance(d_val, (int, float)) else 0):>+6.3f}"
        )

    print()
    print("  Significance markers apply to Holm-Bonferroni-CORRECTED p-values (the honest column).")
    print("  *** p_adj<0.001  ** p_adj<0.01  * p_adj<0.05")

    print(f"\n  Total cost (sum of all cells): ${result['total_cost_usd']:.4f}")

    # Save full result.
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n  Saved aggregate to: {out_path}")

    # Build the frontend-consumable version (smaller).
    frontend = {
        "benchmarks": args.benchmarks,
        "configs": args.configs,
        "seeds": args.seeds,
        "summary": {
            bench: {
                cfg: {
                    "mean_judge": result["summary"][bench].get(cfg, {}).get("mean_judge_across_seeds"),
                    "mean_recall": result["summary"][bench].get(cfg, {}).get("mean_recall_across_seeds"),
                    "judge_95ci": result["summary"][bench].get(cfg, {}).get("judge_95ci_pooled"),
                    "n_questions_pooled": result["summary"][bench].get(cfg, {}).get("n_questions_pooled"),
                }
                for cfg in args.configs
            }
            for bench in args.benchmarks
        },
        "paired_ttests": result["paired_ttests_judge"],
        "total_cost_usd": result["total_cost_usd"],
    }
    frontend_path = ROOT / args.frontend_out
    frontend_path.parent.mkdir(parents=True, exist_ok=True)
    frontend_path.write_text(json.dumps(frontend, indent=2, default=str))
    print(f"  Frontend data: {frontend_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
