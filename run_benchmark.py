"""
Full benchmark runner — 10 memory systems × 3 environments × 50 episodes.
Outputs results to results/benchmark_results.json and prints a summary table.
Run with: python run_benchmark.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from evaluation.benchmark import run_full_benchmark, print_benchmark_table, save_benchmark_results
from evaluation.statistics import bootstrap_ci

# Learned thetas from Phase 7 ES (best found per environment)
LEARNED_THETAS = {
    "Key-Door":         (0.309, 0.188, 0.843),
    "Goal-Room":        (0.740, 0.652, 0.514),
    "MultiHop-KeyDoor": (0.956, 0.378, 1.000),
}

N_EPISODES = 50


def main() -> None:
    print("=" * 70)
    print("FULL BENCHMARK — Learnable Memory Construction")
    print(f"  Systems: 10 | Environments: 3 | Episodes per cell: {N_EPISODES}")
    print("=" * 70)

    results = run_full_benchmark(
        n_episodes=N_EPISODES,
        learned_thetas=LEARNED_THETAS,
        verbose=True,
    )

    print_benchmark_table(results)

    # Save JSON
    out_path = Path("results") / "benchmark_results.json"
    out_path.parent.mkdir(exist_ok=True)
    save_benchmark_results(results, out_path)

    # Also print a plain compact summary for easy reading
    print("\n\n--- COMPACT SUMMARY (reward | precision) ---")
    envs = list(results.keys())
    systems = list(list(results.values())[0].keys())
    print(f"{'System':<22}", end="")
    for env in envs:
        print(f"  {env[:16]:<20}", end="")
    print()
    print("-" * (22 + 22 * len(envs)))
    for sys_name in systems:
        print(f"{sys_name:<22}", end="")
        for env in envs:
            res = results.get(env, {}).get(sys_name, {})
            if "error" in res:
                print(f"  {'ERR':<20}", end="")
            else:
                r = res.get("mean_reward", 0.0)
                p = res.get("retrieval_precision")
                prec = f"{p:.3f}" if p is not None else " N/A"
                print(f"  {r:.4f}|{prec:<13}", end="")
        print()

    # Per-env ranking
    print("\n\n--- RANKINGS BY ENVIRONMENT ---")
    for env_name, sys_results in results.items():
        ranked = sorted(
            [(s, r.get("mean_reward", -1)) for s, r in sys_results.items() if "error" not in r],
            key=lambda x: x[1], reverse=True
        )
        print(f"\n{env_name}:")
        for rank, (sname, reward) in enumerate(ranked, 1):
            res = sys_results[sname]
            prec = res.get("retrieval_precision")
            prec_str = f"{prec:.3f}" if prec is not None else " N/A"
            ci_lo = res.get("ci_lower", 0)
            ci_hi = res.get("ci_upper", 0)
            eff = res.get("efficiency", 0)
            print(f"  #{rank:2d}  {sname:<22}  reward={reward:.4f} [{ci_lo:.3f},{ci_hi:.3f}]"
                  f"  prec={prec_str}  eff={eff:.6f}")

    print("\n\nDone. Results saved to results/benchmark_results.json")


if __name__ == "__main__":
    main()
