"""
Full benchmark runner — 12 memory systems × 4 environments.
Outputs results to results/benchmark_results.json and prints a summary table.

Usage (PowerShell):
    python run_benchmark.py
    python run_benchmark.py RAGMemory          # skip RAGMemory (e.g. if sentence-transformers is broken)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from evaluation.benchmark import run_full_benchmark, print_benchmark_table, save_benchmark_results

# Learned thetas from Phase 7 ES (best found per environment)
LEARNED_THETAS = {
    "Key-Door":         (0.309, 0.188, 0.843),
    "Goal-Room":        (0.740, 0.652, 0.514),
    "MultiHop-KeyDoor": (0.956, 0.378, 1.000),
}

N_EPISODES = 50
# MegaQuestRoom is 1000 steps per episode; use fewer episodes to keep runtime acceptable
MEGAQUEST_EPISODES = 20


def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    skip_systems = list(argv)  # positional args are interpreted as systems to skip
    print("=" * 70)
    print("FULL BENCHMARK — Learnable Memory Construction")
    skip_str = ", ".join(skip_systems) if skip_systems else "none"
    print(f"  Systems: 12 | Environments: 4 | Episodes: {N_EPISODES} (MegaQuestRoom: {MEGAQUEST_EPISODES})")
    print(f"  Skipped systems: {skip_str}")
    print("=" * 70)

    results = run_full_benchmark(
        n_episodes=N_EPISODES,
        episodes_per_env={"MegaQuestRoom": MEGAQUEST_EPISODES},
        learned_thetas=LEARNED_THETAS,
        skip_systems=skip_systems,
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
