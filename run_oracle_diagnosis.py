"""
Phase 0 of the Reflexion plan — oracle diagnosis on MegaQuestRoom.

Question: when V4's memory works (precision 0.94+) but reward is 0.0, is the
bottleneck the policy?

This script runs three configurations on MegaQuestRoom and reports mean
reward + retrieval precision for each:

  1. ExplorationPolicy + V4 memory   (current setup; reward expected ~0.0)
  2. OraclePolicy     + V4 memory    (snake-sweep + plan; same memory)
  3. OraclePolicy     + FlatMemory   (no learnable memory at all — pure policy upper bound)

Decision gate per the plan:
  - Oracle reward >= 0.5 -> bottleneck is policy, proceed to Phase 1 (V6 lessons).
  - Oracle reward <  0.05 -> abort plan; investigate env mechanics or hint parsing.

Usage:
    python run_oracle_diagnosis.py
    python run_oracle_diagnosis.py --episodes 30 --seed 42
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Oracle diagnosis on MegaQuestRoom")
    parser.add_argument("--episodes", type=int, default=30,
                        help="Episodes per configuration (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for policy RNGs and episode seeds")
    parser.add_argument("--k", type=int, default=8, help="Retrieval top-k")
    args = parser.parse_args()

    # Imports here so failures surface in the runner's own traceback.
    from agent.loop import run_episode_with_any_memory
    from agent.policy import ExplorationPolicy
    from agent.oracle_policy import OraclePolicy, OmniscientOraclePolicy
    from environment.mega_quest import MegaQuestRoom
    from memory.flat_memory import FlatMemory
    from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4
    from results.manifest import build_manifest

    # Use the post-MiniLM V4 best params (matches what the rest of the project uses).
    v4_params = MemoryParamsV4(
        theta_store=0.348, theta_novel=0.442, theta_erich=0.312, theta_surprise=0.418,
        theta_entity=0.063, theta_temporal=0.732, theta_decay=0.718,
        w_graph=1.188, w_embed=1.313, w_recency=1.207,
        mode="learnable",
    )

    # Each entry: (name, policy_factory, memory_factory)
    # policy_factory takes (seed, env) and returns a policy instance —
    # `env` is consumed by OmniscientOraclePolicy and ignored by the others.
    configs = [
        ("Exploration   + V4",
         lambda s, env: ExplorationPolicy(seed=s),
         lambda: GraphMemoryV4(v4_params)),
        ("OracleHonest  + V4",
         lambda s, env: OraclePolicy(seed=s),
         lambda: GraphMemoryV4(v4_params)),
        ("OracleHonest  + FlatMemory",
         lambda s, env: OraclePolicy(seed=s),
         lambda: FlatMemory(window_size=50)),
        ("OracleOmni    + FlatMemory",
         lambda s, env: OmniscientOraclePolicy(env=env, seed=s),
         lambda: FlatMemory(window_size=50)),
    ]

    print("=" * 78)
    print(f"  Oracle diagnosis on MegaQuestRoom  ({args.episodes} eps × 3 configs)")
    print("=" * 78)

    t0 = time.time()
    results: dict[str, dict] = {}

    for name, policy_factory, mem_factory in configs:
        rewards: list[float] = []
        precisions: list[float] = []
        mem_sizes: list[float] = []
        steps_to_first_door: list[int] = []
        n_doors_opened: list[int] = []

        print(f"\n  [{name}]")
        for ep in range(args.episodes):
            ep_seed = 5000 + ep
            env = MegaQuestRoom(seed=ep_seed)
            env.reset()  # so OmniscientOracle can read internal state immediately
            policy = policy_factory(args.seed + ep, env)
            mem = mem_factory()

            _, _, stats = run_episode_with_any_memory(
                env, policy, mem, k=args.k, episode_seed=ep_seed
            )

            r = stats.get("reward", 0.0)
            p = stats.get("retrieval_precision")
            ms = stats.get("memory_size", 0)

            rewards.append(r)
            if p is not None:
                precisions.append(p)
            mem_sizes.append(ms)
            n_doors_opened.append(int(round(r * 6)))  # partial_score = doors / 6

        mean_r = statistics.mean(rewards)
        mean_p = statistics.mean(precisions) if precisions else None
        mean_doors = statistics.mean(n_doors_opened)

        results[name.strip()] = {
            "mean_reward": mean_r,
            "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
            "mean_precision": mean_p,
            "mean_memory_size": statistics.mean(mem_sizes),
            "mean_doors_opened": mean_doors,
            "rewards": rewards,
            "n_episodes": args.episodes,
        }

        print(f"    reward = {mean_r:.4f}  (std {results[name.strip()]['std_reward']:.3f})")
        print(f"    precision = {mean_p:.4f}" if mean_p is not None else "    precision = N/A")
        print(f"    mean doors opened = {mean_doors:.2f} / 6")
        print(f"    mem size (final) = {results[name.strip()]['mean_memory_size']:.1f}")

    elapsed = time.time() - t0

    # Decision gate
    omni_mean = results.get("OracleOmni    + FlatMemory", {}).get("mean_reward", 0.0)
    honest_mean = results.get("OracleHonest  + V4", {}).get("mean_reward", 0.0)
    print("\n" + "=" * 78)
    print("  DIAGNOSIS")
    print("=" * 78)
    if omni_mean >= 0.5:
        verdict = "POLICY-BOTTLENECK CONFIRMED (env is solvable)"
        action = (
            "OmniscientOracle proves the env can be solved with full info. "
            "Proceed to Phase 1 (V6 with Reflexion lessons + better navigation)."
        )
    elif omni_mean >= 0.1:
        verdict = "ENV PARTIALLY SOLVABLE"
        action = (
            f"OmniscientOracle gets {omni_mean:.2f} — env is partially solvable; "
            "policy+planning is the bottleneck but Reflexion alone may not close it. "
            "Consider also extending step budget or simplifying env."
        )
    else:
        verdict = "ENV TOO HARD AT 1000 STEPS"
        action = (
            f"Even the cheating OmniscientOracle gets {omni_mean:.2f}. "
            "1000 steps is insufficient for 6 keys × 6 doors of Manhattan paths in 20×20. "
            "Recommend redesigning MegaQuest (longer budget or smaller grid) before Reflexion."
        )

    print(f"  OracleHonest + V4 reward       = {honest_mean:.4f}")
    print(f"  OracleOmni  + FlatMemory reward = {omni_mean:.4f}  (env solvability upper bound)")
    print(f"  Verdict:  {verdict}")
    print(f"  Action:   {action}")
    print(f"  Elapsed:  {elapsed:.1f}s")

    # Persist
    out = {
        "_manifest": build_manifest(seed=args.seed, extra={
            "experiment": "oracle_diagnosis",
            "n_episodes_per_config": args.episodes,
            "env": "MegaQuestRoom",
            "verdict": verdict,
        }),
        "experiment": "oracle_diagnosis",
        "configs": list(results.keys()),
        "results": results,
        "elapsed_s": elapsed,
        "verdict": verdict,
        "action": action,
    }
    out_path = Path("results/oracle_diagnosis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
