"""
Phase 3 of the Reflexion plan — same-env retry experiment.

Reflexion's standard setup is "same task, multiple tries, lessons accumulate
across tries". This runner implements that:

  * For each env_seed in --env-seeds, instantiate a NEW env layout.
  * For each try_idx in 0..--n-tries-1, run a single episode against that
    env. V6 variants persist memory across tries (lessons accumulate);
    V4-base resets memory each try.
  * Across all (env_seed, try_idx) cells we measure reward, then aggregate
    by try_idx to see if reward improves over tries.

Three variants:
  * V4-base       — fresh-memory-per-try V4 (control; no cross-try learning).
  * V6-w-lessons  — persistent V6, lessons recorded each try, base policy.
                    (Tests whether *retrieval* from lessons helps even
                    without explicit policy-injection.)
  * V6-Reflexion  — persistent V6 + ReflexionPolicy: lessons are injected
                    as synthetic past_events so the rule-based policy's
                    hint regex picks them up. Headline experiment.

Usage:
    python run_reflexion_ablation.py
    python run_reflexion_ablation.py --env-seeds 1000 1001 1002 --n-tries 5
    python run_reflexion_ablation.py --envs MultiHop-KeyDoor --w-lesson 2.0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _env_factory_for(name: str):
    """Return a callable(seed) -> env for the named environment."""
    if name == "MegaQuestRoom":
        from environment.mega_quest import MegaQuestRoom
        return lambda s: MegaQuestRoom(seed=s)
    if name == "MultiHop-KeyDoor":
        from environment.env import MultiHopKeyDoor
        return lambda s: MultiHopKeyDoor(seed=s)
    if name == "Goal-Room":
        from environment.env import GoalRoom
        return lambda s: GoalRoom(seed=s)
    if name == "HardKeyDoor":
        from environment.env import HardKeyDoor
        return lambda s: HardKeyDoor(seed=s)
    if name == "Key-Door":
        from environment.env import ToyEnvironment
        return lambda s: ToyEnvironment(seed=s)
    raise ValueError(f"Unknown env: {name!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-env-seeds", type=int, default=10,
                        help="Number of distinct env layouts to evaluate (default: 10)")
    parser.add_argument("--n-tries", type=int, default=5,
                        help="Tries per env layout (default: 5; more tries gives "
                             "Reflexion more episodes to learn from)")
    parser.add_argument("--envs", nargs="*",
                        default=["MultiHop-KeyDoor"],
                        help="Envs to evaluate on (default: MultiHop-KeyDoor)")
    parser.add_argument("--w-lesson", type=float, default=2.0,
                        help="V6 w_lesson for the with-lessons variants (default: 2.0)")
    parser.add_argument("--theta-lesson-decay", type=float, default=0.0,
                        help="V6 theta_lesson_decay (default: 0.0 — no decay within a run)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-seed-base", type=int, default=7000)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from evaluation.reflexion_eval import compare_retry_variants, summarize_by_try
    from evaluation.statistics import full_comparison
    from memory.graph_memory_v4 import MemoryParamsV4
    from memory.graph_memory_v6 import MemoryParamsV6
    from results.manifest import build_manifest

    # V4 best params (post-MiniLM, from Phase 3 / S1 era).
    v4_params = MemoryParamsV4(
        theta_store=0.348, theta_novel=0.442, theta_erich=0.312, theta_surprise=0.418,
        theta_entity=0.063, theta_temporal=0.732, theta_decay=0.718,
        w_graph=1.188, w_embed=1.313, w_recency=1.207,
        mode="learnable",
    )
    v6_params = MemoryParamsV6(
        theta_store=v4_params.theta_store, theta_novel=v4_params.theta_novel,
        theta_erich=v4_params.theta_erich, theta_surprise=v4_params.theta_surprise,
        theta_entity=v4_params.theta_entity, theta_temporal=v4_params.theta_temporal,
        theta_decay=v4_params.theta_decay,
        w_graph=v4_params.w_graph, w_embed=v4_params.w_embed,
        w_recency=v4_params.w_recency,
        w_lesson=args.w_lesson, theta_lesson_decay=args.theta_lesson_decay,
        mode="learnable",
    )

    env_seeds = list(range(args.env_seed_base, args.env_seed_base + args.n_env_seeds))

    print("=" * 78)
    print(f"  Reflexion retry experiment "
          f"({args.n_env_seeds} envs * {args.n_tries} tries * 3 variants * "
          f"{len(args.envs)} env-types)")
    print("=" * 78)
    print(f"  V6 w_lesson           = {args.w_lesson}")
    print(f"  V6 theta_lesson_decay = {args.theta_lesson_decay}")
    print(f"  env_seeds             = {env_seeds[:3]}{'...' if len(env_seeds) > 3 else ''}")

    t0 = time.time()
    by_env: dict[str, dict] = {}

    for env_name in args.envs:
        print(f"\n  -- Environment: {env_name} --")
        env_factory = _env_factory_for(env_name)
        variants = compare_retry_variants(
            env_factory=env_factory,
            v4_params=v4_params,
            v6_params=v6_params,
            env_seeds=env_seeds,
            n_tries=args.n_tries,
            k=args.k,
            policy_seed=args.seed,
            verbose=args.verbose,
        )

        # Aggregate by try_idx to see learning curves.
        print(f"\n  variant         try   mean_reward   std       n")
        per_try: dict[str, dict[int, dict]] = {}
        for vname, results in variants.items():
            per_try[vname] = summarize_by_try(results)
            for tidx, summary in per_try[vname].items():
                print(
                    f"  {vname:<14}  {tidx:2d}    {summary['mean_reward']:.4f}     "
                    f"{summary['std_reward']:.3f}     {summary['n']}"
                )
            print()

        # Overall mean (across all tries).
        print(f"  variant         overall_mean   final_try_mean")
        for vname, results in variants.items():
            all_r = [r.reward for r in results]
            final_r = [r.reward for r in results if r.try_idx == args.n_tries - 1]
            om = sum(all_r) / len(all_r) if all_r else 0.0
            fm = sum(final_r) / len(final_r) if final_r else 0.0
            print(f"  {vname:<14}  {om:.4f}         {fm:.4f}")

        # Pairwise significance on the FINAL-try rewards (most diagnostic for
        # whether learning over tries actually helped).
        print(f"\n  Pairwise on final-try rewards (n={args.n_env_seeds} env layouts):")
        baseline_final = [r.reward for r in variants["V4-base"]
                          if r.try_idx == args.n_tries - 1]
        pairwise: dict[str, dict] = {}
        for vname in ["V6-w-lessons", "V6-Reflexion"]:
            other_final = [r.reward for r in variants[vname]
                           if r.try_idx == args.n_tries - 1]
            comp = full_comparison(
                baseline_final, other_final,
                label_a="V4-base", label_b=vname,
            )
            sig = comp["ttest"]["p_value"] < 0.05
            marker = "**" if comp["ttest"]["p_value"] < 0.01 else "*" if sig else "  "
            print(
                f"  {marker} V4-base vs {vname:<14} "
                f"diff={comp['improvement']:+.4f}  "
                f"p={comp['ttest']['p_value']:.4f}  "
                f"d={comp['cohens_d']['d']:+.3f} ({comp['cohens_d']['magnitude']})"
            )
            comp["V4-base"].pop("values", None)
            comp[vname].pop("values", None)
            pairwise[vname] = comp

        by_env[env_name] = {
            "per_try": per_try,
            "pairwise_final_try": pairwise,
            "raw": {
                vname: [
                    {
                        "env_seed": r.env_seed,
                        "try_idx": r.try_idx,
                        "reward": r.reward,
                        "precision": r.precision,
                        "memory_size": r.memory_size,
                        "lesson_recorded": r.lesson_recorded,
                    }
                    for r in res
                ]
                for vname, res in variants.items()
            },
        }

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s")

    out = {
        "_manifest": build_manifest(seed=args.seed, extra={
            "experiment": "reflexion_retry",
            "n_env_seeds": args.n_env_seeds,
            "n_tries": args.n_tries,
            "envs": args.envs,
            "w_lesson": args.w_lesson,
            "theta_lesson_decay": args.theta_lesson_decay,
            "env_seed_base": args.env_seed_base,
        }),
        "experiment": "reflexion_retry",
        "config": {
            "n_env_seeds": args.n_env_seeds,
            "n_tries": args.n_tries,
            "envs": args.envs,
            "w_lesson": args.w_lesson,
            "theta_lesson_decay": args.theta_lesson_decay,
            "k": args.k,
            "seed": args.seed,
            "env_seed_base": args.env_seed_base,
        },
        "v6_params": vars(v6_params),
        "by_env": by_env,
        "elapsed_s": elapsed,
    }
    out_path = Path("results/reflexion_retry.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
