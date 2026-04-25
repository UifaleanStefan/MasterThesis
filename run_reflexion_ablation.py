"""
Phase 2/5 of the Reflexion plan — 4-variant comparison runner.

Variants compared (paired episode seeds → paired t-tests downstream):
  * V4-base       — fresh-memory-per-episode V4 (control)
  * V6-no-lessons — fresh-memory-per-ep, w_lesson=0 (sanity check that V6
                    with the lesson channel disabled matches V4-base)
  * V6-w-lessons  — persistent V6 across eps, lessons recorded but not
                    injected into the policy via Reflexion wrapper
  * V6-Reflexion  — persistent V6 + ReflexionPolicy injecting lessons as
                    synthetic past_events for the rule-based policy to parse

Usage:
    python run_reflexion_ablation.py
    python run_reflexion_ablation.py --episodes 50 --envs MegaQuestRoom MultiHop-KeyDoor
    python run_reflexion_ablation.py --w-lesson 2.0 --theta-lesson-decay 0.0
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
    parser.add_argument("--episodes", type=int, default=30,
                        help="Episodes per variant per env (default: 30)")
    parser.add_argument("--envs", nargs="*",
                        default=["MegaQuestRoom", "MultiHop-KeyDoor"],
                        help="Envs to evaluate on (default: MegaQuestRoom + MultiHop)")
    parser.add_argument("--w-lesson", type=float, default=2.0,
                        help="V6 w_lesson for the with-lessons variants (default: 2.0)")
    parser.add_argument("--theta-lesson-decay", type=float, default=0.0,
                        help="V6 theta_lesson_decay (default: 0.0 — no decay)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-offset", type=int, default=6000)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from evaluation.reflexion_eval import compare_variants
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

    print("=" * 78)
    print(f"  Reflexion ablation  ({args.episodes} eps × 4 variants × {len(args.envs)} envs)")
    print("=" * 78)
    print(f"  V6 w_lesson           = {args.w_lesson}")
    print(f"  V6 theta_lesson_decay = {args.theta_lesson_decay}")
    print(f"  Seed offset           = {args.seed_offset}")

    t0 = time.time()
    by_env: dict[str, dict[str, dict]] = {}
    by_env_pairwise: dict[str, dict[str, dict]] = {}

    for env_name in args.envs:
        print(f"\n  -- Environment: {env_name} --")
        env_factory = _env_factory_for(env_name)
        results = compare_variants(
            env_factory=env_factory,
            v4_params=v4_params,
            v6_params=v6_params,
            n_episodes=args.episodes,
            k=args.k,
            seed_offset=args.seed_offset,
            policy_seed=args.seed,
            verbose=args.verbose,
        )
        # Print summary
        print(f"\n  variant         reward     std    precision  mem    n_lessons")
        for name in ["V4-base", "V6-no-lessons", "V6-w-lessons", "V6-Reflexion"]:
            r = results[name]
            prec = f"{r.mean_precision:.3f}" if r.mean_precision is not None else "  N/A"
            print(
                f"  {name:<14}  {r.mean_reward:.4f}  ±{r.std_reward:.3f}  "
                f"{prec}     {r.mean_mem_size:5.1f}  {r.n_lessons_final:5d}"
            )

        # Pairwise significance
        baseline = results["V4-base"].rewards
        pairwise: dict[str, dict] = {}
        for vname in ["V6-no-lessons", "V6-w-lessons", "V6-Reflexion"]:
            comp = full_comparison(
                baseline,
                results[vname].rewards,
                label_a="V4-base",
                label_b=vname,
            )
            sig = comp["ttest"]["p_value"] < 0.05
            marker = (
                "**" if comp["ttest"]["p_value"] < 0.01
                else "*" if sig else "  "
            )
            print(
                f"  {marker} V4-base vs {vname:<14} "
                f"diff={comp['improvement']:+.4f}  "
                f"p={comp['ttest']['p_value']:.4f}  "
                f"d={comp['cohens_d']['d']:+.3f} ({comp['cohens_d']['magnitude']})"
            )
            # Strip the per-episode arrays from the saved comp (they're in the
            # variant results above).
            comp[f"V4-base"].pop("values", None)
            comp[vname].pop("values", None)
            pairwise[vname] = comp

        by_env[env_name] = {name: r.to_dict() for name, r in results.items()}
        by_env_pairwise[env_name] = pairwise

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s")

    out = {
        "_manifest": build_manifest(seed=args.seed, extra={
            "experiment": "reflexion_ablation",
            "n_episodes_per_variant": args.episodes,
            "envs": args.envs,
            "w_lesson": args.w_lesson,
            "theta_lesson_decay": args.theta_lesson_decay,
            "seed_offset": args.seed_offset,
        }),
        "experiment": "reflexion_ablation",
        "config": {
            "n_episodes": args.episodes,
            "envs": args.envs,
            "w_lesson": args.w_lesson,
            "theta_lesson_decay": args.theta_lesson_decay,
            "k": args.k,
            "seed": args.seed,
            "seed_offset": args.seed_offset,
        },
        "v6_params": vars(v6_params),
        "by_env": by_env,
        "pairwise": by_env_pairwise,
        "elapsed_s": elapsed,
    }
    out_path = Path("results/reflexion_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
