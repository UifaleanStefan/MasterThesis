"""
Pairwise statistical significance — V4 vs every other system on MultiHopKeyDoor.

The 0.005 absolute reward gap between V4 (0.178) and EpisodicSemantic (0.173)
is plausibly within bootstrap CI overlap at n=200. This script answers the
question definitively by loading per-episode rewards from
``benchmark_results.json`` (now retained after Phase 1 / B1) and running a
paired t-test + bootstrap CI + Cohen's d for every pairing.

Output: ``results/pairwise_significance.json`` with one entry per (system_a,
system_b) tuple, including p-value, mean difference, Cohen's d, and a
plain-language conclusion.

Usage (PowerShell, from project root):
    python scripts/run_pairwise_significance.py
    python scripts/run_pairwise_significance.py --env MultiHop-KeyDoor --baseline GraphMemoryV4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.statistics import full_comparison
from results.manifest import build_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default="results/benchmark_results.json",
                        help="Source benchmark JSON (must contain per-episode `rewards`)")
    parser.add_argument("--output", type=str, default="results/pairwise_significance.json",
                        help="Destination JSON")
    parser.add_argument("--env", type=str, default="MultiHop-KeyDoor",
                        help="Environment to evaluate (key in benchmark JSON)")
    parser.add_argument("--baseline", type=str, default="GraphMemoryV4",
                        help="Reference system to pair every other system against")
    args = parser.parse_args(argv)

    src = Path(args.input)
    if not src.exists():
        print(f"ERROR: {src} not found. Run `python run_benchmark.py` first.")
        return 1
    data = json.loads(src.read_text())

    if args.env not in data:
        envs_available = [k for k in data.keys() if not k.startswith("_")]
        print(f"ERROR: env {args.env!r} not in input. Available: {envs_available}")
        return 1
    env_data = data[args.env]

    if args.baseline not in env_data:
        systems = list(env_data.keys())
        print(f"ERROR: baseline {args.baseline!r} not in env data. Systems: {systems}")
        return 1

    base_entry = env_data[args.baseline]
    base_rewards = base_entry.get("rewards")
    if not base_rewards or not isinstance(base_rewards, list):
        print(f"ERROR: baseline {args.baseline!r} lacks per-episode `rewards` array. "
              f"Re-run `python run_benchmark.py` after Phase 1 / B1.")
        return 1

    print(f"Pairwise significance: {args.baseline} vs N-1 systems on {args.env}")
    print(f"  baseline n_episodes = {len(base_rewards)}")
    print(f"  baseline mean       = {sum(base_rewards) / len(base_rewards):.4f}")

    pairwise: dict = {}
    skipped: list[str] = []
    for sys_name, entry in env_data.items():
        if sys_name == args.baseline:
            continue
        rewards = entry.get("rewards")
        if not rewards:
            skipped.append(sys_name)
            continue
        n_pair = min(len(base_rewards), len(rewards))
        comp = full_comparison(
            rewards[:n_pair], base_rewards[:n_pair],
            label_a=sys_name, label_b=args.baseline,
        )
        # Drop the per-episode arrays from the per-pair payload to keep JSON small;
        # they're already in the source benchmark JSON.
        comp[args.baseline].pop("values", None)
        comp[sys_name].pop("values", None)
        pairwise[sys_name] = comp
        ttest = comp["ttest"]
        d = comp["cohens_d"]
        marker = "**" if ttest["p_value"] < 0.01 else "*" if ttest["p_value"] < 0.05 else "  "
        print(f"  {marker} {args.baseline} vs {sys_name:<24} "
              f"diff={comp['improvement']:+.4f}  p={ttest['p_value']:.4f}  d={d['d']:+.3f} ({d['magnitude']})")

    if skipped:
        print(f"\n[skipped — no per-episode rewards stored] {skipped}")

    payload = {
        "_manifest": build_manifest(extra={
            "experiment": "pairwise_significance",
            "env": args.env,
            "baseline": args.baseline,
            "source": str(src),
        }),
        "env": args.env,
        "baseline": args.baseline,
        "pairwise": pairwise,
        "skipped": skipped,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved {len(pairwise)} pairings to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
