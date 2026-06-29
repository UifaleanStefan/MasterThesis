"""
Multi-seed aggregation for Stage-3 judged cells (audit A4-C).

The thesis asserts corpus-mode evaluation is "near-deterministic at temperature 0,
so multi-seed replication would tighten CIs rather than move point estimates."
That was an assertion; this script measures it. It discovers all judged cells
named {benchmark}__{config}__{mode}__seed{N}, groups by (benchmark, config, mode),
and for every group reports per-seed mean judge score, the cross-seed mean/std/
range, and a pooled bootstrap CI over all questions. Cross-seed std is the
quantity that backs (or refutes) the determinism claim.

Pure aggregation over committed results.jsonl — no LLM, no judging here.

Output: results/stage3/multiseed_summary.json

Usage:
    python scripts/aggregate_multiseed.py            # all groups
    python scripts/aggregate_multiseed.py --min-seeds 2   # only replicated groups
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.statistics import bootstrap_ci

QUEUE = ROOT / "results" / "stage3" / "judge_queue"
# {benchmark}__{config}__{mode}__seed{N}; benchmark/config may contain hyphens,
# mode is one of the known modes, seed is trailing.
_CELL = re.compile(r"^(?P<bench>[a-z0-9]+)__(?P<config>.+)__(?P<mode>online|batch|batch_calib|calibration|scaling)__seed(?P<seed>\d+)$")


def _scores(path: Path) -> list[float]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            j = json.loads(line)
            s = j.get("judge_score")
            if s is not None:
                out.append(float(s))
    return out


def summarize_group(seed_to_scores: dict[int, list[float]]) -> dict:
    """Given {seed: [judge_scores]}, return per-seed + cross-seed summary."""
    per_seed = {str(s): {"n": len(v), "mean": round(sum(v) / len(v), 4)}
                for s, v in sorted(seed_to_scores.items()) if v}
    seed_means = [d["mean"] for d in per_seed.values()]
    n_seeds = len(seed_means)
    cross_mean = round(sum(seed_means) / n_seeds, 4) if n_seeds else None
    if n_seeds >= 2:
        m = sum(seed_means) / n_seeds
        cross_std = round((sum((x - m) ** 2 for x in seed_means) / n_seeds) ** 0.5, 4)
        spread = round(max(seed_means) - min(seed_means), 4)
    else:
        cross_std = None
        spread = None
    pooled = [s for v in seed_to_scores.values() for s in v]
    ci = bootstrap_ci(pooled) if len(pooled) >= 5 else None
    return {
        "n_seeds": n_seeds,
        "seeds": sorted(str(s) for s in seed_to_scores),
        "per_seed": per_seed,
        "cross_seed_mean": cross_mean,
        "cross_seed_std": cross_std,
        "cross_seed_spread": spread,
        "pooled_n": len(pooled),
        "pooled_ci": ({"lower": round(ci["ci_lower"], 4),
                       "upper": round(ci["ci_upper"], 4)} if ci else None),
    }


def discover_groups() -> dict[tuple, dict[int, list[float]]]:
    groups: dict[tuple, dict[int, list[float]]] = {}
    for d in sorted(QUEUE.glob("*/")):
        m = _CELL.match(d.name)
        rj = d / "results.jsonl"
        if not m or not rj.exists():
            continue
        key = (m["bench"], m["config"], m["mode"])
        groups.setdefault(key, {})[int(m["seed"])] = _scores(rj)
    return groups


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-seeds", type=int, default=1,
                    help="only report groups with at least this many seeds")
    args = ap.parse_args()

    groups = discover_groups()
    out = {}
    for (bench, config, mode), s2s in sorted(groups.items()):
        summ = summarize_group(s2s)
        if summ["n_seeds"] >= args.min_seeds:
            out[f"{bench}__{config}__{mode}"] = summ

    replicated = {k: v for k, v in out.items() if v["n_seeds"] >= 2}
    path = ROOT / "results" / "stage3" / "multiseed_summary.json"
    path.write_text(json.dumps({
        "n_groups": len(out),
        "n_replicated": len(replicated),
        "max_cross_seed_std": (round(max(v["cross_seed_std"] for v in replicated.values()), 4)
                               if replicated else None),
        "groups": out,
    }, indent=2))
    print(f"groups: {len(out)}  replicated(>=2 seeds): {len(replicated)}")
    if replicated:
        print("Cross-seed std on replicated cells (low => near-deterministic):")
        for k, v in sorted(replicated.items(), key=lambda kv: -(kv[1]["cross_seed_std"] or 0)):
            print(f"  {k:<48} seeds={v['seeds']} mean={v['cross_seed_mean']} std={v['cross_seed_std']}")
    else:
        print("No multi-seed cells yet (all single-seed). Re-run after seed 7/100 cells land.")
    print(f"-> saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
