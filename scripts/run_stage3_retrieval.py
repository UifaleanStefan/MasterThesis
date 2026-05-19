"""
Stage 3 Phase 1.5 workstream A — multi-document retrieval study.

Runs all memory systems x specified benchmarks x ``--n-docs`` docs each
through the recall@k path. NO LLM in the loop -> $0 cost.

Optionally consumes the per-benchmark tuned thetas produced by
`tuning.tune_v4_per_benchmark` so the table includes a `V4-tuned-<benchmark>`
row alongside the 12 default memory systems.

Output:
  * `results/stage3/retrieval_<benchmark>.json` — per-benchmark detail
    with per-doc means + manifest.
  * `results/stage3/retrieval_summary.json` — aggregate
    `{system: {benchmark: mean_recall}}` table (headline figure).

Usage:
    python scripts/run_stage3_retrieval.py --benchmarks all --n-docs 30
    python scripts/run_stage3_retrieval.py --benchmarks hotpotqa qasper --n-docs 15
    python scripts/run_stage3_retrieval.py --benchmarks all --load-tuned-thetas
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from environment.benchmarks import ADAPTERS
from evaluation.benchmark_memory_eval import (
    aggregate_across_benchmarks,
    run_benchmark_memory_eval,
)
from results.manifest import build_manifest


def _load_tuned_thetas(out_dir: Path) -> dict[str, list[float]]:
    """Read per-benchmark tuned thetas from `results/stage3/tuned_theta_*.json`."""
    tuned: dict[str, list[float]] = {}
    for path in out_dir.glob("tuned_theta_*.json"):
        try:
            data = json.loads(path.read_text())
            if data.get("status") != "ok":
                continue
            vec = data.get("tuned_theta_vec")
            name = data.get("name")
            if name and isinstance(vec, list) and len(vec) == 10:
                tuned[name] = vec
        except Exception as e:
            print(f"  [WARN] failed to load {path.name}: {e!r}")
    return tuned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks", nargs="*", default=sorted(ADAPTERS.keys()),
        help="Subset of benchmarks. Default: all 6.",
    )
    parser.add_argument(
        "--n-docs", type=int, default=30,
        help="Docs per benchmark (default 30).",
    )
    parser.add_argument("--k", type=int, default=8, help="Retrieval top-k.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-systems", nargs="*", default=None,
        help="Memory systems to skip (e.g., RAGMemory if slow).",
    )
    parser.add_argument(
        "--load-tuned-thetas", action="store_true",
        help="Load per-benchmark tuned thetas from tuned_theta_*.json files "
             "in --out-dir and add a V4-tuned-<benchmark> system per benchmark.",
    )
    parser.add_argument(
        "--out-dir", default="results/stage3",
        help="Output dir for per-benchmark JSONs and the summary.",
    )
    args = parser.parse_args()

    # Allow `--benchmarks all` as a shorthand for "every registered adapter".
    if len(args.benchmarks) == 1 and args.benchmarks[0].lower() == "all":
        args.benchmarks = sorted(ADAPTERS.keys())

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tuned_thetas: dict[str, list[float]] = {}
    if args.load_tuned_thetas:
        tuned_thetas = _load_tuned_thetas(out_dir)
        print(f"[tuned thetas] loaded for: {sorted(tuned_thetas.keys())}")

    print()
    print("=" * 78)
    print(f"  Stage 3 retrieval study (n_docs={args.n_docs}, k={args.k})")
    print(f"  benchmarks: {args.benchmarks}")
    print("=" * 78)

    per_benchmark: dict[str, dict] = {}
    t_total = time.time()
    for name in args.benchmarks:
        if name not in ADAPTERS:
            print(f"  [SKIP] unknown benchmark: {name!r}")
            continue
        print()
        print(f"---- {name} ----")
        try:
            result = run_benchmark_memory_eval(
                adapter_name=name,
                n_docs=args.n_docs,
                k=args.k,
                seed=args.seed,
                tuned_thetas=tuned_thetas,
                skip_systems=args.skip_systems,
                verbose=True,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [FAIL] {name}: {e!r}")
            continue
        per_benchmark[name] = result
        # Save per-benchmark detail
        per_path = out_dir / f"retrieval_{name}.json"
        payload = {
            "_manifest": build_manifest(seed=args.seed, extra={
                "experiment": "stage3_retrieval",
                "benchmark": name,
                "n_docs": args.n_docs,
                "k": args.k,
                "tuned_theta_loaded": name in tuned_thetas,
            }),
            "experiment": "stage3_retrieval",
            "benchmark": name,
            "n_docs": args.n_docs,
            "k": args.k,
            "systems": result,
        }
        per_path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"  -> saved {per_path}")

    elapsed = time.time() - t_total
    print()
    print("=" * 78)
    print(f"  RETRIEVAL STUDY SUMMARY (elapsed {elapsed:.1f}s)")
    print("=" * 78)

    summary = aggregate_across_benchmarks(per_benchmark)
    # Pretty-print summary table.
    header = f"  {'system':<22}  " + "  ".join(f"{n[:12]:>12}" for n in args.benchmarks)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for sys_name in sorted(summary.keys()):
        row = summary[sys_name]
        cells = []
        for b in args.benchmarks:
            v = row.get(b)
            cells.append(f"{v:>12.4f}" if isinstance(v, (int, float)) else f"{'  n/a':>12}")
        print(f"  {sys_name:<22}  " + "  ".join(cells))

    # Save aggregate summary
    summary_path = out_dir / "retrieval_summary.json"
    summary_payload = {
        "_manifest": build_manifest(seed=args.seed, extra={
            "experiment": "stage3_retrieval_summary",
            "benchmarks": args.benchmarks,
            "n_docs": args.n_docs,
            "k": args.k,
            "tuned_thetas_loaded": sorted(tuned_thetas.keys()),
        }),
        "experiment": "stage3_retrieval_summary",
        "config": {
            "benchmarks": args.benchmarks,
            "n_docs": args.n_docs,
            "k": args.k,
            "seed": args.seed,
        },
        "summary": summary,
        "elapsed_s": elapsed,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, default=str))
    print(f"\n  Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
