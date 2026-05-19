"""
Compare narrow vs wide CMA-ES tuning runs for each benchmark.

Reads:
  results/stage3/tuned_theta_<bench>.json        — narrow run (n_docs=8, n_generations=10)
  results/stage3/tuned_theta_wide_<bench>.json   — wide run  (n_docs=20, n_generations=30)

Reports a 2-row × N-benchmark table of canonical / narrow-tuned / wide-tuned
recall and the per-benchmark improvement. Writes a comparison summary to
`results/stage3/theta_width_comparison.json`.

When wide >= narrow by `--threshold` (default 0.02), the wide theta is
flagged as preferred. Phase-1.5 retrieval study + Phase-4 orchestrator
both consume `tuned_theta_<bench>.json` so this script does NOT rename
files automatically — the preferred-flag is advisory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TUNED_DIR = ROOT / "results" / "stage3"


def _load_theta_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("status") != "ok":
            return None
        return data
    except Exception as e:
        print(f"  [WARN] could not parse {path.name}: {e!r}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks", nargs="*", default=["qasper", "cuad"],
        help="Benchmarks to compare. Defaults to the two that had headroom.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.02,
        help="Minimum wide-vs-narrow lift to flag wide as preferred (default 0.02).",
    )
    args = parser.parse_args()

    results: dict[str, dict] = {}
    for bench in args.benchmarks:
        narrow = _load_theta_json(TUNED_DIR / f"tuned_theta_{bench}.json")
        wide = _load_theta_json(TUNED_DIR / f"tuned_theta_wide_{bench}.json")
        if narrow is None:
            print(f"  [SKIP] {bench}: narrow result missing (expected tuned_theta_{bench}.json)")
            results[bench] = {"status": "missing_narrow"}
            continue
        if wide is None:
            print(f"  [SKIP] {bench}: wide result missing (expected tuned_theta_wide_{bench}.json)")
            results[bench] = {"status": "missing_wide", "narrow_recall": narrow.get("tuned_recall")}
            continue

        canonical = narrow.get("canonical_recall")
        narrow_recall = narrow.get("tuned_recall")
        wide_recall = wide.get("tuned_recall")
        delta_narrow = narrow_recall - canonical if (canonical is not None and narrow_recall is not None) else None
        delta_wide = wide_recall - canonical if (canonical is not None and wide_recall is not None) else None
        wide_vs_narrow = wide_recall - narrow_recall if (narrow_recall is not None and wide_recall is not None) else None
        prefer_wide = (wide_vs_narrow is not None and wide_vs_narrow >= args.threshold)

        results[bench] = {
            "status": "ok",
            "canonical_recall": canonical,
            "narrow_recall": narrow_recall,
            "wide_recall": wide_recall,
            "delta_narrow_vs_canonical": delta_narrow,
            "delta_wide_vs_canonical": delta_wide,
            "delta_wide_vs_narrow": wide_vs_narrow,
            "prefer_wide": prefer_wide,
            "narrow_n_docs": narrow.get("n_docs"),
            "narrow_n_generations": narrow.get("n_generations"),
            "wide_n_docs": wide.get("n_docs"),
            "wide_n_generations": wide.get("n_generations"),
        }

    # Print table.
    print()
    print("=" * 88)
    print(f"  CMA-ES narrow vs wide comparison (threshold = {args.threshold:+.2f})")
    print("=" * 88)
    header = f"  {'benchmark':<14}  {'canonical':>10}  {'narrow':>10}  {'wide':>10}  {'+narrow':>10}  {'+wide':>10}  {'+w-n':>8}  prefer"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for bench, r in results.items():
        if r["status"] != "ok":
            print(f"  {bench:<14}  {r['status']}")
            continue
        prefer_marker = "WIDE" if r["prefer_wide"] else "narrow"
        print(
            f"  {bench:<14}  "
            f"{r['canonical_recall']:>10.4f}  "
            f"{r['narrow_recall']:>10.4f}  "
            f"{r['wide_recall']:>10.4f}  "
            f"{r['delta_narrow_vs_canonical']:>+10.4f}  "
            f"{r['delta_wide_vs_canonical']:>+10.4f}  "
            f"{r['delta_wide_vs_narrow']:>+8.4f}  {prefer_marker}"
        )

    out_path = TUNED_DIR / "theta_width_comparison.json"
    out_path.write_text(json.dumps({
        "threshold": args.threshold,
        "benchmarks": list(results.keys()),
        "results": results,
    }, indent=2, default=str))
    print(f"\n  Saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
