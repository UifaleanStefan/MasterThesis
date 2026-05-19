"""
Build the frontend data file `web/public/data/stage3_retrieval.json` from
`results/stage3/retrieval_summary.json` and `tuned_theta_*.json`.

Output schema (consumed by `web/src/sections/Stage3.tsx`):

    {
      "benchmarks": ["cuad", "financebench", ...],
      "systems": ["FlatWindow(50)", "GraphMemoryV4", ..., "V4-tuned"],
      "table": {
        system_name: {benchmark: mean_recall_or_null}
      },
      "tuned_vs_canonical": {
        benchmark: {
          "canonical_recall": float,
          "tuned_recall": float,
          "improvement": float,
        }
      },
      "manifest": { git_sha, embedding_backend, timestamp_utc, ... }
    }

The `V4-tuned-<benchmark>` rows from the retrieval summary are collapsed
into a single `V4-tuned` row keyed by benchmark (since each benchmark
has its own tuned theta — they're not comparable cross-benchmark).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = ROOT / "results" / "stage3" / "retrieval_summary.json"
TUNED_DIR = ROOT / "results" / "stage3"
OUT_PATH = ROOT / "web" / "public" / "data" / "stage3_retrieval.json"


def main() -> int:
    if not SUMMARY_PATH.exists():
        print(f"[ERROR] {SUMMARY_PATH} missing — run scripts/run_stage3_retrieval.py first")
        return 1
    data = json.loads(SUMMARY_PATH.read_text())
    summary = data["summary"]
    benchmarks = data["config"]["benchmarks"]

    # Collapse V4-tuned-<benchmark> rows into one "V4-tuned" row.
    v4_tuned_row: dict[str, float | None] = {b: None for b in benchmarks}
    base_systems: dict[str, dict] = {}
    for sys_name, row in summary.items():
        if sys_name.startswith("V4-tuned-"):
            bench = sys_name.removeprefix("V4-tuned-")
            if bench in benchmarks:
                # row may map every benchmark; we only want the one this theta was tuned for.
                v4_tuned_row[bench] = row.get(bench)
        else:
            base_systems[sys_name] = row
    base_systems["V4-tuned"] = v4_tuned_row

    # Build tuned-vs-canonical comparison from per-benchmark tuning JSONs.
    tuned_vs_canonical: dict[str, dict] = {}
    for benchmark in benchmarks:
        path = TUNED_DIR / f"tuned_theta_{benchmark}.json"
        if not path.exists():
            continue
        tj = json.loads(path.read_text())
        if tj.get("status") != "ok":
            continue
        tuned_vs_canonical[benchmark] = {
            "canonical_recall": tj.get("canonical_recall"),
            "tuned_recall": tj.get("tuned_recall"),
            "improvement": tj.get("improvement"),
            "n_gold_questions": tj.get("n_gold_questions"),
        }

    out = {
        "benchmarks": benchmarks,
        "systems": sorted(base_systems.keys()),
        "table": base_systems,
        "tuned_vs_canonical": tuned_vs_canonical,
        "manifest": data.get("_manifest", {}),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"[OK] wrote {OUT_PATH}")
    print(f"  benchmarks: {benchmarks}")
    print(f"  systems:    {sorted(base_systems.keys())}")
    print(f"  tuned-v-canonical: {list(tuned_vs_canonical.keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
