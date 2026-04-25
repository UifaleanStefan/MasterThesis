"""
Build web-ready data files from results/*.json into web/public/data/.

The frontend at web/ is a static SPA; it loads its data from JSON shipped
under public/data/. This script:

  1. Slims neural_controller_v2_results.json (12 MB -> ~10 KB) by dropping
     the per-generation `mean` arrays (5,674 floats each) — the site shows
     learning curves of best_fitness/sigma, not the full weight trajectory.
  2. Copies the other result JSONs verbatim — they're all under 50 KB.
  3. Writes a manifest.json summarising which embedding backend produced
     these numbers and when, so the site can show provenance.

Invoked automatically before `npm run build` via the Vite prebuild hook,
and manually for `npm run dev`.

Usage (from project root):
    python scripts/build_web_data.py
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
WEB_DATA_DIR = ROOT / "web" / "public" / "data"


# Source -> web filename mapping. Source files that don't exist are skipped
# silently — the frontend renders graceful fallbacks for missing data.
_DIRECT_COPIES: dict[str, str] = {
    "benchmark_results.json": "benchmark.json",
    "graphmemory_v4_cmaes_results.json": "v4_cmaes.json",
    "ablation_results.json": "ablation.json",
    "transfer_results.json": "transfer.json",
    "sensitivity_results.json": "sensitivity.json",
    "pairwise_significance.json": "pairwise.json",
    "document_qa_memory_results.json": "docqa.json",
    "multisession_results.json": "multisession.json",
    "w_graph_sweep_results.json": "w_graph_sweep.json",
}


def _slim_neural_v2(src: Path, dst: Path) -> None:
    """
    Drop the 5,674-float `mean` arrays from each generation history entry.
    Keeps generation, best_fitness, sigma — sufficient for plotting learning
    curves without shipping 12 MB of MLP weights.
    """
    raw: dict[str, Any] = json.loads(src.read_text())
    history = raw.get("training", {}).get("history", [])
    slim_history = []
    for h in history:
        slim_history.append({
            "generation": h.get("generation"),
            "best_fitness": h.get("best_fitness"),
            "sigma": h.get("sigma"),
        })
    if "training" in raw and isinstance(raw["training"], dict):
        raw["training"]["history"] = slim_history
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(raw, indent=2, default=str))


def _build_manifest(written: list[str]) -> dict[str, Any]:
    """Aggregate provenance from each source file's _manifest block."""
    backends: set[str] = set()
    git_shas: set[str] = set()
    timestamps: list[str] = []
    for src_name in written:
        src = RESULTS_DIR / src_name
        if not src.exists():
            continue
        try:
            data = json.loads(src.read_text())
        except Exception:
            continue
        m = data.get("_manifest") or {}
        if isinstance(m, dict):
            backend = m.get("embedding_backend")
            if backend:
                backends.add(backend)
            sha = m.get("git_sha")
            if sha:
                git_shas.add(sha)
            ts = m.get("timestamp_utc")
            if ts:
                timestamps.append(ts)
    return {
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedding_backends": sorted(backends) or ["unknown"],
        "git_shas": sorted(git_shas),
        "latest_result_timestamp_utc": max(timestamps) if timestamps else None,
        "files_present": [Path(name).name for name in written],
    }


def main() -> int:
    if not RESULTS_DIR.exists():
        print(f"ERROR: results dir not found at {RESULTS_DIR}", file=sys.stderr)
        return 1
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    skipped: list[str] = []

    # 1. Direct copies (small files).
    for src_name, dst_name in _DIRECT_COPIES.items():
        src = RESULTS_DIR / src_name
        dst = WEB_DATA_DIR / dst_name
        if not src.exists():
            skipped.append(src_name)
            continue
        shutil.copyfile(src, dst)
        written.append(src_name)
        size_kb = dst.stat().st_size / 1024
        print(f"  [copy] {src_name:<46} -> {dst_name:<22} ({size_kb:.1f} KB)")

    # 2. Slim NeuralV2 (the 12 MB outlier).
    nv2_src = RESULTS_DIR / "neural_controller_v2_results.json"
    nv2_dst = WEB_DATA_DIR / "neural_v2.json"
    if nv2_src.exists():
        before_kb = nv2_src.stat().st_size / 1024
        _slim_neural_v2(nv2_src, nv2_dst)
        after_kb = nv2_dst.stat().st_size / 1024
        written.append(nv2_src.name)
        print(f"  [slim] neural_controller_v2_results.json   -> neural_v2.json         "
              f"({before_kb:.1f} KB -> {after_kb:.1f} KB)")
    else:
        skipped.append(nv2_src.name)

    # 3. Build aggregated manifest.
    manifest = _build_manifest(written)
    (WEB_DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  [meta] manifest.json with backends={manifest['embedding_backends']}")

    if skipped:
        print(f"\n  [skipped — source missing] {skipped}")
    print(f"\n  [OK] wrote {len(written) + 1} files to {WEB_DATA_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
