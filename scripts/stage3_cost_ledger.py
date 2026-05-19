"""
Stage 3 cost ledger + data-inventory tracker.

Scans every Phase-4 artifact under `results/stage3/` and prints:
  1. Running cost total (across all tiers, all seeds, all sweeps).
  2. Per-cell data inventory (which {benchmark, config, seed} cells exist).
  3. What's missing (so we know if a partial rerun is needed).
  4. Data layout map (file paths for the thesis appendix).

Run this anytime to get a status snapshot — does NOT spend API.

Usage:
    python scripts/stage3_cost_ledger.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE3_DIR = ROOT / "results" / "stage3"
CELLS_DIR = STAGE3_DIR / "cells"


def _scan_summaries() -> list[dict]:
    """Find every stage3_runs*.json variant and return their cost lines.

    Dedup logic: stage3_runs.json at root is TRANSIENT — overwritten by every
    orchestrator run. If any *named* archive (tier_a/, tier_b_runs_seed*, etc)
    has the same total_actual_cost_usd, drop the transient copy.
    """
    entries: list[dict] = []
    patterns = [
        "tier_a/stage3_runs_tier_a.json",
        "tier_b_runs_seed*.json",
        "k_sweep_k*_runs.json",
        "k_sweep.json",
    ]
    seen_costs = set()
    for pat in patterns:
        for path in STAGE3_DIR.rglob(pat) if "/" in pat else STAGE3_DIR.glob(pat):
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            cost = data.get("total_actual_cost_usd")
            if cost is not None:
                seen_costs.add(round(cost, 6))
            entries.append({
                "source": str(path.relative_to(STAGE3_DIR)),
                "experiment": data.get("config", {}).get("mode") or data.get("experiment"),
                "model": data.get("config", {}).get("model"),
                "seed": data.get("config", {}).get("seed"),
                "n_questions": data.get("config", {}).get("n_questions"),
                "benchmarks": data.get("config", {}).get("benchmarks"),
                "configs": data.get("config", {}).get("configs"),
                "k": data.get("config", {}).get("k"),
                "total_actual_cost_usd": data.get("total_actual_cost_usd"),
                "total_projected_cost_usd": data.get("total_projected_cost_usd"),
                "total_prompt_tokens": data.get("total_prompt_tokens"),
                "n_cells": len(data.get("cells", [])),
                "elapsed_s": data.get("elapsed_s") or data.get("elapsed_total_seconds"),
            })

    # Add stage3_runs.json (the transient current-orchestrator output) IF
    # its cost differs from any archived entry — otherwise it's a duplicate
    # of the latest archive snapshot.
    current = STAGE3_DIR / "stage3_runs.json"
    if current.exists():
        try:
            data = json.loads(current.read_text())
            cost = data.get("total_actual_cost_usd")
            cost_key = round(cost, 6) if cost is not None else None
            if cost_key is None or cost_key not in seen_costs:
                entries.append({
                    "source": "stage3_runs.json (LIVE)",
                    "experiment": data.get("config", {}).get("mode") or data.get("experiment"),
                    "model": data.get("config", {}).get("model"),
                    "seed": data.get("config", {}).get("seed"),
                    "n_questions": data.get("config", {}).get("n_questions"),
                    "benchmarks": data.get("config", {}).get("benchmarks"),
                    "configs": data.get("config", {}).get("configs"),
                    "k": data.get("config", {}).get("k"),
                    "total_actual_cost_usd": data.get("total_actual_cost_usd"),
                    "total_projected_cost_usd": data.get("total_projected_cost_usd"),
                    "total_prompt_tokens": data.get("total_prompt_tokens"),
                    "n_cells": len(data.get("cells", [])),
                    "elapsed_s": data.get("elapsed_s") or data.get("elapsed_total_seconds"),
                })
        except Exception:
            pass
    return entries


def _scan_cells() -> dict:
    """Inventory cells/{bench}__{cfg}__seed{seed}.json files."""
    inventory: dict[tuple[str, str], list[int]] = {}
    total_size = 0
    for path in CELLS_DIR.glob("*.json"):
        total_size += path.stat().st_size
        stem = path.stem
        try:
            parts = stem.split("__")
            bench, cfg, seed_part = parts[0], parts[1], parts[2]
            seed = int(seed_part.replace("seed", ""))
        except (ValueError, IndexError):
            continue
        inventory.setdefault((bench, cfg), []).append(seed)
    return {
        "n_files": sum(1 for _ in CELLS_DIR.glob("*.json")) if CELLS_DIR.exists() else 0,
        "total_size_bytes": total_size,
        "by_cell": {f"{b}__{c}": sorted(seeds) for (b, c), seeds in sorted(inventory.items())},
    }


def main() -> int:
    print()
    print("=" * 84)
    print("  STAGE 3 PHASE 4 — COST LEDGER + DATA INVENTORY")
    print("=" * 84)
    print()
    if not STAGE3_DIR.exists():
        print(f"  [WARN] {STAGE3_DIR} does not exist yet")
        return 1

    entries = _scan_summaries()
    if not entries:
        print("  No stage3_runs*.json files found yet — nothing to report.")
    else:
        print("  Cost ledger (chronological by file):")
        print(f"  {'source':<46} | {'cost USD':>10} | {'tokens':>10} | {'cells':>6} | {'seed':>5}")
        print("  " + "-" * 86)
        total_actual = 0.0
        total_tokens = 0
        for e in entries:
            cost = e.get("total_actual_cost_usd") or 0.0
            tokens = e.get("total_prompt_tokens") or 0
            seed = e.get("seed") if e.get("seed") is not None else ""
            total_actual += cost
            total_tokens += tokens
            print(f"  {e['source']:<46} | ${cost:>9.4f} | {tokens:>10,} | {e['n_cells']:>6} | {seed!s:>5}")
        print("  " + "-" * 86)
        print(f"  {'TOTAL':<46} | ${total_actual:>9.4f} | {total_tokens:>10,}")
        print(f"\n  (Total ACTUAL spend so far: ${total_actual:.4f})")
        if total_actual > 0:
            print(f"  (Per-question avg: ${total_actual / max(total_tokens, 1) * 1000:.5f} per 1k tokens)")

    print()
    print("  Per-cell data inventory:")
    inv = _scan_cells()
    print(f"  {inv['n_files']} cell files in {CELLS_DIR.relative_to(ROOT)}, "
          f"{inv['total_size_bytes']/1024:.1f} KB total")
    if inv["by_cell"]:
        print(f"\n  {'cell key':<40} | seeds available")
        print("  " + "-" * 70)
        for key, seeds in inv["by_cell"].items():
            print(f"  {key:<40} | {seeds}")

    # Check completeness: 6 benchmarks x 3 configs x N seeds for Tier B.
    print()
    print("  Tier B completeness check (6 benchmarks x 3 configs x 3 seeds = 54 cells):")
    expected_benchmarks = ["cuad", "financebench", "hotpotqa", "longmemeval", "narrativeqa", "qasper"]
    expected_configs = ["v4-canonical", "v4-tuned", "flat-50"]
    expected_seeds = [42, 7, 100]
    missing = []
    present = 0
    for b in expected_benchmarks:
        for c in expected_configs:
            seeds_have = inv["by_cell"].get(f"{b}__{c}", [])
            for s in expected_seeds:
                if s in seeds_have:
                    present += 1
                else:
                    missing.append((b, c, s))
    print(f"  Present: {present}/54  Missing: {len(missing)}")
    if missing and len(missing) <= 20:
        for b, c, s in missing[:20]:
            print(f"    - missing {b}__{c}__seed{s}")
    elif missing:
        print(f"    (showing first 10 of {len(missing)})")
        for b, c, s in missing[:10]:
            print(f"    - missing {b}__{c}__seed{s}")

    # Aggregate analyses run? Check phase4_summary.json + lambda_sweep.json + k_sweep.json
    print()
    print("  Aggregate analyses status:")
    for name in ["phase4_summary.json", "lambda_sweep.json", "k_sweep.json"]:
        path = STAGE3_DIR / name
        if path.exists():
            print(f"    [OK]   {name} ({path.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"    [TODO] {name}")

    # Data layout map for thesis appendix.
    print()
    print("  Canonical data layout (Stage 3 Phase 4):")
    layout = [
        ("results/stage3/cells/",                  "per-cell JSON: predicted, gold, judge, recall, retrieved_steps, tokens, cost"),
        ("results/stage3/stage3_runs.json",        "most recent orchestrator summary (overwritten per run)"),
        ("results/stage3/tier_a/",                 "Tier A 30q smoke (frozen)"),
        ("results/stage3/tier_b_runs_seed*.json",  "per-seed Tier B summaries (3 files)"),
        ("results/stage3/k_sweep_k*_runs.json",    "per-k Tier C k-sweep summaries"),
        ("results/stage3/k_sweep.json",            "k-sweep aggregate"),
        ("results/stage3/phase4_summary.json",     "headline aggregate (3-seed pooling, paired t-tests)"),
        ("results/stage3/lambda_sweep.json",       "lambda-sensitivity (post-hoc on cells)"),
        ("results/stage3/*.log",                   "stdout transcripts per run"),
        ("results/stage3/tuned_theta_*.json",      "Phase 1.5 CMA-ES outputs"),
        ("results/stage3/retrieval_*.json",        "Phase 1.5 retrieval-quality numbers (no LLM)"),
        ("web/public/data/stage3_*.json",          "frontend-consumable subsets"),
    ]
    for path, desc in layout:
        print(f"    {path:<45}  {desc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
