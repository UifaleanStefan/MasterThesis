"""Merge Phase 4 Claude judge scores back into the cell JSONs — ALL benchmarks.

Generalizes scripts/merge_p4_fb_claude_judge.py to cover all 6 Phase 4 benchmarks
(FinanceBench, CUAD, QASPER, HotpotQA, LongMemEval, NarrativeQA).

For each cell file at results/stage3/cells/{bench}__{cfg}__seed{N}.json:
  * Looks up Claude judge results at
    results/stage3/judge_queue/p4_main__{bench}__{cfg}__seed{N}/results.jsonl
    (or, for FinanceBench legacy cells, finbench_p4__{cfg}__seed{N}/results.jsonl
    — falls back automatically).
  * Joins by qid suffix (q000…q099) → question index.
  * Adds `claude_judge_score` + `claude_judge_rationale` per question.
  * Computes `mean_claude_judge_score`, `n_with_claude_judge`,
    `claude_minus_gpt4omini_delta` per cell.

Emits results/stage3/stage3_phase4_claude_summary.json with:
  * per_cell — every cell's gpt-4o-mini mean, Claude mean, delta, n
  * per_config_per_benchmark — Claude/gpt pooled by (benchmark, config)
  * per_benchmark — Claude/gpt pooled by benchmark
  * coverage — which cells had partial vs full Claude coverage

Run:
    python scripts/merge_p4_all_claude_judges.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
CELLS_DIR = REPO_ROOT / "results" / "stage3" / "cells"
QUEUE_ROOT = REPO_ROOT / "results" / "stage3" / "judge_queue"
SUMMARY_PATH = REPO_ROOT / "results" / "stage3" / "stage3_phase4_claude_summary.json"


def load_jsonl(p: Path) -> list[dict]:
    with p.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def find_queue_dir(bench: str, cfg: str, seed: int) -> Path | None:
    """Locate the judge_queue dir for a given (bench, cfg, seed).

    Try several historical naming conventions:
      1. p4_main__{bench}__{cfg}__seed{N}  (current standard)
      2. finbench_p4__{cfg}__seed{N}        (FB legacy)
      3. p4_tierb__{bench}__{cfg}__seed{N}  (held-out tier B)
    """
    candidates = [
        QUEUE_ROOT / f"p4_main__{bench}__{cfg}__seed{seed}",
        QUEUE_ROOT / f"finbench_p4__{cfg}__seed{seed}" if bench == "financebench" else None,
        QUEUE_ROOT / f"p4_tierb__{bench}__{cfg}__seed{seed}",
    ]
    for c in candidates:
        if c is not None and c.exists() and (c / "results.jsonl").exists():
            return c
    return None


def main() -> None:
    per_cell: dict[str, dict] = {}
    missing: list[str] = []
    partial: list[str] = []

    for cell_path in sorted(CELLS_DIR.glob("*__*__seed*.json")):
        stem = cell_path.stem  # {bench}__{cfg}__seed{N}
        parts = stem.split("__")
        if len(parts) != 3:
            continue
        bench, config, seed_str = parts
        seed = int(seed_str.replace("seed", ""))

        queue_dir = find_queue_dir(bench, config, seed)
        if queue_dir is None:
            missing.append(stem)
            continue

        results = load_jsonl(queue_dir / "results.jsonl")
        # Map qid suffix (qNNN) → (claude_score, rationale)
        claude_by_suffix: dict[str, tuple[float, str]] = {}
        for r in results:
            suffix = r["qid"].split("__")[-1]
            claude_by_suffix[suffix] = (r["judge_score"], r.get("rationale", ""))

        cell = json.loads(cell_path.read_text(encoding="utf-8"))
        questions = cell.get("questions", [])
        claude_scores: list[float] = []
        gpt_scores: list[float] = []
        for q_idx, q in enumerate(questions):
            suffix = f"q{q_idx:03d}"
            entry = claude_by_suffix.get(suffix)
            if entry is None:
                continue
            cscore, crat = entry
            q["claude_judge_score"] = cscore
            q["claude_judge_rationale"] = crat
            claude_scores.append(cscore)
            gpt_scores.append(q.get("judge_score", 0.0))

        if not claude_scores:
            missing.append(stem)
            continue
        if len(claude_scores) < len(questions):
            partial.append(stem)

        cell_mean_claude = mean(claude_scores) if claude_scores else 0.0
        cell_mean_gpt = mean(gpt_scores) if gpt_scores else 0.0
        cell["mean_claude_judge_score"] = cell_mean_claude
        cell["n_with_claude_judge"] = len(claude_scores)
        cell["claude_minus_gpt4omini_delta"] = cell_mean_claude - cell_mean_gpt
        cell["queue_dir_used"] = queue_dir.name
        cell_path.write_text(json.dumps(cell, indent=2, ensure_ascii=False), encoding="utf-8")

        per_cell[stem] = {
            "benchmark": bench,
            "config": config,
            "seed": seed,
            "queue_dir": queue_dir.name,
            "n": len(claude_scores),
            "n_total": len(questions),
            "mean_claude": round(cell_mean_claude, 4),
            "mean_gpt4omini": round(cell_mean_gpt, 4),
            "delta": round(cell_mean_claude - cell_mean_gpt, 4),
        }

    # Per (benchmark, config) aggregation — pool across seeds
    per_cfg_bench: dict[str, dict] = {}
    cells_by_bc: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for v in per_cell.values():
        cells_by_bc[(v["benchmark"], v["config"])].append(v)
    for (bench, cfg), cells in cells_by_bc.items():
        key = f"{bench}__{cfg}"
        per_cfg_bench[key] = {
            "benchmark": bench,
            "config": cfg,
            "n_cells": len(cells),
            "n_questions_pooled": sum(c["n"] for c in cells),
            "mean_claude_pooled": round(mean(c["mean_claude"] for c in cells), 4),
            "mean_gpt4omini_pooled": round(mean(c["mean_gpt4omini"] for c in cells), 4),
            "mean_delta": round(mean(c["delta"] for c in cells), 4),
            "seeds": sorted({c["seed"] for c in cells}),
        }

    # Per benchmark aggregation — pool across all configs + seeds
    per_bench: dict[str, dict] = {}
    cells_by_b: dict[str, list[dict]] = defaultdict(list)
    for v in per_cell.values():
        cells_by_b[v["benchmark"]].append(v)
    for bench, cells in cells_by_b.items():
        per_bench[bench] = {
            "n_cells": len(cells),
            "n_questions_pooled": sum(c["n"] for c in cells),
            "mean_claude_pooled": round(mean(c["mean_claude"] for c in cells), 4),
            "mean_gpt4omini_pooled": round(mean(c["mean_gpt4omini"] for c in cells), 4),
            "mean_delta": round(mean(c["delta"] for c in cells), 4),
        }

    summary = {
        "n_cells": len(per_cell),
        "n_questions_total": sum(c["n"] for c in per_cell.values()),
        "n_missing_cells": len(missing),
        "n_partial_cells": len(partial),
        "missing_cells": missing,
        "partial_cells": partial,
        "per_cell": per_cell,
        "per_config_per_benchmark": per_cfg_bench,
        "per_benchmark": per_bench,
        "method": (
            "Claude Opus 4.7 max manual judge per evaluation/claude_judge_protocol.md "
            "(5-point rubric, 5% numeric tolerance, refusals count against gold). "
            "Cross-vendor: judge=Anthropic, answerer=OpenAI gpt-4o-mini."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {SUMMARY_PATH.relative_to(REPO_ROOT)}")
    print()
    print(f"Total cells merged: {len(per_cell)}")
    print(f"Total Claude judgments: {sum(c['n'] for c in per_cell.values())}")
    print(f"Missing (no Claude data): {len(missing)}")
    print(f"Partial (< 100 judged): {len(partial)}")
    if missing:
        print(f"Missing cells: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    print()
    print("Per benchmark (sorted by Claude mean desc):")
    for bench, agg in sorted(per_bench.items(), key=lambda x: -x[1]["mean_claude_pooled"]):
        print(f"  {bench:14s} claude={agg['mean_claude_pooled']:.4f}  "
              f"gpt={agg['mean_gpt4omini_pooled']:.4f}  "
              f"delta={agg['mean_delta']:+.4f}  "
              f"(n_cells={agg['n_cells']}, n_q={agg['n_questions_pooled']})")
    print()
    print("Per (benchmark, config) — Claude mean (sorted desc within benchmark):")
    bench_order = sorted(per_bench.keys(), key=lambda b: -per_bench[b]["mean_claude_pooled"])
    for bench in bench_order:
        cfgs = sorted(
            [v for v in per_cfg_bench.values() if v["benchmark"] == bench],
            key=lambda x: -x["mean_claude_pooled"],
        )
        print(f"  {bench}:")
        for c in cfgs:
            print(f"    {c['config']:20s} claude={c['mean_claude_pooled']:.4f}  "
                  f"gpt={c['mean_gpt4omini_pooled']:.4f}  "
                  f"delta={c['mean_delta']:+.4f}  "
                  f"(n_cells={c['n_cells']}, seeds={c['seeds']})")


if __name__ == "__main__":
    main()
