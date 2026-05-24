"""Merge Phase 4 FinanceBench Claude judge scores back into the cell JSONs.

Reads results/stage3/judge_queue/finbench_p4__{cfg}__seed{N}/results.jsonl,
joins by qid suffix to the corresponding cells/financebench__{cfg}__seed{N}.json,
adds `claude_judge_score` per question + `mean_claude_judge_score` aggregate.
Preserves original GPT-4o-mini `judge_score` field intact.

Also emits results/stage3/finbench_phase4_claude_summary.json with the
per-cell + per-config Claude means and the GPT-4o-mini vs Claude delta.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
CELLS_DIR = REPO_ROOT / "results" / "stage3" / "cells"
QUEUE_ROOT = REPO_ROOT / "results" / "stage3" / "judge_queue"
SUMMARY_PATH = REPO_ROOT / "results" / "stage3" / "finbench_phase4_claude_summary.json"


def load_jsonl(p: Path) -> list[dict]:
    with p.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    per_cell = {}
    for cell_path in sorted(CELLS_DIR.glob("financebench__*.json")):
        stem = cell_path.stem  # financebench__{cfg}__seed{N}
        parts = stem.split("__")
        if len(parts) != 3:
            continue
        config = parts[1]
        seed = int(parts[2].replace("seed", ""))

        results_path = QUEUE_ROOT / f"finbench_p4__{config}__seed{seed}" / "results.jsonl"
        if not results_path.exists():
            print(f"  [SKIP] {stem}: no results file")
            continue

        results = load_jsonl(results_path)
        # Map qid suffix (qNNN) → claude_score
        claude_by_suffix = {r["qid"].split("__")[-1]: (r["judge_score"], r.get("rationale", "")) for r in results}

        cell = json.loads(cell_path.read_text(encoding="utf-8"))
        questions = cell.get("questions", [])
        claude_scores = []
        gpt_scores = []
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

        cell_mean_claude = mean(claude_scores) if claude_scores else 0.0
        cell_mean_gpt = mean(gpt_scores) if gpt_scores else 0.0
        cell["mean_claude_judge_score"] = cell_mean_claude
        cell["n_with_claude_judge"] = len(claude_scores)
        cell["claude_minus_gpt4omini_delta"] = cell_mean_claude - cell_mean_gpt

        cell_path.write_text(json.dumps(cell, indent=2, ensure_ascii=False), encoding="utf-8")
        per_cell[stem] = {
            "config": config,
            "seed": seed,
            "n": len(claude_scores),
            "mean_claude": round(cell_mean_claude, 4),
            "mean_gpt4omini": round(cell_mean_gpt, 4),
            "delta": round(cell_mean_claude - cell_mean_gpt, 4),
        }
        print(f"  [OK] {stem}: claude={cell_mean_claude:.4f}  gpt={cell_mean_gpt:.4f}  delta={cell_mean_claude-cell_mean_gpt:+.4f}")

    # Per-config aggregation
    per_config = {}
    for cfg in {v["config"] for v in per_cell.values()}:
        cfg_cells = [v for v in per_cell.values() if v["config"] == cfg]
        per_config[cfg] = {
            "n_cells": len(cfg_cells),
            "n_questions_pooled": sum(c["n"] for c in cfg_cells),
            "mean_claude_pooled": round(mean(c["mean_claude"] for c in cfg_cells), 4),
            "mean_gpt4omini_pooled": round(mean(c["mean_gpt4omini"] for c in cfg_cells), 4),
            "mean_delta": round(mean(c["delta"] for c in cfg_cells), 4),
        }

    summary = {
        "n_cells": len(per_cell),
        "n_questions_total": sum(c["n"] for c in per_cell.values()),
        "per_cell": per_cell,
        "per_config": per_config,
        "method": "Claude Opus 4.7 max manual judge per evaluation/claude_judge_protocol.md (5-point rubric, 5% numeric tolerance, refusals count against gold). Cross-vendor: judge=Anthropic, answerer=OpenAI gpt-4o-mini.",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {SUMMARY_PATH.relative_to(REPO_ROOT)}")
    print()
    print("Per-config:")
    for cfg, agg in sorted(per_config.items(), key=lambda x: -x[1]["mean_claude_pooled"]):
        print(f"  {cfg:20s} claude={agg['mean_claude_pooled']:.4f}  gpt={agg['mean_gpt4omini_pooled']:.4f}  delta={agg['mean_delta']:+.4f}  (n={agg['n_questions_pooled']})")


if __name__ == "__main__":
    main()
