"""Generalized Phase 1.9 cross-benchmark aggregator (CUAD, LME, HQA, NQA, QASPER, FB).

Walks results/stage3/judge_queue/{bench}__{cfg}__{mode}__seed42/ for each
benchmark in turn and aggregates Claude-judge means per cell + the theta
contrast table for the 4-shift visualization.

Output: results/stage3/multi_corpus_summary.json with chapter-ready figures.
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUDGE_QUEUE = ROOT / "results" / "stage3" / "judge_queue"
CORPUS_TRACES = ROOT / "results" / "stage3" / "corpus_traces"
OUT = ROOT / "results" / "stage3" / "multi_corpus_summary.json"

BENCHMARKS = ["financebench", "qasper", "cuad", "longmemeval", "hotpotqa", "narrativeqa"]

CONFIGS = [
    {"key": "v4t-canonical", "label": "V4ₜ canonical (grid θ)", "family": "v4"},
    {"key": "v4t-tuned", "label": "V4ₜ per-doc tuned", "family": "v4"},
    {"key": "v4t-corpus-tuned", "label": "V4ₜ corpus-cumulative tuned", "family": "v4"},
    {"key": "bm25-corpus", "label": "BM25 sparse retrieval", "family": "baseline"},
    {"key": "attention-corpus-tuned", "label": "Attention memory (corpus-tuned)", "family": "baseline"},
    {"key": "dump-all", "label": "Dump-all (context-stuffing)", "family": "baseline"},
]

CANONICAL_THETA = {
    "theta_store": 0.293,
    "w_graph": 0.000,
    "w_embed": 1.079,
    "w_recency": 3.777,
}


def cell_mean(bench: str, cfg: str, mode: str) -> tuple[float | None, int]:
    """Return (mean, n) for a judge cell, or (None, 0) if missing."""
    path = JUDGE_QUEUE / f"{bench}__{cfg}__{mode}__seed42" / "results.jsonl"
    if not path.exists():
        return None, 0
    scores: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            scores.append(json.loads(line)["judge_score"])
        except (json.JSONDecodeError, KeyError):
            continue
    if not scores:
        return None, 0
    return statistics.mean(scores), len(scores)


def cell_recall(bench: str, cfg: str, mode: str) -> float | None:
    path = CORPUS_TRACES / f"{bench}__{cfg}" / "qa_summary.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d.get(mode, {}).get("mean_recall_at_k") if d.get(mode) else None
    except (json.JSONDecodeError, KeyError):
        return None


def get_corpus_theta(bench: str) -> dict | None:
    """Read tuned_theta_v4t_corpus_{bench}.json if available."""
    path = ROOT / "results" / "stage3" / f"tuned_theta_v4t_corpus_{bench}.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        p = d.get("tuned_params", {})
        return {
            "theta_store": p.get("theta_store"),
            "w_graph": p.get("w_graph"),
            "w_embed": p.get("w_embed"),
            "w_recency": p.get("w_recency"),
            "canonical_recall": d.get("canonical_recall"),
            "tuned_recall": d.get("tuned_recall"),
            "improvement": d.get("improvement"),
            "limit_docs": d.get("limit_docs"),
            "n_eval_questions": d.get("n_eval_questions"),
        }
    except (json.JSONDecodeError, KeyError):
        return None


def main() -> None:
    out: dict = {
        "schema_version": 1,
        "judge_model": "claude-opus-4.7-1m",
        "judge_protocol": "v1",
        "seed": 42,
        "canonical_theta": CANONICAL_THETA,
        "benchmarks": {},
        "four_shift_summary": {},
    }

    for bench in BENCHMARKS:
        bench_data: dict = {
            "configs": {},
            "corpus_tuned_theta": get_corpus_theta(bench),
        }
        for cfg_info in CONFIGS:
            cfg = cfg_info["key"]
            modes = {}
            for mode in ["online", "batch"]:
                mean, n = cell_mean(bench, cfg, mode)
                recall = cell_recall(bench, cfg, mode)
                if n > 0 or recall is not None:
                    modes[mode] = {"n": n, "claude_judge_mean": mean, "recall_at_k_8": recall}
            if modes:
                bench_data["configs"][cfg] = {
                    "label": cfg_info["label"],
                    "family": cfg_info["family"],
                    "modes": modes,
                }
        out["benchmarks"][bench] = bench_data

        # Four-shift summary per benchmark
        theta = bench_data["corpus_tuned_theta"]
        if theta:
            shifts = {}
            for param in ["w_recency", "w_embed", "theta_store", "w_graph"]:
                canonical = CANONICAL_THETA[param]
                tuned = theta.get(param)
                if tuned is not None:
                    delta = tuned - canonical
                    expected_dir = "down" if param in ["w_recency", "theta_store"] else "up"
                    actual_dir = "down" if delta < -0.001 else ("up" if delta > 0.001 else "stable")
                    shifts[param] = {
                        "canonical": canonical,
                        "corpus_tuned": tuned,
                        "delta": delta,
                        "expected": expected_dir,
                        "actual": actual_dir,
                        "replicated": (actual_dir == expected_dir),
                    }
            out["four_shift_summary"][bench] = {
                "shifts": shifts,
                "replicated_count": sum(1 for s in shifts.values() if s["replicated"]),
                "total_shifts": len(shifts),
                "recall_lift": theta.get("improvement"),
            }

    # Headline cross-benchmark table
    headline_table = {}
    for bench in BENCHMARKS:
        bd = out["benchmarks"].get(bench, {})
        configs = bd.get("configs", {})
        canon_batch = configs.get("v4t-canonical", {}).get("modes", {}).get("batch", {}).get("claude_judge_mean")
        corpus_batch = configs.get("v4t-corpus-tuned", {}).get("modes", {}).get("batch", {}).get("claude_judge_mean")
        lift = (corpus_batch - canon_batch) if (canon_batch is not None and corpus_batch is not None) else None
        fs = out["four_shift_summary"].get(bench, {})
        headline_table[bench] = {
            "v4t_canonical_batch": canon_batch,
            "v4t_corpus_tuned_batch": corpus_batch,
            "lift": lift,
            "four_shifts_replicated": f"{fs.get('replicated_count', 0)}/{fs.get('total_shifts', 4)}" if fs else "—",
            "recall_lift_from_tuning": fs.get("recall_lift") if fs else None,
        }
    out["headline_table"] = headline_table

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")
    print()
    print("=== Four-shift replication summary ===")
    print(f"{'Benchmark':<14} {'Recency↓':<12} {'Embed↑':<10} {'Store↓':<10} {'Graph↑':<10} {'Score':<10}")
    for bench in BENCHMARKS:
        fs = out["four_shift_summary"].get(bench, {})
        shifts = fs.get("shifts", {})
        marks = []
        for p in ["w_recency", "w_embed", "theta_store", "w_graph"]:
            s = shifts.get(p, {})
            if s.get("replicated"):
                marks.append(f"OK {s['corpus_tuned']:.3f}")
            elif p in shifts:
                marks.append(f"NO {s['corpus_tuned']:.3f}")
            else:
                marks.append("MISS")
        score = f"{fs.get('replicated_count', 0)}/{fs.get('total_shifts', 4)}"
        print(f"{bench:<14} {marks[0]:<12} {marks[1]:<10} {marks[2]:<10} {marks[3]:<10} {score:<10}")
    print()
    print("=== Headline V4-corpus-tuned vs V4-canonical batch judge lift ===")
    for bench, row in headline_table.items():
        c = row["v4t_canonical_batch"]
        ct = row["v4t_corpus_tuned_batch"]
        lift = row["lift"]
        c_str = f"{c:.3f}" if c is not None else "  —  "
        ct_str = f"{ct:.3f}" if ct is not None else "  —  "
        lift_str = f"{lift:+.3f}" if lift is not None else "   —   "
        recall = row["recall_lift_from_tuning"]
        recall_str = f"recall +{recall:.3f}" if recall is not None else "no tuning"
        print(f"  {bench:<14} canonical={c_str}  corpus={ct_str}  lift={lift_str}  ({recall_str})")


if __name__ == "__main__":
    main()
