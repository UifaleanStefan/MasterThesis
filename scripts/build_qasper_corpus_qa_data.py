"""Phase 1.9 — QASPER Aggregator (parallel to build_finbench_corpus_qa_data.py).

Walks results/stage3/judge_queue/qasper__{cfg}__{mode}__seed42/ and aggregates
Claude-judge means per cell. Loads tuned theta files for the four-shift table.
Emits results/stage3/qasper_corpus_summary.json with chapter-ready figures.

Handles both Protocol A (online/batch) and Protocol B (calibration/batch_calib).
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUDGE_QUEUE = ROOT / "results" / "stage3" / "judge_queue"
CORPUS_TRACES = ROOT / "results" / "stage3" / "corpus_traces"
OUT = ROOT / "results" / "stage3" / "qasper_corpus_summary.json"

CONFIGS = [
    {"key": "v4t-canonical", "label": "V4ₜ canonical (grid θ)", "available_modes": ["online", "batch"], "family": "v4"},
    {"key": "v4t-tuned", "label": "V4ₜ per-doc tuned", "available_modes": ["online", "batch"], "family": "v4"},
    {"key": "v4t-corpus-tuned", "label": "V4ₜ corpus-cumulative tuned", "available_modes": ["online", "batch"], "family": "v4"},
    {"key": "bm25-corpus", "label": "BM25 sparse retrieval", "available_modes": ["batch"], "family": "baseline"},
    {"key": "attention-corpus-tuned", "label": "Attention memory (corpus-tuned)", "available_modes": ["batch"], "family": "baseline"},
    {"key": "dump-all", "label": "Dump-all (context-stuffing)", "available_modes": ["batch"], "family": "baseline"},
]


def cell_mean(cfg: str, mode: str) -> tuple[float | None, int, list[float]]:
    """Return (mean, n, scores) for a cell, or (None, 0, []) if missing."""
    path = JUDGE_QUEUE / f"qasper__{cfg}__{mode}__seed42" / "results.jsonl"
    if not path.exists():
        return None, 0, []
    scores: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            scores.append(json.loads(line)["judge_score"])
        except (json.JSONDecodeError, KeyError):
            continue
    if not scores:
        return None, 0, []
    return statistics.mean(scores), len(scores), scores


def cell_mean_with_behavior(cfg: str, mode: str) -> dict:
    """For Protocol B cells: return mean split by expected_behavior.

    Returns dict with keys: overall_mean, n, answer_mean, answer_n,
    acknowledge_missing_mean, acknowledge_missing_n.
    """
    path = JUDGE_QUEUE / f"qasper__{cfg}__{mode}__seed42" / "results.jsonl"
    if not path.exists():
        return {}
    answer_scores: list[float] = []
    ack_scores: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            score = rec["judge_score"]
            behavior = rec.get("expected_behavior", "answer")
            if behavior == "answer":
                answer_scores.append(score)
            else:
                ack_scores.append(score)
        except (json.JSONDecodeError, KeyError):
            continue
    all_scores = answer_scores + ack_scores
    if not all_scores:
        return {}
    return {
        "overall_mean": round(statistics.mean(all_scores), 4),
        "n": len(all_scores),
        "answer_mean": round(statistics.mean(answer_scores), 4) if answer_scores else None,
        "answer_n": len(answer_scores),
        "acknowledge_missing_mean": round(statistics.mean(ack_scores), 4) if ack_scores else None,
        "acknowledge_missing_n": len(ack_scores),
    }


def calibration_trajectory(cfg: str) -> list[dict] | None:
    """Build the per-decile calibration trajectory for one config.

    Returns list of {docs_seen, n, mean_score, answer_n, answer_mean,
    ack_n, ack_mean} sorted by docs_seen, or None if no results yet.
    """
    path = JUDGE_QUEUE / f"qasper__{cfg}__calibration__seed42" / "results.jsonl"
    queue_path = JUDGE_QUEUE / f"qasper__{cfg}__calibration__seed42" / "queue.jsonl"
    if not path.exists():
        return None
    # Read the queue to get docs_seen metadata (in results.jsonl for Phase 1.9)
    by_docs: dict[int, list] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            docs_seen = rec.get("docs_seen")
            if docs_seen is None:
                # Fall back: parse from qid "afterN__seed42"
                qid = rec.get("qid", "")
                import re
                m = re.search(r"__after(\d+)__seed", qid)
                docs_seen = int(m.group(1)) + 1 if m else None
            if docs_seen is not None:
                by_docs.setdefault(docs_seen, []).append(rec)
        except (json.JSONDecodeError, KeyError):
            continue
    if not by_docs:
        return None
    trajectory = []
    for docs_seen in sorted(by_docs.keys()):
        entries = by_docs[docs_seen]
        scores = [e["judge_score"] for e in entries]
        ans_scores = [e["judge_score"] for e in entries if e.get("expected_behavior") == "answer"]
        ack_scores = [e["judge_score"] for e in entries if e.get("expected_behavior") != "answer"]
        trajectory.append({
            "docs_seen": docs_seen,
            "n": len(scores),
            "mean_score": round(statistics.mean(scores), 4),
            "answer_n": len(ans_scores),
            "answer_mean": round(statistics.mean(ans_scores), 4) if ans_scores else None,
            "acknowledge_missing_n": len(ack_scores),
            "acknowledge_missing_mean": round(statistics.mean(ack_scores), 4) if ack_scores else None,
        })
    return trajectory


def get_summary(cfg: str, mode: str) -> dict | None:
    """Get summary.json metrics for a cell."""
    path = CORPUS_TRACES / f"qasper__{cfg}" / "qa_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get(mode, None)


def get_theta_contrast() -> dict:
    """Build the 4-shift theta table for QASPER vs FB."""
    # FB tuned theta (from prior runs, hardcoded for chapter-ready reproducibility)
    fb_canonical = {"theta_store": 0.293, "w_graph": 0.000, "w_embed": 1.079, "w_recency": 3.777}
    fb_corpus = {"theta_store": 0.010, "w_graph": 1.627, "w_embed": 2.633, "w_recency": 0.003}

    # QASPER from existing tuning file
    qasper_path = ROOT / "results" / "stage3" / "tuned_theta_v4t_corpus_qasper.json"
    qasper_canonical = {"theta_store": 0.293, "w_graph": 0.000, "w_embed": 1.079, "w_recency": 3.777}  # same canonical baseline
    qasper_corpus = None
    if qasper_path.exists():
        d = json.loads(qasper_path.read_text(encoding="utf-8"))
        p = d["tuned_params"]
        qasper_corpus = {"theta_store": p["theta_store"], "w_graph": p["w_graph"],
                         "w_embed": p["w_embed"], "w_recency": p["w_recency"]}

    return {
        "four_shift": {
            "param_order": ["w_recency", "w_embed", "theta_store", "w_graph"],
            "fb": {"canonical": fb_canonical, "corpus_tuned": fb_corpus},
            "qasper": {"canonical": qasper_canonical, "corpus_tuned": qasper_corpus},
        },
        "expected_directions": {
            "w_recency": "decrease (collapses)",
            "w_embed": "increase",
            "theta_store": "decrease",
            "w_graph": "increase from 0",
        },
    }


def main() -> None:
    table: dict = {}
    judge_table: dict = {}
    recall_table: dict = {}

    # Protocol B per-config data
    protocol_b: dict = {}

    for cfg_info in CONFIGS:
        cfg = cfg_info["key"]
        modes = cfg_info["available_modes"]
        cell_data: dict = {"label": cfg_info["label"], "family": cfg_info["family"], "modes": {}}
        for mode in modes:
            mean, n, _ = cell_mean(cfg, mode)
            summary = get_summary(cfg, mode)
            recall = summary.get("mean_recall_at_k") if summary else None
            cell_data["modes"][mode] = {
                "n": n,
                "claude_judge_mean": mean,
                "recall_at_k_8": recall,
            }
        table[cfg] = cell_data

        # Flat tables for chapter rendering
        for mode in ["online", "batch"]:
            cell = cell_data["modes"].get(mode, {})
            judge_table.setdefault(mode, {})[cfg] = cell.get("claude_judge_mean")
            recall_table.setdefault(mode, {})[cfg] = cell.get("recall_at_k_8")

        # Protocol B — calibration trajectory + batch_calib
        calib_data = cell_mean_with_behavior(cfg, "calibration")
        batch_calib_data = cell_mean_with_behavior(cfg, "batch_calib")
        trajectory = calibration_trajectory(cfg)
        if calib_data or batch_calib_data:
            protocol_b[cfg] = {
                "calibration": calib_data or None,
                "batch_calib": batch_calib_data or None,
                "trajectory": trajectory,
            }

    theta = get_theta_contrast()

    # Compute key deltas
    deltas: dict = {}
    if (jb := judge_table.get("batch", {})).get("v4t-corpus-tuned") is not None:
        for cfg in ["v4t-canonical", "v4t-tuned", "bm25-corpus", "attention-corpus-tuned", "dump-all"]:
            if jb.get(cfg) is not None:
                deltas[f"v4t-corpus-tuned vs {cfg} (batch)"] = round(jb["v4t-corpus-tuned"] - jb[cfg], 4)

    # Protocol B end-of-corpus comparison (batch_calib means for available configs)
    protocol_b_end: dict = {}
    for cfg in ["v4t-canonical", "v4t-tuned", "v4t-corpus-tuned", "bm25-corpus", "attention-corpus-tuned"]:
        bc = protocol_b.get(cfg, {}).get("batch_calib")
        if bc:
            protocol_b_end[cfg] = bc.get("overall_mean")

    out = {
        "benchmark": "qasper",
        "n_docs_ingested": 281,
        "n_questions_per_cell": 94,
        "n_cells": sum(len(c["available_modes"]) for c in CONFIGS),
        "judge_model": "claude-opus-4.7-1m",
        "judge_protocol": "v1",
        "seed": 42,
        "config_table": table,
        "judge_table_flat": judge_table,
        "recall_at_k_8_flat": recall_table,
        "theta_contrast": theta,
        "v4_corpus_tuned_advantage": deltas,
        "protocol_b": {
            "configs": protocol_b,
            "end_of_corpus_batch_calib": protocol_b_end,
            "dump_all_excluded": True,
            "dump_all_exclusion_reason": (
                "MemoryError at doc 25/281: progressive calibration accumulates all stored events "
                "into context_lines; dump-all cannot scale to large corpora under Protocol B."
            ),
            "n_calibration_per_config": "281 doc-ends × 5 q/doc = ~1,405",
            "n_batch_calib_per_config": "~1,005 (full question pool)",
        },
        "fb_comparison": {
            "fb_v4t_corpus_tuned_batch": 0.665,
            "qasper_v4t_corpus_tuned_batch": judge_table.get("batch", {}).get("v4t-corpus-tuned"),
            "fb_v4t_canonical_batch": 0.450,
            "qasper_v4t_canonical_batch": judge_table.get("batch", {}).get("v4t-canonical"),
            "fb_lift_corpus_vs_canonical_batch": 0.215,
            "qasper_lift_corpus_vs_canonical_batch":
                round(judge_table["batch"]["v4t-corpus-tuned"] - judge_table["batch"]["v4t-canonical"], 4)
                if judge_table.get("batch", {}).get("v4t-corpus-tuned") is not None
                   and judge_table.get("batch", {}).get("v4t-canonical") is not None
                else None,
        },
        "notes": {
            "attention-corpus-tuned_batch":
                "Compromised by OpenAI API quota exhaustion (~59/94 fallback predictions). Real-answer-only mean: 0.421 on 35 entries.",
            "dump-all_batch":
                "61/94 entries are context-overflow fallbacks; real-answer-only mean: 0.106 on 33 entries. Identical to FB dump-all collapse (0.038 vs 0.037).",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")
    print(f"Protocol A — Cells: {out['n_cells']} | Q/cell: 94 | Total: {out['n_cells'] * 94}")
    print(f"Protocol B — Configs with calibration data: {len(protocol_b)}")
    if protocol_b_end:
        print(f"Protocol B end-of-corpus batch_calib means:")
        for cfg, mean in sorted(protocol_b_end.items(), key=lambda x: -(x[1] or 0)):
            print(f"  {cfg:>35} = {mean:.4f}" if mean is not None else f"  {cfg:>35} = None")
    print(f"V4-corpus-tuned batch advantage:")
    for k, v in deltas.items():
        print(f"  {k:>40} = {v:+.3f}")


if __name__ == "__main__":
    main()
