"""Phase 1.9 — Generalized benchmark corpus-QA aggregator.

Usage:
    python scripts/build_benchmark_corpus_data.py <bench>

where <bench> is one of: longmemeval, hotpotqa, narrativeqa, qasper, cuad

Produces web/public/data/stage3_{bench}_corpus.json consumed by the frontend.
Also writes results/stage3/{bench}_corpus_summary.json for chapter figures.

Designed for small benchmarks (LME, HQA, NQA) with 9-12 Protocol A cells
and 12 Protocol B cells. Works equally for larger ones (QASPER, CUAD) though
those have their own dedicated scripts with extra features.
"""
from __future__ import annotations
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUDGE_QUEUE = ROOT / "results" / "stage3" / "judge_queue"
CORPUS_TRACES = ROOT / "results" / "stage3" / "corpus_traces"
WEB_DATA = ROOT / "web" / "public" / "data"

BENCH_META = {
    "longmemeval": {
        "label": "LongMemEval",
        "description": "Multi-session dialogue QA (10 sessions)",
        "n_docs": 10,
        "configs_with_online": ["v4t-canonical", "v4t-tuned", "v4t-corpus-tuned"],
    },
    "hotpotqa": {
        "label": "HotpotQA",
        "description": "Multi-hop Wikipedia QA (10 articles)",
        "n_docs": 10,
        "configs_with_online": ["v4t-canonical", "v4t-tuned", "v4t-corpus-tuned"],
    },
    "narrativeqa": {
        "label": "NarrativeQA",
        "description": "Literary reading comprehension (1 work per corpus)",
        "n_docs": 10,
        "configs_with_online": ["v4t-canonical", "v4t-tuned", "v4t-corpus-tuned",
                                 "bm25-corpus", "attention-corpus-tuned", "dump-all"],
    },
    "qasper": {
        "label": "QASPER",
        "description": "NLP paper QA (30 papers)",
        "n_docs": 30,
        "configs_with_online": ["v4t-canonical", "v4t-tuned", "v4t-corpus-tuned"],
    },
    "cuad": {
        "label": "CUAD",
        "description": "Contract clause QA (10 contracts)",
        "n_docs": 10,
        "configs_with_online": ["v4t-canonical", "v4t-tuned", "v4t-corpus-tuned"],
    },
}

CONFIGS = [
    {"key": "v4t-canonical",        "label": "V4ₜ canonical (grid θ)",           "family": "v4"},
    {"key": "v4t-tuned",            "label": "V4ₜ per-doc tuned",                "family": "v4"},
    {"key": "v4t-corpus-tuned",     "label": "V4ₜ corpus-cumulative tuned",      "family": "v4"},
    {"key": "bm25-corpus",          "label": "BM25 sparse retrieval",            "family": "baseline"},
    {"key": "attention-corpus-tuned","label": "Attention memory (corpus-tuned)", "family": "baseline"},
    {"key": "dump-all",             "label": "Dump-all (context-stuffing)",      "family": "baseline"},
]

CANONICAL_THETA = {"theta_store": 0.293, "w_graph": 0.000, "w_embed": 1.079, "w_recency": 3.777}


def cell_scores(bench: str, cfg: str, mode: str) -> tuple[float | None, int, list[float]]:
    path = JUDGE_QUEUE / f"{bench}__{cfg}__{mode}__seed42" / "results.jsonl"
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


def cell_protocol_b(bench: str, cfg: str, mode: str) -> dict:
    """Return ack/ans split for Protocol B cells (calibration or batch_calib)."""
    path = JUDGE_QUEUE / f"{bench}__{cfg}__{mode}__seed42" / "results.jsonl"
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
            if rec.get("expected_behavior") == "answer":
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


def calibration_trajectory(bench: str, cfg: str) -> list[dict] | None:
    """Per-decile calibration trajectory (for Protocol B calibration phase)."""
    import re
    path = JUDGE_QUEUE / f"{bench}__{cfg}__calibration__seed42" / "results.jsonl"
    if not path.exists():
        return None
    by_docs: dict[int, list] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            docs_seen = rec.get("docs_seen")
            if docs_seen is None:
                qid = rec.get("qid", "")
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


def get_corpus_theta(bench: str) -> dict | None:
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
        }
    except (json.JSONDecodeError, KeyError):
        return None


def cell_recall(bench: str, cfg: str, mode: str) -> float | None:
    path = CORPUS_TRACES / f"{bench}__{cfg}" / "qa_summary.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d.get(mode, {}).get("mean_recall_at_k") if d.get(mode) else None
    except (json.JSONDecodeError, KeyError):
        return None


def main(bench: str) -> None:
    meta = BENCH_META[bench]
    online_cfgs = set(meta["configs_with_online"])

    # ---- Protocol A: judge means + recall ----
    judge_table: dict = {"online": {}, "batch": {}}
    recall_table: dict = {"online": {}, "batch": {}}
    configs_out: list = []

    for cfg_info in CONFIGS:
        cfg = cfg_info["key"]
        modes: list[str] = []
        if cfg in online_cfgs:
            modes.append("online")
        modes.append("batch")

        cfg_data: dict = {
            "key": cfg,
            "label": cfg_info["label"],
            "family": cfg_info["family"],
            "modes": {},
        }
        for mode in modes:
            mean, n, _ = cell_scores(bench, cfg, mode)
            recall = cell_recall(bench, cfg, mode)
            cfg_data["modes"][mode] = {
                "n": n,
                "claude_judge_mean": round(mean, 4) if mean is not None else None,
                "recall_at_k_8": round(recall, 4) if recall is not None else None,
            }
            if mean is not None:
                judge_table[mode][cfg] = round(mean, 4)
            if recall is not None:
                recall_table[mode][cfg] = round(recall, 4)
        configs_out.append(cfg_data)

    # ---- Protocol B: calibration + batch_calib ----
    protocol_b: dict = {}
    for cfg_info in CONFIGS:
        cfg = cfg_info["key"]
        calib = cell_protocol_b(bench, cfg, "calibration")
        batch_c = cell_protocol_b(bench, cfg, "batch_calib")
        if calib or batch_c:
            protocol_b[cfg] = {
                "calibration": calib,
                "batch_calib": batch_c,
                "trajectory": calibration_trajectory(bench, cfg),
            }

    # ---- Theta contrast ----
    corpus_theta = get_corpus_theta(bench)
    theta_contrast = {
        "canonical": CANONICAL_THETA,
        "corpus_tuned": corpus_theta,
        "four_shift_score": None,
    }
    if corpus_theta and all(v is not None for v in corpus_theta.values()):
        shifts = [
            corpus_theta["w_recency"] < CANONICAL_THETA["w_recency"],  # ↓
            corpus_theta["w_embed"] > CANONICAL_THETA["w_embed"],      # ↑
            corpus_theta["theta_store"] < CANONICAL_THETA["theta_store"],  # ↓
            corpus_theta["w_graph"] > CANONICAL_THETA["w_graph"],      # ↑
        ]
        theta_contrast["four_shift_score"] = f"{sum(shifts)}/4"
    elif corpus_theta:
        theta_contrast["four_shift_score"] = "tuning failed"

    # ---- Headline summary ----
    canonical_batch = judge_table.get("batch", {}).get("v4t-canonical")
    ct_batch = judge_table.get("batch", {}).get("v4t-corpus-tuned")
    lift = round(ct_batch - canonical_batch, 4) if (canonical_batch is not None and ct_batch is not None) else None

    summary = {
        "_manifest": {
            "benchmark": bench,
            "label": meta["label"],
            "description": meta["description"],
            "n_docs": meta["n_docs"],
            "total_judgments": sum(
                cfg_data["modes"].get(m, {}).get("n", 0)
                for cfg_data in configs_out
                for m in cfg_data["modes"]
            ),
            "judge_model": "claude-opus-4.7-1m",
        },
        "configs": configs_out,
        "modes": list({m for cfg_data in configs_out for m in cfg_data["modes"]}),
        "judge_table": judge_table,
        "recall_table": recall_table,
        "calibration_data": protocol_b,
        "theta_contrast": theta_contrast,
        "headline": {
            "v4t_canonical_batch": canonical_batch,
            "v4t_corpus_tuned_batch": ct_batch,
            "lift": lift,
            "four_shifts_replicated": theta_contrast.get("four_shift_score"),
        },
    }

    # Write web frontend data
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    out_web = WEB_DATA / f"stage3_{bench}_corpus.json"
    out_web.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_web} ({out_web.stat().st_size:,} bytes)")

    # Also write chapter-facing summary
    out_results = ROOT / "results" / "stage3" / f"{bench}_corpus_summary.json"
    out_results.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_results} ({out_results.stat().st_size:,} bytes)")

    # Print quick chapter-ready table
    print(f"\n=== {meta['label']} Protocol A batch judge means ===")
    for cfg, mean in sorted(judge_table.get("batch", {}).items(), key=lambda x: -x[1]):
        print(f"  {cfg:<35} {mean:.4f}")
    if judge_table.get("online"):
        print(f"\n=== {meta['label']} Protocol A online judge means ===")
        for cfg, mean in sorted(judge_table.get("online", {}).items(), key=lambda x: -x[1]):
            print(f"  {cfg:<35} {mean:.4f}")
    if protocol_b:
        print(f"\n=== {meta['label']} Protocol B batch_calib means ===")
        for cfg, d in sorted(protocol_b.items(), key=lambda x: -(x[1].get("batch_calib", {}).get("overall_mean") or 0)):
            bc = d.get("batch_calib", {})
            if bc:
                print(f"  {cfg:<35} {bc.get('overall_mean', 0):.4f}")
    if lift is not None:
        print(f"\nLift v4t-corpus-tuned vs canonical (batch): {lift:+.4f}")
        print(f"Four-shift score: {theta_contrast.get('four_shift_score')}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_benchmark_corpus_data.py <bench>")
        print("  bench: one of", list(BENCH_META.keys()))
        sys.exit(1)
    bench = sys.argv[1].lower()
    if bench not in BENCH_META:
        print(f"Unknown benchmark: {bench!r}. Choices: {list(BENCH_META.keys())}")
        sys.exit(1)
    main(bench)
