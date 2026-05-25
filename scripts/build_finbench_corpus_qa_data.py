"""Aggregate FinanceBench corpus-mode QA results into a single frontend JSON.

Reads the 12 manual-judge queues + corpus traces + tuned-theta file + audit
JSON and emits ``web/public/data/stage3_finbench_corpus.json`` consumed by
``<FinanceBenchCorpus>`` in the frontend.

Output shape: judge_table, theta_contrast, cross_doc_scatter,
question_types, drilldown_examples, configs, modes, _manifest.

Run via:
    python scripts/build_finbench_corpus_qa_data.py
"""

from __future__ import annotations

import json
import re
import subprocess
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
JUDGE_QUEUE = REPO_ROOT / "results" / "stage3" / "judge_queue"
TRACES = REPO_ROOT / "results" / "stage3" / "corpus_traces"
OUT_PATH = REPO_ROOT / "web" / "public" / "data" / "stage3_finbench_corpus.json"

# Each config declares which queue modes are available + which queue mode supplies
# the "batch" column (Protocol A batch for original 6; Protocol B batch_calib for
# extension 4 that lack Protocol A runs but produce the same end-of-corpus 150q).
CONFIGS = [
    {"name": "v4t-corpus-tuned",       "label": "V4ₜ corpus-tuned",        "family": "v4-corpus", "color": "#22d3ee",
     "queue_for": {"online": "online", "batch": "batch"}, "has_protocol_a": True},
    {"name": "attention-corpus-tuned", "label": "Attention corpus-tuned",  "family": "attention", "color": "#f59e0b",
     "queue_for": {"online": "online", "batch": "batch"}, "has_protocol_a": True},
    {"name": "dump-all",               "label": "Dump-all (188 paras)",    "family": "dump-all",  "color": "#f43f5e",
     "queue_for": {"online": "online", "batch": "batch"}, "has_protocol_a": True},
    {"name": "bm25-corpus",            "label": "BM25-corpus",             "family": "bm25",      "color": "#64748b",
     "queue_for": {"online": "online", "batch": "batch"}, "has_protocol_a": True},
    {"name": "v4t-canonical",          "label": "V4ₜ canonical (grid)",    "family": "v4-grid",   "color": "#0891b2",
     "queue_for": {"online": "online", "batch": "batch"}, "has_protocol_a": True},
    {"name": "v4t-tuned",              "label": "V4ₜ per-doc tuned",       "family": "v4-perdoc", "color": "#06b6d4",
     "queue_for": {"online": "online", "batch": "batch"}, "has_protocol_a": True},
    # ----- Phase 1.9 extension: 4 additional baselines, calibration-only -----
    {"name": "rag-corpus",             "label": "RAG (MiniLM cosine)",     "family": "rag",       "color": "#10b981",
     "queue_for": {"batch": "batch_calib"}, "has_protocol_a": False},
    {"name": "flat-corpus",            "label": "FlatMemory (window=50)",  "family": "flat",      "color": "#fb7185",
     "queue_for": {"batch": "batch_calib"}, "has_protocol_a": False},
    {"name": "v5t-corpus",             "label": "V5 graph (canonical θ)",  "family": "v5",        "color": "#a78bfa",
     "queue_for": {"batch": "batch_calib"}, "has_protocol_a": False},
    {"name": "semantic-corpus",        "label": "Semantic (TF-IDF)",       "family": "tfidf",     "color": "#fbbf24",
     "queue_for": {"batch": "batch_calib"}, "has_protocol_a": False},
]
MODES = ["online", "batch"]
# Calibration data (Protocol B, 1500q/config) loaded separately for the chapter
# narrative + optional trajectory panel.
CALIBRATION_MODE = "calibration"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load_jsonl(p: Path) -> list[dict]:
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def bootstrap_ci(scores: list[float], n_resamples: int = 1000, seed: int = 42) -> tuple[float, float]:
    import random
    rng = random.Random(seed)
    n = len(scores)
    if n == 0:
        return (0.0, 0.0)
    means = []
    for _ in range(n_resamples):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples)])


def categorize_question(q: str) -> str:
    """Heuristic regex categorization. Disclosed in manifest + chapter footnote."""
    ql = q.lower()
    if re.search(r"\b(capital[-\s]?intensive|capital[-\s]?light|is .* a (good|bad|reasonable|stable)|does .* (improve|worsen|grow|decline)|would you|assess(?:ment)? of|qualitative)\b", ql):
        return "qualitative judgement"
    if re.search(r"\b(ratio|margin|fcf conversion|asset turnover|cagr|growth rate|return on (assets|equity|capital)|debt[-\s]?to[-\s]?equity|free cash flow conversion|operating cash flow / |dividend payout|interest coverage|quick ratio|current ratio|days (sales|receivable|payable|inventory)|days outstanding)\b", ql):
        return "multi-formula calc"
    if re.search(r"\b(what are the (names|three|four|five)|list (the|all)|enumerate|name the)\b", ql):
        return "list extraction"
    if re.search(r"\bwhat is the\b.*\b(fy\d{4}|fiscal\s+\d{4}|q[1-4]\s+\d{4}|\d{4})\b", ql):
        return "simple extraction"
    return "other"


def build_step_to_doc_map(snapshots_path: Path) -> dict[int, int]:
    """Map global_step -> doc_idx using doc_start snapshots as boundaries."""
    snapshots = json.loads(snapshots_path.read_text(encoding="utf-8"))
    starts = sorted([(s["global_step"], s["doc_idx"]) for s in snapshots if s["kind"] == "doc_start"])
    if not starts:
        return {}
    boundary_steps = [b[0] for b in starts]
    boundary_docs = [b[1] for b in starts]
    max_step = max(s["global_step"] for s in snapshots)

    out: dict[int, int] = {}
    for step in range(max_step + 1):
        idx = bisect_right(boundary_steps, step) - 1
        if idx < 0:
            idx = 0
        out[step] = boundary_docs[idx]
    return out


def in_doc_ratio(retrieved: list[int], own_doc_idx: int, step_to_doc: dict[int, int]) -> float:
    if not retrieved:
        return 0.0
    same = sum(1 for s in retrieved if step_to_doc.get(s) == own_doc_idx)
    return same / len(retrieved)


def main() -> None:
    # Step → doc_idx map. Snapshots only exist for V4t variants; canonical/tuned
    # share the same paragraph order so v4t-canonical's snapshot map applies to
    # canonical AND tuned AND corpus-tuned. For dump-all / bm25-corpus /
    # attention-corpus-tuned the "retrieved_steps" semantic is identical
    # (global step in the canonical ingestion order), so we reuse the same map.
    canonical_snap = TRACES / "financebench__v4t-canonical" / "snapshots.json"
    step_to_doc = build_step_to_doc_map(canonical_snap)

    # --- Pass 1: per-cell aggregates ----------------------------------------
    judge_table: dict[str, dict[str, dict]] = {}
    per_cell_questions: dict[str, list[dict]] = {}  # for question_types + drilldowns
    calibration_data: dict[str, dict] = {}  # per-config calibration mean + per-decile trajectory

    def _empty_cell() -> dict:
        """Sentinel for missing data — frontend renders '—' when n == 0."""
        return {"mean_judge": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                "mean_recall": 0.0, "n": 0, "cost_usd": 0.0, "in_doc_ratio": 0.0}

    def _load_cell(cfg: dict, display_mode: str, queue_mode: str) -> tuple[dict, list[dict]]:
        cname = cfg["name"]
        queue_dir = JUDGE_QUEUE / f"financebench__{cname}__{queue_mode}__seed42"
        if not (queue_dir / "queue.jsonl").exists():
            return _empty_cell(), []

        queue = load_jsonl(queue_dir / "queue.jsonl")
        results = load_jsonl(queue_dir / "results.jsonl")
        scores = [r["judge_score"] for r in results]
        mean_judge = sum(scores) / len(scores) if scores else 0.0
        ci_lo, ci_hi = bootstrap_ci(scores)

        # mean_recall from qa_summary (only original 6 have corpus_traces)
        mean_recall = 0.0
        cost = 0.0
        qa_summary_path = TRACES / f"financebench__{cname}" / "qa_summary.json"
        if qa_summary_path.exists():
            qa_summary = json.loads(qa_summary_path.read_text(encoding="utf-8"))
            cost = float(qa_summary.get("total_cost_usd", 0.0))
            # original 6: display_mode (online/batch) maps directly to qa_summary key
            mean_recall = float(qa_summary.get(display_mode, {}).get("mean_recall_at_k", 0.0))

        # in_doc_ratio averaged across questions
        score_by_qid = {r["qid"]: r["judge_score"] for r in results}
        ratios = []
        cell_qs = []
        for q in queue:
            qid = q["qid"]
            own_doc = q["doc_idx"]
            retrieved = q.get("retrieved_steps", [])
            ratios.append(in_doc_ratio(retrieved, own_doc, step_to_doc))
            cell_qs.append({
                "qid": qid,
                "doc_idx": own_doc,
                "question": q["question"],
                "gold_answer": q["gold_answer"],
                "predicted": q["predicted"],
                "judge_score": score_by_qid.get(qid, 0.0),
                "retrieved_steps": retrieved,
            })

        cell = {
            "mean_judge": round(mean_judge, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "mean_recall": round(mean_recall, 4),
            "n": len(scores),
            "cost_usd": round(cost, 4),
            "in_doc_ratio": round(sum(ratios) / len(ratios) if ratios else 0.0, 4),
        }
        return cell, cell_qs

    for cfg in CONFIGS:
        cname = cfg["name"]
        judge_table[cname] = {}

        # ---- main judge_table (online + batch) ----
        for display_mode in MODES:
            queue_mode = cfg["queue_for"].get(display_mode)
            if queue_mode is None:
                # Extension config without this display mode (e.g., extension lacks online).
                judge_table[cname][display_mode] = _empty_cell()
                continue
            cell, cell_qs = _load_cell(cfg, display_mode, queue_mode)
            judge_table[cname][display_mode] = cell
            per_cell_questions[f"{cname}__{display_mode}"] = cell_qs

        # ---- calibration (Protocol B, 1500q) — for chapter narrative + trajectory ----
        calib_queue_dir = JUDGE_QUEUE / f"financebench__{cname}__{CALIBRATION_MODE}__seed42"
        if (calib_queue_dir / "results.jsonl").exists():
            calib_results = load_jsonl(calib_queue_dir / "results.jsonl")
            calib_scores = [r["judge_score"] for r in calib_results]
            n = len(calib_scores)
            # Per-decile trajectory (150 entries / decile assuming sequential corpus order)
            decile_size = max(1, n // 10)
            deciles = []
            for i in range(10):
                start, end = i * decile_size, (i + 1) * decile_size if i < 9 else n
                chunk = calib_scores[start:end]
                deciles.append(round(sum(chunk) / len(chunk), 4) if chunk else 0.0)
            calibration_data[cname] = {
                "mean_judge": round(sum(calib_scores) / n, 4) if n else 0.0,
                "n": n,
                "deciles": deciles,
            }
        else:
            calibration_data[cname] = {"mean_judge": 0.0, "n": 0, "deciles": [0.0] * 10}

    # --- theta_contrast -----------------------------------------------------
    canonical_theta = {
        "theta_store": 0.293, "theta_decay": 0.668,
        "w_graph": 0.000, "w_embed": 1.079, "w_recency": 3.777,
    }
    perdoc_tuned_path = REPO_ROOT / "results" / "stage3" / "tuned_theta_financebench.json"
    perdoc = {}
    if perdoc_tuned_path.exists():
        perdoc_raw = json.loads(perdoc_tuned_path.read_text(encoding="utf-8"))
        params = perdoc_raw.get("tuned_theta_params") or perdoc_raw.get("tuned_params") or {}
        perdoc = {k: round(params.get(k, 0.0), 4) for k in ["theta_store", "theta_decay", "w_graph", "w_embed", "w_recency"]}
    corpus_tuned_raw = json.loads((REPO_ROOT / "results" / "stage3" / "tuned_theta_v4t_corpus_financebench.json").read_text(encoding="utf-8"))
    corpus_params = corpus_tuned_raw["tuned_params"]
    corpus_theta = {k: round(corpus_params.get(k, 0.0), 4) for k in ["theta_store", "theta_decay", "w_graph", "w_embed", "w_recency"]}

    theta_contrast = {
        "params": ["theta_store", "theta_decay", "w_graph", "w_embed", "w_recency"],
        "rows": [
            {"name": "canonical (grid-world)", "values": [canonical_theta[p] for p in ["theta_store", "theta_decay", "w_graph", "w_embed", "w_recency"]]},
            {"name": "per-doc tuned",          "values": [perdoc.get(p, 0.0)  for p in ["theta_store", "theta_decay", "w_graph", "w_embed", "w_recency"]]},
            {"name": "corpus-tuned (winner)",  "values": [corpus_theta[p]     for p in ["theta_store", "theta_decay", "w_graph", "w_embed", "w_recency"]]},
        ],
        "annotations": [
            f"w_recency: {canonical_theta['w_recency']:.3f} -> {corpus_theta['w_recency']:.3f} (recency collapses)",
            f"w_graph: {canonical_theta['w_graph']:.3f} -> {corpus_theta['w_graph']:.3f} (graph comes alive)",
            f"w_embed: {canonical_theta['w_embed']:.3f} -> {corpus_theta['w_embed']:.3f} (embeddings dominate)",
            f"theta_store: {canonical_theta['theta_store']:.3f} -> {corpus_theta['theta_store']:.3f} (stores nearly everything)",
        ],
    }

    # --- cross_doc_scatter --------------------------------------------------
    scatter = []
    for cfg in CONFIGS:
        for mode in MODES:
            cell = judge_table[cfg["name"]][mode]
            if cell["n"] == 0:
                continue  # extension configs lack online; don't plot empty points
            scatter.append({
                "config": cfg["name"],
                "label": cfg["label"],
                "family": cfg["family"],
                "color": cfg["color"],
                "mode": mode,
                "in_doc_ratio": cell["in_doc_ratio"],
                "mean_judge": cell["mean_judge"],
                "mean_recall": cell["mean_recall"],
                "ci_lower": cell["ci_lower"],
                "ci_upper": cell["ci_upper"],
                "n": cell["n"],
            })

    # --- question_types -----------------------------------------------------
    # Use queue from ONE cell to enumerate the 150 questions canonically
    ref_queue = load_jsonl(JUDGE_QUEUE / "financebench__v4t-corpus-tuned__online__seed42" / "queue.jsonl")
    category_membership = {}
    cat_list = ["multi-formula calc", "qualitative judgement", "list extraction", "simple extraction", "other"]
    for q in ref_queue:
        category_membership[q["qid"]] = categorize_question(q["question"])

    # Question identity (stable across configs/modes) = the doc{N}_qa{M} suffix.
    def qid_suffix(qid: str) -> str:
        return qid.split("__")[-2]

    # always_hard / always_easy / mid bucketing — bucket by SUFFIX (one entry per logical question)
    # Match finbench_audit.json definitions: "always hard" = all ONLINE cells <= 0.25 (no
    # config got more than partial credit); "always easy" = all ONLINE cells >= 0.75.
    online_suffix_scores: dict[str, list[float]] = {}
    all_suffix_scores: dict[str, list[float]] = {}
    for cell_key, qs in per_cell_questions.items():
        for entry in qs:
            sfx = qid_suffix(entry["qid"])
            all_suffix_scores.setdefault(sfx, []).append(entry["judge_score"])
            if cell_key.endswith("__online"):
                online_suffix_scores.setdefault(sfx, []).append(entry["judge_score"])

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0
    # always_hard: mean across 6 online cells <= 0.25 (matches finbench_audit.json's 37);
    # always_easy: mean across 6 online cells >= 0.75 (no config struggles to answer).
    always_hard_sfx = {sfx for sfx, scores in online_suffix_scores.items() if _mean(scores) <= 0.25}
    always_easy_sfx = {sfx for sfx, scores in online_suffix_scores.items() if _mean(scores) >= 0.75}
    mid_sfx = set(online_suffix_scores) - always_hard_sfx - always_easy_sfx
    suffix_scores = all_suffix_scores  # for downstream

    # Category lookup by suffix (using ref_queue qids).
    suffix_to_cat: dict[str, str] = {}
    for q in ref_queue:
        suffix_to_cat[qid_suffix(q["qid"])] = category_membership.get(q["qid"], "other")

    def cat_counts(sfxs: set[str]) -> list[int]:
        out = [0] * len(cat_list)
        for s in sfxs:
            cat = suffix_to_cat.get(s, "other")
            out[cat_list.index(cat)] += 1
        return out

    per_cell_mean_judge: dict[str, dict[str, float]] = {}
    per_cell_n_by_cat: dict[str, dict[str, int]] = {}
    for cell_key, qs in per_cell_questions.items():
        by_cat: dict[str, list[float]] = {c: [] for c in cat_list}
        for entry in qs:
            cat = category_membership.get(entry["qid"], "other")
            by_cat[cat].append(entry["judge_score"])
        per_cell_mean_judge[cell_key] = {c: round(sum(v) / len(v), 4) if v else 0.0 for c, v in by_cat.items()}
        per_cell_n_by_cat[cell_key] = {c: len(v) for c, v in by_cat.items()}

    question_types = {
        "categories": cat_list,
        "category_method": "heuristic regex on question text; not human-coded",
        "counts": {
            "always_hard": cat_counts(always_hard_sfx),
            "always_easy": cat_counts(always_easy_sfx),
            "mid":         cat_counts(mid_sfx),
        },
        "category_membership": category_membership,
        "per_cell_mean_judge": per_cell_mean_judge,
        "per_cell_n_by_cat": per_cell_n_by_cat,
        "always_hard_n": len(always_hard_sfx),
        "always_easy_n": len(always_easy_sfx),
        "mid_n": len(mid_sfx),
    }

    # --- drilldown_examples -------------------------------------------------
    def predictions_for_qid(qid: str) -> dict[str, dict]:
        out = {}
        for cell_key, qs in per_cell_questions.items():
            for entry in qs:
                if entry["qid"].split("__")[-2] == qid.split("__")[-2]:  # match doc_idx + qa_idx
                    out[cell_key] = {
                        "predicted": entry["predicted"][:400],
                        "judge_score": entry["judge_score"],
                    }
                    break
        return out

    # Build example sets from ref_queue (which is the corpus-tuned online queue;
    # has 150 entries, qid suffix matches doc_idx + qa_idx).
    ref_by_doc_qa = {q["qid"].split("__")[-2]: q for q in ref_queue}

    def base_meta(q: dict) -> dict:
        return {
            "doc_idx": q["doc_idx"],
            "question": q["question"][:300],
            "gold_answer": q["gold_answer"][:300],
            "category": category_membership.get(q["qid"], "other"),
        }

    # always hard: rank by total-across-cells score (lowest first)
    def total_score_for_qid_suffix(suffix: str) -> float:
        total = 0.0
        for qs in per_cell_questions.values():
            for entry in qs:
                if entry["qid"].endswith(f"__{suffix}__seed42"):
                    total += entry["judge_score"]
        return total

    hard_sorted = sorted(always_hard_sfx, key=total_score_for_qid_suffix)
    easy_sorted = sorted(always_easy_sfx, key=lambda s: -total_score_for_qid_suffix(s))

    # corpus_tuned wins / regressions vs canonical (online comparison)
    corpus_wins: list[tuple[str, float]] = []
    corpus_losses: list[tuple[str, float]] = []
    for entry in per_cell_questions["v4t-corpus-tuned__online"]:
        suffix = qid_suffix(entry["qid"])
        c_score = entry["judge_score"]
        n_score = next((e["judge_score"] for e in per_cell_questions["v4t-canonical__online"] if qid_suffix(e["qid"]) == suffix), 0.0)
        diff = c_score - n_score
        if diff > 0.5:
            corpus_wins.append((suffix, diff))
        elif diff < -0.25:
            corpus_losses.append((suffix, diff))
    corpus_wins.sort(key=lambda x: -x[1])
    corpus_losses.sort(key=lambda x: x[1])

    def example_from_suffix(suffix: str) -> dict | None:
        q = ref_by_doc_qa.get(suffix)
        if not q:
            return None
        out = base_meta(q)
        out["predictions"] = predictions_for_qid(q["qid"])
        return out

    drilldown_examples = {
        "always_hard": [e for e in (example_from_suffix(s) for s in hard_sorted[:10]) if e],
        "always_easy": [e for e in (example_from_suffix(s) for s in easy_sorted[:5]) if e],
        "corpus_tuned_wins_vs_canonical": [e for e in (example_from_suffix(s) for s, _ in corpus_wins[:10]) if e],
        "corpus_tuned_regressions_vs_canonical": [e for e in (example_from_suffix(s) for s, _ in corpus_losses[:14]) if e],
    }

    # --- assemble ----------------------------------------------------------
    out = {
        "_manifest": {
            "git_sha": _git_sha(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_questions": 150,
            "k": 8,
            "model": "gpt-4o-mini",
            "manual_judge": True,
            "categorization_method": "heuristic regex on question text; not human-coded",
            "cross_doc_bleed_metric": "in_doc_ratio = (same-doc retrieved steps) / k; uses global_step->doc_idx from v4t-canonical snapshots (paragraph order is identical across configs)",
            "judge_protocol": "evaluation/claude_judge_protocol.md",
        },
        "configs": CONFIGS,
        "modes": MODES,
        "judge_table": judge_table,
        "calibration_data": calibration_data,  # Protocol B 1500q means + 10-decile trajectory per config
        "theta_contrast": theta_contrast,
        "cross_doc_scatter": scatter,
        "question_types": question_types,
        "drilldown_examples": drilldown_examples,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    n_populated = sum(1 for c in CONFIGS for m in MODES if judge_table[c["name"]][m]["n"] > 0)
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"  configs={len(out['configs'])}  cells_total={sum(len(v) for v in out['judge_table'].values())}  cells_populated={n_populated}")
    print(f"  scatter_points={len(scatter)}  calibration_configs={len([k for k, v in calibration_data.items() if v['n'] > 0])}")
    print(f"  examples=hard:{len(drilldown_examples['always_hard'])} easy:{len(drilldown_examples['always_easy'])} wins:{len(drilldown_examples['corpus_tuned_wins_vs_canonical'])} losses:{len(drilldown_examples['corpus_tuned_regressions_vs_canonical'])}")
    print(f"  bucket counts: hard={question_types['always_hard_n']} easy={question_types['always_easy_n']} mid={question_types['mid_n']}")


if __name__ == "__main__":
    main()
