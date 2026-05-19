"""
Cross-benchmark theta-transfer ablation — Stage 3 Phase 1.6 workstream C.

Given two benchmarks (default: QASPER + CUAD — the only ones where
per-benchmark tuning produced measurable lift over canonical), compute
a 3-row x 2-column matrix:

                        QASPER eval     CUAD eval
    canonical theta        0.107           0.458
    QASPER-tuned theta     0.464*          ??.??
    CUAD-tuned   theta     ??.??           0.687*

* = diagonal cells, copied from the narrow tuning run.
?? = off-diagonal cells, evaluated by this script.

Expected pattern: diagonal cells dominate; off-diagonal cells fall back
to near-canonical performance. That would confirm "task-specific theta
does not transfer across long-haystack QA tasks", mirroring the
cross-environment finding from the Stage 1 grid-world chapter.

Output: `results/stage3/theta_transfer_matrix.json` with the matrix +
diagonal_lift_avg, off_diagonal_avg, interpretation string, and manifest.

Uses the WIDE theta if available, falling back to narrow. Picks up
both via the same `tuned_theta_[wide_]<bench>.json` convention as
`scripts/run_stage3_retrieval.py`.

Usage:
    python scripts/run_theta_transfer.py
    python scripts/run_theta_transfer.py --benchmarks qasper cuad --n-docs 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from environment.benchmarks import ADAPTERS, get_adapter
from environment.document_qa import DocumentQA
from evaluation.document_qa_memory import _recall_at_k_for_qa, _run_reading_phase
from memory.graph_memory_v4 import GraphMemoryV4, MemoryParamsV4
from results.manifest import build_manifest
from tuning.tune_v4_per_benchmark import CANONICAL_THETA_VEC, vec_to_params

TUNED_DIR = ROOT / "results" / "stage3"


def _load_theta_vec_named(bench: str, variant: str) -> tuple[list[float], str] | None:
    """Load a specific variant's theta for `bench`.

    Variants: "default" (narrow Phase 1.5), "wide" (Phase 1.6), "heldout" (Phase 1.7).
    """
    if variant == "default":
        path = TUNED_DIR / f"tuned_theta_{bench}.json"
    elif variant == "wide":
        path = TUNED_DIR / f"tuned_theta_wide_{bench}.json"
    elif variant == "heldout":
        path = TUNED_DIR / f"tuned_theta_heldout_{bench}.json"
    else:
        return None
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if data.get("status") != "ok":
        return None
    vec = data.get("tuned_theta_vec")
    if isinstance(vec, list) and len(vec) == 10:
        return vec, variant
    return None


def _load_theta_vec_for(bench: str) -> tuple[list[float], str] | None:
    """Pick the BEST tuned theta for `bench`.

    Preference order:
      1. Whichever of (narrow, wide) the comparison JSON flags as preferred.
      2. Wide if both exist and no comparison is present.
      3. Narrow as fallback.

    Returns (vec, source_label) or None.
    """
    wide_path = TUNED_DIR / f"tuned_theta_wide_{bench}.json"
    narrow_path = TUNED_DIR / f"tuned_theta_{bench}.json"
    comparison_path = TUNED_DIR / "theta_width_comparison.json"

    prefer_wide = True  # default to wide when both exist
    if comparison_path.exists():
        try:
            comp = json.loads(comparison_path.read_text())
            bench_data = comp.get("results", {}).get(bench, {})
            if "prefer_wide" in bench_data:
                prefer_wide = bool(bench_data["prefer_wide"])
        except Exception:
            pass

    order = [(wide_path, "wide"), (narrow_path, "narrow")] if prefer_wide \
        else [(narrow_path, "narrow"), (wide_path, "wide")]
    for path, label in order:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            if data.get("status") != "ok":
                continue
            vec = data.get("tuned_theta_vec")
            if isinstance(vec, list) and len(vec) == 10:
                return vec, label
        except Exception:
            continue
    return None


def eval_theta_on_benchmark(
    theta_vec: list[float] | np.ndarray,
    benchmark: str,
    n_docs: int,
    k: int,
    seed: int,
) -> dict:
    """Run mean recall@k over `n_docs` docs of `benchmark` using V4 with this theta.

    Returns {mean_recall, n_questions_with_gold, n_questions_total, elapsed_s}.
    """
    params = vec_to_params(np.asarray(theta_vec, dtype=np.float64))
    adapter = get_adapter(benchmark)
    docs = list(adapter.iter_documents(limit=n_docs))

    recalls: list[float] = []
    n_with_gold = 0
    n_total = 0
    t0 = time.time()
    for doc in docs:
        memory = GraphMemoryV4(params)
        env = DocumentQA(document=doc, seed=seed, question_shuffle=False)
        _run_reading_phase(env, memory, episode_seed=seed)
        n_paragraphs = len(doc["paragraphs"])
        for qa_idx, qa in enumerate(doc["qa_pairs"]):
            n_total += 1
            relevant = qa.get("relevant_paragraphs", []) or []
            if not relevant:
                continue
            n_with_gold += 1
            current_step = n_paragraphs + qa_idx
            r = _recall_at_k_for_qa(
                memory, qa["question"], relevant,
                k=k, current_step=current_step,
            )
            recalls.append(r)
    elapsed = time.time() - t0
    return {
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
        "n_questions_with_gold": n_with_gold,
        "n_questions_total": n_total,
        "n_docs": len(docs),
        "elapsed_s": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks", nargs="*", default=["qasper", "cuad"],
        help="Benchmarks to include in the matrix (default qasper + cuad).",
    )
    parser.add_argument("--n-docs", type=int, default=15)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", default="results/stage3/theta_transfer_matrix.json",
    )
    args = parser.parse_args()

    for b in args.benchmarks:
        if b not in ADAPTERS:
            print(f"  [FAIL] unknown benchmark: {b!r}")
            return 1

    # Load tuned thetas: both the default-variant tuned (Phase 1.5 narrow / 1.6 wide)
    # AND the held-out-tuned variants (Phase 1.7), if available. Heldout tuners
    # use 25 disjoint docs and are the honest source of cross-task generalization
    # evidence — adding them roughly doubles the transfer matrix rows.
    tuned: dict[str, tuple[list[float], str]] = {}
    heldout: dict[str, tuple[list[float], str]] = {}
    for b in args.benchmarks:
        loaded = _load_theta_vec_for(b)
        if loaded is None:
            print(f"  [WARN] no default-tuned theta for {b}")
        else:
            tuned[b] = loaded
        # Add heldout variant if Phase 1.7 generated one.
        ho = _load_theta_vec_named(b, "heldout")
        if ho is not None:
            heldout[b] = ho

    if not tuned and not heldout:
        print("  [FAIL] no tuned thetas found anywhere — run tuners first")
        return 1

    # Build the matrix:
    #   rows: canonical + [<bench>-tuned for bench in tuned]
    #                  + [<bench>-heldout for bench in heldout]
    rows: list[tuple[str, list[float], str]] = [
        ("canonical", CANONICAL_THETA_VEC.tolist(), "grid_world_baseline")
    ]
    for b, (vec, label) in tuned.items():
        rows.append((f"{b}-tuned", vec, label))
    for b, (vec, label) in heldout.items():
        rows.append((f"{b}-heldout", vec, label))

    print()
    print("=" * 78)
    print(f"  Cross-benchmark theta-transfer matrix")
    print(f"  rows: {[r[0] for r in rows]}")
    print(f"  cols: {args.benchmarks}")
    print(f"  n_docs={args.n_docs}, k={args.k}, seed={args.seed}")
    print("=" * 78)

    matrix: dict[str, dict[str, dict]] = {}
    t_total = time.time()
    for row_name, vec, source_label in rows:
        print(f"\n  --- {row_name} (source: {source_label}) ---")
        row_results: dict[str, dict] = {}
        for col_bench in args.benchmarks:
            print(f"    eval on {col_bench}...", end=" ", flush=True)
            res = eval_theta_on_benchmark(
                vec, col_bench, n_docs=args.n_docs, k=args.k, seed=args.seed,
            )
            row_results[col_bench] = res
            print(
                f"mean_recall={res['mean_recall']:.4f}  "
                f"({res['n_questions_with_gold']}/{res['n_questions_total']} gold, "
                f"{res['elapsed_s']:.1f}s)"
            )
        matrix[row_name] = row_results
    elapsed_total = time.time() - t_total

    # Compute summary diagonals / off-diagonals.
    # Skip canonical row for diagonal averages — diagonal is "X-tuned on X".
    diag_cells: list[float] = []
    offdiag_cells: list[float] = []
    canonical_cells: list[float] = []
    for row_name, row_results in matrix.items():
        if row_name == "canonical":
            for c, cell in row_results.items():
                canonical_cells.append(cell["mean_recall"])
            continue
        # row_name is like "qasper-tuned"
        tuned_bench = row_name.rsplit("-", 1)[0]
        for col_bench, cell in row_results.items():
            if col_bench == tuned_bench:
                diag_cells.append(cell["mean_recall"])
            else:
                offdiag_cells.append(cell["mean_recall"])

    diag_avg = float(np.mean(diag_cells)) if diag_cells else None
    offdiag_avg = float(np.mean(offdiag_cells)) if offdiag_cells else None
    canonical_avg = float(np.mean(canonical_cells)) if canonical_cells else None
    diag_lift_vs_canonical = (diag_avg - canonical_avg) if (diag_avg is not None and canonical_avg is not None) else None
    offdiag_lift_vs_canonical = (offdiag_avg - canonical_avg) if (offdiag_avg is not None and canonical_avg is not None) else None

    interpretation = []
    if diag_lift_vs_canonical is not None and diag_lift_vs_canonical > 0.05:
        interpretation.append(
            f"Diagonal cells (matched theta) beat canonical by avg {diag_lift_vs_canonical:+.3f}."
        )
    if offdiag_lift_vs_canonical is not None and abs(offdiag_lift_vs_canonical) < 0.05:
        interpretation.append(
            f"Off-diagonal cells (mismatched theta) stay within +-0.05 of canonical "
            f"(avg lift {offdiag_lift_vs_canonical:+.3f}) — confirming task-specific theta "
            f"does NOT transfer cross-task within document QA."
        )
    elif offdiag_lift_vs_canonical is not None and offdiag_lift_vs_canonical > 0.05:
        interpretation.append(
            f"Off-diagonal cells exceed canonical by avg {offdiag_lift_vs_canonical:+.3f} — "
            f"some cross-task transfer DOES happen between the long-haystack benchmarks."
        )
    elif offdiag_lift_vs_canonical is not None and offdiag_lift_vs_canonical < -0.05:
        interpretation.append(
            f"Off-diagonal cells fall BELOW canonical by avg {abs(offdiag_lift_vs_canonical):.3f} — "
            f"mismatched theta actively hurts retrieval more than canonical does."
        )

    print()
    print("=" * 78)
    print("  TRANSFER MATRIX SUMMARY")
    print("=" * 78)
    print(f"  canonical avg recall:    {canonical_avg:.4f}" if canonical_avg is not None else "")
    print(f"  diagonal     avg recall: {diag_avg:.4f}" if diag_avg is not None else "")
    print(f"  off-diagonal avg recall: {offdiag_avg:.4f}" if offdiag_avg is not None else "")
    if diag_lift_vs_canonical is not None:
        print(f"  diagonal     lift vs canonical: {diag_lift_vs_canonical:+.4f}")
    if offdiag_lift_vs_canonical is not None:
        print(f"  off-diagonal lift vs canonical: {offdiag_lift_vs_canonical:+.4f}")
    for line in interpretation:
        print(f"  -> {line}")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_manifest": build_manifest(seed=args.seed, extra={
            "experiment": "theta_transfer_matrix",
            "benchmarks": args.benchmarks,
            "n_docs": args.n_docs,
            "k": args.k,
        }),
        "rows": [r[0] for r in rows],
        "row_sources": {r[0]: r[2] for r in rows},
        "cols": args.benchmarks,
        "matrix": matrix,
        "diagonal_avg": diag_avg,
        "off_diagonal_avg": offdiag_avg,
        "canonical_avg": canonical_avg,
        "diagonal_lift_vs_canonical": diag_lift_vs_canonical,
        "off_diagonal_lift_vs_canonical": offdiag_lift_vs_canonical,
        "interpretation": " ".join(interpretation),
        "elapsed_s": elapsed_total,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  Saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
