"""
Held-out vs tuned-on split for corpus-mode QA results (critique remediation).

The V4t corpus CMA-ES tuner (tuning/tune_v4t_corpus.py) optimized recall@k
on qa_pairs[0] of the first `limit_docs` docs of each benchmark. Those exact
questions are part of the Protocol A evaluation set, so pooled cell means mix
tuned-on and held-out questions:

    benchmark      tuning scope          eval n   tuned-on   held-out
    financebench   docs 0-49, qa0          150        50        100
    qasper         docs 0-29, qa0           94        30         64
    cuad           docs 0-29, qa0          132        10        122

(HotpotQA and LongMemEval have NO held-out questions at the original
10-doc eval scale — they are handled by fresh held-out runs, not here.)

This script re-aggregates the existing per-question Claude judgments into
tuned-on and held-out splits, with cluster-bootstrap CIs (clusters = docs)
on the held-out means and paired Wilcoxon tests (vs v4t-canonical) with
Holm-Bonferroni correction. No new LLM calls, no re-judging — pure
re-aggregation of results/stage3/judge_queue/*/results.jsonl.

Output: results/stage3/holdout_split_summary.json

Usage:
    python scripts/build_holdout_split.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.statistics import (  # noqa: E402
    cluster_bootstrap_ci,
    holm_bonferroni,
    wilcoxon_signed_rank,
)

QUEUE_ROOT = ROOT / "results" / "stage3" / "judge_queue"
OUT_PATH = ROOT / "results" / "stage3" / "holdout_split_summary.json"

# Tuning scope per benchmark: CMA-ES objective = qa0 of docs 0..limit-1.
# Source of truth: results/stage3/tuned_theta_v4t_corpus_<bench>.json
# (limit_docs field).
TUNE_LIMIT = {"financebench": 50, "qasper": 30, "cuad": 30}

CONFIGS = [
    "v4t-canonical",
    "v4t-tuned",
    "v4t-corpus-tuned",
    "bm25-corpus",
    "attention-corpus-tuned",
    "dump-all",
]
MODES = ["online", "batch"]
BASELINE_CONFIG = "v4t-canonical"

_QID_RE = re.compile(r"__doc(\d+)_qa(\d+)__")


def load_cell(bench: str, cfg: str, mode: str) -> dict[str, dict] | None:
    """Return {qid: {score, doc_idx, qa_idx}} for one judged cell, or None."""
    path = QUEUE_ROOT / f"{bench}__{cfg}__{mode}__seed42" / "results.jsonl"
    if not path.exists():
        return None
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        qid = rec.get("qid", "")
        m = _QID_RE.search(qid)
        if not m:
            continue
        out[qid] = {
            "score": float(rec["judge_score"]),
            "doc_idx": int(m.group(1)),
            "qa_idx": int(m.group(2)),
        }
    return out or None


def split_cell(entries: dict[str, dict], limit: int) -> tuple[list[dict], list[dict]]:
    """Split into (tuned_on, held_out) by the tuner's question scope."""
    tuned, held = [], []
    for e in entries.values():
        (tuned if (e["doc_idx"] < limit and e["qa_idx"] == 0) else held).append(e)
    return tuned, held


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def main() -> int:
    report: dict = {"tune_limit": TUNE_LIMIT, "benchmarks": {}}

    for bench, limit in TUNE_LIMIT.items():
        bench_out: dict = {"configs": {}, "paired_tests": []}
        cells: dict[tuple[str, str], dict[str, dict]] = {}

        for cfg in CONFIGS:
            for mode in MODES:
                entries = load_cell(bench, cfg, mode)
                if entries is None:
                    continue
                cells[(cfg, mode)] = entries
                tuned, held = split_cell(entries, limit)
                held_scores = [e["score"] for e in held]
                held_docs = [e["doc_idx"] for e in held]
                ci = (
                    cluster_bootstrap_ci(held_scores, held_docs)
                    if len(held_scores) >= 5
                    else None
                )
                bench_out["configs"].setdefault(cfg, {})[mode] = {
                    "n_total": len(entries),
                    "pooled_mean": mean([e["score"] for e in entries.values()]),
                    "n_tuned_on": len(tuned),
                    "tuned_on_mean": mean([e["score"] for e in tuned]),
                    "n_held_out": len(held),
                    "held_out_mean": mean(held_scores),
                    "held_out_ci95": (
                        [ci["ci_lower"], ci["ci_upper"]] if ci else None
                    ),
                }

        # Paired Wilcoxon vs canonical on the HELD-OUT subset only,
        # paired by (doc_idx, qa_idx) which is identical across configs
        # (same seed, same question inventory).
        pvals: list[float] = []
        tests: list[dict] = []
        for cfg in CONFIGS:
            if cfg == BASELINE_CONFIG:
                continue
            for mode in MODES:
                if (cfg, mode) not in cells or (BASELINE_CONFIG, mode) not in cells:
                    continue
                base = cells[(BASELINE_CONFIG, mode)]
                test = cells[(cfg, mode)]

                def held_map(entries: dict[str, dict]) -> dict[tuple, float]:
                    return {
                        (e["doc_idx"], e["qa_idx"]): e["score"]
                        for e in entries.values()
                        if not (e["doc_idx"] < limit and e["qa_idx"] == 0)
                    }

                bmap, tmap = held_map(base), held_map(test)
                keys = sorted(set(bmap) & set(tmap))
                if len(keys) < 5:
                    continue
                a = [tmap[k] for k in keys]
                b = [bmap[k] for k in keys]
                w = wilcoxon_signed_rank(a, b)
                tests.append({
                    "config": cfg,
                    "mode": mode,
                    "n_paired": len(keys),
                    "held_out_lift": mean(a) - mean(b),
                    "wilcoxon_p_two_sided": w["p_two_sided"],
                    "n_nonzero_diffs": w["n_nonzero"],
                })
                pvals.append(w["p_two_sided"])

        # Holm correction across this benchmark's test family.
        if pvals:
            adj = holm_bonferroni(pvals)
            for t, a in zip(tests, adj):
                t["p_holm"] = a["p_adjusted"]
                t["significant_holm"] = a["significant"]
        bench_out["paired_tests"] = tests
        report["benchmarks"][bench] = bench_out

    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"[build_holdout_split] wrote {OUT_PATH}")

    # Human-readable table.
    for bench, b in report["benchmarks"].items():
        print(f"\n=== {bench} (tuned on qa0 of docs 0-{TUNE_LIMIT[bench]-1}) ===")
        print(f"{'config':<24} {'mode':<7} {'pooled':>7} {'tuned-on':>9} {'held-out':>9} {'CI95':>18}")
        for cfg, modes in b["configs"].items():
            for mode, s in modes.items():
                ci = s["held_out_ci95"]
                ci_s = f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci else "-"
                print(
                    f"{cfg:<24} {mode:<7} "
                    f"{s['pooled_mean']:>7.3f} "
                    f"{(s['tuned_on_mean'] if s['tuned_on_mean'] is not None else float('nan')):>9.3f} "
                    f"{(s['held_out_mean'] if s['held_out_mean'] is not None else float('nan')):>9.3f} "
                    f"{ci_s:>18}"
                )
        for t in b["paired_tests"]:
            sig = "SIG" if t.get("significant_holm") else "ns"
            print(
                f"  paired vs canonical [{t['mode']:<6}] {t['config']:<24} "
                f"held-out lift={t['held_out_lift']:+.3f}  "
                f"p={t['wilcoxon_p_two_sided']:.4f}  p_holm={t.get('p_holm', float('nan')):.4f}  {sig}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
