"""
Fair-baseline comparison for the Stage-3 headline (audit B6).

Once the corpus-tuned baselines are judged (Phase 1), this answers the honest
question: does the learned V4t-corpus memory still beat the baselines when the
baselines are tuned on the SAME budget? It loads the judged batch cells, reports
each config's mean judge score with a bootstrap 95% CI, and runs a paired
Wilcoxon signed-rank test of V4t-corpus-tuned vs each baseline (aligned by qid),
with Holm--Bonferroni correction across the baseline family.

Honest by construction: if a tuned baseline ties or beats V4t, the script says so
--- that is the point of the fairness pass.

Pure analysis over committed results.jsonl (no LLM, no judging).
Output: results/stage3/fair_baseline_comparison.json

Usage:
    python scripts/compare_fair_baselines.py
    python scripts/compare_fair_baselines.py --benchmarks qasper cuad
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.statistics import bootstrap_ci, holm_bonferroni, wilcoxon_signed_rank

QUEUE = ROOT / "results" / "stage3" / "judge_queue"

LEARNED = "v4t-corpus-tuned"
# Baselines to compare against, in report order. Corpus-tuned variants are the
# fair ones; the stock variants are kept for the before/after contrast.
BASELINES = ["bm25-corpus-tuned", "bm25-corpus", "attention-corpus-tuned", "dump-all"]
BENCHMARKS = ["financebench", "qasper", "cuad", "hotpotqa", "longmemeval"]


def load_cell(benchmark: str, config: str, mode: str = "batch", seed: int = 42) -> dict:
    """Map a CONFIG-INDEPENDENT question key -> judge_score for a judged cell.

    qids embed the config (e.g. financebench__v4t-corpus-tuned__batch__doc0_qa0__seed42),
    so to pair the same question across configs we strip the config token, leaving
    e.g. financebench__batch__doc0_qa0__seed42 as the alignment key."""
    rj = QUEUE / f"{benchmark}__{config}__{mode}__seed{seed}" / "results.jsonl"
    if not rj.exists():
        return {}
    out = {}
    for line in rj.read_text(encoding="utf-8").splitlines():
        if line.strip():
            j = json.loads(line)
            if j.get("judge_score") is not None:
                key = j["qid"].replace(f"__{config}__", "__")
                out[key] = float(j["judge_score"])
    return out


def _mean_ci(scores: list[float]) -> dict:
    if not scores:
        return {"n": 0, "mean": None, "ci": None}
    ci = bootstrap_ci(scores) if len(scores) >= 5 else None
    return {"n": len(scores), "mean": round(sum(scores) / len(scores), 4),
            "ci": ([round(ci["ci_lower"], 4), round(ci["ci_upper"], 4)] if ci else None)}


def compare(learned: dict, baselines: dict) -> dict:
    """learned: qid->score; baselines: {name: qid->score}. Paired Wilcoxon of
    learned vs each baseline on shared qids, Holm-corrected across baselines."""
    configs = {LEARNED: _mean_ci(list(learned.values()))}
    tests = []
    pvals = []
    for name, scores in baselines.items():
        configs[name] = _mean_ci(list(scores.values()))
        shared = sorted(set(learned) & set(scores))
        if len(shared) >= 5:
            a = [learned[q] for q in shared]     # learned
            b = [scores[q] for q in shared]       # baseline
            w = wilcoxon_signed_rank(a, b)
            # Direction from the MEAN difference: judge scores are discrete so the
            # median paired diff is often exactly 0 even when the means differ and
            # Wilcoxon is significant -- using the median for direction is wrong.
            mean_diff = round(sum(a) / len(a) - sum(b) / len(b), 4)
            entry = {"baseline": name, "n_paired": len(shared),
                     "learned_mean_minus_baseline": mean_diff,
                     "median_paired_diff": w.get("median_diff"),
                     "p_two_sided": w.get("p_two_sided")}
            tests.append(entry)
            pvals.append(w.get("p_two_sided"))
    # Holm across the baseline family
    if pvals:
        holm = holm_bonferroni(pvals)
        learned_n = configs[LEARNED]["n"]
        for t, h in zip(tests, holm):
            t["p_holm"] = round(h["p_adjusted"], 5)
            t["significant_holm"] = bool(h["significant"])
            # Guard: if the two cells were evaluated at very different scales
            # (e.g. learned at 50 docs, baseline at 10), the paired test compares
            # different memory states and the verdict is NOT trustworthy.
            base_n = configs[t["baseline"]]["n"]
            hi = max(learned_n, base_n) or 1
            t["scale_mismatch"] = abs(learned_n - base_n) / hi > 0.2
            md = t["learned_mean_minus_baseline"]
            if t["scale_mismatch"]:
                t["verdict"] = (f"NOT COMPARABLE (different scale: learned n={learned_n} "
                                f"vs baseline n={base_n}) -- re-run baseline at matched scale")
            else:
                t["verdict"] = (
                    "learned wins" if (t["significant_holm"] and md > 0)
                    else "baseline wins" if (t["significant_holm"] and md < 0)
                    else "tie (n.s.)")
    return {"configs": configs, "paired_tests": tests}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", nargs="+", default=BENCHMARKS)
    ap.add_argument("--mode", default="batch")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = {}
    for bench in args.benchmarks:
        learned = load_cell(bench, LEARNED, args.mode, args.seed)
        if not learned:
            out[bench] = {"status": "learned cell not judged yet"}
            continue
        baselines = {}
        for b in BASELINES:
            s = load_cell(bench, b, args.mode, args.seed)
            if s:
                baselines[b] = s
        out[bench] = {"status": "ok", **compare(learned, baselines)}

    path = ROOT / "results" / "stage3" / "fair_baseline_comparison.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:2000])
    print(f"\n-> saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
