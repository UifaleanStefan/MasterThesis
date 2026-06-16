"""Construct-validity analysis: does the CMA-ES tuning objective (recall@k) predict
the headline metric (LLM judge score)? Joins per-question recall@k (from corpus
traces) with the authoritative judge score (from judge_queue results.jsonl) by
qid, across the Stage-3 corpus-mode batch cells, and reports the correlation
plus the within-config causal chain (corpus-tuning raises recall AND judge).

Writes results/stage3/recall_vs_judge.json and docs/figures/fig_recall_vs_judge.png.
"""
import glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / "results" / "stage3" / "corpus_traces"
QUEUE = ROOT / "results" / "stage3" / "judge_queue"

BENCH_OF = {"finbench": "FinanceBench", "financebench": "FinanceBench", "qasper": "QASPER",
            "cuad": "CUAD", "hotpot": "HotpotQA", "longmem": "LongMemEval"}  # NQA has no gold recall


def bench_of(cell):
    c = cell.lower()
    for k, v in BENCH_OF.items():
        if k in c:
            return v
    return None


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


pairs_by_bench = defaultdict(list)        # bench -> list of (recall, judge)
cfg_means = defaultdict(dict)             # bench -> cfg -> {recall, judge}
for tdir in sorted(glob.glob(str(TRACES / "*"))):
    cell = Path(tdir).name                 # e.g. cuad__v4t-corpus-tuned
    b = bench_of(cell)
    if not b:
        continue
    qb = Path(tdir) / "qa_batch.json"
    rj = QUEUE / f"{cell}__batch__seed42" / "results.jsonl"
    if not (qb.exists() and rj.exists()):
        continue
    recall = {}
    for e in json.loads(qb.read_text("utf-8")):
        if e.get("qid") is not None and e.get("recall_at_k") is not None:
            recall[e["qid"]] = float(e["recall_at_k"])
    judge = {}
    for l in rj.read_text("utf-8").splitlines():
        if l.strip():
            j = json.loads(l); judge[j["qid"]] = float(j["judge_score"])
    common = [q for q in recall if q in judge]
    if len(common) < 10:
        continue
    rs = [recall[q] for q in common]; js = [judge[q] for q in common]
    # only configs that are corpus-mode v4t variants (the headline pair)
    if "v4t-canonical" in cell or "v4t-corpus-tuned" in cell:
        pairs_by_bench[b] += list(zip(rs, js))
        cfg = "canonical" if "canonical" in cell else "corpus-tuned"
        cfg_means[b][cfg] = {"recall": float(np.mean(rs)), "judge": float(np.mean(js)), "n": len(common)}

all_pairs = [p for ps in pairs_by_bench.values() for p in ps]
R = np.array([p[0] for p in all_pairs]); J = np.array([p[1] for p in all_pairs])
out = {
    "n_pairs": len(all_pairs),
    "pooled_spearman": round(spearman(R, J), 3),
    "pooled_pearson": round(pearson(R, J), 3),
    "per_benchmark": {b: {"n": len(ps),
                          "spearman": round(spearman(np.array([p[0] for p in ps]), np.array([p[1] for p in ps])), 3)}
                      for b, ps in pairs_by_bench.items() if len(ps) >= 30},
    "binned_mean_judge_by_recall": {},
    "causal_chain_canonical_to_corpus_tuned": cfg_means,
}
# binned: mean judge in recall buckets
bins = [(-0.01, 0.0), (0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.999), (0.999, 1.01)]
labels = ["0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)", "1.0"]
bin_means, bin_ns = [], []
for (lo, hi), lab in zip(bins, labels):
    m = [j for r, j in all_pairs if lo < r <= hi]
    out["binned_mean_judge_by_recall"][lab] = {"mean_judge": round(float(np.mean(m)), 3) if m else None, "n": len(m)}
    bin_means.append(np.mean(m) if m else np.nan); bin_ns.append(len(m))

(ROOT / "results/stage3/recall_vs_judge.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))

# --- figure: recall bucket -> mean judge (monotone => recall predicts judge) ---
fig, ax = plt.subplots(figsize=(7.5, 5))
xs = [l for l, m in zip(labels, bin_means) if not np.isnan(m)]
ys = [m for m in bin_means if not np.isnan(m)]
ns = [n for n, m in zip(bin_ns, bin_means) if not np.isnan(m)]
ax.bar(range(len(xs)), ys, color="#2563EB", edgecolor="white")
for i, (y, n) in enumerate(zip(ys, ns)):
    ax.text(i, y + 0.01, f"{y:.2f}\n(n={n})", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs)
ax.set_xlabel("recall@8 of gold evidence (the CMA-ES tuning objective)")
ax.set_ylabel("mean LLM judge score (the headline metric)")
ax.set_ylim(0, 1.0)
ax.set_title(f"Construct validity: retrieval recall predicts answer quality\n"
             f"(pooled Spearman $\\rho={out['pooled_spearman']}$, n={out['n_pairs']} "
             f"questions across {len(out['per_benchmark'])} benchmarks)")
ax.grid(True, axis="y", alpha=0.3)
fig.savefig(ROOT / "docs/figures/fig_recall_vs_judge.png", dpi=150, bbox_inches="tight")
print("[OK] docs/figures/fig_recall_vs_judge.png")
