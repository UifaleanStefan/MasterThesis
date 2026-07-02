"""Full-corpus scaling aggregation. For the two large domain-coherent benchmarks,
report the judged end-of-corpus batch accuracy at the COMPLETE corpus (CUAD 510
contracts, QASPER 281 papers) using the already-judged batch_calib cells (mode=batch,
expected_behavior=none, docs_seen=full). Paired by doc{D}_qa{Q} suffix.

Emits per-config bootstrap 95% CI, the corpus-tuning lift (corpus-tuned vs canonical),
and fair verdicts (corpus-tuned vs each baseline) with paired Wilcoxon + Holm.
Writes results/stage3/fullcorpus_scaling.json and prints LaTeX-ready rows.

Configs are discovered dynamically, so re-running after new baseline cells are judged
(e.g. bm25-corpus-tuned) updates the table automatically.
"""
from __future__ import annotations
import json, re, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.statistics import bootstrap_ci, wilcoxon_signed_rank, holm_bonferroni

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "results" / "stage3" / "judge_queue"
# preferred display order; only those with a judged full-corpus cell are shown
ORDER = ["v4t-corpus-tuned", "v4t-tuned", "v4t-canonical",
         "bm25-corpus-tuned", "attention-corpus-tuned", "bm25-corpus", "dump-all"]
BENCHES = {"cuad": ("CUAD", 510), "qasper": ("QASPER", 281)}

def cell_maxdocs(cell):
    """Max docs_seen in the cell's queue — the corpus scope it was run at.
    Guards against mixing a 10-doc baseline with a 510-doc headline cell."""
    p = Q / cell / "queue.jsonl"
    if not p.exists():
        return -1
    mx = 0
    for l in p.read_text(encoding="utf-8").splitlines():
        if l.strip():
            mx = max(mx, json.loads(l).get("docs_seen", 0) or 0)
    return mx

def load(cell, require_docs):
    rp = Q / cell / "results.jsonl"
    if not rp.exists():
        return None
    if cell_maxdocs(cell) != require_docs:   # only accept FULL-corpus cells
        return None
    out = {}
    for l in rp.read_text(encoding="utf-8").splitlines():
        if l.strip():
            d = json.loads(l)
            out[re.search(r"(doc\d+_qa\d+)", d["qid"]).group(1)] = float(d["judge_score"])
    return out

def run_bench(bench):
    ndocs = BENCHES[bench][1]
    cells = {c: load(f"{bench}__{c}__batch_calib__seed42", ndocs) for c in ORDER}
    cells = {c: d for c, d in cells.items() if d}  # keep judged full-corpus only
    if "v4t-corpus-tuned" not in cells:
        return None
    common = sorted(set.intersection(*[set(d) for d in cells.values()]))
    ct = [cells["v4t-corpus-tuned"][s] for s in common]
    rows, pvals, comp = {}, [], []
    for c, d in cells.items():
        arr = [d[s] for s in common]
        ci = bootstrap_ci(arr)
        rows[c] = {"mean": ci["point_estimate"], "lo": ci["ci_lower"], "hi": ci["ci_upper"]}
        if c != "v4t-corpus-tuned":
            w = wilcoxon_signed_rank(arr, ct)
            rows[c]["delta_vs_ct"] = sum(ct)/len(common) - sum(arr)/len(common)
            rows[c]["p_raw"] = w["p_two_sided"]
            pvals.append(w["p_two_sided"]); comp.append(c)
    holm = holm_bonferroni(pvals)
    for c, h in zip(comp, holm):
        rows[c]["p_holm"] = h["p_adjusted"]
    return {"n": len(common), "configs": rows}

def main():
    out = {}
    for bench, (label, ndocs) in BENCHES.items():
        r = run_bench(bench)
        if r is None:
            print(f"{label}: no full-corpus cells"); continue
        out[bench] = {"label": label, "n_docs": ndocs, **r}
        print(f"\n=== {label} full corpus ({ndocs} docs, n={r['n']}) ===")
        for c in ORDER:
            if c in r["configs"]:
                x = r["configs"][c]
                extra = ""
                if "delta_vs_ct" in x:
                    extra = f"  delta_vs_ct={x['delta_vs_ct']:+.4f} p_holm={x['p_holm']:.3g}"
                print(f"  {c:24s} {x['mean']:.4f} [{x['lo']:.3f},{x['hi']:.3f}]{extra}")
    (ROOT / "results" / "stage3" / "fullcorpus_scaling.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print("\nwrote results/stage3/fullcorpus_scaling.json")

if __name__ == "__main__":
    main()
