"""Aggregate the QASPER head-to-head: V4t corpus-tuned vs HippoRAG vs MemGPT/Letta,
all at the matched n=1005 (batch_calib) calibration-question scope, paired by the
doc{D}_qa{Q} suffix. Emits per-system bootstrap 95% CI, paired Wilcoxon vs V4t
(Holm-corrected across the two-system memory family), and LaTeX-ready row values.

Reference (for the caption only): tuned-lexical QASPER cells live at the n=94
fair-baseline scope (Table tab:fairbaseline); the n=1005 attention-corpus-tuned
and stock bm25-corpus cells are also reported here for context.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.statistics import bootstrap_ci, wilcoxon_signed_rank, holm_bonferroni

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "results" / "stage3" / "judge_queue"

def load(cell: str) -> dict[str, float]:
    out = {}
    for l in (Q / cell / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if not l.strip(): continue
        d = json.loads(l)
        suf = re.search(r"(doc\d+_qa\d+)", d["qid"]).group(1)
        out[suf] = float(d["judge_score"])
    return out

def main() -> int:
    v4 = load("qasper__v4t-corpus-tuned__batch_calib__seed42")
    hp = load("qasper__hipporag-corpus__batch__seed42")
    lt = load("qasper__letta-corpus__batch__seed42")
    common = sorted(set(v4) & set(hp) & set(lt))
    print(f"paired n = {len(common)} (v4={len(v4)} hp={len(hp)} lt={len(lt)})")
    av4 = [v4[s] for s in common]
    ahp = [hp[s] for s in common]
    alt = [lt[s] for s in common]

    def row(name, arr):
        ci = bootstrap_ci(arr)
        print(f"  {name:22s} mean={ci['point_estimate']:.4f} "
              f"CI[{ci['ci_lower']:.2f}, {ci['ci_upper']:.2f}]  n={ci['n']}")
        return ci
    print("=== QASPER head-to-head (paired, n=1005) ===")
    civ4 = row("V4t corpus-tuned", av4)
    cihp = row("HippoRAG", ahp)
    cilt = row("Letta", alt)

    # paired Wilcoxon vs V4t (a=v4, b=system -> tests system-v4)
    w_hp = wilcoxon_signed_rank(av4, ahp)
    w_lt = wilcoxon_signed_rank(av4, alt)
    holm = holm_bonferroni([w_hp["p_two_sided"], w_lt["p_two_sided"]])
    print("=== paired Wilcoxon vs V4t (Holm across the 2-system memory family) ===")
    for label, w, h in [("HippoRAG", w_hp, holm[0]), ("Letta", w_lt, holm[1])]:
        d = (sum(ahp if label=='HippoRAG' else alt)/len(common)) - (sum(av4)/len(common))
        print(f"  V4t vs {label:9s}: delta(system-v4)={d:+.4f}  "
              f"p_raw={w['p_two_sided']:.4g}  p_Holm={h['p_adjusted']:.4g}  "
              f"reject={h['significant']}")

    # reference cells at n=1005 (context only)
    print("=== n=1005 reference cells (context) ===")
    for name, cell in [("attention-tuned", "qasper__attention-corpus-tuned__batch_calib__seed42"),
                       ("bm25-stock", "qasper__bm25-corpus__batch_calib__seed42")]:
        try:
            ref = load(cell); arr = [ref[s] for s in common if s in ref]
            ci = bootstrap_ci(arr)
            print(f"  {name:16s} mean={ci['point_estimate']:.4f} "
                  f"CI[{ci['ci_lower']:.2f}, {ci['ci_upper']:.2f}] n={ci['n']}")
        except FileNotFoundError:
            print(f"  {name}: MISSING")

    print("\n=== LaTeX rows (QASPER column, n=1005) ===")
    for name, ci in [("V4t corpus-tuned (learned)", civ4), ("HippoRAG (reimpl.)", cihp),
                     ("MemGPT/Letta (reimpl.)", cilt)]:
        print(f"  {name:30s} $ {ci['point_estimate']:.3f}$ "
              f"{{\\scriptsize[{ci['ci_lower']:.2f}, {ci['ci_upper']:.2f}]}}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
